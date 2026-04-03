#!/usr/bin/env python3
"""
Fetch Test Data — Downloads publicly available aerospace PDFs for local testing.

Sources are NASA Technical Reports Server (NTRS) documents with stable download
URLs. These cover anomaly investigation, telemetry, fault management, and systems
engineering — the core domain of the application.

Usage:
    python scripts/fetch_test_data.py              # Download all
    python scripts/fetch_test_data.py --list       # List available sources
    python scripts/fetch_test_data.py --verify     # Check URLs without downloading
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

# ── Curated public aerospace PDFs (NASA NTRS + open-access) ──
# These are direct-download links to publicly funded research documents.
TEST_SOURCES = [
    # ── NASA ──
    {
        "id": "nasa-anomaly-detection-launch-ops",
        "name": "NASA — Anomaly Detection for Next-Gen Space Launch Ground Operations",
        "url": "https://ntrs.nasa.gov/api/citations/20100027325/downloads/20100027325.pdf",
        "description": "Fault Detection, Isolation, and Recovery (FDIR) system for launch control using the Inductive Monitoring System (IMS).",
        "expected_size_mb": 2.2,
    },
    {
        "id": "nasa-fault-management-handbook",
        "name": "NASA Fault Management Handbook (NASA-HDBK-1002 Draft)",
        "url": "https://www.nasa.gov/wp-content/uploads/2015/04/636372main_NASA-HDBK-1002_Draft.pdf",
        "description": "Guidelines for fault management in flight systems — fault detection, isolation, diagnosis, and response.",
        "expected_size_mb": 3.0,
    },
    {
        "id": "nasa-systems-engineering-handbook",
        "name": "NASA Systems Engineering Handbook (SP-2016-6105 Rev2)",
        "url": "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf",
        "description": "Full lifecycle systems engineering — requirements, design, integration, verification, and operations.",
        "expected_size_mb": 3.6,
    },
    {
        "id": "nasa-system-safety-handbook",
        "name": "NASA System Safety Handbook (NASA/SP-2010-580)",
        "url": "https://ntrs.nasa.gov/api/citations/20120003291/downloads/20120003291.pdf",
        "description": "Hazard analysis, risk assessment, and safety assurance for space systems across all mission phases.",
        "expected_size_mb": 2.1,
    },
    {
        "id": "nasa-mars-climate-orbiter-mishap",
        "name": "Mars Climate Orbiter Mishap Investigation Board Phase I Report",
        "url": "https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf",
        "description": "Investigation into loss of Mars Climate Orbiter due to metric/imperial unit conversion failure.",
        "expected_size_mb": 1.4,
    },
    {
        "id": "nasa-mars-polar-lander-failure",
        "name": "Loss of Mars Polar Lander and Deep Space 2 Missions Report",
        "url": "https://ntrs.nasa.gov/api/citations/20000061966/downloads/20000061966.pdf",
        "description": "NASA investigation — failure modes, fault trees, and recommendations for spacecraft testing.",
        "expected_size_mb": 9.4,
    },
    {
        "id": "nasa-fault-mgmt-principles",
        "name": "Development of NASA Fault Management Handbook (JHUAPL)",
        "url": "https://flightsoftware.jhuapl.edu/files/2011/FSW11_Fesq.pdf",
        "description": "Fault management principles, spacecraft autonomy, and fault detection/response architecture.",
        "expected_size_mb": 2.2,
    },
    # ── SpaceX ──
    {
        "id": "spacex-falcon-users-guide",
        "name": "SpaceX Falcon Payload Users Guide (2025)",
        "url": "https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf",
        "description": "Falcon 9/Heavy payload interfaces, environments, spacecraft operations, and mission integration.",
        "expected_size_mb": 8.3,
    },
    # ── ISRO ──
    {
        "id": "isro-annual-report-2023",
        "name": "ISRO Annual Report 2022-2023",
        "url": "https://www.isro.gov.in/media_isro/pdf/AnnualReport/Annual_Report_2022_23_Eng.pdf",
        "description": "ISRO annual report — satellite missions, launch vehicle operations, telemetry infrastructure.",
        "expected_size_mb": 97.3,
    },
    # ── JAXA ──
    {
        "id": "jaxa-spacecraft-telemetry-standard",
        "name": "JAXA Spacecraft Information Base Definition (Telemetry & Telecommand)",
        "url": "https://sma.jaxa.jp/TechDoc/Docs/JAXA-JERG-2-700-TP004.pdf",
        "description": "JAXA technical standard for telemetry and telecommand value processing and monitoring.",
        "expected_size_mb": 1.8,
    },
    {
        "id": "jaxa-epsilon-users-manual",
        "name": "JAXA Epsilon Launch Vehicle Users Manual",
        "url": "https://global.jaxa.jp/projects/rockets/epsilon/pdf/EpsilonUsersManual_e.pdf",
        "description": "Epsilon launch vehicle — payload interfaces, telemetry systems, flight operations.",
        "expected_size_mb": 6.8,
    },
    # ── ESA ──
    {
        "id": "esa-ecss-ground-systems",
        "name": "ESA/ECSS Ground Systems and Operations Standard (ECSS-E-ST-70C)",
        "url": "https://eop-cfi.esa.int/Repo/PUBLIC/DOCUMENTATION/SYSTEM_SUPPORT_DOCS/ECSS%20Standards%20for%20Ground%20Segments/ECSS-E-ST-70C-Ground%20systems.pdf",
        "description": "European space standard — telemetry, telecommand, mission control, spacecraft operations interfaces.",
        "expected_size_mb": 1.4,
    },
    # ── FAA ──
    {
        "id": "faa-flight-safety-analysis",
        "name": "FAA Flight Safety Analysis Handbook",
        "url": "https://www.faa.gov/about/office_org/headquarters_offices/ast/media/Flight_Safety_Analysis_Handbook_final_9_2011v1.pdf",
        "description": "Flight safety analysis of launch and reentry vehicles — risk assessment, debris analysis, failure modeling.",
        "expected_size_mb": 3.0,
    },
    {
        "id": "faa-ato-safety-management",
        "name": "FAA Air Traffic Organization Safety Management System Manual",
        "url": "https://www.faa.gov/air_traffic/publications/media/ATO-SMS-Manual.pdf",
        "description": "Safety risk management processes, safety assurance, and anomaly reporting.",
        "expected_size_mb": 3.7,
    },
]

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
SOURCES_JSON = Path(__file__).resolve().parent.parent / "app" / "config" / "sources.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (AerospaceMLOps/1.0; research-paper-downloader)",
}
TIMEOUT = 120.0


def download_pdf(source: dict, dest_dir: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    dest = dest_dir / f"{source['id']}.pdf"

    if dest.exists() and dest.stat().st_size > 10_000:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {source['name']} — already exists ({size_mb:.1f} MB)")
        return True

    print(f"  [GET]  {source['name']}")
    print(f"         {source['url']}")

    try:
        with httpx.stream(
            "GET",
            source["url"],
            headers=HEADERS,
            follow_redirects=True,
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()

            content_length = resp.headers.get("content-length")
            total = int(content_length) if content_length else None

            with open(dest, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"         {source['id']}",
                    leave=False,
                ) as pbar:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [OK]   Saved ({size_mb:.1f} MB)")
        return True

    except httpx.HTTPStatusError as e:
        print(f"  [FAIL] HTTP {e.response.status_code} — {source['url']}")
        dest.unlink(missing_ok=True)
        return False
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"  [FAIL] {type(e).__name__} — {source['url']}")
        dest.unlink(missing_ok=True)
        return False


def verify_urls() -> None:
    """HEAD-check each URL without downloading."""
    print("Verifying source URLs...\n")
    for src in TEST_SOURCES:
        try:
            resp = httpx.head(
                src["url"],
                headers=HEADERS,
                follow_redirects=True,
                timeout=30.0,
            )
            status = resp.status_code
            ct = resp.headers.get("content-type", "unknown")
            symbol = "OK" if status == 200 else f"HTTP {status}"
            print(f"  [{symbol}] {src['id']} ({ct})")
        except Exception as e:
            print(f"  [ERR]  {src['id']} — {type(e).__name__}")


def update_sources_json(downloaded: list[dict]) -> None:
    """Overwrite sources.json with successfully downloaded entries."""
    entries = [
        {
            "id": s["id"],
            "name": s["name"],
            "url": s["url"],
            "description": s["description"],
        }
        for s in downloaded
    ]

    payload = {"sources": entries}
    SOURCES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[sources.json] Updated with {len(entries)} entries → {SOURCES_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Download aerospace test PDFs")
    parser.add_argument("--list", action="store_true", help="List sources without downloading")
    parser.add_argument("--verify", action="store_true", help="HEAD-check URLs without downloading")
    parser.add_argument("--max", type=int, default=None, help="Max number of PDFs to download")
    args = parser.parse_args()

    if args.list:
        print("Available test sources:\n")
        for s in TEST_SOURCES:
            print(f"  {s['id']}")
            print(f"    {s['name']}")
            print(f"    ~{s['expected_size_mb']} MB")
            print(f"    {s['description']}\n")
        total = sum(s["expected_size_mb"] for s in TEST_SOURCES)
        print(f"Total estimated download: ~{total:.0f} MB")
        return

    if args.verify:
        verify_urls()
        return

    sources = TEST_SOURCES[: args.max] if args.max else TEST_SOURCES

    print("=" * 60)
    print("Aerospace Test Data Downloader")
    print("=" * 60)
    total_est = sum(s["expected_size_mb"] for s in sources)
    print(f"Sources: {len(sources)} | Estimated total: ~{total_est:.0f} MB")
    print(f"Destination: {DATA_DIR}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed = []

    for source in sources:
        if download_pdf(source, DATA_DIR):
            succeeded.append(source)
        else:
            failed.append(source)
        print()

    # Update sources.json with whatever succeeded
    if succeeded:
        update_sources_json(succeeded)

    print("=" * 60)
    print(f"Downloaded: {len(succeeded)}/{len(sources)}")
    if failed:
        print(f"Failed:     {', '.join(s['id'] for s in failed)}")
    print("=" * 60)

    if succeeded:
        print(f"\nNext step: run the ingestion pipeline:")
        print(f"  python -m app.ingest")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
