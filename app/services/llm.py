"""
External LLM provider abstraction.

Supports OpenRouter (OpenAI-compatible) and Google AI Studio.
API keys are passed per-request — nothing stored server-side.
"""

import google.generativeai as genai
from openai import OpenAI

DEFAULT_MODELS = {
    "openrouter": "google/gemini-2.0-flash-001",
    "google": "gemini-2.0-flash",
}

SYSTEM_PROMPT = (
    "You are an expert aerospace anomaly and telemetry analysis assistant. "
    "You help engineers interpret spacecraft telemetry data, diagnose anomalies, "
    "and understand aerospace systems engineering documents.\n\n"
    "When answering, reference the provided context documents by source name and "
    "page number. If the context does not contain enough information to fully "
    "answer the question, say so clearly rather than guessing."
)


def generate_openrouter(
    query: str,
    context_text: str,
    api_key: str,
    model: str | None = None,
) -> tuple[str, str]:
    """Call OpenRouter (OpenAI-compatible). Returns (answer, model_used)."""
    model_used = model or DEFAULT_MODELS["openrouter"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model_used,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"### Retrieved Context\n{context_text}\n\n"
                    f"### Question\n{query}"
                ),
            },
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    return response.choices[0].message.content, model_used


def generate_google(
    query: str,
    context_text: str,
    api_key: str,
    model: str | None = None,
) -> tuple[str, str]:
    """Call Google AI Studio. Returns (answer, model_used)."""
    model_used = model or DEFAULT_MODELS["google"]
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(
        model_name=model_used,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = f"### Retrieved Context\n{context_text}\n\n### Question\n{query}"
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1500,
            temperature=0.3,
        ),
    )
    return response.text, model_used


def generate(
    query: str,
    context_text: str,
    provider: str,
    api_key: str,
    model: str | None = None,
) -> tuple[str, str]:
    """Dispatch to the correct provider. Returns (answer, model_used)."""
    if provider == "openrouter":
        return generate_openrouter(query, context_text, api_key, model)
    elif provider == "google":
        return generate_google(query, context_text, api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
