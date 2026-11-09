import os
from openai import OpenAI, APIError
import time

from cleo.config import (
    PROVIDERS,
    TIMEOUT_S,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    FALLBACK_REPLY,
    MAX_HISTORY,
)


def build_clients() -> dict:
    """One OpenAI client per provider, built once at startup.

    The first provider's key is required. Later providers are skipped if
    their key is missing, so the app still runs with just Groq set up.
    """
    clients = {}
    for i, provider in enumerate(PROVIDERS):
        key = os.environ.get(provider["key_env"])
        if not key:
            if i == 0:
                raise RuntimeError(
                    f"{provider['key_env']} is missing. Add it to your .env file."
                )
            print(f"Skipping {provider['name']} (no {provider['key_env']})")
            continue
        clients[provider["name"]] = OpenAI(
            api_key=key,
            base_url=provider["base_url"],
            timeout=TIMEOUT_S,
            max_retries=0,  # fail over right away instead of retrying the same provider
        )
    return clients


def ask(client: OpenAI, provider: dict, messages: list) -> tuple[str, float, float]:
    """Call one provider. Returns (reply_text, time_to_first_token, total_time)."""
    start = time.perf_counter()
    first_chunk_time = None
    parts: list[str] = []

    request_kwargs = {
        "model": provider["model"],
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_TOKENS,
    }
    effort = provider.get("reasoning_effort")
    if effort:
        request_kwargs["reasoning_effort"] = effort

    stream = client.chat.completions.create(**request_kwargs)
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if not content:
            continue
        parts.append(content)
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start

    reply = "".join(parts).strip()
    total_time = time.perf_counter() - start
    if first_chunk_time is None:
        first_chunk_time = total_time
    return reply, first_chunk_time, total_time


def remember(history: list, user_text: str, reply: str) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY]


def think(clients: dict, history: list, user_text: str) -> tuple[str, dict]:
    """Return (reply, meta). History is updated in place on success.

    meta holds the provider name and latencies, so main can log them.
    """
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_text}]
    )

    for provider in PROVIDERS:
        client = clients.get(provider["name"])
        if client is None:
            continue
        try:
            reply, ttft, total = ask(client, provider, messages)
        except APIError as e:
            # covers rate limits, timeouts, connection errors, bad keys, bad model ids
            print(
                f"[{provider['name']}] {type(e).__name__}: {e}. Trying next provider."
            )
            continue
        if not reply:
            print(f"[{provider['name']}] empty reply. Trying next provider.")
            continue

        print(f"[{provider['name']}] first token {ttft:.2f}s, total {total:.2f}s")

        remember(history, user_text, reply)

        meta = {
            "provider": provider["name"],
            "model": provider["model"],
            "ttft": round(ttft, 3),
        }
        return reply, meta

    return FALLBACK_REPLY, {"provider": None}
