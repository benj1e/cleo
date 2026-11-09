from pathlib import Path
import os
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
import json

load_dotenv(find_dotenv())

SAMPLE_RATE = 16000
CHANNELS = 1
HOTKEY = "shift+windows"
MODEL_SIZE = "small.en"
MIN_SECONDS = 0.5

LOG_FILE = Path(__file__).resolve().parent / "data" / "log.jsonl"

# LLM settings
MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")
MAX_HISTORY = (
    10  # messages, not turns. Keep it even so user/assistant pairs stay together.
)
MAX_TOKENS = 1024
TIMEOUT_S = 5
SYSTEM_PROMPT = (
    "You are a voice assistant. Your replies are spoken aloud, so answer in one or "
    "two short sentences of plain, conversational English. No markdown, lists, "
    "emojis, or special symbols. If you don't know something, say so briefly."
)
FALLBACK_REPLY = "Sorry, I couldn't reach my language model just now."

# Tried in order; the first one that works wins.
# Remove "reasoning_effort" if you switch to a model that doesn't support it.
PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": MODEL,
        "reasoning_effort": "low",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",  # TODO: check current free model ids
    },
]


def log_to_json(file, text, latency, **extra):
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "content": text,
        "latency": latency,
        **extra,
    }
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")
