"""Actions the assistant can perform.

One registry, two callers: the router (rules now, your trained classifier
later) calls run_tool directly, and the LLM sees the same tools as function
schemas. Every tool returns a short string that is safe to speak aloud.
"""

import subprocess
import threading
import webbrowser
from urllib.parse import quote_plus

import keyboard

# Spoken name -> what Windows `start` should launch. Extend with your own apps
# and check that each one actually opens on your machine.
APP_ALIASES = {
    "calculator": "calc",
    "notepad": "notepad",
    "chrome": "chrome",
    "settings": "ms-settings:",
    "file explorer": "explorer",
    "task manager": "taskmgr",
    "command prompt": "cmd",
    "spotify": "spotify:",  # needs the Spotify desktop app installed
}

TOOLS: dict = {}
_notify = print


def set_notify(fn):
    """Where timer alerts go. Swap in speak() once text-to-speech exists."""
    global _notify
    _notify = fn


def tool(name, description, parameters):
    """Register a function as a tool: name, description, JSON-schema parameters."""

    def decorator(fn):
        TOOLS[name] = {"fn": fn, "description": description, "parameters": parameters}
        return fn

    return decorator


def _format_duration(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " ".join(parts) or "0 seconds"


@tool(
    "set_timer",
    "Start a countdown timer.",
    {
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Total duration in seconds."},
            "label": {"type": "string", "description": "Optional name for the timer."},
        },
        "required": ["seconds"],
    },
)
def set_timer(seconds: int, label: str = "") -> str:
    seconds = int(seconds)
    if seconds <= 0:
        return "I need a duration longer than zero."
    name = label or "timer"
    timer = threading.Timer(seconds, lambda: _notify(f"Your {name} is done."))
    timer.daemon = True
    timer.start()
    return f"{name.capitalize()} set for {_format_duration(seconds)}."


@tool(
    "open_app",
    "Open an application on the computer.",
    {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "enum": sorted(APP_ALIASES),
                "description": "Which app to open.",
            }
        },
        "required": ["app_name"],
    },
)
def open_app(app_name: str) -> str:
    # Only launch targets from APP_ALIASES, never raw spoken text, so a
    # mis-transcription or a model mistake can't run an arbitrary command.
    target = APP_ALIASES.get(app_name.lower().strip())
    if target is None:
        return f"I don't know how to open {app_name}."
    subprocess.Popen(f'start "" "{target}"', shell=True)  # Windows only
    return f"Opening {app_name}."


@tool(
    "volume",
    "Change the system volume.",
    {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "mute"]},
            "steps": {
                "type": "integer",
                "description": "How many small steps to move. Default 5.",
            },
        },
        "required": ["direction"],
    },
)
def volume(direction: str, steps: int = 5) -> str:
    if direction == "mute":
        keyboard.send("volume mute")
        return "Toggled mute."
    if direction not in ("up", "down"):
        return "I can only turn the volume up or down, or mute it."
    for _ in range(max(1, min(int(steps), 25))):
        keyboard.send(f"volume {direction}")
    return f"Volume {direction}."


MEDIA_KEYS = {
    "play_pause": "play/pause media",
    "next": "next track",
    "previous": "previous track",
    "stop": "stop media",
}


@tool(
    "media_control",
    "Control whatever media is currently playing.",
    {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": sorted(MEDIA_KEYS)}},
        "required": ["action"],
    },
)
def media_control(action: str) -> str:
    key = MEDIA_KEYS.get(action)
    if key is None:
        return "I can pause, skip, go back, or stop the media."
    keyboard.send(key)
    return "Done."


@tool(
    "search_web",
    "Search the web in the default browser.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."}
        },
        "required": ["query"],
    },
)
def search_web(query: str) -> str:
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
    return f"Searching for {query}."


def run_tool(name: str, args: dict | None = None) -> str:
    """Run a tool by name. Never raises; always returns something speakable."""
    spec = TOOLS.get(name)
    if spec is None:
        return f"I don't have a tool called {name}."
    try:
        return spec["fn"](**(args or {}))
    except TypeError as e:
        print(f"[tool {name}] bad arguments: {e}")
        return "I'm missing some details for that."
    except Exception as e:
        print(f"[tool {name}] {type(e).__name__}: {e}")
        return "Something went wrong running that."


def openai_tool_schemas() -> list:
    """The registry in the format chat-completions APIs expect."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in TOOLS.items()
    ]
