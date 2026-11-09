"""Tier 0 router: fast rules. route(text) returns (tool_name, args) or None.

None means "no rule matched", and the request goes to the LLM. Later, your
trained intent classifier replaces or sits in front of these rules with the
same return shape, so nothing downstream has to change.
"""

import re

from cleo.tools import APP_ALIASES


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (keeping dots inside numbers), collapse spaces."""
    t = re.sub(r"[^\w\s.]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip(" .")


# --- durations -------------------------------------------------------------
UNIT_SECONDS = {
    "second": 1,
    "sec": 1,
    "minute": 60,
    "min": 60,
    "hour": 3600,
    "hr": 3600,
}
DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b"
)


def parse_duration(t: str) -> int | None:
    """'1 hour 30 minutes' -> 5400. Digits only for now.

    TODO: Whisper usually writes digits, but add number words ("five minutes").
    """
    total = 0.0
    for value, unit in DURATION_RE.findall(t):
        total += float(value) * UNIT_SECONDS[unit.rstrip("s")]
    return int(total) if total > 0 else None


# --- rules: each takes normalized text, returns (tool, args) or None ---------
def timer_rule(t: str):
    if re.search(r"\btimer\b", t):
        seconds = parse_duration(t)
        if seconds:
            return "set_timer", {"seconds": seconds}
    return None


OPEN_RE = re.compile(
    r"^(?:please )?(?:open|launch|start)(?: up)? (?:the |my )?(?P<app>.+?)(?: app| application)?$"
)


def open_rule(t: str):
    m = OPEN_RE.match(t)
    if m and m.group("app") in APP_ALIASES:
        return "open_app", {"app_name": m.group("app")}
    return None


VOLUME_WORDS = {"volume", "sound", "audio", "louder", "quieter"}
UP_WORDS = {"up", "raise", "increase", "louder", "higher"}
DOWN_WORDS = {"down", "lower", "decrease", "quieter", "softer", "reduce"}


def volume_rule(t: str):
    words = set(t.split())
    if not (words & VOLUME_WORDS or "mute" in words or "unmute" in words):
        return None
    if "mute" in words or "unmute" in words:
        return "volume", {"direction": "mute"}
    if words & UP_WORDS:
        return "volume", {"direction": "up"}
    if words & DOWN_WORDS:
        return "volume", {"direction": "down"}
    return None  # e.g. "set volume to 30 percent" falls through to the LLM


MEDIA_WORDS = {"music", "song", "track", "playback", "podcast", "media"}


def media_rule(t: str):
    words = set(t.split())
    # Needs a media word, or to be a one-word command like "pause" or "skip".
    if len(words) > 6 or not (words & MEDIA_WORDS or len(words) == 1):
        return None
    if words & {"pause", "resume", "unpause"}:
        return "media_control", {"action": "play_pause"}
    if words & {"skip", "next"}:
        return "media_control", {"action": "next"}
    if words & {"previous", "back"}:
        return "media_control", {"action": "previous"}
    if "stop" in words:
        return "media_control", {"action": "stop"}
    return None


SEARCH_RE = re.compile(
    r"^(?:please )?(?:google|search (?:the web|the internet|google) for)\s+(?P<q>.+)$"
)


def search_rule(t: str):
    m = SEARCH_RE.match(t)
    if m:
        return "search_web", {"query": m.group("q")}
    return None


RULES = [timer_rule, open_rule, volume_rule, media_rule, search_rule]


def route(text: str):
    t = normalize(text)
    for rule in RULES:
        result = rule(t)
        if result:
            return result
    return None
