"""Record the spoken test set: audio files plus intent labels.

Usage:
    python record_test.py            # live transcript shown after each take
    python record_test.py --no-asr   # skip live transcription (faster)

Each take is saved as a 16 kHz wav in data/spoken_test/audio/ and described by
one line in data/spoken_test/labels.jsonl. You can quit any time (q) and resume
later; it counts what you already recorded per intent.

Needs: pip install pyyaml soundfile
Run it from the project root, next to listen.py.
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import keyboard
import soundfile as sf
import yaml

from listen import (
    HOTKEY,
    MIN_SECONDS,
    MODEL_SIZE,
    SAMPLE_RATE,
    record_while_held,
    transcribe,
)

INTENTS_FILE = Path("data/intents.yaml")
OUT_DIR = Path("data/spoken_test")
AUDIO_DIR = OUT_DIR / "audio"
LABELS_FILE = OUT_DIR / "labels.jsonl"

TARGET_PER_INTENT = 8
TARGET_OUT_OF_SCOPE = 20

# Random delivery style per prompt, so the test set is not one flat voice.
STYLES = [
    "normal voice",
    "short and blunt",
    "longer and polite",
    "fast and casual",
    "quiet or a bit mumbled",
    "include a filler word like um or uh",
    "with a fan, music, or other noise on",
]

OUT_OF_SCOPE_HINT = (
    "Say something the assistant should NOT handle: trivia, chit-chat, or a "
    "near-miss that reuses command words without being a command."
)


def load_intents(path):
    """Return {intent_name: description} from intents.yaml."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {name: (spec or {}).get("description", "") for name, spec in data.items()}


def load_counts():
    """Count takes already saved per intent, so a session can be resumed."""
    counts = Counter()
    if LABELS_FILE.exists():
        with open(LABELS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    counts[json.loads(line)["intent"]] += 1
    return counts


def build_todo(intents, counts):
    """Shuffled list with one entry per remaining take, so intents interleave."""
    todo = []
    for name in intents:
        target = TARGET_OUT_OF_SCOPE if name == "out_of_scope" else TARGET_PER_INTENT
        todo += [name] * max(0, target - counts[name])
    random.shuffle(todo)
    return todo


def save_take(intent, style, audio, live_text, model_name):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_path = AUDIO_DIR / f"{intent}_{stamp}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")

    row = {
        "file": f"audio/{wav_path.name}",
        "intent": intent,
        "slots": {},  # fill in by hand later: duration, app_name, query, ...
        "style": style,
        "duration_s": round(len(audio) / SAMPLE_RATE, 2),
        "live_transcript": live_text,  # feedback only, not ground truth
        "live_model": model_name,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-asr", action="store_true", help="skip live transcription")
    parser.add_argument(
        "--intents", default=str(INTENTS_FILE), help="path to intents.yaml"
    )
    args = parser.parse_args()

    intents = load_intents(args.intents)
    todo = build_todo(intents, load_counts())
    if not todo:
        print("Targets already met. Change TARGET_* constants to record more.")
        return

    model = None
    if not args.no_asr:
        from faster_whisper import WhisperModel

        print(f"Loading {MODEL_SIZE} for live feedback...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print(f"{len(todo)} takes to go. Hold {HOTKEY} to record. Ctrl+C to quit.\n")

    while todo:
        intent = todo[-1]
        style = random.choice(STYLES)

        print(f"[{len(todo)} left] Say a command for: {intent}")
        if intent == "out_of_scope":
            print(f"  {OUT_OF_SCOPE_HINT}")
        else:
            print(f"  what it does: {intents[intent]}")
        print(f"  style: {style}")

        keyboard.wait(HOTKEY)
        audio = record_while_held()

        if len(audio) / SAMPLE_RATE < MIN_SECONDS:
            print("  Too short, try again.\n")
            continue

        live_text = transcribe(model, audio) if model else ""
        if model:
            print(f"  heard: {live_text or '(nothing)'}")

        choice = (
            input("  Enter = keep, r = redo, s = skip intent, q = quit: ")
            .strip()
            .lower()
        )
        if choice == "q":
            break
        if choice == "r":
            print()
            continue
        if choice == "s":
            todo = [t for t in todo if t != intent]
            print()
            continue

        save_take(intent, style, audio, live_text, MODEL_SIZE if model else None)
        todo.pop()
        print()

    print("Done for now. Labels are in", LABELS_FILE)


if __name__ == "__main__":
    main()
