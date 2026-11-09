import argparse
from faster_whisper import WhisperModel
from cleo.config import MODEL_SIZE
from cleo.llm import build_clients, think, remember
from cleo.stt import record_while_held, transcribe
from cleo.config import SAMPLE_RATE, HOTKEY, MIN_SECONDS, LOG_FILE, log_to_json
import cleo.tools as tools
import keyboard
import time
import cleo.router as router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", action="store_true", help="type instead of speaking (skips Whisper)"
    )
    args = parser.parse_args()

    model = None
    if not args.text:
        model = WhisperModel(
            MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=5
        )
    clients = build_clients()
    history = []

    if args.text:
        print("Text mode. Type a message, Ctrl+C to quit.")
    else:
        print(f"Hold {HOTKEY} to talk. Ctrl+C to quit.")

    tools.set_notify(lambda msg: print(f"\n[alert] {msg}"))  # swap for speak() later

    while True:
        if args.text:
            text = input("> ").strip()
            stt_latency = 0.0
        else:
            keyboard.wait(HOTKEY)
            audio = record_while_held()
            seconds = len(audio) / SAMPLE_RATE
            if seconds < MIN_SECONDS:
                print("Too short, ignored")
                continue
            print(audio.shape, audio.dtype, f"{seconds:.2f}s")

            start = time.perf_counter()
            text = transcribe(model, audio)
            stt_latency = time.perf_counter() - start
            print(f"{text or '(nothing heard)'}  [{stt_latency:.2f}s]")

        if not text:
            continue

        start = time.perf_counter()
        routed = router.route(text)

        if routed:  # tier 0: a rule matched, no LLM call
            tool_name, tool_args = routed
            reply = tools.run_tool(tool_name, tool_args)
            meta = {"tier": "rules", "tool": tool_name, "args": tool_args}
            remember(history, text, reply)
        else:  # no rule matched: the LLM answers
            reply, meta = think(clients, history, text)
            meta["tier"] = "llm"

        latency = time.perf_counter() - start
        print(f"{reply}  [{meta['tier']}, {latency:.2f}s]")

        log_to_json(
            LOG_FILE,
            text,
            stt_latency,
            reply=reply,
            action_latency=round(latency, 3),
            **meta,
        )
