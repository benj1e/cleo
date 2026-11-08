import queue, time
import numpy as np
import sounddevice as sd
import keyboard
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
HOTKEY = "shift+windows"
MODEL_SIZE = "base"
MIN_SECONDS = 0.5


def record_while_held() -> np.ndarray:
    q = queue.Queue()
    audio_in = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(
        channels=CHANNELS, samplerate=SAMPLE_RATE, callback=callback, dtype="float32"
    ):
        print("Listening... release the keys to stop")
        while keyboard.is_pressed(HOTKEY):
            audio_in.append(q.get())

    # stream is closed, so anything left in the queue is safe to drain
    while not q.empty():
        audio_in.append(q.get_nowait())

    if not audio_in:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(audio_in).flatten()


def transcribe(model, audio: np.ndarray) -> str:
    segments, info = model.transcribe(
        audio=audio,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000),
        beam_size=5,
    )

    # segments is a generator: one pass only, so collect text as we go
    parts = [segment.text for segment in segments]
    return "".join(parts).strip()


def main():
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print(f"Hold {HOTKEY} to talk. Ctrl+C to quit.")
    while True:
        keyboard.wait(HOTKEY)
        audio = record_while_held()
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            print("Too short, ignored")
            continue
        print(audio.shape, audio.dtype, f"{seconds:.2f}s")

        start = time.perf_counter()
        text = transcribe(model, audio)
        latency = time.perf_counter() - start

        print(f"{text or '(nothing heard)'}  [{latency:.2f}s]")


if __name__ == "__main__":
    main()