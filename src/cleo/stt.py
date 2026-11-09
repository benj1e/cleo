import queue
import numpy as np
import sounddevice as sd
import keyboard
from faster_whisper import WhisperModel
from openai import APIError, OpenAI

from cleo.config import SAMPLE_RATE, CHANNELS, HOTKEY, log_to_json


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


def transcribe(model: WhisperModel | None, audio: np.ndarray) -> str:
    if model is None:
        raise RuntimeError("transcribe was called without a model")

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
