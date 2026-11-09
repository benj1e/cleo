from voicellm import VoiceManager
import time

def build_tts() -> VoiceManager:
    """Build a TTS engine once at startup."""
    return VoiceManager()

def speak(tts: VoiceManager, text: str) -> None:
    tts.speak(text)

