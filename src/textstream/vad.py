"""Silero VAD wrapper for speech/non-speech filtering."""

import threading

import numpy as np
from silero_vad_lite import SileroVAD

_vad = None
_vad_lock = threading.Lock()


def _get_vad():
    global _vad
    with _vad_lock:
        if _vad is None:
            _vad = SileroVAD(16000)
    return _vad


def contains_speech(audio_float32_16k: np.ndarray, threshold: float = 0.4) -> bool:
    """Check if a 16kHz float32 mono audio chunk contains speech.

    Processes in 512-sample (32ms) windows and short-circuits on
    first detection above threshold.
    """
    vad = _get_vad()
    window = 512
    for i in range(0, len(audio_float32_16k) - window + 1, window):
        chunk = audio_float32_16k[i : i + window]
        if not chunk.flags["C_CONTIGUOUS"]:
            chunk = np.ascontiguousarray(chunk)
        prob = vad.process(memoryview(chunk.data))
        if prob >= threshold:
            return True
    return False
