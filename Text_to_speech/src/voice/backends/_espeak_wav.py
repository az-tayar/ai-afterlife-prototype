"""Render one UTF-8 phrase from stdin using the bundled eSpeak NG library.

The caller starts a separate process per phrase because eSpeak's active voice
and audio callback are process-global.
"""

import ctypes
import os
import sys
import wave
from pathlib import Path

import espeakng_loader


def render(text: bytes, path: str, rate_wpm: int) -> None:
    library = ctypes.CDLL(espeakng_loader.get_library_path())
    data_root = os.fsencode(Path(espeakng_loader.get_data_path()).parent)
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    library.espeak_SetSynthCallback.argtypes = (callback_type,)
    library.espeak_SetSynthCallback.restype = None
    library.espeak_Initialize.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    library.espeak_SetVoiceByName.argtypes = (ctypes.c_char_p,)
    library.espeak_SetParameter.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    library.espeak_Synth.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint,
                                     ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                     ctypes.c_void_p, ctypes.c_void_p)
    # Synchronous retrieval delivers 16-bit mono PCM to the callback.
    sample_rate = library.espeak_Initialize(2, 0, data_root, 0x8000)
    if sample_rate <= 0:
        raise RuntimeError("eSpeak NG could not load its voice data")
    chunks = []

    @callback_type
    def collect(samples, count, events):
        if samples and count > 0:
            chunks.append(ctypes.string_at(samples, count * ctypes.sizeof(ctypes.c_short)))
        return 0

    library.espeak_SetSynthCallback(collect)
    if library.espeak_SetVoiceByName(b"en-us") or library.espeak_SetParameter(1, rate_wpm, 0):
        raise RuntimeError("eSpeak NG could not configure the English voice")
    # Start at character zero, interpret UTF-8, and leave a natural end pause.
    if (library.espeak_Synth(text, len(text) + 1, 0, 1, 0, 0x1001, None, None)
            or library.espeak_Synchronize() or not chunks):
        raise RuntimeError("eSpeak NG did not render the phrase")
    with wave.open(path, "wb") as output:
        output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(chunks))


if __name__ == "__main__":
    render(sys.stdin.buffer.read(), sys.argv[1], int(sys.argv[2]))
