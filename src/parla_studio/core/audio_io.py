"""Audio reading, writing and resampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".opus", ".m4a")


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (float32 audio [frames, channels], sample rate)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(x), int(sr)


def info(path: str | Path) -> tuple[int, int, int]:
    """(sample rate, channels, frames) without reading the whole file."""
    i = sf.info(str(path))
    return int(i.samplerate), int(i.channels), int(i.frames)


def quantize24(x: np.ndarray) -> np.ndarray:
    """Quantization identical to the C++/Rust version: round(clamp(x) * 8388607), aligned to 32 bits."""
    q = np.round(np.clip(x, -1.0, 1.0).astype(np.float64) * 8388607.0).astype(np.int32)
    return q << 8


def save_wav24(path: str | Path, x: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x if x.ndim == 2 else x[:, None]
    sf.write(str(path), quantize24(x), sr, subtype="PCM_24")


def save_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)


# export formats: key → (name, extension, libsndfile format, subtype)
EXPORT_FORMATS = {
    "wav24": ("WAV 24-bit", ".wav", "WAV", "PCM_24"),
    "wav16": ("WAV 16-bit", ".wav", "WAV", "PCM_16"),
    "wav32": ("WAV 32-bit float", ".wav", "WAV", "FLOAT"),
    "flac24": ("FLAC 24-bit", ".flac", "FLAC", "PCM_24"),
    "flac16": ("FLAC 16-bit", ".flac", "FLAC", "PCM_16"),
    "ogg": ("OGG Vorbis (high quality)", ".ogg", "OGG", "VORBIS"),
    "mp3": ("MP3 320 kbps", ".mp3", "MP3", "MPEG_LAYER_III"),
}


def ext_for(fmt: str) -> str:
    return EXPORT_FORMATS[fmt][1]


def with_ext(path: str | Path, fmt: str) -> Path:
    return Path(path).with_suffix(ext_for(fmt))


def write_audio(target, x: np.ndarray, sr: int, fmt: str = "wav24") -> None:
    """Write x [frames, channels] in format `fmt` (see EXPORT_FORMATS). `target` is a path
    (the caller sets the extension, see with_ext) or an in-memory file (io.BytesIO)."""
    _, _, f, sub = EXPORT_FORMATS[fmt]
    x = x if x.ndim == 2 else x[:, None]
    kw = {}
    if f == "MP3":
        if sr > 48000:  # MP3 supports up to 48 kHz
            x, sr = resample(x, sr, 48000), 48000
        kw = dict(compression_level=0.0, bitrate_mode="CONSTANT")  # highest quality = 320 kbps
    elif f == "OGG":
        kw = dict(compression_level=0.1)  # Vorbis quality ≈ 9
    if sub == "PCM_24":
        data = quantize24(x)
    elif sub == "FLOAT":
        data = x.astype(np.float32, copy=False)
    else:
        data = np.clip(x, -1.0, 1.0)
    if isinstance(target, (str, Path)):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = str(target)
    sf.write(target, data, sr, format=f, subtype=sub, **kw)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr_from, sr_to, quality="HQ").astype(np.float32, copy=False)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1, dtype=np.float32) if x.ndim == 2 else x.astype(np.float32, copy=False)
