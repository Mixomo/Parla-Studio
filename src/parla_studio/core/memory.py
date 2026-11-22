"""Automatic memory budget: sizes the separation blocks, the RoFormer GPU batch and the SAVA window from the RAM
and VRAM that are free right now, so each machine goes as fast as it can without spilling VRAM into RAM.

Every value is measured at the moment a process starts (other programs may be using memory). The estimates are
conservative and every step still has its out-of-memory fallback (halving the block, the batch or the window)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

GB = 2 ** 30
RAM_RESERVE_GB = 3.0       # left for the system, the interface and the audio already loaded
RAM_SHARE = 0.5            # share of the remaining free RAM a separation block may use
VRAM_SHARE = 0.8           # share of the free VRAM (inside the app's cap) a step may plan for
BLOCK_MIN_S, BLOCK_MAX_S = 30.0, 600.0
# RAM per second of block: input, 44.1 kHz copy, stems, resampled result and crossfade (~10 float32 copies)
SEP_RAM_COPIES = 10
SEP_SR = 44100
# voice-isolator (Inworld) processes the whole block on the GPU: VRAM grows with it (estimate, OOM halves it)
ISO_VRAM_GB_BASE, ISO_VRAM_GB_PER_S = 1.0, 0.02
# RoFormer: weights + workspace, plus each chunk of the GPU batch (grows with the model's chunk length)
MSS_VRAM_GB_BASE, MSS_VRAM_GB_PER_CHUNK_S = 1.5, 0.12
MSS_BATCH_MAX = 8


def ram_free_gb() -> float:
    """Available RAM (what can be used without swapping), in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / GB
    except Exception:  # noqa: BLE001
        pass
    try:  # Windows without psutil
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        m = _MS()
        m.len = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / GB
    except Exception:  # noqa: BLE001
        return 8.0


def vram_free_gb() -> float:
    from . import models
    return models.vram_free_gb()


def _clamp_block(s: float) -> float:
    return float(max(BLOCK_MIN_S, min(BLOCK_MAX_S, s)))


def separation_block_s(engine: str, channels: int = 2, ram_gb: float | None = None,
                       vram_gb: float | None = None) -> float:
    """Longest separation block that fits: RAM for every engine, and VRAM too for voice-isolator."""
    ram = ram_free_gb() if ram_gb is None else ram_gb
    per_s = SEP_SR * max(1, channels) * 4 * SEP_RAM_COPIES / GB
    s = max(0.0, ram - RAM_RESERVE_GB) * RAM_SHARE / per_s
    if "inworld" in engine.split("+"):
        vram = vram_free_gb() if vram_gb is None else vram_gb
        if vram > 0:
            s = min(s, (vram * VRAM_SHARE - ISO_VRAM_GB_BASE) / ISO_VRAM_GB_PER_S)
    return _clamp_block(s)


@lru_cache(maxsize=None)
def _chunk_s(yaml_path: str) -> float:
    """Length of the model's inference chunk (audio.chunk_size at 44.1 kHz), in seconds."""
    try:
        txt = Path(yaml_path).read_text(encoding="utf-8")
        m = re.search(r"^\s*chunk_size:\s*(\d+)", txt, re.M)
        return int(m.group(1)) / SEP_SR if m else 12.0
    except OSError:
        return 12.0


def mss_batch(yaml_path: str | Path, vram_gb: float | None = None) -> int:
    """How many chunks a RoFormer can process at once on the GPU with the free VRAM (1..8). The output is the same
    as with batch 1; a larger batch only uses the GPU better."""
    vram = vram_free_gb() if vram_gb is None else vram_gb
    if vram <= 0:
        return 1   # CPU: batching does not help
    per = MSS_VRAM_GB_PER_CHUNK_S * _chunk_s(str(yaml_path))
    return int(max(1, min(MSS_BATCH_MAX, (vram * VRAM_SHARE - MSS_VRAM_GB_BASE) // per)))


def sava_window_s(vram_gb: float | None = None) -> float:
    """Longest SAVA window that fits in the free VRAM (measured cost: 0.25 GB + 0.135 GB per minute)."""
    from . import models
    vram = vram_free_gb() if vram_gb is None else vram_gb
    if vram <= 0:
        return 600.0
    return 60.0 * int(max(1, min(60, (vram * VRAM_SHARE - models.SAVA_GB_BASE) / models.SAVA_GB_PER_MIN)))


def describe() -> str:
    return f"RAM free {ram_free_gb():.1f} GB · VRAM free {vram_free_gb():.1f} GB"
