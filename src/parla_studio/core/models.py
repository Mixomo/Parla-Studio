"""AI models, loaded on demand and shared between tabs.

- SAVA (inworld-ai/sava, Apache-2.0): voice probability per 40 ms frame (16 kHz).
- voice-isolator (inworld-ai/voice-isolator, Apache-2.0): separates voice from background (48 kHz).
- CrisperWhisper 2.0 turbo (nyralabs, non-commercial license): verbatim transcription with word timestamps.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import numpy as np

from . import audio_io
from ..i18n import tr as _tr

_lock = threading.Lock()
_cache: dict[str, object] = {}

SAVA_SR = 16000
SAVA_FRAME_S = 0.04
ISO_SR = 48000


VRAM_FRACTION = 0.92   # cap for PyTorch: if something doesn't fit, it fails and is retried in parts (no spill to RAM)
_guarded = False


def device() -> str:
    global _guarded
    import torch
    if torch.cuda.is_available():
        if not _guarded:
            # Without this cap, the Windows driver "spills" VRAM into RAM and everything becomes extremely slow.
            torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
            _guarded = True
        return "cuda"
    return "cpu"


def vram_free_gb() -> float:
    """Free VRAM for this app (within the cap), in GB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        cap = total * VRAM_FRACTION - (total - free)
        return max(0.0, min(free, cap + torch.cuda.memory_reserved()) / 2**30)
    except Exception:  # noqa: BLE001
        return 0.0


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


# approximate SAVA VRAM in window mode (measured on an RTX 3090): 0.25 GB + 0.135 GB per minute of window
SAVA_GB_BASE, SAVA_GB_PER_MIN = 0.25, 0.135


def sava_max_window_min() -> int:
    from . import memory
    return int(memory.sava_window_s() // 60) if vram_free_gb() else 10


def device_description() -> str:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} · {p.total_memory / 2**30:.0f} GB · CUDA {torch.version.cuda}"
    return "CPU (no CUDA GPU: it will be slow)"


def unload(name: str | None = None) -> None:
    """Free VRAM (one model or all of them). The app calls it after every process."""
    import torch
    with _lock:
        for k in [name] if name else list(_cache):
            _cache.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loaded() -> list[str]:
    return list(_cache)


LOW_VRAM_GB = 3.0   # before loading a model with less free VRAM than this, the other models are unloaded


def _get(name: str, factory):
    with _lock:
        if name not in _cache:
            if _cache and device() == "cuda" and vram_free_gb() < LOW_VRAM_GB:
                _cache.clear()   # make room: one model at a time when VRAM is short (never spill to RAM)
                gc.collect()
                free_vram()
            _cache[name] = factory()
        return _cache[name]


# ---------------------------------------------------------------- SAVA

def sava_model():
    def make():
        from inworld import sava
        return sava.load("inworld/sava", device=device()).eval()
    return _get("sava", make)


def speech_probability(mono: np.ndarray, sr: int, ctx=None, window_s: float | None = None,
                       context_s: float = 60.0) -> np.ndarray:
    """P(voice) per 40 ms frame (frame t covers [t*0.04, (t+1)*0.04) s).

    Full SAVA pass over long windows (10 min by default) with `context_s` of preceding context:
    ~100 times faster than frame-by-frame streaming with the same verdict (99.99 % in tests).
    VRAM grows with the window (~0.135 GB per minute); if it doesn't fit, the window halves itself."""
    from inworld.sava import streaming
    from .blocks import Eta
    x16 = audio_io.resample(mono, sr, SAVA_SR)
    if window_s is None:   # automatic: the longest window the free VRAM allows (measured before loading SAVA)
        from . import memory
        loaded_now = "sava" in _cache
        window_s = memory.sava_window_s(vram_free_gb() + (SAVA_GB_BASE if loaded_now else 0.0))
        window_s = min(window_s, max(60.0, np.ceil(x16.size / SAVA_SR / 60.0) * 60.0))   # never longer than the audio
        if ctx is not None:
            ctx.log(f"Automatic memory · {memory.describe()} · SAVA window {window_s / 60:.0f} min", 0)
    mdl = sava_model()
    F = SAVA_SR * SAVA_FRAME_S
    F = int(F)
    frames = streaming.num_frames(x16.size)
    W = max(F, int(window_s * SAVA_SR) // F * F)
    C = int(context_s * SAVA_SR) // F * F
    eta = Eta("Voice detection (SAVA)", x16.size / SAVA_SR, getattr(ctx, "log", None))
    out, a = [], 0
    while a < x16.size:
        if ctx is not None:
            ctx.check()
        lo, hi = max(0, a - C), min(x16.size, a + W)
        try:
            c = streaming.run(mdl, x16[lo:hi]).speech
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or W <= 60 * SAVA_SR:
                raise
            free_vram()
            W = W // 2 // F * F
            if ctx is not None:
                ctx.log(f"SAVA: the window did not fit in VRAM; reduced to {W / SAVA_SR / 60:.1f} min", 1)
            continue
        k = (a - lo) // F
        out.append(c[k:k + -(-(hi - a) // F)])
        a = hi
        frac = a / max(1, x16.size)
        if ctx is not None:
            ctx.progress(frac)
        eta.tick(frac)
    eta.done()
    return np.concatenate(out).astype(np.float32)[:frames]


def speech_on_hops(prob: np.ndarray, n_hops: int, hop_s: float) -> np.ndarray:
    """Resample the 40 ms probability onto the leveler's hop grid."""
    t = (np.arange(n_hops) + 0.5) * hop_s
    xp = (np.arange(prob.size) + 0.5) * SAVA_FRAME_S
    return np.interp(t, xp, prob, left=prob[0] if prob.size else 0.0, right=prob[-1] if prob.size else 0.0)


def words_with_voice(words: list, speech: np.ndarray, threshold: float = 0.2, pad_s: float = 0.1) -> tuple[list, list]:
    """(kept, dropped): a word is dropped only when SAVA hears no voice anywhere in its span (± pad_s),
    the typical Whisper hallucination over music, noise or silence."""
    kept, dropped = [], []
    for w in words:
        a = max(0, int((w.start - pad_s) / SAVA_FRAME_S))
        b = min(speech.size, max(a + 1, int(np.ceil((w.end + pad_s) / SAVA_FRAME_S))))
        (kept if speech.size == 0 or a >= speech.size or float(speech[a:b].max()) >= threshold else dropped).append(w)
    return kept, dropped


# ---------------------------------------------------------------- voice-isolator

def isolator():
    def make():
        from inworld import voice_isolation
        return voice_isolation.VoiceIsolator("inworld/voice-isolator", device=device())
    return _get("isolator", make)


def free_vram() -> None:
    """Free the video memory left over from an inference (e.g. after aborting)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def isolate_voice(x: np.ndarray, sr: int, ctx=None, engine: str = "inworld", block_s: float | None = None) -> np.ndarray:
    """Isolated voice with the same shape as x [frames, channels] and the same sample rate.
    Processed in blocks (with crossfade): real progress, time left and abort within seconds.
    `engine`: "inworld" (voice-isolator) or a key of MSS_MODELS (Mel-Band RoFormer via pymss).
    `block_s` None = automatic: the longest block the free RAM / VRAM allows (see core.memory)."""
    if "+" in engine:  # chained engines (e.g. voice/music, then noise removal)
        parts = engine.split("+")
        y = x
        for n, e in enumerate(parts):
            sub = ctx.sub(n / len(parts), (n + 1) / len(parts)) if ctx is not None else None
            y = isolate_voice(y, sr, sub, e, block_s)
        return y
    if engine != "inworld":
        return separate_voice_mss(x, sr, engine, ctx, block_s)
    from .blocks import Eta, process_blocks
    if block_s is None:
        from . import memory
        block_s = memory.separation_block_s(engine, x.shape[1])
        if ctx is not None:
            ctx.log(f"Automatic memory · {memory.describe()} · voice-isolator: block {block_s:.0f} s", 0)
    iso = isolator()
    eta = Eta("Voice isolation (voice-isolator)", x.shape[0] / sr, getattr(ctx, "log", None))

    def block(seg: np.ndarray, _inner) -> np.ndarray:
        out = np.empty_like(seg, dtype=np.float32)
        for k in range(seg.shape[1]):
            v = iso.isolate(np.ascontiguousarray(seg[:, k]), sr, output_sample_rate=sr)
            v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float32).reshape(-1)
            n = min(v.size, seg.shape[0])
            out[:n, k] = v[:n]
            out[n:, k] = 0.0
        return out

    def progress(f: float) -> None:
        if ctx is not None:
            ctx.progress(f)
        eta.tick(f)

    y = process_blocks(x, sr, block, block_s, 3.0, progress, ctx.check if ctx is not None else None,
                       getattr(ctx, "log", None))
    eta.done()
    return y


# ---------------------------------------------------------------- Mel-Band RoFormer (pymss, MIT)

from pathlib import Path as _Path  # noqa: E402

from ..paths import MODELS as _MODELS  # noqa: E402

MSS_DIR = _MODELS / "mss"
_MSS_CFG = _Path(__file__).resolve().parent / "mss_configs"  # (small) configs inside the repo
# Checkpoints download automatically the first time from their official Hugging Face repos (sizes verified)
MSS_MODELS = {
    "becruily_deux": {
        "label": "Mel-Band RoFormer · voice / music (becruily deux)",
        "ckpt": MSS_DIR / "vocal" / "becruily_deux.ckpt",
        "yaml": _MSS_CFG / "becruily_deux.yaml",
        "repo": "becruily/mel-band-roformer-deux", "file": "becruily_deux.ckpt", "mb": 435,
        "voice": "Vocals",
    },
    "voc_fv7": {
        "label": "Mel-Band RoFormer · isolate voice (voc_fv7)",
        "ckpt": MSS_DIR / "vocal" / "voc_fv7.ckpt",
        "yaml": _MSS_CFG / "voc_fv7.yaml",
        "repo": "GaboxR67/MelBandRoformers", "file": "melbandroformers/vocals/voc_fv7.ckpt", "mb": 490,
        "voice": "Vocals",
    },
    "gabox_denoise": {
        "label": "Mel-Band RoFormer · remove noise / bleed (gabox denoise-debleed)",
        "ckpt": MSS_DIR / "denoise" / "mel_band_roformer_denoise_debleed_gabox.ckpt",
        "yaml": _MSS_CFG / "gabox_denoise.yaml",
        "repo": "GaboxR67/MelBandRoformers", "file": "melbandroformers/instrumental/denoisedebleed.ckpt", "mb": 913,
        "voice": "Instrumental",  # in this model the "Instrumental" target is the clean voice (verified)
    },
    "becruily_karaoke": {
        "label": "BS-RoFormer · lead voice only (becruily karaoke)",
        "ckpt": MSS_DIR / "vocal" / "bs_roformer_karaoke_frazer_becruily.ckpt",
        "yaml": _MSS_CFG / "becruily_karaoke.yaml",
        "repo": "becruily/bs-roformer-karaoke", "file": "bs_roformer_karaoke_frazer_becruily.ckpt", "mb": 195,
        "voice": "Vocals",           # "Vocals" = lead voice; backing vocals go with "Instrumental"
        "model_type": "bs_roformer",
    },
}


def ensure_mss(key: str, log=None) -> None:
    """Download the checkpoint if missing (only once, into models/mss)."""
    m = MSS_MODELS[key]
    if m["ckpt"].exists():
        return
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download
    if log:
        log(f"Downloading {m['label']} (~{m['mb']} MB, only once) from huggingface.co/{m['repo']}...", 0)
    m["ckpt"].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=m["ckpt"].parent) as tmp:
        p = hf_hub_download(m["repo"], m["file"], local_dir=tmp)
        shutil.move(p, m["ckpt"])
MSS_SR = 44100

# Engines to separate the voice from the rest (the key is passed to isolate_voice)
ENGINES = {
    "inworld": "voice-isolator (Inworld)",
    "becruily_deux": "becruily deux · voice / music",
    "voc_fv7": "voc_fv7 · isolate voice",
    "gabox_denoise": "gabox · remove noise and bleed",
    "becruily_deux+gabox_denoise": "becruily + gabox · music and noise",
    "voc_fv7+gabox_denoise": "voc_fv7 + gabox · voice and noise",
    "becruily_karaoke": "becruily karaoke · lead voice only",
    "becruily_karaoke+gabox_denoise": "karaoke + gabox · lead voice and noise",
    "voc_fv7+gabox_denoise+becruily_karaoke": "voc_fv7 + gabox + karaoke · lead voice only",
}
# default engine of every tab: isolate the voice, remove the noise, then the residual background voices
DEFAULT_ENGINE = "voc_fv7+gabox_denoise+becruily_karaoke"
ENGINE_HINTS = {
    "inworld": "Separates the voice from everything else (noise, music, other voices). 4.3 s per 30 s of audio.",
    "becruily_deux": "Mel-Band RoFormer: separates voice and music. ~0.8 s per 30 s.",
    "voc_fv7": "Mel-Band RoFormer: isolates the voice (music and ambience). ~0.8 s per 30 s.",
    "gabox_denoise": "Mel-Band RoFormer: removes noise and bleed, but not music. ~0.5 s per 30 s.",
    "becruily_deux+gabox_denoise": "First separates the music (becruily), then removes the noise (gabox).",
    "voc_fv7+gabox_denoise": "First isolates the voice (voc_fv7), then removes the noise (gabox).",
    "becruily_karaoke": "BS-RoFormer: keeps only the lead voice and removes backing or background voices. ~0.8 s per 30 s.",
    "becruily_karaoke+gabox_denoise": "First keeps only the lead voice (karaoke), then removes the noise (gabox).",
    "voc_fv7+gabox_denoise+becruily_karaoke": "Isolates the voice (voc_fv7), removes the noise (gabox) and then the "
                                              "residual background voices (karaoke): only the main voice remains.",
}


def _mss_name(key: str, batch: int) -> str:
    return f"mss:{key}:b{batch}"


def mss_loaded_batch(key: str) -> int | None:
    """GPU batch of the RoFormer already in VRAM for `key` (None if not loaded)."""
    for name in loaded():
        if name.startswith(f"mss:{key}:b"):
            return int(name.rsplit(":b", 1)[1])
    return None


def mss_separator(key: str, batch: int = 1):
    def make():
        import logging
        from pymss import MSSeparator, get_separation_logger
        from pymss.logger import set_log_level
        m = MSS_MODELS[key]
        ensure_mss(key)
        log = get_separation_logger(console_level=logging.ERROR)
        set_log_level(log, logging.ERROR)  # the app reports progress
        log.setLevel(logging.ERROR)
        sep = MSSeparator(model_type=m.get("model_type", "mel_band_roformer"), model_path=str(m["ckpt"]), config_path=str(m["yaml"]),
                          device=device(), device_ids=[0], output_format="wav", use_tta=False,
                          store_dirs={}, logger=log, debug=False,
                          inference_params={"batch_size": int(batch), "overlap_size": None, "chunk_size": None,
                                            "standardize": None, "normalize": False, "mask_mode": None})
        set_log_level(sep.logger, logging.ERROR)
        return sep
    return _get(_mss_name(key, batch), make)


def separate_voice_mss(x: np.ndarray, sr: int, key: str, ctx=None, block_s: float | None = None) -> np.ndarray:
    """Voice separated with a RoFormer (pymss), same shape and sample rate as x.
    Blocks at 44.1 kHz; within each block pymss reports progress and aborting stops the inference.
    Automatic (block_s None): the block follows the free RAM and the GPU batch the free VRAM; if a batch does
    not fit after all, it halves itself and the block is repeated (the output does not depend on the batch)."""
    from . import memory
    from .blocks import Eta, process_blocks
    ensure_mss(key, getattr(ctx, "log", None))  # first time: download with a notice in the log
    m = MSS_MODELS[key]
    auto = block_s is None
    if auto:
        block_s = memory.separation_block_s(key, x.shape[1])
    batch = mss_loaded_batch(key) or memory.mss_batch(m["yaml"])   # measured before loading the weights
    state = {"batch": batch, "sep": mss_separator(key, batch)}
    if ctx is not None:
        name = m["label"].split("(")[-1].rstrip(")")
        ctx.log(f"Automatic memory · {memory.describe()} · {name}: block {block_s:.0f} s"
                f"{'' if auto else ' (manual)'} · GPU batch {batch}", 0)
    eta = Eta(m["label"].split("·")[1].strip().capitalize(), x.shape[0] / sr, getattr(ctx, "log", None))

    def block(seg: np.ndarray, inner) -> np.ndarray:
        def cb(done, total, _msg=""):
            if ctx is not None:
                ctx.check()  # raises Cancelled inside the inference: stops immediately
            inner(float(done) / float(total) if total else 0.0)
        s44 = audio_io.resample(np.ascontiguousarray(seg), sr, MSS_SR)
        while True:
            sep = state["sep"]
            sep.progress_callback = cb
            try:
                res = sep.separate(s44.T if s44.shape[1] > 1 else s44[:, 0], pbar=False, stems=[m["voice"]])
                break
            except Exception as e:  # noqa: BLE001
                if not is_oom(e) or state["batch"] <= 1:
                    raise
                sep.progress_callback = None
                unload(_mss_name(key, state["batch"]))
                free_vram()
                state["batch"] //= 2
                if ctx is not None:
                    ctx.log(f"GPU batch did not fit in VRAM: retrying the block with batch {state['batch']}", 1)
                state["sep"] = mss_separator(key, state["batch"])
        v = np.asarray(next(iter(res.values())), dtype=np.float32)
        v = v if v.ndim == 2 else v[:, None]
        if v.shape[1] != seg.shape[1]:
            v = v.mean(axis=1, keepdims=True) if seg.shape[1] == 1 else np.repeat(v[:, :1], seg.shape[1], axis=1)
        return audio_io.resample(v, MSS_SR, sr)

    def progress(f: float) -> None:
        if ctx is not None:
            ctx.progress(f)
        eta.tick(f)

    try:
        y = process_blocks(x, sr, block, block_s, 4.0, progress, ctx.check if ctx is not None else None,
                           getattr(ctx, "log", None))
    finally:
        state["sep"].progress_callback = None
    eta.done()
    return y


# ---------------------------------------------------------------- CrisperWhisper 2.0

@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0
    speaker: int | None = None   # fixed if the word comes from a separated voice (TIGER)


CW_LANGUAGES = {_tr(k): v for k, v in {
    "English": "en", "Spanish": "es", "German": "de", "French": "fr", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Polish": "pl", "Japanese": "ja", "Chinese": "zh",
}.items()}


# CrisperWhisper 2.0 turbo: ~6 s per minute of audio on an RTX 3090 (~11 s with hallucination mitigation).
# In Spanish it got more proper names and voseo right than the large variant, at almost 3x its speed.
CW_MODEL = "nyralabs/CrisperWhisper2.0_turbo"


def crisper():
    def make():
        from crisperwhisper import CrisperWhisperModel
        return CrisperWhisperModel(CW_MODEL, backend="transformers", device=device())
    return _get("crisper", make)


CW_PIECE_S = 300.0   # long audio is transcribed in pieces of ~5 min: progress, ETA and abort between pieces
CW_CUT_SEARCH_S = 20.0   # each cut goes to the quietest 100 ms within the last seconds of the piece


def transcription_pieces(n: int, sr: int, x: np.ndarray | None = None, piece_s: float = CW_PIECE_S) -> list[tuple[int, int]]:
    """[(a, b)] sample ranges covering [0, n), each about piece_s long, cut at a quiet point (between words)."""
    piece, search, hop = int(piece_s * sr), int(CW_CUT_SEARCH_S * sr), max(1, int(0.1 * sr))
    if n <= piece + piece // 4:
        return [(0, n)]
    out, a = [], 0
    while n - a > piece + piece // 4:
        b = a + piece
        if x is not None:
            seg = x[b - search:b]
            k = seg[: (seg.size // hop) * hop].reshape(-1, hop)
            b = b - search + int(np.argmin((k.astype(np.float64) ** 2).mean(axis=1))) * hop + hop // 2
        out.append((a, b))
        a = b
    out.append((a, n))
    return out


def _result_words(res) -> list[tuple[str, float, float, float]]:
    """(text, start, end, prob) of a CrisperWhisper result; empty when the model returned no words (None)."""
    out = []
    for w in (getattr(res, "words", None) or []):
        t = str(getattr(w, "word", "") or "").strip()
        if t and w.start is not None and w.end is not None:
            out.append((t, float(w.start), float(w.end), float(getattr(w, "probability", 1.0) or 1.0)))
    return out


def transcribe(mono: np.ndarray, sr: int, language: str, verbatim: bool = True,
               hallucination_mitigation: bool = True, ctx=None) -> tuple[str, list[Word]]:
    """Transcription with word timestamps (seconds of the original audio).

    Audio longer than ~6 min goes in pieces of CW_PIECE_S cut at quiet points: the progress bar advances,
    the log shows piece k/n with the remaining time, and Abort stops it between pieces."""
    import time
    x16 = audio_io.resample(mono, sr, 16000)
    model = crisper()
    pieces = transcription_pieces(x16.size, 16000, x16)
    texts, words = [], []
    t0 = time.perf_counter()
    total = max(1, x16.size)
    if ctx is not None:
        ctx.progress(0.0)
        if len(pieces) > 1:
            ctx.log(f"CrisperWhisper: {x16.size / 16000 / 60:.1f} min in {len(pieces)} pieces", 0)
    for k, (a, b) in enumerate(pieces):
        if ctx is not None:
            ctx.check()
        piece = x16[a:b]
        res = model.transcribe(piece, sr=16000, language=language, mode="verbatim" if verbatim else "intended",
                               word_confidence=True, hallucination_mitigation=hallucination_mitigation)
        found = _result_words(res)
        if not found and piece.size and float(np.sqrt(np.mean(piece.astype(np.float64) ** 2))) > 1e-3:
            # an audible piece came back empty (the model gave up on every 30 s chunk): one retry with the
            # other hallucination setting before accepting it as silence
            if ctx is not None:
                ctx.log(f"CrisperWhisper: piece {k + 1}/{len(pieces)} came back empty, retrying", 1)
                ctx.check()
            res = model.transcribe(piece, sr=16000, language=language, mode="verbatim" if verbatim else "intended",
                                   word_confidence=True, hallucination_mitigation=not hallucination_mitigation)
            found = _result_words(res)
            if not found and ctx is not None:
                ctx.log(f"CrisperWhisper: piece {k + 1}/{len(pieces)} ({a / 16000 / 60:.1f}-{b / 16000 / 60:.1f} min)"
                        " has no transcribable speech", 1)
        off = a / 16000
        words += [Word(t, s0 + off, s1 + off, pr) for t, s0, s1, pr in found]
        texts.append(str(getattr(res, "text", "") or "").strip())
        if ctx is not None:
            done = b / total
            ctx.progress(done)
            if len(pieces) > 1:
                el = time.perf_counter() - t0
                eta = el / done * (1 - done)
                ctx.log(f"CrisperWhisper: piece {k + 1}/{len(pieces)} · {b / 16000 / 60:.1f}/{total / 16000 / 60:.1f} min"
                        f" · elapsed {el / 60:.1f} min · remaining ~{eta / 60:.1f} min", 0)
    return " ".join(t for t in texts if t), words


# ---------------------------------------------------------------- diarization and identification

DIAR_ID = "nvidia/Nemotron-3-Diarization"   # OpenMDW-1.1 · up to 8 speakers, 10 ms, overlaps
SV_ID = "microsoft/wavlm-base-plus-sv"      # 512-dimensional x-vectors (Microsoft WavLM)
DIAR_FRAME_S = 0.01


def diarizer():
    def make():
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        proc = AutoProcessor.from_pretrained(DIAR_ID)
        dt = torch.bfloat16 if device() == "cuda" else torch.float32
        model = AutoModelForAudioFrameClassification.from_pretrained(DIAR_ID, dtype=dt).to(device()).eval()
        return proc, model
    return _get("diarizer", make)


def speaker_probabilities(mono: np.ndarray, sr: int) -> np.ndarray:
    """P(speech) per speaker every 10 ms: [T, 8], speakers in order of appearance."""
    import torch
    proc, model = diarizer()
    x16 = audio_io.resample(mono, sr, 16000)
    inputs = proc(x16, sampling_rate=16000, return_tensors="pt").to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    return logits.float().sigmoid()[0].cpu().numpy()


def speaker_verifier():
    def make():
        from transformers import AutoFeatureExtractor, WavLMForXVector
        return AutoFeatureExtractor.from_pretrained(SV_ID), WavLMForXVector.from_pretrained(SV_ID).to(device()).eval()
    return _get("verifier", make)


def voice_embedding(pieces: list[np.ndarray], sr: int) -> np.ndarray:
    """Voice print (unit vector) averaging the given fragments (mono, any sample rate)."""
    import torch
    fe, model = speaker_verifier()
    embs = []
    for p in pieces:
        x16 = audio_io.resample(p, sr, 16000)
        if x16.size < 8000:  # < 0.5 s: no aporta
            continue
        inp = fe(x16, sampling_rate=16000, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            e = torch.nn.functional.normalize(model(**inp).embeddings, dim=-1)[0].float().cpu().numpy()
        embs.append(e)
    if not embs:
        raise ValueError("not enough voice for a voice print (minimum 0.5 s)")
    m = np.mean(embs, axis=0)
    return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- TIGER (separation of 2 overlapping voices)

TIGER_ID = "JusperLee/TIGER-speech"          # weights Apache-2.0 · MIT code included in vendor/tiger
TIGER_REV = "f0340340b2d9bbf72074edf8c076dcab59a10ba2"
TIGER_SR = 16000


def separator():
    def make():
        from ..vendor.tiger import TIGER
        return TIGER.from_pretrained(TIGER_ID, revision=TIGER_REV).to(device()).eval()
    return _get("tiger", make)


def separate_two(mono: np.ndarray, sr: int) -> np.ndarray:
    """Separate two overlapping voices: returns [2, n] at the input sample rate."""
    import torch
    x16 = audio_io.resample(mono, sr, TIGER_SR)
    with torch.inference_mode():
        y = separator()(torch.from_numpy(np.ascontiguousarray(x16))[None, None].to(device()))[0]
    y = y.float().cpu().numpy()
    out = np.stack([audio_io.resample(np.ascontiguousarray(v), TIGER_SR, sr) for v in y])
    n = mono.size
    return out[:, :n] if out.shape[1] >= n else np.pad(out, ((0, 0), (0, n - out.shape[1])))
