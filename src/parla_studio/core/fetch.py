"""Audio from online videos (YouTube and any other site yt-dlp supports).

yt-dlp downloads the best audio-only stream as is (no ffmpeg needed) and PyAV decodes it to a 24-bit WAV at the
stream's own sample rate, optionally trimmed to a time range. YouTube needs a JavaScript runtime to solve its
challenges: the bundled `deno` package is used, or deno/node/bun from PATH.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from .leveler import Cancelled, Ctx

_URL = re.compile(r"https?://\S+")


def parse_time(text: str) -> float | None:
    """'90', '1:30', '1:02:03' or '83.5' → seconds; empty → None."""
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"Invalid time: {text}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def split_urls(text: str) -> list[str]:
    """Links found in free text (one per line, or separated by spaces), without duplicates, in order."""
    return list(dict.fromkeys(_URL.findall(text)))


def _js_runtimes() -> dict:
    runtimes = {}
    try:
        import deno
        runtimes["deno"] = {"path": deno.find_deno_bin()}
    except Exception:  # noqa: BLE001 - the package is optional: fall back to PATH
        if shutil.which("deno"):
            runtimes["deno"] = {}
    for name in ("node", "bun"):
        if shutil.which(name):
            runtimes[name] = {}
    return runtimes


class _Logger:
    """yt-dlp messages → app log (debug lines are dropped)."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.ctx.log(f"yt-dlp: {msg}", 1)

    def error(self, msg: str) -> None:
        self.ctx.log(f"yt-dlp: {msg}", 2)


def _options(ctx: Ctx, out_dir: Path, hook=None) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title).80B [%(id)s].%(ext)s"),
        "windowsfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "logger": _Logger(ctx),
        "js_runtimes": _js_runtimes(),
        "retries": 5,
        "fragment_retries": 5,
    }
    if hook:
        opts["progress_hooks"] = [hook]
    return opts


def expand(urls: list[str], ctx: Ctx | None = None) -> list[str]:
    """Playlists and channels → their video links (flat listing, nothing is downloaded)."""
    import yt_dlp
    ctx = ctx or Ctx()
    out = []
    opts = _options(ctx, Path("."))
    opts.update(extract_flat="in_playlist", noplaylist=False)
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            ctx.check()
            info = ydl.extract_info(url, download=False)
            if info.get("_type") in ("playlist", "multi_video"):
                found = [e.get("url") or e.get("webpage_url") for e in info.get("entries") or [] if e]
                ctx.log(f"{info.get('title') or url}: {len(found)} videos", 0)
                out += [u for u in found if u]
            else:
                out.append(info.get("webpage_url") or url)
    return list(dict.fromkeys(out))


def decode_to_wav(src: Path, dst: Path, start: float | None = None, end: float | None = None,
                  ctx: Ctx | None = None, duration: float | None = None) -> Path:
    """Decode any audio/video file with PyAV to a 24-bit WAV (streamed: bounded memory), optionally trimmed."""
    import av
    import soundfile as sf
    ctx = ctx or Ctx()
    tmp = dst.with_name(dst.name + ".part")
    with av.open(str(src)) as box:
        stream = box.streams.audio[0]
        sr = stream.codec_context.sample_rate or stream.rate
        ch = stream.codec_context.channels or 1
        layout = stream.codec_context.layout.name if stream.codec_context.layout else ("stereo" if ch == 2 else "mono")
        total = duration or (float(stream.duration * stream.time_base) if stream.duration else
                             (box.duration / 1e6 if box.duration else None))
        first = int(round((start or 0.0) * sr))
        last = int(round(end * sr)) if end is not None else None
        if start:
            box.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream, backward=True)
        res = av.AudioResampler(format="flt", layout=layout, rate=sr)
        written = 0
        span = ((end or total) - (start or 0)) if (end or total) else None
        try:
            with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=ch, subtype="PCM_24", format="WAV") as out:
                def put(frames) -> bool:
                    nonlocal written
                    for fr in frames:
                        x = fr.to_ndarray().reshape(-1, ch)
                        t0 = int(round(float(fr.pts * fr.time_base) * sr)) if fr.pts is not None else first + written
                        a, b = max(first - t0, 0), x.shape[0] if last is None else min(x.shape[0], last - t0)
                        if b > a:
                            out.write(np.clip(x[a:b], -1.0, 1.0))
                            written += b - a
                        if last is not None and t0 + x.shape[0] >= last:
                            return False
                    return True

                for frame in box.decode(stream):
                    ctx.check()
                    if not put(res.resample(frame)):
                        break
                    if span:
                        ctx.progress(min(1.0, written / sr / span))
                else:
                    put(res.resample(None))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("The selected range has no audio")
    tmp.replace(dst)
    return dst


def fetch_audio(url: str, out_dir: Path, start: float | None = None, end: float | None = None,
                ctx: Ctx | None = None) -> Path:
    """Download the audio of a video and return a WAV ready to load. Already downloaded videos are reused."""
    import yt_dlp
    ctx = ctx or Ctx()
    out_dir.mkdir(parents=True, exist_ok=True)
    if start is not None and end is not None and end <= start:
        raise ValueError("The end must be after the start")
    dl = ctx.sub(0.0, 0.7)

    def hook(d: dict) -> None:
        if ctx.cancel.is_set():
            raise yt_dlp.utils.DownloadCancelled("Download aborted")
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                dl.progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    with yt_dlp.YoutubeDL(_options(ctx, out_dir, hook)) as ydl:
        info = ydl.extract_info(url, download=False)
        ctx.check()
        if info.get("_type") in ("playlist", "multi_video"):
            raise ValueError("This link is a playlist: open a single video, or use it in the Voice dataset tab")
        stem = Path(ydl.prepare_filename(info)).stem
        rng = "" if start is None and end is None else f" ({start or 0:g}-{'' if end is None else f'{end:g}'} s)"
        wav = out_dir / f"{stem}{rng}.wav"
        if wav.exists():
            ctx.log(f"Already downloaded: {wav.name}", 0)
            ctx.progress(1.0)
            return wav
        ctx.log(f"Downloading {info.get('title') or url} · {info.get('duration_string') or '?'}", 0)
        try:
            info = ydl.process_ie_result(info, download=True)
        except yt_dlp.utils.DownloadCancelled:
            raise Cancelled() from None
        if ctx.cancel.is_set():
            raise Cancelled()
    reqs = info.get("requested_downloads") or [{}]
    src = Path(reqs[0].get("filepath") or ydl.prepare_filename(info))
    try:
        ctx.log(f"Decoding {src.name}", 0)
        decode_to_wav(src, wav, start, end, ctx.sub(0.7, 1.0), info.get("duration"))
    finally:
        src.unlink(missing_ok=True)
    ctx.progress(1.0)
    return wav
