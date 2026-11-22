"""Forced word alignment: three interchangeable engines on top of CrisperWhisper's timestamps.

CrisperWhisper gives good timestamps, but on weak endings (single syllables, a plural "s", voiceless
consonants) it sometimes closes the word too early. The aligner pins every word against the audio:

- "mms"  · MMS-300m (1130 languages): character CTC with Viterbi; the text is romanized with uroman. 20 ms frames.
- "w2v"  · wav2vec2 XLSR-53 for the language: same CTC, but with the language's own alphabet (ñ, accents) and an
           acoustic model of that language. If the language has no dedicated model, MMS is used.
- "qwen" · Qwen3-ForcedAligner-0.6B (11 languages): predicts the start and end time of every word (not CTC),
           in 80 ms steps; spans of up to 5 min (here ≤ 30 s).

For all of them: audio is processed in spans with a margin (a final word may extend into the margin) and a
correction is accepted only if it is reasonable (≤ 0.6 s); otherwise the original time is kept.
"""
from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io, models

ALIGNERS = {
    "mms": "MMS-300m · 1130 languages",
    "w2v": "wav2vec2 XLSR-53 per language",
    "qwen": "Qwen3-ForcedAligner 0.6B · 11 languages",
}
ALIGNER_HINTS = {
    "mms": "Any language and script (romanized with uroman). Character CTC, 20 ms frames. Fast and robust.",
    "w2v": "Acoustic model of the language with its own alphabet (ñ, accents). CTC, 20 ms. Languages: es, en, fr, de, it, pt, nl, pl, ja, zh, ru; falls back to MMS otherwise.",
    "qwen": "Predicts the start and end of every word (not CTC), in 80 ms steps. Languages: es, en, fr, de, it, pt, ja, zh, ko, ru, yue; falls back to MMS otherwise.",
}
MMS_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
QWEN_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
W2V_IDS = {  # the same ones WhisperX uses per language
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}
QWEN_LANGS = {"es", "en", "fr", "de", "it", "pt", "ja", "zh", "ko", "ru", "yue"}
CHUNK_S = {"mms": 20.0, "w2v": 20.0, "qwen": 30.0}
MARGIN_S = 0.5
MAX_SHIFT_S = 0.6
MIN_WORD_S = 0.05   # an aligned word with 2+ letters shorter than this collapsed: CrisperWhisper's times are kept
FILLER = re.compile(r"^\[[^\]]*\]$")


@dataclass
class AlignRow:
    text: str
    cw_start: float
    cw_end: float
    prob: float
    start: float
    end: float
    status: str  # "aligned", "no letters", "large shift", "collapsed", "span not aligned", "filler"


def resolve(engine: str, language: str) -> str:
    """Engine that will actually be used (MMS if the language has no dedicated model)."""
    if engine == "w2v" and language not in W2V_IDS:
        return "mms"
    if engine == "qwen" and language not in QWEN_LANGS:
        return "mms"
    return engine if engine in ALIGNERS else "mms"


def describe(engine: str, language: str) -> str:
    e = resolve(engine, language)
    name = {"mms": MMS_ID, "w2v": W2V_IDS.get(language, ""), "qwen": QWEN_ID}[e]
    return f"{ALIGNERS[e]} ({name})" + (" · the language has no dedicated model: using MMS" if e != engine else "")


# ---------------------------------------------------------------- CTC (MMS and wav2vec2)

def _ctc_model(model_id: str):
    def make():
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if models.device() == "cuda" else torch.float32
        mdl = AutoModelForCTC.from_pretrained(model_id, dtype=dt).to(models.device()).eval()
        return tok, mdl
    return models._get(f"aligner:{model_id}", make)


_uroman = None


def _romanize(text: str) -> str:
    global _uroman
    try:
        if _uroman is None:
            import uroman
            _uroman = uroman.Uroman()
        text = _uroman.romanize_string(text)
    except Exception:  # noqa: BLE001 - without uroman: stripping accents is enough for Latin alphabets
        pass
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _emissions(mdl, x16: np.ndarray) -> np.ndarray:
    """log P(symbol) per frame [T, V]."""
    import torch
    x = (x16 - x16.mean()) / np.sqrt(x16.var() + 1e-7)
    with torch.inference_mode():
        t = torch.from_numpy(x.astype(np.float32))[None].to(mdl.device, dtype=mdl.dtype)
        logits = mdl(t).logits[0].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy()


def _viterbi(lp: np.ndarray, tokens: list[int], blank: int) -> np.ndarray | None:
    """Forced CTC alignment: state (index in the sequence with blanks) of every frame, or None if it does not fit."""
    T = lp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    if L == 0 or T < L:
        return None
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    skip = np.zeros(S, dtype=bool)  # the previous blank can be skipped (different letters)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -1e30
    dp = np.full(S, neg)
    dp[0], dp[1] = lp[0, blank], lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate(([neg], dp[:-1]))
        prev2 = np.where(skip, np.concatenate(([neg, neg], dp[:-2])), neg)
        cand = np.stack([stay, prev1, prev2])
        arg = np.argmax(cand, axis=0)
        dp = cand[arg, np.arange(S)] + lp[t, ext]
        back[t] = arg
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[s] <= neg / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])  # int(): with NumPy 2, int - int8 stays int8 and overflows
    return path


class _CTC:
    """Character CTC aligner (MMS with uroman, or wav2vec2 with the language's alphabet)."""

    def __init__(self, model_id: str, romanize: bool) -> None:
        self.tok, self.mdl = _ctc_model(model_id)
        vocab = self.tok.get_vocab()
        # CTC blank: "<blank>" in MMS (index 0; its tokenizer's "<pad>" is NOT the blank: with it, the silence
        # sticks to the last letter); in HF wav2vec2 models the blank is "<pad>"
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.blank = vocab.get("<blank>", pad)
        self.letters = {ch: i for ch, i in vocab.items() if len(ch) == 1 and ch.strip() and ch != "|"}
        self.romanize = romanize

    def tokens(self, text: str) -> list[int]:
        t = _romanize(text) if self.romanize else unicodedata.normalize("NFC", text).lower()
        return [self.letters[ch] for ch in t if ch in self.letters]

    def chunk(self, x16: np.ndarray, texts: list[str]) -> list[tuple[float, float] | None]:
        """(start, end) in seconds within the span, per word (None = could not be aligned)."""
        toks = [self.tokens(t) for t in texts]
        lp = _emissions(self.mdl, x16)
        frame_s = (x16.size / 16000) / lp.shape[0]
        seq = [tk for tl in toks for tk in tl]
        path = _viterbi(lp, seq, self.blank)
        if path is None:
            return [None] * len(texts)
        frames_of: dict[int, tuple[int, int]] = {}
        letter = np.where(path % 2 == 1, path // 2, -1)  # state 2m+1 = letter m of the sequence
        for t, m in enumerate(letter):
            if m >= 0:
                lo, hi = frames_of.get(int(m), (t, t))
                frames_of[int(m)] = (min(lo, t), max(hi, t))
        out, m = [], 0
        for tl in toks:
            first, last = m, m + len(tl) - 1
            m += len(tl)
            out.append((frames_of[first][0] * frame_s, (frames_of[last][1] + 1) * frame_s)
                       if first in frames_of and last in frames_of else None)
        return out


# ---------------------------------------------------------------- Qwen3-ForcedAligner

def _qwen_model():
    def make():
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        proc = AutoProcessor.from_pretrained(QWEN_ID)
        dt = torch.bfloat16 if models.device() == "cuda" else torch.float32
        mdl = AutoModelForTokenClassification.from_pretrained(QWEN_ID, dtype=dt).to(models.device()).eval()
        return proc, mdl
    return models._get("aligner:qwen", make)


class _Qwen:
    def __init__(self, language: str) -> None:
        self.proc, self.mdl = _qwen_model()
        self.language = language

    def tokens(self, text: str) -> list[str]:
        return self.proc.split_words_for_alignment(text, self.language)

    def chunk(self, x16: np.ndarray, texts: list[str]) -> list[tuple[float, float] | None]:
        import torch
        per_word = [self.tokens(t) for t in texts]
        inputs, word_lists = self.proc.prepare_forced_aligner_inputs(audio=x16, transcript=" ".join(texts),
                                                                     language=self.language)
        flat = [p for pw in per_word for p in pw]
        if word_lists[0] != flat:  # the word split of the full text does not match the per-word split
            return [None] * len(texts)
        inputs = inputs.to(self.mdl.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.mdl.dtype)
        with torch.inference_mode():
            logits = self.mdl(**inputs).logits
        ts = self.proc.decode_forced_alignment(logits=logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                                               timestamp_token_id=self.mdl.config.timestamp_token_id)[0]
        out, m = [], 0
        for pw in per_word:
            if not pw:
                out.append(None)
                continue
            out.append((float(ts[m]["start_time"]), float(ts[m + len(pw) - 1]["end_time"])))
            m += len(pw)
        return out


# ---------------------------------------------------------------- shared

def align(mono: np.ndarray, sr: int, words: list[models.Word], ctx=None, engine: str = "mms",
          language: str = "es") -> tuple[list[models.Word], list[AlignRow]]:
    """Return (words with corrected times, per-word report)."""
    engine = resolve(engine, language)
    eng = (_CTC(MMS_ID, True) if engine == "mms" else _CTC(W2V_IDS[language], False) if engine == "w2v"
           else _Qwen(language))
    alignable = [bool(eng.tokens(w.text)) and not FILLER.match(w.text) for w in words]
    rows = [AlignRow(w.text, w.start, w.end, w.prob, w.start, w.end, "") for w in words]
    total = mono.size / sr
    chunks, a = [], 0
    for k in range(1, len(words) + 1):  # spans of consecutive words
        if k == len(words) or words[k].end - words[a].start > CHUNK_S[engine]:
            chunks.append((a, k))
            a = k
    for n, (i, j) in enumerate(chunks):
        if ctx is not None:
            ctx.check()
            ctx.progress(n / max(1, len(chunks)))
        idx = [k for k in range(i, j) if alignable[k]]
        for k in range(i, j):
            if not alignable[k]:
                rows[k].status = "filler" if FILLER.match(words[k].text) else "no letters"
        if not idx:
            continue
        t0 = max(0.0, words[i].start - MARGIN_S, words[i - 1].end if i > 0 else 0.0)
        t1 = min(total, words[j - 1].end + MARGIN_S, words[j].start if j < len(words) else total)
        x16 = audio_io.resample(mono[int(t0 * sr):int(t1 * sr)], sr, 16000).astype(np.float32)
        res = eng.chunk(x16, [words[k].text for k in idx]) if x16.size >= 1600 else [None] * len(idx)
        for k, r in zip(idx, res):
            if r is None:
                rows[k].status = "span not aligned"
                continue
            s, e = t0 + r[0], t0 + r[1]
            if abs(s - words[k].start) > MAX_SHIFT_S or abs(e - words[k].end) > MAX_SHIFT_S:
                rows[k].status = "large shift"
                continue
            if e - s < MIN_WORD_S and sum(ch.isalnum() for ch in words[k].text) >= 2:
                rows[k].status = "collapsed"
                continue
            rows[k].start, rows[k].end, rows[k].status = s, e, "aligned"
    out = []
    for w, r in zip(words, rows):
        out.append(models.Word(w.text, r.start, max(r.end, r.start + 0.01), w.prob, w.speaker))
    for k in range(1, len(out)):  # keep order, no overlaps
        if out[k].start < out[k - 1].end:
            mid = (out[k].start + out[k - 1].end) / 2
            out[k - 1].end, out[k].start = max(out[k - 1].start + 0.01, mid), mid
    if ctx is not None:
        ctx.progress(1.0)
    return out, rows


def logs_dir() -> Path:
    from ..paths import LOGS
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS


def write_log(name: str, rows: list[AlignRow], header: list[str], tail: list[str] | None = None,
              kind: str = "dataset", engine: str = "aligner") -> Path:
    """Diagnostic log: CrisperWhisper's literal output (start, end, confidence, tag) and what the aligner
    corrected, word by word."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = logs_dir() / f"{kind}_{re.sub(r'[^\w.-]+', '_', Path(name).stem)[:80]}_{stamp}.log"
    lines = list(header) + ["", f"WORDS · CrisperWhisper (verbatim) → {engine}"]
    lines.append(f"{'#':>5}  {'cw_in':>9} {'cw_out':>9} {'conf':>5}  {'al_in':>9} {'al_out':>9} "
                 f"{'Δin ms':>7} {'Δout ms':>7}  {'status':<17} text")
    for k, r in enumerate(rows):
        di, do = (r.start - r.cw_start) * 1000, (r.end - r.cw_end) * 1000
        lines.append(f"{k:>5}  {r.cw_start:9.3f} {r.cw_end:9.3f} {r.prob:5.2f}  {r.start:9.3f} {r.end:9.3f} "
                     f"{di:+7.0f} {do:+7.0f}  {r.status or '-':<17} {r.text}")
    st: dict[str, int] = {}
    for r in rows:
        st[r.status or "-"] = st.get(r.status or "-", 0) + 1
    ends = np.array([(r.end - r.cw_end) * 1000 for r in rows if r.status == "aligned"])
    lines.insert(len(header), "SUMMARY: " + " · ".join(f"{k}: {v}" for k, v in st.items()) +
                 (f" · Δend mean {ends.mean():+.0f} ms, median {np.median(ends):+.0f} ms, "
                  f"{(ends > 20).sum()} words extended ≥ 20 ms" if ends.size else ""))
    if tail:
        lines += [""] + tail
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
