"""Preventive censoring by lexicon: modular, per language and downloadable on demand.

A language's lexicon = custom list (editable) + downloaded sources (optional, each with its license).

List syntax (one term per line, `#` = comment):
    mierda          exact word (accents and case are ignored; ñ DOES matter: coño ≠ cono)
    put*            starts with "put" (puta, puto, putas, putísima...): one line instead of a hundred variants
    *mierda*        contains "mierda" (comemierda, mierdero...)
    *ón             ends with "ón"
    hijo de puta    multi-word phrase (each word accepts wildcards)
    !computadora    exception: never censored, even if a wildcard matches it

Matching also neutralizes obfuscations: interleaved symbols (pu!ta, p.u.t.a), digits as letters
(m1erd4, put@) and stretched letters (mieeeerda). Japanese and Chinese are matched by substring.
"""
from __future__ import annotations

import datetime
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import models
from .censor import Match

# ---------------------------------------------------------------- normalization

_PUNCT = re.compile(r"[!¡?¿\[\]().,;:\"'«»“”‘’—–\-…*_~^`´]")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
CJK = {"ja", "zh"}


def fold(text: str) -> str:
    """Lowercase, without accents or diaeresis but KEEPING ñ (and ç), without interleaved symbols and
    with digits/symbols used as letters translated (only if the word also has letters)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text).lower()
    s = s.replace("ñ", "\x00").replace("ç", "\x01")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ").replace("\x01", "ç")
    s = _PUNCT.sub("", s).strip()
    if re.search(r"[^\W\d_]", s) and re.search(r"[0-9@$]", s):
        s = s.translate(_LEET)
    return s


def collapse(s: str) -> str:
    """Collapse repeated letters (to compare stretched words: mieeeerda → mierda)."""
    return re.sub(r"(.)\1+", r"\1", s)


def _stretched(s: str) -> bool:
    return re.search(r"(.)\1\1", s) is not None


# ---------------------------------------------------------------- patrones

@dataclass(frozen=True)
class Pattern:
    kind: str   # "exact", "prefix", "suffix", "contains"
    text: str

    @classmethod
    def parse(cls, raw: str) -> "Pattern":
        pre, post = raw.startswith("*"), raw.endswith("*") and len(raw) > 1
        core = fold(raw.strip("*"))
        kind = "contains" if pre and post else "suffix" if pre else "prefix" if post else "exact"
        return cls(kind, core)

    def matches(self, tok: str) -> bool:
        if not tok or not self.text:
            return False
        if self.kind == "exact":
            return tok == self.text or (_stretched(tok) and collapse(tok) == collapse(self.text))
        if self.kind == "prefix":
            return tok.startswith(self.text) or (_stretched(tok) and collapse(tok).startswith(collapse(self.text)))
        if self.kind == "suffix":
            return tok.endswith(self.text) or (_stretched(tok) and collapse(tok).endswith(collapse(self.text)))
        return self.text in tok or (_stretched(tok) and collapse(self.text) in collapse(tok))


@dataclass
class Lexicon:
    """A language's lexicon, compiled for fast search."""
    lang: str
    exact: dict = field(default_factory=dict)         # normalized term → source
    exact_collapsed: dict = field(default_factory=dict)
    wild: list = field(default_factory=list)          # (Pattern, source) of a wildcard word
    phrases: list = field(default_factory=list)       # (tuple of Pattern, source) of several words
    exceptions: set = field(default_factory=set)
    sources: dict = field(default_factory=dict)       # source → number of terms

    def add_line(self, raw: str, source: str) -> None:
        raw = unicodedata.normalize("NFC", raw).strip()
        if not raw or raw.startswith("#"):
            return
        if raw.startswith("!"):
            self.exceptions.add(fold(raw[1:]))
            return
        toks = raw.split()
        self.sources[source] = self.sources.get(source, 0) + 1
        if len(toks) > 1:
            self.phrases.append((tuple(Pattern.parse(t) for t in toks), source))
            return
        p = Pattern.parse(raw)
        if not p.text:
            return
        if p.kind == "exact":
            self.exact.setdefault(p.text, source)
            self.exact_collapsed.setdefault(collapse(p.text), source)
        else:
            self.wild.append((p, source))

    @property
    def size(self) -> int:
        return sum(self.sources.values())

    def _word_hit(self, tok: str) -> str | None:
        """Source that censors the word (None if none)."""
        if not tok or tok in self.exceptions:
            return None
        if tok in self.exact:
            return self.exact[tok]
        if _stretched(tok) and collapse(tok) in self.exact_collapsed:
            return self.exact_collapsed[collapse(tok)]
        return next((src for p, src in self.wild if p.matches(tok)), None)

    def scan(self, words: list[models.Word]) -> list[Match]:
        """All occurrences in the transcript. Phrases first (the longest wins)."""
        if self.lang in CJK:
            return self._scan_cjk(words)
        toks = [fold(w.text) for w in words]
        phrases = sorted(self.phrases, key=lambda ps: len(ps[0]), reverse=True)
        out, i, n = [], 0, len(words)
        while i < n:
            if not toks[i] or toks[i] in self.exceptions:
                i += 1
                continue
            hit, src = 0, None
            for ph, ps in phrases:
                k = len(ph)
                if i + k <= n and all(p.matches(toks[i + j]) for j, p in enumerate(ph)):
                    hit, src = k, ps
                    break
            if not hit:
                src = self._word_hit(toks[i])
                hit = 1 if src else 0
            if hit:
                out.append(_match(words, i, i + hit, src))
                i += hit
            else:
                i += 1
        return out

    def _scan_cjk(self, words: list[models.Word]) -> list[Match]:
        """No spaces between words: each term is searched as a substring of the continuous text."""
        text, owner = "", []
        for k, w in enumerate(words):
            t = fold(w.text)
            text += t
            owner += [k] * len(t)
        terms = {p.text: src for p, src in self.wild} | dict(self.exact) | \
            {"".join(p.text for p in ph): src for ph, src in self.phrases}
        spans = []
        for term in sorted((t for t in terms if t), key=len, reverse=True):
            start = text.find(term)
            while start >= 0:
                spans.append((start, start + len(term), terms[term]))
                start = text.find(term, start + 1)
        out, used = [], set()
        for a, b, src in sorted(spans):
            i, j = owner[a], owner[b - 1] + 1
            if any(k in used for k in range(i, j)):
                continue
            used.update(range(i, j))
            out.append(_match(words, i, j, src))
        return out


def _match(words: list[models.Word], i: int, j: int, src: str | None = None) -> Match:
    ctx = " ".join(w.text for w in words[max(0, i - 6):i]) + " [" + " ".join(w.text for w in words[i:j]) + "] " + \
          " ".join(w.text for w in words[j:j + 6])
    return Match(i, j, words[i].start, words[j - 1].end, " ".join(w.text for w in words[i:j]), ctx.strip(),
                 source=f"DB · {src}" if src else "DB")


# ---------------------------------------------------------------- downloadable sources

@dataclass(frozen=True)
class Source:
    key: str
    name: str
    license: str
    url: str
    languages: tuple
    note: str = ""


SOURCES = {
    "textdetox": Source(
        "textdetox", "TextDetox multilingual toxic lexicon (Hugging Face)", "OpenRAIL++",
        "https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon",
        ("es", "en", "de", "fr", "it", "zh", "ja"), "The broadest (≈1.2k es, 3.4k en). With accents and ñ."),
    "ldnoobw": Source(
        "ldnoobw", "List of Dirty, Naughty, Obscene and Otherwise Bad Words (Shutterstock)", "CC-BY-4.0",
        "https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words",
        ("es", "en", "de", "fr", "it", "pt", "nl", "pl", "ja", "zh"),
        "Short and with some non-offensive words: worth reviewing."),
    "dsojevic": Source(
        "dsojevic", "dsojevic/profanity-list", "MIT", "https://github.com/dsojevic/profanity-list", ("en",),
        "English only; includes variants and wildcards."),
}

_UA = {"User-Agent": "parla-studio"}


def _get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        return r.read().decode("utf-8")


def _fetch(key: str, lang: str) -> list[str]:
    if key == "ldnoobw":
        return _get("https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
                    f"master/{lang}").splitlines()
    if key == "textdetox":
        words, offset = [], 0
        while True:
            q = urllib.parse.urlencode({"dataset": "textdetox/multilingual_toxic_lexicon", "config": "default",
                                        "split": lang, "offset": offset, "length": 100})
            data = json.loads(_get(f"https://datasets-server.huggingface.co/rows?{q}"))
            rows = [r["row"]["text"] for r in data.get("rows", [])]
            words += rows
            offset += len(rows)
            if not rows or offset >= int(data.get("num_rows_total", 0)):
                return words
    if key == "dsojevic":
        out = []
        for item in json.loads(_get("https://raw.githubusercontent.com/dsojevic/profanity-list/main/en.json")):
            out += [m.strip() for m in str(item.get("match", "")).split("|") if m.strip()]
        return out
    raise ValueError(f"unknown source: {key}")


class LexiconStore:
    """data/lexicons/ folder: custom lists per language, downloaded sources and their manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.src_dir = self.root / "sources"
        self._migrate_names()
        self.manifest_path = self.src_dir / "manifest.json"
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.manifest = {}

    def _migrate_names(self) -> None:
        """Rename lexicon files saved by older versions (Spanish names) to the current ones."""
        old = self.root / "fuentes"
        if old.is_dir() and not self.src_dir.exists():
            old.rename(self.src_dir)
        for p in self.root.glob("propia_*.txt"):
            new = self.root / ("custom_" + p.name[len("propia_"):])
            if not new.exists():
                p.rename(new)

    def own_path(self, lang: str) -> Path:
        return self.root / f"custom_{lang}.txt"

    def source_path(self, key: str, lang: str) -> Path:
        return self.src_dir / f"{key}_{lang}.txt"

    def downloaded(self, lang: str) -> list[str]:
        return [k for k in SOURCES if self.source_path(k, lang).exists()]

    def download(self, key: str, lang: str) -> int:
        """Download a source for a language (only once; it stays on disk with its license)."""
        src = SOURCES[key]
        if lang not in src.languages:
            raise ValueError(f"{src.name} has no list for \"{lang}\"")
        words = [unicodedata.normalize("NFC", w).strip() for w in _fetch(key, lang)]
        words = sorted({w for w in words if w and not w.startswith("#")}, key=str.lower)
        self.src_dir.mkdir(parents=True, exist_ok=True)
        header = f"# {src.name} · {lang} · license {src.license} · {src.url}\n# downloaded {datetime.date.today()}\n"
        self.source_path(key, lang).write_text(header + "\n".join(words) + "\n", encoding="utf-8")
        self.manifest[f"{key}_{lang}"] = {"source": src.name, "lang": lang, "license": src.license, "url": src.url,
                                          "count": len(words), "date": str(datetime.date.today())}
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(words)

    def remove(self, key: str, lang: str) -> None:
        self.source_path(key, lang).unlink(missing_ok=True)
        self.manifest.pop(f"{key}_{lang}", None)
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self, lang: str, enabled: set | None = None) -> Lexicon:
        """Custom list + enabled downloaded sources. Exceptions in the custom list apply to all of them."""
        lex = Lexicon(lang)
        own = self.own_path(lang)
        if own.exists():
            for line in own.read_text(encoding="utf-8").splitlines():
                lex.add_line(line, "custom")
        for key in self.downloaded(lang):
            if enabled is None or key in enabled:
                for line in self.source_path(key, lang).read_text(encoding="utf-8").splitlines():
                    lex.add_line(line, key)
        return lex

    def add_own(self, lang: str, term: str) -> bool:
        p = self.own_path(lang)
        existing = {fold(l.strip()) for l in p.read_text(encoding="utf-8").splitlines()} if p.exists() else set()
        if not term.strip() or fold(term) in existing:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(term.strip() + "\n")
        return True


OWN_HEADER = """# Custom censor list · language: {lang}
# One term per line. Accents and case are ignored; the letter ñ DOES matter (coño ≠ cono).
#   word           exact match       put*        starts with "put"
#   *word*         contains          *ón         ends with "ón"
#   multi word     phrase            !word       exception: never censored
"""


def migrate_legacy(store: "LexiconStore", legacy: Path, lang: str = "es") -> dict | None:
    """Convert an old list (e.g. profanity_app/database.txt, without ñ and with every variant) into the
    language's custom list: restores ñ, removes duplicates and folds families into wildcards.
    The old list is kept as a backup copy."""
    if not legacy.exists():
        return None
    refs: set = set()
    for k in store.downloaded(lang):
        refs |= {l.strip() for l in store.source_path(k, lang).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")}
    lines, report = sanitize(legacy.read_text(encoding="utf-8").splitlines(), refs)
    own = store.own_path(lang)
    previous = own.read_text(encoding="utf-8").splitlines() if own.exists() else []
    keep = [l for l in previous if l.strip() and not l.startswith("#")]
    merged = sorted({*keep, *lines}, key=lambda t: t.lstrip("!*"))
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text(OWN_HEADER.format(lang=lang) + f"# migrated from {legacy.name} ({report['input']} lines → "
                   f"{report['output']}; ñ restored: {len(report['enye_restored'])}; wildcard families: "
                   f"{len(report['families'])})\n" + "\n".join(merged) + "\n", encoding="utf-8")
    backup = legacy.with_name(legacy.stem + "_original" + legacy.suffix)
    if not backup.exists():
        legacy.rename(backup)
    report["backup"] = str(backup)
    return report


# ---------------------------------------------------------------- cleaning old lists

# Words with ñ that old lists stored without it ("coño" → "cono"); completed with the
# words with ñ from the downloaded sources.
KNOWN_ENYE = {"coño", "coños", "coña", "coñas", "coñazo", "coñazos", "coñito", "coñitos", "coñete", "puñeta",
              "puñetas", "puñetero", "puñetera", "puñeteros", "puñeteras", "puñetazo", "ñoño", "ñoña", "ñoños",
              "ñoñas", "cabroncete"}


def sanitize(lines: list[str], references: set[str] | None = None, min_family: int = 4,
             min_stem: int = 6) -> tuple[list[str], dict]:
    """Clean an old list:
    1) restore ñ where the word with ñ exists in the references (coño, puñeta...);
    2) remove duplicates (comparing without accents);
    3) fold variant families into a wildcard: if ≥`min_family` terms share a beginning of
       ≥`min_stem` letters, they are replaced by "beginning*" (future variants stay covered).
    Returns the new lines and a report."""
    refs = {fold(w) for w in (references or set()) | KNOWN_ENYE if "ñ" in fold(w)}
    by_plain = {r.replace("ñ", "n"): r for r in refs}
    report = {"input": 0, "enye_restored": [], "duplicates": 0, "families": [], "output": 0}
    seen, words = set(), []
    for raw in lines:
        raw = unicodedata.normalize("NFC", raw).strip()
        if not raw or raw.startswith("#"):
            continue
        report["input"] += 1
        f = fold(raw)
        if " " not in f and "*" not in raw and f in by_plain:
            report["enye_restored"].append(f"{raw} → {by_plain[f]}")
            f = by_plain[f]
        if f in seen:
            report["duplicates"] += 1
            continue
        seen.add(f)
        words.append(f)
    singles = sorted(w for w in words if " " not in w and "*" not in w and not w.startswith("!"))
    others = [w for w in words if w not in set(singles)]
    groups: dict[str, list[str]] = {}
    for w in singles:
        groups.setdefault(w[:min_stem], []).append(w)
    out = list(others)
    for key, fam in sorted(groups.items()):
        if len(fam) >= min_family and len(key) >= min_stem:
            stem = _common_prefix(fam)
            out.append(stem + "*")
            report["families"].append(f"{stem}* ({len(fam)}: {', '.join(fam[:4])}{'…' if len(fam) > 4 else ''})")
        else:
            out += fam
    out = sorted(set(out), key=lambda s: s.lstrip("!*"))
    report["output"] = len(out)
    return out, report


def _common_prefix(ws: list[str]) -> str:
    p = ws[0]
    for w in ws[1:]:
        while not w.startswith(p):
            p = p[:-1]
    return p


def merge(db_hits: list[Match], search_hits: list[Match]) -> list[Match]:
    """Merge lexicon detections and search hits without duplicates (same word or phrase)."""
    by_span = {(m.i, m.j): m for m in db_hits}
    for m in search_hits:
        prev = by_span.get((m.i, m.j))
        if prev is not None:
            prev.source = f"{prev.source} + search"
        else:
            by_span[(m.i, m.j)] = m
    return sorted(by_span.values(), key=lambda m: m.start)
