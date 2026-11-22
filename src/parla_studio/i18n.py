"""Interface language (English / Spanish).

The source code is written in English. In Spanish mode every text that reaches the screen goes
through `tr()`: exact phrases from `i18n_es.ES`, f-string templates ("{}" = variable part) and
" · "-separated fragments are translated on the fly. Transcripts, file names and anything typed by
the user are never translated: only Qt display calls are hooked (labels, buttons, tooltips, combo
items, tabs, dialogs, notifications and the session log).

The language is read once at startup from data/settings.json; changing it restarts the app.
"""
from __future__ import annotations

import json
import re

from . import paths

LANGUAGES = {"en": "English", "es": "Español"}
DEFAULT = "en"


def _load_settings() -> dict:
    try:
        return json.loads(paths.SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_language(lang: str) -> None:
    data = _load_settings()
    data["language"] = lang
    paths.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    paths.SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")


LANG = _load_settings().get("language", DEFAULT)
if LANG not in LANGUAGES:
    LANG = DEFAULT

_exact: dict[str, str] = {}
_templates: list[tuple[re.Pattern, str]] = []
_cache: dict[str, str] = {}
_LETTERS = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{2}")
_SEP = " · "


def _add(src: str, dst: str) -> None:
    if src == dst:
        return
    if "{" in src:
        parts = re.split(r"\{\d*\}", src)
        rx = "^" + "(.*?)".join(re.escape(p) for p in parts) + "$"
        _templates.append((re.compile(rx, re.S), dst))
    else:
        _exact[src] = dst


def _build() -> None:
    from .i18n_es import ES
    for src, dst in ES.items():
        _add(src, dst)
        # every " · " fragment of a phrase can also be found on its own (composed messages)
        ps, pd = src.split(_SEP), dst.split(_SEP)
        if len(ps) > 1 and len(ps) == len(pd):
            for a, b in zip(ps, pd):
                if a not in ES:
                    _add(a, b)
    # most specific templates (more fixed text) first
    _templates.sort(key=lambda t: -len(re.sub(r"\(\.\*\?\)|\\", "", t[0].pattern)))


def _one(s: str, depth: int) -> str | None:
    if s in _exact:
        return _exact[s]
    for rx, dst in _templates:
        m = rx.match(s)
        if m:
            groups = [_tr(g, depth + 1) if depth < 2 else g for g in m.groups()]
            try:
                return dst.format(*groups)
            except (IndexError, ValueError):
                return None
    return None


def _tr(s: str, depth: int = 0) -> str:
    if not s or not _LETTERS.search(s):
        return s
    lead = s[:len(s) - len(s.lstrip())]
    trail = s[len(s.rstrip()):]
    core = s.strip()
    out = _one(core, depth)
    if out is None and "\n" in core:
        out = "\n".join(_tr(line, depth) for line in core.split("\n"))
    if out is None and _SEP in core:
        out = _SEP.join(_tr(p, depth) for p in core.split(_SEP))
    if out is None and ", " in core:
        parts = core.split(", ")
        done = [_tr(p, depth) for p in parts]
        if done != parts:
            out = ", ".join(done)
    if out is None:
        return s
    return lead + out + trail


def tr(s):
    """Translate a display string to the interface language (no-op in English)."""
    if LANG == "en" or not isinstance(s, str):
        return s
    if not _exact:
        _build()
    hit = _cache.get(s)
    if hit is None:
        hit = _tr(s)
        if len(_cache) > 20000:
            _cache.clear()
        _cache[s] = hit
    return hit


# ------------------------------------------------------------------ Qt hooks
def _wrap(cls, name: str, positions=(0,), keys=()) -> None:
    orig = getattr(cls, name)
    if getattr(orig, "_i18n", False):
        return

    def f(self, *a, **k):
        a = list(a)
        for i in positions:
            if i < len(a):
                a[i] = tr(a[i])
        for key in keys:
            if key in k:
                k[key] = tr(k[key])
        return orig(self, *a, **k)

    f._i18n = True
    setattr(cls, name, f)


def _wrap_static(cls, name: str, positions=(1,)) -> None:
    orig = getattr(cls, name)
    if getattr(orig, "_i18n", False):
        return

    def f(*a, **k):
        a = list(a)
        for i in positions:
            if i < len(a):
                a[i] = tr(a[i])
        if "caption" in k:
            k["caption"] = tr(k["caption"])
        if "title" in k:
            k["title"] = tr(k["title"])
        return orig(*a, **k)

    f._i18n = True
    setattr(cls, name, staticmethod(f))


def install(app) -> None:
    """Hook Qt display calls (Spanish only). Call after QApplication exists, before building windows."""
    if LANG == "en":
        return
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import (QAbstractButton, QColorDialog, QFileDialog, QLabel, QLineEdit, QListWidget,
                                   QPlainTextEdit, QTableWidget, QWidget)
    from qfluentwidgets import FluentWindow, InfoBar, LineEdit, Pivot, SwitchButton
    from qfluentwidgets.components.widgets.combo_box import ComboBoxBase

    _wrap(QLabel, "setText")
    _wrap(QAbstractButton, "setText")
    _wrap(QWidget, "setToolTip")
    _wrap(QWidget, "setWindowTitle")
    _wrap(QLineEdit, "setPlaceholderText")
    _wrap(LineEdit, "setPlaceholderText")
    _wrap(QPlainTextEdit, "setPlaceholderText")
    _wrap(QListWidget, "addItem")
    _wrap(ComboBoxBase, "addItem")
    _wrap(Pivot, "addItem", positions=(1,), keys=("text",))
    _wrap(SwitchButton, "setOnText")
    _wrap(SwitchButton, "setOffText")
    _wrap(FluentWindow, "addSubInterface", positions=(2,), keys=("text",))

    orig_hdr = QTableWidget.setHorizontalHeaderLabels

    def hdr(self, labels):
        return orig_hdr(self, [tr(x) for x in labels])

    QTableWidget.setHorizontalHeaderLabels = hdr

    for kind in ("info", "success", "warning", "error"):
        orig = getattr(InfoBar, kind).__func__

        def bar(cls, title, content, *a, _orig=orig, **k):
            return _orig(cls, tr(title), tr(content), *a, **k)

        setattr(InfoBar, kind, classmethod(bar))

    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
        _wrap_static(QFileDialog, name, positions=(1, 3) if name != "getExistingDirectory" else (1,))
    _wrap_static(QColorDialog, "getColor", positions=(2,))

    class _ShowFilter(QObject):
        """Translate texts set by C++ constructors when a window or dialog first appears."""

        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
                translate_tree(obj)
            return False

    app._i18n_filter = _ShowFilter(app)
    app.installEventFilter(app._i18n_filter)


def translate_tree(root) -> None:
    if LANG == "en":
        return
    from PySide6.QtWidgets import QAbstractButton, QLabel, QWidget
    for w in [root, *root.findChildren(QWidget)]:
        if isinstance(w, (QLabel, QAbstractButton)):
            t = w.text()
            n = tr(t)
            if n != t:
                w.setText(n)
        tip = w.toolTip()
        if tip:
            n = tr(tip)
            if n != tip:
                w.setToolTip(n)
