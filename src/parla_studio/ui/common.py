"""Shared pieces: worker threads, log and level meter."""
from __future__ import annotations

import datetime
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from ..core import leveler as lv
from ..i18n import tr as _tr

ACCENT = "#34d399"
DANGER = "#f87171"


def format_combo(default: str = "wav24"):
    """Export format selector (keys of audio_io.EXPORT_FORMATS); `.key()` returns the chosen key."""
    from qfluentwidgets import ComboBox
    from ..core import audio_io
    keys = list(audio_io.EXPORT_FORMATS)
    cb = ComboBox()
    cb.setMinimumWidth(10)
    cb.addItems([audio_io.EXPORT_FORMATS[k][0] for k in keys])
    cb.setCurrentIndex(keys.index(default))
    cb.key = lambda: keys[cb.currentIndex()]
    return cb


def format_filter(fmt: str) -> str:
    from ..core import audio_io
    name, ext = audio_io.EXPORT_FORMATS[fmt][:2]
    return f"{name} (*{ext})"


LOG_STYLE = ("font-family: Consolas; font-size: 12px; background: #141414; color: #d6d6d6; "
             "border: 1px solid #2d2d2d; border-radius: 6px; padding: 4px")


class LogBus(QObject):
    """Session log: signal for the interface and the data/app.log file."""
    line = Signal(str, int)

    def __init__(self, file: Path) -> None:
        super().__init__()
        self.file = file
        self._lock = threading.Lock()

    def push(self, msg: str, level: int = 0) -> None:
        msg = _tr(msg)
        tag = _tr({0: "INFO", 1: "WARN", 2: "ERROR"}.get(level, "INFO"))
        text = f"[{datetime.datetime.now():%H:%M:%S}] {tag}: {msg}"
        with self._lock:
            try:
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except OSError:
                pass
            _console(text, level)
        self.line.emit(text, level)


_COLORS = {1: "\x1b[33m", 2: "\x1b[31m"}
_RESET = "\x1b[0m"


def _console(text: str, level: int) -> None:
    """Every log line also goes to the console window opened by "2- Start.bat" (all tabs, not only the Leveler)."""
    out = sys.stdout
    if out is None:  # pythonw: no console
        return
    try:
        color = _COLORS.get(level, "") if out.isatty() else ""
        out.write(color + text + (_RESET if color else "") + "\n")
        out.flush()
    except (OSError, ValueError, UnicodeError):
        pass


# ---------------------------------------------------------------- finish chime
_SOUNDS: dict[str, str] = {}


def _tone_file(kind: str) -> str | None:
    """Short chime (rising two notes: done · falling: error · one low note: aborted), generated once."""
    if kind in _SOUNDS:
        return _SOUNDS[kind]
    try:
        import numpy as np
        import soundfile as sf
        from .. import paths
        sr = 44100
        notes = {"done": (659.25, 987.77), "error": (440.0, 293.66), "cancel": (392.0,)}[kind]
        parts = []
        for k, f in enumerate(notes):
            last = k == len(notes) - 1
            n = int(sr * (0.38 if last else 0.14))
            t = np.arange(n) / sr
            env = np.minimum(1.0, t / 0.006) * np.exp(-t * (6.0 if last else 9.0))
            wave = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
            parts.append((0.22 * env * wave).astype(np.float32))
        p = paths.DATA / "sounds" / f"chime_{kind}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.concatenate(parts), sr, subtype="PCM_16")
        _SOUNDS[kind] = str(p)
        return str(p)
    except Exception:  # noqa: BLE001 - a sound must never break a task
        return None


def chime(kind: str = "done") -> None:
    """Sound at the end of a task (Settings: Sound when a task finishes)."""
    from .. import settings
    if not settings.get("chime", True):
        return
    try:
        if os.name == "nt":
            import winsound
            f = _tone_file(kind)
            if f:
                winsound.PlaySound(f, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                return
        from PySide6.QtWidgets import QApplication
        QApplication.beep()
    except Exception:  # noqa: BLE001
        pass


class Worker(QThread):
    """Run fn(ctx) outside the interface thread. `ctx.cancel` is set by cancel().

    When it ends (done, failed or aborted) a chime sounds (tasks longer than CHIME_AFTER_S) and, once no other
    process is running, every AI model is unloaded: VRAM is free between processes and tabs."""
    CHIME_AFTER_S = 1.5
    releases_models = True
    _running = 0
    _running_lock = threading.Lock()
    progress = Signal(float)
    log = Signal(str, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self.fn = fn
        self.ctx = lv.Ctx(progress=self.progress.emit, log=self.log.emit)

    def cancel(self) -> None:
        self.ctx.cancel.set()

    def run(self) -> None:
        t0, kind = time.perf_counter(), "done"
        if self.releases_models:
            with Worker._running_lock:
                Worker._running += 1
        try:
            result = self.fn(self.ctx)
            if self.releases_models:   # before `done`: the next step of the interface may start another process
                self._release_models()
            self.done.emit(result)
        except lv.Cancelled:
            kind = "cancel"
            self._release_models() if self.releases_models else self._free()
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 - reported in the interface and the log
            kind = "error"
            self._release_models() if self.releases_models else self._free()
            self.log.emit(traceback.format_exc(), 2)
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            if time.perf_counter() - t0 >= self.CHIME_AFTER_S:
                chime(kind)

    def _release_models(self) -> None:
        """Once per worker: the last running process unloads every model (VRAM back to the next one)."""
        if getattr(self, "_released", False):
            return
        self._released = True
        with Worker._running_lock:
            Worker._running -= 1
            last = Worker._running <= 0
        from ..core import models
        if last and models.loaded():
            self.log.emit("Models unloaded from VRAM: " + ", ".join(models.loaded()), 0)
            models.unload()
        else:
            self._free()

    @staticmethod
    def _free() -> None:
        import gc
        gc.collect()
        from ..core import models
        models.free_vram()


class MeterWidget(QWidget):
    """Level meter: RMS and LUFS-M bars (with a peak mark) and RMS/M/S/Peak/I readouts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values = {"rms": -200.0, "m": -200.0, "s": -200.0, "peak": -200.0}
        self.integrated: float | None = None
        self.setMinimumWidth(150)
        self.setMaximumWidth(190)

    def set_values(self, values: dict, integrated: float | None) -> None:
        changed = integrated != self.integrated or any(
            abs(max(values[k], -70) - max(self.values[k], -70)) > 0.05 for k in values)
        self.values, self.integrated = dict(values), integrated
        if changed:
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0b0b0c"))
        p.drawRoundedRect(r, 6, 6)
        mono = QFont("Consolas", 10)
        p.setFont(mono)
        p.setPen(QColor("#d6d6d6"))
        p.drawText(r.adjusted(8, 6, -8, 0), Qt.AlignLeft | Qt.AlignTop, _tr("METER"))
        p.drawText(r.adjusted(8, 6, -8, 0), Qt.AlignRight | Qt.AlignTop, "dB")
        top, bottom = r.top() + 30, r.bottom() - 150
        scale_w = 32

        def y_of(db: float) -> float:
            d = min(3.0, max(-60.0, db))
            return bottom - (d + 60) / 63 * (bottom - top)

        for db in (0, -6, -12, -18, -24, -36, -48, -60):
            y = y_of(db)
            p.setPen(QColor("#464646"))
            p.drawLine(int(r.left() + scale_w), int(y), int(r.right() - 4), int(y))
            p.setPen(QColor("#d6d6d6"))
            p.drawText(QRectF(r.left() + 2, y - 8, scale_w - 4, 16), Qt.AlignLeft | Qt.AlignVCenter, str(db))
        colw = (r.width() - scale_w - 8) / 2
        for i, (name, key) in enumerate((("RMS", "rms"), ("LUFS-M", "m"))):
            x0 = r.left() + scale_w + 2 + i * colw
            bw = colw - 8
            p.fillRect(QRectF(x0, top, bw, bottom - top), QColor(40, 40, 40))
            v = self.values[key]
            if v > -60:
                y = y_of(v)
                p.fillRect(QRectF(x0, y, bw, bottom - y), QColor(DANGER if v > -1 else ACCENT))
            if i == 0 and self.values["peak"] > -60:
                pk = self.values["peak"]
                p.fillRect(QRectF(x0, y_of(pk) - 1, bw, 2), QColor(DANGER if pk > -1 else "#ffffff"))
            p.setPen(QColor("#d6d6d6"))
            p.drawText(QRectF(x0 - 4, bottom + 4, bw + 8, 16), Qt.AlignLeft, name)

        def fmt(v):
            return f"{v:+.1f}" if v is not None and v > -70 else "--"

        rows = (("RMS", self.values["rms"]), ("M", self.values["m"]), ("S", self.values["s"]),
                (_tr("Peak"), self.values["peak"]), ("I", self.integrated))
        y = bottom + 30
        for name, v in rows:
            row = QRectF(r.left() + 8, y, r.width() - 16, 20)
            p.setFont(mono)
            p.setPen(QColor("#d6d6d6"))
            p.drawText(row, Qt.AlignLeft | Qt.AlignVCenter, name)
            p.setFont(QFont("Consolas", 12))
            p.setPen(QColor("#ffffff"))
            p.drawText(row, Qt.AlignRight | Qt.AlignVCenter, fmt(v))
            y += 22
        p.end()


class History:
    """Undo / redo of a tab's editable state (snapshots).

    The tab gives `snapshot()` (a copy of its state) and `restore(state)`. Call `record()` *before* each change;
    changes with the same `merge` key in a row (typing, dragging a slider) are one single step.
    """

    def __init__(self, snapshot, restore, on_change=None, limit: int = 100) -> None:
        self.snapshot, self.restore, self.on_change, self.limit = snapshot, restore, on_change, limit
        self.undo_stack: list = []
        self.redo_stack: list = []
        self._last_merge = None

    def record(self, merge=None) -> None:
        if merge is not None and merge == self._last_merge and self.undo_stack:
            return
        self.undo_stack.append(self.snapshot())
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()
        self._last_merge = merge
        self._changed()

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._last_merge = None
        self._changed()

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self) -> None:
        if self.undo_stack:
            self.redo_stack.append(self.snapshot())
            self._last_merge = None
            self.restore(self.undo_stack.pop())
            self._changed()

    def redo(self) -> None:
        if self.redo_stack:
            self.undo_stack.append(self.snapshot())
            self._last_merge = None
            self.restore(self.redo_stack.pop())
            self._changed()

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()


def undo_buttons(page: QWidget, history: History) -> tuple:
    """Undo / redo buttons (Ctrl+Z, Ctrl+Y / Ctrl+Shift+Z) for a tab. Text fields keep their own Ctrl+Z."""
    from PySide6.QtGui import QIcon, QKeySequence, QShortcut, QTransform
    from qfluentwidgets import FluentIcon as FIF, ToolButton
    undo, redo = ToolButton(FIF.CANCEL), ToolButton()  # FIF.CANCEL is the curved "↶" arrow
    pix = FIF.CANCEL.icon().pixmap(64, 64)
    redo.setIcon(QIcon(pix.transformed(QTransform().scale(-1, 1))))
    undo.setToolTip("Undo (Ctrl+Z)")
    redo.setToolTip("Redo (Ctrl+Y)")
    undo.clicked.connect(history.undo)
    redo.clicked.connect(history.redo)
    for key, fn in (("Ctrl+Z", history.undo), ("Ctrl+Y", history.redo), ("Ctrl+Shift+Z", history.redo)):
        QShortcut(QKeySequence(key), page, activated=fn, context=Qt.WidgetWithChildrenShortcut)
    prev = history.on_change

    def refresh() -> None:
        undo.setEnabled(history.can_undo())
        redo.setEnabled(history.can_redo())
        if prev:
            prev()

    history.on_change = refresh
    refresh()
    return undo, redo
