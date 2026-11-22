"""Leveler tab: classic or voice-aware (voice-isolator + SAVA)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, PlainTextEdit,
                            PrimaryPushButton, ProgressBar, PushButton, SegmentedWidget, Slider, SmoothScrollArea,
                            StrongBodyLabel, SubtitleLabel, SwitchButton, ToolButton)

from ..core import audio_io, leveler as lv, models, smart
from PySide6.QtWidgets import QSizePolicy

from ..i18n import tr as _tr
from .common import ACCENT, LOG_STYLE, History, LogBus, MeterWidget, Worker, format_combo, undo_buttons


class _WrapHeight(QObject):
    """Word-wrapped labels inside horizontal rows (switch rows, lists) do not pass their height-for-width to the
    layout: Qt gives them one line and the second one is clipped or drawn over the next row. This keeps each such
    label's minimum height equal to what its text needs at its current width (on resize and on text changes)."""

    def eventFilter(self, obj, ev):  # noqa: N802 - Qt name
        if ev.type() in (QEvent.Resize, QEvent.Show, QEvent.FontChange):
            sync_wrap_height(obj)
        return False


_WRAP = None


def sync_wrap_height(label) -> None:
    if not label.wordWrap() or label.maximumHeight() < 16777215 or label.width() <= 10:
        return   # fixed-height labels (fit(lines=...)) keep their height
    h = label.heightForWidth(label.width())
    if h > 0 and h != label.minimumHeight():
        label.setMinimumHeight(h)


def wrap_height(label):
    """Install the height fix on a word-wrapped label (also re-checked whenever its text changes)."""
    global _WRAP
    if _WRAP is None:
        _WRAP = _WrapHeight()
    if not label.property("_wrap_fixed"):
        label.setProperty("_wrap_fixed", True)
        label.installEventFilter(_WRAP)
        orig = label.setText

        def set_text(text, _orig=orig, _label=label):
            _orig(text)
            QTimer.singleShot(0, lambda: sync_wrap_height(_label))
        label.setText = set_text
    return label


def fit(label, lines: int = 0):
    """Label that fits the available width (word wrap), never widens its column and always gets the height of all
    its lines."""
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label.setMinimumWidth(10)
    if lines:
        h = label.fontMetrics().lineSpacing() * lines + 4
        label.setFixedHeight(h)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    else:
        wrap_height(label)
    return label


def tame(widget) -> None:
    """Make sure nothing inside a fixed-width column widens it: word-wrapped labels and combos,
    fields and buttons that can shrink (long text is clipped instead of pushing)."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QLineEdit
    for w in widget.findChildren(QLabel):
        if isinstance(w.parent(), SwitchButton) or w.maximumWidth() < 16777215:
            continue  # inner text of switches and fixed-width labels (slider values)
        w.setWordWrap(True)
        w.setSizePolicy(QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())
        w.setMinimumWidth(10)
        wrap_height(w)
    for w in widget.findChildren(QComboBox) + widget.findChildren(QLineEdit):
        w.setMinimumWidth(10)
        w.setSizePolicy(QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())
    for w in widget.findChildren(QAbstractButton):
        if w.sizePolicy().horizontalPolicy() != QSizePolicy.Fixed and w.minimumSizeHint().width() > 200:
            w.setMinimumWidth(10)
            w.setSizePolicy(QSizePolicy.Ignored, w.sizePolicy().verticalPolicy())


def switch_row(lay, text: str, on: bool):
    row = QHBoxLayout()
    lbl = fit(BodyLabel(text))
    row.addWidget(lbl, 1)
    sw = SwitchButton()
    sw.setOnText("Yes")
    sw.setOffText("No")
    sw.setChecked(on)
    row.addWidget(sw)
    lay.addLayout(row)
    return sw
from .player import Player
from .waveform import WaveView


def fmt_time(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(s, 3600)
    return f"{int(h)}:{int(rem // 60):02}:{rem % 60:05.2f}" if h else f"{int(rem // 60):02}:{rem % 60:05.2f}"


class ParamSlider(QWidget):
    """Label + slider + value, with decimals."""

    def __init__(self, title: str, lo: float, hi: float, value: float, decimals: int, parent=None) -> None:
        super().__init__(parent)
        self.k = 10 ** decimals
        self.decimals = decimals
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.addWidget(fit(StrongBodyLabel(title)), 1)
        self.val = BodyLabel()
        self.val.setFixedWidth(62)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.val)
        lay.addLayout(top)
        self.slider = Slider(Qt.Horizontal)
        self.slider.setRange(int(round(lo * self.k)), int(round(hi * self.k)))
        self.slider.valueChanged.connect(self._show)
        lay.addWidget(self.slider)
        self.set(value)

    def _show(self, v: int) -> None:
        self.val.setText(f"{v / self.k:.{self.decimals}f}")

    def get(self) -> float:
        return self.slider.value() / self.k

    def set(self, v: float) -> None:
        self.slider.setValue(int(round(v * self.k)))
        self._show(self.slider.value())


class LevelPage(QWidget):
    def __init__(self, bus: LogBus, player: Player, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("levelPage")
        self.bus, self.player = bus, player
        self.path: Path | None = None
        self.audio: np.ndarray | None = None
        self.orig: np.ndarray | None = None  # audio as loaded ("Original" goes back to it)
        self.sr = 48000
        self.integrated: dict[str, float | None] = {}
        self.worker: Worker | None = None
        self.out_paths: list[Path] = []
        self._last_result: dict | None = None  # last processing (kept in the project)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 8, 8)
        root.setSpacing(10)

        # ---------------- left column: parameters (scrollable) + process bar (fixed)
        left = QVBoxLayout()
        left.setSpacing(8)
        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent")
        panel = QWidget()
        panel.setStyleSheet("background: transparent")
        c = QVBoxLayout(panel)
        c.setContentsMargins(0, 0, 14, 8)
        c.setSpacing(6)
        title = SubtitleLabel("Dialogue leveler")
        title.setStyleSheet(f"color: {ACCENT}")
        self.title_row = QHBoxLayout()  # the main window adds the Project button here
        self.title_row.addWidget(title, 1)
        c.addLayout(self.title_row)
        self.project_box = QVBoxLayout()  # ... and the project name here
        self.project_box.setContentsMargins(0, 0, 0, 0)
        c.addLayout(self.project_box)
        self.open_btn = PushButton(FIF.FOLDER, "Select audio")
        self.open_btn.clicked.connect(self.open_file)
        c.addWidget(self.open_btn)
        self.file_lbl = fit(CaptionLabel("No file"))
        c.addWidget(self.file_lbl)
        from .youtube import YouTubeField
        self.yt = YouTubeField(self.bus, lambda ps: self.load(ps[0]), tab_busy=lambda: self.worker is not None and self.worker.isRunning())
        c.addWidget(self.yt)
        from .treatment import SeparationPanel  # local import: treatment.py imports from this module
        c.addSpacing(4)
        self.sw_sep = switch_row(c, "Isolate the voice (AI mode: levels only the voice)", True)
        self.sw_sep.setToolTip("AI-assisted mode: separates voice and background with the engine below and levels only "
                               "the voice. Off: SAVA works on the original. The classic mode does not use it")
        self.sw_sep.checkedChanged.connect(lambda *_: self._mode_changed())
        self.sep = SeparationPanel()
        self.sep.s_block.setToolTip("Longer blocks are a bit faster (~10-20 %) and use more RAM; the separator's VRAM "
                                    "does not change. Aborting waits for one block at most.")
        self.sep.run_btn.setToolTip("Separates the current audio now and keeps it as A · Original: it can be chained (e.g. music, then noise) and the result leveled afterwards")
        self.sep.run_btn.clicked.connect(self.separate_now)
        self.sep.restore_btn.clicked.connect(lambda: self.orig is not None and self._change_audio(self.orig, "original"))
        c.addWidget(self.sep)
        self.stems_box = QWidget()
        stl = QVBoxLayout(self.stems_box)
        stl.setContentsMargins(0, 0, 0, 0)
        self.sw_stems = switch_row(stl, "Save voice and background stems", False)
        c.addWidget(self.stems_box)

        c.addSpacing(6)
        c.addWidget(StrongBodyLabel("Mode"))
        self.mode = SegmentedWidget()
        self.mode.addItem("classic", "Classic")
        self.mode.addItem("smart", "AI-assisted")
        self.mode.setCurrentItem("smart")
        self.mode.currentItemChanged.connect(lambda *_: self._mode_changed())
        c.addWidget(self.mode)
        self.mode_hint = fit(CaptionLabel())
        c.addWidget(self.mode_hint)

        c.addSpacing(6)
        c.addWidget(StrongBodyLabel("Final measurement"))
        self.measure = SegmentedWidget()
        self.measure.addItem("rms", "RMS")
        self.measure.addItem("lufs", "LUFS")
        self.measure.setCurrentItem("rms")
        self.measure.currentItemChanged.connect(lambda k: self.s_final.set(-16.0 if k == "lufs" else -18.0))
        c.addWidget(self.measure)
        self.s_final = ParamSlider("Final common level (dB)", -30, -0.1, -18.0, 1)
        self.s_window = ParamSlider("Analysis window (s)", 0.1, 0.6, 0.25, 3)
        self.s_attack = ParamSlider("Attack (s)", 0.02, 0.3, 0.05, 3)
        self.s_release = ParamSlider("Release (s)", 0.05, 1.0, 0.25, 3)
        for s_ in (self.s_final, self.s_window, self.s_attack, self.s_release):
            c.addWidget(s_)

        # ---- AI-assisted: voice detection (SAVA) and optional separation
        self.smart_box = QWidget()
        sb = QVBoxLayout(self.smart_box)
        sb.setContentsMargins(0, 6, 0, 0)
        sb.setSpacing(6)
        sb.addWidget(StrongBodyLabel("Voice detection"))
        self.sw_sava = switch_row(sb, "Detect voice with SAVA", True)
        self.sw_sava.setToolTip("SAVA (AI voice activity) marks where there is speech. Off: speech is detected from "
                                "the energy of the voice; fine on a clean separated voice, no extra model or VRAM")
        self.sw_sava.checkedChanged.connect(lambda *_: self._mode_changed())
        self.s_thr = ParamSlider("Voice threshold", 0.1, 0.9, 0.5, 2)
        self.s_hang = ParamSlider("Margin around the voice (s)", 0.0, 0.5, 0.15, 2)
        sb.addWidget(self.s_thr)
        sb.addWidget(self.s_hang)
        self.sw_sava_auto = switch_row(sb, "Automatic SAVA window", True)
        self.sw_sava_auto.setToolTip("The longest window that fits in the free VRAM when the process starts "
                                     "(longer = a bit faster)")
        self.sw_sava_auto.checkedChanged.connect(lambda *_: self._mode_changed())
        self.s_sava = ParamSlider("SAVA window (min)", 1, 60, 10, 0)
        self.s_sava.slider.setMaximum(max(1, models.sava_max_window_min()))  # never more than what fits in VRAM
        self.sava_hint = fit(CaptionLabel())
        self.s_sava.slider.valueChanged.connect(lambda *_: self._sava_hint())
        sb.addWidget(self.s_sava)
        sb.addWidget(self.sava_hint)
        self._sava_hint()
        sb.addSpacing(6)
        # visible with and without separation (without separation they act on the pauses SAVA marks)
        self.s_bg = ParamSlider("Background vs. original (dB)", -40, 6, 0.0, 1)
        sb.addWidget(self.s_bg)
        self.sw_remove = switch_row(sb, "Remove the background", False)
        self.sw_remove.checkedChanged.connect(lambda on: self.s_bg.setEnabled(not on))
        self.bg_hint = fit(CaptionLabel())
        sb.addWidget(self.bg_hint)
        c.addWidget(self.smart_box)
        c.addSpacing(6)
        c.addWidget(StrongBodyLabel("Output format"))
        self.fmt = format_combo("wav24")
        c.addWidget(self.fmt)
        c.addStretch(1)
        scroll.setWidget(panel)
        tame(panel)
        left.addWidget(scroll, 1)

        # ---- process bar: always visible, never scrolls or widens with the messages
        run = QWidget()
        run.setObjectName("runCard")
        run.setStyleSheet("#runCard {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);"
                          " border-radius: 8px}")
        rl = QVBoxLayout(run)
        rl.setContentsMargins(12, 10, 12, 10)
        rl.setSpacing(6)
        self.progress = ProgressBar()
        self.progress.setRange(0, 1000)  # tenths of a point: on hours-long audio the bar always moves
        self.progress.setValue(0)
        rl.addWidget(self.progress)
        self.status = fit(CaptionLabel("Ready"), lines=3)
        rl.addWidget(self.status)
        btns = QHBoxLayout()
        self.run_btn = PrimaryPushButton(FIF.PLAY_SOLID, "Process")
        self.run_btn.clicked.connect(lambda: self.start_run())
        self.abort_btn = PushButton(FIF.CANCEL, "Abort")
        self.abort_btn.clicked.connect(self.abort)
        btns.addWidget(self.run_btn, 3)
        btns.addWidget(self.abort_btn, 2)
        rl.addLayout(btns)
        left.addWidget(run)
        holder = QWidget()
        holder.setLayout(left)
        holder.setMinimumWidth(320)  # fluid column: shrinks on small screens
        holder.setMaximumWidth(380)
        left.setContentsMargins(0, 0, 0, 0)
        root.addWidget(holder)

        # ---------------- centro: vista + transporte
        center = QVBoxLayout()
        center.setSpacing(6)
        bar = QHBoxLayout()
        self.view = SegmentedWidget()
        for k, t in (("sup", "Overlaid"), ("orig", "Original"), ("out", "Leveled"), ("voice", "Voice / background")):
            self.view.addItem(k, t)
        self.view.setCurrentItem("orig")
        self.view.currentItemChanged.connect(lambda *_: self._apply_view())
        bar.addWidget(self.view)
        bar.addSpacing(12)
        for icon, tip, fn in ((FIF.ZOOM_OUT, "Zoom out (−)", lambda: self.wave.zoom(1.5)),
                              (FIF.ZOOM_IN, "Zoom in (+)", lambda: self.wave.zoom(1 / 1.5)),
                              (FIF.FIT_PAGE, "Fit", lambda: self.wave.fit())):
            b = ToolButton(icon)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        self._audio_what = "original"
        self.history = History(self._snapshot, self._restore, limit=10)  # whole audios: few steps
        for b in undo_buttons(self, self.history):
            b.setToolTip(_tr(b.toolTip()) + " · " + _tr("separation chain"))
            self.sep.row.addWidget(b)
        bar.addStretch(1)
        self.time_lbl = BodyLabel("00:00.00 / 00:00.00")
        self.time_lbl.setStyleSheet("font-family: Consolas; font-size: 15px")
        bar.addWidget(self.time_lbl)
        center.addLayout(bar)
        self.wave = WaveView()
        self.wave.seekRequested.connect(self.seek_seconds)
        center.addWidget(self.wave, 1)

        tr = QHBoxLayout()
        self.play_btn = PrimaryPushButton(FIF.PLAY, "Play")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.clicked.connect(self.toggle_play)
        tr.addWidget(self.play_btn)
        stop = PushButton(FIF.PAUSE, "Stop")
        stop.clicked.connect(self.stop)
        tr.addWidget(stop)
        tr.addSpacing(12)
        self.source = SegmentedWidget()
        for k, t in (("A", "A · Original"), ("B", "B · Leveled"), ("voice", "Voice"), ("bg", "Background")):
            self.source.addItem(k, t)
        self.source.setCurrentItem("A")
        self.source.currentItemChanged.connect(self._source_changed)
        tr.addWidget(self.source)
        tr.addStretch(1)
        center.addLayout(tr)
        hint = fit(CaptionLabel("Space: play · ←/→: 5 s · Home · A: toggle A/B · Click: seek · Drag: scroll · Wheel: zoom · Ctrl+Wheel: height"))
        center.addWidget(hint)
        logbar = QHBoxLayout()
        logbar.addWidget(StrongBodyLabel("Log"))
        logbar.addStretch(1)
        self.log_toggle = ToolButton(FIF.CHEVRON_DOWN_MED)
        self.log_toggle.setToolTip("Show or hide the log")
        self.log_toggle.clicked.connect(lambda: self._toggle_log())
        logbar.addWidget(self.log_toggle)
        center.addLayout(logbar)
        self.log_view = PlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(150)
        self.log_view.setLineWrapMode(PlainTextEdit.NoWrap)
        self.log_view.setStyleSheet(LOG_STYLE)
        center.addWidget(self.log_view)
        root.addLayout(center, 1)

        self.meter = MeterWidget()
        root.addWidget(self.meter)

        for key, fn in (("Space", self.toggle_play), ("Left", lambda: self.nudge(-5)), ("Right", lambda: self.nudge(5)),
                        ("Home", lambda: self.seek_seconds(0)), ("A", self.toggle_ab),
                        ("+", lambda: self.wave.zoom(1 / 1.5)), ("-", lambda: self.wave.zoom(1.5))):
            QShortcut(QKeySequence(key), self, activated=fn, context=Qt.WidgetWithChildrenShortcut)

        self.bus.line.connect(self._log_line)
        self.timer = QTimer(self, interval=33, timeout=self._tick)
        self.timer.start()
        self._mode_changed()
        self._update_enabled()

    # ------------------------------------------------------------ util
    def _sava_hint(self) -> None:
        if getattr(self, "sw_sava_auto", None) is not None and self.sw_sava_auto.isChecked():
            from ..core import memory
            m = memory.sava_window_s() / 60
            gb = models.SAVA_GB_BASE + models.SAVA_GB_PER_MIN * m
            self.sava_hint.setText(f"Now: {m:.0f} min (≈ {gb:.1f} GB of VRAM) · {memory.describe()}")
            return
        m = self.s_sava.get()
        gb = models.SAVA_GB_BASE + models.SAVA_GB_PER_MIN * m
        self.sava_hint.setText(f"≈ {gb:.1f} GB of VRAM. Long windows are faster (10 min ≈ 0.3 s per 10 min of audio); the limit is set by free VRAM ({self.s_sava.slider.maximum()} min).")

    def _toggle_log(self) -> None:
        on = not self.log_view.isVisible()
        self.log_view.setVisible(on)
        self.log_toggle.setIcon(FIF.CHEVRON_DOWN_MED if on else FIF.UP)

    def _log_line(self, text: str, level: int) -> None:
        color = {1: "#fbbf24", 2: "#f87171"}.get(level)
        self.log_view.appendHtml(f'<span style="color:{color}">{text}</span>' if color else text.replace("<", "&lt;"))

    def _info(self, title: str, text: str, error: bool = False) -> None:
        (InfoBar.error if error else InfoBar.success)(title, text, duration=4000 if not error else 8000,
                                                     position=InfoBarPosition.TOP_RIGHT, parent=self.window())

    def _mode_changed(self) -> None:
        smart_on = self.mode.currentRouteKey() == "smart"
        sep_on = self.sw_sep.isChecked()
        self.smart_box.setVisible(smart_on)
        self.sw_sep.setEnabled(smart_on)          # the classic mode levels the whole audio: no separation
        self.stems_box.setVisible(smart_on and sep_on)
        self.sw_sava_auto.setEnabled(self.sw_sava.isChecked())
        self.s_sava.setVisible(self.sw_sava.isChecked())
        self.s_sava.setEnabled(not self.sw_sava_auto.isChecked())
        self.sava_hint.setVisible(self.sw_sava.isChecked())
        self._sava_hint()
        self.bg_hint.setText("The separated background keeps its original level plus this offset; \"Remove\" drops it." if sep_on
                             else "Without separation, the offset applies to voiceless pauses (SAVA); \"Remove\" fades them to silence: it cleans pause noise without other models.")
        if not smart_on:
            text = "Classic leveling of the whole audio (voice and noise alike)."
        elif sep_on:
            text = ("Separates voice from background, detects where there is speech (SAVA) and levels only the voice: gain freezes in pauses and the background is not raised.")
        else:
            text = ("SAVA detects speech in the original and levels only those parts: gain freezes in pauses. Ideal when there is little or no background; much faster.")
        self.mode_hint.setText(text)

    def _update_enabled(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        self.open_btn.setEnabled(not busy)
        self.yt.set_tab_busy(busy)
        self.run_btn.setEnabled(not busy and self.audio is not None)
        self.abort_btn.setEnabled(busy)
        self.sep.run_btn.setEnabled(not busy and self.audio is not None)
        self.sep.restore_btn.setEnabled(not busy and self.orig is not None and self.audio is not self.orig)
        has = set(self.player.sources)
        for k in ("B", "voice", "bg"):
            item = self.source.items.get(k)
            if item is not None:
                item.setEnabled(k in has)

    def _start(self, fn, on_done, label: str) -> None:
        self.worker = Worker(fn, self)
        self._stage = ""

        def on_log(msg: str, level: int) -> None:
            self.bus.push(msg, level)
            if level == 0 and "\n" not in msg:
                self._stage = msg  # current stage with its progress and time left
                self.status.setText(f"{label} · total {self.progress.value() / 10:.1f}% · {msg}")

        self.worker.progress.connect(lambda v: (self.progress.setValue(int(v * 1000)),
                                                self.status.setText(f"{label} · total {v * 100:.1f}%" +
                                                                    (f" · {self._stage}" if self._stage else ""))))
        self.worker.log.connect(on_log)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(lambda e: (self.status.setText("Error (see log)"), self._info("Error", e, True)))
        self.worker.cancelled.connect(lambda: (self.status.setText("Process aborted"), self.progress.setValue(0),
                                               self.bus.push("Process aborted by the user", 1)))
        self.worker.finished.connect(self._update_enabled)
        self.status.setText(f"{label}...")
        self.progress.setValue(0)
        self.worker.start()
        self._update_enabled()

    # ------------------------------------------------------------ carga
    def open_file(self) -> None:
        start = str(self.path.parent) if self.path else ""
        f, _ = QFileDialog.getOpenFileName(self, "Select audio", start,
                                           "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All (*.*)")
        if f:
            self.load(Path(f))

    def load(self, path: Path) -> None:
        self.player.pause()
        self.bus.push(f"Loading {path}")

        def job(ctx):
            t = time.perf_counter()
            x, sr = audio_io.load(path)
            ctx.progress(0.5)
            try:
                integ = lv.lufs_integrated(lv.k_subblocks(x, sr))
            except ValueError:
                integ = None
            return path, x, sr, integ, time.perf_counter() - t

        self._start(job, self._loaded, "Loading")

    def _loaded(self, res) -> None:
        path, x, sr, integ, secs = res
        self.path, self.audio, self.sr = path, x, sr
        self.orig = x
        self._audio_what = "original"
        self.history.clear()
        self.integrated = {"A": integ}
        self.out_paths = []
        self._last_result = None
        self.file_lbl.setText(f"{path.name}\n{x.shape[0] / sr / 60:.1f} min · {sr} Hz · {x.shape[1]} channel(s)")
        self.wave.clear()
        self.wave.set_layer("orig", audio_io.to_mono(x), sr, "#9a9a9a", "fill", z=0)
        self.wave.set_speech(None)
        self.player.set_sources(sr, {"A": x}, "A")
        self.player.seek(0)
        self.source.setCurrentItem("A")
        self.view.setCurrentItem("orig")  # on import: original view and listen to A
        self._apply_view()
        self.wave.fit()
        self.progress.setValue(0)
        self.status.setText("Ready. Process the audio.")
        self.bus.push(f"Audio loaded in {secs:.1f} s · integrated LUFS {integ:.2f}" if integ is not None else
                      f"Audio loaded in {secs:.1f} s")
        self._update_enabled()

    # ------------------------------------------------------------ project
    def project_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def project_files(self) -> list[str]:
        return [str(self.path)] if self.path else []

    def project_state(self) -> dict:
        from .projects import capture_controls
        return {"controls": capture_controls(self, {"view", "source"}), "path": str(self.path) if self.path else None,
                "audio": None if self.audio is self.orig else self.audio, "what": self._audio_what,
                "result": self._last_result, "view": self.view.currentRouteKey(), "source": self.source.currentRouteKey()}

    def project_apply(self, st: dict) -> None:
        from .projects import apply_controls
        self.project_clear()
        apply_controls(self, st.get("controls", {}))
        if not st.get("path"):
            return
        path = Path(st["path"])
        x, sr = audio_io.load(path)
        try:
            integ = lv.lufs_integrated(lv.k_subblocks(x, sr))
        except ValueError:
            integ = None
        self._loaded((path, x, sr, integ, 0.0))
        if st.get("audio") is not None:
            self._set_audio(st["audio"], st.get("what", "separated"))
        res = st.get("result")
        if res and getattr(res, "output", None) is not None:   # None: its audio was not in the media folder
            self._processed(res, quiet=True)
            self.status.setText("Project opened")
        for w, key in ((self.view, "view"), (self.source, "source")):
            if st.get(key) in w.items:
                w.setCurrentItem(st[key])

    def project_clear(self) -> None:
        self.player.pause()
        self.path = self.audio = self.orig = None
        self.integrated, self.out_paths, self._last_result, self._audio_what = {}, [], None, "original"
        self.history.clear()
        self.wave.clear()
        self.wave.set_speech(None)
        self.player.set_sources(self.sr, {})
        self.file_lbl.setText(_tr("No file"))
        self.progress.setValue(0)
        self.status.setText(_tr("Ready"))
        self._update_enabled()

    # ------------------------------------------------------------ separation (chainable)
    def separate_now(self) -> None:
        if self.audio is None:
            return
        from .treatment import run_separation
        x, sr, name = self.audio, self.sr, self.sep.engine_name()
        run_separation(self, [(self.path.name, lambda: (x, sr))], self.sep,
                       lambda res: self._change_audio(res[0][0], f"separated with {name}"))

    def _change_audio(self, y: np.ndarray, what: str) -> None:
        """A step of the separation chain (can be undone)."""
        self.history.record()
        self._set_audio(y, what)

    def _snapshot(self):
        return self.audio, self._audio_what

    def _restore(self, state) -> None:
        y, what = state
        if y is not None and y is not self.audio:
            self._set_audio(y, what)

    def _set_audio(self, y: np.ndarray, what: str) -> None:
        """The result becomes A · Original: it shows on the waveform and anything processed afterwards starts from it."""
        self._audio_what = what
        self.player.pause()
        pos = self.player.pos
        self.audio = y
        try:
            self.integrated = {"A": lv.lufs_integrated(lv.k_subblocks(y, self.sr))}
        except ValueError:
            self.integrated = {"A": None}
        self.out_paths = []
        self._last_result = None
        self.wave.clear()
        self.wave.set_layer("orig", audio_io.to_mono(y), self.sr, "#9a9a9a", "fill", z=0)
        self.wave.set_speech(None)
        self.player.set_sources(self.sr, {"A": y}, "A")
        self.player.seek(min(pos, y.shape[0]))
        self.source.setCurrentItem("A")
        self.view.setCurrentItem("orig")
        self._apply_view()
        self.progress.setValue(self.progress.maximum())
        self.status.setText(f"Audio {what}. You can separate again (chain) or process.")
        self.bus.push(f"{self.path.name}: audio {what}")
        self._update_enabled()

    # ------------------------------------------------------------ processing
    def start_run(self, out_dir: str | None = None) -> None:
        if self.audio is None:
            return
        out_dir = out_dir or QFileDialog.getExistingDirectory(self, "Choose output folder", str(self.path.parent))
        if not out_dir:
            self.bus.push("Processing cancelled: no output folder chosen", 1)
            return
        x, sr, path = self.audio, self.sr, self.path
        lufs = self.measure.currentRouteKey() == "lufs"
        base = dict(window_s=self.s_window.get(), attack_s=self.s_attack.get(), release_s=self.s_release.get(),
                    lufs=lufs, final_db=self.s_final.get())
        smart_on = self.mode.currentRouteKey() == "smart"
        stems = self.sw_stems.isChecked()
        prm = smart.SmartParams(**base, engine=self.sep.engine_key(),
                                separate=self.sw_sep.isChecked(), use_sava=self.sw_sava.isChecked(),
                                sava_window_s=None if self.sw_sava_auto.isChecked() else self.s_sava.get() * 60,
                                sep_block_s=self.sep.block_s(),
                                speech_threshold=self.s_thr.get(), hangover_s=self.s_hang.get(),
                                background_db=self.s_bg.get(), remove_background=self.sw_remove.isChecked()) \
            if smart_on else lv.Params(**base)
        fmt = self.fmt.key()
        ext = audio_io.ext_for(fmt)
        out = Path(out_dir) / f"{path.stem}_nivelado{'_ia' if smart_on else ''}{ext}"
        if out.resolve() == path.resolve():
            self._info("Error", "The output matches the input; choose another folder", True)
            return
        self.bus.push(f"Processing ({"voice-aware" if smart_on else "classic"}) → {out}")
        self.player.remove_source("B")
        self.player.remove_source("voice")
        self.player.remove_source("bg")

        def job(ctx):
            t = time.perf_counter()
            if smart_on:
                r = smart.run_smart(x, sr, prm, ctx)
                y, gain, voice, bg, speech = r.output, r.voice_gain, r.voice, r.background, r.speech
            else:
                r = lv.run_pipeline(x, sr, prm, ctx)
                y, gain, voice, bg, speech = r.output, r.gain, None, None, None
            audio_io.write_audio(out, y, sr, fmt)
            files = [out]
            if smart_on and stems:
                for name, data in (("voice", voice), ("background", bg)):
                    p = out.with_name(f"{path.stem}_{name}{ext}")
                    audio_io.write_audio(p, data, sr, fmt)
                    files.append(p)
            try:
                integ = lv.lufs_integrated(lv.k_subblocks(y, sr))
            except ValueError:
                integ = None
            return dict(y=y, gain=gain, voice=voice, bg=bg, speech=speech, files=files, integ=integ,
                        secs=time.perf_counter() - t, smart=smart_on,
                        separated=smart_on and prm.separate)

        self._start(job, self._processed, "Processing")

    def _processed(self, r: dict, quiet: bool = False) -> None:
        sr = self.sr
        self._last_result = r
        self.out_paths = r["files"]
        self.integrated["B"] = r["integ"]
        sources = {"A": self.audio, "B": r["y"]}
        self.wave.set_layer("out", audio_io.to_mono(r["y"]), sr, ACCENT, "fill", z=1)
        self.wave.set_layer("orig", audio_io.to_mono(self.audio), sr, "#ffffff", "outline", z=2)
        if r["smart"] and r.get("separated"):
            sources["voice"], sources["bg"] = r["voice"], r["bg"]
            self.wave.set_layer("voice", audio_io.to_mono(r["voice"]), sr, "#60a5fa", "fill", z=1)
            self.wave.set_layer("bg", audio_io.to_mono(r["bg"]), sr, "#f59e0b", "fill", z=2)
        else:
            self.wave.remove_layer("voice")
            self.wave.remove_layer("bg")
        self.wave.set_speech(r["speech"], models.SAVA_FRAME_S) if r["speech"] is not None else self.wave.set_speech(None)
        g = r["gain"]
        self.wave.set_gain((np.arange(g.db.size) + 0.5) * g.hop / sr, g.db)
        pos = self.player.pos
        self.player.set_sources(sr, sources, "B")
        self.player.seek(pos)
        # after rendering: overlaid view and listen to B · Leveled
        self.view.setCurrentItem("sup")
        self.source.setCurrentItem("B")
        self._apply_view()
        self.progress.setValue(1000)
        self.status.setText(f"Done in {r['secs']:.1f} s")
        if not quiet:
            self.bus.push(f"Finished in {r['secs']:.1f} s · output integrated LUFS {r['integ']:.2f}"
                          if r["integ"] is not None else f"Finished in {r['secs']:.1f} s")
            self._info("Ready", "Saved: " + ", ".join(p.name for p in r["files"]))
        self._update_enabled()

    def abort(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("Aborting: stops when the current step ends (seconds)...")
            self.bus.push("Aborting at the user's request...", 1)

    # ------------------------------------------------------------ vista y transporte
    def _apply_view(self) -> None:
        mode = self.view.currentRouteKey()
        has_out = "out" in self.wave.layers
        show = {
            "sup": {"orig", "out"}, "orig": {"orig"}, "out": {"out"}, "voice": {"voice", "bg"},
        }[mode]
        if not has_out:
            show = {"orig"}
        # the original is filled only when shown alone
        if "orig" in self.wave.layers and self.audio is not None:
            style = "outline" if (mode == "sup" and has_out) else "fill"
            if self.wave.layers["orig"]["style"] != style:
                self.wave.set_layer("orig", audio_io.to_mono(self.audio), self.sr,
                                    "#ffffff" if style == "outline" else "#9a9a9a", style, z=2 if style == "outline" else 0)
        for name in list(self.wave.layers):
            self.wave.set_visible(name, name in show)

    def _source_changed(self, key: str) -> None:
        if key in self.player.sources:
            self.player.set_active(key)
        else:
            self.source.setCurrentItem(self.player.active or "A")

    def toggle_ab(self) -> None:
        if "B" in self.player.sources:
            self.source.setCurrentItem("A" if self.player.active != "A" else "B")

    def toggle_play(self) -> None:
        if not self.player.sources:
            return
        self.player.pause() if self.player.playing else self.player.play()

    def stop(self) -> None:
        self.player.stop()
        self.wave.set_cursor(0.0)

    def seek_seconds(self, s: float) -> None:
        self.player.seek(int(s * self.player.sr))
        self.wave.set_cursor(s)

    def nudge(self, secs: float) -> None:
        self.seek_seconds(self.player.pos / self.player.sr + secs)

    def _tick(self) -> None:
        if not self.isVisible():
            return
        sr = self.player.sr
        if self.player.sources:
            t = self.player.position() / sr
            if self.player.playing:
                self.wave.set_cursor(t, follow=True)
            self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.player.length / sr)}")
        playing = self.player.playing
        want = _tr("Pause") if playing else _tr("Play")
        if self.play_btn.text() != want:
            self.play_btn.setText(want)
            self.play_btn.setIcon(FIF.PAUSE if playing else FIF.PLAY)
        self.meter.set_values(self.player.meter, self.integrated.get(self.player.active))
