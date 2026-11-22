"""Voice dataset tab: naturally cut clips + text (CrisperWhisper 2.0 turbo).

Detected clips show as regions on the waveform: drag them to move them and pull their edges to adjust the
in or out point. On release, the clip is rebuilt (audio, text and checks).
"""
from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QTableWidgetItem,
                               QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, LineEdit,
                            ListWidget, PlainTextEdit, PrimaryPushButton, ProgressBar, PushButton, SegmentedWidget,
                            SmoothScrollArea, StrongBodyLabel, SubtitleLabel, SwitchButton, TableWidget)

from ..core import audio_io, dataset, models
from ..i18n import tr as _tr
from .common import ACCENT, DANGER, History, LogBus, Worker, format_combo, LOG_STYLE, undo_buttons
from .level_page import ParamSlider, fit, tame
from .treatment import SeparationPanel, run_separation
from .player import Player
from .waveform import WaveView

COLS = ["✓", "File", "Speaker", "Start", "Dur. (s)", "Text", "Conf.", "Voice", "Reason"]
TEXT_COL = 5
OTHER = _tr("other speaker")
ALL = _tr("All speakers")


def switch_row(lay, text: str, on: bool) -> SwitchButton:
    row = QHBoxLayout()
    lbl = BodyLabel(text)
    lbl.setWordWrap(True)
    row.addWidget(lbl, 1)
    sw = SwitchButton()
    sw.setOnText("Yes")
    sw.setOffText("No")
    sw.setChecked(on)
    row.addWidget(sw)
    lay.addLayout(row)
    return sw


def side_panel(width: int = 380) -> tuple[SmoothScrollArea, QVBoxLayout]:
    scroll = SmoothScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(width - 80)  # shrinks on small screens
    scroll.setMaximumWidth(width)
    scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("background: transparent")
    panel = QWidget()
    panel.setStyleSheet("background: transparent")
    c = QVBoxLayout(panel)
    c.setContentsMargins(0, 0, 18, 8)
    c.setSpacing(8)
    scroll.setWidget(panel)
    return scroll, c


class DatasetPage(QWidget):
    def __init__(self, bus: LogBus, store=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("datasetPage")
        self.bus, self.store = bus, store
        self.player = Player()
        self.clips: list[dataset.Clip] = []
        self.sources: dict[str, dataset.Source] = {}
        self.shown: str | None = None           # file shown on the waveform
        self.region_clips: list[int] = []      # clip index per visible region
        self.worker: Worker | None = None
        self.prm = dataset.DatasetParams()
        self._playing_clip: dataset.Clip | None = None
        self.pre: dict[str, tuple] = {}  # file → (separated audio, sr) from "Separate now"

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 8, 8)
        root.setSpacing(10)
        scroll, c = side_panel()
        t = SubtitleLabel("Voice dataset")
        t.setStyleSheet(f"color: {ACCENT}")
        self.title_row = QHBoxLayout()  # the main window adds the Project button here
        self.title_row.addWidget(t, 1)
        c.addLayout(self.title_row)
        self.project_box = QVBoxLayout()  # ... and the project name here
        self.project_box.setContentsMargins(0, 0, 0, 0)
        c.addLayout(self.project_box)
        intro = CaptionLabel("Voice clips cut at natural pauses, with verbatim transcription and word timestamps (CrisperWhisper 2.0 turbo). Adjust clips by dragging their edges on the waveform.")
        intro.setWordWrap(True)
        c.addWidget(intro)
        row = QHBoxLayout()
        add = PushButton(FIF.ADD, "Add audio files")
        add.clicked.connect(lambda: self.add_files())
        rem = PushButton(FIF.DELETE, "Remove")
        rem.clicked.connect(self.remove_files)
        row.addWidget(add)
        row.addWidget(rem)
        c.addLayout(row)
        self.files = ListWidget()
        self.files.setFixedHeight(100)
        c.addWidget(self.files)
        from .youtube import YouTubeField
        self.yt = YouTubeField(self.bus, lambda ps: self.add_files([str(p) for p in ps]), multi=True)
        c.addWidget(self.yt)
        c.addWidget(StrongBodyLabel("Language"))
        self.lang = ComboBox()
        self.lang.addItems(list(models.CW_LANGUAGES))
        self.lang.setCurrentText("English")
        c.addWidget(self.lang)
        self.sw_iso = switch_row(c, "Isolate the voice first", True)
        self.sep = SeparationPanel()
        self.sep.run_btn.setToolTip("Separates the selected files (or all of them) and shows the result on the waveform; the analysis then uses the separated audio")
        self.sep.run_btn.clicked.connect(self.separate_now)
        self.sep.restore_btn.clicked.connect(self.restore_original)
        c.addWidget(self.sep)
        self.sw_diar = switch_row(c, "Split by speaker (cut at turn changes)", False)
        self.sw_verb = switch_row(c, "Verbatim transcription", True)
        self.sw_fill = switch_row(c, "Keep fillers [UM] [UH]", False)
        self.sw_hall = switch_row(c, "Mitigate hallucinations (slower)", True)
        self.sw_norm = switch_row(c, "Normalize internally before processing", True)
        self.sw_align = switch_row(c, "Align words (more precise edges)", True)
        self.sw_sava = switch_row(c, "Detect pauses with SAVA", False)
        self.sw_sava.setToolTip("SAVA (voice activity) finds the real silences to cut at. Turn it off when music or "
                                "effects stay under the voice: pauses are then taken from the word timestamps")
        from ..core import aligner as _al
        self._aligner_keys = list(_al.ALIGNERS)
        self.aligner_cb = ComboBox()
        self.aligner_cb.setMinimumWidth(10)
        self.aligner_cb.addItems([_al.ALIGNERS[k] for k in self._aligner_keys])
        self.aligner_hint = fit(CaptionLabel(_al.ALIGNER_HINTS[self._aligner_keys[0]]))
        self.aligner_cb.currentIndexChanged.connect(lambda i: self.aligner_hint.setText(
            _al.ALIGNER_HINTS[self._aligner_keys[i]]))
        self.sw_align.checkedChanged.connect(lambda on: (self.aligner_cb.setEnabled(on), self.aligner_hint.setEnabled(on)))
        self.aligner_cb.setCurrentIndex(self._aligner_keys.index("qwen"))  # the most precise one on word endings
        c.addWidget(self.aligner_cb)
        c.addWidget(self.aligner_hint)
        self.s_min = ParamSlider("Minimum duration (s)", 0.5, 10, 2.0, 1)
        self.s_max = ParamSlider("Maximum duration (s)", 3, 30, 12.0, 1)
        self.s_pause = ParamSlider("Minimum pause to cut (s)", 0.05, 1.0, 0.2, 2)
        self.s_pad = ParamSlider("Silence padding (s)", 0.0, 0.5, 0.12, 2)
        self.s_conf = ParamSlider("Minimum word confidence", 0.0, 1.0, 0.2, 2)
        for s in (self.s_min, self.s_max, self.s_pause, self.s_pad, self.s_conf):
            c.addWidget(s)
        c.addWidget(StrongBodyLabel("Output sample rate"))
        self.out_sr = ComboBox()
        self.out_sr.addItems(["16000", "22050", "24000", "44100", "48000"])
        self.out_sr.setCurrentText("24000")
        c.addWidget(self.out_sr)
        self.speaker_lbl = StrongBodyLabel("Speaker")
        c.addWidget(self.speaker_lbl)
        self.speaker = LineEdit()
        self.speaker.setText("speaker")
        c.addWidget(self.speaker)
        # the speaker name goes with the speakers option: hidden while it is off
        self.sw_diar.checkedChanged.connect(lambda on: (self.speaker_lbl.setVisible(on), self.speaker.setVisible(on)))
        self.speaker_lbl.setVisible(self.sw_diar.isChecked())
        self.speaker.setVisible(self.sw_diar.isChecked())
        self.progress = ProgressBar()
        self.progress.setValue(0)
        c.addWidget(self.progress)
        self.status = fit(CaptionLabel("Ready"), lines=3)
        c.addWidget(self.status)
        self.run_btn = PrimaryPushButton(FIF.SYNC, "Analyze and cut")
        self.run_btn.clicked.connect(self.analyze)
        self.abort_btn = PushButton(FIF.CANCEL, "Abort")
        self.abort_btn.setToolTip("Stops the running process (analysis, separation, export...) within seconds")
        self.abort_btn.clicked.connect(self.abort)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.abort_btn, 2)
        c.addLayout(run_row)
        c.addWidget(StrongBodyLabel("Dataset layout (TTS project)"))
        self.layout_keys = list(dataset.EXPORT_LAYOUTS)
        self.layout_cb = ComboBox()
        self.layout_cb.setMinimumWidth(10)
        self.layout_cb.addItems([dataset.EXPORT_LAYOUTS[k] for k in self.layout_keys])
        self.layout_hint = fit(CaptionLabel(dataset.LAYOUT_HINTS[self.layout_keys[0]]))
        self.layout_cb.currentIndexChanged.connect(lambda i: self.layout_hint.setText(
            dataset.LAYOUT_HINTS[self.layout_keys[i]]))
        c.addWidget(self.layout_cb)
        c.addWidget(self.layout_hint)
        self.s_eval = ParamSlider("Clips for evaluation (%)", 0, 20, 5, 0)
        c.addWidget(self.s_eval)
        c.addWidget(StrongBodyLabel("Clip format"))
        self.fmt = format_combo("wav16")
        c.addWidget(self.fmt)
        c.addWidget(StrongBodyLabel("Output"))
        self.package = SegmentedWidget()
        self.package.addItem("zip", "ZIP file")
        self.package.addItem("folder", "Folder (uncompressed)")
        self.package.setCurrentItem("zip")
        c.addWidget(self.package)
        self.export_btn = PrimaryPushButton(FIF.SAVE, "Export dataset")
        self.export_btn.clicked.connect(lambda: self.export())
        c.addWidget(self.export_btn)
        lic = CaptionLabel("License: CrisperWhisper 2.0 transcripts are for non-commercial use only.")
        lic.setWordWrap(True)
        c.addWidget(lic)
        c.addStretch(1)
        tame(scroll.widget())
        root.addWidget(scroll)

        center = QVBoxLayout()
        self.wave = WaveView()
        self.wave.setMaximumHeight(300)
        self.wave.aux.hide()
        self.wave.seekRequested.connect(self._seek)
        self.wave.regionClicked.connect(self._region_clicked)
        self.wave.regionCtrlClicked.connect(self._region_ctrl_clicked)
        self.wave.regionChanged.connect(self._region_changed)
        center.addWidget(self.wave)
        # row 1: history, summary, autoplay and speaker filter · row 2: clip actions
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.history = History(self._snapshot, self._restore)
        for b in undo_buttons(self, self.history):
            bar.addWidget(b)
        self.include_all_btn = PushButton(FIF.ACCEPT_MEDIUM, "Include all")
        self.include_all_btn.setToolTip("Include every clip that was excluded automatically or by hand")
        self.include_all_btn.clicked.connect(self.include_all)
        bar.addWidget(self.include_all_btn)
        bar.addSpacing(6)
        self.summary = BodyLabel("No clips")
        bar.addWidget(self.summary)
        bar.addStretch(1)
        bar.addWidget(BodyLabel("Autoplay clips"))
        self.sw_play = SwitchButton()
        self.sw_play.setOnText("Yes")
        self.sw_play.setOffText("No")
        self.sw_play.setChecked(True)
        self.sw_play.setToolTip("Play a clip as soon as it is selected (never with a multiple selection)")
        bar.addWidget(self.sw_play)
        bar.addSpacing(12)
        self.spk_filter = ComboBox()   # "All speakers" / one speaker
        self.spk_filter.setToolTip("Show and keep only the clips of this speaker")
        self.spk_filter.setMinimumWidth(150)
        self.spk_filter.addItem(ALL)
        self.spk_filter.currentTextChanged.connect(self._filter_speaker)
        bar.addWidget(self.spk_filter)
        center.addLayout(bar)
        acts = QHBoxLayout()
        acts.setSpacing(6)
        for icon, text, fn, primary, tip in (
                (FIF.PLAY, "Play clip", self.play_selected, True, ""),
                (FIF.ACCEPT, "Include / exclude", self.toggle_selected, False,
                 "Include or exclude the selected clips; with several selected, if any is excluded all are "
                 "included, otherwise all are excluded (X)"),
                (FIF.ADD, "New clip", self.new_clip, False, ""),
                (FIF.CUT, "Split at playhead", self.split_at_playhead, False,
                 "Cut the clip under the playhead (or the selected one) in two at the playhead (S)"),
                (FIF.LINK, "Merge clips", self.merge_selected, False,
                 "Select two or more clips of the same file (Ctrl+click, Shift+click) and join them into one: "
                 "times and transcript (Ctrl+M)"),
                (FIF.DELETE, "Delete clip", self.delete_selected, False, "")):
            b = (PrimaryPushButton if primary else PushButton)(icon, text)
            b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            acts.addWidget(b)
            if fn == self.merge_selected:
                self.merge_btn = b
            elif fn == self.split_at_playhead:
                self.split_btn = b
        acts.addStretch(1)
        center.addLayout(acts)
        self.table = TableWidget()
        self.table.setColumnCount(len(COLS))
        self.table.setHorizontalHeaderLabels([f"  {c}  " for c in COLS])  # padding: the Fluent style uses a wider font
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)   # Ctrl/Shift+click: several (merge)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        for i in range(len(COLS)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(TEXT_COL, QHeaderView.Stretch)
        self.table.setWordWrap(True)  # long text wraps instead of being clipped with "..."
        self.table.setTextElideMode(Qt.ElideNone)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.itemChanged.connect(self._edited)
        center.addWidget(self.table, 1)
        center.addWidget(StrongBodyLabel("Full clip transcript"))
        self.full_text = PlainTextEdit()
        self.full_text.setPlaceholderText("Select a clip to see (and correct) its whole transcript")
        self.full_text.setFixedHeight(72)
        self.full_text.textChanged.connect(self._full_text_edited)
        center.addWidget(self.full_text)
        keys = CaptionLabel("Waveform: drag a clip to move it, drag its edges to adjust in and out · Double-click the text to correct it · Space: play · X: include/exclude · Ctrl+click / Shift+click: select several · Ctrl+M: merge · S: split at the playhead · Del: delete · Ctrl+Z / Ctrl+Y: undo / redo")
        keys.setWordWrap(True)
        center.addWidget(keys)
        self.log_view = PlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        self.log_view.setStyleSheet(LOG_STYLE)
        center.addWidget(self.log_view)
        root.addLayout(center, 1)

        # Space: play / pause from the playhead, anywhere in the tab (text fields
        # still receive the space: Qt gives them priority)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Delete"), self, activated=self._delete_key, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("X"), self, activated=self.toggle_selected, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.merge_selected, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("S"), self, activated=self.split_at_playhead, context=Qt.WidgetWithChildrenShortcut)
        self.bus.line.connect(lambda text, lvl: self.log_view.appendPlainText(text))
        self.timer = QTimer(self, interval=40, timeout=self._tick)
        self.timer.start()
        self._update_enabled()

    # ------------------------------------------------------------ estado
    def abort(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("Aborting: stops when the current step ends (seconds)...")
            self.bus.push("Aborting at the user's request...", 1)

    def _update_enabled(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        self.run_btn.setEnabled(not busy and self.files.count() > 0)
        self.abort_btn.setEnabled(busy)
        self.sep.run_btn.setEnabled(not busy and self.files.count() > 0)
        self.sep.restore_btn.setEnabled(not busy and bool(self.pre))
        self.export_btn.setEnabled(not busy and any(c.keep for c in self.clips))
        if hasattr(self, "include_all_btn"):
            self.include_all_btn.setEnabled(not busy and any(not c.keep for c in self.clips))

    def _params(self) -> dataset.DatasetParams:
        return dataset.DatasetParams(
            language=models.CW_LANGUAGES[self.lang.currentText()], hallucination_mitigation=self.sw_hall.isChecked(),
            normalize=self.sw_norm.isChecked(), align=self.sw_align.isChecked(), use_sava=self.sw_sava.isChecked(),
            aligner=self._aligner_keys[self.aligner_cb.currentIndex()],
            isolate=self.sw_iso.isChecked(), engine=self.sep.engine_key(), sep_block_s=self.sep.block_s(), diarize=self.sw_diar.isChecked(), verbatim=self.sw_verb.isChecked(), keep_fillers=self.sw_fill.isChecked(),
            min_s=self.s_min.get(), max_s=max(self.s_max.get(), self.s_min.get() + 0.5), min_pause_s=self.s_pause.get(),
            pad_s=self.s_pad.get(), out_sr=int(self.out_sr.currentText()), min_word_prob=self.s_conf.get(),
            speaker=self.speaker.text().strip() or "speaker")

    # ------------------------------------------------------------ files
    def add_files(self, files: list[str] | None = None) -> None:
        if not files:
            files, _ = QFileDialog.getOpenFileNames(self, "Add audio files", "",
                                                    "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All (*.*)")
        known = {self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())}
        for f in files:
            if f not in known:
                self.files.addItem(Path(f).name)
                self.files.item(self.files.count() - 1).setData(Qt.UserRole, f)
        self._update_enabled()

    # ------------------------------------------------------------ separation
    def _targets(self) -> list[Path]:
        sel = [Path(it.data(Qt.UserRole)) for it in self.files.selectedItems()]
        return sel or [Path(self.files.item(i).data(Qt.UserRole)) for i in range(self.files.count())]

    def _forget(self, keys: set[str]) -> None:
        """Already analyzed clips of these files are outdated: they are removed (they must be analyzed again)."""
        if any(c.source in keys for c in self.clips):
            self.clips = [c for c in self.clips if c.source not in keys]
            self.history.clear()
            self.bus.push("Clips of the reprocessed files were removed: analyze them again", 1)
        for k in keys:
            self.sources.pop(k, None)

    def _show_audio(self, key: str, x, sr: int) -> None:
        self.player.pause()
        self.wave.clear()
        mono = audio_io.to_mono(x)
        self.wave.set_layer("src", mono, sr, "#9a9a9a", "fill")
        self.wave.fit()
        self.shown = key
        self._load_player(mono, sr)
        self._fill_table()

    def _load_player(self, mono, sr: int) -> None:
        """The visible audio stays in the player: Space plays it from the playhead."""
        self.player.pause()
        self.player.stop_at = None
        self.player.set_sources(sr, {"src": mono[:, None] if mono.ndim == 1 else mono}, "src")
        self._playing_clip = None

    def separate_now(self) -> None:
        paths = self._targets()
        pre = dict(self.pre)
        # separate what is there: if it was already separated, it chains (e.g. music first, then noise)
        items = [(p.name, lambda p=p: pre.get(str(p)) or audio_io.load(p)) for p in paths]

        def done(res) -> None:
            keys = {str(p) for p in paths}
            for p, r in zip(paths, res):
                self.pre[str(p)] = r
            self._forget(keys)
            self._show_audio(str(paths[-1]), *res[-1])
            self.progress.setValue(100)
            self.status.setText(f"Separated: {len(paths)} file(s) with {self.sep.engine_name()}. Analyze to cut the clips.")
            self.bus.push(f"Dataset: {len(paths)} file(s) separated with {self.sep.engine_name()}")
            self._update_enabled()

        run_separation(self, items, self.sep, done)

    def restore_original(self) -> None:
        paths = [p for p in self._targets() if str(p) in self.pre]
        if not paths:
            return
        for p in paths:
            self.pre.pop(str(p), None)
        self._forget({str(p) for p in paths})
        x, sr = audio_io.load(paths[-1])
        self._show_audio(str(paths[-1]), x, sr)
        self.status.setText(f"{len(paths)} file(s) back to the original")
        self._update_enabled()

    def remove_files(self) -> None:
        for it in self.files.selectedItems():
            self.pre.pop(it.data(Qt.UserRole), None)
            self.files.takeItem(self.files.row(it))
        self._update_enabled()

    # ------------------------------------------------------------ analysis
    def analyze(self) -> None:
        paths = [Path(self.files.item(i).data(Qt.UserRole)) for i in range(self.files.count())]
        self.prm = prm = self._params()
        self.bus.push(f"Dataset: {len(paths)} file(s) · language {prm.language} · {prm.min_s:.1f}-{prm.max_s:.1f} s")

        pre = dict(self.pre)

        def job(ctx):
            all_clips, sources = [], {}
            for k, p in enumerate(paths):
                ctx.check()
                sub = type(ctx)(cancel=ctx.cancel, log=ctx.log, progress=lambda v, k=k: ctx.progress((k + v) / len(paths)))
                clips, src = dataset.build_clips(p, prm, sub, self.store, pre.get(str(p)))
                all_clips += clips
                sources[str(p)] = src
                ctx.progress((k + 1) / len(paths))
            return all_clips, sources

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda v: (self.progress.setValue(int(v * 100)),
                                                self.status.setText(f"Analyzing... {v * 100:.0f}%")))
        self.worker.log.connect(self.bus.push)
        self.worker.done.connect(self._analyzed)
        self.worker.failed.connect(lambda e: (self.status.setText("Error (see log)"),
                                              InfoBar.error("Error", e, duration=8000, position=InfoBarPosition.TOP_RIGHT,
                                                            parent=self.window())))
        self.worker.cancelled.connect(lambda: self.status.setText("Analysis aborted"))
        self.worker.finished.connect(self._update_enabled)
        self.worker.start()
        self._update_enabled()

    def _analyzed(self, res) -> None:
        self.clips, self.sources = res
        self.history.clear()
        self.shown = None
        names = sorted({c.speaker for c in self.clips if c.speaker})
        self.spk_filter.blockSignals(True)
        self.spk_filter.clear()
        self.spk_filter.addItems([ALL] + names)
        self.spk_filter.blockSignals(False)
        self.progress.setValue(100)
        self.status.setText("Ready")
        self._fill_table()
        if self.clips:
            self._select_quietly(0)
        self._update_enabled()

    # ------------------------------------------------------------ table
    def _row_values(self, c: dataset.Clip) -> list[str]:
        return ["✓" if c.keep else "✗", Path(c.source).name, c.speaker, f"{c.start:.2f}", f"{c.duration:.2f}", c.text,
                f"{c.min_prob:.2f}", f"{100 * c.speech_ratio:.0f}%", ", ".join(_tr(x) for x in c.reason.split(", ")) if c.reason else ("" if c.keep else _tr("excluded by hand"))]

    def _set_row(self, r: int) -> None:
        c = self.clips[r]
        self.table.blockSignals(True)
        for k, v in enumerate(self._row_values(c)):
            it = self.table.item(r, k)
            if it is None:
                it = QTableWidgetItem()
                self.table.setItem(r, k, it)
            it.setText(v)
            if k == TEXT_COL:
                it.setToolTip(v)
            if k != TEXT_COL:
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            if k in (3, 4, 6, 7):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setForeground(QColor("#ffffff") if c.keep else QColor("#8a8a8a"))
        self.table.blockSignals(False)

    def _fill_table(self) -> None:
        self.clips.sort(key=lambda c: (c.source, c.start))
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.clips))
        self.table.blockSignals(False)
        for r in range(len(self.clips)):
            self._set_row(r)
        self._summary()
        self._refresh_regions()
        if hasattr(self, "run_btn"):
            self._update_enabled()

    def _summary(self) -> None:
        keep = [c for c in self.clips if c.keep]
        self.summary.setText(f"{len(keep)} of {len(self.clips)} clips · {sum(c.duration for c in keep) / 60:.1f} min")

    def _edited(self, item: QTableWidgetItem) -> None:
        if item.column() == TEXT_COL and 0 <= item.row() < len(self.clips):
            c = self.clips[item.row()]
            if item.text().strip() == c.text:
                return
            self.history.record(merge=("text", id(c)))
            c.text, c.text_edited = item.text().strip(), True
            item.setToolTip(c.text)
            if item.row() == self._row():
                self._show_full_text(c)

    def _show_full_text(self, c) -> None:
        if self.full_text.toPlainText() != c.text:
            self.full_text.blockSignals(True)
            self.full_text.setPlainText(c.text)
            self.full_text.blockSignals(False)

    def _full_text_edited(self) -> None:
        r = self._row()
        if not (0 <= r < len(self.clips)):
            return
        c = self.clips[r]
        text = self.full_text.toPlainText().replace("\n", " ").strip()
        if text != c.text:
            self.history.record(merge=("text", id(c)))
            c.text, c.text_edited = text, True
            self._set_row(r)

    def _row(self) -> int:
        """The current clip (the last one clicked) when it is selected; otherwise the first selected."""
        rows = sorted(r.row() for r in self.table.selectionModel().selectedRows())
        cur = self.table.currentRow()
        return cur if cur in rows else (rows[0] if rows else -1)

    def _rows(self) -> list[int]:
        return sorted(r.row() for r in self.table.selectionModel().selectedRows())

    # ------------------------------------------------------------ waveform and regions
    def _show_source(self, source: str) -> None:
        if self.shown == source:
            return
        src = self.sources.get(source)
        self.wave.clear()
        self.shown = source
        if src is not None:
            self.wave.set_layer("src", src.mono, src.sr, "#9a9a9a", "fill")
            self._load_player(src.mono, src.sr)
        self._refresh_regions()

    def _refresh_regions(self) -> None:
        if self.shown is None:
            self.wave.set_regions([])
            self.region_clips = []
            return
        self.region_clips = [k for k, c in enumerate(self.clips) if c.source == self.shown]
        self.wave.set_regions([(self.clips[k].start, self.clips[k].end) for k in self.region_clips],
                              [ACCENT if self.clips[k].keep else DANGER for k in self.region_clips])
        r = self._row()
        if r in self.region_clips:
            self.wave.select_region(self.region_clips.index(r), center=False)
        self._mark_selection()

    def _mark_selection(self) -> None:
        """Every selected clip is highlighted on the waveform when several are selected."""
        rows = self._rows()
        self.wave.mark_regions([self.region_clips.index(r) for r in rows if r in self.region_clips])
        n = len(rows)
        if n > 1:
            self.summary.setText(f"{n} clips selected · {sum(self.clips[r].duration for r in rows):.1f} s")
        else:
            self._summary()

    def _select_quietly(self, r: int) -> None:
        """Select a row from code: shows the clip without auto-playing it."""
        self._quiet_select = True
        try:
            self.table.selectRow(r)
        finally:
            self._quiet_select = False

    def _selected(self) -> None:
        r = self._row()
        if r < 0:
            self._mark_selection()
            return
        c = self.clips[r]
        multi = len(self._rows()) > 1
        self._show_full_text(c)
        self._show_source(c.source)
        if r in self.region_clips:
            self.wave.select_region(self.region_clips.index(r), center=not multi)
        self._mark_selection()
        if multi:   # building a multiple selection: no seeking or autoplay
            if self.player.playing and self._playing_clip is not None:
                self.player.pause()
            return
        self._seek(c.start)
        if self.sw_play.isChecked() and not getattr(self, "_quiet_select", False):
            self.play_selected(toggle=False)

    def _region_clicked(self, idx: int) -> None:
        if 0 <= idx < len(self.region_clips):
            r = self.region_clips[idx]
            if r != self._row():
                self.table.selectRow(r)
                self.table.scrollToItem(self.table.item(r, 0))

    def _region_ctrl_clicked(self, idx: int) -> None:
        """Ctrl/Shift+click on the waveform: add the clip to the selection or take it out."""
        if not (0 <= idx < len(self.region_clips)):
            return
        from PySide6.QtCore import QItemSelectionModel
        r = self.region_clips[idx]
        flag = QItemSelectionModel.Deselect if r in self._rows() else QItemSelectionModel.Select
        self.table.selectionModel().select(self.table.model().index(r, 0), flag | QItemSelectionModel.Rows)
        self.table.scrollToItem(self.table.item(r, 0))

    def _region_changed(self, idx: int, a: float, b: float) -> None:
        """Rebuild the clip with the new in/out (audio, text unless corrected by hand, and checks)."""
        if not (0 <= idx < len(self.region_clips)):
            return
        r = self.region_clips[idx]
        old = self.clips[r]
        src = self.sources[old.source]
        self.history.record()
        new = dataset.make_clip(src, a, b, self.prm, old.text if old.text_edited else None)
        new.text_edited, new.name = old.text_edited, old.name
        self._apply_filter(new)
        self.clips[r] = new
        self._set_row(r)
        self.wave.update_region(idx, new.start, new.end, ACCENT if new.keep else DANGER)
        self._summary()
        self._update_enabled()
        if r != self._row():
            self._select_quietly(r)

    def _seek(self, s: float) -> None:
        self.wave.set_cursor(s)
        if self.player.sources:
            self.player.stop_at = None
            self._playing_clip = None
            self.player.seek(int(s * self.player.sr))

    def toggle_play(self) -> None:
        """Space: play from the playhead (or pause)."""
        if not self.player.sources:
            return
        if self.player.playing:
            self.player.pause()
            return
        self.player.stop_at = None
        self._playing_clip = None
        self.player.play()

    # ------------------------------------------------------------ speaker filter
    def _apply_filter(self, c: dataset.Clip) -> None:
        sel = self.spk_filter.currentText()
        reasons = [r for r in c.reason.split(", ") if r and r != OTHER]
        if sel not in ("", ALL) and c.speaker != sel:
            reasons.append(OTHER)
            c.keep = False
        elif OTHER in c.reason:
            c.keep = not reasons  # back to its state according to its own reasons
        c.reason = ", ".join(reasons)

    def _filter_speaker(self, _text: str) -> None:
        for c in self.clips:
            self._apply_filter(c)
        self._fill_table()
        self._update_enabled()

    # ------------------------------------------------------------ acciones
    def play_selected(self, toggle: bool = True) -> None:
        """Play the selected clip on the waveform audio (from its start to its end) and stop by itself."""
        r = self._row()
        if r < 0:
            return
        c = self.clips[r]
        self._show_source(c.source)
        if "src" not in self.player.sources:
            return
        if toggle and self.player.playing and self._playing_clip is c:
            self.player.pause()
            return
        sr = self.player.sr
        self.player.pause()
        self.player.seek(int(c.start * sr))
        self.player.stop_at = int(c.end * sr)
        self._playing_clip = c
        self.player.play()

    def _set_keep(self, rows: list[int], keep: bool) -> None:
        self.history.record()
        for r in rows:
            c = self.clips[r]
            c.keep = keep
            self._set_row(r)
            if r in self.region_clips:
                self.wave.update_region(self.region_clips.index(r), c.start, c.end, ACCENT if c.keep else DANGER)
        if len(self._rows()) > 1:
            self._mark_selection()
        else:
            self._summary()
        self._update_enabled()

    def toggle_selected(self) -> None:
        """Include / exclude every selected clip: if any is excluded, all are included; otherwise all are excluded."""
        rows = self._rows()
        if not rows:
            r = self._row()
            if r < 0:
                return
            rows = [r]
        self._set_keep(rows, not all(self.clips[r].keep for r in rows))

    def include_all(self) -> None:
        rows = [k for k, c in enumerate(self.clips) if not c.keep]
        if rows:
            self._set_keep(rows, True)

    def new_clip(self) -> None:
        """New 3 s clip at the playhead (or the center of the view) of the visible file; adjust it afterwards."""
        if self.shown is None or self.shown not in self.sources:
            return
        src = self.sources[self.shown]
        t = float(self.wave.cursor.value())
        x0, x1 = self.wave.vb.viewRange()[0]
        if not (x0 <= t <= x1):
            t = (x0 + x1) / 2
        clip = dataset.make_clip(src, t, min(src.duration, t + 3.0), self.prm)
        clip.keep = True
        self.history.record()
        self.clips.append(clip)
        self._fill_table()
        self._select_quietly(self.clips.index(clip))
        self.bus.push(f"New clip at {clip.start:.2f} s: adjust its edges on the waveform")

    def delete_selected(self) -> None:
        """Delete the selected clip(s)."""
        rows = self._rows()
        if not rows:
            return
        self.history.record()
        for r in reversed(rows):
            del self.clips[r]
        self._fill_table()
        if self.clips:
            self._select_quietly(min(rows[0], len(self.clips) - 1))
        self._update_enabled()

    def _info(self, title: str, text: str, error: bool = False) -> None:
        (InfoBar.error if error else InfoBar.success)(title, text, duration=6000 if error else 3000,
                                                     position=InfoBarPosition.TOP_RIGHT, parent=self.window())

    def split_at_playhead(self) -> None:
        """Cut a clip in two at the playhead: the clip under the playhead in the file shown (or the selected one)."""
        t = float(self.wave.cursor.value())
        under = [k for k in self.region_clips if self.clips[k].start < t < self.clips[k].end]
        r = self._row()
        k = r if r in under else (under[0] if under else -1)
        if k < 0:
            self._info("Split clip", "Place the playhead inside a clip (click on the waveform)", True)
            return
        c = self.clips[k]
        src = self.sources.get(c.source)
        if src is None:
            return
        try:
            a, b = dataset.split_clip(src, c, t, self.prm)
        except ValueError:
            self._info("Split clip", "The playhead is too close to an edge of the clip", True)
            return
        for x in (a, b):
            self._apply_filter(x)
        self.player.pause()
        self.history.record()
        self.clips[k:k + 1] = [a, b]
        self._fill_table()
        self._select_quietly(self.clips.index(b))
        self._update_enabled()
        self.bus.push(f"Clip split at {t:.2f} s: {a.duration:.2f} s + {b.duration:.2f} s")

    def merge_selected(self) -> None:
        """Join the selected clips (2 or more, same file) into one: from the first start to the last end, with
        the transcript of the whole span. Clips lying inside that span are absorbed too."""
        rows = self._rows()
        picked = [self.clips[r] for r in rows]
        if len(picked) < 2:
            self._info("Merge clips", "Select two or more clips (Ctrl+click or Shift+click)", True)
            return
        if len({c.source for c in picked}) > 1:
            self._info("Merge clips", "The clips must belong to the same file", True)
            return
        src = self.sources.get(picked[0].source)
        if src is None:
            return
        a, b = min(c.start for c in picked), max(c.end for c in picked)
        inside = [c for c in self.clips if c.source == picked[0].source and c.start >= a - 1e-6 and c.end <= b + 1e-6]
        group = picked + [c for c in inside if all(c is not p for p in picked)]
        new = dataset.merge_clips(src, group, self.prm)
        self._apply_filter(new)
        self.history.record()
        k = min(self.clips.index(c) for c in group)
        self.clips = [c for c in self.clips if all(c is not g for g in group)]
        self.clips.insert(k, new)
        self._fill_table()
        self._select_quietly(self.clips.index(new))
        self._update_enabled()
        self.bus.push(f"Merged {len(group)} clips: {new.start:.2f}-{new.end:.2f} s ({new.duration:.2f} s)")

    def _delete_key(self) -> None:
        """Del: removes the selected files when the file list has the focus; otherwise, the selected clip."""
        if self.files.hasFocus():
            self.remove_files()
        else:
            self.delete_selected()

    # ------------------------------------------------------------ project
    def project_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def project_files(self) -> list[str]:
        return [self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())]

    def project_state(self) -> dict:
        from .projects import capture_controls
        r = self._row()
        # the audio of every clip is rebuilt from its source when the project opens: it is not stored twice
        clips = [copy.copy(c) for c in self.clips]
        for c in clips:
            c.audio = None
        return {"controls": capture_controls(self, {"spk_filter"}), "files": self.project_files(), "pre": self.pre,
                "sources": self.sources, "prm": self.prm, "clips": clips, "shown": self.shown, "row": r,
                "filter": self.spk_filter.currentText() if self.spk_filter.currentIndex() > 0 else None}

    def project_apply(self, st: dict) -> None:
        from .projects import apply_controls
        self.project_clear()
        apply_controls(self, st.get("controls", {}))
        self.add_files([str(f) for f in st.get("files", [])])
        # derived audio missing from the project's media folder: that file has to be analyzed again
        self.pre = {k: v for k, v in (st.get("pre") or {}).items()
                    if v is not None and not (isinstance(v, tuple) and any(e is None for e in v))}
        self.sources = {k: s for k, s in (st.get("sources") or {}).items() if getattr(s, "mono", None) is not None}
        self.prm = st.get("prm") or self.prm
        self.clips = []
        for c in st.get("clips") or []:
            src = self.sources.get(c.source)
            if src is None:
                continue
            c.audio = dataset.clip_audio(src, c.start, c.end, self.prm)
            self.clips.append(c)
        names = sorted({c.speaker for c in self.clips if c.speaker})
        self.spk_filter.blockSignals(True)
        self.spk_filter.clear()
        self.spk_filter.addItems([ALL] + names)
        if st.get("filter") in names:
            self.spk_filter.setCurrentText(st["filter"])
        self.spk_filter.blockSignals(False)
        self.shown = None
        self._fill_table()
        if st.get("shown") in self.sources:
            self._show_source(st["shown"])
        if self.clips:
            self._select_quietly(min(max(st.get("row", 0), 0), len(self.clips) - 1))
        self.history.clear()
        self.status.setText("Project opened")
        self._update_enabled()

    def project_clear(self) -> None:
        self.player.pause()
        self.files.clear()
        self.pre, self.sources, self.clips, self.shown = {}, {}, [], None
        self.history.clear()
        self.spk_filter.blockSignals(True)
        self.spk_filter.clear()
        self.spk_filter.addItem(ALL)
        self.spk_filter.blockSignals(False)
        self.wave.clear()
        self.player.set_sources(48000, {})
        self._fill_table()
        self.full_text.blockSignals(True)
        self.full_text.clear()
        self.full_text.blockSignals(False)
        self.progress.setValue(0)
        self.status.setText(_tr("Ready"))
        self._update_enabled()

    # ------------------------------------------------------------ undo / redo
    def _snapshot(self):
        r = self._row()
        return [copy.copy(c) for c in self.clips], (self.clips[r].source, self.clips[r].start) if r >= 0 else None

    def _restore(self, state) -> None:
        clips, sel = state
        self.clips = [copy.copy(c) for c in clips]
        for c in self.clips:
            self._apply_filter(c)  # the speaker filter may have changed since
        self._fill_table()
        if sel and self.clips:
            k = min(range(len(self.clips)), key=lambda i: (self.clips[i].source != sel[0], abs(self.clips[i].start - sel[1])))
            self._select_quietly(k)
        self._update_enabled()

    def _tick(self) -> None:
        if self.player.playing and self.player.sources:
            self.wave.set_cursor(self.player.position() / self.player.sr, follow=True)

    # ------------------------------------------------------------ export
    def export(self, out_dir: str | None = None) -> None:
        """ZIP: asks for the file. Folder: asks where, and creates a new "dataset" subfolder there."""
        package = self.package.currentRouteKey()
        if package == "zip":
            d = out_dir or QFileDialog.getSaveFileName(self, "Save dataset", "dataset.zip", "ZIP (*.zip)")[0]
            if not d:
                return
            out = Path(d).with_suffix(".zip")
        else:
            d = out_dir or QFileDialog.getExistingDirectory(self, "Choose where to create the dataset folder")
            if not d:
                return
            out = Path(d) / "dataset"
            k = 2
            while out.exists() or out.with_name(out.name + ".part").exists():
                out = Path(d) / f"dataset_{k}"
                k += 1
        clips, prm, fmt = list(self.clips), self.prm, self.fmt.key()
        layout, eval_pct = self.layout_keys[self.layout_cb.currentIndex()], self.s_eval.get()
        self.worker = Worker(lambda ctx: dataset.export(clips, out, prm, ctx, fmt, layout, eval_pct, package), self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.done.connect(lambda n: InfoBar.success("Dataset exported", f"{n} clips in {out}", duration=5000,
                                                           position=InfoBarPosition.TOP_RIGHT, parent=self.window()))
        self.worker.failed.connect(lambda e: InfoBar.error("Error", e, duration=8000, position=InfoBarPosition.TOP_RIGHT,
                                                           parent=self.window()))
        self.worker.finished.connect(self._update_enabled)
        self.worker.start()
        self._update_enabled()
