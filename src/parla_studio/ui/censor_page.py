"""Censor tab: find words in the transcript and censor them with a beep, silence or cut."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout,
                               QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, LineEdit,
                            PrimaryPushButton, ProgressBar, PushButton, SegmentedWidget, StrongBodyLabel, SubtitleLabel,
                            SwitchButton,
                            TableWidget, ToolButton)

from ..core import audio_io, censor, leveler as lv, models, profanity, speakers as spk
from ..i18n import tr as _tr
from .common import ACCENT, DANGER, History, LogBus, Worker, format_combo, format_filter, undo_buttons
from .dataset_page import side_panel, switch_row
from .level_page import ParamSlider, fit, fmt_time, tame
from .treatment import SeparationPanel, run_separation
from .player import Player
from .waveform import WaveView

COLS = ["✓", "#", "Time", "Speaker", "Word", "Source", "Context"]
DB_COLOR = "#fb923c"   # matches from the lexicon (preventive censoring)
ALL = _tr("All speakers")
OFF = "#7a7a7a"


class CensorPage(QWidget):
    def __init__(self, bus: LogBus, store=None, db_root: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("censorPage")
        self.bus, self.store = bus, store
        # preventive censor lexicon: custom list per language + sources downloaded on demand
        self.lexstore = profanity.LexiconStore(Path(db_root) if db_root else __import__("parla_studio.paths", fromlist=["LEXICONS"]).LEXICONS)
        self._settings = self.lexstore.root / "censor.json"
        try:
            self._enabled = json.loads(self._settings.read_text(encoding="utf-8")).get("enabled", {})
        except (OSError, ValueError):
            self._enabled = {}
        legacy = self.lexstore.root / "profanidad.txt"
        if legacy.exists() and not self.lexstore.own_path("es").exists():
            rep = profanity.migrate_legacy(self.lexstore, legacy, "es")  # old database without ñ: cleaned once
            if rep:
                bus.push(f"Censor database migrated to custom_es.txt: {rep['input']} → {rep['output']} terms · ñ restored: {', '.join(rep['enye_restored']) or 'none'}")
        self.lex = profanity.Lexicon("es")
        self.diar: spk.Diarization | None = None
        self.names: dict[int, str] = {}
        self.player = Player()
        self.path: Path | None = None
        self.audio: np.ndarray | None = None
        self.orig: np.ndarray | None = None   # audio as loaded ("Original" goes back to it)
        self.voice: np.ndarray | None = None
        self.sr = 48000
        self.words: list[models.Word] = []
        self.matches: list[censor.Match] = []
        self.current = -1
        self.worker: Worker | None = None
        self.rendered: np.ndarray | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 8, 8)
        root.setSpacing(10)
        scroll, c = side_panel()
        t = SubtitleLabel("Word censoring")
        t.setStyleSheet(f"color: {ACCENT}")
        self.title_row = QHBoxLayout()  # the main window adds the Project button here
        self.title_row.addWidget(t, 1)
        c.addLayout(self.title_row)
        self.project_box = QVBoxLayout()  # ... and the project name here
        self.project_box.setContentsMargins(0, 0, 0, 0)
        c.addLayout(self.project_box)
        intro = CaptionLabel("Transcribes with CrisperWhisper 2.0 turbo, finds every occurrence and covers it with a beep, silence or a cut. With the voice isolated, music and ambience keep playing.")
        intro.setWordWrap(True)
        c.addWidget(intro)
        self.open_btn = PushButton(FIF.FOLDER, "Select audio")
        self.open_btn.clicked.connect(lambda: self.open_file())
        c.addWidget(self.open_btn)
        self.file_lbl = fit(CaptionLabel("No file"))
        self.file_lbl.setWordWrap(True)
        c.addWidget(self.file_lbl)
        from .youtube import YouTubeField
        self.yt = YouTubeField(self.bus, lambda ps: self.open_file(str(ps[0])), tab_busy=lambda: self.worker is not None and self.worker.isRunning())
        c.addWidget(self.yt)
        c.addWidget(StrongBodyLabel("Language"))
        self.lang = ComboBox()
        self.lang.addItems(list(models.CW_LANGUAGES))
        self.lang.setCurrentText("English")
        c.addWidget(self.lang)
        self.sw_iso = switch_row(c, "Isolate the voice (keeps the background when censoring)", True)
        self.sep = SeparationPanel()
        self.sep.run_btn.clicked.connect(self.separate_now)
        self.sep.restore_btn.clicked.connect(lambda: self.orig is not None and self._set_audio(self.orig))
        c.addWidget(self.sep)
        self.sw_norm = switch_row(c, "Normalize internally before analyzing", True)
        self.sw_align = switch_row(c, "Align words (more precise censoring)", True)
        self.sw_sava = switch_row(c, "Detect voice with SAVA", False)
        self.sw_sava.setToolTip("SAVA (AI voice activity) checks where there really is speech: words transcribed where nobody speaks (hallucinations over music or silence) are dropped and the speakers are only marked where there is voice")
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
        self.progress = ProgressBar()
        self.progress.setValue(0)
        c.addWidget(self.progress)
        self.status = fit(CaptionLabel("Ready"), lines=3)
        c.addWidget(self.status)
        self.run_btn = PrimaryPushButton(FIF.SYNC, "Analyze (transcribe)")
        self.run_btn.clicked.connect(self.analyze)
        self.abort_btn = PushButton(FIF.CANCEL, "Abort")
        self.abort_btn.setToolTip("Stops the running process (analysis, separation, export...) within seconds")
        self.abort_btn.clicked.connect(self.abort)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.abort_btn, 2)
        c.addLayout(run_row)

        c.addWidget(StrongBodyLabel("Search"))
        self.query = LineEdit()
        self.query.setPlaceholderText("words or phrases, comma-separated · * = wildcard")
        self.query.setClearButtonEnabled(True)
        self.query.returnPressed.connect(self.search)
        c.addWidget(self.query)
        self.sw_acc = switch_row(c, "Ignore accents and case", True)
        self.sw_whole = switch_row(c, "Whole words only", True)
        self.search_btn = PrimaryPushButton(FIF.SEARCH, "Search")
        self.search_btn.clicked.connect(self.search)
        c.addWidget(self.search_btn)
        self.count_lbl = fit(BodyLabel("No search"))
        c.addWidget(self.count_lbl)

        c.addWidget(StrongBodyLabel("Preventive censoring (per-language lexicon)"))
        self.sw_db = switch_row(c, "Automatically censor what is in the lexicon", False)
        self.sw_db.setToolTip("Off by default: the lexicons are not included with the app. Download or write a list "
                              "below and turn this on to censor its terms automatically")
        self.sw_db.checkedChanged.connect(lambda *_: self.search())
        self.db_lbl = fit(CaptionLabel())
        self.db_lbl.setWordWrap(True)
        c.addWidget(self.db_lbl)
        self.src_box = QVBoxLayout()
        self.src_box.setSpacing(2)
        c.addLayout(self.src_box)
        row = QHBoxLayout()
        for icon, tip, fn in ((FIF.EDIT, "Open this language's own list in the editor", self._db_open),
                              (FIF.SYNC, "Reload the lexicon", self._db_reload),
                              (FIF.ADD, "Add the terms typed in \"Search\" to the own list (supports * and !)",
                               self._db_add),
                              (FIF.BROOM, "Import and clean an old list (restores ñ, removes duplicates and folds variants into wildcards)", self._db_import)):
            b = ToolButton(icon)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        c.addLayout(row)
        self.lang.currentTextChanged.connect(lambda *_: self._db_reload(quiet=True))
        self._db_reload(quiet=True)
        c.addWidget(StrongBodyLabel("Censor only"))
        self.who = ComboBox()
        self.who.addItem(ALL)
        self.who.currentTextChanged.connect(lambda *_: self._apply_who())
        c.addWidget(self.who)

        c.addWidget(StrongBodyLabel("Censor"))
        self.mode = SegmentedWidget()
        for k, txt in (("beep", "Beep"), ("silence", "Silence"), ("cut", "Cut")):
            self.mode.addItem(k, txt)
        self.mode.setCurrentItem("beep")
        self.mode.currentItemChanged.connect(lambda *_: self._schedule_render())
        c.addWidget(self.mode)
        self.s_freq = ParamSlider("Beep tone (Hz)", 300, 3000, 1000, 0)
        self.s_level = ParamSlider("Beep vs. voice (dB)", -20, 6, -3, 1)
        self.s_pad = ParamSlider("Margin around the word (ms)", 0, 150, 30, 0)
        for s in (self.s_freq, self.s_level):
            c.addWidget(s)
            s.slider.sliderReleased.connect(self._schedule_render)
        c.addWidget(self.s_pad)
        self.s_pad.slider.sliderReleased.connect(self._repad)
        cap = CaptionLabel("Cut removes the whole segment (voice and background) and shortens the audio.")
        cap.setWordWrap(True)
        c.addWidget(cap)
        c.addWidget(StrongBodyLabel("Export format"))
        self.fmt = format_combo("wav24")
        c.addWidget(self.fmt)
        self.export_btn = PrimaryPushButton(FIF.SAVE, "Export censored audio")
        self.export_btn.clicked.connect(lambda: self.export())
        c.addWidget(self.export_btn)
        c.addStretch(1)
        tame(scroll.widget())
        root.addWidget(scroll)

        center = QVBoxLayout()
        bar = QHBoxLayout()
        self.prev_btn = ToolButton(FIF.LEFT_ARROW)
        self.prev_btn.setToolTip("Previous (Shift+F3)")
        self.prev_btn.clicked.connect(lambda: self.jump(self.current - 1))
        self.next_btn = ToolButton(FIF.RIGHT_ARROW)
        self.next_btn.setToolTip("Next (F3)")
        self.next_btn.clicked.connect(lambda: self.jump(self.current + 1))
        self.pos_lbl = StrongBodyLabel("0 / 0")
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.pos_lbl)
        bar.addWidget(self.next_btn)
        bar.addSpacing(8)
        self.sw_auto = switch_row(bar, "Play when jumping", True)
        bar.addSpacing(8)
        for icon, tip, fn in ((FIF.ZOOM_OUT, "Zoom out", lambda: self.wave.zoom(1.5)),
                              (FIF.ZOOM_IN, "Zoom in", lambda: self.wave.zoom(1 / 1.5)),
                              (FIF.FIT_PAGE, "Fit", lambda: self.wave.fit())):
            b = ToolButton(icon)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(8)
        self.history = History(self._snapshot, self._restore)
        for b in undo_buttons(self, self.history):
            bar.addWidget(b)
        bar.addStretch(1)
        self.time_lbl = BodyLabel("00:00.00 / 00:00.00")
        self.time_lbl.setStyleSheet("font-family: Consolas; font-size: 15px")
        bar.addWidget(self.time_lbl)
        center.addLayout(bar)
        self.wave = WaveView()
        self.wave.aux.hide()
        self.wave.seekRequested.connect(self._seek)
        self.wave.regionClicked.connect(lambda k: self.jump(k, play=False))
        self.wave.regionChanged.connect(self._region_changed)
        center.addWidget(self.wave, 3)
        tr = QHBoxLayout()
        self.play_btn = PrimaryPushButton(FIF.PLAY, "Play")
        self.play_btn.setMinimumWidth(135)  # "Reproducir" / "Pausar" whole
        self.play_btn.clicked.connect(self.toggle_play)
        tr.addWidget(self.play_btn)
        self.source = SegmentedWidget()
        self.source.addItem("A", "A · Original")
        self.source.addItem("B", "B · Censored")
        self.source.setCurrentItem("A")
        self.source.currentItemChanged.connect(self._source_changed)
        tr.addWidget(self.source)
        tr.addSpacing(6)
        self.prev_cens = PrimaryPushButton(FIF.PLAY, "Censored (P)")
        self.prev_cens.setToolTip("Plays the current match already censored, from 1 s before to 1 s after")
        self.prev_cens.clicked.connect(lambda: self.preview(True))
        tr.addWidget(self.prev_cens)
        self.prev_orig = PushButton(FIF.PLAY, "Original (O)")
        self.prev_orig.setToolTip("The same segment uncensored, to compare")
        self.prev_orig.clicked.connect(lambda: self.preview(False))
        tr.addWidget(self.prev_orig)
        tr.addStretch(1)
        toggle = PushButton(FIF.ACCEPT, "Censor / don't censor")
        toggle.setToolTip("X")
        toggle.clicked.connect(self.toggle_current)
        tr.addWidget(toggle)
        rm = ToolButton(FIF.DELETE)
        rm.setToolTip("Remove the match from the list (Del)")
        rm.clicked.connect(self.delete_current)
        tr.addWidget(rm)
        center.addLayout(tr)
        self.table = TableWidget()
        self.table.setColumnCount(len(COLS))
        self.table.setHorizontalHeaderLabels([f"  {c}  " for c in COLS])  # padding: the Fluent style uses a wider font
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for i in range(len(COLS)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.cellClicked.connect(lambda r, _c: self.jump(r))
        self.table.currentCellChanged.connect(lambda r, _c, _pr, _pc: r >= 0 and r != self.current and self.jump(r))
        center.addWidget(self.table, 2)
        keys = CaptionLabel("F3 / Shift+F3: next / previous · P / O: play the censored / original segment · Space: play · X: censor or not · Del: remove · A: toggle A/B · Ctrl+Z / Ctrl+Y: undo / redo · Region edges: adjust")
        keys.setWordWrap(True)
        center.addWidget(keys)
        root.addLayout(center, 1)

        self._cut_map: tuple[float, float, float] | None = None  # (cut point, offset before, offset after)
        for key, fn in (("F3", lambda: self.jump(self.current + 1)), ("Shift+F3", lambda: self.jump(self.current - 1)),
                        ("Space", self.toggle_play), ("X", self.toggle_current), ("Delete", self.delete_current),
                        ("A", self.toggle_ab),
                        ("P", lambda: self.preview(True)), ("O", lambda: self.preview(False))):
            QShortcut(QKeySequence(key), self, activated=fn, context=Qt.WidgetWithChildrenShortcut)
        self._render_timer = QTimer(self, singleShot=True, interval=250, timeout=self.render_preview)
        self.timer = QTimer(self, interval=33, timeout=self._tick)
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
        ready = bool(self.words)
        self.open_btn.setEnabled(not busy)
        self.yt.set_tab_busy(busy)
        self.abort_btn.setEnabled(busy)
        self.run_btn.setEnabled(not busy and self.path is not None)
        self.sep.run_btn.setEnabled(not busy and self.audio is not None)
        self.sep.restore_btn.setEnabled(not busy and self.orig is not None and self.audio is not self.orig)
        self.search_btn.setEnabled(ready)
        self.query.setEnabled(ready)
        self.export_btn.setEnabled(not busy and any(m.enabled for m in self.matches))
        self.prev_btn.setEnabled(bool(self.matches))
        self.next_btn.setEnabled(bool(self.matches))

    def _info(self, title: str, text: str, error: bool = False) -> None:
        (InfoBar.error if error else InfoBar.success)(title, text, duration=8000 if error else 4000,
                                                     position=InfoBarPosition.TOP_RIGHT, parent=self.window())

    # ------------------------------------------------------------ loading and analysis
    def open_file(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select audio", "",
                                                  "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All (*.*)")
        if not path:
            return
        self.player.pause()
        self.path = Path(path)
        self.audio, self.sr = audio_io.load(self.path)
        self.orig = self.audio
        self.voice, self.words, self.matches, self.rendered, self.current = None, [], [], None, -1
        self.history.clear()
        self.file_lbl.setText(f"{self.path.name}\n{self.audio.shape[0] / self.sr / 60:.1f} min · {self.sr} Hz · {self.audio.shape[1]} channel(s)")
        self.wave.clear()
        self.wave.set_layer("orig", audio_io.to_mono(self.audio), self.sr, "#9a9a9a", "fill")
        self.wave.fit()
        self.player.set_sources(self.sr, {"A": self.audio}, "A")
        self.source.setCurrentItem("A")
        self._fill_table()
        self.status.setText("Analyze the audio to be able to search")
        self.count_lbl.setText("No search")
        self._update_enabled()

    # ------------------------------------------------------------ separation
    def separate_now(self) -> None:
        if self.audio is None:
            return
        x, sr = self.audio, self.sr  # separate what is there: allows chaining (e.g. music, then noise)
        run_separation(self, [(self.path.name, lambda: (x, sr))], self.sep,
                       lambda res: (self._set_audio(res[0][0]), self.progress.setValue(100),
                                    self.status.setText(f"Separated with {self.sep.engine_name()}"),
                                    self.bus.push(f"{self.path.name}: separated with {self.sep.engine_name()}")))

    def _set_audio(self, y: np.ndarray) -> None:
        """Replace the working audio (times don't change: the transcript and censoring stay valid)."""
        self.player.pause()
        pos = self.player.pos
        self.audio, self.rendered = y, None
        sources = {"A": y}
        if self.voice is not None:
            sources["voice"] = self.voice
        self.wave.set_layer("orig", audio_io.to_mono(y), self.sr, "#9a9a9a", "fill")
        self.player.set_sources(self.sr, sources, "A")
        self.player.seek(min(pos, y.shape[0]))
        self.source.setCurrentItem("A")
        if any(m.enabled for m in self.matches):
            self._schedule_render()
        self._update_enabled()

    def analyze(self) -> None:
        if self.audio is None:
            return
        x, sr, iso, store = self.audio, self.sr, self.sw_iso.isChecked(), self.store
        lang = models.CW_LANGUAGES[self.lang.currentText()]
        name = self.path.name
        engine, block = self.sep.engine_key(), self.sep.block_s()
        norm, do_align, use_sava = self.sw_norm.isChecked(), self.sw_align.isChecked(), self.sw_sava.isChecked()
        al_engine = self._aligner_keys[self.aligner_cb.currentIndex()]

        def job(ctx):
            t = time.perf_counter()
            notes = [f"File: {name}", f"Language: {lang}"]
            # internal normalization: only for the analysis; censoring is applied to the audio at its original level
            xn, g = lv.normalize_loudness(x, sr) if norm else (x, 0.0)
            if norm:
                ctx.log(f"{name}: internal normalization {g:+.1f} dB (analysis only)", 0)
                notes.append(f"Internal normalization: {g:+.1f} dB (analysis only)")
            voice = None
            if iso:
                ctx.log(f"{name}: isolating the voice ({models.ENGINES.get(engine, engine)})...", 0)
                voice = models.isolate_voice(xn, sr, ctx.sub(0.0, 0.3), engine, block)
                notes.append(f"Separation: {models.ENGINES.get(engine, engine)}")
            ctx.progress(0.3 if iso else 0.0)
            ctx.check()
            ctx.log(f"{name}: transcribing with CrisperWhisper 2.0 turbo ({lang})...", 0)
            src_a = voice if voice is not None else xn
            if voice is not None and norm:
                # like the dataset pipeline: the separated voice is normalized again for transcription and
                # alignment (separation lowers the level; the aligner is more precise on a well-levelled voice)
                src_a, g2 = lv.normalize_loudness(voice, sr)
                notes.append(f"Normalization after separation: {g2:+.1f} dB (analysis only)")
            mono = audio_io.to_mono(src_a)
            _, words = models.transcribe(mono, sr, lang, verbatim=True, ctx=ctx.sub(0.3 if iso else 0.0, 0.6))
            dur = mono.size / sr
            outside = [w for w in words if w.start >= dur - 0.05]   # Whisper hallucination at the very end
            words = [w for w in words if w.start < dur - 0.05]
            for w in words:
                w.end = min(w.end, dur)
            if outside:
                ctx.log(f"{name}: dropped {len(outside)} word(s) outside the audio: {' '.join(w.text for w in outside)}", 1)
            speech = None
            if use_sava:
                ctx.check()
                ctx.log(f"{name}: detecting voice activity with SAVA...", 0)
                speech = models.speech_probability(mono, sr, ctx.sub(0.6, 0.62))
                words, dropped = models.words_with_voice(words, speech)
                notes.append(f"SAVA: {len(dropped)} word(s) without voice dropped")
                if dropped:
                    ctx.log(f"{name}: SAVA: dropped {len(dropped)} word(s) where nobody speaks: "
                            f"{' '.join(w.text for w in dropped[:30])}", 1)
            from ..core import aligner
            rows = None
            if do_align and words:
                desc = aligner.describe(al_engine, lang)
                ctx.log(f"{name}: aligning {len(words)} words with {desc}...", 0)
                notes.append(f"Aligner: {desc}")
                cw_words = words
                aligned, rows = aligner.align(mono, sr, words, ctx.sub(0.6, 0.7), al_engine, lang)
                ok = sum(r.status == "aligned" for r in rows)
                ctx.log(f"{name}: aligner: {ok} of {len(rows)} words corrected against the audio", 0)
                # censoring covers both estimates (CrisperWhisper and aligner); refine trims it to the voice
                words = censor.union_spans(cw_words, aligned)
                notes.append("Censor spans: union of CrisperWhisper and aligner times, trimmed to the voice energy")
            try:
                rows = rows or [aligner.AlignRow(w.text, w.start, w.end, w.prob, w.start, w.end, "not aligned")
                                for w in words]
                lp = aligner.write_log(name, rows, notes, kind="censor",
                                       engine=aligner.describe(al_engine, lang) if do_align else "not aligned")
                ctx.log(f"{name}: word log → {lp}", 0)
            except OSError:
                pass
            if voice is not None and g:
                voice = (voice * np.float32(10.0 ** (-g / 20.0))).astype(np.float32)  # voice at the original's level
            ctx.log(f"{name}: diarizing (who says each word)...", 0)
            diar = spk.diarize(audio_io.to_mono(xn), sr)
            if speech is not None:
                diar = spk.gate_with_speech(diar, speech)
            people = spk.describe(diar, mono, sr, store, 0.80)
            ctx.log(f"{name}: {len(words)} words · {len(people)} speaker(s) in {time.perf_counter() - t:.0f} s", 0)
            return voice, words, diar, {p.idx: p.name for p in people}

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.log.connect(lambda m, l: self.status.setText(m.split(": ", 1)[-1]))
        self.worker.done.connect(self._analyzed)
        self.worker.failed.connect(lambda e: (self.status.setText("Error (see log)"), self._info("Error", e, True)))
        self.worker.cancelled.connect(lambda: self.status.setText("Analysis aborted"))
        self.worker.finished.connect(self._update_enabled)
        self.status.setText("Analyzing...")
        self.progress.setValue(0)
        self.worker.start()
        self._update_enabled()

    def _analyzed(self, res) -> None:
        self.voice, self.words, self.diar, self.names = res
        self.who.blockSignals(True)
        self.who.clear()
        self.who.addItems([ALL] + list(self.names.values()))
        self.who.blockSignals(False)
        if self.voice is not None:
            self.player.add_source("voice", self.voice)
        self.progress.setValue(100)
        self.status.setText(f"Ready: {len(self.words)} words transcribed")
        self._update_enabled()
        self.search()  # preventive censoring: the lexicon is applied as soon as there is a transcript

    # ------------------------------------------------------------ censor lexicon
    def _lang_code(self) -> str:
        return models.CW_LANGUAGES.get(self.lang.currentText(), "es")

    def _enabled_for(self, lang: str) -> set:
        """Active sources for the language (by default: every downloaded one except LDNOOBW, which includes
        non-offensive words and is worth reviewing)."""
        if lang in self._enabled:
            return set(self._enabled[lang])
        return {k for k in self.lexstore.downloaded(lang) if k != "ldnoobw"}

    def _save_enabled(self) -> None:
        try:
            self._settings.write_text(json.dumps({"enabled": self._enabled}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
        except OSError:
            pass

    def _db_reload(self, quiet: bool = False) -> None:
        lang = self._lang_code()
        self.lex = self.lexstore.load(lang, self._enabled_for(lang))
        parts = [f"{k} {n}" for k, n in self.lex.sources.items()]
        self.db_lbl.setText(f"{self.lang.currentText()}: {self.lex.size} active terms" +
                            (f" ({' · '.join(parts)})" if parts else " · no own list or sources"))
        self._build_sources(lang)
        if not quiet:
            self.bus.push(f"Censor lexicon ({lang}) reloaded: {self.lex.size} terms")
        self.search()

    def _build_sources(self, lang: str) -> None:
        while self.src_box.count():
            item = self.src_box.takeAt(0)
            w = item.widget() or (item.layout() and None)
            if item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0).widget()
                    if sub:
                        sub.hide()
                        sub.setParent(None)
            if w:
                w.hide()
                w.setParent(None)
        enabled = self._enabled_for(lang)
        for key, src in profanity.SOURCES.items():
            if lang not in src.languages:
                continue
            got = self.lexstore.source_path(key, lang).exists()
            meta = self.lexstore.manifest.get(f"{key}_{lang}", {})
            row = QHBoxLayout()
            sw = SwitchButton()
            sw.setOnText("")
            sw.setOffText("")
            sw.setChecked(got and key in enabled)
            sw.setEnabled(got)
            sw.checkedChanged.connect(lambda on, k=key: self._toggle_source(k, on))
            row.addWidget(sw)
            lbl = fit(CaptionLabel(f"{key} · {meta.get('count', '?')} · {src.license}" if got else
                                   f"{key} · not downloaded · {src.license}"))
            lbl.setToolTip(f"{src.name}\n{src.url}\n{src.note}" + (f"\nDownloaded: {meta.get('date')}" if got else ""))
            row.addWidget(lbl, 1)
            dl = ToolButton(FIF.DOWNLOAD if not got else FIF.UPDATE)
            dl.setToolTip(("Download" if not got else "Update") + f" the {src.name} list for \"{lang}\"")
            dl.clicked.connect(lambda _=False, k=key: self._download(k))
            row.addWidget(dl)
            self.src_box.addLayout(row)
            if self.src_box.parentWidget() is not None:
                tame(self.src_box.parentWidget())
            for k in range(row.count()):
                w = row.itemAt(k).widget()
                if w:
                    w.show()

    def _toggle_source(self, key: str, on: bool) -> None:
        lang = self._lang_code()
        cur = self._enabled_for(lang)
        cur = cur | {key} if on else cur - {key}
        self._enabled[lang] = sorted(cur)
        self._save_enabled()
        self._db_reload(quiet=True)

    def _download(self, key: str) -> None:
        lang, store = self._lang_code(), self.lexstore
        src = profanity.SOURCES[key]
        self.bus.push(f"Downloading {src.name} ({lang})...")
        w = Worker(lambda ctx: store.download(key, lang), self)
        w.done.connect(lambda n: (self.bus.push(f"{src.name} ({lang}): {n} terms · license {src.license}"),
                                  self._enabled.setdefault(lang, sorted(self._enabled_for(lang))),
                                  self._enabled[lang].append(key) if key != "ldnoobw" and key not in self._enabled[lang]
                                  else None,
                                  self._save_enabled(), self._db_reload(quiet=True),
                                  self._info("List downloaded", f"{key} · {lang}: {n} terms")))
        w.failed.connect(lambda e: self._info("Download failed", e, True))
        w.log.connect(self.bus.push)
        self._dl_worker = w
        w.start()

    def _db_open(self) -> None:
        p = self.lexstore.own_path(self._lang_code())
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(profanity.OWN_HEADER.format(lang=self._lang_code()), encoding="utf-8")
        os.startfile(str(p))  # noqa: S606 - opens the .txt in the system editor

    def _db_add(self) -> None:
        terms = [t.strip() for t in self.query.text().replace(";", ",").split(",") if t.strip()]
        lang = self._lang_code()
        added = [t for t in terms if self.lexstore.add_own(lang, t)]
        if added:
            self.bus.push(f"Added to custom_{lang}.txt: {', '.join(added)}")
            self._info("Own list updated", f"{len(added)} term(s)")
            self._db_reload(quiet=True)
        else:
            self._info("Own list", "Type the terms to add in \"Search\" (supports put*, *word*, !exception)",
                       True)

    def _db_import(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Import old censor list", str(self.lexstore.root),
                                           "Text (*.txt);;All (*.*)")
        if not f:
            return
        lang = self._lang_code()
        tmp = self.lexstore.root / f"_import_{Path(f).name}"
        tmp.write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")  # the chosen file is not touched
        rep = profanity.migrate_legacy(self.lexstore, tmp, lang)
        tmp.with_name(tmp.stem + "_original" + tmp.suffix).unlink(missing_ok=True)
        if rep:
            self.bus.push(f"Imported {Path(f).name} → custom_{lang}.txt: {rep['input']} → {rep['output']} terms · ñ restored: {', '.join(rep['enye_restored']) or 'none'} · {len(rep['families'])} families folded into wildcards")
            self._info("List imported and cleaned", f"{rep['input']} → {rep['output']} terms")
            self._db_reload(quiet=True)

    # ------------------------------------------------------------ search
    def search(self) -> None:
        if not self.words:
            return
        q = self.query.text().strip()
        db_hits = self.lex.scan(self.words) if self.sw_db.isChecked() else []
        s_hits = censor.search(self.words, q, self.sw_acc.isChecked(), self.sw_whole.isChecked()) if q else []
        self.matches = profanity.merge(db_hits, s_hits)
        self.history.clear()
        self._apply_pad()
        self._apply_who(refresh=False)
        self._fill_table()
        n = len(self.matches)
        n_db = sum(m.source.startswith("DB") for m in self.matches)
        self.count_lbl.setText(f"{n} match{'es' if n != 1 else ''} · {n_db} from the lexicon"
                               if (q or db_hits) else "No matches")
        if db_hits:
            self.bus.push(f"Preventive censoring (lexicon {self._lang_code()}): {len(db_hits)} detection(s)")
        if q:
            self.bus.push(f"Search \"{q}\": {len(s_hits)} matches")
        self.current = -1
        self._schedule_render()
        self._update_enabled()
        if n:
            self.jump(0, play=False)

    def _speaker_of(self, m: censor.Match) -> str:
        if self.diar is None:
            return ""
        k, _ = self.diar.who(m.start, m.end)
        return self.names.get(k, _tr("Unknown")) if k is not None else _tr("Unknown")

    def _apply_who(self, refresh: bool = True) -> None:
        """With a speaker chosen, only their matches are censored (the rest are unchecked)."""
        if refresh and self.matches:
            self.history.record()
        sel = self.who.currentText()
        for m in self.matches:
            m.enabled = sel in ("", ALL) or self._speaker_of(m) == sel
        if refresh:
            for r in range(len(self.matches)):
                self._set_row(r)
            self._refresh_regions()
            self._schedule_render()
            self._update_enabled()

    def _apply_pad(self) -> None:
        src = self.voice if self.voice is not None else self.audio
        if src is None:
            return
        for m in self.matches:  # go back to the word boundaries before widening and snapping
            m.start, m.end = self.words[m.i].start, self.words[m.j - 1].end
        censor.refine(self.matches, audio_io.to_mono(src), self.sr, pad_s=self.s_pad.get() / 1000.0)

    def _repad(self) -> None:
        if self.matches:
            self.history.record(merge="pad")
        self._apply_pad()
        self._refresh_regions()
        for r in range(len(self.matches)):
            self._set_row(r)
        self._schedule_render()

    # ------------------------------------------------------------ table and regions
    def _set_row(self, r: int) -> None:
        m = self.matches[r]
        origin = "🛡 " + m.source if m.source.startswith("DB") else m.source
        vals = ["✓" if m.enabled else "✗", str(r + 1), fmt_time(m.start), self._speaker_of(m), m.text, _tr(origin), m.context]
        for k, v in enumerate(vals):
            it = self.table.item(r, k) or QTableWidgetItem()
            it.setText(v)
            if k in (1, 2):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if not m.enabled:
                it.setForeground(QColor("#8a8a8a"))
            elif k == 5 and m.source.startswith("DB"):
                it.setForeground(QColor(DB_COLOR))
                it.setToolTip("Censored by the database (preventive)")
            else:
                it.setForeground(QColor("#ffffff"))
            self.table.setItem(r, k, it)

    def _fill_table(self) -> None:
        # signals blocked: refilling moves Qt's current cell, which must not jump (and auto-play) to row 0
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.matches))
        for r in range(len(self.matches)):
            self._set_row(r)
        self.table.blockSignals(False)
        self._refresh_regions()
        self.pos_lbl.setText(f"0 / {len(self.matches)}")

    @staticmethod
    def _color(m: censor.Match) -> str:
        return OFF if not m.enabled else DB_COLOR if m.source.startswith("DB") else DANGER

    def _refresh_regions(self) -> None:
        self.wave.set_regions([(m.start, m.end) for m in self.matches], [self._color(m) for m in self.matches])
        if 0 <= self.current < len(self.matches):
            self.wave.select_region(self.current, center=False)

    def _region_changed(self, k: int, a: float, b: float) -> None:
        if 0 <= k < len(self.matches):
            m = self.matches[k]
            self.history.record()
            m.start, m.end = a, b
            self._set_row(k)
            self.jump(k, play=False)
            self._schedule_render()

    def jump(self, k: int, play: bool | None = None) -> None:
        """Go to match k (wrapping around): select it, center it and, if enabled, play it."""
        if not self.matches:
            return
        k %= len(self.matches)
        self.current = k
        m = self.matches[k]
        self.table.blockSignals(True)
        self.table.selectRow(k)
        self.table.blockSignals(False)
        self.table.scrollToItem(self.table.item(k, 0))
        self.pos_lbl.setText(f"{k + 1} / {len(self.matches)}")
        w = max(6.0, (m.end - m.start) * 4)
        mid = (m.start + m.end) / 2
        self.wave.vb.setXRange(max(0.0, mid - w / 2), mid + w / 2, padding=0)
        self.wave.select_region(k, center=False)
        self._cut_map = None
        self.player.pause()
        self.player.stop_at = None
        self.player.set_active("A")
        self.source.blockSignals(True)
        self.source.setCurrentItem("A")
        self.source.blockSignals(False)
        self._seek(max(0.0, m.start - 1.0))
        if play if play is not None else self.sw_auto.isChecked():
            self.preview(True)

    def preview(self, censored: bool = True, pre: float = 1.0, post: float = 1.0) -> None:
        """Play only the current segment (1 s before and after), censored or original, and stop by itself."""
        if not (0 <= self.current < len(self.matches)) or self.audio is None:
            return
        m = self.matches[self.current]
        a, b = max(0.0, m.start - pre), m.end + post
        self._cut_map = None
        if censored:
            if self._render_timer.isActive() or "B" not in self.player.sources:
                self._render_timer.stop()
                self.render_preview()  # censoring reflects the latest changes (edges, mode, tone)
            key = "B"
            if self.mode.currentRouteKey() == "cut":
                # in cut audio times shift: they are mapped, and the cursor is drawn in original time
                ma, mb = censor.map_time_after_cut(a, self.matches), censor.map_time_after_cut(b, self.matches)
                cut_at = censor.map_time_after_cut(m.start, self.matches)
                self._cut_map = (cut_at, a - ma, b - mb)
                a, b = ma, mb
        else:
            key = "A"
        self.source.blockSignals(True)
        self.source.setCurrentItem(key)
        self.source.blockSignals(False)
        self.player.set_active(key)
        sr = self.player.sr
        self.player.seek(int(a * sr))
        self.player.stop_at = int(b * sr)
        self.player.play()

    def toggle_current(self) -> None:
        if 0 <= self.current < len(self.matches):
            m = self.matches[self.current]
            self.history.record()
            m.enabled = not m.enabled
            self._set_row(self.current)
            self.wave.update_region(self.current, m.start, m.end, self._color(m))
            self._schedule_render()
            self._update_enabled()

    def delete_current(self) -> None:
        """Del: the match leaves the list (it is not censored and is no longer shown)."""
        if not (0 <= self.current < len(self.matches)):
            return
        self.history.record()
        k = self.current
        del self.matches[k]
        self._after_list_change(k)

    def _after_list_change(self, k: int) -> None:
        self._fill_table()
        n = len(self.matches)
        n_db = sum(m.source.startswith("DB") for m in self.matches)
        self.count_lbl.setText(f"{n} match{'es' if n != 1 else ''} · {n_db} from the lexicon" if n else "No matches")
        self.current = -1
        self._schedule_render()
        self._update_enabled()
        if n:
            self.jump(min(max(k, 0), n - 1), play=False)

    # ------------------------------------------------------------ project
    def project_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def project_files(self) -> list[str]:
        return [str(self.path)] if self.path else []

    def project_state(self) -> dict:
        from .projects import capture_controls
        return {"controls": capture_controls(self, {"who", "source"}), "path": str(self.path) if self.path else None,
                "audio": None if self.audio is self.orig else self.audio, "voice": self.voice, "words": self.words,
                "diar": getattr(self, "diar", None), "names": getattr(self, "names", {}), "matches": self.matches,
                "current": self.current, "who": self.who.currentText() if self.who.currentIndex() > 0 else None}

    def project_apply(self, st: dict) -> None:
        from .projects import apply_controls
        self.project_clear()
        apply_controls(self, st.get("controls", {}))
        if not st.get("path"):
            return
        self.open_file(str(st["path"]))
        if st.get("audio") is not None:
            self._set_audio(st["audio"])
        if st.get("words"):
            self._analyzed((st.get("voice"), st["words"], st.get("diar"), st.get("names") or {}))
            self.matches = st.get("matches") or []
            if st.get("who"):
                self.who.blockSignals(True)
                k = self.who.findText(st["who"])
                if k >= 0:
                    self.who.setCurrentIndex(k)
                self.who.blockSignals(False)
            self.history.clear()
            self._after_list_change(max(0, st.get("current", 0)))
        self.status.setText("Project opened")

    def project_clear(self) -> None:
        self.player.pause()
        self.path = self.audio = self.orig = self.voice = self.rendered = None
        self.words, self.matches, self.current = [], [], -1
        self.diar, self.names = None, {}
        self.history.clear()
        self.who.blockSignals(True)
        self.who.clear()
        self.who.addItem(ALL)
        self.who.blockSignals(False)
        self.wave.clear()
        self.player.set_sources(self.sr, {})
        self._fill_table()
        self.count_lbl.setText("No search")
        self.file_lbl.setText(_tr("No file"))
        self.progress.setValue(0)
        self.status.setText(_tr("Ready"))
        self._update_enabled()

    # ------------------------------------------------------------ undo / redo
    def _snapshot(self):
        return [copy.copy(m) for m in self.matches], self.current

    def _restore(self, state) -> None:
        matches, k = state
        self.matches = [copy.copy(m) for m in matches]
        self._after_list_change(k)

    # ------------------------------------------------------------ render y transporte
    def _schedule_render(self) -> None:
        if self.audio is not None:
            self._render_timer.start()

    def _render_now(self) -> np.ndarray:
        mode = self.mode.currentRouteKey()
        return censor.render(self.audio, self.sr, self.matches, mode, voice=self.voice, beep_hz=self.s_freq.get(),
                             beep_rel_db=self.s_level.get())

    def render_preview(self) -> None:
        if self.audio is None:
            return
        self.rendered = self._render_now()
        self.player.add_source("B", self.rendered)
        if self.player.active == "B":
            self.player.seek(min(self.player.pos, self.rendered.shape[0]))

    def _source_changed(self, key: str) -> None:
        self.player.stop_at = None
        self._cut_map = None
        if key == "B" and "B" not in self.player.sources:
            self.render_preview()
        self.player.set_active(key)

    def toggle_ab(self) -> None:
        self.source.setCurrentItem("B" if self.source.currentRouteKey() == "A" else "A")

    def toggle_play(self) -> None:
        if self.player.sources:
            self.player.stop_at = None
            self.player.pause() if self.player.playing else self.player.play()

    def _seek(self, s: float) -> None:
        self.player.stop_at = None
        self._cut_map = None
        self.player.seek(int(s * self.player.sr))
        self.wave.set_cursor(s)

    def _tick(self) -> None:
        if not self.isVisible() or not self.player.sources:
            return
        t = self.player.position() / self.player.sr
        if self._cut_map is not None and self.player.active == "B":
            cut_at, before, after = self._cut_map  # cursor in original time during the cut preview
            t += before if t < cut_at else after
        if self.player.playing:
            self.wave.set_cursor(t, follow=True)
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.player.length / self.player.sr)}")
        want = _tr("Pause") if self.player.playing else _tr("Play")
        if self.play_btn.text() != want:
            self.play_btn.setText(want)
            self.play_btn.setIcon(FIF.PAUSE if self.player.playing else FIF.PLAY)

    # ------------------------------------------------------------ export
    def export(self, out_path: str | None = None) -> None:
        if self.audio is None:
            return
        fmt = self.fmt.key()
        default = str(audio_io.with_ext(self.path.with_name(f"{self.path.stem}_censored"), fmt))
        out = out_path or QFileDialog.getSaveFileName(self, "Save censored audio", default, format_filter(fmt))[0]
        if not out:
            return
        out = audio_io.with_ext(out, fmt)
        if self.worker is not None and self.worker.isRunning():
            return
        mode = self.mode.currentRouteKey()
        audio, sr, voice, matches = self.audio, self.sr, self.voice, [copy.copy(m) for m in self.matches]
        freq, level = self.s_freq.get(), self.s_level.get()

        def job(ctx):
            ctx.log(f"Rendering the censored audio ({mode})...", 0)
            y = censor.render(audio, sr, matches, mode, voice=voice, beep_hz=freq, beep_rel_db=level)
            ctx.progress(0.6)
            ctx.check()
            ctx.log(f"Writing {out.name}...", 0)
            audio_io.write_audio(out, y, sr, fmt)
            censor.export_list(matches, out.with_suffix(".csv"), mode)
            ctx.progress(1.0)
            return sum(m.enabled for m in matches)

        def done(n: int) -> None:
            self.progress.setValue(100)
            self.status.setText("Exported")
            self.bus.push(f"Censored audio exported: {n} segments ({mode}) → {out}")
            self._info("Exported", f"{out.name} · {n} segments censored")

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.log.connect(lambda m, _l: self.status.setText(m))
        self.worker.done.connect(done)
        self.worker.failed.connect(lambda e: (self.status.setText("Error (see log)"), self._info("Error", e, True)))
        self.worker.cancelled.connect(lambda: self.status.setText("Export aborted"))
        self.worker.finished.connect(self._update_enabled)
        self.progress.setValue(0)
        self.worker.start()
        self._update_enabled()
