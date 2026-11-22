"""Speakers tab: who speaks when (Nemotron 3 Diarization), identification of enrolled people
(WavLM voice print), transcript by turn and per-person project export."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QFileDialog, QHBoxLayout, QHeaderView, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, EditableComboBox, FluentIcon as FIF, InfoBar,
                            InfoBarPosition, LineEdit, ListWidget, MessageBoxBase, PrimaryPushButton, ProgressBar,
                            PushButton, StrongBodyLabel, SubtitleLabel, SwitchButton, TableWidget, ToolButton)

from ..core import audio_io, models, speakers as spk
from ..i18n import tr as _tr
from .common import ACCENT, History, LogBus, Worker, format_combo, undo_buttons
from .dataset_page import side_panel, switch_row
from .level_page import ParamSlider, fit, fmt_time, tame
from .treatment import SeparationPanel, run_separation
from .player import Player
from .waveform import WaveView

SP_COLS = ["", "Name", "Recognized person", "Time", "Turns"]


class ReferenceDialog(MessageBoxBase):
    """Enroll or improve a person with reference clips (3-5 clips of 5 s or more is ideal)."""

    def __init__(self, names: list[str], parent=None) -> None:
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel("Enroll a person with reference clips"))
        info = CaptionLabel("Use clean recordings of the person speaking alone: 3 to 5 clips of 5 s or more. If a clip has several voices the main one is used; clips that don't match the rest are discarded.")
        info.setWordWrap(True)
        self.viewLayout.addWidget(info)
        self.viewLayout.addWidget(StrongBodyLabel("Person (new or existing)"))
        self.name = EditableComboBox()
        self.name.addItems(names)
        self.name.setText("")
        self.name.setPlaceholderText("Name")
        self.viewLayout.addWidget(self.name)
        row = QHBoxLayout()
        add = PushButton(FIF.ADD, "Add clips")
        add.clicked.connect(self._add)
        rem = PushButton(FIF.DELETE, "Remove")
        rem.clicked.connect(lambda: [self.files.takeItem(self.files.row(i)) for i in self.files.selectedItems()])
        row.addWidget(add)
        row.addWidget(rem)
        self.viewLayout.addLayout(row)
        self.files = ListWidget()
        self.files.setMinimumHeight(140)
        self.viewLayout.addWidget(self.files)
        row = QHBoxLayout()
        row.addWidget(BodyLabel("Isolate the voice of each clip"))
        row.addStretch(1)
        self.iso = SwitchButton()
        self.iso.setOnText("Yes")
        self.iso.setOffText("No")
        self.iso.setChecked(True)
        row.addWidget(self.iso)
        self.viewLayout.addLayout(row)
        self.yesButton.setText("Create voice print")
        self.cancelButton.setText("Cancel")
        self.widget.setMinimumWidth(520)

    def _add(self) -> None:
        fs, _ = QFileDialog.getOpenFileNames(self, "Reference clips", "",
                                             "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All (*.*)")
        known = {self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())}
        for f in fs:
            if f not in known:
                self.files.addItem(Path(f).name)
                self.files.item(self.files.count() - 1).setData(Qt.UserRole, f)

    def paths(self) -> list[str]:
        return [self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())]

    def validate(self) -> bool:
        return bool(self.name.text().strip()) and self.files.count() > 0
TR_COLS = ["Time", "Speaker", "Text", "Overlap"]


class SlotRow(QWidget):
    """A speaker slot: color, name, reference clips and delete."""

    def __init__(self, page: "SpeakersPage", i: int) -> None:
        super().__init__()
        self.page, self.i = page, i
        slot = page.slots.slots[i]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.color = ToolButton()
        self.color.setFixedSize(28, 28)
        self.color.setToolTip("Speaker color")
        self.color.clicked.connect(self._pick)
        lay.addWidget(self.color)
        self.name = LineEdit()
        self.name.setText(slot.name)
        self.name.editingFinished.connect(self._rename)
        lay.addWidget(self.name, 1)
        ref = ToolButton(FIF.MICROPHONE)
        ref.setToolTip("Load reference clips of this person")
        ref.clicked.connect(self._clips)
        lay.addWidget(ref)
        self.info = CaptionLabel()
        self.info.setFixedWidth(46)
        lay.addWidget(self.info)
        rm = ToolButton(FIF.CLOSE)
        rm.setToolTip("Remove slot")
        rm.clicked.connect(lambda: page.remove_slot(self.i))
        lay.addWidget(rm)
        self.refresh()

    def refresh(self) -> None:
        slot = self.page.slots.slots[self.i]
        self.color.setStyleSheet(f"ToolButton {{background: {slot.color}; border-radius: 6px; border: 1px solid #ffffff55}}")
        prof = self.page.store.profiles.get(slot.name)
        self.info.setText(f"{prof.samples} clip{'s' if prof.samples != 1 else ''}" if prof else "")
        self.info.setToolTip(f"{prof.seconds:.0f} s of reference" if prof else "No reference clips")

    def _pick(self) -> None:
        slot = self.page.slots.slots[self.i]
        c = QColorDialog.getColor(QColor(slot.color), self, f"Color of {slot.name}")
        if c.isValid():
            slot.color = c.name()
            self.page.slot_changed()

    def _rename(self) -> None:
        slot = self.page.slots.slots[self.i]
        new = self.name.text().strip()
        if new and new != slot.name:
            self.page.store.rename(slot.name, new)  # their reference clips are still theirs
            slot.name = new
            self.page.slot_changed()

    def _clips(self) -> None:
        name = self.page.slots.slots[self.i].name
        fs, _ = QFileDialog.getOpenFileNames(self, f"Reference clips of {name}", "",
                                             "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All (*.*)")
        if fs:
            self.page.add_reference(name, fs, True)


class SpeakersPage(QWidget):
    def __init__(self, bus: LogBus, store: spk.ProfileStore, slots: spk.SlotStore | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("speakersPage")
        self.bus, self.store = bus, store
        self.slots = slots or spk.SlotStore(__import__("parla_studio.paths", fromlist=["SLOTS"]).SLOTS)
        self.player = Player()
        self.path: Path | None = None
        self.audio: np.ndarray | None = None
        self.orig: np.ndarray | None = None   # audio as loaded ("Original" goes back to it)
        self.voice: np.ndarray | None = None
        self.sr = 48000
        self.diar: spk.Diarization | None = None
        self.speakers: list[spk.Speaker] = []
        self.separated: list[spk.Separated] = []
        self.words: list[models.Word] = []
        self.turns: list[spk.Turn] = []
        self.language = "es"
        self.worker: Worker | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 8, 8)
        root.setSpacing(10)
        scroll, c = side_panel()
        t = SubtitleLabel("Speakers")
        t.setStyleSheet(f"color: {ACCENT}")
        self.title_row = QHBoxLayout()  # the main window adds the Project button here
        self.title_row.addWidget(t, 1)
        c.addLayout(self.title_row)
        self.project_box = QVBoxLayout()  # ... and the project name here
        self.project_box.setContentsMargins(0, 0, 0, 0)
        c.addLayout(self.project_box)
        intro = CaptionLabel("Who speaks when (up to 8 voices, with overlaps), who each enrolled person is and what they said. Exports one track per person, SRT, RTTM and a JSON project. The original is not modified.")
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
        self.sw_iso = switch_row(c, "Isolate the voice (cleaner transcripts and voice prints)", True)
        self.sep = SeparationPanel()
        self.sep.run_btn.clicked.connect(self.separate_now)
        self.sep.restore_btn.clicked.connect(lambda: self.orig is not None and self._change_audio(self.orig))
        c.addWidget(self.sep)
        self.sw_tr = switch_row(c, "Transcribe (CrisperWhisper 2.0 turbo)", True)
        self.sw_sava = switch_row(c, "Detect voice with SAVA", False)
        self.sw_sava.setToolTip("SAVA (AI voice activity) checks where there really is speech: speaker activity over music, noise or effects is removed (cleaner turns, voice prints and tracks) and words transcribed where nobody speaks are dropped")
        self.sw_sep = switch_row(c, "Separate overlapping voices (TIGER)", True)
        self.s_thr = ParamSlider("Identification threshold", 0.6, 0.95, 0.80, 2)
        c.addWidget(self.s_thr)
        self.progress = ProgressBar()
        self.progress.setValue(0)
        c.addWidget(self.progress)
        self.status = fit(CaptionLabel("Ready"), lines=3)
        self.status.setWordWrap(True)
        c.addWidget(self.status)
        self.run_btn = PrimaryPushButton(FIF.PEOPLE, "Analyze speakers")
        self.run_btn.clicked.connect(self.analyze)
        self.abort_btn = PushButton(FIF.CANCEL, "Abort")
        self.abort_btn.setToolTip("Stops the running process (analysis, separation, export...) within seconds")
        self.abort_btn.clicked.connect(self.abort)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.abort_btn, 2)
        c.addLayout(run_row)

        c.addWidget(StrongBodyLabel("Speakers (slots)"))
        slots_hint = CaptionLabel("Name and color of each speaker. The microphone button loads reference clips of that person: they will be recognized by voice and take this slot.")
        slots_hint.setWordWrap(True)
        c.addWidget(slots_hint)
        self.slot_box = QVBoxLayout()
        self.slot_box.setSpacing(4)
        c.addLayout(self.slot_box)
        self.add_slot_btn = PushButton(FIF.ADD, "Add speaker")
        self.add_slot_btn.clicked.connect(self.add_slot)
        c.addWidget(self.add_slot_btn)
        c.addWidget(StrongBodyLabel("Speaker detection"))
        self.detect = ComboBox()
        self.detect.addItems(["Automatic (adds slots if more voices appear)", "Limit to the slots"])
        c.addWidget(self.detect)
        c.addWidget(StrongBodyLabel("Enrolled people"))
        self.people = ListWidget()
        self.people.setFixedHeight(120)
        c.addWidget(self.people)
        ref = PrimaryPushButton(FIF.MICROPHONE, "Enroll with reference clips")
        ref.clicked.connect(lambda: self.add_reference())
        c.addWidget(ref)
        rm = PushButton(FIF.DELETE, "Delete person")
        rm.clicked.connect(self.remove_person)
        c.addWidget(rm)
        hint = CaptionLabel("You can also enroll a detected speaker: rename them in the table and press \"Enroll as person\". The voice print only uses parts without overlapping voices.")
        hint.setWordWrap(True)
        c.addWidget(hint)

        self.sw_trk = switch_row(c, "Per-person tracks from the isolated voice", False)
        c.addWidget(StrongBodyLabel("Track format"))
        self.fmt = format_combo("wav24")
        c.addWidget(self.fmt)
        self.export_btn = PrimaryPushButton(FIF.SAVE, "Export project")
        self.export_btn.clicked.connect(lambda: self.export())
        c.addWidget(self.export_btn)
        c.addStretch(1)
        tame(scroll.widget())
        root.addWidget(scroll)

        center = QVBoxLayout()
        bar = QHBoxLayout()
        self.play_btn = PrimaryPushButton(FIF.PLAY, "Play")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.clicked.connect(self.toggle_play)
        bar.addWidget(self.play_btn)
        bar.addWidget(BodyLabel("Listen:"))
        self.listen = ComboBox()
        self.listen.setMinimumWidth(160)
        self.listen.currentIndexChanged.connect(self._listen_changed)
        bar.addWidget(self.listen)
        for icon, tip, fn in ((FIF.ZOOM_OUT, "Zoom out", lambda: self.wave.zoom(1.5)),
                              (FIF.ZOOM_IN, "Zoom in", lambda: self.wave.zoom(1 / 1.5)),
                              (FIF.FIT_PAGE, "Fit", lambda: self.wave.fit())):
            b = ToolButton(icon)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        self.history = History(self._snapshot, self._restore, limit=10)  # whole audios: few steps
        for b in undo_buttons(self, self.history):
            b.setToolTip(_tr(b.toolTip()) + " · " + _tr("separation chain and renames"))
            self.sep.row.addWidget(b)
        bar.addStretch(1)
        self.time_lbl = BodyLabel("00:00.00 / 00:00.00")
        self.time_lbl.setStyleSheet("font-family: Consolas; font-size: 15px")
        bar.addWidget(self.time_lbl)
        center.addLayout(bar)
        self.wave = WaveView()
        self.wave.seekRequested.connect(self._seek)
        self.wave.setMinimumHeight(220)   # waveform and level lanes stay readable
        center.addWidget(self.wave, 3)

        sbar = QHBoxLayout()
        sbar.addWidget(StrongBodyLabel("Detected speakers"))
        sbar.addStretch(1)
        for icon, text, fn in ((FIF.PLAY, "Play sample", self.play_sample),
                               (FIF.ADD, "Enroll as person", self.enroll_selected)):
            b = PushButton(icon, text)
            b.clicked.connect(fn)
            sbar.addWidget(b)
        center.addLayout(sbar)
        self.sp_table = self._table(SP_COLS, stretch=1)
        self.sp_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.sp_table.itemChanged.connect(self._renamed)
        self.sp_table.setMinimumHeight(110)
        self.sp_table.setMaximumHeight(190)
        center.addWidget(self.sp_table)
        center.addWidget(StrongBodyLabel("Transcript by turn"))
        self.tr_table = self._table(TR_COLS, stretch=2)
        self.tr_table.cellClicked.connect(lambda r, _c: self._seek(self.turns[r].start) if r < len(self.turns) else None)
        center.addWidget(self.tr_table, 2)
        keys = CaptionLabel("Double-click a name to rename · Click a turn: go to it · Space: play · Ctrl+Z / Ctrl+Y: undo / redo")
        keys.setWordWrap(True)
        center.addWidget(keys)
        root.addLayout(center, 1)

        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play, context=Qt.WidgetWithChildrenShortcut)
        self.timer = QTimer(self, interval=33, timeout=self._tick)
        self.timer.start()
        self._refresh_people()
        self._build_slots()
        self._update_enabled()

    # ------------------------------------------------------------ slots
    def _build_slots(self) -> None:
        rows = getattr(self, "slot_rows", [])
        if len(rows) == len(self.slots.slots):  # same count: update in place
            for r in rows:
                r.name.setText(self.slots.slots[r.i].name)
                r.refresh()
        else:
            while self.slot_box.count():
                w = self.slot_box.takeAt(0).widget()
                if w is not None:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
            self.slot_rows = [SlotRow(self, i) for i in range(len(self.slots.slots))]
            for r in self.slot_rows:
                self.slot_box.addWidget(r)
                r.show()  # the panel is already visible: new widgets don't show by themselves
        self.add_slot_btn.setEnabled(len(self.slots.slots) < spk.MAX_SPEAKERS)
        self.add_slot_btn.setText(f"Add speaker ({len(self.slots.slots)}/{spk.MAX_SPEAKERS})")
        if hasattr(self, "slot_box") and self.slot_box.parentWidget() is not None:
            tame(self.slot_box.parentWidget())

    def add_slot(self) -> None:
        if self.slots.add() is not None:
            self._build_slots()

    def remove_slot(self, i: int) -> None:
        self.slots.remove(i)
        for sp in self.speakers:  # speakers are mapped again
            sp.slot = -1
        self.slot_changed(remap=True)

    def slot_changed(self, remap: bool = False) -> None:
        """A slot's name or color changed: propagate it to speakers, lanes, tables and listening."""
        self.slots.save()
        if remap and self.speakers:
            spk.apply_slots(self.speakers, self.slots, auto_add=False)
        for sp in self.speakers:
            if 0 <= sp.slot < len(self.slots.slots):
                sl = self.slots.slots[sp.slot]
                sp.name, sp.color = sl.name, sl.color
        self._build_slots()
        self._refresh_people()
        self._fill_tables()
        self._draw_lanes()
        self._fill_listen()

    def _table(self, cols: list[str], stretch: int) -> TableWidget:
        t = TableWidget()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels([f"   {c}   " for c in cols])  # padding: the Fluent style uses a wider font
        t.verticalHeader().hide()
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = t.horizontalHeader()
        # the Fluent header paints with a larger font than it measures: each column gets room for its whole title
        fm = hh.fontMetrics()
        for i in range(len(cols)):
            if i == stretch:
                hh.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                hh.setSectionResizeMode(i, QHeaderView.Interactive)
                hh.resizeSection(i, int(fm.horizontalAdvance(t.horizontalHeaderItem(i).text()) * 1.25) + 24)
        return t

    # ------------------------------------------------------------ estado
    def abort(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("Aborting: stops when the current step ends (seconds)...")
            self.bus.push("Aborting at the user's request...", 1)

    def _update_enabled(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        self.open_btn.setEnabled(not busy)
        self.yt.set_tab_busy(busy)
        self.abort_btn.setEnabled(busy)
        self.run_btn.setEnabled(not busy and self.audio is not None)
        self.sep.run_btn.setEnabled(not busy and self.audio is not None)
        self.sep.restore_btn.setEnabled(not busy and self.orig is not None and self.audio is not self.orig)
        self.export_btn.setEnabled(not busy and self.diar is not None)

    def _info(self, title: str, text: str, error: bool = False) -> None:
        (InfoBar.error if error else InfoBar.success)(title, text, duration=8000 if error else 4000,
                                                     position=InfoBarPosition.TOP_RIGHT, parent=self.window())

    def _refresh_people(self) -> None:
        self.people.clear()
        for p in sorted(self.store.profiles.values(), key=lambda p: p.name.lower()):
            secs = f" · {p.seconds:.0f} s" if p.seconds else ""
            self.people.addItem(f"{p.name}  ·  {p.samples} sample{'s' if p.samples != 1 else ''}{secs}")
            self.people.item(self.people.count() - 1).setData(Qt.UserRole, p.name)

    def remove_person(self) -> None:
        for it in self.people.selectedItems():
            name = it.data(Qt.UserRole)
            self.store.remove(name)
            self.bus.push(f"Person deleted: {name}")
        self._refresh_people()

    # ------------------------------------------------------------ enrollment with reference clips
    def add_reference(self, name: str | None = None, files: list[str] | None = None, isolate: bool = True) -> None:
        if name is None or files is None:
            dlg = ReferenceDialog(sorted(self.store.profiles), self.window())
            if not dlg.exec():
                return
            name, files, isolate = dlg.name.text().strip(), dlg.paths(), dlg.iso.isChecked()
        store = self.store
        self.bus.push(f"Enrolling {name} with {len(files)} reference clip(s)...")

        def job(ctx):
            return spk.enroll_from_clips(store, name, [Path(f) for f in files], isolate, ctx=ctx)

        def done(res) -> None:
            prof, reports = res
            for r in reports:
                self.bus.push(f"  {r.file}: {"used" if r.accepted else "not used"} · {r.seconds:.1f} s of voice · similarity {r.similarity:.2f}{' · ' + r.note if r.note else ''}",
                              0 if r.accepted else 1)
            self._refresh_people()
            self._build_slots()
            if prof is None:
                self._info("No voice print", "No clip was usable (see log)", True)
            else:
                used = sum(r.accepted for r in reports)
                self._info("Person enrolled", f"{prof.name}: {used} of {len(reports)} clips · {prof.seconds:.0f} s of reference in total")
                if self.diar is not None:  # identify again with the new voice print
                    for s in self.speakers:
                        if s.embedding is not None:
                            who, score = self.store.identify(s.embedding, self.s_thr.get())
                            s.score = score
                            if who:
                                s.identity, s.name = who, who
                    self.slot_changed(remap=True)

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.done.connect(done)
        self.worker.failed.connect(lambda e: self._info("Error", e, True))
        self.worker.finished.connect(self._update_enabled)
        self.worker.start()
        self._update_enabled()

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
        self.history.clear()
        self.voice, self.diar, self.speakers, self.words, self.turns = None, None, [], [], []
        self.separated = []
        self.file_lbl.setText(f"{self.path.name}\n{self.audio.shape[0] / self.sr / 60:.1f} min · {self.sr} Hz · {self.audio.shape[1]} channel(s)")
        self.wave.clear()
        self.wave.set_lanes(None)
        self.wave.set_layer("orig", audio_io.to_mono(self.audio), self.sr, "#9a9a9a", "fill")
        self.wave.fit()
        self.player.set_sources(self.sr, {"all": self.audio}, "all")
        self._fill_tables()
        self._fill_listen()
        self.status.setText("Analyze the audio to see who speaks")
        self._update_enabled()

    # ------------------------------------------------------------ separation
    def separate_now(self) -> None:
        if self.audio is None:
            return
        x, sr = self.audio, self.sr  # separate what is there: allows chaining (e.g. music, then noise)
        run_separation(self, [(self.path.name, lambda: (x, sr))], self.sep,
                       lambda res: (self._change_audio(res[0][0]), self.progress.setValue(100),
                                    self.status.setText(f"Separated with {self.sep.engine_name()}"),
                                    self.bus.push(f"{self.path.name}: separated with {self.sep.engine_name()}")))

    def _change_audio(self, y: np.ndarray) -> None:
        self.history.record()
        self._set_audio(y)

    # ------------------------------------------------------------ project
    def project_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def project_files(self) -> list[str]:
        return [str(self.path)] if self.path else []

    def project_state(self) -> dict:
        from .projects import capture_controls
        return {"controls": capture_controls(self, {"listen"}), "path": str(self.path) if self.path else None,
                "audio": None if self.audio is self.orig else self.audio, "voice": self.voice, "diar": self.diar,
                "speakers": self.speakers, "words": self.words, "turns": self.turns, "language": self.language,
                "separated": self.separated}

    def project_apply(self, st: dict) -> None:
        from .projects import apply_controls
        self.project_clear()
        apply_controls(self, st.get("controls", {}))
        if not st.get("path"):
            return
        self.open_file(str(st["path"]))
        if st.get("audio") is not None:
            self._set_audio(st["audio"])
        if st.get("diar") is not None:
            self.diar, self.voice = st["diar"], st.get("voice")
            self.speakers, self.words = st.get("speakers") or [], st.get("words") or []
            self.turns, self.separated = st.get("turns") or [], st.get("separated") or []
            self.language = st.get("language", self.language)
            self.progress.setValue(100)
            self._fill_tables()
            self._draw_lanes()
            self._fill_listen()
        self.history.clear()
        self.status.setText("Project opened")
        self._update_enabled()

    def project_clear(self) -> None:
        self.player.pause()
        self.path = self.audio = self.orig = self.voice = self.diar = None
        self.speakers, self.separated, self.words, self.turns = [], [], [], []
        self.history.clear()
        self.wave.clear()
        self.wave.set_lanes(None)
        self.player.set_sources(self.sr, {})
        self._fill_tables()
        self._fill_listen()
        self.file_lbl.setText(_tr("No file"))
        self.progress.setValue(0)
        self.status.setText(_tr("Ready"))
        self._update_enabled()

    # ------------------------------------------------------------ undo / redo (separation chain and renames)
    def _snapshot(self):
        return self.audio, [(sp.idx, sp.name) for sp in self.speakers]

    def _restore(self, state) -> None:
        y, names = state
        if y is not None and y is not self.audio:
            self._set_audio(y)
        by_idx = {sp.idx: sp for sp in self.speakers}
        changed = False
        for idx, name in names:
            sp = by_idx.get(idx)
            if sp is not None and sp.name != name:
                self._rename_speaker(sp, name)
                changed = True
        if changed:
            self.slot_changed()

    def _set_audio(self, y: np.ndarray) -> None:
        """Replace the working audio (times don't change: the analysis stays valid)."""
        self.player.pause()
        pos = self.player.pos
        self.audio = y
        self.wave.set_layer("orig", audio_io.to_mono(y), self.sr, "#9a9a9a", "fill")
        self.player.set_sources(self.sr, {"all": y}, "all")
        self.player.seek(min(pos, y.shape[0]))
        self._fill_listen()
        self._update_enabled()

    def analyze(self) -> None:
        if self.audio is None:
            return
        x, sr, name = self.audio, self.sr, self.path.name
        engine, block = self.sep.engine_key(), self.sep.block_s()
        iso, transcribe, separate = self.sw_iso.isChecked(), self.sw_tr.isChecked(), self.sw_sep.isChecked()
        use_sava = self.sw_sava.isChecked()
        lang = models.CW_LANGUAGES[self.lang.currentText()]
        thr, store = self.s_thr.get(), self.store
        limit = self.detect.currentIndex() == 1
        n_slots = len(self.slots.slots)

        def job(ctx):
            t = time.perf_counter()
            mono = audio_io.to_mono(x)
            ctx.log(f"{name}: diarizing (Nemotron 3 Diarization)...", 0)
            diar = spk.diarize(mono, sr)
            ctx.progress(0.15)
            voice = None
            if iso:
                ctx.check()
                ctx.log(f"{name}: isolating the voice ({models.ENGINES.get(engine, engine)})...", 0)
                voice = models.isolate_voice(x, sr, ctx.sub(0.15, 0.55), engine, block)
                ctx.progress(0.55)
            ref = audio_io.to_mono(voice) if voice is not None else mono
            speech = None
            if use_sava:
                ctx.check()
                ctx.log(f"{name}: detecting voice activity with SAVA...", 0)
                speech = models.speech_probability(ref, sr, ctx.sub(0.55, 0.6))
                diar = spk.gate_with_speech(diar, speech)
                ctx.log(f"{name}: SAVA: voice in {100 * float((speech >= 0.5).mean()) if speech.size else 0:.0f}% "
                        "of the audio; speaker activity outside it removed", 0)
            ctx.check()
            ctx.log(f"{name}: voice prints and identification...", 0)
            speakers = spk.describe(diar, ref, sr, store, thr)
            if limit and len(speakers) > n_slots:
                ctx.log(f"{name}: {len(speakers)} voices detected → merged into {n_slots} (slot limit)", 1)
                speakers = spk.limit_speakers(diar, speakers, n_slots)
            ctx.progress(0.65)
            separated = []
            if separate and diar.overlap.any():
                ctx.check()
                regions = spk.overlap_regions(diar)
                ctx.log(f"{name}: separating {len(regions)} overlapped segment(s) with TIGER...", 0)
                # on the ORIGINAL: the isolator keeps the loudest voice and could erase the other one
                separated = spk.separate_overlaps(mono, sr, diar, speakers)
                spk.identify_with_separated(speakers, separated, store, thr)  # voice prints of whoever almost always talks over others
            ctx.progress(0.75)
            words = []
            if transcribe:
                ctx.check()
                ctx.log(f"{name}: transcribing with CrisperWhisper 2.0 turbo ({lang})...", 0)
                _, words = models.transcribe(ref, sr, lang, ctx=ctx.sub(0.75, 0.95))
                if separated:
                    ctx.log(f"{name}: transcribing the {2 * len(separated)} overlapping voices separately...", 0)
                    words = spk.merge_separated_words(words, separated,
                                                      lambda v, r: models.transcribe(v, r, lang)[1])
                if speech is not None:
                    words, dropped = models.words_with_voice(words, speech)
                    if dropped:
                        ctx.log(f"{name}: SAVA: dropped {len(dropped)} word(s) where nobody speaks: "
                                f"{' '.join(w.text for w in dropped[:30])}", 1)
            known = [s.name for s in speakers if s.identity]
            ctx.log(f"{name}: {len(speakers)} speaker(s) · {diar.overlap.sum() * diar.frame_s:.1f} s overlapped · recognized: {', '.join(known) or 'ninguno'} · {time.perf_counter() - t:.0f} s", 0)
            return diar, voice, speakers, words, lang, separated

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.log.connect(lambda m, l: self.status.setText(m.split(": ", 1)[-1]))
        self.worker.done.connect(self._analyzed)
        self.worker.failed.connect(lambda e: (self.status.setText("Error (see log)"), self._info("Error", e, True)))
        self.worker.cancelled.connect(lambda: self.status.setText("Analysis aborted"))
        self.worker.finished.connect(self._update_enabled)
        self.progress.setValue(0)
        self.worker.start()
        self._update_enabled()

    def _analyzed(self, res) -> None:
        self.diar, self.voice, self.speakers, self.words, self.language, self.separated = res
        self.history.clear()  # new speakers: earlier renames no longer apply
        spk.apply_slots(self.speakers, self.slots, auto_add=self.detect.currentIndex() == 0)
        self._build_slots()
        self.turns = spk.turns(self.words, self.diar, separated=self.separated) if self.words else [
            spk.Turn(s.start, s.end, s.speaker, "", float(self.diar.overlap[int(s.start / 0.01):int(s.end / 0.01)].mean()))
            for s in self.diar.segments()]
        self.progress.setValue(100)
        self._fill_tables()
        self._draw_lanes()
        self._fill_listen()
        self._update_enabled()

    # ------------------------------------------------------------ tablas
    def _names(self) -> dict[int, str]:
        return {s.idx: s.name for s in self.speakers}

    def _fill_tables(self) -> None:
        t = self.sp_table
        t.blockSignals(True)
        t.setRowCount(len(self.speakers))
        for r, s in enumerate(self.speakers):
            vals = ["■", s.name, f"{s.identity} ({s.score:.2f})" if s.identity else
                    (_tr(f"unknown (best {s.score:.2f})") if self.store.profiles else "—"),
                    fmt_time(s.seconds), str(s.turns)]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k != 1:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if k == 0:
                    it.setForeground(QColor(s.color))
                t.setItem(r, k, it)
        t.blockSignals(False)
        names = self._names()
        tt = self.tr_table
        tt.setRowCount(len(self.turns))
        for r, tu in enumerate(self.turns):
            who = names.get(tu.speaker, _tr("Unknown")) if tu.speaker is not None else _tr("Unknown")
            sep = any(sp.start < tu.end and tu.start < sp.end for sp in self.separated)
            vals = [fmt_time(tu.start), who, tu.text,
                    (_tr("separated (TIGER)") if sep else _tr("⚠ yes")) if tu.overlap > 0.2 else ""]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k == 1 and tu.speaker is not None:
                    it.setForeground(QColor(next((s.color for s in self.speakers if s.idx == tu.speaker), "#ffffff")))
                tt.setItem(r, k, it)

    def _draw_lanes(self) -> None:
        if self.diar is None or not self.speakers:
            self.wave.set_lanes(None)
            return
        self.wave.set_lanes(self.diar.active[:, [s.idx for s in self.speakers]], self.diar.frame_s,
                            [s.name for s in self.speakers], [s.color for s in self.speakers])

    def _renamed(self, item: QTableWidgetItem) -> None:
        r = item.row()
        if item.column() == 1 and 0 <= r < len(self.speakers):
            sp = self.speakers[r]
            new = item.text().strip() or sp.name
            if new == sp.name:
                return
            self.history.record()
            self._rename_speaker(sp, new)
            self.slot_changed()

    def _rename_speaker(self, sp: spk.Speaker, new: str) -> None:
        if 0 <= sp.slot < len(self.slots.slots):
            self.store.rename(self.slots.slots[sp.slot].name, new)
            self.slots.slots[sp.slot].name = new
        sp.name = new

    def _sel_speaker(self) -> spk.Speaker | None:
        rows = self.sp_table.selectionModel().selectedRows()
        return self.speakers[rows[0].row()] if rows and rows[0].row() < len(self.speakers) else None

    def enroll_selected(self) -> None:
        s = self._sel_speaker()
        if s is None or s.embedding is None:
            self._info("Enroll", "Choose a speaker with enough non-overlapping voice", True)
            return
        if s.name.startswith("Speaker ") and not s.identity:
            self._info("Enroll", "First type the person's name in the table (double-click)", True)
            return
        p = self.store.enroll(s.name, s.embedding)
        s.identity, s.score = s.name, 1.0
        self.bus.push(f"Person enrolled: {p.name} ({p.samples} sample{'s' if p.samples != 1 else ''})")
        self._refresh_people()
        self._fill_tables()
        self._info("Person enrolled", f"{p.name} will be recognized in future analyses")

    def play_sample(self) -> None:
        s = self._sel_speaker()
        if s is None or self.diar is None:
            return
        segs = [g for g in self.diar.segments() if g.speaker == s.idx]
        if not segs:
            return
        g = max(segs, key=lambda g: g.end - g.start)
        self._set_listen(f"spk{s.idx}")
        self._seek(g.start)
        self.player.stop_at = int(min(g.end, g.start + 8.0) * self.player.sr)
        self.player.play()

    # ------------------------------------------------------------ listening
    def _fill_listen(self) -> None:
        self.listen.blockSignals(True)
        self.listen.clear()
        self._listen_keys = ["all"]
        self.listen.addItem("Everything (original)")
        if self.voice is not None:
            self.listen.addItem("Isolated voice")
            self._listen_keys.append("voice")
        for s in self.speakers:
            self.listen.addItem(f"Only {s.name}")
            self._listen_keys.append(f"spk{s.idx}")
        self.listen.setCurrentIndex(0)
        self.listen.blockSignals(False)
        self.player.set_active("all")

    def _set_listen(self, key: str) -> None:
        if key in getattr(self, "_listen_keys", []):
            self.listen.setCurrentIndex(self._listen_keys.index(key))

    def _listen_changed(self, i: int) -> None:
        keys = getattr(self, "_listen_keys", ["all"])
        if not (0 <= i < len(keys)) or self.audio is None:
            return
        key = keys[i]
        if key not in self.player.sources:
            if key == "voice" and self.voice is not None:
                self.player.add_source("voice", self.voice)
            elif key.startswith("spk") and self.diar is not None:
                self.player.add_source(key, spk.speaker_track(self.audio, self.sr, self.diar, int(key[3:]),
                                                              separated=self.separated))
        self.player.stop_at = None
        self.player.set_active(key)

    def toggle_play(self) -> None:
        if self.player.sources:
            self.player.stop_at = None
            self.player.pause() if self.player.playing else self.player.play()

    def _seek(self, s: float) -> None:
        self.player.stop_at = None
        self.player.seek(int(s * self.player.sr))
        self.wave.set_cursor(s)

    def _tick(self) -> None:
        if not self.isVisible() or not self.player.sources:
            return
        t = self.player.position() / self.player.sr
        if self.player.playing:
            self.wave.set_cursor(t, follow=True)
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self.player.length / self.player.sr)}")
        want = _tr("Pause") if self.player.playing else _tr("Play")
        if self.play_btn.text() != want:
            self.play_btn.setText(want)
            self.play_btn.setIcon(FIF.PAUSE if self.player.playing else FIF.PLAY)

    # ------------------------------------------------------------ export
    def export(self, out_dir: str | None = None) -> None:
        if self.diar is None:
            return
        d = out_dir or QFileDialog.getExistingDirectory(self, "Project folder", str(self.path.parent))
        if not d:
            return
        out = Path(d) / f"{self.path.stem}_project" if not out_dir else Path(d)
        args = (out, self.path, self.audio, self.sr, self.diar, self.speakers, self.words, self.voice, self.language,
                self.sw_trk.isChecked())
        sep, fmt = self.separated, self.fmt.key()
        self.worker = Worker(lambda ctx: spk.export_project(*args, ctx=ctx, separated=sep, fmt=fmt), self)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))
        self.worker.log.connect(self.bus.push)
        self.worker.done.connect(lambda p: (self.bus.push(f"Project exported: {p}"),
                                            self._info("Project exported", str(out))))
        self.worker.failed.connect(lambda e: self._info("Error", e, True))
        self.worker.finished.connect(self._update_enabled)
        self.worker.start()
        self._update_enabled()
