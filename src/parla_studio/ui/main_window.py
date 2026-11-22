"""Main window (Fluent): Leveler, Voice dataset, Speakers, Censor and Settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from PySide6.QtCore import QEvent, QProcess, QTimer
from PySide6.QtWidgets import QApplication, QFrame
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, FluentIcon as FIF, FluentWindow, InfoBar, InfoBarPosition,
                            MessageBox, NavigationItemPosition, PushButton, ScrollArea, SimpleCardWidget,
                            SmoothScrollArea,
                            StrongBodyLabel, SubtitleLabel, SwitchButton)

from .. import __version__, i18n, settings
from ..core import models, speakers
from .common import ACCENT, LogBus, chime
from .censor_page import CensorPage
from .dataset_page import DatasetPage
from .speakers_page import SpeakersPage
from .level_page import LevelPage, fit
from .player import Player


class SettingsPage(QWidget):
    def __init__(self, bus: LogBus, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self.bus = bus
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = SmoothScrollArea()  # the page fits any window height
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent")
        body = QWidget()
        body.setStyleSheet("background: transparent")
        scroll.setWidget(body)
        outer.addWidget(scroll)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        t = SubtitleLabel("Settings")
        t.setStyleSheet(f"color: {ACCENT}")
        lay.addWidget(t)

        card0 = SimpleCardWidget()
        c0 = QHBoxLayout(card0)
        col = QVBoxLayout()
        col.addWidget(StrongBodyLabel("Interface language"))
        col.addWidget(CaptionLabel("Applied when the app restarts."))
        c0.addLayout(col, 1)
        self.ui_lang = ComboBox()
        self._lang_keys = list(i18n.LANGUAGES)
        self.ui_lang.addItems(list(i18n.LANGUAGES.values()))
        for it, name in zip(self.ui_lang.items, i18n.LANGUAGES.values()):
            it.text = name  # each language with its own name (untranslated: "Español", not "Spanish")
        self.ui_lang.setCurrentIndex(self._lang_keys.index(i18n.LANG))
        self.ui_lang.setMinimumWidth(160)
        self.ui_lang.currentIndexChanged.connect(self._change_language)
        c0.addWidget(self.ui_lang)
        lay.addWidget(card0)

        card_s = SimpleCardWidget()
        cs = QHBoxLayout(card_s)
        col = QVBoxLayout()
        col.addWidget(StrongBodyLabel("Sound when a task finishes"))
        col.addWidget(fit(CaptionLabel("A short chime when a process of any tab ends (done, error or aborted).")))
        cs.addLayout(col, 1)
        test = PushButton(FIF.VOLUME, "Test")
        test.clicked.connect(lambda: chime("done"))
        cs.addWidget(test)
        self.chime = SwitchButton()
        self.chime.setOnText(i18n.tr("On"))
        self.chime.setOffText(i18n.tr("Off"))
        self.chime.setChecked(bool(settings.get("chime", True)))
        self.chime.checkedChanged.connect(lambda on: settings.set("chime", bool(on)))
        cs.addWidget(self.chime)
        lay.addWidget(card_s)

        card = SimpleCardWidget()
        cl = QVBoxLayout(card)
        cl.addWidget(StrongBodyLabel("Compute device"))
        self.dev = BodyLabel(models.device_description())
        cl.addWidget(self.dev)
        cl.addWidget(StrongBodyLabel("Models folder"))
        cl.addWidget(BodyLabel(os.environ.get("HF_HOME", "(user's Hugging Face cache)")))
        row = QHBoxLayout()
        free = PushButton(FIF.DELETE, "Free VRAM (unload models)")
        free.clicked.connect(self._free)
        row.addWidget(free)
        row.addStretch(1)
        cl.addLayout(row)
        lay.addWidget(card)

        card2 = SimpleCardWidget()
        c2 = QVBoxLayout(card2)
        c2.addWidget(StrongBodyLabel("Models and licenses"))
        for name, text in (
            ("SAVA (Inworld AI)", "Voice activity detection per 40 ms frame · Apache-2.0"),
            ("voice-isolator (Inworld AI)", "Voice / background separation at 48 kHz · Apache-2.0"),
            ("Nemotron 3 Diarization (NVIDIA)", "Who speaks when, up to 8 voices and overlaps · OpenMDW-1.1"),
            ("WavLM base plus SV (Microsoft)", "Voice prints to recognize enrolled people"),
            ("CrisperWhisper 2.0 turbo (nyra health)", "Verbatim transcription with word timestamps · NON-commercial research license (weights and transcripts)"),
            ("becruily deux · Gabox voc_fv7 / denoise-debleed", "Voice / music / noise separation (Mel-Band RoFormer) · becruily CC-BY-NC-4.0, Gabox"),
            ("Qwen3-ForcedAligner · MMS · wav2vec2 XLSR-53", "Forced word alignment (more precise clip edges and censoring)"),
            ("TIGER", "Overlapping voice separation · Apache-2.0"),
        ):
            c2.addWidget(BodyLabel(name))
            c2.addWidget(fit(CaptionLabel(text)))
        lay.addWidget(card2)
        lay.addWidget(CaptionLabel(f"Parla Studio {__version__} · log in data/app.log"))
        lay.addStretch(1)

    def _change_language(self, i: int) -> None:
        lang = self._lang_keys[i]
        if lang == i18n.LANG:
            return
        i18n.save_language(lang)
        box = MessageBox("Change language", "The app restarts to apply the language. Restart now?",
                         self.window())
        box.yesButton.setText(i18n.tr("Restart now"))
        box.cancelButton.setText(i18n.tr("Later"))
        if box.exec():
            win = self.window()
            if win.close():  # closes players and threads; if something refuses, it doesn't restart
                QProcess.startDetached(sys.executable, ["-m", "parla_studio", *sys.argv[1:]])
                QApplication.quit()

    def _free(self) -> None:
        models.unload()
        self.bus.push("Models unloaded from video memory")
        InfoBar.success("VRAM freed", "Models are loaded again when used", duration=3000,
                        position=InfoBarPosition.TOP_RIGHT, parent=self.window())


class PageFrame(ScrollArea):
    """Each tab inside a scroll area: below a comfortable size (where every text and button is shown whole)
    scroll bars appear instead of squeezing, and the window itself can always shrink to fit the screen
    (its title bar and buttons stay visible). Above that size the tab fills the window."""

    def __init__(self, page: QWidget, comfort: tuple[int, int] = (0, 0)) -> None:
        super().__init__()
        self.page, self.comfort = page, comfort
        self.setObjectName(page.objectName() + "Frame")
        self.setWidgetResizable(False)   # sized by hand in resizeEvent
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QScrollArea {background: transparent; border: none}")
        self.viewport().setStyleSheet("background: transparent")
        self.setWidget(page)

    def _fit(self) -> None:
        vp = self.maximumViewportSize()
        hint = self.page.minimumSizeHint()
        w = max(hint.width(), self.comfort[0])
        h = max(hint.height(), self.comfort[1])
        # a scroll bar takes room from the other direction
        need_h = vp.width() < w
        need_v = vp.height() < h
        bar = self.verticalScrollBar().sizeHint().width()
        vw = vp.width() - (bar if need_v else 0)
        vh = vp.height() - (bar if need_h else 0)
        self.page.resize(max(w, vw), max(h, vh))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._fit()


class MainWindow(FluentWindow):
    MIN_W, MIN_H = 760, 480   # the tabs scroll below their comfortable size
    # size (without the navigation bar) where every text and button of the tab is shown whole
    COMFORT = {"levelPage": (1220, 600), "datasetPage": (1200, 600), "speakersPage": (1180, 600),
               "censorPage": (1232, 600), "projectsPage": (760, 420)}

    def __init__(self, lexicons=None) -> None:
        super().__init__()
        self.setWindowTitle("Parla Studio")
        self.setMicaEffectEnabled(False)  # opaque dark background (Mica is transparent and varies with the desktop)
        from ..paths import APP_LOG, LEXICONS, PROFILES, SLOTS
        self.bus = LogBus(APP_LOG)
        self.player = Player()
        self.level = LevelPage(self.bus, self.player, self)
        self.profiles = speakers.ProfileStore(PROFILES)
        self.dataset = DatasetPage(self.bus, self.profiles, self)
        self.slots = speakers.SlotStore(SLOTS)
        self.speakers = SpeakersPage(self.bus, self.profiles, self.slots, self)
        self.censor = CensorPage(self.bus, self.profiles, lexicons or LEXICONS, self)
        self.settings = SettingsPage(self.bus, self)
        # projects: one independent project per tab, and the manager page
        from ..core import project as prj
        from ..paths import PROJECTS, RECENT_PROJECTS
        from .projects import ProjectController, ProjectsPage
        self.recent = prj.RecentProjects(RECENT_PROJECTS, folder=PROJECTS)
        # every project in the subfolder of its tab: projects/leveler, projects/dataset, ...
        for kind in prj.KINDS:
            try:
                prj.kind_folder(PROJECTS, kind).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        for old, new in prj.organize(PROJECTS):
            self.recent.rename(old, new)
            self.bus.push(f"Project moved to its tab's folder: {new}")
        self.pages = {"leveler": self.level, "dataset": self.dataset, "speakers": self.speakers, "censor": self.censor}
        self.project_ctl = {}
        for kind, page in self.pages.items():
            ctl = ProjectController(page, kind, self.bus, self.recent)
            row = QHBoxLayout()   # own row under the title: the title keeps the whole width of the panel
            row.setSpacing(6)
            for b in ctl.buttons:
                row.addWidget(b, 1)
            page.project_box.addLayout(row)
            page.project_box.addWidget(ctl.label)
            self.project_ctl[kind] = ctl
        self.projects = ProjectsPage(self, self.recent, self)
        for ctl in self.project_ctl.values():
            ctl.on_opened = self.projects.refresh
        self._frames: dict[int, PageFrame] = {}
        for page, icon, name, *pos in ((self.projects, FIF.LIBRARY, "Projects"), (self.level, FIF.MUSIC, "Leveler"),
                                       (self.dataset, FIF.MICROPHONE, "Voice dataset"),
                                       (self.speakers, FIF.PEOPLE, "Speakers"), (self.censor, FIF.REMOVE_FROM, "Censor"),
                                       (self.settings, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)):
            frame = PageFrame(page, self.COMFORT.get(page.objectName(), (0, 0)))
            self._frames[id(page)] = frame
            self.addSubInterface(frame, icon, name, *pos)
        self.navigationInterface.setExpandWidth(200)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.bus.push(f"Parla Studio {__version__} ready · {models.device_description()}")
        self.bus.push(f"Audio output: {self.player.device_name()}")

    def switchTo(self, interface) -> None:
        """Pages are shown inside their scroll frame: switching to a page switches to its frame."""
        super().switchTo(self._frames.get(id(interface), interface))

    # ------------------------------------------------------------ window on screen
    def minimumSizeHint(self):
        return self.minimumSize()   # the pages scroll: never ask for more than MIN_W x MIN_H

    def _keep_on_screen(self) -> None:
        """After restoring (from minimized or maximized) the window must fit the screen it is on, with its title
        bar visible: otherwise it could not be moved, minimized, maximized or closed."""
        if self.isMaximized() or self.isMinimized() or self.isFullScreen():
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        g = self.geometry()
        w, h = min(g.width(), area.width()), min(g.height(), area.height())
        x = min(max(g.x(), area.left()), area.right() - w + 1)
        y = min(max(g.y(), area.top()), area.bottom() - h + 1)
        if (x, y, w, h) != (g.x(), g.y(), g.width(), g.height()):
            self.setGeometry(x, y, w, h)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        QTimer.singleShot(0, self._keep_on_screen)

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._keep_on_screen)

    def open_project(self, path) -> None:
        """Open a project in the tab it belongs to."""
        from ..core import project as prj
        try:
            kind = prj.read_header(Path(path))["kind"]
        except (prj.ProjectError, OSError) as err:
            InfoBar.error("Project", str(err), duration=8000, position=InfoBarPosition.TOP_RIGHT, parent=self)
            return
        self.switchTo(self.pages[kind])
        self.project_ctl[kind].open(path)

    def closeEvent(self, e) -> None:
        self.player.close()
        self.dataset.player.close()
        self.censor.player.close()
        self.speakers.player.close()
        for w in (self.level.worker, self.dataset.worker, self.censor.worker, self.speakers.worker,
                  *(c.worker for c in self.project_ctl.values())):
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(3000)
        super().closeEvent(e)
