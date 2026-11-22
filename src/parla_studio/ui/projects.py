"""Project management: one independent project per tab (.parla), the relink dialog and the Projects page."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (Action, CaptionLabel, DropDownPushButton, FluentIcon as FIF, InfoBar, InfoBarPosition,
                            MessageBox, MessageBoxBase, PrimaryPushButton, PushButton, RoundMenu, SegmentedWidget,
                            SubtitleLabel, TableWidget)

from .. import paths
from ..core import project
from ..i18n import tr as _tr
from .common import ACCENT, LogBus, Worker

def projects_folder() -> Path:
    try:
        paths.PROJECTS.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return paths.PROJECTS


def tab_folder(kind: str) -> Path:
    """projects/<tab>: where the projects of one tab are saved and listed."""
    folder = project.kind_folder(projects_folder(), kind)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return folder


def trash_project(path: Path) -> bool:
    """Send a project and its media folder to the Recycle Bin (recoverable). The original audio is not touched."""
    from PySide6.QtCore import QFile
    path = Path(path)
    ok = True
    for p in (path, project.media_dir(path)):
        if p.exists():
            r = QFile.moveToTrash(str(p))       # PySide6: bool or (bool, path in the trash)
            ok &= bool(r[0] if isinstance(r, tuple) else r)
    return ok


def _inside(p: Path, folder: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def show_projects_folder(kind: str | None = None) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(tab_folder(kind) if kind else projects_folder())))


KIND_NAMES = {"leveler": "Leveler", "dataset": "Voice dataset", "speakers": "Speakers", "censor": "Censor"}
FILTER = f"Parla Studio project (*{project.EXT})"


def _info(parent: QWidget, title: str, text: str, error: bool = False) -> None:
    (InfoBar.error if error else InfoBar.success)(title, text, duration=8000 if error else 4000,
                                                 position=InfoBarPosition.TOP_RIGHT, parent=parent.window())


class _ProjectWorker(Worker):
    releases_models = False
    """Saving / opening a project must not unload the models of the tab."""

    @staticmethod
    def _free() -> None:
        pass


class RelinkDialog(MessageBoxBase):
    """Audio files of the project that are no longer where they were: locate them one by one or search a folder."""

    def __init__(self, missing: list[dict], project_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.missing, self.project_path = missing, project_path
        self.mapping: dict[str, str] = {}
        self.viewLayout.addWidget(SubtitleLabel("Relink audio files"))
        info = CaptionLabel("These original files of the project were moved, renamed or deleted. Point to their new "
                            "location (the project never modifies them).")
        info.setWordWrap(True)
        self.viewLayout.addWidget(info)
        self.table = TableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels([f"  {_tr(c)}  " for c in ("File", "Was in", "Now")])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setMinimumHeight(180)
        self.viewLayout.addWidget(self.table)
        row = QHBoxLayout()
        loc = PushButton(FIF.FOLDER, "Locate selected...")
        loc.clicked.connect(self._locate)
        find = PushButton(FIF.SEARCH, "Search in a folder...")
        find.clicked.connect(self._search)
        row.addWidget(loc)
        row.addWidget(find)
        row.addStretch(1)
        self.viewLayout.addLayout(row)
        self.status = CaptionLabel("")
        self.status.setWordWrap(True)
        self.viewLayout.addWidget(self.status)
        self.yesButton.setText("Apply")
        self.cancelButton.setText("Cancel")
        self.widget.setMinimumWidth(720)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._locate())
        self._fill()

    def _fill(self) -> None:
        self.table.setRowCount(len(self.missing))
        for r, e in enumerate(self.missing):
            now = self.mapping.get(e["path"])
            for k, v in enumerate((e["name"], str(Path(e["path"]).parent), now or _tr("not found"))):
                it = QTableWidgetItem(v)
                it.setToolTip(v)
                if k == 2:
                    it.setForeground(Qt.white if now else Qt.red)
                self.table.setItem(r, k, it)
        left = sum(e["path"] not in self.mapping for e in self.missing)
        self.status.setText(_tr("All files located") if not left else f"{left} {_tr('file(s) still missing')}")
        self.yesButton.setEnabled(left == 0)
        if self.table.currentRow() < 0 and self.missing:
            self.table.selectRow(0)

    def _locate(self) -> None:
        r = self.table.currentRow()
        if not (0 <= r < len(self.missing)):
            return
        e = self.missing[r]
        start = str(self.project_path.parent)
        f, _ = QFileDialog.getOpenFileName(self, f"{_tr('Locate')} {e['name']}", start,
                                           "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff *.opus *.m4a);;All (*.*)")
        if not f:
            return
        self.mapping[e["path"]] = f
        if e.get("size") and os.path.getsize(f) != e["size"]:
            _info(self, "Relink", f"{Path(f).name}: {_tr('the size differs from the original file')}", True)
        # the other missing files are often in the same new folder
        for o in self.missing:
            cand = Path(f).parent / o["name"]
            if o["path"] not in self.mapping and cand.is_file():
                self.mapping[o["path"]] = str(cand)
        self._fill()
        nxt = next((i for i, o in enumerate(self.missing) if o["path"] not in self.mapping), -1)
        if nxt >= 0:
            self.table.selectRow(nxt)

    def _search(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder where the files are now", str(self.project_path.parent))
        if not d:
            return
        names = {e["name"] for e in self.missing if e["path"] not in self.mapping}
        found = project.search_folder(Path(d), names)
        for e in self.missing:
            if e["path"] not in self.mapping and e["name"] in found:
                self.mapping[e["path"]] = found[e["name"]]
        self._fill()


def ask_relink(parent: QWidget, path: Path, entries: list[dict]) -> dict[str, str] | None:
    """Mapping old path → current path for every file of the project, asking the user for the missing ones.
    None if the user cancels."""
    found, missing = project.resolve_files(entries, path)
    if missing:
        dlg = RelinkDialog(missing, path, parent.window())
        if not dlg.exec():
            return None
        found.update(dlg.mapping)
    return found


class ProjectController(QObject):
    """Project of one tab: New / Open / Save / Save as, recent list, Ctrl+S / Ctrl+Shift+S / Ctrl+O.

    The page provides: project_files() → original audio paths, project_state() → dict, project_apply(state),
    project_clear() and project_busy()."""

    def __init__(self, page, kind: str, bus: LogBus, recent: project.RecentProjects) -> None:
        super().__init__(page)
        self.page, self.kind, self.bus, self.recent = page, kind, bus, recent
        self.path: Path | None = None
        self.worker: _ProjectWorker | None = None
        self.on_opened = None   # set by the main window: refreshes the Projects page
        self.button = DropDownPushButton(FIF.FOLDER, "Project")
        self.menu = RoundMenu(parent=self.button)
        self.button.setMenu(self.menu)
        self.menu.aboutToShow.connect(self._build_menu)
        # dedicated buttons next to the Project menu
        self.new_btn = PushButton(FIF.ADD, "New")
        self.new_btn.setToolTip("New project: clears this tab (files, results, edits and settings of the project)")
        self.new_btn.clicked.connect(self.ask_new)
        self.save_btn = PrimaryPushButton(FIF.SAVE, "Save")
        self.save_btn.setToolTip("Save the project of this tab (Ctrl+S)")
        self.save_btn.clicked.connect(self.save)
        self.buttons = (self.new_btn, self.save_btn, self.button)
        self.label = CaptionLabel("")
        self.label.setWordWrap(True)
        for key, fn in (("Ctrl+S", self.save), ("Ctrl+Shift+S", self.save_as), ("Ctrl+O", self.open)):
            QShortcut(QKeySequence(key), page, activated=fn, context=Qt.WidgetWithChildrenShortcut)
        self._show_name()

    # ------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        m = self.menu
        m.clear()
        for old in m.findChildren(RoundMenu):   # submenus of the previous opening
            old.deleteLater()
        m.addAction(Action(FIF.ADD, _tr("New project"), triggered=self.new))
        m.addAction(Action(FIF.FOLDER, _tr("Open project...") + "  Ctrl+O", triggered=lambda: self.open()))
        m.addAction(Action(FIF.SAVE, _tr("Save project") + "  Ctrl+S", triggered=self.save))
        m.addAction(Action(FIF.SAVE_AS, _tr("Save project as...") + "  Ctrl+Shift+S", triggered=self.save_as))
        m.addSeparator()
        lib = self.recent.library(self.kind)
        sub = RoundMenu(f"{_tr('Projects folder')} ({len(lib)})", m)
        sub.setIcon(FIF.LIBRARY)
        for it in lib[:40]:
            sub.addAction(Action(FIF.DOCUMENT, it["name"], triggered=lambda _=False, p=it["path"]: self.open(p)))
        if lib:
            sub.addSeparator()
        sub.addAction(Action(FIF.FOLDER, _tr("Show the projects folder"),
                             triggered=lambda: show_projects_folder(self.kind)))
        m.addMenu(sub)
        lib_paths = {project._norm(it["path"]) for it in lib}
        items = [it for it in self.recent.of(self.kind) if project._norm(it["path"]) not in lib_paths][:10]
        if items:
            sub = RoundMenu(_tr("Recent"), m)
            sub.setIcon(FIF.HISTORY)
            for it in items:
                sub.addAction(Action(it["name"], triggered=lambda _=False, p=it["path"]: self.open(p)))
            m.addMenu(sub)

    def _show_name(self) -> None:
        if self.path:
            self.label.setText(f"{_tr('Project')}: {self.path.stem}")
            self.label.setToolTip(str(self.path))
        else:
            self.label.setText(_tr("Unsaved project"))
            self.label.setToolTip("")

    def busy(self) -> bool:
        return (self.worker is not None and self.worker.isRunning()) or self.page.project_busy()

    # ------------------------------------------------------------ actions
    def ask_new(self) -> None:
        """New project, asking first when the tab has something loaded."""
        if self.busy():
            _info(self.page, "Project", "Wait until the current process finishes", True)
            return
        if self.page.project_files() or self.path is not None:
            box = MessageBox("New project", "Clear this tab and start a new project? Changes not saved in the "
                             "project are lost (the audio files are not touched).", self.page.window())
            if not box.exec():
                return
        self.new()

    def new(self) -> None:
        if self.busy():
            return
        self.page.project_clear()
        self.path = None
        self._show_name()

    def save(self) -> None:
        if self.path is None:
            self.save_as()
        else:
            self._save(self.path)

    def save_as(self) -> None:
        if self.busy():
            return
        files = self.page.project_files()
        folder = tab_folder(self.kind)
        if self.path is not None and _inside(self.path, folder):
            start = self.path            # already in this tab's projects folder (or one of its subfolders)
        else:
            stem = self.path.stem if self.path else (Path(files[0]).stem if files else f"{self.kind}-preset")
            start = folder / f"{stem}{project.EXT}"
        f, _ = QFileDialog.getSaveFileName(self.page, "Save project", str(start), FILTER)
        if f:
            self._save(Path(f))

    def _save(self, path: Path) -> None:
        if self.busy():
            _info(self.page, "Project", "Wait until the current process finishes", True)
            return
        files, state, kind = self.page.project_files(), self.page.project_state(), self.kind
        self.bus.push(f"Saving project {path}")
        self._run(lambda ctx: project.save(path, kind, state, files, ctx), self._saved, "Saving project")

    def _saved(self, path: Path) -> None:
        self.path = path
        self.recent.touch(path, self.kind)
        self._show_name()
        self.bus.push(f"Project saved: {path}")
        _info(self.page, "Project saved", path.name)
        if self.on_opened:
            self.on_opened()

    def open(self, path: str | Path | None = None) -> None:
        if self.busy():
            return
        if not path:
            start = str(self.path.parent) if self.path else str(tab_folder(self.kind))
            path, _ = QFileDialog.getOpenFileName(self.page, "Open project", start, FILTER)
            if not path:
                return
        path = Path(path)
        try:
            doc = project.read_header(path)
        except project.ProjectError as e:
            _info(self.page, "Project", str(e), True)
            return
        except OSError as e:
            _info(self.page, "Project", f"{path.name}: {e}", True)
            return
        if doc["kind"] != self.kind:
            _info(self.page, "Project", f"{_tr('This project belongs to the tab')} {_tr(KIND_NAMES[doc['kind']])}", True)
            return
        mapping = ask_relink(self.page, path, doc["files"])
        if mapping is None:
            return
        self.bus.push(f"Opening project {path}")
        self._run(lambda ctx: (path, project.load(path, mapping, ctx)[1]), self._loaded, "Opening project")

    def _loaded(self, res) -> None:
        path, state = res
        missing = state.pop("__missing_media__", None) if isinstance(state, dict) else None
        if missing:
            self.bus.push(f"Project {path.name}: {len(missing)} derived audio file(s) not found in "
                          f"{project.media_dir(path).name} · separations / results have to be generated again", 1)
            _info(self.page, "Project", f"{_tr('Derived audio not found (separations, results): generate it again')}"
                  f" · {project.media_dir(path).name}", True)
        try:
            self.page.project_apply(state)
        except Exception as e:  # noqa: BLE001 - reported to the user, the tab stays usable
            import traceback
            self.bus.push(traceback.format_exc(), 2)
            _info(self.page, "Project", f"{type(e).__name__}: {e}", True)
            return
        self.path = path
        self.recent.touch(path, self.kind)
        self._show_name()
        self.bus.push(f"Project opened: {path}")
        if self.on_opened:
            self.on_opened()

    def _run(self, fn, done, label: str) -> None:
        self.worker = w = _ProjectWorker(fn, self)
        w.log.connect(self.bus.push)
        w.done.connect(done)
        w.failed.connect(lambda e: _info(self.page, "Project", e, True))
        w.finished.connect(lambda: [b.setEnabled(True) for b in self.buttons])
        for b in self.buttons:
            b.setEnabled(False)
        status = getattr(self.page, "status", None)
        if status is not None:
            status.setText(_tr(label) + "...")
            w.finished.connect(lambda: status.setText(_tr("Ready")))
        w.start()


def title_row(title: QWidget, ctl: ProjectController) -> QVBoxLayout:
    """Tab title with its Project button, and the project name below."""
    col = QVBoxLayout()
    col.setSpacing(2)
    row = QHBoxLayout()
    row.addWidget(title, 1)
    for b in ctl.buttons:
        row.addWidget(b)
    col.addLayout(row)
    col.addWidget(ctl.label)
    return col


class ProjectsPage(QWidget):
    """Project manager: every recent project of every tab, with its status; open, relink, show, forget."""

    def __init__(self, window, recent: project.RecentProjects, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("projectsPage")
        self.win, self.recent = window, recent
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(10)
        t = SubtitleLabel("Projects")
        t.setStyleSheet(f"color: {ACCENT}")
        lay.addWidget(t)
        intro = CaptionLabel("Each tab has its own projects (.parla). A project keeps the whole state of the tab "
                             "(settings, separations, transcripts, clips, matches, speakers and edits) and only "
                             "references the original audio, which is never modified. If the audio moved, relink it. "
                             "Projects are saved in the projects folder, and every tab lists the ones there: a project "
                             "saved with nothing loaded works as a preset of settings. Subfolders group them.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        bar = QHBoxLayout()
        self.filter = SegmentedWidget()
        self.filter.addItem("all", "All")
        for k, name in KIND_NAMES.items():
            self.filter.addItem(k, name)
        self.filter.setCurrentItem("all")
        self.filter.currentItemChanged.connect(lambda *_: self.refresh())
        bar.addWidget(self.filter)
        bar.addStretch(1)
        pf = PushButton(FIF.LIBRARY, "Projects folder")
        pf.setToolTip(str(paths.PROJECTS))
        pf.clicked.connect(lambda: show_projects_folder())
        bar.addWidget(pf)
        op = PrimaryPushButton(FIF.FOLDER, "Open project...")
        op.clicked.connect(self._open_file)
        bar.addWidget(op)
        lay.addLayout(bar)
        self.table = TableWidget()
        cols = ["Project", "Tab", "Folder", "Last opened", "Status"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([f"   {_tr(c)}   " for c in cols])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        for i in range(len(cols)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for icon, text, fn in ((FIF.PLAY, "Open", self._open), (FIF.LINK, "Relink audio...", self._relink),
                               (FIF.FOLDER, "Show in folder", self._show), (FIF.DELETE, "Remove from list", self._forget)):
            b = PushButton(icon, text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        refresh = PushButton(FIF.SYNC, "Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        lay.addLayout(row)
        self.rows: list[dict] = []
        self.refresh()

    def showEvent(self, e) -> None:  # noqa: N802 - Qt name
        super().showEvent(e)
        self.refresh()

    def refresh(self) -> None:
        kind = self.filter.currentRouteKey()
        self.rows = self.recent.everything(None if kind == "all" else kind)
        self.table.setRowCount(len(self.rows))
        for r, it in enumerate(self.rows):
            p = Path(it["path"])
            status, bad = _tr("OK"), False
            if not p.is_file():
                status, bad = _tr("Project file not found"), True
            else:
                try:
                    doc = project.read_header(p)
                    _found, missing = project.resolve_files(doc["files"], p)
                    if missing:
                        status, bad = f"{len(missing)} {_tr('audio file(s) missing: relink')}", True
                except project.ProjectError:
                    status, bad = _tr("Damaged project"), True
            vals = [it["name"], _tr(KIND_NAMES.get(it["kind"], it["kind"])), str(p.parent),
                    it.get("opened", "").replace("T", " "), status]
            for k, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                cell.setToolTip(str(p) if k in (0, 2) else v)
                if k == 4:
                    cell.setForeground(Qt.red if bad else Qt.green)
                self.table.setItem(r, k, cell)

    def _sel(self) -> dict | None:
        r = self.table.currentRow()
        return self.rows[r] if 0 <= r < len(self.rows) else None

    def _open(self) -> None:
        it = self._sel()
        if it:
            self.win.open_project(it["path"])

    def _open_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Open project", str(projects_folder()), FILTER)
        if f:
            self.win.open_project(f)

    def _relink(self) -> None:
        it = self._sel()
        if not it:
            return
        p = Path(it["path"])
        try:
            doc = project.read_header(p)
        except (project.ProjectError, OSError) as e:
            _info(self, "Project", str(e), True)
            return
        found, missing = project.resolve_files(doc["files"], p)
        if not missing and all(k == v for k, v in found.items()):
            _info(self, "Relink", "Every audio file of the project is in place")
            return
        mapping = ask_relink(self, p, doc["files"])
        if mapping is None:
            return
        try:
            project.relink(p, mapping)
        except (OSError, project.ProjectError) as e:
            _info(self, "Relink", str(e), True)
            return
        _info(self, "Relink", f"{p.name}: {_tr('the project now points to the new locations')}")
        self.refresh()

    def _show(self) -> None:
        it = self._sel()
        if not it:
            return
        p = Path(it["path"])
        if p.exists() and os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(p)])  # noqa: S603,S607 - fixed program, path argument
        elif p.parent.exists():
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))

    def _forget(self) -> None:
        it = self._sel()
        if not it:
            return
        if it.get("library"):
            self._delete(it)
            return
        box = MessageBox(_tr("Remove from list"), f"{it['name']}: {_tr('the project file is not deleted, it only leaves this list.')}",
                         self.window())
        if box.exec():
            self.recent.remove(it["path"])
            self.refresh()

    def _delete(self, it: dict) -> None:
        """A project of the projects folder leaves the list by going to the Recycle Bin (with its media folder)."""
        p = Path(it["path"])
        box = MessageBox(_tr("Delete project"),
                         f"{it['name']}: {_tr('the project and its media folder (separations, results) go to the Recycle Bin, where you can restore them. The original audio is not touched.')}",
                         self.window())
        box.yesButton.setText(_tr("Move to the Recycle Bin"))
        if not box.exec():
            return
        for ctl in getattr(self.win, "project_ctl", {}).values():
            if ctl.path is not None and project._norm(str(ctl.path)) == project._norm(str(p)):
                ctl.path = None          # the tab keeps its content, now as an unsaved project
                ctl._show_name()
        if not trash_project(p):
            _info(self, "Delete project", f"{p.name}: {_tr('could not be moved to the Recycle Bin')}", True)
            return
        self.recent.remove(str(p))
        self.refresh()
        _info(self, "Project in the Recycle Bin", p.name)



# ---------------------------------------------------------------- settings of a tab
def _controls(obj, prefix: str = "", depth: int = 0):
    """(name, widget) of every setting of a tab: switches, combos, sliders, segmented selectors and text fields,
    including those of its panels (separation, YouTube). Indices instead of texts: language-independent."""
    from qfluentwidgets import ComboBox, LineEdit, SegmentedWidget as Seg, SwitchButton
    from .level_page import ParamSlider
    for name, w in list(vars(obj).items()):
        if name.startswith("_") or not isinstance(w, QWidget):
            continue
        key = prefix + name
        if isinstance(w, (ParamSlider, SwitchButton, ComboBox, LineEdit, Seg)):
            yield key, w
        elif depth == 0 and type(w).__module__.startswith("parla_studio.ui") and name in ("sep", "yt"):
            yield from _controls(w, key + ".", depth + 1)


def capture_controls(page, skip: set[str] = frozenset()) -> dict:
    from qfluentwidgets import ComboBox, LineEdit, SwitchButton
    from .level_page import ParamSlider
    out = {}
    for key, w in _controls(page):
        if key in skip:
            continue
        if isinstance(w, ParamSlider):
            out[key] = w.get()
        elif isinstance(w, SwitchButton):
            out[key] = w.isChecked()
        elif isinstance(w, ComboBox):
            out[key] = w.currentIndex()
        elif isinstance(w, LineEdit):
            out[key] = w.text()
        else:
            out[key] = w.currentRouteKey()
    return out


def apply_controls(page, values: dict) -> None:
    """Restore the settings (with signals: hints and dependent panels update as if the user changed them)."""
    from qfluentwidgets import ComboBox, LineEdit, SwitchButton
    from .level_page import ParamSlider
    widgets = dict(_controls(page))
    for key, v in values.items():
        w = widgets.get(key)
        if w is None:
            continue
        try:
            if isinstance(w, ParamSlider):
                w.set(float(v))
            elif isinstance(w, SwitchButton):
                w.setChecked(bool(v))
            elif isinstance(w, ComboBox):
                if 0 <= int(v) < w.count():
                    w.setCurrentIndex(int(v))
            elif isinstance(w, LineEdit):
                w.setText(str(v))
            elif v in w.items:
                w.setCurrentItem(v)
        except (TypeError, ValueError):
            continue
