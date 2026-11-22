"""YouTube field for the audio tabs. Each tab has its own instance (own link, range and download), so every tab can
fetch a different video at the same time."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (CaptionLabel, FluentIcon as FIF, InfoBar, InfoBarPosition, LineEdit, ProgressBar,
                            PushButton, StrongBodyLabel)

from .. import paths
from ..core import fetch
from ..i18n import tr as _tr
from .common import LogBus, Worker


class _FetchWorker(Worker):
    releases_models = False
    """A failed download must not unload the models the tab already has in VRAM."""

    @staticmethod
    def _free() -> None:
        pass


class YouTubeField(QWidget):
    """Link + optional time range + Download (click again to abort). Hands the downloaded WAV paths to `on_ready`.

    multi=True (Voice dataset): several links separated by spaces, and playlists add all their videos.
    """

    def __init__(self, bus: LogBus, on_ready: Callable[[list[Path]], None], multi: bool = False,
                 tab_busy: Callable[[], bool] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.bus, self.on_ready, self.multi = bus, on_ready, multi
        self.tab_busy = tab_busy or (lambda: False)
        self.worker: _FetchWorker | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(StrongBodyLabel("From YouTube"))
        self.url = LineEdit()
        self.url.setPlaceholderText("One or more links or playlists" if multi else "YouTube link")
        self.url.setClearButtonEnabled(True)
        self.url.setToolTip("YouTube or any site supported by yt-dlp")
        self.url.returnPressed.connect(self._clicked)
        lay.addWidget(self.url)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.start = LineEdit()
        self.start.setPlaceholderText("Start")
        self.start.setToolTip("Optional, e.g. 1:30 (empty = from the beginning)")
        self.end = LineEdit()
        self.end.setPlaceholderText("End")
        self.end.setToolTip("Optional, e.g. 12:00 (empty = to the end)")
        self.btn = PushButton(FIF.DOWNLOAD, "Download")
        self.btn.setToolTip(f"{_tr('Saved as WAV in')} {paths.DOWNLOADS}")
        self.btn.clicked.connect(self._clicked)
        row.addWidget(self.start, 1)
        row.addWidget(self.end, 1)
        row.addWidget(self.btn, 2)
        lay.addLayout(row)
        self.bar = ProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.msg = CaptionLabel("")
        self.msg.setWordWrap(True)
        self.msg.hide()
        lay.addWidget(self.msg)

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def set_tab_busy(self, busy: bool) -> None:
        """While the tab processes, no new download starts; a running one can still be aborted."""
        free = not busy and not self.busy()
        for w in (self.url, self.start, self.end):
            w.setEnabled(free)
        self.btn.setEnabled(free or self.busy())

    def _say(self, text: str, error: bool = False) -> None:
        self.msg.setStyleSheet("color: #ff6b6b" if error else "")
        self.msg.setText(_tr(text))
        self.msg.setVisible(bool(text))

    def _clicked(self) -> None:
        if self.busy():
            self.worker.cancel()
            self.btn.setText(_tr("Aborting..."))
            return
        if self.tab_busy():
            return
        try:
            urls = fetch.split_urls(self.url.text())
            start, end = fetch.parse_time(self.start.text()), fetch.parse_time(self.end.text())
            if not urls:
                raise ValueError("Paste a link that starts with http")
            if start is not None and end is not None and end <= start:
                raise ValueError("The end must be after the start")
        except ValueError as e:
            self._say(str(e), True)
            return
        self._say("")
        self._run(urls if self.multi else urls[:1], start, end)

    def _run(self, urls: list[str], start: float | None, end: float | None) -> None:
        multi = self.multi

        def job(ctx):
            items = fetch.expand(urls, ctx) if multi else urls
            out, failed = [], []
            for i, url in enumerate(items):
                ctx.check()
                sub = ctx.sub(i / len(items), (i + 1) / len(items))
                try:
                    out.append(fetch.fetch_audio(url, paths.DOWNLOADS, start, end, sub))
                except Exception as e:  # noqa: BLE001 - one bad link must not stop the rest of the list
                    if not multi or ctx.cancel.is_set():
                        raise
                    failed.append(url)
                    ctx.log(f"{url}: {type(e).__name__}: {e}", 2)
            return out, failed

        self.worker = w = _FetchWorker(job, self)
        w.log.connect(self.bus.push)
        w.progress.connect(lambda v: self.bar.setValue(int(v * 1000)))
        w.done.connect(self._done)
        w.failed.connect(lambda e: (self._say(e, True), self._notify("Download failed", e, True)))
        w.cancelled.connect(lambda: (self._say("Download aborted"), self.bus.push("Download aborted by the user", 1)))
        w.finished.connect(self._finished)
        self.bar.setValue(0)
        self.bar.show()
        self.btn.setIcon(FIF.CANCEL)
        self.btn.setText(_tr("Abort"))
        w.start()
        self.set_tab_busy(self.tab_busy())

    def _finished(self) -> None:
        self.bar.hide()
        self.btn.setIcon(FIF.DOWNLOAD)
        self.btn.setText(_tr("Download"))
        self.set_tab_busy(self.tab_busy())

    def _done(self, res) -> None:
        out, failed = res
        for p in out:
            self.bus.push(f"Downloaded: {p}")
        if failed:
            self._say(f"{len(failed)} of {len(failed) + len(out)} links failed (see log)", True)
        elif out:
            self._say(out[0].name if len(out) == 1 else f"{len(out)} files downloaded")
        if out and self.tab_busy():
            self._notify("Download", "Downloaded, but the tab is busy: open the file from the downloads folder when it finishes", True)
        elif out:
            self.on_ready(out)

    def _notify(self, title: str, text: str, error: bool = False) -> None:
        (InfoBar.error if error else InfoBar.success)(title, text, duration=8000 if error else 4000,
                                                     position=InfoBarPosition.TOP_RIGHT, parent=self.window())
