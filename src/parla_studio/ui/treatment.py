"""Reusable separation panel: the same engines as the Leveler tab (voice, noise, music and their
combinations) with a button that runs only the separation and shows the result on the waveform."""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, PushButton, StrongBodyLabel

from ..core import models
from .common import Worker
from .level_page import ParamSlider, fit, switch_row


class SeparationPanel(QWidget):
    """Engine + separation block. `run_btn` separates the loaded audio; `restore_btn` goes back to the original."""

    def __init__(self, parent=None, title: str = "Separation (voice, noise, music)") -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(6)
        lay.addWidget(StrongBodyLabel(title))
        self.engine = ComboBox()
        self.engine.setMinimumWidth(10)
        self._engine_keys = list(models.ENGINES)
        self.engine.addItems([models.ENGINES[k] for k in self._engine_keys])
        default = models.DEFAULT_ENGINE if models.DEFAULT_ENGINE in self._engine_keys else self._engine_keys[0]
        self.engine.setCurrentIndex(self._engine_keys.index(default))  # voc_fv7 + gabox + karaoke: lead voice only
        self.engine_hint = fit(CaptionLabel(models.ENGINE_HINTS.get(default, "")))
        self.engine.currentIndexChanged.connect(lambda *_: self.engine_hint.setText(
            models.ENGINE_HINTS.get(self.engine_key(), "")))
        lay.addWidget(self.engine)
        lay.addWidget(self.engine_hint)
        self.sw_auto = switch_row(lay, "Automatic block and GPU batch", True)
        self.sw_auto.setToolTip("Sizes the separation block from the free RAM and the GPU batch from the free VRAM when "
                                "the process starts: as fast as this machine allows, never spilling VRAM into RAM")
        self.s_block = ParamSlider("Separation block (s)", 30, 600, 120, 0)
        lay.addWidget(self.s_block)
        self.auto_hint = fit(CaptionLabel())
        lay.addWidget(self.auto_hint)
        self.sw_auto.checkedChanged.connect(lambda *_: self.refresh_auto())
        self.engine.currentIndexChanged.connect(lambda *_: self.refresh_auto())
        self.refresh_auto()
        row = QHBoxLayout()
        self.run_btn = PushButton(FIF.MUSIC, "Separate now")
        self.run_btn.setToolTip("Runs only the separation and replaces the waveform audio with the result")
        self.restore_btn = PushButton(FIF.HISTORY, "Original")
        self.restore_btn.setToolTip("Goes back to the audio as it was loaded")
        row.addWidget(self.run_btn, 3)
        row.addWidget(self.restore_btn, 2)
        self.row = row  # tabs may add tools here (undo / redo of the chain)
        lay.addLayout(row)

    def engine_key(self) -> str:
        return self._engine_keys[self.engine.currentIndex()]

    def engine_name(self) -> str:
        return models.ENGINES.get(self.engine_key(), self.engine_key())

    def block_s(self) -> float | None:
        """None = automatic (decided when the process starts, from the memory free at that moment)."""
        return None if self.sw_auto.isChecked() else self.s_block.get()

    def refresh_auto(self) -> None:
        auto = self.sw_auto.isChecked()
        self.s_block.setEnabled(not auto)
        self.auto_hint.setVisible(auto)
        if auto:
            from ..core import memory
            key = self.engine_key()
            mss = [k for k in key.split("+") if k in models.MSS_MODELS]
            batch = min((memory.mss_batch(models.MSS_MODELS[k]["yaml"]) for k in mss), default=0)
            text = f"Now: block {memory.separation_block_s(key):.0f} s"
            if batch:
                text += f" · GPU batch {batch}"
            text += f" · {memory.describe()}"
            self.auto_hint.setText(text)

    def showEvent(self, e) -> None:  # noqa: N802 - Qt name: free memory changes, refresh when shown
        super().showEvent(e)
        self.refresh_auto()


def run_separation(page, items: list, panel: SeparationPanel, on_done) -> None:
    """Separate in the background. `items` = [(name, get)], where get() → (audio, sr) is called in the worker
    thread (so loading files doesn't block the interface); `on_done([(separated, sr), ...])`.
    `page` must have worker, progress (0..100), status, bus and _update_enabled."""
    if not items or (page.worker is not None and page.worker.isRunning()):
        return
    engine, block, label = panel.engine_key(), panel.block_s(), panel.engine_name()

    def job(ctx):
        out = []
        for k, (name, get) in enumerate(items):
            ctx.check()
            x, sr = get()
            ctx.log(f"{name}: separating with {label}...", 0)
            out.append((models.isolate_voice(x, sr, ctx.sub(k / len(items), (k + 1) / len(items)), engine, block), sr))
        return out

    page.worker = Worker(job, page)
    page.worker.progress.connect(lambda v: page.progress.setValue(int(v * page.progress.maximum())))
    page.worker.log.connect(page.bus.push)
    page.worker.log.connect(lambda m, l: page.status.setText(m.split(": ", 1)[-1]))
    page.worker.done.connect(on_done)
    page.worker.failed.connect(lambda e: (page.status.setText("Error (see log)"),
                                          InfoBar.error("Error", e, duration=8000, position=InfoBarPosition.TOP_RIGHT,
                                                        parent=page.window())))
    page.worker.cancelled.connect(lambda: page.status.setText("Separation aborted"))
    page.worker.finished.connect(page._update_enabled)
    page.status.setText(f"Separating with {label}...")
    page.progress.setValue(0)
    page.worker.start()
    page._update_enabled()
