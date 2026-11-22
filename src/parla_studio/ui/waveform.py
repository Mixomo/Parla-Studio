"""Multi-resolution waveform (pyqtgraph) for long audio.

Each layer keeps a min/max pyramid (bins of 64, 256, 1024... frames); when zooming only ~2 points per
visible pixel are computed, so smoothness does not depend on the audio length.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

pg.setConfigOptions(antialias=False, useOpenGL=False, background="#0b0b0c", foreground="#d6d6d6")

BASE, FACTOR = 64, 4


class Pyramid:
    def __init__(self, mono: np.ndarray) -> None:
        self.x = np.ascontiguousarray(mono, dtype=np.float32)
        n = self.x.size // BASE * BASE
        lv = [(self.x[:n].reshape(-1, BASE).min(1), self.x[:n].reshape(-1, BASE).max(1))]
        if n < self.x.size:
            tail = self.x[n:]
            lv[0] = (np.append(lv[0][0], tail.min()), np.append(lv[0][1], tail.max()))
        while lv[-1][0].size > 1:
            mn, mx = lv[-1]
            m = mn.size // FACTOR * FACTOR
            nmn, nmx = mn[:m].reshape(-1, FACTOR).min(1), mx[:m].reshape(-1, FACTOR).max(1)
            if m < mn.size:
                nmn, nmx = np.append(nmn, mn[m:].min()), np.append(nmx, mx[m:].max())
            lv.append((nmn, nmx))
        self.levels = lv

    def envelope(self, start: float, spp: float, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_frames, lo, hi) for `w` columns from `start` with `spp` frames per column."""
        if spp < BASE:
            src_mn = src_mx = self.x
            bin_ = 1
        else:
            li = 0
            while li + 1 < len(self.levels) and BASE * FACTOR ** (li + 1) <= spp:
                li += 1
            src_mn, src_mx = self.levels[li]
            bin_ = BASE * FACTOR ** li
        cols = np.arange(w + 1)
        idx = np.floor((start + cols * spp) / bin_).astype(np.int64)
        idx = np.clip(idx, 0, src_mn.size)
        a, b = idx[:-1], idx[1:]
        valid = a < src_mn.size
        a, b = a[valid], np.maximum(b[valid], a[valid] + 1)
        if a.size == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        end = min(src_mn.size, int(b[-1]))
        # reduceat over contiguous segments [a_i, a_{i+1})
        starts = np.minimum(a, end - 1)
        lo = np.minimum.reduceat(src_mn[:end], starts)
        hi = np.maximum.reduceat(src_mx[:end], starts)
        xs = start + (np.nonzero(valid)[0] + 0.5) * spp
        return xs, lo, hi


class _ViewBox(pg.ViewBox):
    """Wheel: horizontal zoom at the cursor. Ctrl+wheel: height. Drag: scroll."""
    heightWheel = Signal(float)

    def wheelEvent(self, ev, axis=None):
        d = ev.delta() / 120.0
        if ev.modifiers() & Qt.ControlModifier:
            self.heightWheel.emit(d)
            ev.accept()
            return
        s = 0.8 ** d
        center = self.mapToView(ev.pos())
        self.scaleBy(x=s, y=None, center=center)
        ev.accept()


class EditRegion(pg.LinearRegionItem):
    """Editable region: dragging the body moves it, dragging an edge changes the in or out point."""

    def __init__(self, view: "WaveView", idx: int, a: float, b: float, color: QColor) -> None:
        self.view, self.idx = view, idx
        fill = QColor(color)
        fill.setAlpha(45)
        hover = QColor(color)
        hover.setAlpha(80)
        super().__init__((a, b), movable=True, brush=fill, hoverBrush=hover, pen=pg.mkPen(color, width=1.5),
                         hoverPen=pg.mkPen("#ffffff", width=2.5), swapMode="sort", bounds=(0, view.duration))
        self.setZValue(5)
        for line in self.lines:
            line.setCursor(Qt.SizeHorCursor)
        self.sigRegionChangeFinished.connect(lambda r: view._region_moved(self))

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and ev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
            self.view.regionCtrlClicked.emit(self.idx)   # add to / remove from a multiple selection
            ev.accept()
        elif ev.button() == Qt.LeftButton:
            self.view.regionClicked.emit(self.idx)
            t = self.view.vb.mapSceneToView(ev.scenePos()).x()
            self.view.seekRequested.emit(float(np.clip(t, 0, self.view.duration)))
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class WaveView(QWidget):
    seekRequested = Signal(float)  # seconds
    regionChanged = Signal(int, float, float)  # index, start, end (after dragging)
    regionClicked = Signal(int)
    regionCtrlClicked = Signal(int)   # Ctrl/Shift+click on a region: multiple selection

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.vb = _ViewBox()
        self.vb.setMouseEnabled(x=True, y=False)
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.hideAxis("left")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.getAxis("bottom").setStyle(tickTextOffset=4)
        self.vb.setYRange(-1.05, 1.05, padding=0)
        lay.addWidget(self.plot, 5)

        self.vb2 = pg.ViewBox()
        self.vb2.setMouseEnabled(x=True, y=False)
        self.aux = pg.PlotWidget(viewBox=self.vb2)
        self.aux.setMenuEnabled(False)
        self.aux.hideButtons()
        self.aux.setXLink(self.plot)
        self.aux.getAxis("left").setWidth(42)
        self.aux.setLabel("left", "dB")
        self.aux.setMaximumHeight(130)
        lay.addWidget(self.aux, 1)
        self.plot.getAxis("left").setWidth(42)
        self.plot.showAxis("left")
        self.plot.getAxis("left").setTicks([[(-1, "-1"), (0, "0"), (1, "1")]])

        self.sr = 48000
        self.duration = 0.0
        self.height_scale = 1.0
        self.layers: dict[str, dict] = {}
        self.items: dict[str, list] = {}
        self.order: list[str] = []
        self.visible: dict[str, bool] = {}
        self.cursor = pg.InfiniteLine(0, angle=90, pen=pg.mkPen("#ffffff", width=2))
        self.cursor2 = pg.InfiniteLine(0, angle=90, pen=pg.mkPen("#ffffff", width=1))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.aux.addItem(self.cursor2, ignoreBounds=True)
        self.region = pg.LinearRegionItem(movable=False, brush=QColor(52, 211, 153, 40), pen=pg.mkPen("#34d399"))
        self.region.hide()
        self.plot.addItem(self.region, ignoreBounds=True)
        self.regions: list[EditRegion] = []
        self._selected = -1
        self.gain_item = self.aux.plot(pen=pg.mkPen("#34d399", width=1.5))
        self.speech_item = pg.PlotDataItem(pen=None, fillLevel=0, brush=QColor(96, 165, 250, 70), stepMode="center")
        self.aux.addItem(self.speech_item)

        self._timer = QTimer(self, singleShot=True, interval=0)
        self._timer.timeout.connect(self._refresh)
        self.vb.sigXRangeChanged.connect(lambda *_: self._timer.start())
        self.vb.heightWheel.connect(self._height)
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    # ------------------------------------------------------------ data
    def clear(self) -> None:
        for its in self.items.values():
            for it in its:
                self.plot.removeItem(it)
        self.layers.clear()
        self.items.clear()
        self.order.clear()
        self.gain_item.setData([], [])
        self._speech = None
        self.speech_item.setData([], [])
        self.region.hide()
        self.set_regions([])
        self.duration = 0.0

    def set_layer(self, name: str, mono: np.ndarray, sr: int, color: str, style: str = "fill", z: int = 0) -> None:
        """style: 'fill' (min-max bars) or 'outline' (edges only)."""
        self.sr = sr
        self.duration = max(self.duration if self.layers else 0.0, mono.size / sr)
        for it in self.items.pop(name, []):
            self.plot.removeItem(it)
        pen = pg.mkPen(color, width=1 if style == "fill" else 1.4)
        if style == "fill":
            its = [pg.PlotDataItem(pen=pen, connect="pairs")]
        else:
            its = [pg.PlotDataItem(pen=pen), pg.PlotDataItem(pen=pen)]
        for it in its:
            it.setZValue(z)
            self.plot.addItem(it)
        self.layers[name] = {"pyr": Pyramid(mono), "style": style}
        self.items[name] = its
        if name not in self.order:
            self.order.append(name)
        self.visible.setdefault(name, True)
        self.vb.setLimits(xMin=0, xMax=max(self.duration, 0.01), minXRange=64 / sr)
        self.vb2.setLimits(xMin=0, xMax=max(self.duration, 0.01))
        self._timer.start()

    def remove_layer(self, name: str) -> None:
        for it in self.items.pop(name, []):
            self.plot.removeItem(it)
        self.layers.pop(name, None)
        if name in self.order:
            self.order.remove(name)

    def set_visible(self, name: str, on: bool) -> None:
        self.visible[name] = on
        for it in self.items.get(name, []):
            it.setVisible(on)

    def set_gain(self, times: np.ndarray, db: np.ndarray) -> None:
        # downsampling for the plot (pyqtgraph clips to the view)
        step = max(1, db.size // 200000)
        self.gain_item.setData(times[::step], db[::step])
        self.gain_item.setDownsampling(auto=True, method="peak")
        self.gain_item.setClipToView(True)
        self._aux_range()

    def set_speech(self, prob: np.ndarray | None, frame_s: float = 0.04) -> None:
        """Voice probability (0-1) drawn as a blue shade in the lower plot."""
        self._speech = (prob, frame_s) if prob is not None and prob.size else None
        self._aux_range()

    def set_lanes(self, active: np.ndarray | None, frame_s: float = 0.01, names: list[str] | None = None,
                  colors: list[str] | None = None) -> None:
        """Activity lanes per speaker in the lower plot (active: [T, n] boolean).
        Only state changes are drawn, so it stays light with long audio."""
        for it in getattr(self, "_lanes", []):
            self.aux.removeItem(it)
        self._lanes = []
        if active is None or active.size == 0:
            self.aux.getAxis("left").setTicks(None)
            self.aux.setLabel("left", "dB")
            return
        n = active.shape[1]
        for i in range(n):
            a = active[:, i].astype(np.int8)
            ch = np.nonzero(np.diff(np.concatenate(([0], a))))[0]
            xs = np.concatenate(([0], ch, [a.size])) * frame_s
            vals = np.concatenate(([0], a[ch])) if ch.size else np.array([0])
            ys = (n - 1 - i) + 0.8 * vals.astype(np.float32)
            col = QColor(colors[i] if colors else "#60a5fa")
            fill = QColor(col)
            fill.setAlpha(200)
            it = pg.PlotDataItem(xs, ys, stepMode="center", fillLevel=n - 1 - i, brush=fill, pen=pg.mkPen(col, width=1))
            self.aux.addItem(it)
            self._lanes.append(it)
        self.gain_item.setData([], [])
        self.speech_item.setData([], [])
        self.aux.setLabel("left", "")
        self.aux.getAxis("left").setTicks([[(n - 1 - i + 0.4, (names[i] if names else f"S{i + 1}")[:10]) for i in range(n)]])
        self.aux.getAxis("left").setWidth(90)
        self.plot.getAxis("left").setWidth(90)
        self.vb2.setYRange(-0.1, n, padding=0)
        self.aux.setMaximumHeight(max(130, 28 * n + 30))

    def _aux_range(self) -> None:
        g = self.gain_item.yData
        lim = max(float(np.abs(g).max()) if g is not None and len(g) else 0.0, 6.0)
        self.vb2.setYRange(-lim, lim, padding=0.05)
        sp = getattr(self, "_speech", None)
        if sp is None:
            self.speech_item.setData([], [])
        else:
            prob, frame_s = sp
            self.speech_item.setData(np.arange(prob.size + 1) * frame_s, prob * lim, stepMode="center")

    def fit(self) -> None:
        self.height_scale = 1.0
        self.vb.setYRange(-1.05, 1.05, padding=0)
        self.vb.setXRange(0, max(self.duration, 0.01), padding=0)

    def zoom(self, factor: float, center: float | None = None) -> None:
        x0, x1 = self.vb.viewRange()[0]
        c = center if center is not None else (x0 + x1) / 2
        self.vb.setXRange(c - (c - x0) * factor, c + (x1 - c) * factor, padding=0)

    def set_cursor(self, seconds: float, follow: bool = False) -> None:
        self.cursor.setValue(seconds)
        self.cursor2.setValue(seconds)
        if follow:
            x0, x1 = self.vb.viewRange()[0]
            if seconds > x1 or seconds < x0:
                w = x1 - x0
                self.vb.setXRange(seconds - 0.05 * w, seconds + 0.95 * w, padding=0)

    # ------------------------------------------------------------ regiones editables
    def set_regions(self, spans: list[tuple[float, float]], colors: list[str] | None = None) -> None:
        for r in getattr(self, "regions", []):
            self.plot.removeItem(r)
        self.regions: list[EditRegion] = []
        self._selected = -1
        self._marked: set[int] = set()
        for k, (a, b) in enumerate(spans):
            r = EditRegion(self, k, a, b, QColor(colors[k] if colors else "#34d399"))
            self.plot.addItem(r, ignoreBounds=True)
            self.regions.append(r)

    def update_region(self, idx: int, a: float, b: float, color: str | None = None) -> None:
        if 0 <= idx < len(getattr(self, "regions", [])):
            r = self.regions[idx]
            r.blockSignals(True)
            r.setRegion((a, b))
            r.blockSignals(False)
            if color:
                self._color(r, QColor(color))

    def _color(self, r: EditRegion, c: QColor) -> None:
        fill = QColor(c)
        marked = r.idx in getattr(self, "_marked", ())
        fill.setAlpha(110 if r.idx == self._selected else 95 if marked else 45)
        r.setBrush(fill)
        if marked:   # part of a multiple selection: thick white edges, unmistakable
            pen = pg.mkPen("#ffffff", width=3.0)
        else:
            pen = pg.mkPen(c, width=2.5 if r.idx == self._selected else 1.5)
        r.lines[0].setPen(pen)
        r.lines[1].setPen(pen)
        r._base = c
        r.update()

    def mark_regions(self, idxs) -> None:
        """Highlight several regions (multiple selection); an empty set or a single one: normal look."""
        regs = getattr(self, "regions", [])
        new = set(idxs) if len(set(idxs)) > 1 else set()
        old, self._marked = getattr(self, "_marked", set()), new
        for k in old | new:
            if 0 <= k < len(regs):
                self._color(regs[k], getattr(regs[k], "_base", regs[k].lines[0].pen.color()))

    def select_region(self, idx: int, center: bool = True) -> None:
        regs = getattr(self, "regions", [])
        old, self._selected = self._selected, idx
        for k in (old, idx):
            if 0 <= k < len(regs):
                self._color(regs[k], getattr(regs[k], "_base", regs[k].lines[0].pen.color()))
        if center and 0 <= idx < len(regs):
            a, b = regs[idx].getRegion()
            x0, x1 = self.vb.viewRange()[0]
            w = max(x1 - x0, (b - a) * 1.6)
            if not (x0 <= a and b <= x1) or w != x1 - x0:
                m = (a + b) / 2
                self.vb.setXRange(max(0, m - w / 2), m + w / 2, padding=0)

    def _region_moved(self, r: EditRegion) -> None:
        a, b = r.getRegion()
        self.regionChanged.emit(r.idx, float(a), float(b))

    def show_region(self, a: float, b: float, center: bool = True) -> None:
        self.region.setRegion((a, b))
        self.region.show()
        if center:
            w = max((b - a) * 3, 2.0)
            m = (a + b) / 2
            self.vb.setXRange(max(0, m - w / 2), m + w / 2, padding=0)

    # ------------------------------------------------------------ interno
    def _height(self, d: float) -> None:
        self.height_scale = float(np.clip(self.height_scale * (1.25 ** d), 0.05, 2000))
        h = 1.05 / self.height_scale
        self.vb.setYRange(-h, h, padding=0)

    def _clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton or ev.double():
            return
        pos = ev.scenePos()
        if self.plot.sceneBoundingRect().contains(pos):
            t = self.vb.mapSceneToView(pos).x()
            self.seekRequested.emit(float(np.clip(t, 0, self.duration)))

    def _refresh(self) -> None:
        if not self.layers:
            return
        x0, x1 = self.vb.viewRange()[0]
        w = max(1, int(self.vb.width()))
        spp = max(1e-9, (x1 - x0) * self.sr / w)
        start = x0 * self.sr
        for name, layer in self.layers.items():
            xs, lo, hi = layer["pyr"].envelope(start, spp, w)
            t = xs / self.sr
            its = self.items[name]
            if layer["style"] == "fill":
                xx = np.repeat(t, 2)
                yy = np.empty(xx.size, dtype=np.float32)
                yy[0::2], yy[1::2] = lo, hi
                its[0].setData(xx, yy)
            else:
                its[0].setData(t, hi)
                its[1].setData(t, lo)
