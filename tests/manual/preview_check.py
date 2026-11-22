"""Check the segment preview in the Censor tab (no models, synthetic data).

Captures what actually goes out through the audio callback and verifies that it starts 1 s earlier, that it
stops by itself 1 s later, that the beep (not the voice) plays in the segment and that the original doesn't change.
"""
import sys
import time

import numpy as np

from parla_studio.core import censor, models


def main() -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    from parla_studio.ui.censor_page import CensorPage
    from parla_studio.ui.common import LogBus
    from pathlib import Path
    page = CensorPage(LogBus(Path(sys.argv[1])))
    page.show()

    sr = 48000
    t = np.arange(sr * 8) / sr
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
    bg = (0.02 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)[:, None]
    page.audio, page.voice, page.sr = voice + bg, voice, sr
    page.path = Path("sintetico.wav")
    page.words = [models.Word("hola", 1.0, 1.5), models.Word("palabrota", 3.0, 3.6), models.Word("chau", 5.0, 5.5)]
    page.player.set_sources(sr, {"A": page.audio}, "A")
    page.query.setText("palabrota")
    page.search()
    m = page.matches[0]
    print(f"match {m.text} {m.start:.3f}-{m.end:.3f}")

    captured = []
    orig_cb = page.player._callback

    def spy(out, frames, tinfo, status):
        orig_cb(out, frames, tinfo, status)
        if page.player.playing or np.any(out):
            captured.append(out[:, 0].copy())
    page.player.stream.stop()
    page.player.stream.close()
    import sounddevice as sd
    page.player.stream = sd.OutputStream(samplerate=sr, channels=2, dtype="float32", callback=spy)
    page.player.stream.start()

    ok = True
    for mode, censored in (("beep", True), ("beep", False), ("silence", True), ("cut", True)):
        page.mode.setCurrentItem(mode)
        captured.clear()
        page.preview(censored)
        t0 = time.time()
        while page.player.playing and time.time() - t0 < 10:
            app.processEvents()
            time.sleep(0.01)
        time.sleep(0.15)
        app.processEvents()
        y = np.concatenate(captured) if captured else np.zeros(0)
        nz = np.nonzero(np.abs(y) > 1e-6)[0]
        dur = (nz[-1] - nz[0] + 1) / sr if nz.size else 0.0
        expect = (m.end + 1.0) - (m.start - 1.0)
        # content of the censored segment: energy at 1 kHz vs 220 Hz in the middle of the word
        a = nz[0] + int(1.0 * sr) + int(0.1 * sr) if nz.size else 0
        mid = y[a:a + int(0.3 * sr)]
        spec = np.abs(np.fft.rfft(mid * np.hanning(mid.size))) if mid.size else np.zeros(2)
        f = np.fft.rfftfreq(mid.size, 1 / sr) if mid.size else np.zeros(2)
        e1k = spec[(f > 950) & (f < 1050)].sum()
        e220 = spec[(f > 200) & (f < 240)].sum()
        if mode == "cut" and censored:
            expect -= (m.end - m.start)
        label = f"{mode:7s} {'censored' if censored else 'original'}"
        good = abs(dur - expect) < 0.08
        if censored and mode == "beep":
            good &= e1k > 5 * e220
        if not censored:
            good &= e220 > 5 * e1k
        if censored and mode == "silence":
            good &= e220 < 0.05 * max(1e-9, np.abs(np.fft.rfft(voice[:mid.size, 0])).max())
        print(f"{label}: played {dur:.2f} s (expected {expect:.2f}) · 1 kHz {e1k:.1f} · 220 Hz {e220:.1f} → {'OK' if good else 'FALLA'}")
        ok &= good
    page.player.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
