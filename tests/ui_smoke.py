"""Full interface test: loads, processes (AI), analyzes a dataset and saves screenshots.

usage: python tests/ui_smoke.py <audio> <output_folder> <spanish_audio> <overlapping_conversation.wav>
"""
import sys
from pathlib import Path

import numpy as np

from parla_studio.paths import ROOT as _ROOT


def main() -> None:
    import os
    os.environ.setdefault("HF_HOME", str(_ROOT / "models"))
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import Theme, setTheme, setThemeColor

    src, out, es = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    conv = Path(sys.argv[4])  # conversation with overlaps (same language)
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)

    def hook(*exc):  # any error in a step ends the test with the traceback
        import traceback
        traceback.print_exception(*exc)
        app.exit(1)
    sys.excepthook = hook
    setTheme(Theme.DARK)
    setThemeColor("#34d399")
    from parla_studio.ui.main_window import MainWindow
    # test lexicon: custom list with "suerte" as a banned term (doesn't touch the real database)
    (out / "database").mkdir(parents=True, exist_ok=True)
    (out / "database" / "custom_es.txt").write_text("# test\nsuerte\n", encoding="utf-8")
    win = MainWindow(lexicons=out / "database")
    win.resize(1600, 950)
    win.show()
    lp, dp, cp, sp = win.level, win.dataset, win.censor, win.speakers
    steps = []

    def shot(name):
        win.grab().save(str(out / f"{name}.png"))
        print("screenshot", name, flush=True)

    def wait_worker(page, then):
        def check():
            if page.worker is None or page.worker.isRunning():
                QTimer.singleShot(200, check)
            else:
                QTimer.singleShot(400, then)
        QTimer.singleShot(300, check)

    def s1():
        lp.load(src)
        wait_worker(lp, s2)

    def s2():
        print("view mode:", lp.view.currentRouteKey(), "source:", lp.player.active, flush=True)
        shot("1_importado")
        lp.start_run(str(out))
        wait_worker(lp, s3)

    def s3():
        print("view mode:", lp.view.currentRouteKey(), "source:", lp.player.active, "sources:", list(lp.player.sources), flush=True)
        lp.player.seek(int(20 * lp.player.sr))
        lp.player.play()
        QTimer.singleShot(1500, s4)

    def s4():
        lp.wave.zoom(0.2, 21)
        print("live meter:", {k: round(v, 1) for k, v in lp.player.meter.items()}, flush=True)
        shot("2_procesado_ia")
        lp.player.pause()
        lp.view.setCurrentItem("voice")
        QTimer.singleShot(300, s5)

    def s5():
        shot("3_voice_background")
        win.switchTo(dp)
        dp.add_files([str(es)])
        dp.lang.setCurrentText("Spanish")
        dp.analyze()
        wait_worker(dp, s6)

    def s6():
        print("clips:", len(dp.clips), "valid:", sum(c.keep for c in dp.clips), "regions:", len(dp.wave.regions),
              "speakers:", sorted({c.speaker for c in dp.clips}), flush=True)
        dp.table.selectRow(1)
        c = dp.clips[1]
        old = (round(c.start, 2), round(c.end, 2), c.text)
        # simulates dragging the out edge 1.5 s earlier (as the user does on the waveform)
        r = dp.wave.regions[1]
        r.setRegion((c.start, c.end - 1.5))
        r.sigRegionChangeFinished.emit(r)
        c2 = dp.clips[1]
        print("edited clip:", old, "->", (round(c2.start, 2), round(c2.end, 2), c2.text), flush=True)
        QTimer.singleShot(600, s7)

    def s7():
        shot("4_dataset_editado")
        from parla_studio.core import dataset
        print("exported:", dataset.export(dp.clips, out / "dataset", dp.prm), flush=True)
        win.switchTo(cp)
        cp.open_file(str(es))
        cp.analyze()
        wait_worker(cp, s8)

    def s8():
        print("preventive (no search):", cp.count_lbl.text(), [(m.text, m.source) for m in cp.matches[:4]], flush=True)
        cp.query.setText("bomba*, Hiroshima, suerte")
        cp.search()
        print("matches:", len(cp.matches), [(m.text, m.source) for m in cp.matches[:10]], flush=True)
        print("source column:", sorted({cp.table.item(r, 5).text() for r in range(cp.table.rowCount())}), flush=True)
        cp.jump(3, play=False)
        cp.jump(cp.current + 1, play=False)
        print("position:", cp.pos_lbl.text(), flush=True)
        cp.source.setCurrentItem("B")
        cp.render_preview()
        cp.player.seek(int((cp.matches[cp.current].start - 0.3) * cp.sr))
        cp.player.play()
        QTimer.singleShot(1200, s9)

    def s9():
        cp.player.pause()
        shot("5_censor")
        for mode in ("beep", "silence", "cut"):
            cp.mode.setCurrentItem(mode)
            cp.export(str(out / f"censored_{mode}.wav"))
            cp.worker.wait()  # the export runs in the background
            app.processEvents()
        print("censored audio exported", flush=True)
        print("speaker per match:", [cp._speaker_of(m) for m in cp.matches[:5]], flush=True)
        win.switchTo(sp)
        for name in list(sp.store.profiles):
            sp.store.remove(name)
        sp.slots.slots = sp.slots.slots[:4]
        for i, sl in enumerate(sp.slots.slots):
            sl.name = f"Speaker {i + 1}"
        # conversation with real overlapping voices (two speakers of the interview, in English)
        import numpy as np
        from parla_studio.core import audio_io
        x, sr = audio_io.load(src)
        refs = []  # reference clips of "Luis" (second speaker of the interview, other segments)
        for n, (a, b) in enumerate(((28.8, 38.9), (51.4, 62.3), (73.4, 84.3))):
            f = out / f"ref_luis_{n + 1}.wav"
            audio_io.save_wav(f, x[int(a * sr):int(b * sr)], sr)
            refs.append(str(f))
        sp.slots.slots[1].name = "Luis"           # slot 2 = Luis (as typed in the slot)
        sp.slots.save()
        sp._build_slots()
        sp.add_reference("Luis", refs)            # slot microphone: reference clips
        wait_worker(sp, s10)

    def s10():
        print("slots:", [(s.name, s.color) for s in sp.slots.slots], flush=True)
        print("people:", [(p.name, p.samples, round(p.seconds)) for p in sp.store.profiles.values()], flush=True)
        sp.open_file(str(conv))
        sp.lang.setCurrentText("English")
        sp.detect.setCurrentIndex(0)              # automatic
        sp.analyze()
        wait_worker(sp, s11)

    def s11():
        print("speakers:", [(s.name, s.identity, round(s.score, 2), s.slot) for s in sp.speakers],
              "separated segments:", len(sp.separated), flush=True)
        for t in sp.turns:
            who = next((s.name for s in sp.speakers if s.idx == t.speaker), "?")
            print(f"   {t.start:5.2f} {who}: {t.text}", flush=True)
        # custom name and color in the other speaker's slot, and one more slot with "+"
        other = next(s for s in sp.speakers if s.name != "Luis")
        row = sp.slot_rows[other.slot]
        row.name.setText("Ana")
        row._rename()
        sp.slots.slots[other.slot].color = "#ff00aa"
        sp.slot_changed()
        sp.add_slot()
        print("after rename/color/+:", [(s.name, s.color) for s in sp.speakers], "slots:", len(sp.slots.slots), flush=True)
        sp.tr_table.selectRow(2)
        shot("6_speakers")
        sp.export(str(out / "project"))
        wait_worker(sp, s12)

    def s12():
        srt = (out / "project" / "transcripts" / "transcript.srt").read_text(encoding="utf-8")
        print("SRT:", srt.replace("\n", " | ")[:700], flush=True)
        files = sorted(str(p.relative_to(out / "project")) for p in (out / "project").rglob("*") if p.is_file())
        print("project:", files, flush=True)
        app.quit()

    QTimer.singleShot(500, s1)
    app.exec()


if __name__ == "__main__":
    main()
