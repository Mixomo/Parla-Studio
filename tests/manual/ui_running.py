"""Screenshots of the Leveler tab during a long inference and after aborting.

usage: python tests/manual/ui_running.py <long_audio> <folder> <width>x<height>
"""
import sys
import time
from pathlib import Path


def main() -> None:
    import os
    from parla_studio.paths import ROOT as _ROOT
    os.environ.setdefault("HF_HOME", str(_ROOT / "models"))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import Theme, setTheme, setThemeColor
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    w_, h_ = (int(v) for v in sys.argv[3].split("x"))
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    setThemeColor("#34d399")
    from parla_studio.ui.main_window import MainWindow
    win = MainWindow()
    win.resize(w_, h_)
    win.show()
    lp = win.level
    tag = f"{w_}x{h_}"

    def shot(name):
        win.grab().save(str(out / f"{tag}_{name}.png"))
        print("screenshot", tag, name, "·", lp.status.text()[:120], flush=True)

    def s1():
        lp.load(src)
        wait(s2)

    def wait(then):
        def check():
            if lp.worker is None or lp.worker.isRunning():
                QTimer.singleShot(200, check)
            else:
                QTimer.singleShot(300, then)
        QTimer.singleShot(300, check)

    def s2():
        lp.start_run(str(out))
        QTimer.singleShot(9000, s3)

    def s3():
        shot("1_processing")
        lp.abort()
        wait(s4)

    def s4():
        shot("2_aborted")
        lp.sw_sep.setChecked(False)
        QTimer.singleShot(400, lambda: (shot("3_no_separation"), app.quit()))

    QTimer.singleShot(500, s1)
    app.exec()


if __name__ == "__main__":
    main()
