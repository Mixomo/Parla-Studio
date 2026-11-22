"""Entry point: python -m parla_studio [audio | project.parla]"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    from . import paths
    paths.ensure()
    # models inside the project folder: portable, and the user's cache is not touched
    os.environ.setdefault("HF_HOME", str(paths.MODELS))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    if os.name == "nt":
        import ctypes
        try:   # own taskbar button and icon (not the Python one)
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ParlaStudio.App")
        except (AttributeError, OSError):
            pass
        if sys.stdout is not None:
            os.system("")   # enable ANSI colours in the log console

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import Theme, setTheme, setThemeColor

    app = QApplication(sys.argv)
    app.setApplicationName("Parla Studio")
    app.setWindowIcon(QIcon(str(paths.ICON)))
    setTheme(Theme.DARK)
    setThemeColor("#34d399")

    from . import i18n
    i18n.install(app)  # English interface by default (data/settings.json → "language")

    from .ui.main_window import MainWindow

    win = MainWindow()
    win.setWindowIcon(QIcon(str(paths.ICON)))
    if sys.stdout is not None:
        print("Parla Studio - the log of every tab is shown here (also saved in data/app.log). "
              "Closing this window closes the app.", flush=True)
    win.showMaximized()  # native size of the current screen; Qt applies the Windows DPI scaling
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        arg = Path(sys.argv[1])
        if arg.suffix.lower() == ".parla":
            win.open_project(arg)
        else:
            win.level.load(arg)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
