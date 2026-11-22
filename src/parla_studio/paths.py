"""App paths. Everything generated while using it lives in data/ (outside git); the models, in models/."""
from __future__ import annotations

import sys
from pathlib import Path


def root() -> Path:
    """App folder (the repo's, or the executable's when packaged)."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parents[2]


ROOT = root()
MODELS = ROOT / "models"
DATA = ROOT / "data"
LOGS = DATA / "logs"            # diagnostic logs (words and cuts)
LEXICONS = DATA / "lexicons"    # censor lists: custom list per language + downloaded sources
APP_LOG = DATA / "app.log"      # session log
SETTINGS = DATA / "settings.json"
ICON = Path(__file__).resolve().parent / "assets" / "icon.png"
ICON_ICO = ICON.with_suffix(".ico")
PROFILES = DATA / "profiles.json"
SLOTS = DATA / "slots.json"
DOWNLOADS = DATA / "downloads"  # audio fetched from online videos
RECENT_PROJECTS = DATA / "projects.json"  # project manager list
PROJECTS = ROOT / "projects"              # projects (.parla) folder: listed in every tab, like presets


def ensure() -> None:
    for d in (DATA, LOGS, LEXICONS):
        d.mkdir(parents=True, exist_ok=True)
