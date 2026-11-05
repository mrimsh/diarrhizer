"""Source entry point; releases ship the independent Diarrhizer.exe instead."""

import multiprocessing
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from diarrhizer_launcher.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
