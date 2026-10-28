"""Shared test helpers: corpus cache and ffmpeg ebur128 reference runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import corpus  # noqa: E402

_CORPUS_DIR: Optional[str] = None
_MANIFEST: Optional[dict] = None

REAL_DIR = os.environ.get("AUDIO_DIAG_TEST_MUSIC_DIR", os.path.expanduser("~/Música/OMNi"))
REAL_FILES = {
    "master": REAL_DIR + "/Projects/02 - Schranz Hard Techno (158 BPM)/master.wav",
    "melodic": REAL_DIR + "/User Library/Samples/Synths/Melodic Schranz/MS Melodic Schranz Project (Live 11).wav",
    "oldschool": REAL_DIR + "/User Library/Samples/Synths/Melodic Schranz/MS Oldschool Schranz Project (Live 11).wav",
    "vocal": REAL_DIR + "/User Library/Samples/Vocals/Misc/MS Schranz Vocal Tool Project (Live 11).wav",
}


def corpus_dir() -> str:
    """Build the corpus once per test process into a throw-away temp dir."""
    global _CORPUS_DIR, _MANIFEST
    if _CORPUS_DIR is None:
        _CORPUS_DIR = tempfile.mkdtemp(prefix="audiodiag-corpus-")
        _MANIFEST = corpus.build_corpus(_CORPUS_DIR)
        import atexit

        atexit.register(shutil.rmtree, _CORPUS_DIR, True)
    return _CORPUS_DIR


def manifest() -> dict:
    corpus_dir()
    assert _MANIFEST is not None
    return _MANIFEST


def item_path(name: str) -> str:
    return os.path.join(corpus_dir(), manifest()["items"][name]["file"])


def ffmpeg_ebur128(path: str) -> Dict[str, float]:
    """Run ``ffmpeg -af ebur128=peak=true`` (read-only) and parse the summary.

    Returns integrated LUFS, LRA and true peak (dBTP). Momentary/short-term
    maxima are taken from the per-frame log with ``framelog=verbose``.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, check=True)
    text = proc.stderr
    summary = text.split("Summary:")[-1]
    out: Dict[str, float] = {}
    m = re.search(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", summary)
    out["integrated"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"LRA:\s+(-?[\d.]+)\s+LU", summary)
    out["lra"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS", summary)
    out["true_peak"] = float(m.group(1)) if m else float("nan")
    ms = [float(v) for v in re.findall(r"\bM:\s*(-?\d+\.\d+)", text.split("Summary:")[0])]
    ss = [float(v) for v in re.findall(r"\bS:\s*(-?\d+\.\d+)", text.split("Summary:")[0])]
    out["momentary_max"] = max(ms) if ms else float("nan")
    out["short_term_max"] = max(ss) if ss else float("nan")
    return out


def real_files_available() -> bool:
    return all(os.path.exists(p) for p in REAL_FILES.values())
