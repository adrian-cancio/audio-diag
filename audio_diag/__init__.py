"""audio_diag: stereo-master audio measurements in three reliability levels.

Public API: analyze(path, bpm=None, intents=(), thresholds=None) -> dict,
loudness.measure(samples, sample_rate), DecodeError.
"""
__version__ = '0.1.0'

from .decode import DecodeError  # noqa: E402
from .analyze import analyze, DEFAULT_THRESHOLDS  # noqa: E402

__all__ = ['analyze', 'DecodeError', 'DEFAULT_THRESHOLDS', '__version__']
