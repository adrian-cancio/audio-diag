"""Identity and windowed-band crest regression tests."""
import hashlib
import unittest
import numpy as np
from .support import item_path
from audio_diag import analyze
from audio_diag.spectral import SpectrumAccumulator

class TestFinish(unittest.TestCase):
    def test_sha256(self):
        p = item_path('sine1k_m20_pcm24')
        with open(p, 'rb') as f:
            expected = hashlib.file_digest(f, 'sha256').hexdigest()
        self.assertEqual(analyze(p)['file']['sha256'], expected)

    def test_global_and_band_crest(self):
        r = analyze(item_path('sine1k_m20_float32'))
        c = r['level2']['crest_factor']
        self.assertAlmostEqual(c['global_db'], 3.0103, delta=0.02)
        # A Hann-windowed sine has peak A and RMS A*sqrt(3/16): 7.27 dB,
        # unlike the unwindowed continuous sine (3.01 dB).
        self.assertAlmostEqual(c['per_band_db']['mid'], 10*np.log10(16/3), delta=0.2)
        self.assertEqual(len(c['per_band_db']), 6)
        self.assertIn('Hann', c['band_definition'])

    def test_band_crest_chunk_invariance(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, .1, (24000, 2))
        a, b = SpectrumAccumulator(48000, 2), SpectrumAccumulator(48000, 2)
        a.update(x)
        for part in np.array_split(x, 17):
            b.update(part)
        a.result(); b.result()
        for name in a.band_peak:
            self.assertAlmostEqual(a.band_peak[name], b.band_peak[name], places=12)
            self.assertAlmostEqual(a.band_square[name], b.band_square[name], places=10)
