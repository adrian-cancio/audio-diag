"""Independent gain-scaling property of normalized power comparisons."""
import unittest
import numpy as np
from audio_diag.spectral import SpectrumAccumulator
from audio_diag.profiles import compare, profile


class ProfileTests(unittest.TestCase):
    def test_gain_invariance(self):
        x = np.random.default_rng(41).normal(size=(24000, 2))
        results = []
        for gain in (1e-4, 0.1, 1, 13):
            meter = SpectrumAccumulator(48000, 2)
            meter.update(x * gain)
            r = {'file': {}, 'version': 'test', 'level1': {'bands': meter.result()}}
            results.append(profile(r))
        for p in results[1:]:
            np.testing.assert_allclose(list(p['relative_bands_db'].values()),
                                       list(results[0]['relative_bands_db'].values()), atol=1e-10)

    def test_extrema_not_average(self):
        result = compare([{'relative_bands_db': {'sub': -9}},
                          {'relative_bands_db': {'sub': -2}}])
        self.assertEqual(result['dispersion']['sub'],
                         {'min_db': -9, 'max_db': -2, 'known_count': 2, 'unknown_count': 0})

    def test_unknown_and_empty(self):
        self.assertIsNone(compare([{'relative_bands_db': {'sub': None}}])['dispersion']['sub']['min_db'])
        with self.assertRaises(ValueError):
            compare([])
