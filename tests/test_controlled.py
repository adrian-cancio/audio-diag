"""Analytic controls for phase and intentional silence, independent of real music."""
import unittest
import numpy as np
from audio_diag.spectral import SpectrumAccumulator
from audio_diag.level2 import FeatureAccumulator, analyze_level2


class ControlledTests(unittest.TestCase):
    def test_exact_independent_low_inversion(self):
        from scripts.phase4_controls import invert_right_low
        sr = 48000
        t = np.arange(sr * 2) / sr
        low = 0.2 * np.sin(2 * np.pi * 60 * t)
        high = 0.1 * np.sin(2 * np.pi * 2000 * t)
        x = np.column_stack((low + high, low + high))
        y = invert_right_low(x, sr)
        np.testing.assert_array_equal(y[:, 0], x[:, 0])
        np.testing.assert_allclose(y[:, 1], -low + high, atol=1e-12)
        a, b = np.fft.rfft(x[:, 1]), np.fft.rfft(y[:, 1])
        mask = np.fft.rfftfreq(len(x), 1 / sr) < 120
        np.testing.assert_allclose(b[mask], -a[mask], atol=1e-10)
        np.testing.assert_allclose(b[~mask], a[~mask], atol=1e-10)

    def test_low_phase_cancellation(self):
        sr = 48000
        t = np.arange(sr * 2) / sr
        low = 0.2 * np.sin(2 * np.pi * 60 * t)
        high = 0.1 * np.sin(2 * np.pi * 2000 * t)
        outputs = []
        for sign in (1, -1):
            meter = SpectrumAccumulator(sr, 2)
            meter.update(np.column_stack((low + high, sign * low + high)))
            outputs.append(meter.result()['stereo']['bands'])
        self.assertGreater(outputs[0]['below_120']['correlation'], 0.99)
        self.assertLess(outputs[1]['below_120']['correlation'], -0.99)
        self.assertGreater(outputs[1]['above_500']['correlation'], 0.99)

    def test_intentional_break_stays_heuristic(self):
        sr = 48000
        t = np.arange(sr * 16) / sr
        x = 0.2 * np.sin(2 * np.pi * 60 * t)
        x[(t >= 8) & (t < 12)] = 0
        feature = FeatureAccumulator(sr)
        feature.update(np.column_stack((x, x)))
        result = analyze_level2(feature, sr, 158, {'L': -14, 'R': -14},
                                {'L': -18, 'R': -18}, False, ['break 8–12 s'])
        drops = result['energy_drops']
        self.assertEqual(drops['confidence'], 'heuristic')
        self.assertTrue(any(d['kind'] == 'break' and 7 < d['start_s'] < 9
                            for d in drops['drops']))
        self.assertNotIn('fail', drops.values())
