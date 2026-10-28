"""Expectation mapping tests without official audio dependencies."""
from pathlib import Path
import tempfile
import unittest
from scripts.validate_ebu import expectation, compare, validate_output
from audio_diag.safety import PROTECTED_DIRS


class OutputSafety(unittest.TestCase):
    def test_reject_protected_and_internal_outputs(self):
        dataset, repo = Path('/external/dataset'), Path('/external/repo')
        for root in (dataset, repo, *(Path(p).resolve() for p in PROTECTED_DIRS)):
            for output in (root, root / 'results'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output(dataset, output, repo)

    def test_require_fresh_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                validate_output(root / 'dataset', root / 'repo', root)
            validate_output(root / 'dataset', root / 'repo', root / 'new-results')


class Expectations(unittest.TestCase):
    def test_file_maxima(self):
        for case, metric in ((10, 'short_term_max_lufs'), (13, 'momentary_max_lufs')):
            for index in range(1, 21):
                checks, reason = expectation(f'seq-3341-{case}-{index}-24bit.wav.wav')
                self.assertIsNone(reason)
                self.assertEqual(checks[0]['metric'], metric)
                self.assertEqual(checks[0]['expected'], -23)

    def test_true_peak_asymmetric(self):
        for case in range(15, 24):
            c = expectation(f'seq-3341-{case}-24bit.wav.wav')[0][0]
            expected = -6 if case <= 18 else 3 if case == 19 else 0
            self.assertAlmostEqual(c['lower'], expected-.4)
            self.assertAlmostEqual(c['upper'], expected+.2)
            self.assertEqual(compare(c, {'true_peak_dbtp': c['lower']})['status'], 'pass')
            self.assertEqual(compare(c, {'true_peak_dbtp': c['upper']+.001})['status'], 'fail')
            self.assertEqual(compare(c, {})['status'], 'unknown')

    def test_time_series_not_replaced_with_integrated(self):
        for case in (9, 11, 12, 14):
            checks, reason = expectation(f'seq-3341-{case}-24bit.wav')
            self.assertIn('API limitation', reason)
            self.assertNotIn('integrated_lufs', [c['metric'] for c in checks])

    def test_shared_programmes(self):
        for name, expected in (('seq-3341-7_seq-3342-5-24bit.wav', 5), ('seq-3341-2011-8_seq-3342-6-24bit-v02.wav', 15)):
            checks, reason = expectation(name)
            self.assertIsNone(reason)
            self.assertEqual([c['expected'] for c in checks], [-23, expected])

    def test_multichannel_and_auxiliary(self):
        self.assertIn('NOT SUPPORTED', expectation('seq-3341-6-6channels-WAVEEX-16bit.wav')[1])
        self.assertEqual(expectation('1kHz Sine -20 LUFS-16bit.wav')[0], [])
