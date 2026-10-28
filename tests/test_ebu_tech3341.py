"""EBU Tech 3341 minimum-requirement cases, rebuilt from their published descriptions.

The signals are synthesised from the test-case descriptions (levels and durations);
the EBU zip is NOT used. Expected values are the ones in the Tech 3341 table
(+-0.1 LU tolerance). Cases 6 (5.0 channels), 7 and 8 (authentic programme)
and the "live" cases 11 and 14 are not reproduced.

Our implementation is exercised through the public ``loudness.measure``; the
reference values come from the standard, not from ffmpeg.
"""

import os
import tempfile
import unittest

from .support import corpus

from audio_diag import analyze, loudness

TOL = 0.1


def measure(case):
    return loudness.measure(case["signal"], corpus.SR)


class TestEbuTech3341(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = corpus.ebu_cases()

    def check(self, name):
        case = self.cases[name]
        m = measure(case)
        for key, field in (("integrated", "integrated_lufs"), ("momentary_max", "momentary_max_lufs"),
                           ("short_term_max", "short_term_max_lufs")):
            if key in case["expected"]:
                self.assertAlmostEqual(m[field], case["expected"][key], delta=TOL, msg=f"{name} {key}")
        return m

    def test_case01_sine_m23(self):
        m = self.check("case01")
        self.assertAlmostEqual(m["lra_lu"], 0.0, delta=TOL)

    def test_case02_sine_m33(self):
        self.check("case02")

    def test_case03_absolute_gate(self):
        self.check("case03")

    def test_case04_absolute_gate_with_silence_floor(self):
        self.check("case04")

    def test_case05_relative_gate(self):
        self.check("case05")

    def test_case09_short_term_alternating(self):
        self.check("case09")

    def test_case12_momentary_alternating(self):
        self.check("case12")

    def test_case10_short_term_file_based(self):
        for name in [n for n in self.cases if n.startswith("case10")]:
            with self.subTest(name=name):
                self.check(name)

    def test_case13_momentary_file_based(self):
        for name in [n for n in self.cases if n.startswith("case13")]:
            with self.subTest(name=name):
                self.check(name)

    def test_lra_of_stationary_and_stepped_signals(self):
        """LRA is not tabulated by Tech 3341 (the table's "0.0 LU" is loudness relative to
        target), so these are analytic: a stationary tone has LRA 0, and a tone that steps
        between two levels has an LRA equal to the step (percentile spread of two plateaus)
        provided both plateaus pass the -20 LU relative gate."""
        for name in ("case01", "case02"):
            self.assertAlmostEqual(measure(self.cases[name])["lra_lu"], 0.0, delta=TOL, msg=name)
        # case 3: 10 s at -36, 60 s at -23, 10 s at -36. P95 sits on the -23 plateau and P10 on
        # the -36 plateau (20 s of 80 s -> the 10th percentile falls on it): LRA = 13 LU.
        self.assertAlmostEqual(measure(self.cases["case03"])["lra_lu"], 13.0, delta=0.2)

    def test_file_path_matches_library_path(self):
        """The decode+analyze path gives the same answer as the in-memory path (case 5)."""
        case = self.cases["case05"]
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "c5.wav")
            corpus.write_wav(p, case["signal"], corpus.SR, "float32")
            r = analyze(p)
        self.assertAlmostEqual(r["level1"]["loudness"]["integrated_lufs"], -23.0, delta=TOL)


if __name__ == "__main__":
    unittest.main()
