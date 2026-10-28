"""Cross-check of our independent numpy loudness/true-peak against ``ffmpeg -af ebur128=peak=true``.

IMPORTANT: agreement with ffmpeg only validates that our implementation and
libebur128/ffmpeg compute the same thing (integration and gating details). It
is NOT an independent proof of correctness. Absolute correctness is covered by
``test_corpus_expected`` (analytic values) and ``test_ebu_tech3341`` (values
from the standard's test descriptions).

Real local audio files are only ever opened read-only by ffmpeg
and by the decoder.
"""

import unittest

from .support import REAL_FILES, ffmpeg_ebur128, item_path, manifest, real_files_available

from audio_diag import analyze

LU_TOL = 0.1
TP_TOL = 0.2

# Values measured earlier with ffmpeg ebur128 and given in the contract.
CONTRACT_REFERENCE = {
    "melodic": (-9.2, 4.9, 0.2),
    "oldschool": (-8.5, 1.8, 0.8),
    "vocal": (-12.1, 4.8, -1.9),
}


def compare(testcase, path, lra_tol=0.1, compare_lra=True):
    ref = ffmpeg_ebur128(path)
    r = analyze(path)
    ours = r["level1"]["loudness"]
    testcase.assertAlmostEqual(ours["integrated_lufs"], ref["integrated"], delta=LU_TOL, msg="integrated")
    if compare_lra:
        testcase.assertAlmostEqual(ours["lra_lu"], ref["lra"], delta=lra_tol, msg="LRA")
    testcase.assertAlmostEqual(r["level1"]["peaks"]["true_peak_dbtp"], ref["true_peak"], delta=TP_TOL, msg="true peak")
    return ref, r


class TestCorpusVsFfmpeg(unittest.TestCase):
    def test_corpus_items(self):
        names = [n for n, v in manifest()["items"].items()
                 if n not in ("silence", "short_300ms") and v["duration_s"] >= 3.0]
        self.assertGreaterEqual(len(names), 8)
        for name in names:
            with self.subTest(name=name):
                # LRA is skipped where it is ill-defined, and ffmpeg is the odd one out:
                #  * ~3.0 s files: ffmpeg feeds its first, partially filled 3 s window into the
                #    LRA statistics (a constant tone gets LRA 20 LU; the analytic value is 0).
                #  * silence_padded: only ~23 gated blocks on a steep ramp, so one percentile
                #    rank is worth ~0.25 LU (floor/ceil/interp/ffmpeg histogram: 2.32/2.07/2.27/2.6).
                lra_ok = manifest()["items"][name]["duration_s"] >= 5.0 and name != "silence_padded"
                compare(self, item_path(name), compare_lra=lra_ok)

    def test_momentary_and_short_term_max(self):
        for name in ("sine1k_m20_pcm24", "pink_14lufs", "dc_offset"):
            with self.subTest(name=name):
                ref = ffmpeg_ebur128(item_path(name))
                ours = analyze(item_path(name))["level1"]["loudness"]
                self.assertAlmostEqual(ours["momentary_max_lufs"], ref["momentary_max"], delta=0.15)
                self.assertAlmostEqual(ours["short_term_max_lufs"], ref["short_term_max"], delta=0.15)


@unittest.skipUnless(real_files_available(), "real audio files not present")
class TestRealFilesVsFfmpeg(unittest.TestCase):
    def test_against_live_ffmpeg(self):
        for key, path in REAL_FILES.items():
            with self.subTest(file=key):
                compare(self, path)

    def test_against_contract_numbers(self):
        """The contract quotes values from an earlier ffmpeg run (1-decimal rounding)."""
        for key, (lufs, lra, tp) in CONTRACT_REFERENCE.items():
            with self.subTest(file=key):
                r = analyze(REAL_FILES[key])
                self.assertAlmostEqual(r["level1"]["loudness"]["integrated_lufs"], lufs, delta=LU_TOL + 0.05)
                self.assertAlmostEqual(r["level1"]["loudness"]["lra_lu"], lra, delta=LU_TOL + 0.05)
                self.assertAlmostEqual(r["level1"]["peaks"]["true_peak_dbtp"], tp, delta=TP_TOL)


if __name__ == "__main__":
    unittest.main()
