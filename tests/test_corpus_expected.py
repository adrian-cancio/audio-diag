"""Analyzer output versus ANALYTICALLY known corpus values (no ffmpeg involved)."""

import json
import math
import unittest

from .support import item_path, manifest

import audio_diag
from audio_diag import analyze


def run(name, **kw):
    return analyze(item_path(name), **kw)


def no_nonfinite(obj):
    json.dumps(obj, allow_nan=False)


class TestSine(unittest.TestCase):
    def test_sine_loudness_and_peaks_all_containers(self):
        for sub in ("pcm16", "pcm24", "float32"):
            with self.subTest(sub=sub):
                exp = manifest()["items"][f"sine1k_m20_{sub}"]["expected"]
                r = run(f"sine1k_m20_{sub}")
                l1 = r["level1"]
                self.assertAlmostEqual(l1["loudness"]["integrated_lufs"], exp["integrated_lufs"], delta=0.02)
                self.assertAlmostEqual(l1["loudness"]["momentary_max_lufs"], exp["momentary_max_lufs"], delta=0.02)
                self.assertAlmostEqual(l1["loudness"]["short_term_max_lufs"], exp["short_term_max_lufs"], delta=0.02)
                self.assertAlmostEqual(l1["loudness"]["lra_lu"], 0.0, delta=0.05)
                self.assertAlmostEqual(l1["peaks"]["sample_peak_dbfs"], exp["sample_peak_dbfs"], delta=0.001)
                self.assertAlmostEqual(l1["peaks"]["true_peak_dbtp"], -20.0, delta=0.05)
                self.assertEqual(l1["clipping"]["runs_total"], 0)
                self.assertAlmostEqual(l1["dc_offset"]["left"], 0.0, delta=1e-4)

    def test_bit_depth_reported(self):
        self.assertEqual(run("sine1k_m20_pcm16")["file"]["bit_depth"], 16)
        self.assertEqual(run("sine1k_m20_pcm24")["file"]["bit_depth"], 24)
        self.assertEqual(run("sine1k_m20_float32")["file"]["bit_depth"], 32)
        f = run("sine1k_m20_float32")["file"]
        self.assertEqual((f["sample_rate"], f["channels"], f["lossy_source"]), (48000, 2, False))
        self.assertAlmostEqual(f["duration_s"], 5.0, delta=0.01)

    def test_44k_and_mono(self):
        e44 = manifest()["items"]["sine1k_m20_44k"]["expected"]
        self.assertAlmostEqual(run("sine1k_m20_44k")["level1"]["loudness"]["integrated_lufs"], e44["integrated_lufs"], delta=0.02)
        em = manifest()["items"]["sine1k_m20_mono"]["expected"]
        r = run("sine1k_m20_mono")
        self.assertAlmostEqual(r["level1"]["loudness"]["integrated_lufs"], em["integrated_lufs"], delta=0.02)
        self.assertEqual(r["level1"]["stereo"]["status"], "not_applicable")


class TestSilence(unittest.TestCase):
    def test_digital_silence_is_clean(self):
        r = run("silence")
        no_nonfinite(r)
        l1 = r["level1"]
        self.assertIsNone(l1["loudness"]["integrated_lufs"])
        self.assertIn("integrated_lufs_reason", l1["loudness"])
        self.assertIsNone(l1["peaks"]["true_peak_dbtp"])
        self.assertTrue(l1["silence"]["total_silent"])
        self.assertEqual(l1["clipping"]["runs_total"], 0)
        self.assertEqual(r["verdicts"]["loudness_target"]["status"], "unknown")


class TestPink(unittest.TestCase):
    def test_pink_loudness_slope_and_bands(self):
        for name in ("pink_14lufs", "pink_23lufs"):
            with self.subTest(name=name):
                exp = manifest()["items"][name]["expected"]
                r = run(name)
                l1 = r["level1"]
                self.assertAlmostEqual(l1["loudness"]["integrated_lufs"], exp["integrated_lufs"], delta=0.1)
                self.assertAlmostEqual(l1["peaks"]["sample_peak_dbfs"], exp["sample_peak_dbfs"], delta=0.01)
                self.assertAlmostEqual(l1["spectral_slope"]["slope_db_per_octave"], -3.0, delta=0.3)
                for band, want in exp["band_energy_rel_total_db"].items():
                    got = l1["bands"]["bands"][band]["energy_db"]
                    self.assertAlmostEqual(got, want, delta=1.0, msg=band)


class TestClipping(unittest.TestCase):
    def test_clipped_sine_runs(self):
        exp = manifest()["items"]["clipped_sine"]["expected"]
        c = run("clipped_sine")["level1"]["clipping"]
        self.assertEqual(c["runs_total"], exp["clipping_runs"])
        self.assertEqual(c["runs_min_len_2"], exp["clipping_runs_min_len_2"])
        self.assertEqual(c["clipped_samples"], exp["clipped_samples"])
        self.assertEqual(c["max_run_length"], exp["max_run_length"])
        r = run("clipped_sine")
        self.assertEqual(r["verdicts"]["clipping"]["status"], "fail")
        self.assertGreaterEqual(r["level1"]["peaks"]["true_peak_dbtp"], -0.001)


class TestStereoBands(unittest.TestCase):
    def test_inverted_lows_in_phase_highs(self):
        st = run("phase_inverted_lows")["level1"]["stereo"]["bands"]
        self.assertLess(st["below_120"]["correlation"], -0.99)
        self.assertLess(st["below_120"]["mono_loss_db"], -40.0)
        self.assertGreater(st["120_500"]["correlation"], 0.99)
        self.assertAlmostEqual(st["120_500"]["mono_loss_db"], 0.0, delta=0.05)
        self.assertGreater(st["above_500"]["correlation"], 0.99)
        self.assertAlmostEqual(st["above_500"]["mono_loss_db"], 0.0, delta=0.05)


class TestTimeDomain(unittest.TestCase):
    def test_dc_offset(self):
        dc = run("dc_offset")["level1"]["dc_offset"]
        self.assertAlmostEqual(dc["left"], 0.05, delta=1e-4)
        self.assertAlmostEqual(dc["right"], 0.05, delta=1e-4)

    def test_leading_trailing_silence(self):
        s = run("silence_padded")["level1"]["silence"]
        self.assertAlmostEqual(s["leading_s"], 1.5, delta=0.01)
        self.assertAlmostEqual(s["trailing_s"], 0.75, delta=0.01)


class TestLevel2(unittest.TestCase):
    def test_all_level2_flagged_heuristic(self):
        r = run("clicks_155", bpm=155)
        for k, v in r["level2"].items():
            with self.subTest(k=k):
                self.assertEqual(v["confidence"], "heuristic")
                self.assertTrue(v["warning"])

    def test_needs_bpm_without_bpm(self):
        r = run("clicks_155")
        self.assertEqual(r["level2"]["energy_drops"]["status"], "needs_bpm")
        self.assertEqual(r["level2"]["kick_onset_deviation"]["status"], "needs_bpm")

    def test_onsets_on_grid(self):
        k = run("clicks_155", bpm=155)["level2"]["kick_onset_deviation"]
        exp = manifest()["items"]["clicks_155"]["expected"]
        self.assertAlmostEqual(k["onsets"], len(exp["onsets_s"]), delta=1)
        self.assertLess(k["deviation_ms"]["max_abs"], 6.0)

    def test_grid_coherence_flags_unreliable_onsets(self):
        good = run("clicks_155", bpm=155)["level2"]["kick_onset_deviation"]
        self.assertEqual(good["reliability"], "ok")
        self.assertGreater(good["phase_coherence"], 0.9)
        # same clicks, but analysed against a wrong tempo -> no grid structure -> flagged low
        bad = run("clicks_155", bpm=131)["level2"]["kick_onset_deviation"]
        self.assertEqual(bad["reliability"], "low")
        self.assertIn("reliability_note", bad)

    def test_shifted_kick_is_seen(self):
        k = run("clicks_155_shift", bpm=155)["level2"]["kick_onset_deviation"]
        self.assertEqual(k["onsets_beyond_10ms"], 1)
        # injected 30 ms shift; resolution is ~5.3 ms
        self.assertAlmostEqual(k["deviation_ms"]["max_abs"], 30.0, delta=8.0)

    def test_dropout_found_and_classified(self):
        exp = manifest()["items"]["dropout_beat7"]["expected"]
        d = run("dropout_beat7", bpm=155)["level2"]["energy_drops"]
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["count"], 1)
        drop = d["drops"][0]
        self.assertEqual(drop["kind"], "dropout")
        self.assertAlmostEqual(drop["start_s"], exp["silent_window_s"][0], delta=0.1)
        self.assertAlmostEqual(drop["duration_beats"], 1.0, delta=0.4)

    def test_sustained_break_is_classified_as_break(self):
        import numpy as np
        import os
        import tempfile
        from .support import corpus

        bpm, period, fs = 120.0, 0.5, 48000
        rng = np.random.default_rng(3)
        n = int(32 * period * fs)
        x = 0.1 * np.stack([corpus.pink_noise(rng, n, fs)] * 2, axis=1)
        x[int(12 * period * fs) : int(20 * period * fs)] *= 0.05  # 8-beat break, -26 dB
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "break.wav")
            corpus.write_wav(p, x, fs, "float32")
            d = analyze(p, bpm=bpm)["level2"]["energy_drops"]
        self.assertEqual(d["count"], 1)
        self.assertEqual(d["drops"][0]["kind"], "break")
        self.assertAlmostEqual(d["drops"][0]["duration_beats"], 8.0, delta=1.0)

    def test_crest_factor_of_sine(self):
        c = run("sine1k_m20_float32")["level2"]["crest_factor"]
        self.assertAlmostEqual(c["per_channel_db"]["left"], 3.0103, delta=0.02)


class TestLevel3(unittest.TestCase):
    FIELDS = ("kick_fundamental", "kick_envelope", "kick_rumble_ratio", "sub_masking", "key")

    def test_level3_never_numeric(self):
        for name in ("clicks_155", "pink_14lufs", "sine1k_m20_pcm16"):
            l3 = run(name, bpm=155)["level3"]
            self.assertEqual(set(l3), set(self.FIELDS))
            for k, v in l3.items():
                self.assertEqual(set(v), {"status", "reason"}, k)
                self.assertEqual(v["status"], "not_determinable")
                self.assertTrue(v["reason"])
                self.assertFalse(any(isinstance(x, (int, float)) for x in v.values()))


class TestIntents(unittest.TestCase):
    def test_intents_echoed_and_not_defects(self):
        r = run("dropout_beat7", bpm=155, intents=["break at bars 33-40", "mono bass on purpose"])
        texts = [i["text"] for i in r["intentional_decisions"]]
        self.assertEqual(texts, ["break at bars 33-40", "mono bass on purpose"])
        self.assertNotIn("defect", json.dumps(r["verdicts"]).lower())


class TestShortAndMono(unittest.TestCase):
    def test_short_file(self):
        r = run("short_300ms")
        no_nonfinite(r)
        self.assertLess(r["file"]["duration_s"], 1.0)
        self.assertTrue(any("shorter than 1 s" in w for w in r["warnings"]))
        self.assertAlmostEqual(r["level1"]["peaks"]["sample_peak_dbfs"], -20.0, delta=0.01)

    def test_json_strict_for_every_corpus_item(self):
        for name in manifest()["items"]:
            with self.subTest(name=name):
                no_nonfinite(run(name, bpm=155))


if __name__ == "__main__":
    unittest.main()
