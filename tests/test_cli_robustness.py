"""CLI behaviour, error paths, lossy sources, output safety and memory bounds."""

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest

from .support import corpus, item_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path, mode="r"):
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
        return fh.read()


def cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "audio_diag", *args], cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class TestCli(unittest.TestCase):
    def test_json_stdout_markdown_stderr(self):
        p = cli("analyze", item_path("sine1k_m20_pcm24"))
        self.assertEqual(p.returncode, 0, p.stderr)
        d = json.loads(p.stdout)
        self.assertEqual(d["tool"], "audio-diag")
        self.assertIn("#", p.stderr)

    def test_output_files_and_intent(self):
        with tempfile.TemporaryDirectory() as td:
            j, m = os.path.join(td, "o.json"), os.path.join(td, "o.md")
            p = cli("analyze", item_path("clicks_155"), "--bpm", "155", "--json", j, "--md", m,
                    "--intent", "break at bars 33-40", "--intent", "mono bass")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(j, encoding="utf-8") as fh:
                d = json.load(fh, parse_constant=lambda c: self.fail(f"non-finite {c}"))
            self.assertEqual([i["text"] for i in d["intentional_decisions"]], ["break at bars 33-40", "mono bass"])
            md = read(m)
            self.assertIn("break at bars 33-40", md)
            self.assertIn("heuristic", md.lower())
            self.assertIn("not determinable", md.lower().replace("not_determinable", "not determinable"))

    def test_usage_error_exit_2(self):
        self.assertEqual(cli().returncode, 2)
        self.assertEqual(cli("analyze").returncode, 2)
        self.assertEqual(cli("analyze", item_path("silence"), "--bpm", "5").returncode, 2)

    def test_missing_file_exit_2(self):
        p = cli("analyze", "/nonexistent/file.wav")
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.strip())

    def test_corrupt_file_exit_2_with_message(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.wav")
            with open(bad, "wb") as fh:
                fh.write(b"RIFF\x00\x00\x00\x00WAVEjunk" + os.urandom(2000))
            p = cli("analyze", bad)
            self.assertEqual(p.returncode, 2)
            self.assertNotIn("Traceback", p.stderr)
            self.assertTrue(p.stderr.strip())
            empty = os.path.join(td, "empty.wav")
            open(empty, "wb").close()
            p = cli("analyze", empty)
            self.assertEqual(p.returncode, 2)
            self.assertNotIn("Traceback", p.stderr)

    def test_silent_and_short_exit_0(self):
        for name in ("silence", "short_300ms", "sine1k_m20_mono"):
            p = cli("analyze", item_path(name))
            self.assertEqual(p.returncode, 0, (name, p.stderr))
            json.loads(p.stdout, parse_constant=lambda c: self.fail(f"non-finite {c}"))


class TestOutputSafety(unittest.TestCase):
    def test_refuses_to_overwrite_input_and_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.wav")
            shutil.copy(item_path("sine1k_m20_pcm16"), src)
            before = read(src, "rb")
            link = os.path.join(td, "link.wav")
            os.symlink(src, link)
            hard = os.path.join(td, "hard.wav")
            os.link(src, hard)
            for out in (src, link, hard):
                with self.subTest(out=os.path.basename(out)):
                    p = cli("analyze", src, "--json", out)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(read(src, "rb"), before)

    def test_refuses_output_under_music_dir(self):
        from audio_diag.safety import get_protected_dirs
        protected_list = get_protected_dirs()
        self.assertTrue(len(protected_list) > 0)
        target = os.path.join(protected_list[0], "should_not_exist.json")
        p = cli("analyze", item_path("silence"), "--json", target)
        self.assertEqual(p.returncode, 2)
        self.assertFalse(os.path.exists(target))

    def test_refuses_output_under_env_protected_dir(self):
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "forbidden.json")
            env = dict(os.environ, AUDIO_DIAG_PROTECTED_DIRS=td)
            p = subprocess.run([sys.executable, "-m", "audio_diag", "analyze", item_path("silence"),
                                "--json", target], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(os.path.exists(target))

    def test_schema_cli_and_load(self):
        p = cli("schema")
        self.assertEqual(p.returncode, 0, p.stderr)
        d = json.loads(p.stdout)
        self.assertEqual(d.get("title"), "audio-diag v1 report")
        from audio_diag import load_schema
        self.assertEqual(load_schema().get("title"), "audio-diag v1 report")

    def test_input_unchanged_after_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.wav")
            shutil.copy(item_path("pink_14lufs"), src)
            os.chmod(src, 0o444)
            before = (read(src, "rb"), os.stat(src).st_mtime_ns)
            self.assertEqual(cli("analyze", src).returncode, 0)
            self.assertEqual((read(src, "rb"), os.stat(src).st_mtime_ns), before)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg missing")
class TestLossy(unittest.TestCase):
    def test_lossy_sources_flagged_and_labelled(self):
        for codec, ext in (("libmp3lame", "mp3"), ("libopus", "opus")):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, f"x.{ext}")
                subprocess.run(["ffmpeg", "-v", "error", "-i", item_path("pink_14lufs"), "-c:a", codec,
                                "-b:a", "128k", out], check=True)
                p = cli("analyze", out)
                self.assertEqual(p.returncode, 0, p.stderr)
                d = json.loads(p.stdout)
                self.assertTrue(d["file"]["lossy_source"])
                self.assertIn("not valid for calibration", d["level1"]["peaks"]["true_peak_validity"])
                self.assertIn("not valid for calibration", d["level1"]["loudness"]["validity"])
                self.assertIn("not valid for calibration", d["level2"]["crest_factor"]["validity"])
                p = cli("analyze", out, "--md", os.path.join(td, "r.md"))
                self.assertIn("not valid for calibration", read(os.path.join(td, "r.md")))

    def test_lossless_is_not_flagged(self):
        d = json.loads(cli("analyze", item_path("sine1k_m20_pcm16")).stdout)
        self.assertFalse(d["file"]["lossy_source"])
        self.assertNotIn("validity", d["level1"]["loudness"])


class TestMemoryBound(unittest.TestCase):
    def test_ten_minute_stereo_48k_is_bounded(self):
        """Analyse a 10 min stereo 48 kHz file in a child process and check its peak RSS."""
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "long.wav")
            rng = np.random.default_rng(1)
            # 10 min float32 stereo = 230 MB on disk; written in 30 s pieces to keep the test bounded.
            block = 0.05 * np.stack([corpus.pink_noise(rng, 48000 * 30, 48000)] * 2, axis=1).astype("<f4")
            n_blocks = 20
            total = block.nbytes * n_blocks
            with open(src, "wb") as fh:
                import struct

                fh.write(b"RIFF" + struct.pack("<I", 36 + total) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 2, 48000, 48000 * 8, 8, 32))
                fh.write(b"data" + struct.pack("<I", total))
                for _ in range(n_blocks):
                    fh.write(block.tobytes())
            code = ("import resource,sys,audio_diag;"
                    "r=audio_diag.analyze(sys.argv[1]);"
                    "print(r['file']['duration_s'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
            p = subprocess.run([sys.executable, "-c", code, src], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            dur, rss_kb = p.stdout.split()
            self.assertAlmostEqual(float(dur), 600.0, delta=0.1)
            file_mb = os.path.getsize(src) / 1e6
            rss_mb = int(rss_kb) / 1024
            self.assertLess(rss_mb, 400, f"peak RSS {rss_mb:.0f} MB for a {file_mb:.0f} MB file")


if __name__ == "__main__":
    unittest.main()
