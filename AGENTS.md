# audio-diag — agent rules

Standalone Python CLI/library that classifies audio measurements into three
reliability levels: level 1 (reliable), level 2 (heuristic hints), level 3
(not determinable from a stereo master). It must work without any MCP.

## Hard rules

- **Audio under `/home/cancio/Música/` is strictly read-only.** Open it
  `O_RDONLY` only. Never write, move, rename, or delete it. Never write
  output next to it; the CLI must refuse an output path that resolves to the
  input file.
- **Never upload or send audio to any external service.**
- **No third-party dependencies.** Python 3.14 stdlib + numpy + system
  `ffmpeg`/`ffprobe` only. No scipy. New packages are installed by the user
  with `dnf`, not `pip`.
- **Do not commit generated WAV files.** The test corpus is generated
  deterministically from code (`tests/corpus.py`) into a temp dir.
- **Do not commit or push unless the user asks.**
- Code, comments, and docs are written in English. Generated phase reports
  go to the Obsidian vault (Spanish notes), not to this repo.
- Level-3 measurements (kick fundamental/envelope, kick/rumble ratio, sub
  masking, key) always return `{"status": "not_determinable", "reason": ...}`
  for a stereo master. Do not add estimators for them.
- Level-2 results are always `confidence: "heuristic"` with a warning and are
  never reported as defects. User `--intent` regions are echoed under
  `intentional_decisions` and never labelled as defects.
- JSON output must never contain NaN/inf: use `null` plus a reason.
- Production loudness/true-peak numbers come from our own numpy
  implementation. ffmpeg is used for decoding, and `ffmpeg -af ebur128` is
  used only as an external cross-check in tests.

## Commands

```
python3 -m unittest discover                      # full test suite
python3 -m unittest tests.test_corpus_expected    # one module
python3 -m audio_diag analyze FILE [--bpm N] [--json OUT] [--md OUT] [--intent TEXT ...]
```

Exit codes: `0` ok, `2` usage or decode error.

## Layout

- `audio_diag/` — package (decode, loudness, spectral, stereo, level 2/3, report, CLI)
- `tests/corpus.py` — deterministic synthetic corpus + analytic expected values
  + EBU Tech 3341 test-signal reconstructions (no EBU download)
- `tests/` — unittest modules
- `README.md` — usage, schema, limitations
