# audio-diag

Standalone Python CLI and library for stereo-master audio diagnostics with explicit reliability tiers. It operates completely offline, requires no external services or telemetry, and classifies measurements into three clear levels of trustworthiness.

- **Level 1 (reliable):** BS.1770-4 / EBU R128 loudness (integrated, momentary max, short-term max, LRA), sample peak, true peak (4x polyphase Kaiser sinc oversampling), sample-level clipping runs, DC offset, leading and trailing silence, 8-band energy distribution, spectral slope, per-band stereo correlation and mono compatibility.
- **Level 2 (heuristic hints):** Beat-window energy drops (dropouts vs. sustained breaks), kick-onset deviation from a BPM grid, section energy contrast, crest factor (global and FFT band projections). Always reported with `confidence: "heuristic"` and a warning; never treated as an automatic defect.
- **Level 3 (not determinable from a stereo master):** Kick fundamental frequency, kick envelope, kick/rumble ratio, sub masking, musical key. Always reports `{"status": "not_determinable", "reason": ...}`. No estimators exist for these from mixed stereo audio.

## Requirements

- Python >= 3.12
- `numpy`
- `ffmpeg` and `ffprobe` installed and available on `PATH`

System `ffmpeg` is used solely to decode audio into float32 PCM; all loudness, true-peak, spectral, and time-domain calculations are performed directly by audio-diag's internal NumPy implementation.

## Installation

Install via pip from source:

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

## CLI Usage

The package installs the `audio-diag` executable and can also be run via `python3 -m audio_diag`.

```bash
# Analyze an audio file
audio-diag analyze track.wav

# Pass tempo and declare intentional arrangement decisions
audio-diag analyze master.wav --bpm 155 --intent "break at bars 33-40" --intent "mono bass"

# Export machine-readable JSON and human-readable Markdown
audio-diag analyze master.wav --json report.json --md report.md

# Inspect the bundled JSON Schema
audio-diag schema
```

### Options

```
usage: audio_diag analyze [-h] [--bpm BPM] [--json OUT] [--md OUT]
                          [--intent TEXT] [--target-lufs TARGET_LUFS]
                          [--tolerance-lu TOLERANCE_LU] [--tp-max TP_MAX]
                          [--silence-dbfs SILENCE_DBFS] [--drop-db DROP_DB]
                          file
```

- `file`: Path to input audio file (opened strictly read-only).
- `--bpm N`: Tempo in beats per minute; enables grid-aligned Level 2 onset hints.
- `--json OUT`: Write formatted JSON report to `OUT` instead of stdout.
- `--md OUT`: Write Markdown report to `OUT` instead of stderr.
- `--intent TEXT`: Declare an intentional artistic choice (repeatable). Echoed in `intentional_decisions` and never reported as a defect.
- `--target-lufs N`: Integrated loudness target in LUFS (default: `-8.0`).
- `--tolerance-lu N`: Loudness tolerance in LU (default: `2.0`).
- `--tp-max N`: Maximum true peak in dBTP (default: `-0.3`).
- `--silence-dbfs N`: Threshold for digital silence detection (default: `-60.0`).
- `--drop-db N`: Energy drop threshold for dropout/break detection (default: `6.0`).

### Path Safety and Safeguards

- The input file is opened strictly read-only (`O_RDONLY`).
- The CLI refuses any output path that matches the input file (resolving symlinks and hardlinks).
- The CLI refuses to write output inside protected directories:
  - Standard user music directories (`~/Music`, `~/Música`).
  - Directories configured via the `AUDIO_DIAG_PROTECTED_DIRS` environment variable (colon-separated).
- Exit codes: `0` on success, `2` on invalid arguments, decoding failure, or unsafe output paths.

## Python API

```python
from audio_diag import analyze, load_schema, to_markdown, DecodeError

# Analyze an audio file
try:
    report = analyze("master.wav", bpm=158, intents=["break at bars 33-40"])
except DecodeError as err:
    print(f"Decoding failed: {err}")

# Access Level 1 metrics
loudness = report["level1"]["loudness"]
print(f"Integrated LUFS: {loudness['integrated_lufs']}")
print(f"True Peak dBTP:  {report['level1']['peaks']['true_peak_dbtp']}")

# Render Markdown report
md_text = to_markdown(report)

# Load the JSON Schema definition
schema = load_schema()
```

### Direct PCM Measurement

Measure in-memory float32 PCM samples without decoding from disk:

```python
import numpy as np
from audio_diag.loudness import measure

# samples: 2D numpy array of shape (num_samples, channels)
stats = measure(samples, sample_rate=48000)
print(stats["integrated_lufs"], stats["true_peak_dbtp"])
```

### Spectral Profiles and Dispersion

Compare relative spectral balance across tracks using gain-normalized power profiles:

```python
from audio_diag.profiles import profile, compare

p1 = profile(analyze("track1.wav"))
p2 = profile(analyze("track2.wav"))
dispersion = compare([p1, p2])
```

Profiles normalize by total spectral power ($P_{band} / P_{total}$), canceling gain differences. Comparisons produce observed minimum and maximum envelopes per band rather than an artificial average or arbitrary pass/fail gate.

## Schema & Report Invariants

- Schema definition is bundled as `report.schema.json` and accessible via `audio_diag.load_schema()`.
- Standard keys: `tool`, `version`, `file`, `thresholds`, `level1`, `level2`, `level3`, `intentional_decisions`, `warnings`, `verdicts`.
- Finite JSON: values are never `NaN` or `inf`. Missing or invalid numbers are represented as `null` with an accompanying `<metric>_reason` explanation.
- Lossy inputs (e.g. MP3, Opus, AAC) set `file.lossy_source: true`. True peak, crest factor, and absolute loudness are explicitly marked `"not valid for calibration (lossy source)"` because lossy codecs alter peaks and levels dynamically.

## Metric Summary

- **Loudness:** ITU-R BS.1770-4 / EBU R128 biquad K-weighting, 400 ms blocks with 75% overlap, -70 LUFS absolute and -10 LU relative gating. Maxima evaluated on a 10 ms grid. LRA uses 3 s windows, -20 LU relative gate, P10-P95 span.
- **True Peak:** 4x polyphase Kaiser-windowed sinc oversampling filter (32 taps).
- **Clipping:** Identifies consecutive runs of full-scale samples (`|x| >= 1.0 - 1e-6`) per channel.
- **Bands:** Relative energy across 8 sub-bands computed via Hann-windowed periodogram (8192-point FFT, 50% overlap).
- **Mono Compatibility:** Inter-channel correlation and mono energy ratio $10 \log_{10} \frac{E[((L+R)/2)^2]}{E[(L^2+R^2)/2]}$ per frequency band.

## Testing

The test suite runs with Python's standard `unittest`:

```bash
# Run the entire test suite
python3 -m unittest discover

# Run specific test modules
python3 -m unittest tests.test_cli_robustness
python3 -m unittest tests.test_corpus_expected
python3 -m unittest tests.test_ebu_tech3341
```

- `test_corpus_expected`: Verifies measurements against analytically calculated mathematical ground truths (sine LUFS via K-weighting filter transfer function, pink noise LUFS via Parseval's theorem, exact clipping runs, synthesized click trains).
- `test_ebu_tech3341`: Reconstructs EBU Tech 3341 compliance signals and asserts tolerances according to the specification ($\pm 0.1$ LU).
- `test_ffmpeg_crosscheck`: Cross-checks loudness and true-peak results against system `ffmpeg -af ebur128=peak=true`.
- `test_cli_robustness`: Verifies CLI exit codes, error handling, path traversal and overwrite protection, memory limits, and schema output.

Test audio signals are generated deterministically in memory or temporary directories and are not committed to source control.

## License

MIT License. Copyright (c) 2026 Adrián Cancio / contributors. See `LICENSE` for details.
