# audio-diag

Standalone Python CLI/library that measures an audio file and states how far
each measurement can be trusted. It needs no MCP server and sends nothing
anywhere.

- **Level 1 (reliable):** loudness (BS.1770-4 / EBU R128), true peak, sample
  peak, clipping, DC offset, leading/trailing silence, per-band energy,
  spectral slope, stereo correlation and mono compatibility per band.
- **Level 2 (heuristic hints):** beat-window energy drops, kick-onset deviation
  from a BPM grid, section energy contrast, crest factor. Always
  `confidence: "heuristic"` with a warning. Never an automatic defect.
- **Level 3 (not determinable from a stereo master):** kick fundamental,
  kick envelope, kick/rumble ratio, sub masking, key. Always
  `{"status": "not_determinable", "reason": ...}`. No estimators exist.

## Requirements

Python 3.14 (stdlib), numpy, and `ffmpeg`/`ffprobe` on `PATH`. ffmpeg is only
used to decode to float32 PCM; the loudness and true-peak numbers come from this
package's own numpy implementation.

## Usage

```
python3 -m audio_diag analyze FILE [--bpm N] [--json OUT] [--md OUT] [--intent TEXT ...]
                              [--target-lufs -8] [--tolerance-lu 2] [--tp-max -0.3]
                              [--silence-dbfs -60] [--drop-db 6]
```

JSON goes to stdout unless `--json` is given; markdown goes to stderr unless
`--md` is given. Exit codes: `0` ok, `2` usage, decode or output-path error.
The input is opened read-only and the CLI refuses an output path that is the
input (also via symlink or hardlink) or that lies under `~/Música`.

Library use:

```python
from audio_diag import analyze, DecodeError
result = analyze("master.wav", bpm=158, intents=["break at bars 33-40"])
```

`audio_diag.loudness.measure(samples, sample_rate)` exposes the loudness meter
directly on in-memory PCM.

## Output

Top-level keys: `tool, version, file, thresholds, level1, level2, level3,
intentional_decisions, warnings, verdicts`. JSON never contains NaN/inf; an
unavailable value is `null` with a sibling `<key>_reason`.

Defaults: club target -8 LUFS +-2 integrated, true peak <= -0.3 dBTP.
`verdicts` compares level-1 numbers to these thresholds only.

`--intent TEXT` (repeatable) is echoed under `intentional_decisions` and is
never reported as a defect.

## Metric definitions (short)

- Loudness: K-weighting biquads, 400 ms blocks with 75 % overlap, -70 LUFS
  absolute and -10 LU relative gate. Momentary/short-term maxima use a 10 ms
  grid. LRA: 3 s windows, -20 LU gate, P10-P95.
- True peak: 4x polyphase oversampling (32-tap Kaiser sinc).
- Clipping: a run is consecutive samples with `|x| >= 1 - 1e-6` in one channel.
- Band energy: relative to the total (dB), from an averaged Hann periodogram
  (8192 samples, 50 % overlap).
- Mono loss (per band): `10*log10(E[((L+R)/2)^2] / E[(L^2+R^2)/2])`.
  0 dB = identical channels, -3 dB = uncorrelated, very negative = out of phase.
- Level 2 `energy_drops`: one-beat sliding window; short regions (<= 2 beats)
  are `dropout`, sustained (>= 4 beats) are `break`.

## Lossy sources

Lossy input (mp3, opus, vorbis, aac, ...) sets `file.lossy_source: true`. True
peak, crest factor and absolute loudness are then labelled
"not valid for calibration (lossy source)" in JSON and markdown: decoding
changes peaks and level by amounts the file does not tell you.

## Limits

- Level 2 is heuristic. `kick_onset_deviation` uses low-band (<200 Hz) energy
  flux with ~5 ms resolution. On a full mix, bass notes and rumble also
  trigger it. When detected onsets do not cluster on the BPM grid
  (`phase_coherence < 0.6`) it reports `reliability: "low"`; do not trust its
  deviation numbers then. Tempo is never estimated; pass `--bpm`.
- Energy-drop durations and start times are approximate (1/8-beat steps).
- More than 2 channels are downmixed to stereo by ffmpeg (with a warning).
- Files shorter than 0.4 s / 3 s have no momentary / short-term values (null
  plus reason).
- Very short or abruptly started tones can differ from ffmpeg's true peak by
  up to ~0.4 dB (ffmpeg itself overshoots the analytic value there). On
  realistic material the difference is ~0.01-0.05 dB.
- LRA on tiny gated sets is sensitive to percentile rank handling (about
  0.25 LU on a 2 s ramp), and ffmpeg counts its first partially filled 3 s
  window for ~3 s files. Both are noted in the tests.

## Render identity and schema

`file.sha256` is the SHA-256 of the entire source file, read in bounded chunks;
markdown includes the same hash. Keep the source immutable during analysis.
`report.schema.json` is the machine-readable JSON Schema (2020-12) for the
report envelope and reliability invariants; detailed metric definitions above
remain authoritative. No third-party schema validator is required at runtime.

Global crest uses maximum channel sample peak divided by pooled channel RMS.
`level2.crest_factor.per_band_db` uses peak/RMS of brick-wall FFT band projections
of Hann-windowed 8192-sample frames (50% overlap, pooled channels). It is **not**
a continuous bandpass crest measurement: Hann-windowing raises the crest of a
stationary sine from 3.01 to about 7.27 dB. Boundary leakage affects weak bands.
These values are heuristic comparisons only; silence gives null.

This is not a claim of full EBU conformance. No EBU dataset was downloaded.
Intentional breaks never automatically fail: only level-1 target comparisons
produce verdicts, not arrangement heuristics.

## Offline reference profiles and controlled checks

`audio_diag.profiles.profile(report)` extracts each file's normalized relative
band powers. `compare(profiles)` preserves every profile and returns independent
per-band min/max dispersion (including missing-value counts), never an average
sound or an automatic acceptance gate. Normalization precedes comparison:
`P_band / P_total = (g*g*P_band) / (g*g*P_total)` for nonzero common gain.
This is unit-total-spectral-power normalization, not LUFS normalization. The
property is tested across four gains. Silence remains unknown.

Run the bounded local evidence job from this repository:

```
python3 -m scripts.phase4_offline
```

This job expects the seven specifically mapped local Opus references, three
project WAV exports and the existing 158 BPM master. It hashes sources before
and after measurement, writes individual full reports and comparison JSON under
ignored `data/phase4/`, and generates a 24-second excerpt plus independent ffmpeg
manipulations only under `~/.hermes/cache/scratch/opencode/phase4-bde-v2/`.
It refuses to overwrite excerpt files (`ffmpeg -n`); use a new scratch directory
for a fresh generation. Exact generation argv and directional assertions are
recorded in `controlled-checks.json`. No network, live OMNi, or audio-source writes
are involved. Opus absolute loudness/peak/crest are not targets; project WAVs
are not finished-master targets either. The expected filter directions are not
claims of calibrated numerical truth or full musical utility.

Generate the measured Spanish audit and complete example with
`python3 -m scripts.phase4_vault_report` after the evidence job passes.
The low-phase control uses an exact whole-excerpt FFT polarity flip below
120 Hz on R only (periodic-boundary/ringing limitations apply). The original
failed causal-filter control is preserved in `controlled-checks-v1-failed.json`.
Mute assertions check zero interior energy, unchanged exterior and a localized
heuristic, not just any changed result dictionary.

The Spanish vault artifacts under `Personal/ProdMusic/_Plan_Maestro/_Historial_Auditorias/`, including `INFORME_PRODUCCION.md`,
`Fase4_Escucha_Referencias.md`, and `Fase4_BDE_Informe.md` hold the production
report template, seven pending listening sheets, and the measured audit.
Copy the template for each new SHA-bound render; keep source/device readback,
measurements, hypotheses, intentional decisions and human listening separate.

## Validation

`python3 -m unittest discover` runs:

| Suite | What it proves |
|---|---|
| `test_corpus_expected` | analyzer vs **analytically known** values of a seeded synthetic corpus (sine LUFS from the K-weighting transfer function, pink-noise LUFS by Parseval, clipping run counts, stereo phase, click-train onsets, dropout) |
| `test_ebu_tech3341` | EBU Tech 3341 cases 1-5, 9, 10, 12, 13 rebuilt from the published descriptions, expected values from the standard (+-0.1 LU). Cases 6, 7, 8, 11, 14 are not reproduced. The EBU zip is not used |
| `test_ffmpeg_crosscheck` | our numpy meter vs `ffmpeg -af ebur128=peak=true` (0.1 LU, 0.2 dB TP). **This only shows that our implementation and ffmpeg's agree** (integration, gating, filters); it is not independent proof of correctness |
| `test_cli_robustness` | exit codes, corrupt/empty/missing input, lossy labelling, output safety, 10-minute memory bound |

The corpus WAVs are generated at test time into a temporary directory and are
not committed.
