"""Deterministic synthetic corpus for audio-diag tests.

Every item is generated from code with a fixed seed. Nothing here calls into
``audio_diag``: the EXPECTED values are derived analytically (or from the
frequency-domain spectrum of the generated signal), so tests that compare the
analyzer against them check the analyzer, not itself.

Expected-value derivations
--------------------------
* Sine LUFS: ``L = -0.691 + 10*log10(sum_ch G(f) * A^2 / 2)`` where ``G(f)`` is
  the squared magnitude of the K-weighting transfer function evaluated at the
  sine frequency (``k_weight_power_gain``). No time-domain filtering involved.
* Pink noise LUFS: the K-weighted mean square is computed with Parseval's
  theorem from the rfft of the generated signal and ``|H(f)|^2``; the signal
  is then rescaled so the expected integrated loudness hits the target
  (a closed-form gain, so no iteration is required). Gating is ignored in the
  expectation (stationary noise), which the tests cover with a 0.1 LU margin.
* Pink noise band energies: for a 1/f power spectrum limited to
  [20 Hz, Nyquist], band energy is proportional to ``ln(f_hi / f_lo)``.
* Clipping: runs are counted directly from the generated integer samples.
* Click train: onset times are the quantised sample positions of the pulses.
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Dict, Iterable, List, Tuple

import numpy as np

# --------------------------------------------------------------------------
# K-weighting transfer function (BS.1770-4), independent re-implementation.
# --------------------------------------------------------------------------

_SHELF_F0 = 1681.974450955533
_SHELF_G = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_HP_F0 = 38.13547087602444
_HP_Q = 0.5003270373238773


def _k_biquads(fs: float):
    """Return ((b_shelf, a_shelf), (b_hp, a_hp)) for sample rate ``fs``."""
    k = math.tan(math.pi * _SHELF_F0 / fs)
    vh = 10.0 ** (_SHELF_G / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / _SHELF_Q + k * k
    b_s = np.array(
        [
            (vh + vb * k / _SHELF_Q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / _SHELF_Q + k * k) / a0,
        ]
    )
    a_s = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _SHELF_Q + k * k) / a0])
    k = math.tan(math.pi * _HP_F0 / fs)
    d = 1.0 + k / _HP_Q + k * k
    b_h = np.array([1.0, -2.0, 1.0])
    a_h = np.array([1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / _HP_Q + k * k) / d])
    return (b_s, a_s), (b_h, a_h)


def k_weight_power_gain(freqs, fs: float) -> np.ndarray:
    """Squared magnitude |H(f)|^2 of the K-weighting cascade at ``freqs``."""
    freqs = np.asarray(freqs, dtype=np.float64)
    w = 2.0 * np.pi * freqs / fs
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    gain = np.ones_like(freqs)
    for b, a in _k_biquads(fs):
        num = b[0] + b[1] * z1 + b[2] * z2
        den = a[0] + a[1] * z1 + a[2] * z2
        gain = gain * (np.abs(num) ** 2) / (np.abs(den) ** 2)
    return gain


def sine_lufs(freq: float, amp_peak: float, fs: float, channels: int = 2) -> float:
    """Analytic loudness of identical sines on ``channels`` channels."""
    g = float(k_weight_power_gain([freq], fs)[0])
    return -0.691 + 10.0 * math.log10(channels * g * amp_peak**2 / 2.0)


def spectral_lufs(x: np.ndarray, fs: float) -> float:
    """Ungated loudness of a (channels, N) float array via Parseval."""
    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    h2 = k_weight_power_gain(freqs, fs)
    total = 0.0
    for ch in x:
        spec = np.fft.rfft(ch)
        p = np.abs(spec) ** 2
        w = np.full(p.shape, 2.0)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        total += float(np.sum(w * p * h2)) / (n * n)
    return -0.691 + 10.0 * math.log10(total)


def db(x: float) -> float:
    return 20.0 * math.log10(x)


# --------------------------------------------------------------------------
# WAV writer (PCM 16/24/32 and IEEE float32) - stdlib + numpy only.
# --------------------------------------------------------------------------


def write_wav(path: str, data: np.ndarray, fs: int, subtype: str = "pcm16") -> None:
    """Write ``data`` (frames, channels) float in [-1, 1] to a WAV file."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    frames, channels = data.shape
    if subtype == "float32":
        payload = data.astype("<f4").tobytes()
        tag, bits = 3, 32
    else:
        bits = {"pcm16": 16, "pcm24": 24, "pcm32": 32}[subtype]
        scale = float(2 ** (bits - 1))
        ints = np.clip(np.rint(data.astype(np.float64) * scale), -scale, scale - 1).astype(np.int64)
        if bits == 16:
            payload = ints.astype("<i2").tobytes()
        elif bits == 32:
            payload = ints.astype("<i4").tobytes()
        else:
            raw = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
            payload = raw.tobytes()
        tag = 1
    block = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, tag, channels, fs, fs * block, block, bits)
    header += b"data" + struct.pack("<I", len(payload))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)


def quantize(data: np.ndarray, bits: int) -> np.ndarray:
    """Round-trip ``data`` through integer quantisation, as the WAV would."""
    scale = float(2 ** (bits - 1))
    return np.clip(np.rint(data * scale), -scale, scale - 1) / scale


# --------------------------------------------------------------------------
# Signal generators
# --------------------------------------------------------------------------

SR = 48000


def tone(freq: float, amp: float, seconds: float, fs: int = SR) -> np.ndarray:
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


def stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1)


def pink_noise(rng: np.random.Generator, n: int, fs: int, f_lo: float = 20.0) -> np.ndarray:
    """1/f-power noise limited to [f_lo, Nyquist], unit-ish variance."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(freqs)
    mask = freqs >= f_lo
    shape[mask] = 1.0 / np.sqrt(freqs[mask])
    out = np.fft.irfft(spec * shape, n)
    return out / np.sqrt(np.mean(out**2))


def pink_stereo_at_lufs(seed: int, seconds: float, fs: int, target_lufs: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs))
    x = np.stack([pink_noise(rng, n, fs), pink_noise(rng, n, fs)], axis=0) * 0.1
    gain = 10.0 ** ((target_lufs - spectral_lufs(x, fs)) / 20.0)
    return (x * gain).T  # (frames, channels)


def kick_pulse(fs: int, freq: float = 55.0, tau: float = 0.06, dur: float = 0.25, amp: float = 0.8):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.cos(2.0 * np.pi * freq * t) * np.exp(-t / tau)


def click_train(bpm: float, seconds: float, fs: int, first: float = 0.5, shifts: Dict[int, float] | None = None):
    """Kick-like pulses on a BPM grid. Returns (signal, onset_times_s)."""
    shifts = shifts or {}
    period = 60.0 / bpm
    pulse = kick_pulse(fs)
    x = np.zeros(int(round(seconds * fs)))
    onsets: List[float] = []
    k = 0
    while True:
        t = first + k * period + shifts.get(k, 0.0)
        pos = int(round(t * fs))
        if pos + len(pulse) >= len(x):
            break
        x[pos : pos + len(pulse)] += pulse
        onsets.append(pos / fs)
        k += 1
    return x, onsets


def count_runs(x: np.ndarray, thresh: float) -> Tuple[int, int, int, int]:
    """(runs_total, runs_min_len_2, clipped_samples, max_run) over channels."""
    runs = runs2 = clipped = max_run = 0
    for ch in np.atleast_2d(x.T):
        hit = np.abs(ch) >= thresh
        clipped += int(hit.sum())
        padded = np.concatenate([[0], hit.astype(np.int8), [0]])
        d = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        lens = ends - starts
        runs += len(lens)
        runs2 += int((lens >= 2).sum())
        if len(lens):
            max_run = max(max_run, int(lens.max()))
    return runs, runs2, clipped, max_run


# --------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------

BANDS = [
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 120.0),
    ("low_mid", 120.0, 500.0),
    ("mid", 500.0, 4000.0),
    ("high", 4000.0, 10000.0),
    ("air", 10000.0, None),  # up to Nyquist
]


def pink_band_expectations(fs: int) -> Dict[str, float]:
    """Band energy relative to total (dB) for 1/f noise on [20 Hz, fs/2]."""
    nyq = fs / 2.0
    total = math.log(nyq / 20.0)
    out = {}
    for name, lo, hi in BANDS:
        hi = nyq if hi is None else hi
        out[name] = 10.0 * math.log10(math.log(hi / lo) / total)
    return out


def build_corpus(outdir: str, seed: int = 20261028) -> dict:
    """Write every WAV into ``outdir`` and return (and save) the manifest."""
    os.makedirs(outdir, exist_ok=True)
    items: Dict[str, dict] = {}

    def add(name, data, fs, subtype, expected, **extra):
        fname = f"{name}.wav"
        write_wav(os.path.join(outdir, fname), data, fs, subtype)
        frames = data.shape[0]
        items[name] = {
            "file": fname,
            "sample_rate": fs,
            "channels": 1 if data.ndim == 1 else data.shape[1],
            "subtype": subtype,
            "duration_s": frames / fs,
            "expected": expected,
            **extra,
        }

    # 1. digital silence
    add("silence", np.zeros((SR * 5, 2)), SR, "pcm16", {"silent": True, "sample_peak_dbfs": None})

    # 2. 1 kHz sine at -20 dBFS (several containers, same analytic expectation)
    amp = 10.0 ** (-20.0 / 20.0)
    for sub in ("pcm16", "pcm24", "float32"):
        sig = stereo(tone(1000.0, amp, 5.0))
        stored = sig if sub == "float32" else quantize(sig, {"pcm16": 16, "pcm24": 24}[sub])
        add(
            f"sine1k_m20_{sub}",
            sig,
            SR,
            sub,
            {
                "sample_peak_dbfs": db(float(np.max(np.abs(stored)))),
                "integrated_lufs": sine_lufs(1000.0, amp, SR),
                "momentary_max_lufs": sine_lufs(1000.0, amp, SR),
                "short_term_max_lufs": sine_lufs(1000.0, amp, SR),
                "lra_lu": 0.0,
                "true_peak_dbtp": -20.0,
                "dc_offset": 0.0,
                "clipping_runs": 0,
            },
        )
    # 44.1 kHz and mono variants
    sig = stereo(tone(1000.0, amp, 5.0, 44100))
    add("sine1k_m20_44k", sig, 44100, "pcm24",
        {"integrated_lufs": sine_lufs(1000.0, amp, 44100), "true_peak_dbtp": -20.0, "sample_peak_dbfs": -20.0})
    mono = tone(1000.0, amp, 5.0)
    add("sine1k_m20_mono", mono, SR, "pcm24",
        {"integrated_lufs": sine_lufs(1000.0, amp, SR, channels=1), "true_peak_dbtp": -20.0, "sample_peak_dbfs": -20.0})

    # 3. pink noise at a known integrated LUFS
    for target in (-14.0, -23.0):
        pink = pink_stereo_at_lufs(seed, 30.0, SR, target)
        peak = float(np.max(np.abs(pink)))
        assert peak < 1.0, "pink corpus item would clip; lower the target"
        add(
            f"pink_{int(-target)}lufs",
            pink,
            SR,
            "float32",
            {
                "integrated_lufs": spectral_lufs(pink.T, SR),
                "sample_peak_dbfs": db(peak),
                "spectral_slope_db_oct": -3.0,
                "band_energy_rel_total_db": pink_band_expectations(SR),
                "clipping_runs": 0,
            },
        )

    # 4. forced clipping: 2x-overdriven 440 Hz sine, hard clipped at full scale (float32)
    sine = tone(441.0, 2.0, 3.0)
    clipped = np.clip(sine, -1.0, 1.0)
    sig = stereo(clipped)
    runs, runs2, nclip, maxrun = count_runs(sig, 1.0)
    add(
        "clipped_sine",
        sig,
        SR,
        "float32",
        {
            "sample_peak_dbfs": 0.0,
            "clipping_runs": runs,
            "clipping_runs_min_len_2": runs2,
            "clipped_samples": nclip,
            "max_run_length": maxrun,
            "channels_clipped": 2,
        },
    )

    # 5. stereo: inverted phase below 120 Hz, in-phase above
    n = SR * 6
    t = np.arange(n) / SR
    lows = 0.2 * np.sin(2 * np.pi * 50.0 * t) + 0.2 * np.sin(2 * np.pi * 90.0 * t)
    mids = 0.2 * np.sin(2 * np.pi * 250.0 * t)
    highs = 0.1 * np.sin(2 * np.pi * 1000.0 * t) + 0.1 * np.sin(2 * np.pi * 3000.0 * t) + 0.05 * np.sin(2 * np.pi * 7000.0 * t)
    sig = np.stack([lows + mids + highs, -lows + mids + highs], axis=1)
    add(
        "phase_inverted_lows",
        sig,
        SR,
        "float32",
        {
            "band_correlation": {"below_120": -1.0, "120_500": 1.0, "above_500": 1.0},
            "band_mono_loss_db": {"120_500": 0.0, "above_500": 0.0},
            "below_120_mono_loss_db": "<= -40",
        },
    )

    # 6. click train at 155 BPM
    bpm = 155.0
    x, onsets = click_train(bpm, 12.0, SR)
    add("clicks_155", stereo(x), SR, "float32", {"bpm": bpm, "onsets_s": onsets, "grid_period_s": 60.0 / bpm})
    shifted, onsets2 = click_train(bpm, 12.0, SR, shifts={8: 0.030})
    add("clicks_155_shift", stereo(shifted), SR, "float32",
        {"bpm": bpm, "onsets_s": onsets2, "shifted_beat_index": 8, "shift_ms": 30.0})

    # 7. beat-window dropout: pink noise with beat 7 (grid-aligned) silenced
    period = 60.0 / bpm
    rng = np.random.default_rng(seed + 1)
    n = int(round(16 * period * SR))
    noise = 0.1 * np.stack([pink_noise(rng, n, SR), pink_noise(rng, n, SR)], axis=1)
    s0, s1 = int(round(7 * period * SR)), int(round(8 * period * SR))
    noise[s0:s1] = 0.0
    add("dropout_beat7", noise, SR, "float32",
        {"bpm": bpm, "silent_window_s": [s0 / SR, s1 / SR], "beat_index": 7})

    # 8. very short file (< 1 s)
    add("short_300ms", stereo(tone(1000.0, amp, 0.3)), SR, "pcm16",
        {"integrated_lufs": sine_lufs(1000.0, amp, SR)})

    # 9. DC offset
    dc = stereo(tone(1000.0, 0.1, 3.0) + 0.05)
    add("dc_offset", dc, SR, "float32", {"dc_offset": 0.05})

    # 10. leading/trailing silence around a tone
    sig = np.concatenate([np.zeros(int(1.5 * SR)), tone(1000.0, 0.3, 3.0), np.zeros(int(0.75 * SR))])
    add("silence_padded", stereo(sig), SR, "float32", {"leading_silence_s": 1.5, "trailing_silence_s": 0.75})

    manifest = {"seed": seed, "items": items}
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# --------------------------------------------------------------------------
# EBU Tech 3341 "minimum requirements" cases, rebuilt from the published
# test-signal descriptions (no EBU download). Each case is (signal, expected).
# Tolerance in the standard is +-0.1 LU.
# --------------------------------------------------------------------------


def _seg(level_dbfs: float, seconds: float, fs: int = SR) -> np.ndarray:
    return tone(1000.0, 10.0 ** (level_dbfs / 20.0), seconds, fs)


def _silence(seconds: float, fs: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * fs)))


def ebu_cases(fs: int = SR) -> Dict[str, dict]:
    """Return {name: {"signal": (frames, 2), "expected": {...}}} for cases 1-5, 9, 10, 12, 13."""
    cases: Dict[str, dict] = {}

    def put(name, mono, **expected):
        cases[name] = {"signal": stereo(mono), "expected": expected}

    put("case01", _seg(-23.0, 20, fs), integrated=-23.0, momentary_max=-23.0, short_term_max=-23.0)
    put("case02", _seg(-33.0, 20, fs), integrated=-33.0, momentary_max=-33.0, short_term_max=-33.0)
    put("case03", np.concatenate([_seg(-36, 10, fs), _seg(-23, 60, fs), _seg(-36, 10, fs)]), integrated=-23.0)
    put(
        "case04",
        np.concatenate([_seg(-72, 10, fs), _seg(-36, 10, fs), _seg(-23, 60, fs), _seg(-36, 10, fs), _seg(-72, 10, fs)]),
        integrated=-23.0,
    )
    put("case05", np.concatenate([_seg(-26, 20, fs), _seg(-20, 20.1, fs), _seg(-26, 20, fs)]), integrated=-23.0)
    # case 9: (1.34 s at -20 dBFS; 1.66 s at -30 dBFS) x 5 -> S constant -23.0 after 3 s
    put("case09", np.concatenate([np.concatenate([_seg(-20, 1.34, fs), _seg(-30, 1.66, fs)]) for _ in range(5)]),
        short_term_max=-23.0)
    # case 12: (0.18 s at -20; 0.22 s at -30) x 25 -> M constant -23.0 after 1 s
    put("case12", np.concatenate([np.concatenate([_seg(-20, 0.18, fs), _seg(-30, 0.22, fs)]) for _ in range(25)]),
        momentary_max=-23.0)
    # case 10 (file-based): i*0.15 s silence, 3 s at -23 dBFS, 1 s silence -> max S = -23.0
    for i in (0, 1, 5, 13, 19):
        put(f"case10_i{i}", np.concatenate([_silence(i * 0.15, fs), _seg(-23, 3, fs), _silence(1, fs)]),
            short_term_max=-23.0)
    # case 13 (file-based): i*20 ms silence, 400 ms at -23 dBFS, 1 s silence -> max M = -23.0
    for i in (0, 1, 7, 19):
        put(f"case13_i{i}", np.concatenate([_silence(i * 0.02, fs), _seg(-23, 0.4, fs), _silence(1, fs)]),
            momentary_max=-23.0)
    return cases


if __name__ == "__main__":  # pragma: no cover - manual use
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "corpus_out"
    print(json.dumps(build_corpus(target)["items"].keys().__repr__()))
