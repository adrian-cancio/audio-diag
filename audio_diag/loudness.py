"""Independent numpy implementation of ITU-R BS.1770-4 / EBU R128 measurements.

No ffmpeg measurement filter is used here. ffmpeg only decodes PCM.

Definitions
-----------
* K-weighting: two biquads (high shelf + RLB high-pass) at the sample rate. The
  cascade is realised by FFT overlap-save convolution with its (truncated,
  0.25 s) impulse response, which is exact to ~1e-20 for supported rates.
* Momentary: 400 ms blocks, 75 % overlap (100 ms hop). Short-term: 3 s windows,
  100 ms hop. Channel weights are 1.0 (mono and stereo only).
* Integrated: absolute gate -70 LUFS, then relative gate -10 LU below the mean
  of the blocks that passed the absolute gate.
* LRA: short-term blocks, absolute gate -70 LUFS, relative gate -20 LU, then
  95th minus 10th percentile (linear interpolation) of the gated values.
* True peak: 4x oversampling with a 4-branch, 32-tap Kaiser(beta 9, cutoff
  0.97) windowed-sinc polyphase interpolator (the libswresample design that
  ffmpeg's ebur128 true-peak mode uses; the 49-tap Hann sinc of the BS.1770
  informative annex was measured to deviate up to ~0.3 dB from it on
  full-scale square-ish signals). Also combined with the raw sample peak.
"""
import itertools
import math

import numpy as np

ABS_GATE = -70.0
OFFSET = -0.691
SILENCE_ENERGY = 1e-12  # ~ -120 LUFS: treated as digital silence


def k_weighting_coefficients(sample_rate):
    """Return ((shelf_b, shelf_a), (highpass_b, highpass_a)); a[0] == 1.

    The high-pass numerator is [1, -2, 1] (not normalised); denominator
    is normalised.
    """
    f, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f / sample_rate)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = (np.array([vh + vb * k / q + k * k, 2 * (k * k - vh), vh - vb * k / q + k * k]) / a0,
             np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]))
    q2 = 0.5003270373238773
    k = math.tan(math.pi * 38.13547087602444 / sample_rate)
    a0 = 1 + k / q2 + k * k
    hp = (np.array([1.0, -2.0, 1.0]),
          np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q2 + k * k) / a0]))
    return shelf, hp


def _impulse(b, a, n):
    """Impulse response of a biquad (direct form I recurrence, n samples)."""
    out = np.zeros(n)
    y1 = y2 = 0.0
    for i in range(n):
        y = (b[i] if i < 3 else 0.0) - a[1] * y1 - a[2] * y2
        out[i] = y
        y2, y1 = y1, y
    return out


def k_weighting_impulse(sample_rate, seconds=0.25):
    """Impulse response of the full K-weighting cascade."""
    n = max(512, int(sample_rate * seconds))
    shelf, hp = k_weighting_coefficients(sample_rate)
    nfft = 1 << (2 * n).bit_length()
    h = np.fft.irfft(np.fft.rfft(_impulse(*shelf, n), nfft) * np.fft.rfft(_impulse(*hp, n), nfft), nfft)
    return h[:n]


class _FIR:
    """Streaming overlap-save FIR over (frames, channels) arrays."""

    def __init__(self, h, channels):
        self.h = np.asarray(h, dtype=float)
        self.tail = np.zeros((len(self.h) - 1, channels))
        self._cache = {}

    def process(self, x):
        z = np.concatenate((self.tail, x))
        n = 1 << (len(z) + len(self.h)).bit_length()
        if n not in self._cache:
            self._cache = {n: np.fft.rfft(self.h, n)[:, None]}
        y = np.fft.irfft(np.fft.rfft(z, n, axis=0) * self._cache[n], n, axis=0)
        start = len(self.h) - 1
        if start:
            self.tail = z[-start:].copy()
        return y[start:start + len(x)]


def k_weight(x, sample_rate):
    """Apply K-weighting to a whole (frames,) or (frames, channels) array."""
    x = np.asarray(x, dtype=float)
    one = x.ndim == 1
    x = x[:, None] if one else x
    out = _FIR(k_weighting_impulse(sample_rate), x.shape[1]).process(x)
    return out[:, 0] if one else out


def _polyphase_taps(oversample=4, taps=32, cutoff=0.97, beta=9.0):
    """Four polyphase branches (fractional delays 0, 1/4, 2/4, 3/4 sample) of a
    Kaiser-windowed sinc, the same design family as libswresample's default
    resampler (32 taps, cutoff 0.97, beta 9). Each branch is DC-normalised.
    Returns a list of (taps, group delay in samples)."""
    n = int(math.ceil(taps / cutoff))
    c = (n - 1) // 2
    i = np.arange(n)
    branches = []
    for ph in range(oversample):
        x = math.pi * ((i - c) - ph / oversample) * cutoff
        safe = np.where(x == 0, 1.0, x)
        y = np.where(x == 0, 1.0, np.sin(safe) / safe)
        w = 2 * x / (cutoff * n * math.pi)
        y = y * np.i0(beta * np.sqrt(np.maximum(1 - w * w, 0)))
        branches.append((y / y.sum(), c))
    return branches


def lufs_of(energy):
    return OFFSET + 10.0 * np.log10(np.maximum(energy, 1e-300))


class LoudnessMeter:
    """Bounded-memory streaming meter. update() takes float arrays (frames, channels)."""

    def __init__(self, sample_rate, channels=2):
        self.sample_rate, self.channels = int(sample_rate), channels
        self._k = _FIR(k_weighting_impulse(self.sample_rate), channels)
        # Energies are stored per 10 ms sub-block so momentary/short-term maxima are evaluated on a
        # 10 ms grid (Tech 3341 cases 10/13 shift bursts in 20 ms/150 ms steps). Gating blocks and
        # LRA windows still advance in 100 ms steps (every 10th window), as in BS.1770-4.
        self._hop = max(1, round(self.sample_rate * 0.01))
        self._pending = np.empty(0)
        self._energies = []
        self._branches = [(_FIR(t, channels), d) for t, d in _polyphase_taps()]
        self._peak = np.zeros(channels)
        self._sample_peak = np.zeros(channels)
        self.frames = 0

    def update(self, samples):
        x = np.asarray(samples, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if not len(x):
            return
        self.frames += len(x)
        self._sample_peak = np.maximum(self._sample_peak, np.abs(x).max(axis=0))
        self._peak = np.maximum(self._peak, np.abs(x).max(axis=0))
        for fir, _ in self._branches:
            self._peak = np.maximum(self._peak, np.abs(fir.process(x)).max(axis=0))
        y = self._k.process(x)
        e = np.concatenate((self._pending, np.einsum('ij,ij->i', y, y)))
        n = len(e) // self._hop
        if n:
            self._energies.extend(e[:n * self._hop].reshape(n, self._hop).mean(axis=1).tolist())
        self._pending = e[n * self._hop:]

    def finish(self):
        """Return the measurement dict (all values finite or None with *_reason)."""
        if self.frames:
            flush = np.zeros((40, self.channels))
            for fir, _ in self._branches:
                self._peak = np.maximum(self._peak, np.abs(fir.process(flush)).max(axis=0))
        e = np.asarray(self._energies)

        def windows(n):
            if len(e) < n:
                return np.empty(0)
            c = np.concatenate(([0.0], np.cumsum(e)))
            return (c[n:] - c[:-n]) / n

        def gate(v, rel):
            v = v[lufs_of(v) > ABS_GATE]
            if not len(v):
                return v
            return v[lufs_of(v) > lufs_of(v.mean()) + rel]

        m, s = windows(40), windows(300)  # 400 ms / 3 s windows on the 10 ms grid
        g, sg = gate(m[::10], -10.0), gate(s[::10], -20.0)  # gating blocks: 100 ms steps
        res = {
            'integrated_lufs': float(lufs_of(g.mean())) if len(g) else None,
            'momentary_max_lufs': float(lufs_of(m.max())) if len(m) and m.max() > SILENCE_ENERGY else None,
            'short_term_max_lufs': float(lufs_of(s.max())) if len(s) and s.max() > SILENCE_ENERGY else None,
            'lra_lu': None,
        }
        if len(sg):
            lv = lufs_of(sg)
            res['lra_lu'] = float(np.percentile(lv, 95) - np.percentile(lv, 10))
        for k in ('integrated_lufs', 'momentary_max_lufs', 'short_term_max_lufs', 'lra_lu'):
            if res[k] is None:
                need = 3.0 if k in ('short_term_max_lufs', 'lra_lu') else 0.4
                if self.frames / self.sample_rate < need:
                    res[k + '_reason'] = f'file shorter than the {need:g} s measurement window'
                elif k == 'lra_lu' and len(s) and s.max() > SILENCE_ENERGY and not len(sg):
                    res[k + '_reason'] = 'no short-term block passed the gates'
                else:
                    res[k + '_reason'] = 'signal below -70 LUFS absolute gate (digital silence)'
        tp = [float(20 * math.log10(p)) if p > 0 else None for p in self._peak]
        res['true_peak_dbtp_per_channel'] = tp
        vals = [v for v in tp if v is not None]
        res['true_peak_dbtp'] = max(vals) if vals else None
        if not vals:
            res['true_peak_dbtp_reason'] = 'digital silence'
        return res


def measure(samples, sample_rate, channels=None):
    """Measure PCM; ``samples`` is an ndarray (frames[, channels]) or an iterable of
    such chunks. Returns dict: integrated_lufs, momentary_max_lufs,
    short_term_max_lufs, lra_lu, true_peak_dbtp, true_peak_dbtp_per_channel
    (list), plus ``<key>_reason`` text for every value that is None."""
    if isinstance(samples, np.ndarray):
        x = samples
        channels = channels or (x.shape[1] if x.ndim > 1 else 1)
        step = 1 << 18
        chunks = (x[i:i + step] for i in range(0, len(x), step))
    else:
        chunks = iter(samples)
        first = next(chunks, None)
        if first is None:
            raise ValueError('empty PCM stream')
        first = np.asarray(first)
        channels = channels or (first.shape[1] if first.ndim > 1 else 1)
        chunks = itertools.chain((first,), chunks)
    meter = LoudnessMeter(sample_rate, channels)
    for c in chunks:
        meter.update(c)
    return meter.finish()
