"""Level 2: heuristic hints. Every item is confidence "heuristic" with a warning and is
never a defect verdict. Level 3 (not determinable from a stereo master) is in level3()."""
import math

import numpy as np

from .util import db10

FRAME = 2048
HOP = 256
LOW_HZ = 200.0
CONF = 'heuristic'
GENERIC_WARNING = ('heuristic estimate from a stereo master; may be wrong for intentional '
                   'arrangement choices. Not a defect.')
LEVEL3_FIELDS = {
    'kick_fundamental': 'kick fundamental cannot be isolated from a mixed stereo master',
    'kick_envelope': 'kick amplitude envelope is masked by other sources in a stereo master',
    'kick_rumble_ratio': 'kick and rumble overlap spectrally and cannot be separated in a mixdown',
    'sub_masking': 'masking between sub sources needs the individual stems',
    'key': 'musical key is not determinable from this measurement set',
}


def level3():
    return {k: {'status': 'not_determinable', 'reason': r} for k, r in LEVEL3_FIELDS.items()}


class FeatureAccumulator:
    """Streams a mono mix and records per-hop features (bounded: a few floats per 256 samples)."""

    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.win = np.hanning(FRAME + 1)[:FRAME]
        self.freqs = np.fft.rfftfreq(FRAME, 1 / sample_rate)
        self.low = self.freqs < LOW_HZ
        self.low_w = np.full(len(self.freqs), 2.0)
        self.low_w[0] = 1.0
        self.low_energy, self.mean_sq, self.peak = [], [], []
        self._buf = np.empty(0)
        self._hop_buf = np.empty(0)
        self._pos = 0
        self.mono_peak = 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        m = x.mean(axis=1) if x.ndim > 1 else x
        if not len(m):
            return
        self._hop_buf = np.concatenate((self._hop_buf, m))
        n = len(self._hop_buf) // HOP
        if n:
            h = self._hop_buf[:n * HOP].reshape(n, HOP)
            self.mean_sq.extend((h * h).mean(axis=1).tolist())
            self.peak.extend(np.abs(h).max(axis=1).tolist())
            self._hop_buf = self._hop_buf[n * HOP:]
        self._buf = np.concatenate((self._buf, m))
        if len(self._buf) >= FRAME:
            nfr = (len(self._buf) - FRAME) // HOP + 1
            for i in range(0, nfr, 256):
                k = min(256, nfr - i)
                view = np.lib.stride_tricks.sliding_window_view(self._buf, FRAME)[i * HOP:(i + k) * HOP:HOP]
                spec = np.fft.rfft(view * self.win, axis=1)
                p = (np.abs(spec[:, self.low]) ** 2) * self.low_w[self.low]
                self.low_energy.extend(p.sum(axis=1).tolist())
            self._buf = self._buf[nfr * HOP:]

    def arrays(self):
        return (np.asarray(self.low_energy), np.asarray(self.mean_sq), np.asarray(self.peak))


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


def _item(**kw):
    d = {'confidence': CONF, 'warning': GENERIC_WARNING}
    d.update(kw)
    return d


def _needs_bpm(what):
    return _item(status='needs_bpm', reason=f'{what} requires --bpm (tempo is not estimated)')


def _insufficient(reason):
    return _item(status='insufficient_data', reason=reason)


def energy_drops(ms, sr, bpm, drop_db=6.0, intents=()):
    """1-beat-window energy drops, classified as dropout (short) or break (sustained).

    A window of one beat slides in 1/8-beat steps. A window is "low" when its mean-square level is
    >= drop_db below the median of the unflagged windows of the preceding 8 beats (the reference is
    frozen while low, so long breaks stay flagged). Consecutive low windows form a region; its
    duration is estimated as span + half a window (exact for digital silence at a 6 dB threshold).
    <= 2 beats -> "dropout", >= 4 beats -> "break", otherwise "ambiguous".
    """
    if not bpm:
        return _needs_bpm('energy drops')
    beat_hops = 60.0 / bpm * sr / HOP
    w = max(1, round(beat_hops))
    step = max(1, round(beat_hops / 8))
    if len(ms) < w * 6:
        return _insufficient('fewer than 6 beats of audio')
    csum = np.concatenate(([0.0], np.cumsum(ms)))
    starts = np.arange(0, len(ms) - w + 1, step)
    energy = (csum[starts + w] - csum[starts]) / w
    lv = np.where(energy > 1e-12, 10 * np.log10(np.maximum(energy, 1e-300)), -120.0)
    hop_s = HOP / sr
    min_hist = 4 * 8  # 4 beats of history at 8 windows per beat
    hist, regions, cur = [], [], None
    for k in range(len(starts)):
        ref = float(np.median(hist[-64:])) if len(hist) >= min_hist else None
        low = ref is not None and ref > -60 and ref - lv[k] >= drop_db
        if low:
            if cur is None:
                cur = {'a': k, 'b': k, 'ref': ref, 'min': lv[k]}
            else:
                cur['b'], cur['min'] = k, min(cur['min'], lv[k])
        else:
            if cur is not None:
                regions.append(cur)
                cur = None
            hist.append(float(lv[k]))
    if cur is not None:
        regions.append(cur)
    drops = []
    for r in regions:
        span = (r['b'] - r['a']) * step * hop_s
        wl = w * hop_s
        beats = (span + 0.5 * wl) / (60.0 / bpm)
        kind = 'dropout' if beats <= 2.5 else ('break' if beats >= 3.5 else 'ambiguous')
        drops.append({'start_s': round(r['a'] * step * hop_s + 0.25 * wl, 3), 'duration_beats': round(beats, 2),
                      'drop_db': round(r['ref'] - r['min'], 2), 'kind': kind})
    return _item(status='ok', bpm=bpm, beat_s=60.0 / bpm, threshold_db=drop_db, count=len(drops), drops=drops[:50],
                 definition='1-beat sliding window level >= threshold below the median of the preceding unflagged '
                            'windows; short regions (<=2 beats) are dropouts, sustained (>=4 beats) are breaks; '
                            'durations/starts are approximate (1/8-beat resolution)')


def kick_onset_deviation(low, sr, bpm):
    """Low-band (<200 Hz) energy-flux onsets compared with a BPM grid anchored at the median phase."""
    if not bpm:
        return _needs_bpm('kick onset deviation')
    beat = 60.0 / bpm
    if len(low) < 20:
        return _insufficient('audio too short')
    env = np.sqrt(low)
    flux = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    if flux.max() <= 0:
        return _insufficient('no low-frequency onsets found')
    thr = max(float(np.median(flux) + 4 * np.median(np.abs(flux - np.median(flux)))), 0.2 * float(flux.max()))
    min_gap = max(1, int(0.3 * beat * sr / HOP))
    idx = []
    for i in range(1, len(flux) - 1):
        if flux[i] >= thr and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1] and (not idx or i - idx[-1] >= min_gap):
            idx.append(i)
        elif idx and flux[i] >= thr and flux[i] > flux[idx[-1]] and i - idx[-1] < min_gap:
            idx[-1] = i
    # Frame i covers [i*HOP, i*HOP+FRAME): its centre is the onset-time reference.
    times = (np.asarray(idx) * HOP + FRAME / 2) / sr
    if len(times) < 4:
        return _insufficient('fewer than 4 low-band onsets detected')
    # Circular median phase: anchor on the phase that minimises the median absolute wrapped offset.
    ph = times % beat
    ang = ph / beat * 2 * math.pi
    anchor = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * math.pi)) / (2 * math.pi) * beat
    dev = _wrap(ph - anchor, beat)
    # Phase coherence = mean resultant length of the onset phases (1 = all on one grid phase, ~0 =
    # uniformly random). Low values mean the detected low-band events are not (only) kicks on this
    # BPM grid, so the deviation numbers below are not meaningful.
    coherence = float(math.hypot(np.sin(ang).mean(), np.cos(ang).mean()))
    anchor = (anchor + float(np.median(dev))) % beat
    dev_ms = _wrap(ph - anchor, beat) * 1000.0
    extra = {}
    if coherence < 0.6:
        extra['reliability'] = 'low'
        extra['reliability_note'] = ('detected low-band onsets do not cluster on the BPM grid (basslines, rumble or a '
                                     'wrong --bpm are likely); deviation values are not meaningful')
    else:
        extra['reliability'] = 'ok'
    return _item(status='ok', bpm=bpm, onsets=len(times), anchor_phase_s=float(anchor),
                 phase_coherence=coherence, **extra,
                 deviation_ms={'median_abs': float(np.median(np.abs(dev_ms))), 'mean_abs': float(np.abs(dev_ms).mean()),
                               'max_abs': float(np.abs(dev_ms).max()), 'std': float(dev_ms.std())},
                 onsets_beyond_10ms=int((np.abs(dev_ms) > 10).sum()),
                 resolution_ms=HOP / sr * 1000.0,
                 definition='onsets = peaks of low-band (<200 Hz) energy flux; deviation = signed ms from '
                            'the BPM grid anchored at the median phase')


def section_contrast(ms, sr, bpm=None, bars=8):
    """RMS level spread between consecutive sections (8 bars with bpm, otherwise 15 s)."""
    sec = bars * 4 * 60.0 / bpm if bpm else 15.0
    per = max(1, round(sec * sr / HOP))
    nb = len(ms) // per
    if nb < 2:
        return _insufficient('fewer than two sections')
    lv = [db10(ms[i * per:(i + 1) * per].mean()) for i in range(nb)]
    if any(v is None for v in lv):
        return _insufficient('silent section(s) make contrast undefined')
    steps = np.abs(np.diff(lv))
    return _item(status='ok', section_s=sec, sections=nb, levels_db=[round(float(v), 2) for v in lv[:64]],
                 range_db=float(max(lv) - min(lv)), max_step_db=float(steps.max()),
                 definition='mean-square level (dB) per section; contrast = max-min and max consecutive step')


def crest_factor(peaks_dbfs, rms_dbfs, ms, pk, sr, lossy=False):
    """Overall and 3 s block crest factor (peak/RMS, dB)."""
    out = {}
    for name in rms_dbfs:
        p, r = peaks_dbfs.get(name), rms_dbfs.get(name)
        out[name] = (p - r) if p is not None and r is not None else None
    per = max(1, round(3 * sr / HOP))
    blocks = []
    for i in range(0, len(ms) - per + 1, per):
        e = ms[i:i + per].mean()
        if e > 1e-12:
            blocks.append(20 * math.log10(max(pk[i:i + per].max(), 1e-12)) - 10 * math.log10(e))
    item = _item(status='ok', per_channel_db=out,
                 block_median_db=float(np.median(blocks)) if blocks else None,
                 definition='crest = sample peak - RMS (dB); block values use 3 s windows of the mono mix')
    if not blocks:
        item['block_median_db_reason'] = 'no block long/loud enough'
    if lossy:
        item['validity'] = 'not valid for calibration (lossy source)'
    return item


def analyze_level2(acc, sr, bpm, peaks_dbfs, rms_dbfs, lossy=False, intents=(), drop_db=6.0):
    low, ms, pk = acc.arrays()
    return {
        'energy_drops': energy_drops(ms, sr, bpm, drop_db=drop_db),
        'kick_onset_deviation': kick_onset_deviation(low, sr, bpm),
        'section_contrast': section_contrast(ms, sr, bpm),
        'crest_factor': crest_factor(peaks_dbfs, rms_dbfs, ms, pk, sr, lossy),
    }
