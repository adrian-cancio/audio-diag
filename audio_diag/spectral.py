"""Streaming spectral accumulation: bands, spectral slope, per-band stereo metrics.

Welch-style averaged periodograms (Hann, 8192-sample frames, 50 % overlap) are
accumulated per channel together with the L/R cross-spectrum, so memory is
bounded regardless of file length. Everything is derived from those sums.
"""
import math

import numpy as np

from .util import db10

FRAME = 8192
HOP = FRAME // 2

BANDS = (
    ('sub', 20.0, 60.0),
    ('bass', 60.0, 120.0),
    ('low_mid', 120.0, 500.0),
    ('mid', 500.0, 4000.0),
    ('high', 4000.0, 10000.0),
    ('air', 10000.0, math.inf),
)
STEREO_BANDS = (('below_120', 0.0, 120.0), ('120_500', 120.0, 500.0), ('above_500', 500.0, math.inf))
SLOPE_RANGE = (50.0, 16000.0)
MONO_LOSS_FLOOR_DB = -100.0
SILENT_REL = 1e-12


class SpectrumAccumulator:
    """Feed (frames, channels) float arrays; call result() once at the end."""

    def __init__(self, sample_rate, channels):
        self.sr, self.channels = sample_rate, channels
        self.window = np.hanning(FRAME + 1)[:FRAME]  # periodic Hann
        # Parseval: sum_n (w x)^2 = (1/N) sum_k |X_k|^2, so dividing by N*sum(w^2)
        # turns the summed periodogram into a mean square.
        self.norm = float(FRAME * np.sum(self.window ** 2))
        nb = FRAME // 2 + 1
        self.pll = np.zeros(nb)
        self.prr = np.zeros(nb)
        self.plr = np.zeros(nb)
        self.count = 0
        self.band_peak = {name: 0.0 for name, _, _ in BANDS}
        self.band_square = {name: 0.0 for name, _, _ in BANDS}
        self.band_samples = 0
        self._buf = np.zeros((0, channels))
        self.total = 0
        # one-sided weights so that sum_k w_k P_k equals the mean square
        self.weights = np.full(nb, 2.0)
        self.weights[0] = self.weights[-1] = 1.0
        self.freqs = np.arange(nb) * sample_rate / FRAME

    def _process(self, frames):
        if not len(frames):
            return
        for i in range(0, len(frames), 64):
            seg = frames[i:i + 64] * self.window[None, :, None]
            spec = np.fft.rfft(seg, axis=1)
            left = spec[:, :, 0]
            self.pll += (np.abs(left) ** 2).sum(axis=0)
            if self.channels > 1:
                right = spec[:, :, 1]
                self.prr += (np.abs(right) ** 2).sum(axis=0)
                self.plr += (left * np.conj(right)).real.sum(axis=0)
            # Heuristic band crest: brick-wall FFT band projection of Hann-windowed frames.
            # Both peak and RMS use the same windowed projection (not the averaged PSD).
            for name, lo, hi in BANDS:
                band = np.fft.irfft(spec * self._mask(lo, hi)[None, :, None], n=FRAME, axis=1)
                self.band_peak[name] = max(self.band_peak[name], float(np.abs(band).max()))
                self.band_square[name] += float(np.sum(band * band))
            self.band_samples += seg.size
            self.count += len(seg)

    def _frames(self, buf):
        n = (len(buf) - FRAME) // HOP + 1 if len(buf) >= FRAME else 0
        if n <= 0:
            return None, 0
        view = np.lib.stride_tricks.sliding_window_view(buf, FRAME, axis=0)  # (n_all, ch, FRAME)
        return view[::HOP][:n].transpose(0, 2, 1), n

    def update(self, samples):
        x = np.asarray(samples, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        self.total += len(x)
        self._buf = np.concatenate((self._buf, x)) if len(self._buf) else x.copy()
        frames, n = self._frames(self._buf)
        if n:
            self._process(frames)
            consumed = n * HOP
            self._buf = self._buf[consumed:]

    def result(self):
        """Finalise (zero-padded last frame) and return a dict of derived metrics."""
        # After the last full frame, the first HOP samples of the buffer are already
        # covered by that frame's second half; anything beyond needs a padded frame.
        uncovered = len(self._buf) - (HOP if self.count else 0)
        if self.total and uncovered > 0:
            pad = np.zeros((FRAME, self.channels))
            pad[:len(self._buf)] = self._buf
            self._process(pad[None])
        return self._derive()

    # -- derived metrics -------------------------------------------------
    def _mask(self, lo, hi):
        return (self.freqs >= lo) & (self.freqs < hi)

    def _derive(self):
        n = max(self.count, 1)
        # Mean per-frame periodograms -> power per bin (mean square contribution).
        scale = self.weights / (n * self.norm)
        pll, prr, plr = self.pll * scale, self.prr * scale, self.plr * scale
        if self.channels == 1:
            avg = pll
            prr = plr = None
        else:
            avg = (pll + prr) / 2
        total = float(avg.sum())
        out = {'total_mean_square': total}
        bands = {}
        energies = {}
        for name, lo, hi in BANDS:
            m = self._mask(lo, hi)
            e = float(avg[m].sum()) if m.any() else None
            energies[name] = e
        top = max([e for e in energies.values() if e], default=None)
        for name, lo, hi in BANDS:
            e = energies[name]
            entry = {'lo_hz': lo, 'hi_hz': None if math.isinf(hi) else hi}
            if e is None:
                entry.update(energy_db=None, energy_db_rel_max=None,
                             energy_db_reason='band lies above the Nyquist frequency')
            elif total <= 0 or top is None or e <= total * SILENT_REL:
                entry.update(energy_db=None, energy_db_rel_max=None,
                             energy_db_reason='no energy in band (silent signal)'
                             if total <= 0 else 'band energy below -120 dB relative to total')
            else:
                entry.update(energy_db=db10(e / total), energy_db_rel_max=db10(e / top))
            bands[name] = entry
        out['bands'] = bands
        out['slope'] = self._slope(avg)
        out['stereo'] = self._stereo(pll, prr, plr) if self.channels > 1 else None
        return out

    def _slope(self, avg):
        lo, hi = SLOPE_RANGE
        hi = min(hi, 0.98 * self.sr / 2)
        info = {'range_hz': [lo, hi],
                'definition': 'least-squares slope of 10*log10(averaged PSD) versus log2(frequency); '
                              'PSD averaged in 1/3-octave cells (power mean) so every octave weighs equally'}
        if hi <= lo * 2 or float(avg.sum()) <= 0:
            info.update(slope_db_per_octave=None,
                        slope_db_per_octave_reason='silent signal' if float(avg.sum()) <= 0
                        else 'sample rate too low for the slope range')
            return info
        edges = lo * 2 ** (np.arange(0, math.log2(hi / lo) * 3 + 1) / 3)
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (self.freqs >= a) & (self.freqs < b)
            if m.sum() < 1:
                continue
            p = float(avg[m].mean())
            if p > 0:
                xs.append(math.log2(math.sqrt(a * b)))
                ys.append(10 * math.log10(p))
        if len(xs) < 4:
            info.update(slope_db_per_octave=None, slope_db_per_octave_reason='not enough spectral content')
            return info
        info['slope_db_per_octave'] = float(np.polyfit(xs, ys, 1)[0])
        return info

    def _stereo(self, pll, prr, plr):
        res = {'definition': {
            'correlation': 'Pearson correlation of band-filtered L and R, from the cross-spectrum '
                           '(DC bin excluded); null when either channel is silent in the band',
            'mono_loss_db': '10*log10(E[((L+R)/2)^2] / E[(L^2+R^2)/2]) per band; 0 dB identical '
                            'channels, -3.01 dB uncorrelated, floored at -100 dB'}}
        res['bands'] = {name: self._stereo_band(pll, prr, plr, lo, hi) for name, lo, hi in STEREO_BANDS}
        res['full_range'] = self._stereo_band(pll, prr, plr, 0.0, math.inf)
        return res

    def _stereo_band(self, pll, prr, plr, lo, hi):
        m = self._mask(lo, hi)
        ll, rr, lr = pll[m].sum(), prr[m].sum(), plr[m].sum()
        entry = {'lo_hz': lo, 'hi_hz': None if math.isinf(hi) else hi}
        mc = m.copy()
        mc[0] = False
        cl, cr, cx = pll[mc].sum(), prr[mc].sum(), plr[mc].sum()
        tot = max(pll.sum(), prr.sum(), 1e-300)
        if cl <= tot * SILENT_REL or cr <= tot * SILENT_REL or not mc.any():
            entry.update(correlation=None, correlation_reason='left or right channel is silent in this band')
        else:
            entry['correlation'] = float(max(-1.0, min(1.0, cx / math.sqrt(cl * cr))))
        both = (ll + rr) / 2
        if both <= tot * SILENT_REL:
            entry.update(mono_loss_db=None, mono_loss_db_reason='no energy in this band')
        else:
            mono = (ll + rr + 2 * lr) / 4
            if mono <= both * SILENT_REL:
                entry.update(mono_loss_db=None, mono_loss_db_reason='mono sum is digital silence')
            else:
                entry['mono_loss_db'] = max(MONO_LOSS_FLOOR_DB, db10(mono / both))
        return entry
