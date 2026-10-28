"""Streaming time-domain statistics: sample peak, clipping runs, DC offset, silence."""
import math

import numpy as np

from .util import db20

CLIP_THRESHOLD = 1.0 - 1e-6
DEFAULT_SILENCE_DBFS = -60.0


class TimeDomainAccumulator:
    def __init__(self, sample_rate, channels, silence_dbfs=DEFAULT_SILENCE_DBFS):
        self.sr, self.ch = sample_rate, channels
        self.silence_amp = 10 ** (silence_dbfs / 20)
        self.silence_dbfs = silence_dbfs
        self.frames = 0
        self.peak = np.zeros(channels)
        self.sum = np.zeros(channels)
        self.sumsq = np.zeros(channels)
        self.clipped = np.zeros(channels, dtype=np.int64)
        self.runs1 = np.zeros(channels, dtype=np.int64)
        self.runs2 = np.zeros(channels, dtype=np.int64)
        self.maxrun = np.zeros(channels, dtype=np.int64)
        self._open = np.zeros(channels, dtype=np.int64)  # run length continuing across chunks
        self.first_loud = None
        self.last_loud = None
        self.nonfinite = 0

    def update(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        n = len(x)
        if not n:
            return
        bad = ~np.isfinite(x)
        if bad.any():
            self.nonfinite += int(bad.sum())
            x = np.where(bad, 0.0, x)
        x64 = x.astype(np.float64)
        a = np.abs(x64)
        self.peak = np.maximum(self.peak, a.max(axis=0))
        self.sum += x64.sum(axis=0)
        self.sumsq += np.einsum('ij,ij->j', x64, x64)
        clip = a >= CLIP_THRESHOLD
        for c in range(self.ch):
            self._runs(c, clip[:, c], n)
        loud = np.flatnonzero((a >= self.silence_amp).any(axis=1))
        if len(loud):
            if self.first_loud is None:
                self.first_loud = self.frames + int(loud[0])
            self.last_loud = self.frames + int(loud[-1])
        self.frames += n

    def _runs(self, c, m, n):
        self.clipped[c] += int(m.sum())
        d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        lens = (ends - starts).astype(np.int64)
        carry = int(self._open[c])
        if carry:
            if len(starts) and starts[0] == 0:
                lens[0] += carry  # the run continues across the chunk boundary
            else:
                self._close(c, np.array([carry]))
            self._open[c] = 0
        if len(lens) and ends[-1] == n:  # last run is still open at the chunk end
            self._open[c] = int(lens[-1])
            lens = lens[:-1]
        self._close(c, lens)

    def _close(self, c, lens):
        """Account for runs that are definitely finished."""
        if len(lens):
            self.runs1[c] += len(lens)
            self.runs2[c] += int((lens >= 2).sum())
            self.maxrun[c] = max(self.maxrun[c], int(lens.max()))

    def _flush(self):
        for c in range(self.ch):
            if self._open[c]:
                self._close(c, np.array([self._open[c]]))
                self._open[c] = 0

    def result(self, names=('left', 'right')):
        if self.ch == 1:
            names = ('mono',)
        self._flush()
        dur = self.frames / self.sr
        n = max(self.frames, 1)
        per = {}
        for c, name in enumerate(names):
            pk = db20(self.peak[c]) if self.peak[c] > 0 else None
            e = {'sample_peak_dbfs': pk}
            if pk is None:
                e['sample_peak_dbfs_reason'] = 'digital silence'
            per[name] = e
        peaks = {'definition': 'sample peak = max |x| per channel in dBFS (full scale = 1.0)',
                 'per_channel': per,
                 'sample_peak_dbfs': max((p['sample_peak_dbfs'] for p in per.values()
                                          if p['sample_peak_dbfs'] is not None), default=None)}
        if peaks['sample_peak_dbfs'] is None:
            peaks['sample_peak_dbfs_reason'] = 'digital silence'
        dc = {}
        for c, name in enumerate(names):
            dc[name] = float(self.sum[c] / n)
        clipping = {
            'definition': 'run = maximal sequence of consecutive samples with |x| >= 1 - 1e-6 in one channel',
            'clipped_samples': int(self.clipped.sum()),
            'runs_total': int(self.runs1.sum()),
            'runs_min_len_2': int(self.runs2.sum()),
            'max_run_length': int(self.maxrun.max()) if self.ch else 0,
            'per_channel': {name: {'clipped_samples': int(self.clipped[c]), 'runs_total': int(self.runs1[c]),
                                   'runs_min_len_2': int(self.runs2[c]), 'max_run_length': int(self.maxrun[c])}
                            for c, name in enumerate(names)},
        }
        silence = {'threshold_dbfs': self.silence_dbfs,
                   'definition': 'leading/trailing time where every sample |x| is below the threshold'}
        if self.first_loud is None:
            silence.update(leading_s=None, trailing_s=None, total_silent=True,
                           leading_s_reason='whole file is below the silence threshold',
                           trailing_s_reason='whole file is below the silence threshold')
        else:
            silence.update(leading_s=self.first_loud / self.sr,
                           trailing_s=(self.frames - 1 - self.last_loud) / self.sr, total_silent=False)
        rms = {name: (db20(math.sqrt(self.sumsq[c] / n)) if self.sumsq[c] > 0 else None)
               for c, name in enumerate(names)}
        return {'duration_s': dur, 'peaks': peaks, 'dc_offset': dc, 'clipping': clipping,
                'silence': silence, 'rms_dbfs': rms, 'nonfinite_samples': int(self.nonfinite)}
