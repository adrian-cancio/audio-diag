"""Public analysis entry point: analyze(path, ...) -> JSON-safe dict."""
import math
import os

import numpy as np

from . import __version__
from . import decode, level2
from .loudness import LoudnessMeter
from .spectral import SpectrumAccumulator
from .timedomain import DEFAULT_SILENCE_DBFS, TimeDomainAccumulator
from .util import sanitize

DEFAULT_THRESHOLDS = {
    'target_lufs': -8.0,
    'target_tolerance_lu': 2.0,
    'true_peak_max_dbtp': -0.3,
    'silence_dbfs': DEFAULT_SILENCE_DBFS,
    'energy_drop_db': 6.0,
}
NOT_VALID = 'not valid for calibration (lossy source)'


def analyze(path, bpm=None, intents=(), thresholds=None, chunk_frames=decode.CHUNK_FRAMES):
    """Analyze an audio file (read-only). Returns a strict-JSON-safe dict.

    Raises audio_diag.decode.DecodeError if the file cannot be decoded.
    thresholds: optional dict overriding DEFAULT_THRESHOLDS keys.
    """
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    if bpm is not None:
        bpm = float(bpm)
        if not (math.isfinite(bpm) and 20 <= bpm <= 400):
            raise ValueError('bpm must be a finite number between 20 and 400')
    intents = [str(i) for i in (intents or ())]
    import hashlib
    try:
        with open(path, 'rb') as source:
            sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    except OSError as exc:
        raise decode.DecodeError(f'cannot read input: {exc}') from exc
    meta = decode.probe(path)
    meta['sha256'] = sha256
    warnings = []
    src_ch = meta['channels']
    ch = 1 if src_ch == 1 else 2
    if src_ch > 2:
        warnings.append(f'{src_ch}-channel source was downmixed to stereo by ffmpeg before analysis')
    sr = meta['sample_rate']
    loud = LoudnessMeter(sr, ch)
    spec = SpectrumAccumulator(sr, ch)
    td = TimeDomainAccumulator(sr, ch, th['silence_dbfs'])
    feat = level2.FeatureAccumulator(sr)
    for block in decode.stream(path, ch, warnings, chunk_frames):
        loud.update(block)
        spec.update(block)
        td.update(block)
        feat.update(block)
    lm = loud.finish()
    t = td.result()
    duration = t['duration_s']
    sp = spec.result()
    lossy = bool(meta['lossy_source'])
    meta = dict(meta)
    meta['duration_s'] = duration  # decoded duration is authoritative
    meta['analyzed_channels'] = ch
    file = {'path': os.fspath(path), **meta}

    loudness = {k: lm.get(k) for k in ('integrated_lufs', 'momentary_max_lufs', 'short_term_max_lufs', 'lra_lu')}
    for k in list(loudness):
        if lm.get(k + '_reason'):
            loudness[k + '_reason'] = lm[k + '_reason']
    loudness['definition'] = ('independent numpy BS.1770-4 / EBU R128: K-weighting, 400 ms blocks (75 % overlap), '
                              '-70 LUFS absolute and -10 LU relative gates; LRA from 3 s blocks, -20 LU gate, '
                              'P10-P95')
    peaks = dict(t['peaks'])
    peaks['true_peak_dbtp'] = lm['true_peak_dbtp']
    if lm.get('true_peak_dbtp_reason'):
        peaks['true_peak_dbtp_reason'] = lm['true_peak_dbtp_reason']
    names = list(t['peaks']['per_channel'])
    for n, v in zip(names, lm['true_peak_dbtp_per_channel']):
        peaks['per_channel'][n]['true_peak_dbtp'] = v
    peaks['true_peak_definition'] = '4x oversampled (polyphase Kaiser-windowed sinc, 32 taps) peak in dBTP'
    if lossy:
        loudness['validity'] = NOT_VALID
        peaks['true_peak_validity'] = NOT_VALID
    level1 = {
        'loudness': loudness,
        'peaks': peaks,
        'dc_offset': t['dc_offset'],
        'silence': t['silence'],
        'clipping': t['clipping'],
        'bands': {'definition': 'energy_db = band mean-square relative to the total of the full-range signal '
                                '(L/R power averaged); energy_db_rel_max relative to the loudest band',
                  'bands': sp['bands']},
        'spectral_slope': sp['slope'],
        'stereo': sp['stereo'] if sp['stereo'] else {
            'status': 'not_applicable', 'reason': 'mono source: no stereo metrics'},
    }
    if t['nonfinite_samples']:
        warnings.append(f"{t['nonfinite_samples']} non-finite samples were treated as 0")
    if duration < 1.0:
        warnings.append('file shorter than 1 s: gated loudness metrics may be unavailable')
    if t['silence'].get('total_silent'):
        warnings.append('file is silent (everything below the silence threshold)')
    if lossy:
        warnings.append('lossy source: true peak, crest factor and absolute loudness are ' + NOT_VALID)

    l2 = level2.analyze_level2(feat, sr, bpm, {n: v['sample_peak_dbfs'] for n, v in t['peaks']['per_channel'].items()},
                               t['rms_dbfs'], lossy, intents, drop_db=th['energy_drop_db'])
    l2['crest_factor']['per_band_db'] = {
        name: (20 * math.log10(spec.band_peak[name]) -
               10 * math.log10(spec.band_square[name] / spec.band_samples))
        if spec.band_square[name] > 1e-20 and spec.band_samples else None
        for name in sp['bands']}
    l2['crest_factor']['band_definition'] = ('Peak/RMS of brick-wall FFT band projections of '
        '8192-sample Hann-windowed frames, 50% overlap, pooled channels; window and frame-boundary '
        'effects make this heuristic, not equivalent to continuous bandpass filtering.')
    powers = [10 ** (v / 10) for v in t['rms_dbfs'].values() if v is not None]
    l2['crest_factor']['global_db'] = (peaks['sample_peak_dbfs'] - 10 * math.log10(sum(powers) / ch)
        if powers and peaks['sample_peak_dbfs'] is not None else None)
    if lossy:
        l2['crest_factor']['validity'] = NOT_VALID
    result = {
        'tool': 'audio-diag', 'version': __version__, 'file': file,
        'thresholds': th, 'level1': level1, 'level2': l2, 'level3': level2.level3(),
        'intentional_decisions': [{'text': i, 'note': 'declared by the user; never reported as a defect'}
                                  for i in intents],
        'warnings': warnings,
        'verdicts': _verdicts(level1, th, lossy),
    }
    return sanitize(result)


def _verdicts(l1, th, lossy):
    v = {}
    i = l1['loudness']['integrated_lufs']
    if i is None:
        v['loudness_target'] = {'status': 'unknown', 'reason': l1['loudness'].get('integrated_lufs_reason',
                                                                                  'no integrated loudness')}
    else:
        delta = i - th['target_lufs']
        v['loudness_target'] = {'status': 'pass' if abs(delta) <= th['target_tolerance_lu'] else 'fail',
                                'integrated_lufs': i, 'target_lufs': th['target_lufs'],
                                'tolerance_lu': th['target_tolerance_lu'], 'delta_lu': delta}
    tp = l1['peaks']['true_peak_dbtp']
    if tp is None:
        v['true_peak'] = {'status': 'unknown', 'reason': l1['peaks'].get('true_peak_dbtp_reason', 'silent')}
    else:
        v['true_peak'] = {'status': 'pass' if tp <= th['true_peak_max_dbtp'] else 'fail',
                          'true_peak_dbtp': tp, 'max_dbtp': th['true_peak_max_dbtp']}
    c = l1['clipping']
    v['clipping'] = {'status': 'pass' if c['runs_min_len_2'] == 0 else 'fail',
                     'runs_min_len_2': c['runs_min_len_2'], 'clipped_samples': c['clipped_samples']}
    if lossy:
        for k in ('loudness_target', 'true_peak'):
            v[k]['note'] = NOT_VALID
    return v
