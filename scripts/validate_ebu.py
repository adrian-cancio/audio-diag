"""Local-only official EBU v05 audit; never changes DSP or source signals.

Run from repository root: python3 -m scripts.validate_ebu --dataset DIR --output DIR
Expectations: Tech 3341 (2023) §2.9 Table 1; Tech 3342 (2023) §4 Table 1.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from audio_diag.analyze import analyze
from audio_diag.report import to_markdown
from audio_diag.safety import PROTECTED_DIRS


def validate_output(dataset, output, repo):
    """Require a fresh external directory outside protected audio storage."""
    for root in (dataset, repo, *(Path(p).resolve() for p in PROTECTED_DIRS)):
        if output == root or root in output.parents:
            raise ValueError('output must be outside dataset, repository and protected audio storage')
    if output.exists() or output.is_symlink():
        raise ValueError('output must be a new directory')

SOURCES = {n: f'https://tech.ebu.ch/docs/tech/tech{n}.pdf' for n in (3341, 3342)}


def expectation(name):
    checks = []
    reason = None
    def add(metric, value, tolerance, standard=3341):
        checks.append(dict(metric=metric, expected=value, lower=value-tolerance,
                           upper=value+(0.2 if metric == 'true_peak_dbtp' else tolerance),
                           source=SOURCES[standard], section='§2.9 Table 1' if standard == 3341 else '§4 Table 1'))
    m = re.search(r'seq-3341-(\d+)', name)
    if m:
        n = int(m[1])
        if n == 2011:
            n = 8
        if n in (1, 2):
            for metric in ('integrated_lufs', 'momentary_max_lufs', 'short_term_max_lufs'):
                add(metric, -23 if n == 1 else -33, .1)
        elif 3 <= n <= 8:
            add('integrated_lufs', -23, .1)
            if n == 6:
                reason = 'NOT SUPPORTED: normative multichannel weighting; analyzer downmixes to stereo'
        elif n in (9, 12):
            metric = 'short_term_lufs' if n == 9 else 'momentary_lufs'
            add(metric, -23, .1)
            reason = 'API limitation: constant time-series reading after 3 s' if n == 9 else 'API limitation: constant time-series reading after 1 s'
        elif n in (10, 13):
            add('short_term_max_lufs' if n == 10 else 'momentary_max_lufs', -23, .1)
        elif n in (11, 14):
            checks.append(dict(metric='successive_short_term_max_lufs' if n == 11 else 'successive_momentary_max_lufs', expected=list(range(-38,-18)), tolerance=.1, source=SOURCES[3341], section='§2.9 Table 1'))
            reason = 'API limitation: live successive readings/reset not exposed; file-based counterpart tested separately'
        elif 15 <= n <= 23:
            add('true_peak_dbtp', -6 if n <= 18 else 3 if n == 19 else 0, .4)
    m = re.search(r'seq-3342-(\d+)', name)
    if m:
        add('lra_lu', {1:10, 2:5, 3:20, 4:15, 5:5, 6:15}[int(m[1])], 1, 3342)
    if not checks:
        reason = 'Not a Tech 3341/3342 conformance case: auxiliary alignment/listening signal; filename level is not a normative tolerance'
    return checks, reason


def compare(check, measured):
    value = measured.get(check['metric'])
    return dict(check, measured=value, status='unknown' if value is None else 'pass' if check['lower'] <= value <= check['upper'] else 'fail')


def save(path, obj):
    text = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(text)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dataset, output = a.dataset.resolve(), a.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    try:
        validate_output(dataset, output, repo)
    except ValueError as exc:
        p.error(str(exc))
    files = sorted(dataset.glob('*.wav'))
    if len(files) != 70 or any(f.is_symlink() for f in files):
        p.error('expected exactly 70 non-symlink WAV files')
    output.mkdir(parents=True, exist_ok=False)
    rows = [dict(name=f.name, checks=expectation(f.name)[0], reason=expectation(f.name)[1], status='pending') for f in files]
    ledger = dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), sources=SOURCES, rows=rows)
    def persist():
        assert len(rows) == len({r['name'] for r in rows}) == 70
        ledger['counts'] = dict(Counter(r['status'] for r in rows))
        save(output/'ledger.json', ledger)
    persist()
    for f, row in zip(files, rows):
        start = time.monotonic()
        try:
            result = analyze(f)
            save(output/(f.name+'.json'), result)
            render = to_markdown(result)
            (output/(f.name+'.md')).write_text(render)
            row['render_sha256'] = hashlib.sha256(render.encode()).hexdigest()
            row['input_sha256'] = result['file']['sha256']
            row['duration_s'] = result['file']['duration_s']
            measured = {**result['level1']['loudness'], **result['level1']['peaks']}
            row['measured'] = {k: measured.get(k) for k in ('integrated_lufs','momentary_max_lufs','short_term_max_lufs','lra_lu','true_peak_dbtp')}
            if row['reason']:
                row['status'] = 'not_applicable' if not row['checks'] else 'unsupported'
            else:
                row['checks'] = [compare(c, measured) for c in row['checks']]
                states = {c['status'] for c in row['checks']}
                row['status'] = 'fail' if 'fail' in states else 'unknown' if 'unknown' in states else 'pass'
            if row['status'] == 'fail' or 'lra_lu' in [c['metric'] for c in row['checks']]:
                cmd = ['ffmpeg','-hide_banner','-nostats','-i',str(f),'-af','ebur128=peak=true','-f','null','-']
                cross = subprocess.run(cmd, capture_output=True, text=True)
                (output/(f.name+'.ffmpeg.txt')).write_text(cross.stderr)
                row['ffmpeg_crosscheck'] = dict(command=cmd, returncode=cross.returncode, role='independent comparison, not expected-value authority')
        except Exception as exc:
            row.update(status='error', error=f'{type(exc).__name__}: {exc}')
        row['elapsed_s'] = time.monotonic()-start
        persist()
        print(row['status'], f.name, flush=True)
    print(ledger['counts'])


if __name__ == '__main__':
    main()
