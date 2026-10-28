"""Markdown rendering of an analyze() result."""


def _f(v, nd=2, unit=''):
    return 'n/a' if v is None else f'{v:.{nd}f}{unit}'


def to_markdown(r):
    f, l1, l2, th = r['file'], r['level1'], r['level2'], r['thresholds']
    L, P = l1['loudness'], l1['peaks']
    out = [f"# audio-diag report: {f['path']}", '',
           f"- format: {f.get('format')} / codec {f.get('codec')} / {f.get('sample_rate')} Hz / "
           f"{f.get('channels')} ch / {_f(f.get('duration_s'), 2, ' s')}",
           f"- lossy source: {'yes' if f.get('lossy_source') else 'no'}", '']
    out += [f"- SHA-256: `{f.get('sha256')}`", '']
    if r['warnings']:
        out += ['## Warnings', ''] + [f'- {w}' for w in r['warnings']] + ['']
    out += ['## Verdicts', '', '| check | status | detail |', '|---|---|---|']
    for k, v in r['verdicts'].items():
        detail = v.get('reason') or v.get('note') or ', '.join(
            f'{a}={_f(b) if isinstance(b, float) else b}' for a, b in v.items() if a != 'status')
        out.append(f"| {k} | {v['status']} | {detail} |")
    out += ['', '## Level 1 (reliable)', '', '### Loudness', '',
            f"- integrated: {_f(L['integrated_lufs'], 1, ' LUFS')}",
            f"- momentary max: {_f(L['momentary_max_lufs'], 1, ' LUFS')}",
            f"- short-term max: {_f(L['short_term_max_lufs'], 1, ' LUFS')}",
            f"- LRA: {_f(L['lra_lu'], 1, ' LU')}"]
    for k in ('integrated_lufs', 'momentary_max_lufs', 'short_term_max_lufs', 'lra_lu'):
        if L.get(k + '_reason'):
            out.append(f"  - {k}: {L[k + '_reason']}")
    if L.get('validity'):
        out.append(f"- validity: {L['validity']}")
    out += ['', '### Peaks', '', f"- sample peak: {_f(P['sample_peak_dbfs'], 2, ' dBFS')}",
            f"- true peak: {_f(P['true_peak_dbtp'], 2, ' dBTP')}"]
    if P.get('true_peak_validity'):
        out.append(f"- true peak validity: {P['true_peak_validity']}")
    c = l1['clipping']
    out += ['', '### Clipping', '', f"- clipped samples: {c['clipped_samples']}, runs: {c['runs_total']}, "
            f"runs >=2: {c['runs_min_len_2']}, longest run: {c['max_run_length']}", '',
            '### DC offset / silence', '']
    out += [f'- DC {k}: {v:.6f}' for k, v in l1['dc_offset'].items()]
    s = l1['silence']
    out.append(f"- leading silence: {_f(s.get('leading_s'), 2, ' s')}, trailing: {_f(s.get('trailing_s'), 2, ' s')}")
    out += ['', '### Bands', '', '| band | energy dB (total) | dB rel. max |', '|---|---|---|']
    for k, b in l1['bands']['bands'].items():
        out.append(f"| {k} | {_f(b['energy_db'])} | {_f(b['energy_db_rel_max'])} |")
    out += ['', f"Spectral slope: {_f(l1['spectral_slope'].get('slope_db_per_octave'), 2, ' dB/oct')}", '',
            '### Stereo', '']
    st = l1['stereo']
    if 'bands' in st:
        out += ['| band | correlation | mono loss dB |', '|---|---|---|']
        for k, b in st['bands'].items():
            out.append(f"| {k} | {_f(b.get('correlation'))} | {_f(b.get('mono_loss_db'))} |")
    else:
        out.append(st['reason'])
    out += ['', '## Level 2 (heuristic, not defects)', '']
    for k, v in l2.items():
        out.append(f"- **{k}** [{v['confidence']}]: {v['status']}" + (f" - {v['reason']}" if v.get('reason') else ''))
    crest = l2['crest_factor']
    out += [f"- global crest: {_f(crest.get('global_db'))} dB",
            f"- band crest definition: {crest.get('band_definition')}"]
    out += [f"  - {name}: {_f(value)} dB" for name, value in crest.get('per_band_db', {}).items()]
    if crest.get('validity'):
        out.append(f"- crest validity: {crest['validity']}")
    out += ['', '_' + 'Heuristic estimates from a stereo master; never treat as defects.' + '_', '',
            '## Level 3 (not determinable)', '']
    out += [f"- {k}: not_determinable - {v['reason']}" for k, v in r['level3'].items()]
    if r['intentional_decisions']:
        out += ['', '## Intentional decisions (declared by the user)', '']
        out += [f"- {d['text']}" for d in r['intentional_decisions']]
    return '\n'.join(out) + '\n'
