"""Gain-normalized spectral profiles and dispersion, never an averaged target."""


def profile(report):
    """Power normalization before comparison: P_band/P_total cancels gain squared.

    Equivalent to scaling each decoded signal to unit pooled spectral power before
    band integration. No loudness/peak/crest targets are inferred, especially for
    lossy material. Null bands remain unknown rather than becoming zero.
    """
    return {'file': report['file'], 'version': report['version'],
            'normalization': 'unit total spectral power; 10*log10(P_band/P_total)',
            'relative_bands_db': {k: v['energy_db'] for k, v in
                                  report['level1']['bands']['bands'].items()}}


def compare(profiles):
    """Return individual profiles and observed min/max, not acceptance thresholds."""
    if not profiles:
        raise ValueError('at least one profile is required')
    names = tuple(profiles[0]['relative_bands_db'])
    if any(tuple(p['relative_bands_db']) != names for p in profiles):
        raise ValueError('incompatible band definitions')
    envelope = {}
    for name in names:
        values = [p['relative_bands_db'][name] for p in profiles]
        known = [v for v in values if v is not None]
        envelope[name] = {'min_db': min(known) if known else None,
                          'max_db': max(known) if known else None,
                          'known_count': len(known), 'unknown_count': len(values) - len(known)}
    return {'count': len(profiles), 'profiles': profiles, 'dispersion': envelope,
            'warning': 'Observed independent per-band extrema, not an averaged sound or a defect gate; '
                       'extrema need not coexist in any reference. Listening remains required.'}
