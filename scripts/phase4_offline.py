"""Reproducible offline Phase 4 evidence. Outputs only to data/phase4 and scratch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from audio_diag import decode
from scripts.phase4_controls import invert_right_low, localized_mute_checks

from audio_diag import analyze
from audio_diag.profiles import profile, compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/phase4'
SCRATCH = Path('/home/cancio/.hermes/cache/scratch/opencode/phase4-bde-v2')
MUSIC = Path('/home/cancio/Música')
IDS = ['b3ooXLd6ymU', '6k_H100tMvg', 'UEVKoCoU1u4', 'bDmZcs4qJp4',
       'hDnPaErXipI', '-vBXROMQ2i4', 'lBm7weRu0ik']
WAVS = ['Samples/Synths/Melodic Schranz/MS Melodic Schranz Project (Live 11).wav',
        'Samples/Synths/Melodic Schranz/MS Oldschool Schranz Project (Live 11).wav',
        'Samples/Vocals/Misc/MS Schranz Vocal Tool Project (Live 11).wav']
MASTER = MUSIC / 'OMNi/Projects/02 - Schranz Hard Techno (158 BPM)/master.wav'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, data):
    (OUT / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                               allow_nan=False) + '\n')


def measure(name, path, **kwargs):
    before = digest(path)
    result = analyze(path, **kwargs)
    assert before == result['file']['sha256'] == digest(path), 'source changed'
    assert not result['warnings'] or not any('decoder reported' in w for w in result['warnings'])
    save(name, result)
    print(name, before, flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / 'controlled-checks.json'
    historical = OUT / 'controlled-checks-v1-failed.json'
    if previous.exists() and not historical.exists():
        historical.write_bytes(previous.read_bytes())
    SCRATCH.mkdir(parents=True, exist_ok=True)
    directory = MUSIC / 'Referencias Schranz/Escucha - comprimido'
    files = list(directory.glob('*.opus'))
    assert len(files) == 7
    opus = []
    mapping = []
    for ident in IDS:
        matches = [p for p in files if '[' + ident + ']' in p.name]
        assert len(matches) == 1
        path = matches[0]
        source = path.with_suffix('.info.json')
        metadata = json.loads(source.read_text())
        assert metadata['id'] == ident
        r = measure('opus-' + ident, path)
        opus.append(profile(r))
        mapping.append({'id': ident, 'path': str(path), 'sha256': r['file']['sha256'],
                        'source_metadata': str(source), 'metadata_sha256': digest(source),
                        'source_url': metadata.get('webpage_url'), 'title': metadata.get('title')})
    save('opus-comparison', compare(opus))
    wav = [profile(measure('wav-' + str(i + 1), MUSIC / 'OMNi/User Library' / p))
           for i, p in enumerate(WAVS)]
    save('wav-profiles', {'count': len(wav), 'profiles': wav,
                          'warning': 'Project exports, not finished mastering targets.'})
    master = measure('master', MASTER, bpm=158)
    save('manifest', {'opus_count': 7, 'wav_count': 3, 'opus_mapping': mapping,
                      'wav_mapping': [p['file'] for p in wav], 'master': master['file']})
    # Float PCM preserves unclamped filter output; -n refuses overwriting artifacts.
    clean = SCRATCH / 'clean.wav'
    commands = []

    def generate(source, target, filters=None, excerpt=False):
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-n']
        if excerpt:
            cmd += ['-ss', '30']
        cmd += ['-i', str(source)]
        if excerpt:
            cmd += ['-t', '24']
        if filters:
            cmd += ['-af', filters]
        cmd += ['-c:a', 'pcm_f32le', str(target)]
        subprocess.run(cmd, check=True, capture_output=True)
        commands.append(cmd)

    generate(MASTER, clean, excerpt=True)
    recipes = {
        'low_cut': 'highpass=f=250:p=2',
        'low_phase': 'whole-excerpt rFFT: negate right bins f<120 Hz; preserve left and higher bins; inverse rFFT',
        'harsh': 'equalizer=f=3200:t=o:w=1.3:g=15',
        'clipped': r'volume=8,aeval=clip(val(0)\,-1\,1)|clip(val(1)\,-1\,1)',
        'muted': "volume=0:enable='between(t,8,12)'",
        'intentional_break': "volume=0:enable='between(t,8,12)'",
    }
    results = {'clean': measure('excerpt-clean', clean, bpm=158)}
    for name, recipe in recipes.items():
        target = SCRATCH / (name + '.wav')
        if name == 'low_phase':
            samples = np.concatenate(list(decode.stream(clean, 2)))
            rate = decode.probe(clean)['sample_rate']
            changed = invert_right_low(samples, rate)
            cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'f64le',
                   '-ar', str(rate), '-ac', '2', '-i', '-', '-c:a', 'pcm_f32le', str(target)]
            subprocess.run(cmd, input=changed.astype('<f8').tobytes(), check=True, capture_output=True)
            commands.append(cmd)
        else:
            generate(clean, target, recipe)
        results[name] = measure('excerpt-' + name, target, bpm=158,
                                intents=['Controlled intentional break at excerpt seconds 8–12']
                                if name == 'intentional_break' else [])
    base = results['clean']
    band = lambda r, n: r['level1']['bands']['bands'][n]['energy_db']
    mono = lambda r: r['level1']['stereo']['bands']['below_120']['mono_loss_db']
    original = np.concatenate(list(decode.stream(clean, 2)))
    phase = np.concatenate(list(decode.stream(SCRATCH / 'low_phase.wav', 2)))
    expected_phase = invert_right_low(original, decode.probe(clean)['sample_rate'])
    checks = {
        'phase_readback_matches_exact_split': bool(np.allclose(phase, expected_phase, atol=1e-7, rtol=1e-7)),
        'phase_left_unchanged': bool(np.array_equal(original[:, 0], phase[:, 0])),
        'low_cut_sub_decreases': band(results['low_cut'], 'sub') < band(base, 'sub') - 3,
        'low_phase_mono_loss_worsens': mono(results['low_phase']) < mono(base) - 3,
        'harsh_mid_share_increases': band(results['harsh'], 'mid') > band(base, 'mid') + 1,
        'clipping_runs_increase': results['clipped']['level1']['clipping']['runs_min_len_2'] > base['level1']['clipping']['runs_min_len_2'],
        **localized_mute_checks(
            np.concatenate(list(decode.stream(clean, 2))),
            np.concatenate(list(decode.stream(SCRATCH / 'muted.wav', 2))),
            decode.probe(clean)['sample_rate'], results['muted']['level2']['energy_drops']),
        'intent_echoed': bool(results['intentional_break']['intentional_decisions']),
        'intent_not_a_defect_gate': results['intentional_break']['verdicts'] == results['muted']['verdicts'],
    }
    save('controlled-checks', {'excerpt': {'start_s': 30, 'duration_s': 24},
                              'commands': commands, 'recipes': recipes, 'assertions': checks,
                              'expected_basis': 'Independent filter operations predict directions, not numerical truth. '
                              'Muting and an intentional break are physically identical: intent cannot be inferred.',
                              'observations': {k: {'sub_db': band(r, 'sub'), 'mid_db': band(r, 'mid'),
                                                   'low_mono_loss_db': mono(r),
                                                   'clipping': r['level1']['clipping'],
                                                   'drops': r['level2']['energy_drops']} for k, r in results.items()}})
    assert digest(MASTER) == master['file']['sha256']
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
