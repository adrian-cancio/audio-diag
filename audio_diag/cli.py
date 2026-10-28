"""Command line: python3 -m audio_diag analyze FILE [options].

stdout: JSON (or nothing when --json OUT is given); stderr: markdown report
(or nothing when --md OUT is given). Exit 0 ok, 2 usage/decode/output error.
"""
import argparse
import json
import sys

from .analyze import DEFAULT_THRESHOLDS, analyze
from .decode import DecodeError
from .report import load_schema, to_markdown
from .safety import OutputRefused, check_output_path, write_text_atomic


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write(f'audio_diag: error: {message}\n')
        raise SystemExit(2)


def build_parser():
    p = _Parser(prog='audio_diag', description='Stereo-master audio diagnostics.')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze', help='analyze an audio file (read-only)')
    a.add_argument('file')
    a.add_argument('--bpm', type=float, help='tempo, enables bpm-dependent level-2 hints')
    a.add_argument('--json', metavar='OUT', help='write JSON here instead of stdout')
    a.add_argument('--md', metavar='OUT', help='write markdown here instead of stderr')
    a.add_argument('--intent', action='append', default=[], metavar='TEXT',
                   help='declare an intentional decision (repeatable); never reported as a defect')
    a.add_argument('--target-lufs', type=float, default=DEFAULT_THRESHOLDS['target_lufs'])
    a.add_argument('--tolerance-lu', type=float, default=DEFAULT_THRESHOLDS['target_tolerance_lu'])
    a.add_argument('--tp-max', type=float, default=DEFAULT_THRESHOLDS['true_peak_max_dbtp'],
                   help='maximum true peak in dBTP')
    a.add_argument('--silence-dbfs', type=float, default=DEFAULT_THRESHOLDS['silence_dbfs'])
    a.add_argument('--drop-db', type=float, default=DEFAULT_THRESHOLDS['energy_drop_db'])
    sub.add_parser('schema', help='print report JSON schema')
    return p


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return 2 if e.code not in (0, None) else 0
    if args.command == 'schema':
        schema = load_schema()
        sys.stdout.write(json.dumps(schema, indent=2) + '\n')
        return 0
    try:
        for out in (args.json, args.md):
            if out:
                check_output_path(out, args.file)
        if args.json and args.md:
            from os.path import realpath
            if realpath(args.json) == realpath(args.md):
                raise OutputRefused('--json and --md point at the same file')
    except OutputRefused as exc:
        sys.stderr.write(f'audio_diag: error: {exc}\n')
        return 2
    th = {'target_lufs': args.target_lufs, 'target_tolerance_lu': args.tolerance_lu,
          'true_peak_max_dbtp': args.tp_max, 'silence_dbfs': args.silence_dbfs, 'energy_drop_db': args.drop_db}
    try:
        result = analyze(args.file, bpm=args.bpm, intents=args.intent, thresholds=th)
    except DecodeError as exc:
        sys.stderr.write(f'audio_diag: error: {exc}\n')
        return 2
    except ValueError as exc:
        sys.stderr.write(f'audio_diag: error: {exc}\n')
        return 2
    text = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + '\n'
    md = to_markdown(result)
    try:
        if args.json:
            write_text_atomic(args.json, text, args.file)
        else:
            sys.stdout.write(text)
        if args.md:
            write_text_atomic(args.md, md, args.file)
        else:
            sys.stderr.write(md)
    except (OutputRefused, OSError) as exc:
        sys.stderr.write(f'audio_diag: error: cannot write output: {exc}\n')
        return 2
    return 0
