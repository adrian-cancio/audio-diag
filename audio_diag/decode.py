"""ffprobe metadata and chunked ffmpeg float32 decoding (input opened read-only).

ffmpeg/ffprobe are only used to decode; every reported number is computed in
numpy by this package.
"""
import json
import os
import subprocess

import numpy as np

LOSSY_CODECS = {
    'mp3', 'mp2', 'opus', 'vorbis', 'aac', 'wmav1', 'wmav2', 'wmapro', 'ac3', 'eac3',
    'amr_nb', 'amr_wb', 'musepack7', 'musepack8', 'atrac3', 'atrac3p', 'dts', 'speex',
    'cook', 'ra_144', 'ra_288', 'qdm2', 'adpcm_ima_wav', 'adpcm_ms',
}
CHUNK_FRAMES = 1 << 17


class DecodeError(Exception):
    """Input missing, unreadable, corrupt or without decodable audio."""


def _url(path):
    # 'file:' prefix stops ffmpeg treating odd names as URLs/options.
    return 'file:' + os.path.abspath(os.fspath(path))


def probe(path):
    """Return metadata dict (format, codec, sample_rate, bit_depth, channels, duration_s,
    lossy_source). Raises DecodeError if the file has no readable audio stream."""
    p = os.fspath(path)
    if not os.path.exists(p):
        raise DecodeError(f'input not found: {p}')
    if not os.path.isfile(p):
        raise DecodeError(f'input is not a regular file: {p}')
    try:
        with open(p, 'rb') as fh:  # read-only access check
            fh.read(1)
    except OSError as exc:
        raise DecodeError(f'cannot read input: {exc}') from exc
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format',
                            '-show_streams', '-i', _url(p)], capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DecodeError(f'ffprobe failed: {exc}') from exc
    if r.returncode:
        raise DecodeError('cannot decode audio (corrupt or unsupported file): '
                          + r.stderr.decode(errors='replace').strip()[:300])
    try:
        data = json.loads(r.stdout or b'{}')
    except ValueError as exc:
        raise DecodeError('ffprobe returned invalid output') from exc
    audio = [s for s in data.get('streams') or [] if s.get('codec_type') == 'audio']
    if not audio:
        raise DecodeError('no audio stream found in input')
    s, fmt = audio[0], data.get('format') or {}

    def num(v):
        try:
            v = float(v)
            return v if np.isfinite(v) and v >= 0 else None
        except (TypeError, ValueError):
            return None

    rate = int(num(s.get('sample_rate')) or 0)
    channels = int(num(s.get('channels')) or 0)
    if rate <= 0 or channels <= 0:
        raise DecodeError('audio stream has invalid sample rate or channel count')
    bits = int(num(s.get('bits_per_raw_sample')) or 0) or int(num(s.get('bits_per_sample')) or 0) or None
    codec = s.get('codec_name')
    return {'format': (fmt.get('format_name') or '').split(',')[0] or None, 'codec': codec,
            'sample_rate': rate, 'bit_depth': bits, 'channels': channels,
            'duration_s': num(s.get('duration')) or num(fmt.get('duration')),
            'lossy_source': (codec or '') in LOSSY_CODECS}


def stream(path, channels, warnings=None, chunk_frames=CHUNK_FRAMES):
    """Yield float32 arrays (frames, channels) at the native rate; channels is 1 or 2.

    Memory is bounded by chunk_frames. Raises DecodeError when nothing could be
    decoded; if some audio was decoded before an error, a message is appended to
    ``warnings`` instead (truncated/corrupt tail).
    """
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', _url(path), '-map', '0:a:0', '-vn',
           '-ac', str(channels), '-f', 'f32le', '-acodec', 'pcm_f32le', '-']
    try:
        # stderr to a temp file avoids a pipe deadlock while we read stdout.
        import tempfile
        errf = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, stdin=subprocess.DEVNULL)
    except OSError as exc:
        raise DecodeError(f'cannot run ffmpeg: {exc}') from exc
    size = chunk_frames * channels * 4
    total = 0
    try:
        carry = b''
        while True:
            buf = proc.stdout.read(size)
            if not buf:
                break
            buf = carry + buf
            usable = len(buf) // (4 * channels) * 4 * channels
            carry = buf[usable:]
            if usable:
                total += usable // (4 * channels)
                yield np.frombuffer(buf[:usable], dtype='<f4').reshape(-1, channels)
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        errf.seek(0)
        err = errf.read().decode(errors='replace').strip()
        errf.close()
    if total == 0:
        raise DecodeError('no audio could be decoded (corrupt or empty file)'
                          + (': ' + err[:300] if err else ''))
    if (proc.returncode or err) and warnings is not None:
        warnings.append('decoder reported problems (possibly corrupt/truncated input): '
                        + (err[:300] or f'ffmpeg exit {proc.returncode}'))
