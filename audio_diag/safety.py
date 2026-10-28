"""Output-path safeguards: never overwrite the input, never write under the music library."""
import os

PROTECTED_DIRS = ('/home/cancio/Música',)


class OutputRefused(ValueError):
    """The requested output path is unsafe."""


def _under(path, root):
    root = os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check_output_path(output, input_path):
    """Raise OutputRefused if writing ``output`` could damage ``input_path`` or
    would land inside a protected directory. Handles symlinks (both resolved),
    hardlinks (same device/inode) and dangling links. Returns the resolved path."""
    out = os.fspath(output)
    real_out = os.path.realpath(out)
    real_in = os.path.realpath(os.fspath(input_path))
    if os.path.isdir(real_out):
        raise OutputRefused(f'output path is a directory: {out}')
    if real_out == real_in:
        raise OutputRefused(f'refusing to overwrite the input file: {out}')
    for protected in PROTECTED_DIRS:
        for candidate in (real_out, os.path.abspath(out)):
            if _under(candidate, protected) or _under(os.path.normcase(candidate), protected):
                raise OutputRefused(f'refusing to write under protected directory {protected}: {out}')
    if os.path.lexists(out):
        try:
            if os.path.samefile(out, input_path):
                raise OutputRefused(f'output is the same file as the input (hardlink/symlink): {out}')
        except FileNotFoundError:
            pass
    # Check every existing hardlink alias of input is not the output's target
    parent = os.path.dirname(real_out) or '.'
    if not os.path.isdir(parent):
        raise OutputRefused(f'output directory does not exist: {parent}')
    return real_out


def write_text_atomic(output, text, input_path):
    """Write text to ``output`` (resolved through symlinks) via temp file + rename,
    after re-validating the path. The input file is never opened for writing."""
    real_out = check_output_path(output, input_path)
    parent = os.path.dirname(real_out)
    tmp = os.path.join(parent, f'.{os.path.basename(real_out)}.{os.getpid()}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, 0o644)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        check_output_path(real_out, input_path)
        os.replace(tmp, real_out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
