"""Small shared helpers: dB conversion and strict-JSON sanitising."""
import math

import numpy as np


def db10(x, floor=None):
    """10*log10(x) as float; None when x is not a positive finite number
    (or ``floor`` substituted for non-positive values when given)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x <= 0:
        return floor
    return 10.0 * math.log10(x)


def db20(x):
    v = db10(None if x is None else float(x) ** 2)
    return v


def sanitize(obj, _key=None):
    """Return a JSON-strict copy: numpy scalars/arrays converted, NaN/inf -> None.

    When a dict value is replaced because it was non-finite, a sibling
    ``<key>_reason`` is added (unless one exists)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k = str(k)
            clean = sanitize(v, k)
            out[k] = clean
            if clean is None and _nonfinite(v) and (k + '_reason') not in obj:
                out[k + '_reason'] = 'non-finite value (undefined for this input)'
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    return str(obj)


def _nonfinite(v):
    if isinstance(v, np.generic):
        v = v.item()
    return isinstance(v, float) and not math.isfinite(v)
