"""Independent finite-window manipulations, not analyzer-derived synthesis."""
import numpy as np


def invert_right_low(samples, rate, cutoff=120):
    """Flip R bins below cutoff, preserve L and other R bins exactly.

    Whole-excerpt rectangular DFT implies periodic boundaries and ringing at the
    cutoff/endpoints. This is a controlled test, not a production crossover.
    """
    output = np.array(samples, dtype=float, copy=True)
    spectrum = np.fft.rfft(output[:, 1])
    spectrum[np.fft.rfftfreq(len(output), 1 / rate) < cutoff] *= -1
    output[:, 1] = np.fft.irfft(spectrum, n=len(output))
    return output


def localized_mute_checks(clean, muted, rate, drops):
    """Require loss in the edited interior, unchanged exterior and localized hint."""
    inner = slice(round(8.1 * rate), round(11.9 * rate))
    outside = np.r_[0:round(7.9 * rate), round(12.1 * rate):len(clean)]
    return {
        'mute_interior_zero_with_nonzero_baseline': bool(np.mean(clean[inner] ** 2) > 0
                                                        and np.max(np.abs(muted[inner])) == 0),
        'mute_exterior_unchanged': bool(np.array_equal(clean[outside], muted[outside])),
        'mute_localized_heuristic': bool(drops['confidence'] == 'heuristic' and any(
            7.5 <= d['start_s'] <= 8.5 and d['kind'] == 'break' for d in drops['drops'])),
    }
