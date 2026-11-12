"""Imaging time courses and trial-to-trial noise: per-trial excerpts of traces around odor
onsets and the responses in windows after them, the calcium indicator's kernel (fitted to
mean traces under a drive), and odor-odor distances corrected for the noise between
trials.

Traces are (units, volumes) arrays; trials of one recording are (n_trials, units, odors)
arrays (trial i of every odor is the i-th presentation, or repeat, of that odor).
Distances are 1 - cosine similarity across units, returned as pair Series
(`hong2p.rdm`).

Moved from the KC-remapping experiment queue (model_experiment_queue's kcq.harness),
2026-11-12; the volume rate, the windows and the odors are arguments.
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from hong2p.rdm import PAIR_NAMES, pair_index, pair_list


def window_volumes(window: Tuple[float, float], volume_rate: float) -> Tuple[int, int]:
    """Returns a window (start, end) in seconds from the odor's onset as volumes after the
    onset volume: (round(start * volume_rate), round(end * volume_rate)), end exclusive.

    E.g. at 2.964 volumes / s (2, 8) and (2.0, 8.1) are both volumes 6 to 23.
    """
    a = int(round(float(window[0]) * volume_rate))
    b = int(round(float(window[1]) * volume_rate))
    if not b > a:
        raise ValueError(f'window {window} has no volume at {volume_rate} volumes / s')
    return a, b


def trace_excerpts(traces: np.ndarray, onsets: Sequence[int], n_before: int,
    n_after: int) -> np.ndarray:
    """Returns per-onset excerpts of traces, each with its pre-onset mean subtracted.

    Args:
        traces: (units, volumes).
        onsets: the onset volume of each trial.
        n_before: volumes before the onset (the baseline, whose mean is subtracted).
        n_after: volumes from the onset volume on (the onset volume is the first).

    Returns (trials, units, n_before + n_after) float64; sample j is volume
    onset - n_before + j.
    """
    X = np.asarray(traces, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    n_before, n_after = int(n_before), int(n_after)
    if n_before < 1 or n_after < 1:
        raise ValueError('n_before and n_after must be at least 1')
    if len(on) and (on.min() - n_before < 0 or on.max() + n_after > X.shape[1]):
        raise ValueError(f'excerpts of {n_before} + {n_after} volumes run off a trace of '
            f'{X.shape[1]} volumes'
        )
    out = np.empty((len(on), X.shape[0], n_before + n_after))
    for i, n in enumerate(on):
        seg = X[:, n - n_before:n + n_after]
        out[i] = seg - seg[:, :n_before].mean(axis=1, keepdims=True)
    return out


def windowed_responses(excerpts: np.ndarray, windows: Sequence[Tuple[float, float]], *,
    first_volume: int, volume_rate: Optional[float] = None) -> np.ndarray:
    """Returns the mean of baseline-subtracted excerpts (`trace_excerpts`) in each window.

    Args:
        excerpts: (..., samples), sample j being volume `first_volume` + j relative to
            the onset volume.
        windows: (start, end) pairs, in volumes after the onset volume (end exclusive),
            or in seconds when `volume_rate` is given (`window_volumes`).
        first_volume: the volume of sample 0 relative to the onset volume (-n_before).

    Returns (..., len(windows)). With the window (round(2 v), round(8 v)) on excerpts of a
    z-scored trace, this is the mean z 2 to 8 s after onset minus the pre-onset mean.
    """
    E = np.asarray(excerpts)
    n = E.shape[-1]
    out = []
    for w in windows:
        a, b = (window_volumes(w, volume_rate) if volume_rate is not None
            else (int(w[0]), int(w[1]))
        )
        i, j = a - int(first_volume), b - int(first_volume)
        if i < 0 or j > n or not j > i:
            raise ValueError(f'window {w} (volumes {a} to {b - 1}) is outside the excerpts '
                f'(volumes {first_volume} to {first_volume + n - 1})'
            )
        out.append(E[..., i:j].mean(axis=-1, dtype=np.float64))
    return np.stack(out, axis=-1)


def indicator_kernel(u, rise: float, decay: float) -> np.ndarray:
    """Returns the indicator kernel (1 - exp(-u / rise)) exp(-u / decay) at times `u` since
    a spike (0 for u <= 0; rise 0 is an instant rise, decay np.inf no decay)."""
    u = np.asarray(u, dtype=float)
    pos = u > 0
    up = np.where(pos, u, 0.0)
    k = np.ones_like(up) if rise == 0 else -np.expm1(-up / rise)
    if not np.isinf(decay):
        k = k * np.exp(-up / decay)
    return np.where(pos, k, 0.0)


def indicator_kernel_integral(u, rise: float, decay: float) -> np.ndarray:
    """Returns the integral of `indicator_kernel` from 0 to `u` (0 for u <= 0)."""
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    first = u if np.isinf(decay) else -decay * np.expm1(-u / decay)
    tau = rise if np.isinf(decay) else (rise * decay / (rise + decay) if rise > 0 else 0.0)
    return first if tau == 0 else first + tau * np.expm1(-u / tau)


DRIVES: Tuple[str, ...] = ('onset', 'steady', 'histogram')


def _drive_bins(drive: str, pulse: float, edges, weights
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(edges, weights summing to 1) of a drive, None for the onset impulse."""
    if drive == 'onset':
        return None
    if drive == 'steady':
        if not pulse > 0:
            raise ValueError(f'pulse must be > 0, not {pulse}')
        return np.array([0.0, float(pulse)]), np.array([1.0])
    if drive == 'histogram':
        e = np.asarray(edges, dtype=float)
        w = np.asarray(weights, dtype=float)
        if e.ndim != 1 or len(e) != len(w) + 1 or np.any(np.diff(e) <= 0):
            raise ValueError('a histogram drive needs increasing edges, one more than its '
                'weights'
            )
        if np.any(w < 0) or not w.sum() > 0:
            raise ValueError('a histogram drive needs non-negative weights with a positive '
                'sum'
            )
        return e, w / w.sum()
    raise ValueError(f'drive must be one of {DRIVES}, not {drive!r}')


def kernel_signal(t, rise: float, decay: float, drive: str = 'onset', *,
    pulse: float = 2.0, edges: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Returns the kernel convolved with a drive of total weight 1, at times `t` (s from
    the odor's onset).

    Args:
        drive: 'onset' (all firing at the onset, an impulse at 0), 'steady' (uniform
            firing from 0 to `pulse`) or 'histogram' (firing uniform within each bin
            [edges[i], edges[i + 1]) with total weight weights[i]; the weights are divided
            by their sum).

    Returns float array shaped like `t`.
    """
    t = np.asarray(t, dtype=float)
    bins = _drive_bins(drive, pulse, edges, weights)
    if bins is None:
        return indicator_kernel(t, rise, decay)
    e, w = bins
    keep = w > 0
    lo, hi, w = e[:-1][keep], e[1:][keep], w[keep]
    tt = t.reshape(-1, 1)
    K = (indicator_kernel_integral(tt - lo, rise, decay) -
        indicator_kernel_integral(tt - hi, rise, decay)) / (hi - lo)
    return (K @ w).reshape(t.shape)


def fit_indicator_kernel(t, y, drive: str = 'onset', *, pulse: float = 2.0, edges=None,
    weights=None, rise_bounds: Tuple[float, float] = (0.05, 10.0),
    decay_bounds: Tuple[float, float] = (0.5, 200.0),
    fixed: Optional[Tuple[float, float]] = None) -> Dict[str, object]:
    """Returns the least-squares fit of amplitude * `kernel_signal(t, rise, decay, drive)`
    to a mean trace `y`.

    The amplitude is solved linearly for each (rise, decay) (and kept >= 0); rise and
    decay are fitted on a log scale within their bounds from several starts. With
    `fixed` = (rise, decay) only the amplitude is fitted (e.g. a kernel fitted on one
    trace tested on another).

    Returns dict: 'rise', 'decay', 'amplitude', 'rms' (root mean squared residual, in y's
    units), 'n', 'rise_at_bound', 'decay_at_bound' (within 1 % of a bound), 'drive'.
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 4:
        raise ValueError('need at least 4 finite points')

    def amp_res(rise, decay):
        s = kernel_signal(t, rise, decay, drive, pulse=pulse, edges=edges, weights=weights)
        ss = float(s @ s)
        a = max(float(s @ y) / ss, 0.0) if ss > 0 else 0.0
        return a, y - a * s

    if fixed is not None:
        r, d = float(fixed[0]), float(fixed[1])
        a, res = amp_res(r, d)
        return dict(rise=r, decay=d, amplitude=a, rms=float(np.sqrt(np.mean(res ** 2))),
            n=int(len(t)), rise_at_bound=False, decay_at_bound=False, drive=drive
        )

    lb = np.log([rise_bounds[0], decay_bounds[0]])
    ub = np.log([rise_bounds[1], decay_bounds[1]])

    def fun(p):
        return amp_res(np.exp(p[0]), np.exp(p[1]))[1]

    best = None
    for r0 in (0.2, 0.7, 2.0):
        for d0 in (2.0, 8.0, 30.0):
            x0 = np.clip(np.log([r0, d0]), lb + 1e-9, ub - 1e-9)
            sol = least_squares(fun, x0, bounds=(lb, ub), x_scale=1.0, xtol=1e-10,
                ftol=1e-12, gtol=1e-12
            )
            if best is None or sol.cost < best.cost:
                best = sol

    r, d = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    a, res = amp_res(r, d)

    def at(v, b):
        return bool(v <= b[0] * 1.01 or v >= b[1] / 1.01)

    return dict(rise=r, decay=d, amplitude=a, rms=float(np.sqrt(np.mean(res ** 2))),
        n=int(len(t)), rise_at_bound=at(r, rise_bounds), decay_at_bound=at(d, decay_bounds),
        drive=drive
    )


TRIAL_VALUES: Tuple[str, ...] = ('graded', 'thresholded', 'binary')


def trial_values(trials: np.ndarray, values: str = 'graded',
    thresh: Optional[float] = None) -> np.ndarray:
    """Returns single trials read as `values`: 'graded' (unchanged), 'thresholded' (values
    at or below `thresh` set to 0, trial by trial) or 'binary' (1 above `thresh`, else 0).
    Works on any array shape; `thresh` is needed for the last two."""
    X = np.asarray(trials, dtype=float)
    if values == 'graded':
        return X
    if values not in TRIAL_VALUES:
        raise ValueError(f'values must be one of {TRIAL_VALUES}, not {values!r}')
    if thresh is None:
        raise ValueError(f'values={values!r} needs a threshold')
    if values == 'thresholded':
        return np.where(X > thresh, X, 0.0)
    return (X > thresh).astype(float)


def trial_cosines(trials: np.ndarray) -> np.ndarray:
    """Returns the cosine similarities between every pair of single-trial response vectors
    of one recording.

    Args:
        trials: (n_trials, units, odors).

    Returns (n_trials, odors, n_trials, odors): [i, a, j, b] is the cosine between the
    units' responses to odor a in trial i and to odor b in trial j; NaN where either
    vector is all zero.
    """
    X = np.asarray(trials, dtype=float)
    R, U, O = X.shape
    Y = X.transpose(0, 2, 1).reshape(R * O, U)
    G = Y @ Y.T
    n = np.sqrt(np.diag(G))
    with np.errstate(invalid='ignore', divide='ignore'):
        C = G / np.outer(n, n)
    C[(n == 0)[:, None] | (n == 0)[None, :]] = np.nan
    return C.reshape(R, O, R, O)


REPEAT_PAIRS: Tuple[str, ...] = ('same_repeat', 'different_repeat', 'adjacent',
    'first_third'
)


def repeat_pair_masks(n_rep: int) -> Dict[str, np.ndarray]:
    """Returns {kind: (n_rep, n_rep) bool} of the ordered trial pairs (i, j) of each kind
    (`REPEAT_PAIRS`): the same trial, different trials, trials one apart, two apart."""
    i, j = np.meshgrid(np.arange(n_rep), np.arange(n_rep), indexing='ij')
    return {'same_repeat': i == j, 'different_repeat': i != j,
        'adjacent': np.abs(i - j) == 1, 'first_third': np.abs(i - j) == 2
    }


def mean_over_trial_pairs(C: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns the (odors, odors) mean of `trial_cosines` output C[i, a, j, b] over the
    trial pairs in `mask` (NaN skipped; NaN if none)."""
    if not mask.any():
        return np.full(C.shape[1:4:2], np.nan)
    sel = C.transpose(0, 2, 1, 3)[mask]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(sel, axis=0)


def noise_free_matrix(C: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns the (odors, odors) noise-free distances 1 - C_ab / sqrt(C_aa C_bb) from the
    single-trial cosines `C` (`trial_cosines`) over the trial pairs in `mask` (which must
    exclude i == j)."""
    M = mean_over_trial_pairs(C, mask)
    d = np.diag(M).copy()
    d[~(d > 0)] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1 - M / np.sqrt(np.outer(d, d))


def _pair_series(D: np.ndarray, odors: Sequence, order: Optional[Sequence],
    names: Sequence[str]) -> pd.Series:
    odors = list(odors)
    pl = pair_list(odors, order)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([D[idx[a], idx[b]] for a, b in pl], index=pair_index(pl, names))


def noise_corrected_pair_distances(trials: np.ndarray, odors: Sequence,
    min_reliability: Optional[float] = None, order: Optional[Sequence] = None, *,
    names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the noise-corrected cosine distances of one recording from its trials.

    Each odor's signal power is the squared length of its trial mean minus the
    trial-to-trial variance summed over units divided by the number of trials (floored at
    1e-12): independent noise adds that much to the squared length of a mean and nothing,
    in expectation, to the dot product of two odors' means. The corrected distance is
    1 - (dot product of the trial means) / sqrt(signal power a x signal power b).

    Args:
        trials: (n_trials, units, odors).
        odors: the odors of the last axis.
        min_reliability: pairs with an odor whose signal power is less than this share of
            its trial mean's squared length are NaN (the correction divides by a near-zero
            signal there).
        order: canonical order of the odors (`hong2p.rdm.order_items`).

    Returns a canonical pair Series.
    """
    X = np.asarray(trials, dtype=float)
    n = X.shape[0]
    xbar = X.mean(axis=0)
    s2 = X.var(axis=0, ddof=1).sum(axis=0)
    norm2 = (xbar ** 2).sum(axis=0)
    sig = np.maximum(norm2 - s2 / n, 1e-12)
    dots = xbar.T @ xbar
    D = 1 - dots / np.sqrt(np.outer(sig, sig))
    if min_reliability is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            bad = ~(sig / norm2 >= min_reliability)
        D[bad, :] = np.nan
        D[:, bad] = np.nan
    return _pair_series(D, odors, order, names)


def noise_free_pair_distances(trials: np.ndarray, odors: Sequence, *,
    repeats: str = 'different', values: str = 'graded', thresh: Optional[float] = None,
    order: Optional[Sequence] = None, names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the noise-free cosine distances of one recording, from products of single
    trials of different repeats.

    For each pair (a, b): 1 - C_ab / sqrt(C_aa C_bb), C_ab the mean single-trial cosine
    between odor a in trial i and odor b in trial j over the trial pairs i != j selected
    by `repeats`, C_aa the same for odor a with itself. Noise independent between trials
    adds nothing, in expectation, to such a product, whether or not it is shared by the
    odors of a trial; a drift common to an odor's trials acts on C_ab and C_aa alike and
    cancels in the ratio. Not clipped: a near-identical pair can come out slightly below
    0. (Cosines rather than dot products, so a gain of a whole presentation, which a
    cosine does not see, does not weigh a trial in.)

    Args:
        trials: (n_trials, units, odors).
        odors: the odors of the last axis.
        repeats: 'different' (every i != j), 'adjacent' (|i - j| == 1) or 'first_third'
            (|i - j| == 2).
        values, thresh: how the single trials are read (`trial_values`).
        order: canonical order of the odors (`hong2p.rdm.order_items`).

    Returns a canonical pair Series.
    """
    kind = {'different': 'different_repeat', 'adjacent': 'adjacent',
        'first_third': 'first_third'
    }.get(repeats)
    if kind is None:
        raise ValueError("repeats must be 'different', 'adjacent' or 'first_third', not "
            f'{repeats!r}'
        )
    X = trial_values(trials, values, thresh)
    D = noise_free_matrix(trial_cosines(X), repeat_pair_masks(X.shape[0])[kind])
    return _pair_series(D, odors, order, names)


def nonresponder_trial_sd(recordings: Iterable[np.ndarray], thresh: float) -> float:
    """Returns the trial-to-trial SD of the non-responding entries, pooled over recordings:
    a measure of the measurement noise.

    Args:
        recordings: (n_trials, units, odors) arrays, one per recording.
        thresh: an entry (unit, odor) is non-responding when its trial mean is at or below
            this.

    The root of the mean, over the non-responding entries of all recordings, of their
    variance across trials (ddof 1).
    """
    var = []
    for X in recordings:
        v = X.var(axis=0, ddof=1)
        var.append(v[X.mean(axis=0) <= thresh])
    return float(np.sqrt(np.concatenate(var).mean()))
