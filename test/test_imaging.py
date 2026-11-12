"""Tests of hong2p.imaging on synthetic traces and recordings (moved with the functions from
the KC-remapping experiment queue's test_e84_trace_windows.py, test_e81_repeat_structure.py
and test_readout_low_end.py, 2026-11-12): the response window reproduces a z-scored
per-trial response, the indicator kernel and its fits recover known kernels and drives, and
the noise-corrected and noise-free distances recover a recording's noise-free distances."""
import numpy as np
import pandas as pd
import pytest

from hong2p import imaging, rdm

# a volume rate (59.273 frames / s, 20 frames per volume) and the baseline / window volume
# counts at it
RATE = 59.273 / 20
B, S, P = round(5 * RATE), round(2 * RATE), round(8 * RATE)
ODORS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', '6al', 't2h', '1-8ol', '1-5ol', '1-6ol',
    'ms', 'paa', '-aPine', 'euc', 'B-myr'
]
# numpy >= 2 renamed trapz (numpy 1.x, for the old Python 3.8 environment, has only trapz)
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _session(seed=0, n_units=40, n_trials=12, gap=139, n_before=15, n_after=119):
    """Synthetic traces (units, volumes) with a slow drift, noise and responses after each
    onset, and the onsets."""
    rng = np.random.default_rng(seed)
    T = n_before + n_trials * gap + n_after
    onsets = n_before + 5 + gap * np.arange(n_trials)
    t = np.arange(T)
    x = (100 + 20 * rng.random((n_units, 1)) +
        0.002 * t[None] * rng.standard_normal((n_units, 1))
    )
    x = x + rng.standard_normal((n_units, T))
    for n in onsets:
        amp = rng.gamma(1.0, 2.0, (n_units, 1)) * (rng.random((n_units, 1)) < 0.3)
        u = (t - n) / RATE
        x = x + amp * imaging.indicator_kernel(u, 1.0, 10.0)[None]
    return x, onsets


def test_window_volumes():
    assert imaging.window_volumes((2, 8), RATE) == (S, P) == (6, 24)
    assert imaging.window_volumes((2.0, 8.1), RATE) == (6, 24)
    assert imaging.window_volumes((0, 2), RATE) == (0, 6)
    assert imaging.window_volumes((-2, -1), RATE) == (-6, -3)
    with pytest.raises(ValueError):
        imaging.window_volumes((1.0, 1.1), RATE)


def test_target_window_reproduces_zscored_responses():
    from scipy.stats import zscore

    x, onsets = _session()
    z = zscore(x, axis=1)
    want = np.stack([z[:, n + S:n + P].mean(axis=1) - z[:, n - B:n].mean(axis=1)
        for n in onsets], axis=1
    )
    ex = imaging.trace_excerpts(z, onsets, B, 119)
    assert ex.shape == (len(onsets), x.shape[0], B + 119)
    assert np.allclose(ex[:, :, :B].mean(axis=2), 0, atol=1e-12)
    ours = imaging.windowed_responses(ex, [(S, P)], first_volume=-B)[..., 0].T
    assert np.max(np.abs(ours - want)) < 1e-12
    # the same window given in seconds, and the excerpts of the unscaled trace divided by
    # the whole-trace SD
    sec = imaging.windowed_responses(ex, [(2.0, 8.1)], first_volume=-B,
        volume_rate=RATE)[..., 0].T
    assert np.array_equal(sec, ours)
    exr = imaging.trace_excerpts(x, onsets, B, 119) / x.std(axis=1)[None, :, None]
    assert np.max(np.abs(imaging.windowed_responses(exr, [(S, P)], first_volume=-B
        )[..., 0].T - want)) < 1e-10
    # several windows at once, and windows outside the excerpt refused
    two = imaging.windowed_responses(ex, [(0, 6), (S, P)], first_volume=-B)
    assert two.shape == (len(onsets), x.shape[0], 2) and np.array_equal(two[..., 1].T, ours)
    with pytest.raises(ValueError):
        imaging.windowed_responses(ex, [(-20, 0)], first_volume=-B)
    with pytest.raises(ValueError):
        imaging.trace_excerpts(x, [3], B, 119)


def test_noise_free_of_windowed_trials():
    """The noise-free distances of windowed excerpts are those of the per-trial values."""
    rng = np.random.default_rng(3)
    n_units, n_rep = 300, 3
    signal = np.where(rng.random((n_units, len(ODORS))) < 0.15,
        rng.gamma(2.0, 1.5, (n_units, len(ODORS))), 0.0
    )
    ex = rng.standard_normal((n_rep, len(ODORS), n_units, B + 30)) * 0.5
    ex[..., B + S:B + 24] += signal.T[None, :, :, None]
    ex = ex - ex[..., :B].mean(axis=-1, keepdims=True)
    w = imaging.windowed_responses(ex, [(S, 24)], first_volume=-B)[..., 0]
    trials = w.transpose(0, 2, 1)
    direct = np.stack([[ex[r, o, :, B + S:B + 24].mean(axis=-1) for o in range(len(ODORS))]
        for r in range(n_rep)]).transpose(0, 2, 1)
    assert np.allclose(trials, direct, atol=1e-12)
    a = imaging.noise_free_pair_distances(trials, ODORS, order=ODORS)
    b = imaging.noise_free_pair_distances(direct, ODORS, order=ODORS)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)
    assert list(a.index) == rdm.pair_list(ODORS, ODORS)


def test_kernel_functions():
    u = np.linspace(-1, 30, 6201)
    k = imaging.indicator_kernel(u, 1.3, 20.0)
    assert np.all(k[u <= 0] == 0) and k.max() < 1
    # the integral against a numerical one, and the limits
    K = imaging.indicator_kernel_integral(u, 1.3, 20.0)
    num = np.concatenate([[0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(u))])
    assert np.max(np.abs(K - num)) < 1e-5
    assert np.allclose(imaging.indicator_kernel(u, 0.0, np.inf), (u > 0).astype(float))
    assert np.allclose(imaging.indicator_kernel_integral(u, 0.0, np.inf), np.maximum(u, 0))
    # the onset drive is the kernel; the steady drive a box convolution; a one-bin
    # histogram the steady drive
    t = np.arange(-15, 119) / RATE
    assert np.allclose(imaging.kernel_signal(t, 1.3, 20.0, 'onset'),
        imaging.indicator_kernel(t, 1.3, 20.0)
    )
    s = np.linspace(0, 2, 4001)
    box = np.array([_trapz(imaging.indicator_kernel(tt - s, 0.8, 10.0), s) / 2.0 for tt in t])
    st = imaging.kernel_signal(t, 0.8, 10.0, 'steady', pulse=2.0)
    assert np.max(np.abs(st - box)) < 1e-6
    hist = imaging.kernel_signal(t, 0.8, 10.0, 'histogram', edges=[0.0, 2.0], weights=[5.0])
    assert np.allclose(hist, st)
    two = imaging.kernel_signal(t, 0.8, 10.0, 'histogram', edges=[0.0, 1.0, 2.0],
        weights=[1.0, 1.0]
    )
    assert np.allclose(two, st)
    with pytest.raises(ValueError):
        imaging.kernel_signal(t, 1.0, 5.0, 'ramp')


@pytest.mark.parametrize('drive,rise,decay', [('onset', 1.3, 20.0), ('steady', 0.8, 10.0),
    ('onset', 0.5, 6.0)])
def test_fit_recovers_known_kernel(drive, rise, decay):
    rng = np.random.default_rng(7)
    t = np.arange(-15, 119) / RATE
    amp = 4.0
    clean = amp * imaging.kernel_signal(t, rise, decay, drive, pulse=2.0)
    y = clean + 0.01 * amp * rng.standard_normal(len(t))
    f = imaging.fit_indicator_kernel(t, y, drive, pulse=2.0)
    assert f['drive'] == drive and f['n'] == len(t)
    assert abs(f['rise'] / rise - 1) < 0.1
    assert abs(f['decay'] / decay - 1) < 0.1
    assert abs(f['amplitude'] / amp - 1) < 0.05
    assert f['rms'] < 0.02 * amp
    assert not f['rise_at_bound'] and not f['decay_at_bound']
    g = imaging.fit_indicator_kernel(t, y, drive, pulse=2.0, fixed=(rise, decay))
    assert abs(g['amplitude'] / amp - 1) < 0.02 and g['rise'] == rise and g['decay'] == decay


def test_fit_tells_the_drives_apart_without_noise():
    t = np.arange(-15, 119) / RATE
    y_steady = 3.0 * imaging.kernel_signal(t, 0.8, 10.0, 'steady', pulse=2.0)
    y_onset = 3.0 * imaging.kernel_signal(t, 1.3, 20.0, 'onset')
    for y, right, wrong in ((y_steady, 'steady', 'onset'), (y_onset, 'onset', 'steady')):
        good = imaging.fit_indicator_kernel(t, y, right, pulse=2.0)
        bad = imaging.fit_indicator_kernel(t, y, wrong, pulse=2.0)
        assert good['rms'] < 1e-4 and bad['rms'] > 10 * good['rms']
    y_h = 2.0 * imaging.kernel_signal(t, 1.0, 12.0, 'histogram', edges=[0.0, 0.5, 2.0],
        weights=[3.0, 1.0]
    )
    f = imaging.fit_indicator_kernel(t, y_h, 'histogram', edges=[0.0, 0.5, 2.0],
        weights=[3.0, 1.0]
    )
    assert abs(f['rise'] - 1.0) < 1e-3 and abs(f['decay'] - 12.0) < 1e-2


def test_trial_values_and_cosines():
    X = np.array([[[0.5, 2.0], [3.0, -1.0], [0.0, 1.5]]])
    assert np.array_equal(imaging.trial_values(X, 'thresholded', 1.0),
        [[[0, 2.0], [3.0, 0], [0, 1.5]]]
    )
    assert np.array_equal(imaging.trial_values(X, 'binary', 1.0), [[[0, 1.0], [1.0, 0], [0, 1.0]]])
    assert imaging.trial_values(X) is not None
    with pytest.raises(ValueError):
        imaging.trial_values(X, 'binary')
    with pytest.raises(ValueError):
        imaging.trial_values(X, 'clipped', 1.0)
    C = imaging.trial_cosines(X)
    assert C.shape == (1, 2, 1, 2)
    a, b = X[0, :, 0], X[0, :, 1]
    assert np.isclose(C[0, 0, 0, 1], a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    Z = np.zeros((2, 3, 2))
    Z[0, 0, 0] = 1.0
    C = imaging.trial_cosines(Z)
    assert np.isnan(C[1, 0, 0, 0]) and np.isclose(C[0, 0, 0, 0], 1.0)


def _signal(seed=0, n_units=3000, n_odors=16, p_resp=0.12):
    """A KC-like noise-free signal (units x odors): sparse positive responses, odors sharing
    responders through a few latent groups."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, n_odors)
    unit_pref = rng.random((n_units, 4)) ** 6
    p = np.clip(p_resp * unit_pref[:, groups] / unit_pref.mean(), 0, 1)
    resp = rng.random((n_units, n_odors)) < p
    amp = 2.0 + rng.gamma(2.0, 0.6, (n_units, n_odors))
    return np.where(resp, amp, 0.0)


def _recording(m, *, seed=1, n_rep=3, sd_indep=0.6, sd_shared=0.0, sd_gain=0.0):
    rng = np.random.default_rng(seed)
    U, O = m.shape
    g = 1.0 + sd_gain * rng.standard_normal((n_rep, 1, O))
    u = sd_shared * rng.standard_normal((n_rep, U, 1))
    e = sd_indep * rng.standard_normal((n_rep, U, O))
    return g * m[None] + u + e


def test_independent_noise_estimates_recover_true_distances():
    m = _signal(seed=2)
    true = rdm.pair_distances(pd.DataFrame(m, columns=ODORS), order=ODORS)
    X = _recording(m, seed=3, sd_indep=1.0)
    mean = rdm.pair_distances(pd.DataFrame(X.mean(axis=0), columns=ODORS), order=ODORS)
    # the noise inflates the trial-mean distances well beyond the sampling error
    assert (mean - true).mean() > 0.025
    for d in (imaging.noise_corrected_pair_distances(X, ODORS, order=ODORS),
            imaging.noise_free_pair_distances(X, ODORS, order=ODORS),
            imaging.noise_free_pair_distances(X, ODORS, repeats='adjacent', order=ODORS),
            imaging.noise_free_pair_distances(X, ODORS, repeats='first_third', order=ODORS)):
        err = d - true
        assert abs(err.mean()) < 0.01 and np.sqrt((err ** 2).mean()) < 0.03


def test_shared_state_noise_free_unbiased():
    """With a state shared by one unit's odors within a repeat, the correction that assumes
    independent noise makes the odors too alike, and the noise-free estimate does not (with a
    gain per presentation too, which a cosine does not see)."""
    m = _signal(seed=4)
    true = rdm.pair_distances(pd.DataFrame(m, columns=ODORS), order=ODORS)
    X = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5)
    err_nf = imaging.noise_free_pair_distances(X, ODORS, order=ODORS) - true
    err_c = imaging.noise_corrected_pair_distances(X, ODORS, order=ODORS) - true
    assert abs(err_nf.mean()) < 0.01 and np.sqrt((err_nf ** 2).mean()) < 0.03
    assert err_c.mean() < -0.03
    Xg = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5, sd_gain=0.2)
    assert abs((imaging.noise_free_pair_distances(Xg, ODORS, order=ODORS) - true).mean()) < 0.01


def test_noise_corrected_min_reliability():
    rng = np.random.default_rng(10)
    base = _signal(seed=10, n_units=400, n_odors=6)
    base[:, 0] = 0.0
    odors = ODORS[:6]
    trials = base[None] + rng.normal(0, 0.5, (3,) + base.shape)
    d = imaging.noise_corrected_pair_distances(trials, odors, min_reliability=0.1)
    with_odor0 = [odors[0] in p for p in d.index]
    assert d[with_odor0].isna().all() and d[[not w for w in with_odor0]].notna().all()
    assert imaging.noise_corrected_pair_distances(trials, odors).notna().all()
    # without `order`, the pairs are in sorted order
    assert list(d.index) == rdm.pair_list(odors)


def test_nonresponder_trial_sd():
    rng = np.random.default_rng(11)
    recs = [rng.normal(0, 0.4, (3, 200, 5)) + np.where(rng.random((200, 5)) < 0.1, 3.0, 0.0)
        for _ in range(3)
    ]
    v = np.concatenate([X.var(axis=0, ddof=1)[X.mean(axis=0) <= 1.0] for X in recs])
    assert imaging.nonresponder_trial_sd(recs, 1.0) == pytest.approx(np.sqrt(v.mean()),
        rel=1e-14
    )
    assert 0.35 < imaging.nonresponder_trial_sd(recs, 1.0) < 0.45
