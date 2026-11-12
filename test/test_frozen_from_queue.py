"""The functions that moved here from the KC-remapping experiment queue (hong2p.rdm,
hong2p.stats, hong2p.imaging, hong2p.viz.oklab_diverging_cmap; 2026-11-12) give the
outputs the queue's own code gave before the move, on the same inputs.

frozen_from_queue/hong2p.npz holds small inputs cut from real data (a model's spike counts,
Yang's KC recordings, ...) and every case's output, frozen from the queue's code
(model_experiment_queue's test/frozen_step3/). The cases below call the moved functions
with the queue's parameters: the 16-odor APL panel's order, its odor groups, odor labels
abbreviated as the queue does ('2h @ -3' -> '2h'). Exact in the environment the outputs were
frozen in, else to a relative 1e-9 (`frozen_outputs.check`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hong2p import imaging, rdm, stats
from .frozen_outputs import check, load, objects

FROZEN = Path(__file__).resolve().parent / 'frozen_from_queue' / 'hong2p.npz'
RATE = 59.273 / 20
# the queue's chemical groups of the APL16 odors (its block analysis)
GROUPS = {'2h': 'fruity', 'IaA': 'fruity', 'pa': 'fruity', '2-but': 'small_ester',
    'eb': 'small_ester', 'ep': 'small_ester', '6al': 'ald', 't2h': 'ald', '1-8ol': 'alc',
    '1-5ol': 'alc', '1-6ol': 'alc', 'ms': 'ms', 'paa': 'paa', '-aPine': 'terp',
    'euc': 'terp', 'B-myr': 'terp',
}


def ab(df: pd.DataFrame) -> pd.DataFrame:
    """The queue's abbreviation of odor columns ('1-5ol @ -3' -> '1-5ol')."""
    out = df.copy()
    out.columns = pd.Index([str(c).split(' @ ')[0] for c in df.columns], name='odor')
    return out


def cases(O):
    order = O['odors']
    megamat12 = order[:12]

    def pfs():
        s = ab(O['sq_ctrl'])
        s.index = s.columns
        return rdm.pairs_from_square(s, order)

    def dist(df, *args, **kw):
        return rdm.pair_distances(ab(df), *args, order=order, **kw)

    def model():
        return dist(O['counts_on']), dist(O['counts_off'])

    def reversed_pairs():
        p = pfs()
        return pd.Series(p.to_numpy()[::-1],
            index=pd.MultiIndex.from_tuples([(b, a) for a, b in p.index[::-1]])
        )

    def with_nan():
        p = pfs()
        a = p.copy()
        a.iloc[[3, 50]] = np.nan
        b = rdm.subset_pairs(p, megamat12 + ['paa']) * 0.5
        return a, b

    def boot_fn(m):
        return {'mean_on': float(m['TNT3c'].mean()), 'rho': stats.spearman(m['TNT3c'], m['TNT'])}

    groups = {'TNT3c': O['rec_ctrl'], 'TNT': O['rec_tnt']}
    ex = lambda: imaging.trace_excerpts(O['traces'], O['onsets'], 15, 60)
    t = np.arange(-15, 119) / RATE
    u = np.linspace(-1, 30, 311)
    rd = ((1.3, 20.0), (0.0, np.inf), (0.0, 5.0), (0.7, np.inf))
    return {
        'odor_order': lambda: rdm.order_items(['B-myr', 'zz', '2h', 'aa', 'pa', 'euc'], order),
        'pair_list': lambda: rdm.pair_list(['ms', '2h', 'zz', 'IaA', 'aa'], order),
        'pairs_from_square': pfs,
        'square_from_pairs': lambda: rdm.square_from_pairs(pfs(), order),
        'square_from_pairs_subset': lambda: rdm.square_from_pairs(rdm.subset_pairs(pfs(),
            ['eb', '2h', 'paa', 'ms']), order),
        'canonical_pairs': lambda: rdm.canonical_pairs(reversed_pairs(), order),
        'align_pairs': lambda: rdm.align_pairs(*with_nan()),
        'subset_pairs': lambda: rdm.subset_pairs(pfs(), megamat12),
        'pair_distances_cosine': lambda: dist(O['counts_on']),
        'pair_distances_cosine_off': lambda: dist(O['counts_off']),
        'pair_distances_pearson': lambda: dist(O['orn'], measure='pearson'),
        'pair_distances_pearson_nosilent': lambda: dist(O['counts_on'], 'pearson',
            drop_silent=True),
        'pair_distances_tm': lambda: dist(O['tm_mat']),
        'block_label': lambda: [rdm.block_label(a, b, GROUPS) for a, b in (('2h', 'IaA'),
            ('2h', 'eb'), ('ms', 'paa'), ('euc', '-aPine'), ('zz', 'ms'))],
        'block_means': lambda: rdm.block_means(pfs(), GROUPS),
        'per_odor_means': lambda: rdm.per_odor_means(pfs(), order),
        'pair_design': lambda: rdm.pair_design(pfs().index),
        'pair_design_odors': lambda: rdm.pair_design(rdm.subset_pairs(dist(O['counts_on']),
            ['2h', 'eb', 'ms']).index, ['ms', 'eb', '2h', 'paa']),
        'odor_shares': lambda: rdm.odor_shares(O['data_effect']),
        'density_null_similarity': lambda: rdm.density_null_similarity(ab(O['tm_mat']), order),
        'shuffled_pair_distances': lambda: rdm.shuffled_pair_distances(ab(O['tm_mat']),
            np.random.default_rng(0), order=order, thresh=1.0),
        'shuffled_pair_distances_thr': lambda: rdm.shuffled_pair_distances(ab(O['tm_mat']),
            np.random.default_rng(1), measure='pearson', order=order, values='thresholded',
            cells='responsive', thresh=0.8),
        'transform_matrix': lambda: [rdm.transform_matrix(O['tm_mat'].iloc[:60], values=v,
            cells=cl, thresh=1.0) for v in ('graded', 'binary', 'thresholded')
            for cl in ('all', 'responsive')],
        'transform_matrix_thresh': lambda: rdm.transform_matrix(O['tm_mat'], values='binary',
            thresh=0.75),
        'linear_resid': lambda: stats.linear_resid(*model()),
        'isotonic_resid': lambda: stats.isotonic_resid(*model()),
        'spearman_pearson': lambda: [stats.spearman(*model()), stats.pearson(*model()),
            stats.spearman(*with_nan()), stats.pearson(*with_nan())],
        'resample_recording_means': lambda: stats.resample_group_means(groups,
            np.random.default_rng(1)),
        'resample_recording_means_fly': lambda: stats.resample_group_means(groups,
            np.random.default_rng(2), cluster=('date', 'fly_num')),
        'bootstrap_recordings': lambda: stats.bootstrap_groups(boot_fn, groups, n_boot=30),
        'bootstrap_recordings_fly': lambda: stats.bootstrap_groups(boot_fn, groups, n_boot=25,
            seed=3, ci=(10, 90), cluster=('date', 'fly_num')),
        'split_half_reliability': lambda: stats.split_half_reliability(O['rec_ctrl'],
            n_splits=40),
        'split_half_reliability_pearson': lambda: stats.split_half_reliability(O['rec_tnt'],
            n_splits=30, seed=4, method='pearson'),
        'oklab_diverging_cmap': lambda: _viz().oklab_diverging_cmap(),
        'oklab_diverging_cmap_args': lambda: _viz().oklab_diverging_cmap(11, l_end=0.5,
            c_end=0.2, hue_neg=200.0, hue_pos=40.0),
        'window_volumes': lambda: [imaging.window_volumes(w, RATE)
            for w in ((2, 8), (2.0, 8.1), (0, 2), (-2, -1))],
        'trace_excerpts': ex,
        'windowed_responses': lambda: imaging.windowed_responses(ex(), [(6, 24), (0, 6),
            (-15, 0)], first_volume=-15),
        'windowed_responses_seconds': lambda: imaging.windowed_responses(ex(), [(2.0, 8.1),
            (0.5, 3.0)], first_volume=-15, volume_rate=RATE),
        'indicator_kernel': lambda: [imaging.indicator_kernel(u, r, d) for r, d in rd],
        'indicator_kernel_integral': lambda: [imaging.indicator_kernel_integral(u, r, d)
            for r, d in rd],
        'kernel_signal': lambda: [imaging.kernel_signal(t, 0.8, 10.0, 'onset'),
            imaging.kernel_signal(t, 0.8, 10.0, 'steady', pulse=2.0),
            imaging.kernel_signal(t, 0.8, 10.0, 'histogram', edges=[0.0, 1.0, 2.5],
                weights=[2.0, 1.0])],
        'fit_indicator_kernel': lambda: imaging.fit_indicator_kernel(O['synth_t'],
            O['synth_y'], 'onset'),
        'fit_indicator_kernel_steady': lambda: imaging.fit_indicator_kernel(O['synth_t'],
            O['synth_y'], 'steady', pulse=2.0),
        'fit_indicator_kernel_fixed': lambda: imaging.fit_indicator_kernel(O['synth_t'],
            O['synth_y'], 'onset', fixed=(1.0, 8.0)),
        'trial_values': lambda: [imaging.trial_values(O['trials_ctrl'][:, :40], v, 1.0)
            for v in ('graded', 'thresholded', 'binary')]
            + [imaging.trial_values(O['trials_ctrl'][:, :40], 'thresholded', thresh=0.5)],
        'trial_cosines': lambda: imaging.trial_cosines(O['trials_ctrl']),
        'noise_corrected_pair_distances': lambda: imaging.noise_corrected_pair_distances(
            O['trials_ctrl'], order, order=order),
        'noise_corrected_pair_distances_rel': lambda: imaging.noise_corrected_pair_distances(
            imaging.trial_values(O['trials_tnt'], 'thresholded', 1.0), order,
            min_reliability=0.3, order=order),
        'noise_free_pair_distances': lambda: [imaging.noise_free_pair_distances(
            O['trials_ctrl'], order, repeats=r, values=v, thresh=1.0, order=order)
            for r in ('different', 'adjacent', 'first_third')
            for v in ('graded', 'thresholded')],
        'noise_free_pair_distances_binary': lambda: imaging.noise_free_pair_distances(
            O['trials_tnt'], order, values='binary', thresh=0.5, order=order),
    }


SOLVER = {'fit_indicator_kernel', 'fit_indicator_kernel_steady'}


def _viz():
    return pytest.importorskip('hong2p.viz')


_STATE = {}


def _frozen():
    if not _STATE:
        fz = load(FROZEN)
        _STATE.update(fz=fz, O=objects(fz['in']))
    return _STATE['fz'], _STATE['O']


def _case_names():
    fz, O = _frozen()
    return sorted(cases(O))


@pytest.mark.parametrize('case', _case_names())
def test_frozen_case(case):
    fz, O = _frozen()
    check(fz, case, cases(O)[case](), solver=case in SOLVER)


def test_every_frozen_case_covered():
    fz, O = _frozen()
    frozen_cases = {k.split('/')[0] for k in fz['out']}
    assert frozen_cases == set(cases(O))
