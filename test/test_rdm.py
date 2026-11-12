"""Tests of hong2p.rdm and hong2p.stats (moved from the KC-remapping experiment queue,
2026-11-12), of `hong2p.util.repo_info` / `git_commits` (from pyolfsysm.mb_model and the
queue) and of `hong2p.viz.oklab_diverging_cmap`: pair order and round trips, agreement with
`corr_triangular` / `invert_corr_triangular`, distances against direct formulas, the
additive odor shares, the density null and the resampling."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hong2p import rdm, stats
from hong2p.util import corr_triangular, git_commits, invert_corr_triangular, repo_info

ODORS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep']


def _square(odors=ODORS, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((len(odors), len(odors)))
    a = a + a.T
    np.fill_diagonal(a, 0.0)
    return pd.DataFrame(a, index=odors, columns=odors)


def test_order_and_pairs():
    assert rdm.order_items(['zz', 'eb', '2h', 'aa'], ODORS) == ['2h', 'eb', 'aa', 'zz']
    assert rdm.order_items(['b', 'a']) == ['a', 'b']
    pl = rdm.pair_list(['eb', 'zz', '2h'], ODORS)
    assert pl == [('2h', 'eb'), ('2h', 'zz'), ('eb', 'zz')]
    assert rdm.pair_list(['c', 'a', 'b']) == list(combinations('abc', 2))


def test_pairs_from_square_round_trip_and_corr_triangular():
    sq = _square()
    p = rdm.pairs_from_square(sq, ODORS)
    assert list(p.index.names) == ['odor_a', 'odor_b']
    assert list(p.index) == rdm.pair_list(ODORS, ODORS)
    assert all(p[(a, b)] == sq.loc[a, b] for a, b in p.index)
    back = rdm.square_from_pairs(p, ODORS)
    assert back.equals(sq.loc[ODORS, ODORS])
    # without an order: corr_triangular's sorted pairs, with this module's names
    srt = rdm.pairs_from_square(sq)
    ct = corr_triangular(sq.rename_axis('odor').rename_axis('odor', axis=1))
    assert list(srt.index) == list(ct.index) and np.array_equal(srt.to_numpy(), ct.to_numpy())
    # invert_corr_triangular builds its square with square_from_pairs (diag 1)
    inv = invert_corr_triangular(ct)
    assert np.array_equal(inv.to_numpy(), rdm.square_from_pairs(ct, list(inv.index),
        diag=1.0).to_numpy())
    with pytest.raises(ValueError):
        rdm.pairs_from_square(sq.iloc[:, ::-1])


def test_square_from_pairs_partial_and_canonical():
    sq = _square()
    p = rdm.pairs_from_square(sq, ODORS)
    part = rdm.subset_pairs(p, ['2h', 'eb', 'pa'])
    assert len(part) == 3
    s = rdm.square_from_pairs(part.iloc[:2], ODORS, diag=1.0, missing=np.nan)
    assert list(s.index) == ['2h', 'pa', 'eb'] and s.loc['2h', '2h'] == 1.0
    assert np.isnan(s.loc['pa', 'eb']) and s.loc['eb', '2h'] == p[('2h', 'eb')]
    rev = pd.Series(p.to_numpy()[::-1], index=pd.MultiIndex.from_tuples(
        [(b, a) for a, b in p.index[::-1]]))
    assert rdm.canonical_pairs(rev, ODORS).equals(p)
    with pytest.raises(KeyError):
        rdm.canonical_pairs(rev.iloc[1:], ODORS)


def test_align_pairs():
    sq = _square()
    p = rdm.pairs_from_square(sq, ODORS)
    a = p.copy()
    a.iloc[0] = np.nan
    b = rdm.subset_pairs(p, ODORS[:4])
    x, y = rdm.align_pairs(a, b)
    assert len(x) == len(y) == 5 and list(x.index) == list(y.index)


def test_pair_distances():
    rng = np.random.default_rng(1)
    m = pd.DataFrame(rng.gamma(1.0, 1.0, (50, len(ODORS))), columns=ODORS)
    m.iloc[:, 2] = 0.0
    d = rdm.pair_distances(m, order=ODORS)
    X = m.to_numpy()
    a, b = X[:, 0], X[:, 1]
    assert np.isclose(d[('2h', 'IaA')], 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    assert d[[('pa' in p) for p in d.index]].isna().all()
    dp = rdm.pair_distances(m, 'pearson', order=ODORS)
    assert np.isclose(dp[('2h', 'IaA')], 1 - np.corrcoef(a, b)[0, 1])
    m.iloc[:, 2] = rng.random(50)
    m.iloc[:10] = 0.0
    dn = rdm.pair_distances(m, 'pearson', drop_silent=True, order=ODORS)
    assert np.isclose(dn[('2h', 'IaA')], 1 - np.corrcoef(a[10:], b[10:])[0, 1])
    with pytest.raises(ValueError):
        rdm.pair_distances(m, 'euclidean')


def test_blocks_and_per_odor_means():
    groups = {'2h': 'fruity', 'IaA': 'fruity', 'pa': 'fruity', '2-but': 'small_ester',
        'eb': 'small_ester', 'ep': 'small_ester'}
    assert rdm.block_label('2h', 'IaA', groups) == 'within'
    assert rdm.block_label('eb', 'pa', groups) == 'fruity-small_ester'
    assert rdm.block_label('eb', 'zz', groups) == 'small_ester-zz'
    p = rdm.pairs_from_square(_square(), ODORS)
    bm = rdm.block_means(p, groups)
    assert set(bm.index) == {'within', 'fruity-small_ester'}
    po = rdm.per_odor_means(p, ODORS)
    assert list(po.index) == ODORS
    assert np.isclose(po['2h'], p[[('2h' in q) for q in p.index]].mean())


def test_odor_shares_recovers_additive_effect():
    pairs = pd.MultiIndex.from_tuples(list(combinations(ODORS, 2)), names=rdm.PAIR_NAMES)
    shares = np.array([0.1, 0.2, 0.05, 0.3, 0.0, 0.15])
    effect = pd.Series([shares[ODORS.index(a)] + shares[ODORS.index(b)] for a, b in pairs],
        index=pairs)
    s, r2 = rdm.odor_shares(effect)
    assert np.allclose(s.reindex(ODORS).to_numpy(), shares)
    assert r2 == pytest.approx(1.0)
    X, odors = rdm.pair_design(pairs)
    assert odors == ODORS and X.shape == (15, 6) and (X.sum(axis=1) == 2).all()


def test_density_null_is_the_shuffles_mean():
    rng = np.random.default_rng(2)
    m = pd.DataFrame((rng.random((300, 4)) < 0.2).astype(float), columns=ODORS[:4])
    null = rdm.density_null_similarity(m)
    p = m.mean().to_numpy()
    assert np.isclose(null[('2h', 'IaA')], np.sqrt(p[0] * p[1]))
    shuf = np.mean([1 - rdm.shuffled_pair_distances(m, rng).to_numpy() for _ in range(400)],
        axis=0)
    assert np.allclose(shuf, null.to_numpy(), atol=0.01)


def test_transform_matrix():
    m = pd.DataFrame([[0.5, 2.0], [3.0, -1.0], [0.2, 0.3]], columns=['a', 'b'])
    assert rdm.transform_matrix(m).equals(m)
    assert rdm.transform_matrix(m, values='binary', thresh=1.0).to_numpy().tolist() == \
        [[0, 1], [1, 0], [0, 0]]
    assert len(rdm.transform_matrix(m, cells='responsive', thresh=1.0)) == 2
    with pytest.raises(ValueError):
        rdm.transform_matrix(m, values='binary')
    with pytest.raises(ValueError):
        rdm.transform_matrix(m, values='other', thresh=1.0)


def test_residuals_and_correlations():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.random(40))
    y = 2 * x + 0.1 + rng.normal(0, 0.01, 40)
    r, fit = stats.linear_resid(y, x)
    assert fit['slope'] == pytest.approx(2.0, abs=0.05) and fit['r2'] > 0.99
    ri, fi = stats.isotonic_resid(np.exp(3 * x), x)
    assert fi['r2'] == pytest.approx(1.0) and np.allclose(ri, 0)
    assert stats.spearman(np.exp(x), x) == pytest.approx(1.0)
    assert np.isnan(stats.pearson(x.iloc[:2], y.iloc[:2]))


def _recordings(seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_tuples([('d1', 1, 'L'), ('d1', 1, 'R'), ('d1', 2, 'L'),
        ('d2', 1, 'L'), ('d2', 1, 'R'), ('d2', 3, 'L')], names=['date', 'fly_num', 'hemibrain'])
    base = rng.random(10)
    return pd.DataFrame(base + rng.normal(0, 0.05, (6, 10)), index=idx)


def test_resampling():
    recs = _recordings()
    g = {'a': recs, 'b': recs * 2}
    m = stats.resample_group_means(g, np.random.default_rng(0))
    assert set(m) == {'a', 'b'} and len(m['a']) == 10
    mf = stats.resample_group_means(g, np.random.default_rng(0), cluster=('fly_num', 'date'))
    assert len(mf['a']) == 10
    with pytest.raises(ValueError):
        stats.resample_group_means(g, np.random.default_rng(0), cluster=('fly',))
    bt = stats.bootstrap_groups(lambda mm: {'mean_a': float(mm['a'].mean())}, g, n_boot=50)
    assert set(bt.columns) == {'observed', 'boot_mean', 'boot_sd', 'lo', 'hi'}
    assert bt.loc['mean_a', 'observed'] == pytest.approx(recs.mean().mean())
    assert bt.loc['mean_a', 'lo'] <= bt.loc['mean_a', 'hi']
    sh = stats.split_half_reliability(recs, n_splits=20)
    assert sh['n_recordings'] == 6 and 0 < sh['split_half'] <= 1
    assert sh['ceiling'] == pytest.approx(np.sqrt(sh['spearman_brown_full']))


def test_repo_info_and_git_commits(tmp_path):
    import hong2p
    info = repo_info(hong2p)
    assert Path(info['path']).is_dir() and info['git_hash'] and len(info['git_hash']) == 40
    assert info['dirty'] == bool(info['diff'])
    assert repo_info(tmp_path)['git_hash'] is None
    out = git_commits({'hong2p': Path(hong2p.__file__).parent, 'none': tmp_path})
    assert list(out) == ['hong2p_commit', 'hong2p_dirty', 'none_commit', 'none_dirty']
    assert out['hong2p_commit'] == info['git_hash'] and out['hong2p_dirty'] == info['dirty']
    assert out['none_commit'] is None and out['none_dirty'] is None


def test_oklab_diverging_cmap():
    viz = pytest.importorskip('hong2p.viz')
    cm = viz.oklab_diverging_cmap()
    colors = np.asarray(cm.colors)
    assert colors.shape == (256, 3) and cm.name == 'oklab_div'
    small = np.asarray(viz.oklab_diverging_cmap(11).colors)
    assert np.allclose(small[5], 1.0)
    # blue at the negative end, red at the positive end
    assert small[0, 2] > small[0, 0] and small[-1, 0] > small[-1, 2]


def test_invert_corr_triangular_values():
    """invert_corr_triangular's square (built with rdm.square_from_pairs since 2026-11-12):
    the odor order of its pairs, the diagonal value, NaN kept, odors named `name`, pairs in
    either order, and a missing or duplicated pair refused."""
    idx = pd.MultiIndex.from_tuples([('b', 'a'), ('b', 'c'), ('a', 'c')],
        names=['odor1', 'odor2'])
    ser = pd.Series([0.1, np.nan, 0.3], index=idx)
    sq = invert_corr_triangular(ser)
    assert list(sq.index) == list(sq.columns) == ['b', 'a', 'c']
    assert sq.index.name == sq.columns.name == 'odor'
    want = np.array([[1.0, 0.1, np.nan], [0.1, 1.0, 0.3], [np.nan, 0.3, 1.0]])
    assert np.array_equal(sq.to_numpy(), want, equal_nan=True)
    sq0 = invert_corr_triangular(ser, diag_value=0.0, name=None)
    assert sq0.index.name is None and np.allclose(np.diag(sq0.to_numpy()), 0.0)
    rev = pd.Series([0.3], index=pd.MultiIndex.from_tuples([('c', 'a')], names=idx.names))
    sq_rev = invert_corr_triangular(pd.concat([ser.iloc[:2], rev]), _index=idx)
    assert np.array_equal(sq_rev.to_numpy(), want, equal_nan=True)
    with pytest.raises(AssertionError):
        invert_corr_triangular(ser.iloc[:2], _index=idx)
    with pytest.raises(AssertionError):
        invert_corr_triangular(pd.concat([ser, rev]), _index=idx)
