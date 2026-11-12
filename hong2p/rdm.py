"""Odor-pair matrices: one value per unordered pair of odors, as a "pair Series".

A pair Series holds a value (a distance between two odors' responses, a correlation, the
change in either under some manipulation, ...) for each unordered pair of odors (or of any
other items), indexed by (odor_a, odor_b) with `odor_a` before `odor_b` in a canonical
order: the odors in `order` first, in that order, then any others sorted (`order_items`;
with no `order`, all sorted). `pair_list` gives the pairs of a set of odors in that order,
and the functions here return pair Series indexed that way (levels named `PAIR_NAMES`), so
outputs computed from different matrices line up pair for pair.

Distances between odors are computed across units (cells, glomeruli, model KCs) from a
(units x odors) matrix (`pair_distances`): 1 - cosine similarity, or 1 - Pearson
correlation.

Relation to the other pair helpers in hong2p: `pairs_from_square` takes the triangle with
`hong2p.util.corr_triangular` (in a given order, with this module's index names), and
`hong2p.util.invert_corr_triangular` builds its square with `square_from_pairs`; those two
keep al_analysis' conventions (pairs in sorted or given order named odor1 / odor2, panel
levels, a diagonal value). `hong2p.pandas.melt_symmetric` is the tidy form of a symmetric
matrix (diagonal included, any MultiIndex), for long-format tables rather than pair
statistics.

Moved from the KC-remapping experiment queue (model_experiment_queue's kcq.harness),
2026-11-12, with the 16-odor APL panel's order made the `order` argument.
"""
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PAIR_NAMES: Tuple[str, str] = ('odor_a', 'odor_b')
MEASURES: Tuple[str, ...] = ('cosine', 'pearson')
TRANSFORM_VALUES: Tuple[str, ...] = ('graded', 'binary', 'thresholded')
TRANSFORM_CELLS: Tuple[str, ...] = ('all', 'responsive')


def order_items(items, order: Optional[Sequence] = None) -> List:
    """Returns `items` in canonical order: those in `order` first (in that order), then the
    others sorted.

    >>> order_items(['zz', 'b', 'a', 'c'], order=['c', 'a'])
    ['c', 'a', 'b', 'zz']
    """
    rank = {o: i for i, o in enumerate(() if order is None else order)}
    return sorted(items, key=lambda o: (rank.get(o, len(rank)), o))


def pair_list(items, order: Optional[Sequence] = None) -> List[Tuple]:
    """Returns the unordered pairs of `items`, each (a, b) with a before b in canonical
    order (`order_items`), in that order (all pairs starting with the first item, then
    with the second, ...).

    >>> pair_list(['b', 'a', 'c'])
    [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """
    items = order_items(items, order)
    return [(a, b) for i, a in enumerate(items) for b in items[i + 1:]]


def pair_index(pairs: Sequence[Tuple], names: Sequence[str] = PAIR_NAMES
    ) -> pd.MultiIndex:
    """Returns the (odor_a, odor_b) MultiIndex of a list of pairs."""
    return pd.MultiIndex.from_tuples(pairs, names=list(names))


def pairs_from_square(sq: pd.DataFrame, order: Optional[Sequence] = None, *,
    names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the upper triangle of a symmetric (odor x odor) DataFrame, as a pair Series.

    Args:
        sq: square DataFrame with the same (string) labels on rows and columns, e.g. a
            distance or correlation matrix.
        order: canonical order (`order_items`) of the pairs' odors.
        names: index level names of the output.

    The values are taken with `hong2p.util.corr_triangular`, in canonical pair order.
    """
    from hong2p.util import corr_triangular

    if not sq.index.equals(sq.columns):
        raise ValueError('sq must have the same labels on rows and columns')

    items = order_items(sq.columns, order)
    pairs = pair_list(items, order)
    ser = corr_triangular(sq.loc[items, items], sort=False, check_index_names=False)
    return pd.Series(ser.to_numpy(), index=pair_index(pairs, names))


def square_from_pairs(pairs: pd.Series, order: Optional[Sequence] = None, *,
    diag: float = 0.0, missing: float = 0.0) -> pd.DataFrame:
    """Returns the symmetric (odor x odor) DataFrame of a pair Series.

    Args:
        pairs: Series indexed by (odor_a, odor_b) pairs, in either order within a pair.
        order: canonical order (`order_items`) of the output's rows and columns.
        diag: value on the diagonal.
        missing: value of the pairs `pairs` does not have.
    """
    odors = order_items(set(pairs.index.get_level_values(0)) |
        set(pairs.index.get_level_values(1)), order
    )
    sq = pd.DataFrame(missing, index=odors, columns=odors)
    if diag != missing:
        for o in odors:
            sq.loc[o, o] = diag

    for (a, b), v in pairs.items():
        sq.loc[a, b] = sq.loc[b, a] = v

    return sq


def canonical_pairs(pairs: pd.Series, order: Optional[Sequence] = None, *,
    names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns a pair Series (keys (a, b) in any order within a pair) reindexed to canonical
    pair order (`pair_list`).

    Raises KeyError if a pair of the odors present is missing.
    """
    d = {}
    for (a, b), v in pairs.items():
        key = (a, b) if (a, b) in set(pair_list([a, b], order)) else (b, a)
        d[key] = v

    odors = order_items(set(k for p in d for k in p), order)
    pl = pair_list(odors, order)
    return pd.Series([d[p] for p in pl], index=pair_index(pl, names))


def align_pairs(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Returns two (canonical) pair Series subset to the pairs they share, in `a`'s order,
    without the pairs either has NaN for."""
    common = a.index.intersection(b.index)
    a2, b2 = a.loc[common], b.loc[common]
    ok = a2.notna() & b2.notna()
    return a2[ok], b2[ok]


def subset_pairs(pairs: pd.Series, items: Sequence) -> pd.Series:
    """Returns the pairs with both odors in `items`."""
    keep = [p for p in pairs.index if p[0] in items and p[1] in items]
    return pairs.loc[keep]


def pair_distances(mat: pd.DataFrame, measure: str = 'cosine', drop_silent: bool = False,
    order: Optional[Sequence] = None, *, names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the odor x odor distances of a (units x odors) matrix, across units.

    Args:
        mat: rows units (cells, glomeruli, model KCs, ...), columns odors.
        measure: 'cosine' (1 - cosine similarity) or 'pearson' (1 - Pearson r).
        drop_silent: drop units that are 0 for every odor first (irrelevant for cosine;
            matters for Pearson).
        order: canonical order (`order_items`) of the odors.

    Returns a canonical pair Series (NaN for pairs with an odor no unit responds to).
    """
    m = mat.astype(float)
    if drop_silent:
        m = m.loc[(m != 0).any(axis='columns')]

    odors = order_items(m.columns, order)
    X = m[odors].to_numpy().T
    with np.errstate(invalid='ignore', divide='ignore'):
        if measure == 'cosine':
            norms = np.linalg.norm(X, axis=1)
            S = (X @ X.T) / np.outer(norms, norms)
        elif measure == 'pearson':
            S = np.corrcoef(X)
        else:
            raise ValueError(f'measure must be one of {MEASURES}, not {measure!r}')

    pl = pair_list(odors, order)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([1 - S[idx[a], idx[b]] for a, b in pl], index=pair_index(pl, names))


def transform_matrix(mat: pd.DataFrame, *, values: str = 'graded', cells: str = 'all',
    thresh: Optional[float] = None) -> pd.DataFrame:
    """Returns a (units x odors) response matrix thresholded / restricted before distances.

    Args:
        values: 'graded' (as is), 'binary' (response > `thresh` -> 1.0, else 0.0) or
            'thresholded' (values > `thresh` kept, others 0).
        cells: 'all' units, or 'responsive' (units with a response > `thresh` to at least
            one odor column of `mat`).
        thresh: response threshold (e.g. 1 for z-scored responses); needed by every
            option but 'graded' with 'all'.
    """
    if cells not in TRANSFORM_CELLS:
        raise ValueError(cells)
    if values not in TRANSFORM_VALUES:
        raise ValueError(values)
    if thresh is None and (values != 'graded' or cells != 'all'):
        raise ValueError(f'values={values!r}, cells={cells!r} need a threshold')

    m = mat.astype(float)
    if cells == 'responsive':
        m = m.loc[(m > thresh).any(axis='columns')]

    if values == 'binary':
        m = (m > thresh).astype(float)
    elif values == 'thresholded':
        m = m.where(m > thresh, 0.0)

    return m


def block_label(a, b, groups: Mapping) -> str:
    """Returns the block of a pair given a group for each odor: 'within' when both odors
    are in the same group, else the two groups' names sorted and joined by '-' (odors with
    no group are their own group)."""
    g1, g2 = groups.get(a, a), groups.get(b, b)
    return 'within' if g1 == g2 else '-'.join(sorted([g1, g2]))


def block_means(pairs: pd.Series, groups: Mapping) -> pd.Series:
    """Returns the mean of a pair Series within each block of pairs (`block_label`)."""
    labels = pd.Series([block_label(a, b, groups) for a, b in pairs.index],
        index=pairs.index
    )
    return pairs.groupby(labels).mean()


def per_odor_means(pairs: pd.Series, order: Optional[Sequence] = None) -> pd.Series:
    """Returns, per odor (canonical order), the mean over the pairs that contain it."""
    odors = order_items(set(pairs.index.get_level_values(0)) |
        set(pairs.index.get_level_values(1)), order
    )
    return pd.Series({o: pairs[[o in p for p in pairs.index]].mean() for o in odors})


def pair_design(pairs: pd.Index, odors: Optional[List] = None
    ) -> Tuple[np.ndarray, List]:
    """Returns the additive design over odors of a set of odor pairs, and the odor order.

    One row per pair, one column per odor, 1 for the pair's two odors (so a pair's value
    is modelled as the sum of its odors' shares). `odors` defaults to the odors in `pairs`,
    in order of appearance.
    """
    if odors is None:
        odors = list(dict.fromkeys([o for p in pairs for o in p]))
    col = {o: i for i, o in enumerate(odors)}
    X = np.zeros((len(pairs), len(odors)))
    for r, (a, b) in enumerate(pairs):
        X[r, col[a]] = 1.0
        X[r, col[b]] = 1.0
    return X, odors


def odor_shares(values: pd.Series) -> Tuple[pd.Series, float]:
    """Returns each odor's share of a per-pair quantity, and the variance it explains.

    The shares are the least-squares fit of the additive model `value(a, b) = s_a + s_b`
    over the pairs (`pair_design`); the second output is the fraction of the quantity's
    variance over pairs that the additive model explains (NaN for a constant quantity).
    """
    X, odors = pair_design(values.index)
    y = values.to_numpy(dtype=float)
    s, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ s
    var = float(np.var(y))
    return (pd.Series(s, index=pd.Index(odors, name='odor'), name='share'),
        (float(1 - np.var(y - fit) / var) if var > 0 else np.nan)
    )


def density_null_similarity(mat: pd.DataFrame, order: Optional[Sequence] = None, *,
    names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the expected cosine similarity between the odor columns of a (units x odors)
    matrix if unit identities were shuffled independently within each odor.

    The shuffle keeps each odor's response distribution (and so its number of
    responders). Exact permutation expectation: E[a . b] = N mean(a) mean(b), so
    E[cos] = N mean(a) mean(b) / (|a| |b|); for binary columns this is sqrt(p_a p_b).
    Returned as a canonical pair Series of similarities (not distances).
    """
    m = mat.astype(float)
    odors = order_items(m.columns, order)
    X = m[odors].to_numpy().T
    n = X.shape[1]
    means = X.mean(axis=1)
    norms = np.linalg.norm(X, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = n * np.outer(means, means) / np.outer(norms, norms)
    pl = pair_list(odors, order)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([S[idx[a], idx[b]] for a, b in pl], index=pair_index(pl, names))


def shuffled_pair_distances(mat: pd.DataFrame, rng: np.random.Generator, *,
    measure: str = 'cosine', order: Optional[Sequence] = None, **transform_kws
    ) -> pd.Series:
    """Returns the pair distances after permuting unit identities independently within each
    odor column (one draw of the null `density_null_similarity` is the mean of).

    Args:
        rng: the permutations' generator (one permutation per column, in column order).
        **transform_kws: passed to `transform_matrix` (applied before the shuffle).
    """
    m = transform_matrix(mat, **transform_kws)
    X = m.to_numpy().copy()
    for j in range(X.shape[1]):
        X[:, j] = X[rng.permutation(X.shape[0]), j]
    return pair_distances(pd.DataFrame(X, index=m.index, columns=m.columns),
        measure=measure, order=order
    )
