"""Statistics of paired values and of grouped matrices: residuals around linear and
monotone fits, rank and linear correlations of aligned Series, and resampling of the rows
(e.g. recordings, or flies) of grouped matrices (bootstrap, split halves).

Works on any Series sharing an index, e.g. the pair Series of `hong2p.rdm`, and on
DataFrames whose rows are the units resampled (one row per recording, columns e.g. odor
pairs).

Moved from the KC-remapping experiment queue (model_experiment_queue's kcq.harness),
2026-11-12, with the resampling generalized from Yang's recordings / flies to any rows or
clusters of rows.
"""
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hong2p.rdm import align_pairs


def linear_resid(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Returns the residual of `y` around its least-squares line on `x`, and the fit.

    The two are aligned first (`hong2p.rdm.align_pairs`: shared index, no NaN). The fit is
    a dict with 'slope', 'intercept', 'r2' and 'resid_sd'.
    """
    from scipy import stats

    y, x = align_pairs(y, x)
    slope, intercept, r, _, _ = stats.linregress(x.to_numpy(), y.to_numpy())
    resid = y - (intercept + slope * x)
    return resid, dict(slope=float(slope), intercept=float(intercept), r2=float(r**2),
        resid_sd=float(resid.std())
    )


def isotonic_resid(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Returns the residual of `y` around its best increasing (isotonic) fit on `x`, and
    the fit's 'r2' and 'resid_sd' (aligned first, as `linear_resid`)."""
    from sklearn.isotonic import IsotonicRegression

    y, x = align_pairs(y, x)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    pred = iso.fit(x.to_numpy(), y.to_numpy()).predict(x.to_numpy())
    resid = y - pred
    return resid, dict(r2=float(1 - resid.var() / y.var()), resid_sd=float(resid.std()))


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Returns the Spearman correlation of two aligned Series (NaN if fewer than 3 shared,
    non-NaN entries)."""
    from scipy import stats

    a, b = align_pairs(a, b)
    return float(stats.spearmanr(a, b)[0]) if len(a) > 2 else np.nan


def pearson(a: pd.Series, b: pd.Series) -> float:
    """Returns the Pearson correlation of two aligned Series (as `spearman`)."""
    from scipy import stats

    a, b = align_pairs(a, b)
    return float(stats.pearsonr(a, b)[0]) if len(a) > 2 else np.nan


def _cluster_levels(index: pd.Index, cluster: Sequence[str]) -> Tuple[str, ...]:
    levels = tuple(n for n in index.names if n in set(cluster))
    if len(levels) != len(set(cluster)):
        raise ValueError(f'cluster levels {list(cluster)} not all in {list(index.names)}')
    return levels


def resample_group_means(groups: Mapping[str, pd.DataFrame], rng: np.random.Generator,
    *, cluster: Optional[Sequence[str]] = None) -> Dict[str, pd.Series]:
    """Returns one bootstrap replicate of each group's mean row.

    Args:
        groups: {name: DataFrame}, rows the units resampled (e.g. recordings), columns
            the statistics averaged (e.g. odor pairs).
        rng: draws the resamples, group by group in `groups`' order.
        cluster: index levels whose unique combinations are resampled instead of rows,
            each drawn cluster keeping all its rows (e.g. ('date', 'fly_num') to draw
            flies, keeping both hemispheres of each, which are not independent). None
            draws rows.

    Each group's draws are with replacement, as many as it has rows (or clusters); the
    drawn rows are then averaged with equal weight.
    """
    means = {}
    for g, df in groups.items():
        if cluster is None:
            idx = rng.integers(0, len(df), len(df))
            means[g] = df.iloc[idx].mean()
        else:
            levels = _cluster_levels(df.index, cluster)
            others = [n for n in df.index.names if n not in levels]
            units = df.index.droplevel(others).unique()
            drawn = units[rng.integers(0, len(units), len(units))]
            parts = [df.xs(u, level=levels, drop_level=False) for u in drawn]
            means[g] = pd.concat(parts).mean()
    return means


def bootstrap_groups(fn: Callable[[Dict[str, pd.Series]], Dict[str, float]],
    groups: Mapping[str, pd.DataFrame], *, n_boot: int = 1000, seed: int = 0,
    ci: Tuple[float, float] = (5, 95), cluster: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
    """Returns the bootstrap distribution of statistics of groups' mean rows.

    Args:
        fn: maps {name: mean row} -> {statistic: value}.
        groups: {name: DataFrame} (`resample_group_means`).
        n_boot: number of resamples.
        seed: seeds the resamples' generator.
        ci: percentiles of the interval.
        cluster: see `resample_group_means`.

    Returns DataFrame indexed by statistic: 'observed' (`fn` of the groups' means),
    'boot_mean', 'boot_sd', 'lo', 'hi' (NaN draws ignored).
    """
    rng = np.random.default_rng(seed)
    observed = fn({g: df.mean() for g, df in groups.items()})
    draws = {k: [] for k in observed}
    for _ in range(n_boot):
        means = resample_group_means(groups, rng, cluster=cluster)
        for k, v in fn(means).items():
            draws[k].append(v)

    rows = {}
    for k, v in observed.items():
        arr = np.asarray(draws[k], dtype=float)
        rows[k] = dict(observed=v, boot_mean=np.nanmean(arr), boot_sd=np.nanstd(arr),
            lo=np.nanpercentile(arr, ci[0]), hi=np.nanpercentile(arr, ci[1])
        )
    return pd.DataFrame(rows).T


def split_half_reliability(rows: pd.DataFrame, *, n_splits: int = 200, seed: int = 0,
    method: str = 'spearman') -> Dict[str, float]:
    """Returns the reliability of the mean row of a DataFrame, from random split halves of
    its rows (e.g. of the mean odor-pair distance matrix over recordings).

    Keys: 'n_recordings' (rows), 'split_half' (the correlation between the two halves'
    means, mean over splits; `method` 'spearman' or 'pearson'), 'split_half_lo' / '_hi'
    (its 5th / 95th percentiles), 'spearman_brown_full' (2r / (1 + r), for the full set
    of rows) and 'ceiling' (its square root: the expected correlation with the observed
    mean of a noise-free model that matched the true mean perfectly).
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    n = len(rows)
    corr = stats.spearmanr if method == 'spearman' else stats.pearsonr
    vals = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a = rows.iloc[idx[:n // 2]].mean()
        b = rows.iloc[idx[n // 2:]].mean()
        vals.append(corr(a, b)[0])

    vals = np.asarray(vals)
    r = float(np.nanmean(vals))
    full = 2 * r / (1 + r)
    return dict(n_recordings=n, split_half=r, split_half_lo=float(np.nanpercentile(vals, 5)),
        split_half_hi=float(np.nanpercentile(vals, 95)), spearman_brown_full=full,
        ceiling=float(np.sqrt(full)),
    )
