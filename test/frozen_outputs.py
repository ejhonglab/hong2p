"""Frozen outputs of the functions that moved from the KC-remapping queue (model_experiment_queue's kcq) to hong2p and
pyolfsysm (2026-11-12): outputs flattened to arrays, saved with their inputs in one .npz, and compared exactly.

The same file is in model_experiment_queue (test/frozen_step3/), hong2p (test/) and olfsysm (test/): the queue's copy
made the frozen files from the queue's code before the move, and each repository's tests recompute the cases from the
same inputs (the queue through its re-exports, hong2p and pyolfsysm calling the moved functions with the queue's
parameters) and require the same outputs. Comparisons are exact (rtol = atol = 0) when numpy, scipy, scikit-learn and
pandas are the versions the files were made with; with other versions (e.g. Python 3.8 with pandas 1.5) floats may
differ in the last bits (BLAS, reduction order), so they are compared to rtol 1e-9 (1e-6 for the cases marked as
solver outputs, whose root-finding can amplify such differences).
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _version(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except ImportError:
        return 'missing'


VERSION_MODULES = ('numpy', 'scipy', 'sklearn', 'pandas')


def versions() -> Dict[str, str]:
    return {m: _version(m) for m in VERSION_MODULES}


def _lab(x) -> str:
    """A version-independent string for an index label."""
    if isinstance(x, tuple):
        return '(' + ', '.join(_lab(e) for e in x) + ')'
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if x is None:
        return '<None>'
    return str(x)


def labels(index) -> np.ndarray:
    return np.array([_lab(x) for x in index], dtype=str)


def _values(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == object or a.dtype.kind in 'SUT':
        return np.array([_lab(x) for x in a.ravel()], dtype=str).reshape(a.shape)
    if a.dtype.kind == 'f':
        return a.astype(np.float64)
    if a.dtype.kind in 'iu':
        return a.astype(np.int64)
    return a


def flatten(obj, key: str, out: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """{key/path: array} of an output (scalars, arrays, Series, DataFrames, dicts, lists, colormaps), with every
    label, name and key order recorded."""
    if out is None:
        out = {}
    if obj is None:
        out[key] = np.array(['<None>'])
    elif isinstance(obj, (bool, np.bool_)):
        out[key] = np.array([bool(obj)])
    elif isinstance(obj, (int, np.integer)):
        out[key] = np.array([int(obj)], dtype=np.int64)
    elif isinstance(obj, (float, np.floating)):
        out[key] = np.array([float(obj)], dtype=np.float64)
    elif isinstance(obj, str):
        out[key] = np.array([obj])
    elif isinstance(obj, np.ndarray):
        out[key] = _values(obj)
    elif isinstance(obj, pd.Series):
        out[f'{key}/values'] = _values(obj.to_numpy())
        out[f'{key}/index'] = labels(obj.index)
        out[f'{key}/index_names'] = labels(obj.index.names)
        out[f'{key}/name'] = np.array([_lab(obj.name)])
    elif isinstance(obj, pd.DataFrame):
        for j in range(obj.shape[1]):
            out[f'{key}/col{j}'] = _values(obj.iloc[:, j].to_numpy())
        out[f'{key}/columns'] = labels(obj.columns)
        out[f'{key}/column_names'] = labels(obj.columns.names)
        out[f'{key}/index'] = labels(obj.index)
        out[f'{key}/index_names'] = labels(obj.index.names)
        if obj.attrs:
            flatten(dict(obj.attrs), f'{key}/attrs', out)
    elif isinstance(obj, pd.Index):
        out[key] = labels(obj)
    elif isinstance(obj, dict):
        out[f'{key}/__keys__'] = np.array([_lab(k) for k in obj.keys()], dtype=str)
        for k, v in obj.items():
            flatten(v, f'{key}/{_lab(k)}', out)
    elif isinstance(obj, (list, tuple)):
        out[f'{key}/__len__'] = np.array([len(obj)], dtype=np.int64)
        for i, v in enumerate(obj):
            flatten(v, f'{key}/{i}', out)
    elif hasattr(obj, 'colors') and hasattr(obj, 'N'):
        # a matplotlib ListedColormap
        out[f'{key}/colors'] = np.asarray(obj.colors, dtype=np.float64)
        out[f'{key}/name'] = np.array([obj.name])
    else:
        raise TypeError(f'{key}: cannot flatten {type(obj)}')
    return out


def save(path, inputs: Dict[str, np.ndarray], expected: Dict[str, np.ndarray]) -> None:
    arrays = {f'in/{k}': v for k, v in inputs.items()}
    arrays.update({f'out/{k}': v for k, v in expected.items()})
    arrays.update({f'version/{k}': np.array([v]) for k, v in versions().items()})
    np.savez_compressed(path, **arrays)


def load(path) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return {'in': {k[3:]: v for k, v in d.items() if k.startswith('in/')},
        'out': {k[4:]: v for k, v in d.items() if k.startswith('out/')},
        'version': {k[8:]: str(v[0]) for k, v in d.items() if k.startswith('version/')}}


def same_versions(frozen: Dict[str, Dict[str, np.ndarray]]) -> bool:
    return frozen['version'] == versions()


def check(frozen: Dict[str, Dict[str, np.ndarray]], case: str, output, solver: bool = False) -> None:
    """Asserts that `output` flattens to the frozen arrays of `case` (exactly, unless the library versions differ)."""
    got = flatten(output, case)
    want = {k: v for k, v in frozen['out'].items() if k == case or k.startswith(case + '/')}
    assert want, f'no frozen outputs for {case}'
    assert sorted(got) == sorted(want), (case, sorted(set(got) ^ set(want)))
    exact = same_versions(frozen)
    rtol = 0.0 if exact else (1e-6 if solver else 1e-9)
    for k, w in want.items():
        g = got[k]
        assert g.shape == w.shape, (k, g.shape, w.shape)
        assert g.dtype.kind == w.dtype.kind, (k, g.dtype, w.dtype)
        if w.dtype.kind == 'f':
            if exact:
                ok = np.array_equal(g, w, equal_nan=True)
            else:
                ok = np.allclose(g, w, rtol=rtol, atol=1e-12, equal_nan=True)
            if not ok:
                bad = ~((g == w) | (np.isnan(g) & np.isnan(w)))
                raise AssertionError(f'{k}: {int(bad.sum())} of {g.size} values differ (max abs diff '
                    f'{np.nanmax(np.abs(g - w)):.3g}; exact={exact})')
        else:
            assert np.array_equal(g, w), (k, g[:5], w[:5])


def _frame(inp: Dict[str, np.ndarray], name: str, index_name: Optional[str] = None) -> pd.DataFrame:
    return pd.DataFrame(inp[f'{name}/values'], index=pd.Index(list(inp[f'{name}/index']), name=index_name),
        columns=pd.Index([str(c) for c in inp[f'{name}/columns']]))


def _pairs(inp: Dict[str, np.ndarray], name: str) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(list(zip([str(a) for a in inp[f'{name}/a']], [str(b) for b in inp[f'{name}/b']])),
        names=['odor_a', 'odor_b'])


def objects(inp: Dict[str, np.ndarray]) -> Dict[str, object]:
    """The frozen inputs as the cases use them (DataFrames, pair Series, arrays), for whichever of them `inp` has."""
    names = {k.split('/')[0] for k in inp}
    obj: Dict[str, object] = {}
    for name in ('counts_on', 'counts_off', 'resp_on', 'resp_off'):
        if name in names:
            df = _frame(inp, name, 'kc_id')
            df.index = pd.Index([int(i) for i in df.index], name='kc_id')
            obj[name] = df
    if 'tm_mat' in names:
        df = _frame(inp, 'tm_mat', 'roi')
        df.index = pd.Index([int(i) for i in df.index], name='roi')
        obj['tm_mat'] = df
    for name in ('sq_ctrl', 'orn'):
        if name in names:
            df = _frame(inp, name)
            df.index = pd.Index([str(i) for i in df.index])
            obj[name] = df
    for name in ('rec_ctrl', 'rec_tnt'):
        if name in names:
            idx = pd.MultiIndex.from_arrays([[str(x) for x in inp[f'{name}/date']], [int(x) for x in inp[f'{name}/fly_num']],
                [str(x) for x in inp[f'{name}/hemibrain']]], names=['date', 'fly_num', 'hemibrain'])
            obj[name] = pd.DataFrame(inp[f'{name}/values'], index=idx, columns=_pairs(inp, name))
    for name in ('data_effect', 'data_on', 'data_off'):
        if name in names:
            obj[name] = pd.Series(inp[f'{name}/values'], index=_pairs(inp, name), name=str(inp[f'{name}/name'][0]) or None)
    if 'spikes' in names:
        obj['spikes'] = pd.DataFrame({'kc': inp['spikes/kc'].astype(np.int64), 'odor': inp['spikes/odor'].astype(np.int64),
            'time_s': inp['spikes/time_s']})
        obj['n_kc'] = int(inp['spikes/n_kc'][0])
    for name in ('odors',):
        if name in inp:
            obj[name] = [str(o) for o in inp[name]]
    for name in ('trials_ctrl', 'trials_ctrl2', 'trials_tnt', 'traces', 'onsets', 'ref_amp', 'synth_t', 'synth_y'):
        if name in inp:
            obj[name] = inp[name]
    for name in ('cal_target', 'volume_rate'):
        if name in inp:
            obj[name] = float(inp[name][0])
    return obj


def toy_simulator(n_units: int = 200, n_odors: int = 16, seed: int = 0):
    """A deterministic stand-in for a fixed-parameter KC simulator (`simulate(params) -> (responses, spike counts)`),
    whose number of responses falls with the threshold and the APL scales: for the tuning cases."""
    rng = np.random.default_rng(seed)
    drive = rng.gamma(2.0, 30.0, (n_units, n_odors))
    pn = rng.gamma(2.0, 1.0, (n_units, n_odors))
    idx = pd.Index(list(range(n_units)), name='kc_id')
    cols = pd.Index([f'o{j}' for j in range(n_odors)])

    def sim(p):
        inh = (p['wAPLKC_scale'] * 40.0 + p['wKCAPL_scale'] * 5.0 + p['wPNAPL_scale'] * 3.0 + p['wAPLPN_scale'] * pn
            + p['wKCKC_scale'] * 2.0)
        spikes = np.floor(np.maximum(drive - p['fixed_thr'] - inh, 0.0) / 10.0)
        return (pd.DataFrame((spikes >= 2).astype(float), index=idx, columns=cols),
            pd.DataFrame(spikes, index=idx, columns=cols))
    return sim


TOY_PARAMS = dict(fixed_thr=30.0, wAPLKC_scale=0.5, wKCAPL_scale=0.4, wAPLPN_scale=1.5, wPNAPL_scale=0.6,
    wKCKC_scale=0.3)
