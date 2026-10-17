import subprocess
import sys

import pytest

import numpy as np
import pandas as pd

from hong2p.pandas import (subset_index, frame_pdist, addlevel, diff_dataframes,
    expand_array_cols, melt_symmetric, invert_melt_symmetric, pd_allclose, pd_isclose,
    dtypes_equal
)
from hong2p.olf import solvent_str


def test_subset_index():
    tuples = [
        ('x', 1, 0),
        ('x', 2, 0),
        ('y', 1, 0),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=['a', 'b', 'c'])

    i2 = subset_index(index, ['a', 'b'], check_unique=True)
    assert i2.equals(index.droplevel('c'))

    with pytest.raises(AssertionError):
        subset_index(index, ['a', 'c'], check_unique=True)

    i2 = subset_index(index, ['a', 'c'], check_unique=False)
    assert i2.equals(index.droplevel('b'))

    # TODO any point to adding a squeeze=True param, that could be set False if someone
    # really did want a 1 level MultiIndex? don't think so...
    i2 = subset_index(index, ['a'], check_unique=False)
    assert not isinstance(i2, pd.MultiIndex)
    assert i2.equals(index.droplevel(['b', 'c']))

    with pytest.raises(KeyError):
        subset_index(index, ['a', 'z'])


# TODO refactor to share w/ test_olf.py (where i copied this from)
def _make_df(rng, odor1, odor2=None, odors_in_index=True, panel=None):
    odor_keys = ['odor1', 'odor2']
    for_df = {
        'odor1': odor1,
        # TODO flag to not make with odor2 at all (and test with that)
        'odor2': [solvent_str] * len(odor1) if odor2 is None else odor2,
        'delta_f': rng.random(len(odor1)),
    }
    if panel is not None:
        for_df['panel'] = panel
        odor_keys.insert(0, 'panel')

    df = pd.DataFrame(for_df)

    if odors_in_index:
        df = df.set_index(odor_keys)

    return df


def test_frame_pdist(rng):
    odors = ['a', 'b', 'c']

    # df1 and df2 will have different randomly-generated data
    df1 = _make_df(rng, odors)
    df2 = _make_df(rng, odors)

    df = pd.concat([df1, df2], axis='columns')
    df.columns = [0, 1]
    df.columns.name = 'cell'

    # want this fn to behave the same as df.corr() (in terms of having same shape
    # output, for same shape input)
    corr1 = df.corr()

    # metric='correlation' is a correlation DISTANCE (1 - corr)
    corr2 = 1 - frame_pdist(df, metric='correlation')

    assert corr1.index.equals(corr2.index)
    assert corr1.columns.equals(corr2.columns)

    # corr1.equals(corr2) fails because of numerical issues
    assert np.allclose(corr1, corr2)


def test_addlevel():
    df = pd.DataFrame(data=[[0,1,2],[4,5,6]], columns=['a', 'b', 'c'])

    n = 'x'
    v = 7
    def check_index(mi, n=n, v=v):
        assert mi.names[0] == n
        unq = mi.get_level_values(n)
        assert len(set(unq)) == 1 and unq[0] == v

    odf = addlevel(df, names=n, values=v)
    odf2 = addlevel(df, names=[n], values=[v])
    assert odf2.equals(odf)
    check_index(odf.index)

    odf = addlevel(df, names='x', values=7, axis='columns')
    check_index(odf.columns)

    ns = [n, 'y']
    vs = [v, 9]
    odf = addlevel(df, names=ns, values=vs)

    def check_index_2added(mi):
        check_index(mi)
        assert len(mi.names) == 3 and mi.names[:2] == ns

    check_index_2added(odf.index)

    # Testing when rows / columns are already multi indices
    df = addlevel(odf, names='x', values='xcol', axis='columns')

    n = 'z'
    v = 3.14

    odf = addlevel(addlevel(df, names=n, values=v),
        names=n, values=v, axis='columns'
    )

    check_index(odf.index, n=n, v=v)
    check_index(odf.columns, n=n, v=v)

    # (these used to silently drop extra names/values)
    with pytest.raises(ValueError):
        addlevel(df, names=['a', 'b'], values=[1])

    with pytest.raises(ValueError):
        addlevel(df, names=['a'], values='xy')

    # TODO add tests where values contains something of length equal to the
    # corresponding axis, rather than just a single unique value (+ one test w/ a mix of
    # each)


def test_expand_array_cols():
    df = pd.DataFrame({
        'x': ['a', 'b', 'c'],
        'trace': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        # (non-array column after array one, and out of alphabetical order, to check
        # output column order matches input)
        'n': [3, 2, 1],
    })
    # So index has no 0 label (as after filtering), which used to cause a KeyError
    df = df.iloc[1:]

    out = expand_array_cols(df)

    assert list(out['x']) == ['b', 'b', 'c', 'c']
    assert list(out['trace']) == [2.0, 3.0, 4.0, 5.0]
    assert out['trace'].dtype == np.float64
    assert list(out['n']) == [2, 2, 1, 1]
    assert list(out.columns) == list(df.columns)

    # Input should not have had list elements converted to arrays
    assert all(type(t) is list for t in df['trace'])

    # Output dtype should come from all values, not just first array (which is int here)
    df = pd.DataFrame({'x': ['a', 'b'], 'trace': [[0, 1], [2.5, 3.5]]})
    out = expand_array_cols(df)
    assert list(out['trace']) == [0.0, 1.0, 2.5, 3.5]
    assert out['trace'].dtype == np.float64


def test_pd_allclose():
    df = pd.DataFrame({'x': [1.0, 2.0], 's': ['a', 'b']})
    assert pd_allclose(df, df.copy())

    close = df.copy()
    close['x'] += 1e-12
    assert pd_allclose(df, close)

    not_close = df.copy()
    not_close['x'] += 1
    assert not pd_allclose(df, not_close)

    str_differs = df.copy()
    str_differs['s'] = ['a', 'c']
    assert not pd_allclose(df, str_differs)

    # str column in only one input. should be False in either argument order (one order
    # used to raise TypeError)
    a = pd.DataFrame({'x': [1.0, 2.0]})
    b = pd.DataFrame({'x': ['a', 'b']})
    assert not pd_allclose(a, b)
    assert not pd_allclose(b, a)


def test_pd_isclose():
    a = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]}, index=['i', 'j'])
    b = a.copy()
    b.loc['j', 'y'] += 1

    isclose = pd_isclose(a, b)
    assert isinstance(isclose, pd.DataFrame)
    assert isclose.index.equals(a.index) and isclose.columns.equals(a.columns)
    assert isclose.values.tolist() == [[True, True], [True, False]]

    isclose = pd_isclose(a['y'], b['y'])
    assert isinstance(isclose, pd.Series)
    assert isclose.tolist() == [True, False]

    with pytest.raises(ValueError):
        pd_isclose(a, b.iloc[::-1])


def test_dtypes_equal():
    ints = pd.DataFrame({'x': [1, 2], 's': ['a', 'b']})
    assert dtypes_equal(ints, ints.copy())

    # (columns in different order)
    assert dtypes_equal(ints, ints[['s', 'x']])

    whole_floats = ints.astype({'x': float})
    floats = whole_floats.copy()
    floats['x'] = [1.5, 2.0]
    nan_floats = whole_floats.copy()
    nan_floats['x'] = [np.nan, 2.0]

    # (cast=True used to be the default, and returned True for all of these)
    assert not dtypes_equal(ints, whole_floats)
    assert not dtypes_equal(ints, floats)

    assert dtypes_equal(ints, whole_floats, cast=True)
    assert not dtypes_equal(ints, floats, cast=True)
    assert not dtypes_equal(ints, nan_floats, cast=True)


@pytest.mark.parametrize('suffixes', [('_a', '_b'), ('', '_b')])
def test_invert_melt_symmetric(suffixes):
    odors = pd.Index(['a', 'b', 'c'], name='odor')
    df = pd.DataFrame([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ], index=odors, columns=odors
    )
    s0, s1 = suffixes

    ser = melt_symmetric(df, suffixes=suffixes, keep_duplicate_values=True)
    square = invert_melt_symmetric(ser, suffixes=suffixes)

    # Levels ending in first suffix are unstacked into columns
    assert square.columns.name == f'odor{s0}'
    assert square.index.name == f'odor{s1}'
    assert np.array_equal(square.values, df.values)


diff_df = pd.DataFrame({
    'str': 'asdf',
    'int': 1,
    'arr1': [[0.3, 0.1, 0.5]],
    'float': 1.1
}, index=[0])

# TODO test series vs length 1 dataframes?
# TODO also test extra cols?
# TODO test w/ >1 / mismatching / cols in index? multiindex?

def test_diff_dataframes_equal():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert diff_dataframes(df1, df2) is None


def test_diff_dataframes_floatclose():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1e-6
    assert not df1.equals(df2)
    assert diff_dataframes(df1, df2) is None


def test_diff_dataframes_arrclose():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    # TODO why does equals check here work, but calls after copying stuff fail?
    # (failed w/ an arr2 that was a numpy array like arr1)
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    v2 = df2.loc[0, 'arr1'][1]
    df2.loc[0, 'arr1'][1] = v2 + 1e-6
    assert not df1.equals(df2)
    assert diff_dataframes(df1, df2) is None


def test_diff_dataframes_other_ne():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df1['str'] = 'qwerty'
    assert not df1.equals(df2)
    diff = diff_dataframes(df1, df2)
    assert diff is not None
    assert len(diff) == 1
    assert 'str' in diff.index.get_level_values('col')


def test_diff_dataframes_float_ne():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1.0
    assert not df1.equals(df2)
    diff = diff_dataframes(df1, df2)
    assert diff is not None
    assert len(diff) == 1
    assert 'float' in diff.index.get_level_values('col')


def test_diff_dataframes_arr_ne():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    v2 = df2.loc[0, 'arr1'][1]
    df2.loc[0, 'arr1'][1] = v2 + 100
    assert not df1.equals(df2)
    diff = diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_onenan():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = np.nan
    assert not df1.equals(df2)
    diff = diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'float' in diff.index.get_level_values('col')


def test_diff_dataframes_arr_onenan():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df2.loc[0, 'arr1'][1] = np.nan
    assert not df1.equals(df2)
    diff = diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_naneq():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df1.loc[0, 'float'] = np.nan
    df2.loc[0, 'float'] = np.nan
    assert diff_dataframes(df1, df2) is None


def test_diff_dataframes_arr_naneq():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    df1.loc[[0], 'arr1'] = pd.Series([df1.loc[0, 'arr1'].copy()], index=[0])
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df1.loc[0, 'arr1'][1] = np.nan
    df2.loc[0, 'arr1'][1] = np.nan
    assert diff_dataframes(df1, df2) is None


def test_does_not_import_util():
    # hong2p.util imports this module, so this module importing hong2p.util (even inside
    # a function) is a circular dependency. also checks hong2p/__init__.py doesn't import
    # util (e.g. via cli_entry_points).
    code = 'import sys, hong2p.pandas; assert "hong2p.util" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)

    from hong2p import util
    from hong2p.pandas import is_scalar
    # should still be accessible from util (without the deprecation warning the other
    # moved fns have)
    assert util.is_scalar is is_scalar
