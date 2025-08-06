#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd

from hong2p import util

from hong2p.olf import solvent_str


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
    corr2 = 1 - util.frame_pdist(df, metric='correlation')

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

    odf = util.addlevel(df, names=n, values=v)
    odf2 = util.addlevel(df, names=[n], values=[v])
    assert odf2.equals(odf)
    check_index(odf.index)

    odf = util.addlevel(df, names='x', values=7, axis='columns')
    check_index(odf.columns)

    ns = [n, 'y']
    vs = [v, 9]
    odf = util.addlevel(df, names=ns, values=vs)

    def check_index_2added(mi):
        check_index(mi)
        assert len(mi.names) == 3 and mi.names[:2] == ns

    check_index_2added(odf.index)

    # Testing when rows / columns are already multi indices
    df = util.addlevel(odf, names='x', values='xcol', axis='columns')

    n = 'z'
    v = 3.14

    odf = util.addlevel(util.addlevel(df, names=n, values=v),
        names=n, values=v, axis='columns'
    )

    check_index(odf.index, n=n, v=v)
    check_index(odf.columns, n=n, v=v)

    # TODO add tests where values contains something of length equal to the
    # corresponding axis, rather than just a single unique value (+ one test w/ a mix of
    # each)


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
    assert util.diff_dataframes(df1, df2) is None


def test_diff_dataframes_floatclose():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1e-6
    assert not df1.equals(df2)
    assert util.diff_dataframes(df1, df2) is None


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
    assert util.diff_dataframes(df1, df2) is None


def test_diff_dataframes_other_ne():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df1['str'] = 'qwerty'
    assert not df1.equals(df2)
    diff = util.diff_dataframes(df1, df2)
    assert diff is not None
    assert len(diff) == 1
    assert 'str' in diff.index.get_level_values('col')


def test_diff_dataframes_float_ne():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1.0
    assert not df1.equals(df2)
    diff = util.diff_dataframes(df1, df2)
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
    diff = util.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_onenan():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = np.nan
    assert not df1.equals(df2)
    diff = util.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'float' in diff.index.get_level_values('col')


def test_diff_dataframes_arr_onenan():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df2.loc[0, 'arr1'][1] = np.nan
    assert not df1.equals(df2)
    diff = util.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_naneq():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    assert df1.equals(df2)
    df1.loc[0, 'float'] = np.nan
    df2.loc[0, 'float'] = np.nan
    assert util.diff_dataframes(df1, df2) is None


def test_diff_dataframes_arr_naneq():
    df1 = diff_df.copy()
    df2 = diff_df.copy()
    df1.loc[[0], 'arr1'] = pd.Series([df1.loc[0, 'arr1'].copy()], index=[0])
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df1.loc[0, 'arr1'][1] = np.nan
    df2.loc[0, 'arr1'][1] = np.nan
    assert util.diff_dataframes(df1, df2) is None


def test_const_ranges():
    rs = util.const_ranges([])
    assert rs == []

    rs = util.const_ranges([], include_val=True)
    assert rs == []

    rs = util.const_ranges([0])
    assert rs == [(0, 0)]

    rs = util.const_ranges([0, 1])
    assert rs == [(0, 0), (1, 1)]

    rs = util.const_ranges([0, 1, 1])
    assert rs == [(0, 0), (1, 2)]

    rs = util.const_ranges([0, 0, 1])
    assert rs == [(0, 1), (2, 2)]

    rs = util.const_ranges(['a'], include_val=True)
    assert rs == [('a', 0, 0)]

    rs = util.const_ranges(['a', 'a', 'b'], include_val=True)
    assert rs == [('a', 0, 1), ('b', 2, 2)]

    vs = ['a', 'a', 'b', 'b', 'a', 'a']
    rs = util.const_ranges(vs, include_val=True)
    assert rs == [('a', 0, 1), ('b', 2, 3), ('a', 4, 5)]
    rs = util.const_ranges(vs)
    assert rs == [(0, 1), (2, 3), (4, 5)]
