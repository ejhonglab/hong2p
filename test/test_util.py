#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd

from hong2p import util


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

