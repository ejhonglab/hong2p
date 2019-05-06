#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd

import util as u


test_df = pd.DataFrame({
    'str': 'asdf',
    'int': 1,
    'arr1': [[0.3, 0.1, 0.5]],
    'float': 1.1
}, index=[0]) 

# TODO test series vs length 1 dataframes?
# TODO also test extra cols?
# TODO test w/ >1 / mismatching / cols in index? multiindex?

def test_diff_dataframes_equal():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert u.diff_dataframes(df1, df2) is None


def test_diff_dataframes_floatclose():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1e-6
    assert not df1.equals(df2)
    assert u.diff_dataframes(df1, df2) is None


def test_diff_dataframes_arrclose():
    df1 = test_df.copy()
    df2 = test_df.copy()
    # TODO why does equals check here work, but calls after copying stuff fail?
    # (failed w/ an arr2 that was a numpy array like arr1)
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    v2 = df2.loc[0, 'arr1'][1]
    df2.loc[0, 'arr1'][1] = v2 + 1e-6
    assert not df1.equals(df2)
    assert u.diff_dataframes(df1, df2) is None


def test_diff_dataframes_other_ne():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert df1.equals(df2)
    df1['str'] = 'qwerty'
    assert not df1.equals(df2)
    diff = u.diff_dataframes(df1, df2)
    assert diff is not None
    assert len(diff) == 1
    assert 'str' in diff.index.get_level_values('col')


def test_diff_dataframes_float_ne():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = df2.loc[0, 'float'] + 1.0
    assert not df1.equals(df2)
    diff = u.diff_dataframes(df1, df2)
    assert diff is not None
    assert len(diff) == 1
    assert 'float' in diff.index.get_level_values('col')


def test_diff_dataframes_arr_ne():
    df1 = test_df.copy()
    df2 = test_df.copy()
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    v2 = df2.loc[0, 'arr1'][1]
    df2.loc[0, 'arr1'][1] = v2 + 100
    assert not df1.equals(df2)
    diff = u.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_onenan():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert df1.equals(df2)
    df2.loc[0, 'float'] = np.nan
    assert not df1.equals(df2)
    diff = u.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'float' in diff.index.get_level_values('col')


def test_diff_dataframes_arr_onenan():
    df1 = test_df.copy()
    df2 = test_df.copy()
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df2.loc[0, 'arr1'][1] = np.nan
    assert not df1.equals(df2)
    diff = u.diff_dataframes(df1, df2)
    assert len(diff) == 1
    assert 'arr1' in diff.index.get_level_values('col')


def test_diff_dataframes_float_naneq():
    df1 = test_df.copy()
    df2 = test_df.copy()
    assert df1.equals(df2)
    df1.loc[0, 'float'] = np.nan
    df2.loc[0, 'float'] = np.nan
    assert u.diff_dataframes(df1, df2) is None


def test_diff_dataframes_arr_naneq():
    df1 = test_df.copy()
    df2 = test_df.copy()
    df1.loc[[0], 'arr1'] = pd.Series([df1.loc[0, 'arr1'].copy()], index=[0])
    df2.loc[[0], 'arr1'] = pd.Series([df2.loc[0, 'arr1'].copy()], index=[0])
    assert df1.equals(df2)
    df1.loc[0, 'arr1'][1] = np.nan
    df2.loc[0, 'arr1'][1] = np.nan
    assert u.diff_dataframes(df1, df2) is None


