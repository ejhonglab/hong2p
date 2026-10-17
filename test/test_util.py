#!/usr/bin/env python3

import re
import warnings

import pytest
import numpy as np
import pandas as pd

from hong2p import util
from hong2p.util import subset_same_in_all_dicts, is_scalar


def test_is_scalar():
    # showing that old approaches in is_scalar (just checking isinstance before)
    # would not work
    def check_python_checks_fails(xs, t):
        for x in xs:
            # NOTE: isinstance(<np-scalar-float>, float) is actually True
            # but same does not work for bool/int (idk why)
            if not (isinstance(x, np.float64) and (t == float or float in t)):
                assert not isinstance(x, t)

            if type(t) is tuple:
                ts = t
            else:
                ts = (t,)

            for t in ts:
                assert type(x) != t
                assert type(x) is not t
                if not (isinstance(x, np.float64) and t == float):
                    assert not isinstance(x, t)

    arr = np.array([1.0, 0.0])
    bool_arr = arr.astype(bool)
    bool_list = [x for x in bool_arr]
    check_python_checks_fails(bool_list, bool)

    not_scalar = [False, True] + bool_list + ['a', None, arr, pd.Series(arr),
        bool_arr.dtype
    ]
    for x in not_scalar:
        assert not is_scalar(x)

    float_list = list(arr)
    # so that checks against np.float64 make sense in check_python_checks_fail
    assert all(isinstance(x, np.float64) for x in float_list)
    check_python_checks_fails(float_list, (int, float))
    check_python_checks_fails(float_list, float)

    int_arr = arr.astype(int)
    int_list = list(int_arr)
    check_python_checks_fails(int_list, (int, float))
    check_python_checks_fails(int_list, int)

    should_be_scalar = [0, 1, 0.5] + float_list + int_list + [arr.dtype, int_arr.dtype]
    for x in should_be_scalar:
        assert is_scalar(x)


def test_subset_same_in_all_dicts():
    common = {'a': 1}
    ds = [
        common,
        {'b': 0, **common},
        {'b': 2, **common},
    ]
    for xs in [ds, [ds[0]], ds[1:]]:
        assert subset_same_in_all_dicts(xs) == common

    assert subset_same_in_all_dicts([common, {'b': 1}]) == dict()


def test_moved_to_pandas_deprecated():
    import hong2p.pandas

    # Only calling the aliases should warn, not accessing / importing them.
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        for name in util._moved_to_pandas:
            assert hasattr(util, name)
            assert name in dir(util)
            assert getattr(util, name).__wrapped__ is getattr(hong2p.pandas, name)

        from hong2p.util import num_null

    df = pd.DataFrame({'a': [1.0, np.nan]})
    match = re.escape('hong2p.pandas.num_null')
    with pytest.warns(FutureWarning, match=match) as record:
        assert num_null(df) == 1

    # Checking stacklevel attributes warning to caller, rather than to util.py
    assert len(record) == 1 and record[0].filename == __file__

    with pytest.raises(AttributeError):
        util.not_a_real_attribute


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


def test_write_tiff(tmp_path):
    import tifffile

    movie = np.arange(2 * 4 * 5, dtype='<u2').reshape(2, 4, 5)
    tiff_path = tmp_path / 'movie.tif'
    util.write_tiff(tiff_path, movie)

    with tifffile.TiffFile(tiff_path) as tiff:
        # little-endian input should be converted to big-endian (for MATLAB imread_big)
        assert tiff.byteorder == '>'
        assert np.array_equal(tiff.asarray(), movie)
