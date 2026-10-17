#!/usr/bin/env python3

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

    for name in util._moved_to_pandas:
        with pytest.warns(FutureWarning, match=f'hong2p.pandas.{name}'):
            fn = getattr(util, name)

        assert fn is getattr(hong2p.pandas, name)

    with pytest.warns(FutureWarning, match='hong2p.pandas.addlevel'):
        from hong2p.util import addlevel

    assert addlevel is hong2p.pandas.addlevel

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
