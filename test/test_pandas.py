
import pytest

import pandas as pd

from hong2p.pandas import subset_index


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

