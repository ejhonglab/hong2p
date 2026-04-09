
from typing import Any, Sequence

import pandas as pd


# TODO TODO move other fns here, as appropriate. probably many currently in util would
# make more sense here

# NOTE: i would have said Sequence[str], but technically names can be whatever, right?
def subset_index(index: pd.MultiIndex, names: Sequence[Any], *,
    check_unique: bool = True) -> pd.Index:
    """Returns new index that has a subset of names in original.

    Args:
        index: index to subset

        names: list of level names, that should be a subset of `index.names`

        check_unique: if True, will assert that the index is unique, even after
            subsetting levels.

    Raises KeyError if not all `names` in `index.names`
    """
    if not all(n in index.names for n in names):
        raise KeyError(f'not all {names=} in {index.names=}')

    if check_unique:
        assert not index.duplicated().any(), ('check_unique=True and index was already '
            'duplicated'
        )

    # TODO test case where we subset down to one level. don't need to squeeze or
    # something, do we?
    # does not automatically give us an (non-Multi) Index when names is len 1
    if len(names) > 1:
        subset = pd.MultiIndex.from_frame(index.to_frame(index=False)[names])
    else:
        # TODO need the copy? prob not
        subset = index.get_level_values(names[0]).copy()

    if check_unique:
        assert not subset.duplicated().any(), ('check_unique=True and index not unique '
            f'within only {names=}'
        )

    return subset

