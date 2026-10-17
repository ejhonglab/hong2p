
"""
Utility functions for working with pandas objects (DataFrames, Series, and their indices)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd

from hong2p.types import DataFrameOrSeries, Axis


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


# TODO support xarray.DataArrays? (isnull().sum().item(0) should work)
def num_null(df: pd.DataFrame) -> int:
    return df.isna().sum().sum()

# TODO an issue that this returns a float type (at least in case where input is empty)?
def num_notnull(df: pd.DataFrame) -> int:
    return df.notna().sum().sum()


# TODO maybe accept dict of names / values? which pd fn to copy the interfact of
# names/values from (DataFrame creation probably)?
# TODO TODO support xarray dataarrays?
# TODO test
# TODO TODO TODO add support for sequence of values (for a given name (/names entry)).
# at the moment, seems only scalar values supported. or at least rename to indicate it
# only supports scalars in values.
# TODO type of values?
def addlevel(data: DataFrameOrSeries, names: Union[str, List[str]], values, *,
    axis: Axis = 'index') -> DataFrameOrSeries:
    """Add level(s) to pandas MultiIndex. Does not change input.

    Intended to be an inverse to pandas.DataFrame.droplevel. pandas.DataFrame.set_index
    with `append=True` would work *except* that there is no `axis` kwarg to that
    function, so it does not work for the columns. pandas.DataFrame.unstack is almost
    what I would want, but it can seemingly arbitrarily change order of rows.

    Args:
        data: DataFrame/Series to add MultiIndex levels to
        names: `str`/sequence-of-`str` name(s) for the new levels
        values: values for the new levels. If `names` is a sequence, this should be of
            the same length.
        axis: 0/'index' or 1/'columns', defauling to 'index' as in pandas

    Returns: DataFrame with MultiIndex containing names/levels from input
    """
    # https://stackoverflow.com/questions/14744068
    if isinstance(names, str):
        names = [names]
        values = [values]

    for name, value in list(zip(names, values))[::-1]:
        data = pd.concat([data], names=[name], keys=[value], axis=axis)

    return data


def df_map(df: pd.DataFrame, fn: Callable, **kwargs) -> pd.DataFrame:
    """Applies `fn` elementwise. `DataFrame.map` if available, else `.applymap`.

    `DataFrame.map` was added in pandas 2.1 (and `.applymap` removed in 3.0).
    """
    try:
        map_fn = df.map
    # pandas < 2.1
    except AttributeError:
        map_fn = df.applymap

    return map_fn(fn, **kwargs)


def is_str_or_object_dtype(dtype) -> bool:
    """Returns whether dtype is `object` or a pandas string dtype.

    pandas >= 3.0 infers `str` dtype (a `pd.StringDtype`) for string data, where older
    versions would use `object`.
    """
    return dtype == object or isinstance(dtype, pd.StringDtype)


def select_str_and_object_dtypes(df: pd.DataFrame, *, exclude: bool = False
    ) -> pd.DataFrame:
    """Like `df.select_dtypes(include='object')` (`exclude=...` if `exclude=True`),
    but also including pandas >= 3.0 `str` dtype columns, without a deprecation warning.
    """
    kwarg = 'exclude' if exclude else 'include'
    try:
        return df.select_dtypes(**{kwarg: ['object', 'str']})
    # pandas < 3.0:
    # TypeError: string dtypes are not allowed, use 'object' instead
    except TypeError:
        return df.select_dtypes(**{kwarg: 'object'})


def arraylike_cols(df):
    """Returns a list of columns that have only lists or arrays as elements.
    """
    # not using select_dtypes(include='object'), as that also includes pandas>=3.0 'str'
    # dtype columns (and warns about that behavior being deprecated)
    df = df[[c for c in df.columns if df[c].dtype == object]]
    return df.columns[df_map(df, lambda o:
        type(o) is list or isinstance(o, np.ndarray)).all()]


# TODO use in other places that duplicate this functionality
# (like in natural_odors/kc_analysis ?)
def expand_array_cols(df):
    """Expands any list/array entries, with new rows for each entry.

    For any columns in `df` that have all list/array elements (at each row),
    the column in `out_df` will have the type of single elements from those
    arrays.

    The length of `out_df` will be the length of the input `df`, multiplied by
    the length (should be common in each input row) of each set of list/array
    elements.

    Other columns have their values duplicated, to match the lengths of the
    expanded array values.

    Args:
    `df` (pd.DataFrame)

    Returns:
    `out_df` (pd.DataFrame)
    """
    if len(df.index.names) > 1 or df.index.names[0] is not None:
        raise NotImplementedError('numpy repeating may not handle index. '
            'reset_index first.')

    # Will be ['raw_f', 'df_over_f', 'from_onset'] in the main way I'm using
    # this function.
    array_cols = arraylike_cols(df)

    if len(array_cols) == 0:
        raise ValueError('df did not appear to have any columns with all '
            'arraylike elements')

    # Index is unnamed (checked above), so nothing lost by dropping it. Avoids label-based
    # indexing below (e.g. in np.concatenate) failing if there is no 0 label, and also
    # means converting array cols below won't modify the input.
    df = df.reset_index(drop=True)

    orig_dtypes = df.dtypes.to_dict()
    for ac in array_cols:
        df[ac] = df[ac].apply(lambda x: np.array(x))
        assert len(df[ac]) > 0 and len(df[ac].iloc[0]) > 0
        orig_dtypes[ac] = df[ac].iloc[0][0].dtype

    non_array_cols = df.columns.difference(array_cols)

    # TODO true vectorized way to do this?
    # is str.len (on either rows/columns) faster (+equiv)?
    array_lengths = df_map(df[array_cols], len)
    c0 = array_lengths[array_cols[0]]
    for c in array_cols[1:]:
        assert np.array_equal(c0, array_lengths[c])
    array_lengths = c0

    # TODO more idiomatic / faster way to do what this loop is doing?
    n_non_array_cols = len(non_array_cols)
    expanded_rows_list = []
    for row, n_repeats in zip(df[non_array_cols].values, array_lengths):
        # could try subok=True if want to use pandas obj as input rather than
        # stuff from .values?
        expanded_rows = np.broadcast_to(row, (n_repeats, n_non_array_cols))
        expanded_rows_list.append(expanded_rows)
    nac_data = np.concatenate(expanded_rows_list, axis=0)

    ac_data = df[array_cols].apply(np.concatenate)
    assert nac_data.shape[0] == ac_data.shape[0]
    data = np.concatenate((nac_data, ac_data), axis=1)
    assert data.shape[1] == df.shape[1]

    new_cols = list(non_array_cols) + list(array_cols)
    # TODO copy=False is fine here, right? measure the time difference?
    out_df = pd.DataFrame(columns=new_cols, data=data).astype(orig_dtypes,
        copy=False)

    return out_df


# TODO work with NaN in inputs? i don't want NaN != NaN being reported as a difference.
# use my new nan_eq if needed/helpful.
# TODO possible to replace (some of?) this w/ pandas.DataFrame.compare?
def diff_dataframes(df1, df2) -> Optional[pd.DataFrame]:
    """Returns a DataFrame summarizing input differences, or None if no differences.
    """
    # TODO do i want df1 and df2 to be allowed to be series?
    # (is that what they are now? need to modify anything?)
    assert (df1.columns == df2.columns).all(), 'DataFrame column names are different'

    if any(df1.dtypes != df2.dtypes):
        print('Data Types are different, trying to convert')
        df2 = df2.astype(df1.dtypes)

    # TODO is this really necessary? not an empty df in this case anyway?
    if df1.equals(df2):
        return None
    else:
        # TODO unit test w/ descrepencies in each of the cases.
        # TODO also test w/ nan in list / nan in float column (one / both nan)
        floats1 = df1.select_dtypes(include='float')
        floats2 = df2.select_dtypes(include='float')
        assert set(floats1.columns) == set(floats2.columns)
        diff_mask_floats = ~pd.DataFrame(
            columns=floats1.columns,
            index=df1.index,
            # TODO TODO does this already deal w/ nan correctly?
            # otherwise, this part needs to handle possibility of nan
            # (it does not. need to handle.)
            data=np.isclose(floats1, floats2)
        )
        diff_mask_floats = (diff_mask_floats &
            ~(floats1.isnull() & floats2.isnull()))

        # Just assuming, for now, that array-like cols are same across two dfs.
        arr_cols = arraylike_cols(df1)
        # Also assuming, for now, that no elements of these lists / arrays will
        # be nan (which is currently true).
        diff_mask_arr = ~pd.DataFrame(
            columns=arr_cols,
            index=df1.index,
            data=np.vectorize(np.allclose)(df1[arr_cols], df2[arr_cols])
        )

        # TODO sort?
        other_cols = list(set(df1.columns) - set(floats1.columns) - set(arr_cols))
        other_diff_mask = df1[other_cols] != df2[other_cols]

        diff_mask = pd.concat([diff_mask_floats, diff_mask_arr, other_diff_mask],
            axis=1
        )
        if diff_mask.sum().sum() == 0:
            return None

        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        # TODO are these what i want? prob change id (basically just to index?)?
        # TODO get id from index name of input dfs? and assert only one index
        # (assuming this wouldn't work w/ multiindex w/o modification)?
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
            index=changed.index
        )


def first_group(df, group_cols):
    """Returns key tuple and df of first group, grouping df on group_cols.

    Just for ease of interactively testing out functions on DataFrames of a
    groupby.
    """
    gb = df.groupby(group_cols)
    first_group_tuple = list(gb.groups.keys())[0]
    gdf = gb.get_group(first_group_tuple)
    return first_group_tuple, gdf


def pd_index_equal(a: DataFrameOrSeries, b: DataFrameOrSeries, *, axis: str = 'index',
    only_check_shared_levels: bool  = False,
    assert_no_dupes_in_shared_levels: bool = True,
    if_index_mismatch: Optional[str] = None) -> bool:
    """Returns whether `a` and `b` have equal indices along `axis`.

    Args:
        a|b: DataFrames / Series to compare indices of

        axis: 'index' or 'columns'

        only_check_shared_levels: if True, will only check level names shared
            between both indices, but requires that they share at least one level.
            see `assert_no_dupes_in_shared_levels`.

        assert_no_dupes_in_shared_levels: if True, will assert shared levels do not have
            duplicates in either input index (so that they could be a valid index on
            their own, otherwise, the equality may not mean as much). only relevant if
            `only_check_shared_levels=True`

        if_index_mismatch: None (just return False), 'warn' (warn and return False), or
            'err' (raise ValueError) if indices do not match
    """
    # TODO factor this axis checking into fn [decorator?]?
    assert axis in ('index', 'columns')

    index1 = getattr(a, axis)
    index2 = getattr(b, axis)

    # TODO need/want to check [index|columns].name[s] ever? Series case?
    # TODO TODO option to use pd_allclose here (after converting each to DataFrame)?
    if index1.equals(index2):
        return True

    if only_check_shared_levels:
        shared_names = list(set(index1.names) & set(index2.names))
        assert len(shared_names) > 0, ('need overlapping names in indices!\n'
            f'{index1.names=}\n{index2.names=}'
        )
        df1 = index1.to_frame(index=False)
        df2 = index2.to_frame(index=False)

        if assert_no_dupes_in_shared_levels:
            assert not df1[shared_names].duplicated().any(), (f'a {axis} had duplicates'
                f' when restricted to just {shared_names=}'
            )
            assert not df2[shared_names].duplicated().any(), (f'b {axis} had duplicates'
                f' when restricted to just {shared_names=}'
            )

        # TODO TODO option to use allclose here? (or in general?) sometimes necessary if
        # float values...
        if df1[shared_names].equals(df2[shared_names]):
            return True

    if axis == 'index':
        msg = 'row index mismatch'
    elif axis == 'columns':
        msg = 'columns mismatch'

    if if_index_mismatch == 'warn':
        warnings.warn(msg)
    elif if_index_mismatch == 'err':
        raise ValueError(msg)
    else:
        assert if_index_mismatch is None

    return False


# TODO test
# TODO use in al_analysis.py (and other places i'm checking indices equal w/o checking
# data equal)
# TODO add kwarg for also checking index name[s]?
def pd_indices_equal(a: DataFrameOrSeries, b: DataFrameOrSeries, *,
    if_index_mismatch: Optional[str] = None) -> bool:
    """Returns whether row indices (and column indices, for DataFrames) are equal.

    Args:
        a|b: inputs to compare. must both be either DataFrames or Series.

        if_index_mismatch: whether to return False (None), warn (and return False), or
            raise ValueError ('err') if either row/column indices do not match
    """
    if not pd_index_equal(a, b, axis='index', if_index_mismatch=if_index_mismatch):
        return False

    if isinstance(a, pd.DataFrame):
        # TODO don't assert (share if_index_mismatch / similar for this -> raise
        # ValueError?)?
        assert isinstance(b, pd.DataFrame)

        if not pd_index_equal(a, b, axis='columns',
            if_index_mismatch=if_index_mismatch):

            return False
    else:
        # TODO don't?
        assert isinstance(a, pd.Series) and isinstance(b, pd.Series)

    return True


# TODO add support for dtype mismatch (but where one can be cast to the other)?  maybe
# also support for cols in diff orders? utility fns (maybe exposed behind verbose kwarg)
# for printing differences between dtypes / etc? (use new dtypes_equal fn below for
# this?)
# TODO test
# TODO option to ignore values that are NaN in one and defined in the other? (or
# similar fn for that?)? maybe still (optionally?) warning about how many NaN values are
# only in one or the other? (and also for pd_isclose below, if so)
# TODO TODO default to equal_nan=True (maybe warning if there are NaN unless explicitly
# set True?)?
# TODO at test w/ a str column (mixed w/ other numeric cols)
# ```
# TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could
# not be safely coerced to any supported types according to the casting rule ''safe''
# ```
# i'm now getting w/ pd 1.5.0 (when a column has dtype object) and np 1.24.4
def pd_allclose(a: DataFrameOrSeries, b: DataFrameOrSeries, *,
    if_index_mismatch: Optional[str] = None, **kwargs) -> bool:
    """Like `np.allclose`, but also checks pandas indices are equal.

    Scalars and numpy arrays are also supported (compared with `np.isclose` /
    `np.allclose`). str/object columns of DataFrames are compared for exact equality.

    Args:
        a|b: inputs to compare. must both be either DataFrames or Series.

        if_index_mismatch: passed to `pd_indices_equal`

        **kwargs: passed thru to `np.allclose` (`equal_nan=True` may be the most useful)
    """
    # not importing at module level, since util imports from this module
    from hong2p.util import is_scalar

    a_scalar = is_scalar(a)
    b_scalar = is_scalar(b)
    # TODO or still want to consider equal if DataFrame/Series has one element with
    # matching value? bit of a stretch...
    if (a_scalar and not b_scalar) or (not a_scalar and b_scalar):
        return False

    if a_scalar and b_scalar:
        # seems i could also use allclose here, and not sure output would ever be
        # different? allclose and isclose seem to have same kwarg options too.
        return np.isclose(a, b, **kwargs)

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.allclose(a, b, **kwargs)

    if not pd_indices_equal(a, b, if_index_mismatch=if_index_mismatch):
        return False

    # Series don't have select_dtypes
    if isinstance(a, pd.DataFrame):
        a_obj = select_str_and_object_dtypes(a)
        if len(a_obj.columns) > 0:
            b_obj = select_str_and_object_dtypes(b)
            if not a_obj.equals(b_obj):
                return False

            # since np.allclose below can throw TypeError if there are 'object' dtype
            # columns (like str) in input
            a = select_str_and_object_dtypes(a, exclude=True)
            b = select_str_and_object_dtypes(b, exclude=True)

    return np.allclose(a, b, **kwargs)


def pd_isclose(a: DataFrameOrSeries, b: DataFrameOrSeries, **kwargs) -> bool:
    """Like `np.isclose`, but preserves input indices.

    Args:
        a|b: inputs to compare. must both be either DataFrames or Series.

        **kwargs: passed thru to `np.isclose` (`equal_nan=True` may be the most useful)

    Raises ValueError if indices do not match.
    """
    # not sure what isclose behavior is if shapes mismatch, but only really imagine
    # using this fn for like-indexed stuff anyway
    try:
        pd_indices_equal(a, b, if_index_mismatch='err')
    except ValueError:
        raise

    isclose = np.isclose(a, b, **kwargs)

    # can just use metadata from `a`, since we already know `b` has the same index data
    # from above
    if isinstance(a, pd.DataFrame):
        return pd.DataFrame(index=a.index, columns=a.columns, data=isclose)

    assert isinstance(a, pd.Series)
    return pd.Series(index=a.index, data=isclose)


def dtypes_equal(a: pd.DataFrame, b: pd.DataFrame, *, reorder_cols: bool = True,
    cast: bool = True, verbose: bool = False) -> bool:
    """Returns whether column dtypes of `a` and `b` are equal.

    Args:
        a|b: DataFrames to compare. must have same set of columns.

        reorder_cols: if True, `b` columns are reordered to match `a` before comparing.
            otherwise, asserts columns are in the same order.

        cast: if True (and dtypes differ), tries casting `b` to dtypes of `a` before
            comparing.

        verbose: if True, prints mismatching dtypes
    """

    # NOTE: Series do not seem to have dtypes available on a per-key basis anymore. just
    # has one dtype (and a .dtypes attribute, but still just a scalar, not per-column).
    # tested by doing .iloc to get a Series from a DataFrame with non-uniform values in
    # .dtypes.

    if not reorder_cols:
        assert a.columns.equals(b.columns)
    else:
        if not a.columns.equals(b.columns):
            assert not a.columns.duplicated().any()
            assert not b.columns.duplicated().any()
            assert set(a.columns) == set(b.columns)
            b = b[a.columns]

    if cast and not a.dtypes.equals(b.dtypes):
        # TODO how will this fail if casting not possible? need to catch and return
        # False?
        b = b.astype(a.dtypes)

    equal = a.dtypes.equals(b.dtypes)

    if verbose and not equal:
        print('mismatching dtypes:')
        print(a.dtypes[a.dtypes != b.dtypes])

    return equal


# TODO use (/delete)
def subset_levels(index: pd.MultiIndex, subset: List[str]) -> pd.Index:
    """Returns MultiIndex with only levels named in `subset` (see also
    `subset_index`).
    """
    # does pandas really still not have a function like this? am i missing something?
    # TODO work w/ one level index? change type hint to `-> pd.MultiIndex` if so?
    # if not, fix
    return pd.MultiIndex.from_frame(index.to_frame(index=True)[subset])


# TODO type hint in way to indicate if input will have same type as output? some
# pythonic way to do that?
# TODO also replace usage of reindex in natmix (and anything else important. drosolf?
# odor analysis stuff?) with this.
def reindex(x: DataFrameOrSeries, index: pd.Index, checks: bool = True, **kwargs
    ) -> DataFrameOrSeries:
    """Reindexes data to index, after checking index is unique (to avoid surprises).

    Args:
        x: data to reindex

        index: index to conform data to. The output index (or output column index,
            if `x` is a DataFrame and `axis='columns'`) will equal this index.

            Will raise AssertionError if this index contains any duplicates.

        checks: if True, checks output index equals `index`, and that the other index
            (for DataFrames) is unchanged

        **kwargs: passed thru to `<x>.reindex(...)`
    """
    # TODO also support being called w/ columns= / index= (w/ dataframe input), rather
    # than labels= + axis=? (and assert neither is in kwargs for now? and/or that
    # 'labels' is also not?)
    assert all(x not in kwargs for x in ('columns', 'labels', 'index')), \
        'these args currently unsupported by wrapper. would not be hard to add.'

    # TODO allow anything other than index type for this? maybe Series/DataFrame?
    #
    # maybe the "scalar, list-like, dict-like or functions transformations to apply to
    # that axis’ values" that current Series.reindex talk about?
    #
    # or an array-like that labels= (/index/columns) arg for DataFrame.reindex talk
    # about (but those docs do say Index type preferred, so probably don't care to
    # support anything else there)?
    assert isinstance(index, pd.Index), f'{type(index)=} was not Index/MultiIndex'

    # TODO maybe only do this if level= not passed? and do some separate check there?
    # see if that arg is useful to me in al_analysis.mb_model.reindex_to_wPNKC (for
    # `one_row_per_claw and not prat_claws` case)
    assert not index.duplicated().any(), ('index had duplicates! reindex output might '
        'not be as expected!'
    )
    if checks:
        axis = kwargs.get('axis', 'index')
        if axis == 'columns' or axis == 1:
            other_index = x.index.copy()
        else:
            assert axis == 'index' or axis == 0
            if isinstance(x, pd.DataFrame):
                other_index = x.columns.copy()

    reindexed = x.reindex(index, **kwargs)
    if checks:
        if axis == 'columns' or axis == 1:
            assert reindexed.columns.equals(index)
            # check we haven't changed other index in process
            assert other_index.equals(reindexed.index)
        else:
            assert reindexed.index.equals(index)
            if isinstance(x, pd.DataFrame):
                assert other_index.equals(reindexed.columns)

    return reindexed


# TODO add unit tests w/ inputs that are both MultiIndex / not
def index2dict_list(index: pd.Index) -> List[Dict[str, Any]]:
    """Returns list of dicts, each with `index.names` keys and index values.

    Raises ValueError if `index.names` elements are not all of type `str`.
    """
    names = index.names
    if {type(x) for x in index.names} != {str}:
        # TODO also say what types we currently have
        raise ValueError('index.names must all be of type `str`')

    def _index_entry_dict(index: pd.Index, index_entry: Any) -> Dict[str, Any]:
        # If the index is a MultiIndex, iterating over it should yield tuples of the
        # level values, otherwise we'll convert to a tuple so zipping is the same.
        if type(index_entry) is not tuple:
            # only non-MultiIndex index should have non-tuple index_entry (otherwise
            # wrong index may have been passed in)
            # https://stackoverflow.com/questions/21081042
            assert not isinstance(index, pd.MultiIndex)
            index_entry = (index_entry,)

        elif not isinstance(index, pd.MultiIndex):
            # for case where index has only one level with tuple values
            # (e.g. key -> (False, 1, ...) in natmix_data/analysis.py
            # sort_rois_by_response_classes)
            index_entry = [index_entry]
            # TODO also test single level multiindex w/ tuple values? possible? it
            # already have length-1 values (b/c of some outer iterable containing them)?

        # TODO also test case where multiindex has one level w/ tuple values. i assume
        # all this should still work there?

        assert len(names) == len(index_entry), f'{names=}\n{index_entry=}'

        return dict(zip(names, index_entry))

    return [_index_entry_dict(index, x) for x in index]


def frame_pdist(df: pd.DataFrame, *, metric='euclidean') -> pd.DataFrame:
    """Returns a DataFrame matrix of all pairwise distances between input rows.

    Returns output of same shape as DataFrame.corr() (calculating distances between
    all pairs of input *rows*), so output rows and columns will be same as input
    columns.
    """
    from scipy.spatial.distance import pdist, squareform
    return pd.DataFrame(squareform(pdist(df.T, metric=metric)), index=df.columns,
        columns=df.columns
    )


# TODO rename from melt (if that's not the closest-functionality pandas fn)?
# TODO still want all these kwargs?
# TODO get to work w/ xarray input too
# TODO TODO and also support case where input DataArray has one dimension along
# which a series of symmetric matrices have been concatenated, and handle that too
# (e.g. shape (5, 27, 27) series of 5 correlation matrices, one for each of 5 flies)
# TODO test that it also works w/o MultiIndex indices
def melt_symmetric(symmetric_df, drop_constant_levels=True,
    suffixes=('_a', '_b'), name=None, keep_duplicate_values=False):
    """Takes a symmetric DataFrame to a tidy Series with unique values.

    Symmetric means the row and columns indices are equal, and values should
    be a symmetric matrix.

    Output index has levels from both row and column indices, suffixed with `suffixes`
    (unless they already were).

    Args:
        symmetric_df: data to melt. rows/columns that are all NaN are dropped.

        drop_constant_levels: if True, drops (MultiIndex) levels with only one value

        suffixes: added to row and column index level names, respectively

        name: name of output Series

        keep_duplicate_values: if False, only upper triangle (including diagonal) is
            kept. otherwise, all values are kept.

    See `invert_melt_symmetric` for inverse.
    """
    # This is actually not dependent on the level names / .name (if we wanted to check
    # those + the types as well, we'd use .identical(...)), so it's ok if the
    # columns/index are already e.g. called 'odor' & 'odor_b', respectively.
    assert symmetric_df.columns.equals(symmetric_df.index)

    # TODO put behind a flag at the minimum?
    symmetric_df = symmetric_df.copy()
    symmetric_df.dropna(how='all', axis=0, inplace=True)
    symmetric_df.dropna(how='all', axis=1, inplace=True)
    assert symmetric_df.notnull().all(axis=None), 'not tested w/ non-all NaN'

    multiindices = False
    # Assuming index/columns are the same type (either MultiIndex or not)
    if isinstance(symmetric_df.index, pd.MultiIndex):
        multiindices = True

    # If identical(...) fails, that should mean the .names/.name attributes differed
    # (formally the dtypes could have differed too), as the .equals(...) assertion above
    # passed.
    if not symmetric_df.columns.identical(symmetric_df.index):
        # TODO factor out?
        def get_index_names(index):
            return index.names if multiindices else [index.name]

        col_names =  get_index_names(symmetric_df.columns)
        index_names = get_index_names(symmetric_df.index)

        # TODO use ValueError rather than assertions
        assert all(
            x.endswith(suffixes[1]) for x in col_names
        )
        assert all(
            x.endswith(suffixes[0]) for x in index_names
        )
    else:
        # TODO make all of this index name/level names renaming conditional on kwarg?
        if multiindices:
            # TODO adapt to work in non-multiindex case too! (rename there?)
            symmetric_df.index.rename([n + suffixes[0] for n in
                symmetric_df.index.names], inplace=True
            )
            symmetric_df.columns.rename([n + suffixes[1] for n in
                symmetric_df.columns.names], inplace=True
            )
        else:
            # TODO TODO maybe i still want to do this in the multiindex case tho?
            # would it break my old code that used this in kc_natural_mixes?
            old_name = symmetric_df.index.name
            assert old_name == symmetric_df.columns.name
            symmetric_df.index.name = f'{old_name}{suffixes[0]}'
            symmetric_df.columns.name = f'{old_name}{suffixes[1]}'

    # TODO maybe always test that triu equals tril tho (or w/ a _checks=True set)

    # To de-clutter what would otherwise become a highly-nested index.
    if multiindices and drop_constant_levels:
        # TODO may need to call index.remove_unused_levels() first, if using
        # levels here... (see docs of that remove fn)
        constant_levels = [i for i, levels in enumerate(symmetric_df.index.levels)
            if len(levels) == 1
        ]
        symmetric_df = symmetric_df.droplevel(constant_levels, axis='index')
        symmetric_df = symmetric_df.droplevel(constant_levels, axis='columns')

    # TODO maybe an option to interleave the new index names
    # (so it's like name1_a, name1_b, ... rather than *_a, *_b)
    # or would that not ever really be useful?

    if keep_duplicate_values:
        tidy = symmetric_df.stack(level=symmetric_df.columns.names)
        assert tidy.shape == (np.prod(symmetric_df.shape),)
    else:
        # From: https://stackoverflow.com/questions/34417685
        keep = np.triu(np.ones(symmetric_df.shape)).astype('bool')
        masked = symmetric_df.where(keep)
        n_nonnull = masked.notnull().sum().sum()
        # We already know both elements of shape are the same from equality
        # check on indices above.
        n = symmetric_df.shape[0]
        # The right expression is the number of elements expected for the
        # triangular of a square matrix w/ side length n, if the diagonal
        # is INCLUDED.
        assert n_nonnull == (n * (n + 1) / 2)

        # TODO make sure this also still works in non-multiindex case!
        # dropna for pandas >= 3.0, where stack no longer drops NaN by default
        tidy = masked.stack(level=masked.columns.names).dropna()
        assert tidy.shape == (n_nonnull,)

    tidy.name = name
    return tidy


# TODO rename if it could make it more accurate
def invert_melt_symmetric(ser, suffixes=('_a', '_b')):
    """Takes Series (as output by `melt_symmetric`) back to a square DataFrame.

    Levels ending in first suffix (that have a matching level ending in the second
    suffix) are unstacked into columns. Levels without such a pair are dropped. Only
    values in `ser` are filled, so output will have NaN in other triangle, if `ser` only
    has one triangle (as by default in `melt_symmetric`).
    """
    assert len(ser.shape) == 1, 'not a series'
    assert len(ser.index.names) == len(set(ser.index.names)), \
        'index names should be unique'

    assert len(suffixes) == 2 and len(set(suffixes)) == 2
    s0, s1 = suffixes

    levels_to_drop = set(ser.index.names)
    col_prefixes = []
    for c in ser.index.names:
        if type(c) is not str:
            continue

        if c.endswith(s0):
            # Not c[:-len(s0)], which is '' when s0 == ''
            prefix = c[:len(c) - len(s0)]
            if (prefix + s1) in ser.index.names:
                col_prefixes.append(prefix)
                levels_to_drop.remove(prefix + s0)
                levels_to_drop.remove(prefix + s1)

    levels_to_drop = list(levels_to_drop)
    # This does also work in the case where `levels_to_drop` is empty.
    ser = ser.droplevel(levels_to_drop)
    return ser.unstack([p + s0 for p in col_prefixes])


def check_index_vals_unique(df: pd.DataFrame) -> None:
    """Raises AssertionError if any duplicates in column/index indices.
    """
    assert not (df.index.duplicated().any() or df.columns.duplicated().any())


# TODO already have something like this?
# TODO typehint subclass of Index? already good on that?
def get_axis_index(df: pd.DataFrame, axis: Union[int, str]) -> pd.Index:
    """Returns `df.index` (axis=0/'index') or `df.columns` (axis=1/'columns').
    """
    if axis == 0 or axis == 'index':
        return df.index
    elif axis == 1 or axis == 'columns':
        return df.columns
    else:
        raise ValueError(f'invalid axis: {axis}')


def suffix_index_names(df: pd.DataFrame, suffix: str = '_b', axis='columns') -> None:
    """Adds `suffix` to all level names of index along `axis` (inplace).
    """
    # TODO TODO also support non-multiindex w/ .name
    # TODO factor out helper to get df
    index = get_axis_index(df, axis)
    index.names = [f'{x}{suffix}' for x in index.names]
    # TODO make not inplace?
