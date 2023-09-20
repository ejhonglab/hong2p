"""
Utility functions for working with xarray objects, primarily DataArrays
"""

from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
import xarray as xr

from hong2p.util import suffix_index_names


# NOTE: lots of my current MultiIndex related woes should be tractable once development
# on xarray's own "Explicit indexes" completes here:
# https://github.com/pydata/xarray/projects/1
#
# See also: https://github.com/pydata/xarray/issues/6293 (if project number changes)

# TODO fn for getting all coordinates (in index or not?) associated w/ particular
# dimension
# TODO how do i even add new scalar coords? should i make a helper to add a bunch of
# them?
def is_scalar_coord(arr: xr.DataArray) -> bool:
    """Whether item from <DataArray>.coords.values() corresponds to a scalar coordinate.
    """
    # TODO shape == (1,) ever happen?
    return len(arr.shape) == 0


# TODO a to_pandas function, wrapping existing, and using this to add non-dimension
# coordinates to (one of?) the (multi)indices. also have it (at least by default)
# call move_all_coords_to_index first
# TODO maybe add kwarg flag to return as a pandas Series instead?
# TODO non-dim only flag? or kwarg for a specific dim only?
def scalar_coords(arr: xr.DataArray) -> dict:
    # TODO TODO clarify (renaming if necessary) + test whether this should [only] work
    # with scalar coords associated with a particular dimension (and maybe it should
    # only be coordinates not associated with any dimension).
    # TODO will series constructor already convert to Timestamp as i'd want? or nah (is
    # dtype lost?)?
    """Returns all scalar coordinates as a dict

    Converts any datatime64[ns] to pandas Timestamp
    """
    # TODO need to account for endianness (the > or < in the dtype str, e.g.
    # dtype('<M8[ns]')) in conversion to Timestamp?
    # TODO do i want to handle coords that are repeated, but also just have a single
    # unique coordinate?
    return {k: v.item(0)
        if not np.issubdtype(v.dtype, np.datetime64) else pd.Timestamp(v.item(0))
        for k, v in arr.coords.items() if is_scalar_coord(v)
    }


def drop_scalar_coords(arr: xr.DataArray) -> xr.DataArray:
    """Returns DataArray with all scalar coordinates dropped.
    """
    to_drop = [k for k, v in arr.coords.items() if is_scalar_coord(v)]
    return arr.drop_vars(to_drop)


def assign_scalar_coords_to_dim(arr: xr.DataArray, dim: str) -> xr.DataArray:
    """Returns DataArray with all scalar coordinates moved to dimension `dim`.

    Assumes scalar coordinates are non-dimension coordinates.

    May also depend on dim being of size >1 (at least for some intended uses)
    """
    if dim not in arr.sizes:
        raise ValueError(f'{dim} not in {arr.dims=}')

    coord_dict = scalar_coords(arr)
    return arr.assign_coords({
        k: (dim, arr.sizes[dim] * [v]) for k, v in coord_dict.items()
    })


# TODO TODO should this, and/or scalar_coords above, be converting datetime64[ns]
# (xarray represetation) to pd.Timestamp (constructor of latter takes former directly)?
# TODO Any or hashable?
def unique_coord_value(arr: xr.DataArray, coord: Optional[str] = None) -> Any:
    """Returns unique value for a DataArray coordinate, if there is one.

    If `arr[coord]` has multiple values, raises `ValueError`.

    Can also call like `unique_coord_value(arr.<coord_name>)`.
    """
    # To handle case when called like unique_coord_value(arr.<coord_name>), where the
    # arr will still be a DataArray
    if coord is None:
        # arr.<coord_name> gives you a DataArray with name == <coord_name>
        assert arr.name in arr.coords and len(arr.shape) == 1
        coord = arr.name

    ser = arr[coord].to_pandas()
    # TODO TODO also work when len(ser.shape) == 0. docs and linked source code:
    # https://docs.xarray.dev/en/stable/generated/xarray.DataArray.to_pandas.html
    # seem to indicate it should be of type DataArray if the shape is length 0
    # (0-dimensional), but at least now w/ xarray 0.19.0, that seems to not be the case
    # (seems to be of type numpy.datetime64 in one case where that's the value)
    # TODO work w/ old and new version ideally, or at least new versions and upgrade
    # from 0.19.0
    assert len(ser.shape) == 1, f'coord: {coord}, ser: {ser}'

    unique = ser.unique()
    if len(unique) != 1:
        raise ValueError(f"arr['{coord}'] did not have a unique value:\n {unique}")

    return unique[0]


# TODO option to just do for a subset of dimensions?
# TODO rename index->indices?
# TODO allow passing in str list of order for them to be in (could also be the subset of
# names to use as new index, assuming there would be any value add over set_index)
def move_all_coords_to_index(arr: xr.DataArray) -> xr.DataArray:
    """Returns array with all coord variables associated with a dim to index on that dim
    """
    if len(arr.indexes) > 0:
        # arr.indexes is of type xarray.core.indexs.Indexes, and will give names of dims
        # as keys when iterating over it.
        for indexed_dim in arr.indexes:
            arr = arr.reset_index(indexed_dim)

    dim2coords = dict()
    for dim_name in arr.dims:
        # Of type DataArrayCoordinates
        dim_coords = arr[dim_name].coords

        # Iterating over the coordinates just gives us the names of each, like a
        # iterating over a dict would just give us the keys
        dim_coord_names = list(dim_coords)

        dim2coords[dim_name] = dim_coord_names

    return arr.set_index(dim2coords)


def odor_corr_frame_to_dataarray(df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None) -> xr.DataArray:
    """Converts (odor X odor) correlation DataFrame to DataArray.
    """
    # TODO also need to check it's .names that are defined and not .name? .names
    # should be None in that case, causing this to fail, right?
    assert set(df.index.names) == set(df.columns.names)

    for index in (df.index, df.columns):
        # Will often also have 'odor2' and 'repeat' levels, but potentially
        # additional metadata too.
        assert 'odor1' in index.names

    if metadata is not None:
        for k in metadata.keys():
            assert k not in df.index.names, f'metadata key {k} already in index'

    # Since util.suffix_index_names is currently inplace
    df = df.copy()

    # Adds '_b' to end of all column names
    suffix_index_names(df)

    # TODO TODO why does corr.odor1 work if odor1 is an input MultiIndex level but
    # not just the name of an Index? 'odor1' is in index.names either way...
    # for normal input (with MultiIndex rows/cols), coords look like:
    # Coordinates:
    #     odor1     (odor) object '1-5ol @ -3' '1-5ol @ -3' ... 'va @ -3' 'va @ -3'
    #     odor2     (odor) object 'solvent' 'solvent' ... 'solvent' 'solvent'
    #     repeat    (odor) int64 0 1 2 0 1 2 0 1 2 0 1 2 0 ... 0 1 2 0 1 2 0 1 2 0 1 2
    #     odor1_b   (odor_b) object '1-5ol @ -3' '1-5ol @ -3' ... 'va @ -3' 'va @ -3'
    #     odor2_b   (odor_b) object 'solvent' 'solvent' ... 'solvent' 'solvent'
    #     repeat_b  (odor_b) int64 0 1 2 0 1 2 0 1 2 0 1 2 ... 0 1 2 0 1 2 0 1 2 0 1 2
    # Dimensions without coordinates: odor, odor_b
    #
    # (c1)
    # coordinates look like (for input w/ just 'odor1' X 'odor1_b' Index name[s]):
    # Coordinates:
    #   * odor     (odor) object '1-5ol @ -3' '1-6ol @ -3' ... 't2h @ -3' 'va @ -3'
    #   * odor_b   (odor_b) object '1-5ol @ -3' '1-6ol @ -3' ... 't2h @ -3' 'va @ -3'
    #
    # (c2)
    # if i change this call to xr.DataArray(df), then coordinates looks like this:
    # Coordinates:
    #   * odor1    (odor1) object '1-5ol @ -3' '1-6ol @ -3' ... 't2h @ -3' 'va @ -3'
    #   * odor1_b  (odor1_b) object '1-5ol @ -3' '1-6ol @ -3' ... 't2h @ -3' 'va @ -3'
    #
    # c1.set_index({'odor1': 'odor', 'odor1_b': 'odor_b'}
    # just converts c1 (above) to c2
    # TODO what happens if i add an empty level and then drop it though? possible?
    #
    corr = xr.DataArray(df, dims=['odor', 'odor_b'])

    if metadata is not None:
        corr = corr.assign_coords(metadata)

    return corr

