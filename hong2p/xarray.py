"""
Utility functions for working with xarray objects, primarily DataArrays
"""

from typing import Optional, Any

import numpy as np
import pandas as pd
import xarray as xr


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

