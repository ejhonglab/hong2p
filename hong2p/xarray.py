"""
Utility functions for working with xarray objects, primarily DataArrays
"""

from typing import Optional, Any

import xarray as xr


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
    assert len(ser.shape) == 1, f'coord: {coord}, ser: {ser}'

    unique = ser.unique()
    if len(unique) != 1:
        raise ValueError(f"arr['{coord}'] did not have a unique value:\n {unique}")

    return unique[0]


# TODO option to just do for a subset of dimensions?
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

