"""
Utility functions for working with xarray objects, primarily DataArrays
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

from importlib_resources import files
import numpy as np
import pandas as pd
import xarray as xr

from hong2p.util import suffix_index_names
from hong2p.pandas import subset_index


# TODO switch to h5netcdf? docs say it can sometimes be faster. is it, for me?
# h5netcdf (according to latest xarray docs) also supports more compression options
# behind the to_netcdf encoding=<dict> kwarg. from 2026 docs:
# "The h5netcdf engine supports both the NetCDF4-style compression encoding parameters
# {"zlib": True, "complevel": 9} and the h5py ones {"compression": "gzip",
# "compression_opts": 9}. This allows using any compression plugin installed in the HDF5
# library, e.g. LZF."
# TODO add script to benchmark size / load / write times for diff engines and
# compression levels? use claw_sims?
NETCDF_ENGINE: str = 'netcdf4'

def load_dataarray(path: Path) -> xr.DataArray:
    """Loads DataArray saved via `save_datarray`, and restores any MultiIndex coords.

    Makes same assumptions that `save_dataarray` does; namely that we should call
    `move_all_coords_to_index(arr)` on the loaded data, to recreate input to
    `save_datarray`. This may not be true for some NetCDF files saved with `to_netcdf`
    directly, or anything other than `save_dataarray`.
    """
    arr = xr.open_dataarray(path, engine=NETCDF_ENGINE)
    arr = move_all_coords_to_index(arr)
    return arr


# TODO does zarr work w/ MultiIndex in newer xarray (0.19 doesn't support zarr)
# TODO (delete) want to use cf_xarray encode/decode, or just assume everything i'll
# want to write can have move_all_coords_to_index called on load (and then reset
# all indices before saving)? does latter approach produce files that are more useful to
# people not using cf_xarray?
# https://cf-xarray.readthedocs.io/en/latest/coding.html
# https://github.com/arviz-devs/arviz/issues/2165
# https://github.com/pydata/xarray/issues/1077
# current IO docs don't mention MultiIndex:
# https://docs.xarray.dev/en/stable/user-guide/io.html

def save_dataarray(arr: xr.DataArray, path: Path, *, check: bool = True, **kwargs
    ) -> None:
    """Writes `arr` to NetCDF4 at `path`.

    Currently requires that input does not change after being passed through:
    `reset_all_multiindex(move_all_coords_to_index(arr))`, since we need to reset
    MultiIndex coords for saving to netCDF to work, and if we can't assume we can move
    them all back to MultiIndex (when possible) on load, we wouldn't currently know how
    to recreate the saved data.

    Args:
        check: if True, check we get exactly the input `arr`, if we load the saved file

        **kwargs: passed to `<DataArray>.to_netcdf`. can be used to set compression, via
            `encoding=<dict>` kwarg. see xarray `to_netcdf` docs.
    """
    # TODO assert no attrs? can those be saved too? move to scalar coords (assuming even
    # those work... but i can probably make those work?). test
    reset = reset_all_multiindex(arr)

    encoding_kws = dict()
    if 'encoding' in kwargs:
        # TODO try h5netcdf engine w/ compression=gzip, compression_opts=9 (not
        # supported by netcdf4 engine)? and compare it's default performance?
        #
        # so far i've only tried encoding=dict(zlib=True, complevel=9), which i think is
        # all that is supported by netcdf4 engine
        encoding = kwargs.pop('encoding')
        assert isinstance(encoding, dict), f'{encoding=}'
        assert all(type(x) is str for x in encoding.keys()), f'{encoding=}'

        DEFAULT_XARRAY_NETCDF_VAR_NAME: str = '__xarray_dataarray_variable__'
        # from xarray docs: "Only xarray.Dataset objects can be written to netCDF files,
        # so the xarray.DataArray is converted to a xarray.Dataset object containing a
        # single variable.  If the DataArray has no name, or if the name is the same as
        # a coordinate name, then it is given the name "__xarray_dataarray_variable__"."
        if arr.name is None or arr.name in reset.coords:
            var_name = DEFAULT_XARRAY_NETCDF_VAR_NAME
        else:
            var_name = arr.name

        encoding_kws = dict(encoding={var_name: encoding})

    # need to check we can use the same approach to restore all MultiIndex coordinates
    # on load (or else, would need to look into using cf_xarray MultiIndex encoding
    # support, or some other alternatives)
    if not arr.identical(move_all_coords_to_index(reset)):
        raise ValueError('can only use with input where move_all_coords_to_index does '
            'produces the same array+coords as its input'
        )
    reset.to_netcdf(path, engine=NETCDF_ENGINE, **kwargs, **encoding_kws)
    del reset

    if check:
        arr2 = load_dataarray(path)
        assert arr.identical(arr2)


# al_analysis tests that use this also work w/ this
data_dir = files('hong2p')
orn_test_data = data_dir / 'orn_dynamics_test.nc'
def get_example_orn_dynamics() -> xr.DataArray:
    """Returns an example DataArray (small subset of model ORN dynamics)
    """
    arr = load_dataarray(orn_test_data)
    return arr


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


def coords_equal(x: xr.DataArray, y: xr.DataArray) -> bool:
    """Checks set of dims and all variables under coordinates are equal.

    For older versions of xarray without `Coordinates.equals()/.identical()`
    """
    # TODO when was Coordinates.equals()/identical() added? how to implementent in
    # meantime, if i can't upgrade xarray? seems not in 0.19 (what i've been using) or
    # 0.21 (last version before many vYYYY.MM.N versions, starting in 2022)
    x = x.coords
    y = y.coords

    # NOTE: there can be elments of dims that are not in things checked below
    # e.g. orns.reset_index('stim') still has a 'stim' dim, but no associated index
    # levels ('odor' / 'panel' still exist as non-scalar metadata)
    #
    # TODO want to require order to be the same?
    if set(x.dims) != set(y.dims):
        return False

    # if this fails, would need to decide which of keys/items/variables.
    # i don't see any other attributes under dir(coords) (in 0.19) that seem useful.
    # only other thinsg are dims/indexes/values()/xindexes/_data/_names
    def _check_options_equiv(coords):
        k0 = set(x for x in coords)
        keys = set(coords.keys())
        item_keys = set(k for k, v in coords.items())
        var_keys = set(coords.variables.keys())
        assert keys == k0
        assert keys == item_keys
        assert keys == var_keys

    _check_options_equiv(x)
    _check_options_equiv(y)

    # TODO possible to have non-index elements still associated with one dimension
    # (don't think so? can have non-scalar not in indices, but not sure any of those can
    # be associated w/ a dimension?)? what does that look like? still wouldn't show up
    # in .keys() / .variables i assume, since those only contain outer indices?
    xk = set(x.keys())
    yk = set(y.keys())
    if xk != yk:
        return False

    # TODO also sort on names to allow this to not depend on order?
    # (would prob be easiest to convert to frame and sort columns)
    # NOTE: to_index() output has int index for stuff in dims but not in index
    # associated names
    #
    # there should be a level for each variable in each dimension-specific multiindex
    # here, and no longer any name for dimensions (at least not those [only?] containing
    # MultiIndex values) (because index level names can't seem to be equal to dimension
    # name)
    x_index = x.to_index()
    y_index = y.to_index()
    if not x_index.equals(y_index):
        return False

    # we already checked above both of these same in x/y
    dims = x.dims
    index_names = x_index.names
    for k in xk:
        if k in index_names:
            continue

        # was name of dimension with a MultiIndex, and thus can not be a name of a
        # variable to check (i.e. one of the index levels, or something not associated
        # with a dimension)
        if k in dims:
            continue

        vx = x[k]
        vy = y[k]
        if vx.shape == tuple():
            assert vy.shape == tuple()
            if vx.item() != vy.item():
                return False

        eq0 = vx.equals(vy)
        # just checking we don't need to use anything other than .equals()
        # could delete eventually
        assert eq0 == np.array_equal(vx.values, vy.values)
        if not eq0:
            return False

    return True


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


# TODO TODO test when some non-scalar level has already been processed by reset_index()
def all_coord_names(arr: xr.DataArray) -> List[str]:
    """Returns all coordinate variables names, excluding names containing MultiIndex

    Some dims can have a MultiIndex associated with them, where there is one name in the
    coordinates that should equal a name in `dims`, but does not equal any of the names
    of the levels in the MultiIndex it contains. This function will exclude those outer
    names from the output.

    Calling `reset_index()` on all dimensions should produce a `DataArray` where the
    coordinate variable names match those returned here.
    """
    names = []
    coords = arr.coords
    for x in coords:
        index = None
        try:
            index = coords[x].to_index()

        # should get something like this if x is a scalar variable not associated with a
        # coordinate:
        # ValueError: IndexVariable objects must be 1-dimensional
        except ValueError:
            assert isinstance(x, str)
            names.append(x)

        if index is not None:
            if index.nlevels == 1:
                # if we reset_index(c) for some c with a MultiIndex, when x is any level
                # from that MultiIndex, the .names will just contain c, not x
                names.append(x)
            else:
                names.extend(list(index.names))

    return names


# TODO test
# TODO add option to only reset a subset (or a particular one)?
def reset_all_multiindex(arr: xr.DataArray) -> xr.DataArray:
    vars_before = all_coord_names(arr)
    dims = arr.dims
    if len(arr.indexes) > 0:
        # arr.indexes is of type xarray.core.indexs.Indexes, and will give names of dims
        # as keys when iterating over it.
        for indexed_dim, index in arr.indexes.items():
            # now that i've added the `and indexed_dim in dims` part, seems to also work
            # in xarray 2023.1.0, not just the 0.19 i was using before
            if index.nlevels > 1 and indexed_dim in dims:
                # why am i now getting '_' suffixed to names of keys i
                # reset_index() on? (only for things not part of a multiindex) (oh, it's
                # probably because coord names can't match dim names exactly)
                # TODO problem that i'm no longer reset_index()-ing the non-multi
                # indexes? test
                # TODO add flag to have option to still reset those?
                arr = arr.reset_index(indexed_dim)

    vars_after = all_coord_names(arr)
    # TODO sort array coords before checking equal? add flag to coords_equal for
    # that? something already existing in xarray for that?
    assert set(vars_after) == set(vars_before), f'{vars_after} != {vars_before}'
    return arr


# TODO option to just do for a subset of dimensions?
# TODO rename index->indices?
# TODO allow passing in str list of order for them to be in (could also be the subset of
# names to use as new index, assuming there would be any value add over set_index)
# TODO TODO give more sensible error when this is called w/ input that has scalar
# coordinate values (not associated w/ dims). currently get:
# TypeError: unhashable type: 'numpy.ndarray'
# on this input:
# <xarray.DataArray (stim: 2, glomerulus: 3, time_s: 4)>
# array([[[13.26, 13.26, 13.26, 13.26],
#         [13.26, 13.26, 13.26, 13.26],
#         [13.26, 13.26, 13.26, 13.26]],
#
#        [[13.26, 13.26, 13.26, 13.26],
#         [13.26, 13.26, 13.26, 13.26],
#         [13.26, 13.26, 13.26, 13.26]]])
# Coordinates:
#   * stim        (stim) MultiIndex
#   - panel       (stim) object 'megamat' 'megamat'
#   - odor        (stim) object '2h @ -3' 'IaA @ -3'
#   * time_s      (time_s) float64 -0.4995 -0.499 -0.4985 -0.498
#   * glomerulus  (glomerulus) object 'D' 'DA1' 'DA2'
#     x           int64 1
# TODO test
def move_all_coords_to_index(arr: xr.DataArray) -> xr.DataArray:
    """Returns array with all coord variables associated with a dim to index on that dim
    """
    arr = reset_all_multiindex(arr)

    dim2coords = dict()
    for dim_name in arr.dims:
        # Of type DataArrayCoordinates
        dim_coords = arr[dim_name].coords
        # Iterating over the coordinates just gives us the names of each, like a
        # iterating over a dict would just give us the keys
        dim_coord_names = list(dim_coords)
        dim2coords[dim_name] = dim_coord_names

    arr = arr.set_index(dim2coords)
    return arr


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


# TODO how to type hint coordinates in current xr? (replace Any with that)
def find_matching_coords(index: pd.Index, arr: xr.DataArray, *, warn: bool = True
    ) -> Optional[Any]:
    """Checks whether `data` and `arr` have compatible indices, returns matching coords.

    Args:
        index: pandas index to find matching coords for (from a single dim in `arr`)

        arr: DataArray that should contain a single dimension with coordinates that
            match index. Index levels names (and contents, currently including sort
            order) should all match.

            The set of levels that match must also be unique, so that they could
            actually be used to reindex along this dimension. Only dimensions with
            length matching `len(index)` are checked for matching.

        warn: if True, will warn about any coords with same length but no overlapping
            names, as well as any level names in one index but not the other
    """
    matching_coords = None
    for name, coords in arr.coords.items():
        if arr.sizes[name] != len(index):
            continue

        arr_index = coords.to_index()
        if index.equals(arr_index):
            matching_coords = coords
            # TODO or check no other matching coords? very unarrly anything would match
            # below for multiple, or that i'd care which was chosen (if they did)
            break

        index_names = list(index.names)
        arr_names = list(arr_index.names)
        shared_names = [n for n in index_names if n in arr_names]

        # TODO update warning/assertion messages below to refer to the specific coord
        # (/dim) name being checked, rather than just 'arr'

        if len(shared_names) == 0:
            if warn:
                warnings.warn('{name=} size matches, but no matching index level names')
            continue

        if warn and index_names != arr_names:
            index_only_names = [x for x in index_names if x not in arr_names]
            if len(index_only_names) > 0:
                warnings.warn(f'names only in index (not arr): {index_only_names}')

            arr_only_names = [x for x in arr_names if x not in index_names]
            if len(arr_only_names) > 0:
                warnings.warn(f'names only in arr (not index): {arr_only_names}')

        # otherwise would have to decide if it ever matters which order we go with
        # (from the series or the DataArray? prob latter, if this ever trips)
        assert shared_names == [n for n in arr_names if n in shared_names], \
            'index and arr have index names in different orders!'

        common = subset_index(arr_index, shared_names)
        assert not common.duplicated().any(), (f'duplicates in {shared_names=} subset '
            'of arr index'
        )
        index_common = subset_index(index, shared_names)
        # this should imply nothing duplicated in index_common either
        assert common.equals(index_common), ('index and arr indices did not match, even'
            f' after subsetting both to only {shared_names=}. maybe different order?'
        )
        # TODO support stuff in wrong order too, if we can sort into right order?
        # (also warning, if warn=True)

        matching_coords = coords
        break

    return matching_coords


# TODO also implement a version that is just `series2xarray`, and doesn't need to get
# coords from other array
# NOTE: don't seem to need a similar fn for dataframes. xr.DataArray(df) works fine,
# and might just want to pass dims=<list-of-str>
# TODO TODO update xarray? in recent versions, i could probably define coords from
# xr.Coordinates, by passing series index. that constructor is not available in 0.19
def series2xarray_like(ser: pd.Series, like: xr.DataArray, *, warn: bool = True
    ) -> xr.DataArray:
    """Checks `ser` and `like` have compatible indices, and converts `ser` to DataArray.

    Args:
        ser: Series to convert to 1D DataArray

        like: DataArray that should contain one coordinate with index levels (and
            contents, currently including sort order) that match `ser.index`.
            See `find_matching_coords`.

        warn: passed to `find_matching_coords`
    """
    matching_coords = find_matching_coords(ser.index, like, warn=warn)
    assert matching_coords is not None, (f'found no matching coord between:\n'
        f'{ser.index=}\n...and...\n{like.coords=}'
    )
    arr = xr.DataArray(data=ser.values, coords=matching_coords.coords, name=ser.name)
    assert arr.shape == ser.shape
    assert arr.isnull().sum().item() == ser.isnull().sum()
    return arr


def outer_product(x: xr.DataArray, y: xr.DataArray, *, squeeze: bool = True,
    drop: bool = True) -> xr.DataArray:
    """Returns outer product of two DataArrays.

    Output shape is the concatenation of input shapes (`x.shape + y.shape`).

    One should be missing a name (`.name == None`) or both should have matching
    `.name`s.

    Assumed that they will have non-overlapping dims, as otherwise what would be the
    point of an outer product.

    Args:
        x,y: DataArrays to multiply

        squeeze: whether to `.squeeze()` inputs before multiplying

        drop: if `squeeze=True`, passes this to `squeeze()`
    """
    # TODO remove this restriction too? want to support other types of multiplication
    # that are not outer products? would have to rename fn if so...
    overlapping_dims = set(x.dims) & set(y.dims)
    assert len(overlapping_dims) == 0, f'{overlapping_dims=}'

    if squeeze:
        x = x.squeeze(drop=drop)
        y = y.squeeze(drop=drop)

    # oh, actually it does work if both names are None. just didn't work when i was
    # trying to multiply against a Series.
    if x.name is None and y.name is not None:
        x = x.rename(y.name)
    elif y.name is None and x.name is not None:
        y = y.rename(x.name)
    else:
        assert x.name == y.name, ('if both x & y have .name not-None, they must be '
            'equal. you may also set the name you do not wish to keep to None'
        )

    outer = x * y
    # TODO remove this if i make this fn do more than just outer products
    concat_shape = x.shape + y.shape
    assert outer.shape == concat_shape, f'{outer.shape=} != {concat_shape=}'
    #
    return outer

