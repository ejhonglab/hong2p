
import os
from os.path import join, split, exists, getmtime
from pathlib import Path
import pickle
from pprint import pprint
import re
from typing import Optional, Union, List
# TODO replace w/ logging.warning
import warnings
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# This must be my fork at https://github.com/tom-f-oconnell/ijroi
import ijroi
from hong2p import thor, util, viz
from hong2p.types import Pathlike, NumpyOrXArray


# TODO TODO switch order of args, and allow passing just coords. if just coords are
# passed, shift all towards 0 (+ margin). use for e.g. xpix/ypix stats elements in
# suite2p stat output. corresponding pixel weights in lam output would not need to be
# modified.
def crop_to_coord_bbox(matrix, coords, margin=0):
    """Returns matrix cropped to bbox of coords and bounds.
    """
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    assert x_min >= 0 and y_min >= 0, \
        f'mins must be >= 0 (x_min={x_min}, y_min={y_min})'

    # NOTE: apparently i had to use <= or else some cases in old use of this function
    # (e.g. using certain CNMF outputs) would violate this. best to just fix that code
    # if it ever comes up again though.
    assert x_max < matrix.shape[0] and y_max < matrix.shape[1], (
        f'maxes must be < matrix shape = {matrix.shape} (x_max={x_max}' +
        f', y_max={y_max}'
    )

    # Keeping min at 0 to prevent slicing error in that case
    # (I think it will be empty, w/ -1:2, for instance)
    # Capping max not necessary to prevent err, but to make behavior of bounds
    # consistent on both edges.
    # TODO flag to err if margin would take it past edge? / warn?
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(x_max + margin, matrix.shape[0] - 1)
    y_max = min(y_max + margin, matrix.shape[1] - 1)

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


def crop_to_nonzero(matrix, margin=0):
    """
    Returns a matrix just large enough to contain the non-zero elements of the
    input, and the bounding box coordinates to embed this matrix in a matrix
    with indices from (0,0) to the max coordinates in the input matrix.
    """
    # nan_to_num will replace nan w/ 0 by default. infinities also converted but not
    # expected to be in input.
    coords = np.argwhere(np.nan_to_num(matrix) > 0)
    return crop_to_coord_bbox(matrix, coords, margin=margin)


# TODO delete (and other stuff no longer used)
# TODO if these 'db_row2*' fns are really just db related, move to db.py, but
# probably just rename...
# TODO better name?
def db_row2footprint(db_row, shape=None):
    """Returns dense array w/ footprint from row in cells table.
    """
    from scipy.sparse import coo_matrix
    weights, x_coords, y_coords = db_row[['weights','x_coords','y_coords']]
    # TODO maybe read shape from db / metadata on disk? / merging w/ other
    # tables (possible?)?
    footprint = np.array(coo_matrix((weights, (x_coords, y_coords)),
        shape=shape).todense()).T
    return footprint


# TODO delete (and other stuff no longer used)
def db_footprints2array(df, shape):
    """Returns footprints in an array of dims (shape + (n_footprints,)).
    """
    return np.stack([db_row2footprint(r, shape) for _, r in df.iterrows()],
        axis=-1)


# TODO maybe refactor so there is a function does this for single arrays, then concat
# using xarray functions in here? or if i still want both functions, how to dedupe code?
# allow this to accept single rois too (without that component of shape)?
# TODO type hint for roi_indices
def numpy2xarray_rois(rois: np.ndarray, roi_indices: Optional[dict] = None
    ) -> xr.DataArray:
    # TODO doc what keys of roi_indices should be (/ delete).
    # currently only used by hong2p.suite2p.suite2p_stat2rois
    """Takes numpy array of shape ([z,]y,x,roi) to labelled xarray.

    Args:
        roi_indices (None | dict): values must be iterables of length equal to number of
            ROIs. 'roi_num' will be included as an additional ROI index regardless.
    """
    shape = rois.shape
    # TODO check that the fact that i swapped y and x now didn't break how i was using
    # this w/ actual ijrois / anything else. wanted to be more consistent w/ how
    # suite2p, ImageJ, etc seemed to do things.
    if len(shape) == 3:
        dims = ['y', 'x', 'roi']
    elif len(shape) == 4:
        dims = ['z', 'y', 'x', 'roi']
    else:
        raise ValueError('shape must have length 3 or 4')

    # NOTE: 'roi_num' can't be replaced w/ 'roi' b/c conflict w/ name of 'roi' dim
    roi_num_name = 'roi_num'
    roi_index_names = [roi_num_name]
    roi_index_levels = [np.arange(rois.shape[-1])]

    if roi_indices is not None:

        # TODO delete. not useful as is (ijroi_masks subsets output of this fn, before
        # calling code uses it for integer based indexing)
        #
        # Want to be able to rely on roi_num always being [0, len - 1] (in order)
        # (so we can use numpy/iloc/isel indexing in some places)
        assert roi_num_name not in roi_indices

        n_rois = shape[-1]
        for ns, xs in roi_indices.items():
            assert len(xs) == n_rois
            roi_index_names.append(ns)
            roi_index_levels.append(xs)

    roi_index = pd.MultiIndex.from_arrays(roi_index_levels, names=roi_index_names)
    return xr.DataArray(rois, dims=dims, coords={'roi': roi_index})


# TODO doc + test
# TODO complete? check against imagej internal code?
def is_ijroi_name_default(roi: str) -> bool:

    parts = roi.split('-')
    if len(parts) not in (2, 3):
        return False

    for p in parts:
        try:
            int(p)
        except ValueError:
            return False

    return True


# TODO doc + test
def is_ijroi_named(roi: str) -> bool:
    try:
        int(roi)
        return False
    except ValueError:
        return not is_ijroi_name_default(roi)


def is_ijroi_plane_outline(roi: str) -> bool:
    """Returns ROI name indicates it's an outline of a whole structure (e.g. the AL)

    There will typically be exactly one of these per plane, when used. Sub-ROIs (e.g.
    glomeruli) should be contained within this larger region.
    """
    # TODO also support sam's 'plane<x>' syntax?
    return roi == 'AL'


# TODO drop support for '/' (and maybe also '|'?)
ijroi_uncertainty_chars = ('?', '|', '/', '+')
# TODO doc + test
def is_ijroi_certain(roi: str) -> bool:
    """Returns whether an ROI is named indicating it's ID is certain.

    False for names that are default, integers, or include one of
    `ijroi_uncertainty_chars` (e.g. '?'), and True otherwise.
    """
    # Won't contain any of the characters indicating uncertainty if it's just a number.
    if not is_ijroi_named(roi):
        return False

    if any(x in roi for x in ijroi_uncertainty_chars):
        return False

    return True


def ijroi_name_as_if_certain(roi_name: str) -> Optional[str]:
    # TODO or raise ValueError instead of returning None?
    """Removes characters indicating uncertainty, to group ROIs by best guess at ID

    'VM3' -> 'VM3'
    'VM3?' -> 'VM3'
    'VM3+?' -> 'VM3'
    'VM2|VM3?' -> None
    'VM2+VM3?' -> None

    Any time ijroi_uncertainty_chars split remainder of name into more than one part,
    None is returned (ambiguous which name it should refer to).
    """
    # the regex below would work without stripping these first (as long as we remove the
    # final '' string if there was a suffix), but by handling these first we shouldn't
    # need to handle empty strings in split output (don't want to support names that
    # would have them anywhere besides end either)
    allowed_suffix_chars = ('?', '+')
    without_uncertainty_suffix = roi_name.strip(''.join(allowed_suffix_chars))

    pattern = f"[{''.join([re.escape(c) for c in ijroi_uncertainty_chars])}]+"
    parts = re.split(pattern, without_uncertainty_suffix)

    assert len(parts) > 0, f'{roi_name=} had nothing besides {ijroi_uncertainty_chars=}'
    if len(parts) > 1:
        return None

    return parts[0]


def ijroi_comparable_via_name(n: str) -> bool:
    """Returns whether input refers to only a single ROI name
    ('VM3?' and 'VM3')
    """
    if is_ijroi_certain(n):
        return True

    if not is_ijroi_named(n):
        return False

    # TODO is this where i want this? just drop anything with '+' in it
    # (in other stuff in here that currently does '+' suffix dropping)?
    #
    # will currently deal with stuff like: 'VM2+', 'VM2+?', 'VM2+VM3'
    if '+' in n:
        return False
    #

    # 'VM2|VM3', 'VM2+VM3', etc will have None for this, as there are
    # multiple names it could be (not sure which to compare to, or whether
    # signal is just one)
    as_if_certain = ijroi_name_as_if_certain(n)
    if as_if_certain is None:
        return False

    return True


# TODO also test on w/ non-MultiIndex 'roi' columns?
# TODO adapt to also work w/ DataArray input (at least w/ 'roi_name' level, but maybe
# also 'roi' (or 'name', on 'roi' dimension?))
def certain_roi_indices(df: pd.DataFrame) -> np.ndarray:
    """Returns boolean mask of certain ROIs, for DataFrame with 'roi' column level
    """
    # TODO some reason i'm not using Index.map? do i specifically want the numpy array
    # type for some reason?
    return np.array([is_ijroi_certain(x) for x in df.columns.get_level_values('roi')])


# TODO TODO adapt to also work w/ DataArray input
def select_certain_rois(df: pd.DataFrame) -> pd.DataFrame:
    """Returns input subset with certain ROI labels, for input with 'roi' column level
    """
    mask = certain_roi_indices(df)
    # no need to copy, because indexing with a bool mask always does
    return df.loc[:, mask]


# TODO TODO rename / delete one-or-the-other of this and contour2mask etc
# (+ accept ijroi[set] filename or something if actually gonna call it this)
# (ALSO include ijrois2masks in consideration for refactoring. this fn might not be
# necessary)
def ijroi2mask(roi, shape, z: Optional[int] = None):
    """
        z: z-index ROI was drawn on
    """
    # This mask creation was taken from Yusuke N.'s answer here:
    # https://stackoverflow.com/questions/3654289
    from matplotlib.path import Path

    if z is None:
        if len(shape) != 2:
            raise ValueError(f'len(shape) == {len(shape)}. must be 2 if z keyword '
                'argument not passed'
            )

        # TODO check transpose isn't necessary...
        nx, ny = shape

    else:
        if len(shape) != 3:
            raise ValueError(f'shape must be (z, x, y) if z is passed. shape == {shape}'
                ', which has the wrong length'
            )

        # TODO check transpose (of x and y) isn't necessary...
        nz, nx, ny = shape
        if z >= nz:
            raise ValueError(f'z ({z}) out of bounds with z size ({nz}) from shape[0]')

    # TODO test + delete
    #assert nx == ny, 'need to check code shoulnt be tranposing these'
    #

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(roi)
    grid = path.contains_points(points)
    # Transpose makes this correct in testing on some of YZ's data
    mask = grid.reshape((ny, nx)).T

    if z is None:
        return mask
    else:
        vol_mask = np.zeros(shape, dtype='bool')
        vol_mask[z, :, :] = mask
        return vol_mask


# TODO TODO add option to translate ijroi labels to pandas index values?
# (and check trace extraction downstream preserves those!)
# TODO TODO TODO document type / structure expecations of inputs/outputs
# TODO TODO accept either the input or output of ijroi.read_roi[_zip] for ijrois?
# read_roi takes file object and read_roi_zip takes filename
# TODO can ijroi lib be modified to read path to tif things were drawn over (is that
# data there?), and can that be used to get shape? or can i also accept a path to tiff
# / thorimage dir / etc for that?
# TODO TODO option to use one of those scipy sparse arrays for the masks instead?
# TODO TODO maybe update all of my roi stuff that currently has the roi index as the
# last index so that it is the first index instead? feels more intuitive...
# TODO TODO make as_xarray default behavior and remove if other places that use this
# output don't break / change them
# TODO TODO TODO fn to convert suite2p representation of masks to the same [xarray]
# representation of masks this spits out
#
# TODO do try to make it faster. think it's probably a big part of why al_analysis.py is
# slow to recompute ijroi stuff. less of an issue from plot_roi.py now though, since
# only calculating for particular roi index now.
# ~25% of time in ijroi2mask calls, and ~75% of time in np.stack(masks, ...) call
# (would require `from line_profiler import profile` (pip install line-profiler)
#@profile
def ijrois2masks(ijrois, shape, as_xarray: bool = False
    # TODO type hint alias for ndarray|DataArray?
    ) -> Union[np.ndarray, xr.DataArray]:
    # TODO be clear on where shape is coming from (just shape of the data in the TIFF
    # the ROIs were draw in, right?)
    """
    Transforms ROIs loaded from my ijroi fork to an array full of boolean masks,
    of dimensions (shape + (n_rois,)).
    """
    import ijroi

    # TODO delete depending on how refactoring the below into xarray fn goes / whether
    # the non-xarray parts of this fn have the same requirements (which they probably
    # do...)
    if len(shape) not in (2, 3):
        raise ValueError('shape must have length 2 or 3')

    masks = []
    names = []
    if len(shape) == 3:
        roi_z_indices = []
        prefixes = []
        suffixes = []

    for name, roi in ijrois:
        z_index = None

        if len(shape) == 3:
            # Otherwise, it should simply be a numpy array with points.
            # `points_only=False` to either `read_roi_zip` or `read_roi` should produce
            # input suitable for this branch.
            if getattr(roi, 'z', None) is not None:
                z_index = roi.z

            # ROI was drawn on an image w/ only 3 dimensions (assuming the non-XY one
            # was Z)
            elif getattr(roi, 'position', None) is not None:
                z_index = roi.position

            if z_index is not None:
                # This should be the same as the `name` this shadows, just without the
                # '.roi' suffix.
                name = roi.name

                # TODO eventually delete everything that uses parse_z_from_name.
                # is there any old code that depends on it?
                try:
                    _, (prefix, suffix) = ijroi.parse_z_from_name(name,
                        return_unmatched=True
                    )
                except ValueError:
                    prefix = name
                    suffix = ''
            else:
                warnings.warn('trying to parse Z from ROI name. pass points_only=False'
                    ' to ijroi loading function to read Z directly.'
                )
                z_index, (prefix, suffix) = ijroi.parse_z_from_name(name,
                    return_unmatched=True
                )

            # TODO assert z_index in range of corresponding element of shape?

        points = getattr(roi, 'points', roi)

        # TODO may also need to reverse part of shape here, if really was
        # necessary above (test would probably need to be in asymmetric
        # case...)
        masks.append(ijroi2mask(points, shape, z=z_index))
        names.append(name)

        if len(shape) == 3:
            roi_z_indices.append(z_index)

            if len(prefix) == 0:
                prefix = np.nan

            if len(suffix) == 0:
                suffix = np.nan

            prefixes.append(prefix)
            suffixes.append(suffix)

    # This concatenates along the last element of the new shape
    masks = np.stack(masks, axis=-1)
    if not as_xarray:
        return masks

    roi_index_names = ['roi_name']
    roi_index_levels = [names]
    if len(shape) == 3:
        roi_index_names += ['roi_z', 'ijroi_prefix', 'ijroi_suffix']
        roi_index_levels += [roi_z_indices, prefixes, suffixes]

    return numpy2xarray_rois(masks,
        roi_indices=dict(zip(roi_index_names, roi_index_levels))
    )


# TODO maybe add a fn to plot single xarray masks for debugging?
# TODO TODO change `on` default to something like `roi`
def merge_rois(rois, on='ijroi_prefix', merge_fn=None, label_fn=None,
    check_no_overlap=False):
    """
    Args:
        rois (xarray.DataArray): must have at least dims 'x', 'y', and 'roi'.
            'roi' should be indexed by a MultiIndex and one of the levels should have
            the name of the `on` argument. Currently expect the dtype to be 'bool'.

        on (str): name of other ROI metadata dimension to merge on

        label_fn (callable): function mapping the values of the `on` column to labels
            for the ROIs. Only ROIs created via merging will be given these labels,
            while unmerged ROIs will recieve unique number labels. Defaults to the
            identity function.

        check_no_overlap (bool): (optional, default=False) If True, checks that no
            merged rois shared any pixels before being merged. If merged ROIs are all
            on different planes, this should be True because ImageJ ROIs are defined on
            single planes.
    """
    # TODO assert bool before this / rename to something like 'total_weight' that would
    # apply in non-boolean-mask case too
    total_weight_before = rois.sum().item()
    n_rois_before = len(rois.roi)

    def get_nonroi_shape(arr):
        return {k: n for k, n in zip(arr.dims, arr.shape) if k != 'roi'}

    nonroi_shape_before = get_nonroi_shape(rois)

    # TODO maybe i should use sum instead, if i'm not going to make the aggregation
    # function configurable?
    # If `on` contains NaN values, the groupby will not include groups for the NaN
    # values, and there is no argument to configure this (as in pandas). Therefore,
    # we need to drop things that were merged and then add these to what remains.
    # TODO do i need to check that nothing else conflicts w/ what i plan on renaming
    # `on` to ('roi')?
    if merge_fn is None:
        merged = rois.groupby(on).max()
    else:
        #raise NotImplementedError
        # TODO maybe try reduce? might need other inputs tho...
        merged = rois.groupby(on).map(merge_fn)

    # TODO some kind of inplace version of this? or does xarray not really do that?
    merged = merged.rename({on: 'roi'})

    if label_fn is None:
        label_fn = lambda x: x

    merged_roi_labels = [label_fn(x) for x in merged.roi.values]
    # Trying to pass label_fn here instead of calling before didn't work b/c it seems to
    # expect a fn that takes a DataArray (and it's not passing scalar DataArrays either)
    merged = merged.assign_coords(roi=merged_roi_labels)

    not_merged = rois[on].isnull()
    unmerged = rois.where(not_merged, drop=True).reset_index('roi', drop=True)

    n_orig_rois_merged = (~ not_merged).values.sum()
    n_rois_after = n_rois_before - n_orig_rois_merged + len(merged.roi)

    available_labels = [x for x in range(n_rois_after) if x not in merged_roi_labels]
    unmerged_roi_labels = available_labels[:len(unmerged.roi)]
    assert len(unmerged_roi_labels) == len(unmerged.roi)
    unmerged = unmerged.assign_coords(roi=unmerged_roi_labels)

    # The .groupby (seemingly with any function application, as .first() also does it)
    # and .where both change the dtype to float64 from bool
    was_bool = rois.dtype == 'bool'

    rois = xr.concat([merged, unmerged], 'roi')

    if was_bool:
        rois = rois.astype('bool')

    assert n_rois_after == len(rois.roi)
    assert len(set(rois.roi.values)) == len(rois.roi)

    total_weight_after = rois.sum().item()
    if check_no_overlap:
        assert total_weight_before == total_weight_after
    else:
        assert total_weight_before >= total_weight_after

    nonroi_shape_after = get_nonroi_shape(rois)
    assert nonroi_shape_before == nonroi_shape_after

    return rois


def merge_ijroi_masks(masks, **kwargs):
    """
    Args:
        masks (xarray.DataArray): must have at least dims 'x', 'y', and 'roi'.
            'roi' should be indexed by a MultiIndex and one of the levels should have
            the name of the `on` argument. Currently expect the dtype to be 'bool'.

        label_fn (callable): function mapping the values of the `on` column to labels
            for the ROIs. Only ROIs created via merging will be given these labels,
            while unmerged ROIs will recieve unique number labels. Defaults to a
            function that takes strings, removes trailing/leading underscores, and
            parses an int from what remains.
    """
    # TODO probably assert bool
    # TODO assert ijroi_prefix in here / accept kwarg on (defaulting to same), and
    # assert that's here

    return merge_rois(masks, on='ijroi_prefix', **kwargs)


# TODO TODO TODO make another function that groups rois based on spatial overlap
# (params to include [variable-number-of?] dilation steps, fraction of pixels[/weight?]
# that need to overlap, and correlation of responses required) -> generate appropriate
# input to [refactored + renamed] merge_ijroi_masks fn below, particularly the `masks`
# and `on` arguments, and have label_fn be identity

# TODO TODO how to handle a correlation threshold? pass correlations of some kind in
# or something to compute them from (probably the former, or make the correlation
# thresholding a separate step)?
def merge_single_plane_rois(rois, min_overlap_frac=0.3, n_dilations=1):
    """
    For handling single plane ROIs that are on adjacent Z planes, and correspond to the
    same biological feature. This is to merge the single plane ROIs that suite2p
    outputs.
    """
    raise NotImplementedError
    import ipdb; ipdb.set_trace()


# TODO TODO refactor this + hong2p.suite2p.remerge_suite2p_merged to share core code
# here! (this initially copied from other fn and then adapted)
# TODO unit test! (include 2023-04-26/3 ROIs, where some also have '+' in name, that
# ijroi_masks is currently dropping)
# TODO change verbose default to False
def rois2best_planes_only(rois: xr.DataArray, roi_quality: pd.Series,
    verbose: bool = True) -> xr.DataArray:
    """
    Currently assumes input only has non-zero values in a single plane for a given
    unique combination of ROI identifier variables.
    """
    # TODO modify calling code to pass roi_quality in w/ compatible metadata to rois,
    # and try to replace these w/ assertions that metadata is compatible
    assert rois.sizes['roi'] == len(roi_quality)
    # (as calling code is currently only gauranteeing that shapes are the same, so
    # diff index might cause problems...)
    assert list(roi_quality.index) == list(range(len(roi_quality)))

    # TODO delete all this if i can replace roi_index w/ isel / similar
    roi_quality = roi_quality.copy()
    roi_quality.index.name = 'roi_index'
    # TODO remove this from index if i'm going to return other metadata?
    # (and if i don't manage to replace this code w/ isel/similar)
    rois = rois.assign_coords({'roi_index': ('roi', range(rois.sizes['roi']))})

    input_roi_names = set(rois.roi_name.values)

    roi_quality = roi_quality.to_frame(name='roi_quality')

    # TODO better name
    mo_key = 'name'
    roi_quality[mo_key] = rois.roi_name.to_numpy()

    # TODO dropna=False? or at least assert no null in it then...
    gb = roi_quality.groupby(mo_key, sort=False)

    # TODO implement another strategy where as long as the roi_quality are within some
    # tolerance of the best, they are averaged? or weighted according to response stat?
    # weight according to variance of response stat (see examples of weighted averages
    # using something derived from variance for weights online)
    # TODO maybe also use 2/3rd highest lowest frame / percentile rather than actual min
    # / max (for picking 'best' plane), to gaurd against spiking noise

    # TODO rename?
    best_inputs = gb.idxmax()

    # Selecting the only column this DataFrame has (i.e. shape (n, 1))
    # TODO assertion on shape before this (and maybe values / dtype?)
    # TODO rename this and/or `best` below
    best_inputs = best_inputs.iloc[:, 0]

    best = roi_quality.loc[best_inputs]

    assert np.array_equal(best.roi_quality.values, gb.max().roi_quality.values)

    if verbose:
        by_response = roi_quality.dropna().set_index('name', append=True).swaplevel()

    notbest_to_drop = []
    for row in best.itertuples():
        name = row.name
        curr_best = row.Index

        curr_notbest = list(rois.roi[
            (rois.roi_name == name) & (rois.roi_index != curr_best)
        ].roi_index.values)

        notbest_to_drop.extend(curr_notbest)

        # only print if multiple inputs (i.e. actually 'merging' in some sense)
        if verbose and len(curr_notbest) > 0:
            print(f'merging ROI {name}')
            print(f'selecting input ROI {curr_best} as best plane')
            print(f'dropping other input ROIs {curr_notbest}')
            print(by_response.loc[name])
            print()

    rois = rois.sel(roi= ~ rois.roi_index.isin(notbest_to_drop))

    # TODO delete eventually (just getting metadata of subset of `rois` returned, and
    # indexing using that in the future)
    roi_indices = rois.roi_index.values

    output_roi_names = set(rois.roi_name.values)
    assert input_roi_names == output_roi_names
    # If input and ouput name sets are equal, this should imply no duplicated ROI names
    # in output.
    assert rois.sizes['roi'] == len(input_roi_names)

    # TODO TODO TODO remove need for this (so we can keep metadata in output, at least
    # by default)
    # TODO should i also include roi_z? would want to also do / use in s2p case for
    # consistency... also, how to modify this call to accomplish that?
    rois = rois.assign_coords(roi=rois.roi_name)

    # TODO TODO if i can figure out how to keep multiple levels for the roi dimension,
    # do that rather than return multiple things
    return roi_indices, rois


ijroiset_default_basename = 'RoiSet.zip'

def ijroi_filename(ijroiset_dir_or_fname: Pathlike, must_exist: bool = True) -> Path:

    ijroiset_dir_or_fname = Path(ijroiset_dir_or_fname)
    if ijroiset_dir_or_fname.is_dir():
        # TODO if i standardize path to analysis intermediates, update this to look for
        # RoiSet.zip there?
        ijroiset_fname = ijroiset_dir_or_fname / ijroiset_default_basename

        if must_exist and not ijroiset_fname.is_file():
            raise IOError('directory passed for ijroiset_dir_or_fname, but '
                f'{ijroiset_fname} did not exist'
            )

    else:
        # TODO check it's actually loadable?
        ijroiset_fname = ijroiset_dir_or_fname

    return ijroiset_fname


def has_ijrois(ijroiset_dir_or_fname: Pathlike) -> bool:
    # NOTE: not actually checking it contains any
    ijroiset_fname = ijroi_filename(ijroiset_dir_or_fname, must_exist=False)
    return ijroiset_fname.is_file()


def ijroi_mtime(ijroiset_dir_or_fname: Pathlike) -> float:
    """RoiSet.zip (/directory with one) path to Unix timestamp modification time
    """
    ijroiset_fname = ijroi_filename(ijroiset_dir_or_fname)
    return getmtime(ijroiset_fname)


# TODO maybe delete drop_maximal_extent_rois and move code that drops them to a separate
# call (or set default to False?)? initially broke some stuff in al_analysis.py and
# later seemingly plot_roi.py, which expected to be able to index ROI list consistently
# TODO rename *_maximal_extent_rois -> *_nonseparated_rois? something else?
# want to indicate the signal may be a mixture from multiple glomeruli (/ biological
# regions)
#
# 95% of the time this fn took was from ijrois2masks
def ijroi_masks(ijroiset_dir_or_fname: Pathlike, thorimage_dir: Pathlike,
    as_xarray: bool = True, drop_maximal_extent_rois: bool = True,
    indices: Optional[List[int]] = None, **kwargs) -> NumpyOrXArray:

    ijroiset_fname = ijroi_filename(ijroiset_dir_or_fname)

    # TODO format better + put behind verbose flag (/delete)
    #print(f'{ijroiset_fname=}')
    #print(f'{ijroiset_fname.resolve()=}')

    # TODO maybe just fix ijroi.read_roi_zip to work in case input is a single lone .roi
    # file?
    try:
        name_and_roi_list = ijroi.read_roi_zip(ijroiset_fname, points_only=False)

    except BadZipFile as e:
        raise IOError(f'does {ijroiset_fname} only have a single ROI? try adding >1, '
            'so that ImageJ saves ROIs as a zipfile (or deselect ROIs before any manual'
            ' saving)'
        ) from e

    _, (x, y), z, c, _, _ =  thor.load_thorimage_metadata(thorimage_dir)

    assert x == y, 'not tested in case x != y'

    # From what `thor.read_movie` says the output dimensions are (except the first
    # dimension, which is time).
    if z == 1:
        movie_shape_without_time = (y, x)
    else:
        movie_shape_without_time = (z, y, x)

    # since current mask calculation is seems to have some slow steps, and we don't need
    # to compute all of them when calling from plot_roi[_util].py
    if indices is not None:
        name_and_roi_list = [name_and_roi_list[i] for i in indices]

    masks = ijrois2masks(name_and_roi_list, movie_shape_without_time,
        as_xarray=as_xarray
    )

    # 2023-04-26/3 currently is one of just a few flies w/ ROIs using the optional '+'
    # suffix syntax (to indicate the same ROI name w/o the suffix is contained by the
    # suffixed ROI, which might also have more contamination from external signals)
    #
    # TODO TODO assert that for each of these, we have ROIs w/ name matching the
    # non-'+'-suffixed name (we might not tho... could be impossible to get a clean ROI
    # for some glomeruli)?
    # TODO put dropping of these behind flag?
    if drop_maximal_extent_rois:
        # TODO TODO just check if '+' is in roi name? or want to still handle diff if
        # there is a '?' suffix? kinda want to support e.g. 'DM2+DM5' tho
        # TODO TODO TODO also handle '+?' suffix? warn / assert it isn't there?
        maximal_extent_rois = masks.roi_name.str.endswith('+')
        n_maximal_to_drop = maximal_extent_rois.sum().item(0)
        if n_maximal_to_drop > 0:
            # TODO convert to logging.warnings?
            # TODO better message (indicating what they are)?
            warnings.warn(f"dropping {n_maximal_to_drop} ROIs with '+' suffix")

            n_rois_before = masks.sizes['roi']

            masks = masks[dict(roi=~maximal_extent_rois)].copy()
            assert masks.sizes['roi'] == (n_rois_before - n_maximal_to_drop)

    return masks

    ## TODO modify check_no_overlap to make sure it's also erring if two things that
    ## would be merged (by having same name / whatever) are not in the same z-plane
    ## (assuming the intention was to have one per plane, to make a single volumetric
    ## ROI)
    #merged = merge_ijroi_masks(masks, check_no_overlap=True)
    #
    #import ipdb; ipdb.set_trace()
    #
    #return merged


# TODO test / document requirements for type / properties of contour. it's just a
# numpy array of points, right? doesn't need start = end or anything, does it?
def contour2mask(contour, shape):
    """Returns a boolean mask True inside contour and False outside.
    """
    # TODO TODO TODO appropriate checking of contour input. i.e. any requirements on
    # first/last point / order (do some orders imply overlapping edge segments, and if
    # so, check there are none of those)
    import cv2
    # TODO any checking of contour necessary for it to be well behaved in
    # opencv?
    mask = np.zeros(shape, np.uint8)

    # NOTE: at some point i think i needed convexHull in ijroi2mask to get that + this
    # to work as I expected. AS I THINK CONVEXHULL MIGHT RESULT IN SOME UNEXPECTED
    # MODIFICATIONS TO CONTOURS, i need to change that code, and that might break some
    # of this code too
    # TODO TODO TODO if drawContours truly does need convex hull inputs, need to change
    # this function to no longer use drawContours
    # TODO TODO TODO see strategy i recommended to yang recently and consider using it
    # here instead
    # TODO draw into a sparse array maybe? or convert after?
    cv2.drawContours(mask, [contour.astype(np.int32)], 0, 1, -1)

    # TODO TODO TODO investigate need for this transpose
    # (imagej contour repr specific? maybe load to contours w/ dims swapped them
    # call this fn w/o transpose?)
    # (was it somehow still a product of x_coords / y_coords being swapped in
    # db?)
    # not just b/c reshaping to something expecting F order CNMF stuff?
    # didn't correct time averaging w/in roi also require this?
    return mask.astype('bool')


# TODO delete (/ rename "py" to "cnmf" or something)
def py2imagej_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
    # TODO TODO TODO probably just delete any code that actually relied on this?
    # assuming it doesn't still make sense...
    #return array.T
    return array


# TODO maybe move to a submodule for interfacing w/ cnmf?
# TODO TODO probably make a corresponding fn to do the inverse
# (or is one of these not necessary? in one dir, is order='C' and order
def footprints_to_flat_cnmf_dims(footprints):
    """Takes array of (x, y[, z], n_footprints) to (n_pixels, n_footprints).

    There is more than one way this reshaping can be done, and this produces
    output as CNMF expects it.
    """
    frame_pixels = np.prod(footprints.shape[:-1])
    n_footprints = footprints.shape[-1]
    # TODO TODO is this supposed to be order='F' or order='C' matter?
    # wrong setting equivalent to transpose?
    # what's the appropriate test (make unit?)?
    return np.reshape(footprints, (frame_pixels, n_footprints), order='F')


# TODO change this + how plot_roi_util.py calls it, so that we can only calculate mean
# for the relevant subset of frames (e.g. only for baseline frames + within odor
# response window)? would that even save much time? (could make mean same length, but
# NaN, and fill in calculated timepoints, so indexing output would be the same)
#
# 90% of time was from the np.mean line
def extract_traces_bool_masks(movie, footprints, verbose=False):
    # TODO doc shape/type expectations on movie / footprints
    """
    Averages the movie within each boolean mask in footprints
    to make a matrix of traces (n_frames x n_footprints).
    """
    assert footprints.dtype.kind != 'f', 'float footprints are not boolean'
    assert footprints.max() == 1, 'footprints not boolean'
    assert footprints.min() == 0, 'footprints not boolean'
    n_spatial_dims = len(footprints.shape) - 1
    spatial_dims = tuple(range(n_spatial_dims))
    assert np.any(footprints, axis=spatial_dims).all(), 'some zero footprints'
    slices = (slice(None),) * n_spatial_dims
    n_frames = movie.shape[0]
    n_footprints = footprints.shape[-1]
    traces = np.empty((n_frames, n_footprints)) * np.nan

    if verbose:
        print('extracting traces from boolean masks...', end='', flush=True)

    # TODO vectorized way to do this?
    for i in range(n_footprints):
        mask = footprints[slices + (i,)]
        # TODO compare time of this to sparse matrix dot product?
        # + time of MaskedArray->mean w/ mask expanded by n_frames?

        # TODO TODO is this correct? check
        # axis=1 because movie[:, mask] only has two dims (frames x pixels)
        trace = np.mean(movie[:, mask], axis=1)
        assert len(trace.shape) == 1 and len(trace) == n_frames
        traces[:, i] = trace

    if verbose:
        print(' done')

    return traces


# TODO what were these files for again? still want to use? delete?
def autoroi_metadata_filename(ijroi_file):
    path, fname = split(ijroi_file)
    return join(path, '.{}.meta.p'.format(fname))


def template_match(scene, template, method_str='cv2.TM_CCOEFF', hist=False,
    debug=False):

    import cv2

    vscaled_scene = util.baselined_normed_u8(scene)
    # TODO TODO maybe template should only be scaled to it's usual fraction of
    # max of the scene? like scaled both wrt orig_scene.max() / max across all
    # images?
    vscaled_template = util.baselined_normed_u8(template)

    if debug:
        # To check how much conversion to u8 (necessary for cv2 template
        # matching) has reduced the number of pixel levels.
        scene_levels = len(set(scene.flat))
        vs_scene_levels = len(set(vscaled_scene.flat))
        template_levels = len(set(template.flat))
        vs_template_levels = len(set(vscaled_template.flat))
        print(f'Number of scene levels BEFORE scaling: {scene_levels}')
        print(f'Number of scene levels AFTER scaling: {vs_scene_levels}')
        print(f'Number of template levels BEFORE scaling: {template_levels}')
        print(f'Number of template levels AFTER scaling: {vs_template_levels}')

        # So you can see that the relative dimensions and scales of each of
        # these seems reasonable.
        def compare_template_and_scene(template, scene, suptitle,
            same_scale=True):

            smin = scene.min()
            smax = scene.max()
            tmin = template.min()
            tmax = template.max()

            print(f'\n{suptitle}')
            print('scene shape:', scene.shape)
            print('template shape:', template.shape)
            print('scene min:', smin)
            print('scene max:', smax)
            print('template min:', tmin)
            print('template max:', tmax)

            # Default, for this fig at least seemed to be (6.4, 4.8)
            # This has the same aspect ratio.
            fh = 10
            fw = (1 + 1/3) * fh
            fig, axs = plt.subplots(ncols=3, figsize=(fw, fh))

            xlim = (0, max(scene.shape[0], template.shape[0]) - 1)
            ylim = (0, max(scene.shape[1], template.shape[1]) - 1)

            if same_scale:
                vmin = min(smin, tmin)
                vmax = max(smax, tmax)
            else:
                vmin = None
                vmax = None

            ax = axs[0]
            sim = ax.imshow(scene, vmin=vmin, vmax=vmax)
            ax.set_title('scene')

            ax = axs[1]
            tim = ax.imshow(template, vmin=vmin, vmax=vmax)
            ax.set_title('template (real scale)')

            ax = axs[2]
            btim = ax.imshow(template, vmin=vmin, vmax=vmax)
            ax.set_title('template (blown up)')

            # https://stackoverflow.com/questions/31006971
            plt.setp(axs, xlim=xlim, ylim=ylim)

            ax.set_xlim((0, template.shape[0] - 1))
            ax.set_ylim((0, template.shape[0] - 1))

            fig.suptitle(suptitle)

            if same_scale:
                # l, b, w, h
                cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cb = fig.colorbar(sim, cax=cax)
                cb.set_label('shared')
                fig.subplots_adjust(right=0.8)
            else:
                # l, b, w, h
                cax1 = fig.add_axes([0.75, 0.15, 0.025, 0.7])
                cb1 = fig.colorbar(sim, cax=cax1)
                cb1.set_label('scene')

                cax2 = fig.add_axes([0.85, 0.15, 0.025, 0.7])
                cb2 = fig.colorbar(tim, cax=cax2)
                cb2.set_label('template')

                fig.subplots_adjust(right=0.7)

            bins = 50
            fig, axs = plt.subplots(ncols=2, sharex=same_scale)
            ax = axs[0]
            shistvs, sbins, _ = ax.hist(scene.flat, bins=bins, log=True)
            ax.set_title('scene')
            ax.set_ylabel('Frequency (a.u.)')
            ax = axs[1]
            thitvs, tbins, _ = ax.hist(template.flat, bins=bins, log=True)
            ax.set_title('template')
            fig.suptitle(f'{suptitle}\npixel value distributions ({bins} bins)')
            fig.subplots_adjust(top=0.85)

        compare_template_and_scene(template, scene, 'original',
            same_scale=False
        )
        compare_template_and_scene(vscaled_template, vscaled_scene,
            'baselined + scaled'
        )
        print('')
        hist = True

    method = eval(method_str)
    res = cv2.matchTemplate(vscaled_scene, vscaled_template, method)

    # b/c for sqdiff[_normed], find minima. for others, maxima.
    if 'SQDIFF' in method_str:
        res = res * -1

    if hist:
        fh = plt.figure()
        plt.hist(res.flatten())
        plt.title('Matching output values ({})'.format(method_str))

    return res


# TODO TODO TODO try updating to take max of two diff match images,
# created w/ different template scales (try a smaller one + existing),
# and pack appropriate size at each maxima.
# TODO make sure match criteria is comparable across scales (one threshold
# ideally) (possible? using one of normalized metrics sufficient? test this
# on fake test data?)
def greedy_roi_packing(match_images, ds, radii_px, thresholds=None, ns=None,
    exclusion_radius_frac=0.7, min_dist2neighbor_px=15, min_neighbors=3,
    exclusion_mask=None,
    draw_on=None, debug=False, draw_bboxes=True, draw_circles=True,
    draw_nums=True, multiscale_strategy='one_order', match_value_weights=None,
    radii_px_ps=None, scale_order=None, subpixel=False, _src_img_shape=None,
    _show_match_images=False, _show_packing_constraints=False, _show_fit=True,
    _initial_single_threshold=None):
    """
    Args:
    match_images (np.ndarray / iterable of same): 2-dimensional array of match
    value higher means better match of that point to template.

        Shape is determined by the number of possible offsets of the template in
        the original image, so it is smaller than the original image in each
        dimension. As an example, for a 3x3 image and a 2x2 template, the
        template can occupy 4 possible positions in the 3x3 image, and the match
        image will be 2x2.

    ds (int / iterable of same): integer width (and height) of square template.
        related to radius, but differ by margin set outside.

    radii_px (int / iterable of same): radius of cell in pixels.

    exclusion_radius_frac (float): approximately 1 - the fraction of two ROI
        radii that are allowed to overlap.
    """
    # TODO move drawing fns for debug to mpl and remove this if not gonna
    # use for constraints here
    import cv2
    #
    # Use of either this or KDTree seem to cause pytest ufunc size changed
    # warning (w/ pytest at least), though it should be harmless.
    from scipy.spatial import cKDTree

    if subpixel is True:
        raise NotImplementedError

    if thresholds is None and ns is None:
        raise ValueError('specify either thresholds or ns')

    if not ((ns is None and thresholds is not None) or
            (ns is not None and thresholds is None)):
        raise ValueError('only specify either thresholds or ns')

    # For multiscale matching, we require (at lesat) multiple radii, so we test
    # whether it is iterable to determine if we should be using multiscale
    # matching.
    try:
        iter(radii_px)

        if len(radii_px) == 1:
            multiscale = False
        else:
            assert len(set(ds)) == len(ds)
            assert len(set(radii_px)) == len(radii_px)
            multiscale = True

    except TypeError:
        multiscale = False
        # also check most other things are NOT iterable in this case?

        match_images = [match_images]
        ds = [ds]
        radii_px = [radii_px]

    if ns is None:
        total_n = None
        # TODO maybe delete this test and force thresholds (if-specified)
        # to have same length. useless if one threshold is never gonna work.
        try:
            iter(thresholds)
            # If we have multiple thresholds, we must have as many
            # as the things above.
            assert len(thresholds) == len(radii_px)
        except TypeError:
            thresholds = [thresholds] * len(radii_px)

    elif thresholds is None:
        try:
            iter(ns)
            # TODO want this behavior ever? maybe delete try/except...
            # Here, we are specify how many of each size we are looking for.
            assert len(ns) == len(radii_px)
            if len(ns) == 1:
                total_n = ns[0]
                ns = None
            else:
                total_n = None
        except TypeError:
            # Here, we specify a target number of cells of any size to find.
            total_n = ns
            ns = None

    if multiscale:
        n_scales = len(radii_px)
        assert len(match_images) == n_scales
        assert len(ds) == n_scales

        if multiscale_strategy != 'one_order':
            assert match_value_weights is None, ('match_value_weights are only '
                "meaningful in multiscale_strategy='one_order' case, because "
                'they do not change match ordering within a single match scale.'
                ' They only help make one ordering across matching scales.'
            )

        if multiscale_strategy != 'random':
            assert radii_px_ps is None, ('radii_px_ps is only meaningful in '
                "multiscale_strategy='random' case"
            )

        if multiscale_strategy != 'fixed_scale_order':
            assert scale_order is None, ('scale_order is only meaningful in '
                "multiscale_strategy='fixed_scale_order' case"
            )

        if multiscale_strategy == 'one_order':
            # Can still be None here, that just implies that match values
            # at different scales will be sorted into one order with no
            # weighting.
            if match_value_weights is not None:
                assert len(match_value_weights) == n_scales

            # could also accept callable for each element, if a fn (rather than
            # linear scalar) would be more useful to make match values
            # comparable across scales (test for it later, at time-to-use)

        elif multiscale_strategy == 'random':
            assert radii_px_ps is not None
            assert np.isclose(np.sum(radii_px_ps), 1)
            assert all([r >= 0 and r <= 1 for r in radii_px_ps])
            if any([r == 0 or r == 1 for r in radii_px_ps]):
                warnings.warn('Some elements of radii_px_ps were 0 or 1. '
                    "This means using multiscale_strategy='random' may not make"
                    ' sense.'
                )

        elif multiscale_strategy == 'fixed_scale_order':
            # could just take elements from other iterables in order passed
            # in... just erring on side of being explicit
            assert scale_order is not None
            assert len(set(scale_order)) == len(scale_order)
            for i in scale_order:
                try:
                    radii_px[i]
                except IndexError:
                    raise ValueError('scale_order had elements not usable to '
                        'index scales'
                    )

        else:
            raise ValueError(f'multiscale_strategy {multiscale_strategy} not '
                'recognized'
            )

        # Can not assert all match_images have the same shape, because d
        # affects shape of match image (as you can see from line inverting
        # this dependence to calculate orig_shape, below)

    else:
        n_scales = 1
        assert match_value_weights is None
        assert radii_px_ps is None
        assert scale_order is None
        # somewhat tautological. could delete.
        if thresholds is None:
            assert total_n is not None

    # TODO optimal non-greedy alg for this problem? (maximize weight of
    # match_image summed across all assigned ROIs)

    # TODO do away with this copying if not necessary
    # (just don't want to mutate inputs without being clear about it)
    # (multiplication by match_value_weights below)
    match_images = [mi.copy() for mi in match_images]
    orig_shapes = set()
    for match_image, d in zip(match_images, ds):
        # Working through example w/ 3x3 src img and 2x2 template -> 2x2 match
        # image in docstring indicates necessity for - 1 here.
        orig_shape = tuple(x + d - 1 for x in match_image.shape)
        orig_shapes.add(orig_shape)

    assert len(orig_shapes) == 1
    orig_shape = orig_shapes.pop()
    if _src_img_shape is not None:
        assert orig_shape == _src_img_shape
        del _src_img_shape

    if draw_on is not None:
        # if this fails, just fix shape comparison in next assertion and
        # then delete this assert
        assert len(draw_on.shape) == 2
        assert draw_on.shape == orig_shape

        draw_on = util.u8_color(draw_on)
        # upsampling just so cv2 drawing functions look better
        ups = 4
        draw_on = cv2.resize(draw_on,
            tuple([ups * x for x in draw_on.shape[:2]])
        )

    if match_value_weights is not None:
        for i, w in enumerate(match_value_weights):
            match_images[i] = match_images[i] * w

    if debug and _show_match_images:
        # wanted these as subplots w/ colorbar besides each, but colorbars
        # seemed to want to go to the side w/ the simplest attempt
        ncols = 3
        nrows = n_scales % ncols + 1
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

        if not multiscale or multiscale_strategy == 'one_order':
            vmin = min([mi.min() for mi in match_images])
            vmax = max([mi.max() for mi in match_images])
            same_scale = True
        else:
            vmin = None
            vmax = None
            same_scale = False

        for i, (ax, td, match_image) in enumerate(zip(
            axs.flat, ds, match_images)):

            to_show = match_image.copy()
            if thresholds is not None:
                to_show[to_show < thresholds[i]] = np.nan

            im = ax.imshow(to_show)
            if not same_scale:
                # https://stackoverflow.com/questions/23876588
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax)

            title = f'({td}x{td} template)'
            if match_value_weights is not None:
                w = match_value_weights[i]
                title += f' (weight={w:.2f})'
            ax.set_title(title)

        if same_scale:
            # l, b, w, h
            cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('match value')
            fig.subplots_adjust(right=0.8)

        title = 'template matching metric at each template offset'
        if thresholds is not None:
            title += '\n(white space is pixels below corresponding threshold)'
        fig.suptitle(title)
        # TODO may want to decrease wspace if same_scale
        fig.subplots_adjust(wspace=0.7)

    all_flat_vals = [mi.flatten() for mi in match_images]

    if debug:
        print('thresholds for each scale:', thresholds)
        print('min (possibly scaled) match val at each scale:',
            np.array([vs.min() for vs in all_flat_vals])
        )
        print('max (possibly scaled) match val at each scale:',
            np.array([vs.max() for vs in all_flat_vals])
        )
        print('match_value_weights:', match_value_weights)

    if not multiscale or multiscale_strategy == 'one_order':
        # TODO TODO TODO need to sort flat_vals into one order, while
        # maintaining info about which match_image (index) a particular
        # value came from
        # how to do this while also thresholding each one?

        all_vals = []
        all_scale_and_flat_indices = []
        for i, (fv, thresh) in enumerate(zip(all_flat_vals, thresholds)):
            if thresholds is not None:
                flat_idx_at_least_thresh = np.argwhere(fv > thresh)[:,0]
                vals_at_least_thresh = fv[flat_idx_at_least_thresh]
            else:
                flat_idx_at_least_thresh = np.arange(len(fv))
                vals_at_least_thresh = fv

            if debug:
                thr_frac = len(flat_idx_at_least_thresh) / len(fv)
                print(f'scale {i} fraction of (scaled) match values above'
                    ' threshold:', thr_frac
                )

                '''
                # TODO delete after figuring out discrepancy
                thr_fracs = [0.001252, 0.0008177839, 0.00087937249]
                assert np.isclose(thr_frac, thr_fracs[i])

                #print(len(flat_idx_at_least_thresh))
                #import ipdb; ipdb.set_trace()
                #
                '''
                # TODO TODO maybe find range of weights that produce same
                # fraction above thresholds, and see if somewhere in that range
                # is a set of weights that also leads to a global ordering that
                # behaves as I want?

                # TODO delete if not gonna finish
                if _initial_single_threshold is not None:
                    t0 = _initial_single_threshold

                    '''
                    if match_value_weights is not None:
                        # Undoing previous multiplication by weight.
                        w = match_value_weights[i]
                        orig_match_image = match_images[i] / w
                    orig 

                    # TODO TODO TODO fit(?) to find match value weight that
                    # produces same fraction of pixels above threshold
                    import ipdb; ipdb.set_trace()
                    '''
                #

            # TODO maybe just store ranges of indices in concatenated
            # flat_idx... that correspond to each source img?
            src_img_idx = np.repeat(i, len(flat_idx_at_least_thresh))

            scale_and_flat_indices = np.stack(
                [src_img_idx, flat_idx_at_least_thresh]
            )
            all_scale_and_flat_indices.append(scale_and_flat_indices)
            all_vals.append(vals_at_least_thresh)

        all_scale_and_flat_indices = np.concatenate(all_scale_and_flat_indices,
            axis=1
        )

        all_vals = np.concatenate(all_vals)
        # Reversing order so indices corresponding to biggest element is first,
        # and so on, decreasing.
        one_order_indices = np.argsort(all_vals)[::-1]

        '''
        # TODO delete
        #np.set_printoptions(threshold=sys.maxsize)
        out = all_scale_and_flat_indices.T[one_order_indices]
        print('all_scale_and_flat_indices.shape:',
            all_scale_and_flat_indices.shape
        )
        print('one_order_indices.shape:', one_order_indices.shape)

        print('sorted match values:')
        print(all_vals[one_order_indices])

        nlines = 20
        head = out[:nlines]
        tail = out[-nlines:]
        print('head:')
        print(head)
        print('tail:')
        print(tail)

        chead = np.array([[     2, 120520],
               [     1, 108599],
               [     0, 125250],
               [     2, 120521],
               [     2, 120029],
               [     2, 120519],
               [     2, 121011],
               [     2, 120030],
               [     2, 121012],
               [     2, 120028],
               [     2, 121010],
               [     1, 108600],
               [     1, 109096],
               [     1, 108598],
               [     1, 108102],
               [     0, 125750],
               [     0, 125249],
               [     0, 124750],
               [     0, 125251],
               [     1, 124002]])

        ctail = np.array([[     0, 108759],
               [     0, 112252],
               [     0, 111259],
               [     0, 112257],
               [     0, 125723],
               [     0, 124223],
               [     0, 128231],
               [     0, 121728],
               [     0, 128228],
               [     0, 124236],
               [     0, 125736],
               [     0, 121731],
               [     0, 128227],
               [     0, 126236],
               [     0, 126223],
               [     0, 121732],
               [     0, 123723],
               [     0, 128232],
               [     0, 121727],
               [     0, 123736]])

        try:
            assert np.array_equal(chead, head)
            assert np.array_equal(ctail, tail)
        except AssertionError:
            print('arrays did not match')
            print('correct versions (from specific thresholds):')
            print('correct head:')
            print(chead)
            print('correct tail:')
            print(ctail)
            import ipdb; ipdb.set_trace()
        #
        '''

        def match_iter_fn():
            for scale_idx, match_img_flat_idx in all_scale_and_flat_indices.T[
                one_order_indices]:

                match_image = match_images[scale_idx]
                match_pt = np.unravel_index(match_img_flat_idx,
                    match_image.shape
                )
                yield scale_idx, match_pt

    else:
        all_matches = []
        for i, match_image in enumerate(match_images):
            flat_vals = all_flat_vals[i]
            sorted_flat_indices = np.argsort(flat_vals)
            if thresholds is not None:
                idx = np.searchsorted(flat_vals[sorted_flat_indices],
                    thresholds[i]
                )
                sorted_flat_indices = sorted_flat_indices[idx:]
                del idx

            # Reversing order so indices corresponding to biggest element is
            # first, and so on, decreasing.
            sorted_flat_indices = sorted_flat_indices[::-1]
            matches = np.unravel_index(sorted_flat_indices, match_image.shape)
            all_matches.append(matches)

        if multiscale_strategy == 'fixed_scale_order':
            def match_iter_fn():
                for scale_idx in scale_order:
                    matches = all_matches[scale_idx]
                    for match_pt in zip(*matches):
                        yield scale_idx, match_pt

        elif multiscale_strategy == 'random':
            def match_iter_fn():
                per_scale_last_idx = [0] * n_scales
                scale_ps = radii_px_ps
                while True:
                    scale_idx = np.random.choice(n_scales, p=scale_ps)
                    matches = all_matches[scale_idx]

                    if all([last >= len(matches[0]) for last, matches in
                        zip(per_scale_last_idx, all_matches)]):

                        # This should end the generator's iteration.
                        return

                    # Currently just opting to retry sampling when we
                    # got something for which we have already exhausted all
                    # matches, rather than changing probabilities and choices.
                    if per_scale_last_idx[scale_idx] >= len(matches[0]):
                        continue

                    match_idx = per_scale_last_idx[scale_idx]
                    match_pt = tuple(m[match_idx] for m in matches)

                    per_scale_last_idx[scale_idx] += 1

                    yield scale_idx, match_pt

    match_iter = match_iter_fn()

    # TODO and any point to having subpixel circles anyway?
    # i.e. will packing decisions ever differ from those w/ rounded int
    # circles (and then also given that my ijroi currently only supports
    # saving non-subpixel rois...)?

    claimed = []
    center2radius = dict()

    total_n_found = 0
    roi_centers = []
    # roi_ prefix here is to disambiguate this from radii_px input, which
    # describes radii of various template scales to use for matching, but
    # NOT the radii of the particular matched ROI outputs.
    roi_radii_px = []

    if ns is not None:
        n_found_per_scale = [0] * n_scales

    max_exclusion_radius_px = max(exclusion_radius_frac * r for r in radii_px)
    scale_info_printed = [False] * n_scales
    for scale_idx, pt in match_iter:
        if total_n is not None:
            if total_n_found >= total_n:
                break

        elif ns is not None:
            if all([n_found >= n for n_found, n in zip(n_found_per_scale, ns)]):
                break

            if n_found_per_scale[scale_idx] >= ns[scale_idx]:
                continue

        d = ds[scale_idx]
        offset = d / 2
        center = (pt[0] + offset, pt[1] + offset)
        del offset

        if exclusion_mask is not None:
            if not exclusion_mask[tuple(int(round(v)) for v in center)]:
                continue

        radius_px = radii_px[scale_idx]
        exclusion_radius_px = radius_px * exclusion_radius_frac
        if debug:
            if not scale_info_printed[scale_idx]:
                print('template d:', d)
                print('radius_px:', radius_px)
                print('exclusion_radius_frac:', exclusion_radius_frac)
                print('exclusion_radius_px:', exclusion_radius_px)
                scale_info_printed[scale_idx] = True

        # Ideally I'd probably use a data structure that doesn't need to
        # be rebuilt each time (and k-d trees in general don't, but
        # scipy's doesn't support that (perhaps b/c issues w/ accumulating
        # rebalancing costs?), nor do they seem to offer spatial
        # structures that do)
        if len(claimed) > 0:
            tree = cKDTree(claimed)
            # (would need to relax if supporting 3d)
            assert tree.m == 2
            # TODO tests to check whether this is right dist bound
            # ( / 2 ?)
            dists, locs = tree.query(center,
                distance_upper_bound=max_exclusion_radius_px * 2
            )
            # Docs say this indicates no neighbors found.
            if locs != tree.n:
                try:
                    len(dists)
                except:
                    dists = [dists]
                    locs = [locs]

                conflict = False
                for dist, neighbor_idx in zip(dists, locs):
                    # TODO TODO any way to add metadata to tree element to avoid
                    # this lookup? (+ dist bound above)
                    neighbor_r = center2radius[tuple(tree.data[neighbor_idx])]
                    # We already counted the radius about the tentative
                    # new ROI, but that assumes all neighbors are just points.
                    # This prevents small ROIs from being placed inside big
                    # ones.
                    # TODO check these two lines
                    dist -= neighbor_r * exclusion_radius_frac
                    if dist <= exclusion_radius_px:
                        conflict = True
                        break
                if conflict:
                    continue

        total_n_found += 1
        roi_centers.append(center)
        roi_radii_px.append(radius_px)

        if draw_on is not None:
            draw_pt = (ups * pt[0], ups * pt[1])[::-1]
            draw_c = (
                int(round(ups * center[0])),
                int(round(ups * center[1]))
            )[::-1]

            # TODO factor this stuff out into post-hoc drawing fn, so that
            # roi filters in here can exclude stuff? or maybe just factor out
            # the filtering stuff anyway?

            if draw_bboxes:
                cv2.rectangle(draw_on, draw_pt,
                    (draw_pt[0] + ups * d, draw_pt[1] + ups * d), (0,0,255), 2
                )

            # TODO maybe diff colors for diff scales? (random or from kwarg)
            if draw_circles:
                cv2.circle(draw_on, draw_c, int(round(ups * radius_px)),
                    (255,0,0), 2
                )

            if draw_nums:
                cv2.putText(draw_on, str(total_n_found), draw_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
                )

        claimed.append(center)
        center2radius[tuple(center)] = radius_px

    # TODO change to not need to import (all of?) viz? would that help avoid some
    # circular import issues?
    '''
    if debug and _show_packing_constraints:
        title = 'greedy_roi_packing overlap exlusion mask'
        viz.imshow(claimed, title)
    '''

    if debug and draw_on is not None and _show_fit:
        viz.imshow(draw_on, 'greedy_roi_packing fit')

    # TODO also use kdtree for this step
    if not min_neighbors:
        filtered_roi_centers = roi_centers
        filtered_roi_radii = roi_radii_px
    else:
        # TODO maybe extend this to requiring the nth closest be closer than a
        # certain amount (to exclude 2 (or n) cells off by themselves)
        filtered_roi_centers = []
        filtered_roi_radii = []
        for i, (center, radius) in enumerate(zip(roi_centers, roi_radii_px)):
            n_neighbors = 0
            for j, other_center in enumerate(roi_centers):
                if i == j:
                    continue

                dist = util.euclidean_dist(center, other_center)
                if dist <= min_dist2neighbor_px:
                    n_neighbors += 1

                if n_neighbors >= min_neighbors:
                    filtered_roi_centers.append(center)
                    filtered_roi_radii.append(radius)
                    break

    assert len(filtered_roi_centers) == len(filtered_roi_radii)
    return np.array(filtered_roi_centers), np.array(filtered_roi_radii)


def scale_template(template_data, um_per_pixel_xy, target_cell_diam_um=None,
    target_cell_diam_px=None, target_template_d=None, debug=False):
    import cv2

    if target_cell_diam_um is None:
        # TODO make either of other kwargs also work (any of the 3 should
        # be alone)
        raise NotImplementedError

    template = template_data['template']
    margin = template_data['margin']
    # We enforce both elements of shape are same at creation.
    d = template.shape[0]

    target_cell_diam_px = target_cell_diam_um / um_per_pixel_xy

    # TODO which of these is correct? both? assert one is w/in
    # rounding err of other?
    template_cell_diam_px = d - 2 * margin
    template_scale = target_cell_diam_px / template_cell_diam_px
    '''
    template_cell_diam_um = template_data['mean_cell_diam_um']
    print(f'template_cell_diam_um: {template_cell_diam_um}')
    template_scale = target_cell_diam_um / template_cell_diam_um
    '''
    new_template_d = int(round(template_scale * d))
    new_template_shape = tuple([new_template_d] * len(template.shape))

    if debug:
        print(f'\nscale_template:\nd: {d}\nmargin: {margin}')
        print(f'um_per_pixel_xy: {um_per_pixel_xy}')
        print(f'target_cell_diam_um: {target_cell_diam_um}')
        print(f'target_cell_diam_px: {target_cell_diam_px}')
        print(f'template_cell_diam_px: {template_cell_diam_px}')
        print(f'template_scale: {template_scale}')
        print(f'new_template_d: {new_template_d}')
        print('')

    if new_template_d != d:
        scaled_template = cv2.resize(template, new_template_shape)
        scaled_template_cell_diam_px = \
            template_cell_diam_px * new_template_d / d

        return scaled_template, scaled_template_cell_diam_px

    else:
        return template.copy(), template_cell_diam_px


def _get_template_roi_radius_px(template_data, if_template_d=None, _round=True):
    template = template_data['template']
    margin = template_data['margin']
    d = template.shape[0]
    template_cell_diam_px = d - 2 * margin
    template_cell_radius_px = template_cell_diam_px / 2

    radius_frac = template_cell_radius_px / d

    if if_template_d is None:
        if_template_d = d

    radius_px = radius_frac * if_template_d
    if _round:
        radius_px = int(round(radius_px))
    return radius_px


# TODO test this w/ n.5 centers / radii
def get_circle_ijroi_input(center_px, radius_px):
    """Returns appropriate first arg for my ijroi.write_roi
    """
    min_corner = [center_px[0] - radius_px, center_px[1] - radius_px]
    max_corner = [
        min_corner[0] + 2 * radius_px,
        min_corner[1] + 2 * radius_px
    ]
    bbox = np.array([min_corner, max_corner])
    return bbox


# TODO move to viz? or maybe move all roi stuff to a new module?
def plot_circles(draw_on, centers, radii):
    import cv2
    draw_on = cv2.normalize(draw_on, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    draw_on = cv2.equalizeHist(draw_on)

    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.imshow(draw_on, cmap='gray')
    for center, radius in zip(centers, radii):
        roi_circle = plt.Circle((center[1] - 0.5, center[0] - 0.5), radius,
            fill=False, color='r'
        )
        ax.add_artist(roi_circle)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def fit_circle_rois(tif, template_data=None, avg=None, movie=None,
    method_str='cv2.TM_CCOEFF_NORMED', thresholds=None,
    exclusion_radius_frac=0.8, min_neighbors=None,
    debug=False, _packing_debug=False, show_fit=None,
    write_ijrois=False, _force_write_to=None, overwrite=False,
    exclude_dark_regions=None, dark_fraction_beyond_dhist_min=0.6,
    max_n_rois=650, min_n_rois=150,
    per_scale_max_n_rois=None,
    per_scale_min_n_rois=None, threshold_update_factor=0.7,
    update_factor_shrink_factor=0.7, max_threshold_tries=4,
    _um_per_pixel_xy=None, multiscale=True, roi_diams_px=None,
    roi_diams_um=None, roi_diams_from_kmeans_k=None,
    multiscale_strategy='one_order', template_d2match_value_scale_fn=None,
    allow_duplicate_px_scales=False, _show_scaled_templates=False,
    verbose=False, **kwargs):
    """
    Even if movie or avg is passed in, tif is used to find metadata and
    determine where to save ImageJ ROIs.

    _um_per_pixel_xy only used for testing. Normally, XML is found from `tif`,
    and that is loaded to get this value.

    Returns centers_px, radii_px
    (both w/ coordinates and conventions ijrois uses)
    """
    import tifffile
    import cv2
    import ijroi
    from scipy.cluster.vq import vq

    if debug and show_fit is None:
        show_fit = True

    # TODO update all kwargs to go through a dict (store existing defaults as
    # dict at module level?) + need to handle passing of remaining to greedy_...
    # appropriately (don't pass stuff it won't take / don't unwrap and modify
    # so it only uses relevant ones)
    method_str2defaults = {
        # Though this does not work at all scales
        # (especially sensitive since not normed)
        'cv2.TM_CCOEFF': {'threshold': 4000.0, 'exclude_dark_regions': False},
        'cv2.TM_CCOEFF_NORMED': {'threshold': 0.3, 'exclude_dark_regions': True}
    }
    if method_str in method_str2defaults:
        for k, v in method_str2defaults[method_str].items():
            if k not in kwargs or kwargs[k] is None:
                kwargs[k] = v

    threshold = kwargs.pop('threshold')
    exclude_dark_regions = kwargs.pop('exclude_dark_regions')

    # Will divide rather than multiply by this,
    # if we need to increase threshold.
    assert threshold_update_factor < 1 and threshold_update_factor > 0

    # TODO also provide fitting for this fn in extract_template?
    mvw_key = 'match_value_weights'
    if template_d2match_value_scale_fn is not None:
        assert multiscale and multiscale_strategy == 'one_scale'
        assert mvw_key not in kwargs
        match_value_weights = []
    else:
        try:
            match_value_weights = kwargs.pop(mvw_key)
        except KeyError:
            match_value_weights = None

    if template_data is None:
        # TODO maybe options to cache this data across calls?
        # might not matter...
        template_data = load_template_data(err_if_missing=True)

    template = template_data['template']
    margin = template_data['margin']
    mean_cell_diam_um = template_data['mean_cell_diam_um']
    frame_shape = template_data['frame_shape']

    if _um_per_pixel_xy is None:
        keys = util.tiff_filename2keys(tif)
        ti_dir = util.thorimage_dir(*tuple(keys))
        xmlroot = thor.get_thorimage_xmlroot(ti_dir)
        um_per_pixel_xy = thor.get_thorimage_pixelsize_um(xmlroot)
        del keys, ti_dir, xmlroot
    else:
        um_per_pixel_xy = _um_per_pixel_xy

    if multiscale:
        # Centroids are scalars in units of um diam
        kmeans_k2cluster_cell_diams = \
            template_data['kmeans_k2cluster_cell_diams']

        if roi_diams_px is not None:
            assert roi_diams_um is None and roi_diams_from_kmeans_k is None
            roi_diams_um = [rd_px * um_per_pixel_xy for rd_px in roi_diams_px]

        if roi_diams_um is None and roi_diams_from_kmeans_k is None:
            roi_diams_from_kmeans_k = 2

        if roi_diams_um is None:
            roi_diams_um = kmeans_k2cluster_cell_diams[roi_diams_from_kmeans_k]

            if verbose:
                in_px = roi_diams_um / um_per_pixel_xy
                print(f'Using ROI diameters {roi_diams_um} um ({in_px} px) from'
                    f' K-means (k={roi_diams_from_kmeans_k}) on data used to '
                    'generate template.'
                )
                del in_px

            if multiscale_strategy == 'random':
                all_cell_diams_um = template_data['all_cell_diams_um']
                clusters, _ = vq(all_cell_diams_um, roi_diams_um)
                count_clusters, counts = np.unique(clusters, return_counts=True)
                # otherwise would need to reindex the counts
                assert np.array_equal(count_clusters, np.sort(count_clusters))
                radii_px_ps = counts / np.sum(counts)
                kwargs['radii_px_ps'] = radii_px_ps
                print('Calculated these probabilities from template data:',
                    radii_px_ps
                )
    else:
        assert roi_diams_px is None
        assert roi_diams_um is None
        assert roi_diams_from_kmeans_k is None
        roi_diams_um = [mean_cell_diam_um]

    n_scales = len(roi_diams_um)

    if thresholds is None:
        thresholds = [threshold] * n_scales
    else:
        # TODO better way to specify thresholds in kmeans case, where
        # user may not know # thresholds needed in advance?
        assert len(thresholds) == n_scales
    del threshold
    thresholds = np.array(thresholds)

    if write_ijrois or _force_write_to is not None:
        write_ijrois = True

        path, tiff_last_part = split(tif)
        tiff_parts = tiff_last_part.split('.tif')
        assert len(tiff_parts) == 2 and tiff_parts[1] == ''
        fname = join(path, tiff_parts[0] + '_rois.zip')

        # TODO TODO change. fname needs to always be under
        # analysis_output_root (or just change input in
        # kc_natural_mixes/populate_db.py).
        # TODO or at least err if not subdir of it
        # see: https://stackoverflow.com/questions/3812849

        if _force_write_to is not None:
            if _force_write_to == True:
                fname = join(path, tiff_parts[0] + '_auto_rois.zip')
            else:
                fname = _force_write_to

        # TODO also check for modifications before overwriting (mtime in that hidden
        # file)
        elif not overwrite and exists(fname):
            print(fname, 'already existed. returning.')
            return None, None, None, None

    if avg is None:
        if movie is None:
            movie = tifffile.imread(tif)
        avg = movie.mean(axis=0)
    assert avg.shape[0] == avg.shape[1]
    orig_frame_d = avg.shape[0]

    # It seemed to me that picking a new threshold on cv2.TM_CCOEFF_NORMED was
    # not sufficient to reproduce cv2.TM_CCOEFF performance, so even if the
    # normed version were useful to keep the same threshold across image scales,
    # it seems other problems prevent me from using that in my case, so I'm
    # rescaling the image to match against.

    frame_downscaling = 1.0
    if avg.shape != frame_shape:
        scaled_avg = cv2.resize(avg, frame_shape)

        new_frame_d = scaled_avg.shape[0]
        frame_downscaling = orig_frame_d / new_frame_d
        del new_frame_d
        um_per_pixel_xy *= frame_downscaling
    else:
        scaled_avg = avg

    if debug:
        print('frame downscaling:', frame_downscaling)
        print('scaled_avg.shape:', scaled_avg.shape)

    if exclude_dark_regions:
        histvals, bins = np.histogram(scaled_avg.flat, bins=100, density=True)
        hv_deltas = np.diff(histvals)
        # TODO get the + 3 from a parameter controller percentage beyond
        # count delta min
        # min from: histvals[idx + 1] - histvals[idx]
        idx = np.argmin(hv_deltas)

        # TODO if this method of calculating dark_thresh doesn't seem robust,
        # compare robustness to thresholds from percetile of overal image,
        # or fixed thresholds on image scaled to [0,1], or fixed fractional
        # adjustment from delta hist threshold

        # Originally, dark_thresh was from bins[idx + 4], and that seemed to
        # work OK, so on one image, I calculated initial value (~0.5 -> 0.5)
        # of this from: ((scaled_avg <= bins[idx + 4]).sum() -
        # (scaled_avg <= bins[idx]).sum()) / scaled_avg.size (=0.543...)
        #dark_thresh = bins[idx + 4]
        dh_min_fractile = (scaled_avg <= bins[idx]).sum() / scaled_avg.size
        dark_thresh = np.percentile(scaled_avg,
            100 * (dark_fraction_beyond_dhist_min + dh_min_fractile)
        )

        exclusion_mask = scaled_avg >= dark_thresh
        if debug:
            fig, axs = plt.subplots(ncols=2)
            axs[0].imshow(scaled_avg)
            axs[1].imshow(exclusion_mask)
            axs[1].set_title('exclusion mask')
    else:
        exclusion_mask = None

    # We enforce earlier that template must be symmetric.
    d, d2 = template.shape
    assert d == d2

    match_images = []
    template_ds = []
    per_scale_radii_px = []
    for i, roi_diam_um in enumerate(roi_diams_um):
        scaled_template, scaled_template_cell_diam_px = scale_template(
            template_data, um_per_pixel_xy, roi_diam_um, debug=debug
        )
        scaled_radius_px = scaled_template_cell_diam_px / 2
        if debug:
            print('scaled template shape:', scaled_template.shape)

        if debug and _show_scaled_templates:
            fig, ax = plt.subplots()
            ax.imshow(scaled_template)
            title = f'scaled template (roi_diam_um={roi_diam_um:.2f})'
            if roi_diams_px is not None:
                title += f'\n(roi_diam_px={roi_diams_px[i]:.1f})'
            ax.set_title(title)

        # see note below about what i'd need to do to continue using
        # a check like this
        '''
        if template.shape != scaled_template.shape:
            # Just for checking that conversion back to original coordinates
            # (just scale diff) seems to be working.
            radius_px_before_scaling = int(round((d - 2 * margin) / 2))
        '''

        match_image = template_match(scaled_avg, scaled_template,
            method_str=method_str
        )
        if debug:
            print(f'scaled_template_cell_diam_px: '
                f'{scaled_template_cell_diam_px}'
            )
            print(f'scaled_radius_px: {scaled_radius_px}')

        template_d = scaled_template.shape[0]
        if (match_value_weights is not None and
            template_d2match_value_scale_fn is not None):

            match_value_weights.append(
                template_d2match_value_scale_fn(template_d)
            )

        match_images.append(match_image)
        template_ds.append(template_d)
        per_scale_radii_px.append(scaled_radius_px)

    if debug:
        print('template_ds:', template_ds)

    if len(set(template_ds)) != len(template_ds):
        if not allow_duplicate_px_scales:
            raise ValueError(f'roi_diams_um: {roi_diams_um} led to duplicate '
                f'pixel template scales ({template_ds})'
            )
        else:
            # TODO would still probably have to de-duplicate before passing to
            # packing fn
            raise NotImplementedError

    # TODO one fn that just returns circles, another to draw?
    draw_on = scaled_avg

    if per_scale_max_n_rois is not None or per_scale_min_n_rois is not None:
        if per_scale_max_n_rois is not None:
            assert len(per_scale_max_n_rois) == n_scales, \
                f'{len(per_scale_max_n_rois)} != {n_scales}'

        if per_scale_min_n_rois is not None:
            assert len(per_scale_min_n_rois) == n_scales, \
                f'{len(per_scale_min_n_rois)} != {n_scales}'

        print('Per-scale bounds on number of ROIs overriding global bounds.')
        min_n_rois = None
        max_n_rois = None
        per_scale_n_roi_bounds = True
    else:
        per_scale_n_roi_bounds = False

    threshold_tries_remaining = max_threshold_tries
    while threshold_tries_remaining > 0:
        # Regarding exclusion_radius_frac: 0.3 allowed too much overlap, 0.5
        # borderline too much w/ non-normed method (0.7 OK there)
        # (r=4,er=4,6 respectively, in 0.5 and 0.7 cases)
        if debug:
            print('per_scale_radii_px:', per_scale_radii_px)

        centers_px, radii_px = greedy_roi_packing(match_images, template_ds,
            per_scale_radii_px, thresholds=thresholds,
            min_neighbors=min_neighbors, exclusion_mask=exclusion_mask,
            exclusion_radius_frac=exclusion_radius_frac, draw_on=draw_on,
            draw_bboxes=False, draw_nums=False,
            multiscale_strategy=multiscale_strategy, debug=_packing_debug,
            match_value_weights=match_value_weights,
            _src_img_shape=scaled_avg.shape, **kwargs
        )

        n_found_per_scale = {r_px: 0 for r_px in per_scale_radii_px}
        for r_px in radii_px:
            n_found_per_scale[r_px] += 1
        assert len(centers_px) == sum(n_found_per_scale.values())

        if debug:
            print('number of ROIs found at each pixel radius scale:')
            pprint(n_found_per_scale)

        if per_scale_n_roi_bounds:
            wrong_num = False
            for i in range(n_scales):
                r_px = per_scale_radii_px[i]
                thr = thresholds[i]
                n_found = n_found_per_scale[r_px]

                sstr = f' w/ radius={r_px}px @ thr={thr:.2f}'
                have_retries = threshold_tries_remaining > 1
                if have_retries:
                    sstr += f'\nthr:={{:.2f}}'

                if per_scale_max_n_rois is not None:
                    smax = per_scale_max_n_rois[i]
                    if smax < n_found:
                        thresholds[i] /= threshold_update_factor
                        print((f'too many ROIs ({n_found} > {smax}){sstr}'
                            ).format(thresholds[i] if have_retries else tuple()
                        ))
                        wrong_num = True

                if per_scale_min_n_rois is not None:
                    smin = per_scale_min_n_rois[i]
                    if n_found < smin:
                        thresholds[i] *= threshold_update_factor
                        print(f'too few ROIs ({n_found} < {smin}){sstr}'.format(
                            thresholds[i] if have_retries else tuple()
                        ))
                        wrong_num = True

            if not wrong_num:
                break
            elif debug:
                print('')

        n_rois_found = len(centers_px)
        if not per_scale_n_roi_bounds:
            if ((min_n_rois is None or min_n_rois <= n_rois_found) and
                (max_n_rois is None or n_rois_found <= max_n_rois)):
                break

        threshold_tries_remaining -= 1
        if threshold_tries_remaining == 0:
            if debug or _packing_debug:
                plt.show()

            raise RuntimeError(f'too many/few ({n_rois_found}) ROIs still '
                f'detected after {max_threshold_tries} attempts to modify '
                'threshold. try changing threshold(s)?'
            )

        if not per_scale_n_roi_bounds:
            # TODO maybe squeeze to threshold if just one
            fail_notice_suffix = f', with thresholds={thresholds}'
            if max_n_rois is not None and n_rois_found > max_n_rois:
                thresholds /= threshold_update_factor
                fail_notice = \
                    f'found too many ROIs ({n_rois_found} > {max_n_rois})'

            elif min_n_rois is not None and n_rois_found < min_n_rois:
                thresholds *= threshold_update_factor
                fail_notice = \
                    f'found too few ROIs ({n_rois_found} < {min_n_rois})'

            fail_notice += fail_notice_suffix
            print(f'{fail_notice}\n\nretrying with thresholds={thresholds}')

        if update_factor_shrink_factor is not None:
            threshold_update_factor = \
                1 - (1 - threshold_update_factor) * update_factor_shrink_factor

    if frame_downscaling != 1.0:
        # TODO if i want to keep doing this check, while also supporting
        # multiscale case, gonna need to check (the set of?) radii returned
        # (would i need more info for that?)
        '''
        # This is to invert any previous scaling into coordinates for matching
        radius_px = scaled_radius_px * frame_downscaling

        # always gonna be true? seems like if a frame were 7x7, converting size
        # down to say 2x2 and back up by same formula would yield same result
        # as a 6x6 input or something, no?
        assert radius_px == radius_px_before_scaling
        del radius_px_before_scaling
        '''
        centers_px = centers_px * frame_downscaling
        radii_px  = radii_px * frame_downscaling

    # TODO would some other (alternating?) rounding rule help?
    # TODO random seed then randomly choose between floor and ceil for stuff
    # at 0.5?
    # TODO TODO or is rounding wrong? do some tests to try to figure out
    centers_px = np.round(centers_px).astype(np.uint16)
    radii_px = np.round(radii_px).astype(np.uint16)
    # this work if centers is empty?
    assert np.all(centers_px >= 0) and np.all(centers_px < orig_frame_d)

    if show_fit:
        fig, ax = plot_circles(avg, centers_px, radii_px)
        if tif is None:
            title = 'fit circles'
        else:
            title = util.tiff_title(tif)
        ax.set_title(title)

        roi_plot_dir = 'auto_rois'
        if not exists(roi_plot_dir):
            print(f'making directory {roi_plot_dir} for plots of ROI fits')
            os.makedirs(roi_plot_dir)

        roi_plot_fname = join(roi_plot_dir, title.replace('/','_') + '.png')
        print(f'Writing image showing fit ROIs to {roi_plot_fname}')
        fig.savefig(roi_plot_fname)

    if write_ijrois:
        auto_md_fname = autoroi_metadata_filename(fname)

        name2bboxes = list()
        for i, (center_px, radius_px) in enumerate(zip(centers_px, radii_px)):
            # TODO TODO test that these radii are preserved across
            # round trip save / loads?
            bbox = get_circle_ijroi_input(center_px, radius_px)
            name2bboxes.append((str(i), bbox))

        print('Writing ImageJ ROIs to {} ...'.format(fname))
        # TODO TODO TODO uncomment
        '''
        ijroi.write_oval_roi_zip(name2bboxes, fname)

        with open(auto_md_fname, 'wb') as f:
            data = {
                'mtime': getmtime(fname)
            }
            pickle.dump(data, f)
        '''

    ns_found = [n_found_per_scale[rpx] for rpx in per_scale_radii_px]

    return centers_px, radii_px, thresholds, ns_found


def template_data_file():
    template_cache = 'template.p'
    return join(util.analysis_output_root(), template_cache)


def load_template_data(err_if_missing=False):
    template_cache = template_data_file()
    if exists(template_cache):
        with open(template_cache, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        if err_if_missing:
            raise IOError(f'template data not found at {template_cache}')
        return None


# TODO delete (+ probably delete contour2mask too and replace use of both w/ ijroi2mask)
# don't like this convexHull based approach though...
# (because roi may be intentionally not a convex hull)
def ijroi2cv_contour(roi):
    import cv2

    ## cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## cnts[1][0].shape
    ## cnts[1][0].dtype
    # from inspecting output of findContours, as above:
    #cnt = np.expand_dims(ijroi, 1).astype(np.int32)
    # TODO fix so this isn't necessary. in case of rois that didn't start as
    # circles, the convexHull may occasionally not be equal to what i want
    cnt = cv2.convexHull(roi.astype(np.int32))
    # if only getting cnt from convexHull, this is probably a given...
    assert cv2.contourArea(cnt) > 0
    return cnt
#


def roi_center(roi):
    import cv2
    cnt = ijroi2cv_contour(roi)
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array((cx, cy))


def roi_centers(rois):
    centers = []
    for roi in rois:
        center = roi_center(roi)
        # pretty close to (w/in 0.5 in each dim) np.mean(roi, axis=0),
        # in at least one example i played with
        centers.append(center)
    return np.array(centers)


def point_idx(xys_to_check, pt_xy, swap_xy=False):
    if not swap_xy:
        x, y = pt_xy
    else:
        y, x = pt_xy

    matching_pt = (
        (xys_to_check[:,0] == x) &
        (xys_to_check[:,1] == y)
    )
    assert matching_pt.sum() == 1
    return np.argwhere(matching_pt)[0][0]


def correspond_rois(left_centers_or_seq, *right_centers, cost_fn=None,
    max_cost=None, show=False, write_plots=True, left_name='Left',
    right_name='Right', name_prefix='', draw_on=None, title='', colors=None,
    connect_centers=True, pairwise_plots=True, pairwise_same_style=False,
    roi_numbers=False, jitter=True, progress=None, squeeze=True,
    verbose=False, debug_points=None):
    """
    Args:
    left_centers_or_seq (list): (length n_timepoints) list of (n_rois x 2)
        arrays of ROI center coordinates.

    Returns:
    lr_matches: list of arrays matching ROIs in one timepoint to ROIs in the
        next.

    left_unmatched: list of arrays with ROI labels at time t,
        without a match at time (t + 1)

    right_unmatched: same as left_unmatched, but for (t + 1) with respect to t.

    total_costs: array of sums of costs from matching.

    fig: matplotlib figure handle to the figure with all ROIs on it,
        for modification downstream.
    """
    # TODO doc support for ROI inputs / rewrite to expect them
    # (to use jaccard, etc)

    from scipy.optimize import linear_sum_assignment
    import seaborn as sns

    if cost_fn is None:
        # this was to avoid a circular import if it was in function definition
        cost_fn = util.euclidean_dist

    # TODO maybe unsupport two args case to be more concise
    if len(right_centers) == 0:
        sequence_of_centers = left_centers_or_seq

    elif len(right_centers) == 1:
        right_centers = right_centers[0]
        sequence_of_centers = [left_centers_or_seq, right_centers]

    else:
        raise ValueError('wrong number of arguments')

    if progress is None:
        progress = len(sequence_of_centers) >= 10
    if progress:
        from tqdm import tqdm

    if max_cost is None:
        raise ValueError('max_cost must not be None')

    if verbose:
        print(f'max_cost: {max_cost:.2f}')

    default_two_colors = ['red', 'blue']
    if len(sequence_of_centers) == 2:
        pairwise_plots = False
        scatter_alpha = 0.6
        scatter_marker = None
        labels = [n + ' centers' for n in (left_name, right_name)]
        if colors is None:
            colors = default_two_colors
    else:
        scatter_alpha = 0.8
        scatter_marker = 'x'
        labels = [name_prefix + str(i) for i in range(len(sequence_of_centers))]
        if colors is None:
            colors = sns.color_palette('hls', len(sequence_of_centers))

    # TODO don't copy after removing need for flip
    # Copying so that flip doesn't screw with input data.
    new_sequence_of_centers = []
    for i, centers in enumerate(sequence_of_centers):
        # Otherwise it should be an ndarray representing centers
        # TODO assertion on dims in ndarray case
        if type(centers) is list:
            centers = roi_centers(centers)

        # This is just to make them display right (not transposed).
        # Should not change any of the matching.
        # TODO remove need for this flip
        new_sequence_of_centers.append(np.flip(centers, axis=1))
    sequence_of_centers = new_sequence_of_centers

    fig = None
    if show:
        figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
        if draw_on is None:
            color = 'black'
        else:
            ax.imshow(draw_on, cmap='gray')
            ax.axis('off')
            color = 'yellow'
        fontsize = 8
        text_x_offset = 2
        plot_format = 'png'

        if jitter:
            np.random.seed(50)
            jl = -0.1
            jh = 0.1

    unmatched_left = []
    unmatched_right = []
    lr_matches = []
    cost_totals = []

    if progress:
        centers_iter = tqdm(range(len(sequence_of_centers) - 1))
        print('Matching ROIs across timepoints:')
    else:
        centers_iter = range(len(sequence_of_centers) - 1)

    for ci, k in enumerate(centers_iter):
        left_centers = sequence_of_centers[k]
        right_centers = sequence_of_centers[k + 1]

        # TODO TODO use pdist / something else under scipy.spatial.distance?
        # TODO other / better ways to generate cost matrix?
        # pairwise jacard (would have to not take centers then)?
        # TODO why was there a "RuntimeWarning: invalid valid encounterd in
        # multiply" here ocassionally? it still seems like we had some left and
        # right centers, so idk
        costs = np.empty((len(left_centers), len(right_centers))) * np.nan
        for i, cl in enumerate(left_centers):
            for j, cr in enumerate(right_centers):
                # TODO short circuit as appropriate? better way to loop over
                # coords we need?
                cost = cost_fn(cl, cr)
                costs[i,j] = cost

        '''
        if verbose:
            print(f'(iteration {ci}) fraction of costs >= max_cost:',
                '{:.3f}'.format((costs >= max_cost).sum() / costs.size)
            )
        '''

        # TODO delete. problem does not seem to be in this fn.
        '''
        if debug_points and ci in debug_points:
            print(f'iteration {ci}:')
            ln = 3
            for pt_info in debug_points[ci]:
                name = pt_info['name']
                xy0 = pt_info['xy0']
                # TODO print cost wrt this point
                xy1 = pt_info['xy1']

                # swap_xy etc b/c of flip earlier
                idx = point_idx(left_centers, xy0, swap_xy=True)
                print(f'lowest {ln} costs for point {name} in '
                    'left_centers:'
                )
                # TODO also print to which other points (x,y)
                # correspond to these ln lowest costs
                print(np.sort(costs[idx, :])[:ln])
        '''

        # TODO TODO TODO test that setting these to an arbitrarily large number
        # produces matching equivalent to setting them to max_cost here
        costs[costs >= max_cost] = max_cost

        # TODO was Kellan's method of matching points not equivalent to this?
        # or per-timestep maybe it was (or this was better), but he also
        # had a way to evolve points over time (+ a particular cost)?

        left_idx, right_idx = linear_sum_assignment(costs)
        # Just to double-check properties I assume about the assignment
        # procedure.
        assert len(left_idx) == len(np.unique(left_idx))
        assert len(right_idx) == len(np.unique(right_idx))

        n_not_drawn = None
        if show:
            if jitter:
                left_jitter = np.random.uniform(low=jl, high=jh,
                    size=left_centers.shape)
                right_jitter = np.random.uniform(low=jl, high=jh,
                    size=right_centers.shape)

                left_centers_to_plot = left_centers + left_jitter
                right_centers_to_plot = right_centers + right_jitter
            else:
                left_centers_to_plot = left_centers
                right_centers_to_plot = right_centers

            if pairwise_plots:
                # TODO maybe change multiple pairwise plots to be created as
                # axes within one the axes from one call to subplots
                pfig, pax = plt.subplots(figsize=figsize)
                if pairwise_same_style:
                    pmarker = scatter_marker
                    c1 = colors[k]
                    c2 = colors[k + 1]
                else:
                    pmarker = None
                    c1 = default_two_colors[0]
                    c2 = default_two_colors[1]

                if draw_on is not None:
                    pax.imshow(draw_on, cmap='gray')
                    pax.axis('off')

                pax.scatter(*left_centers_to_plot.T, label=labels[k],
                    color=c1, alpha=scatter_alpha,
                    marker=pmarker
                )
                pax.scatter(*right_centers_to_plot.T, label=labels[k + 1],
                    color=c2, alpha=scatter_alpha,
                    marker=pmarker
                )
                psuffix = f'{k} vs. {k+1}'
                if len(name_prefix) > 0:
                    psuffix = f'{name_prefix} ' + psuffix
                if len(title) > 0:
                    ptitle = f'{title}, ' + psuffix
                else:
                    ptitle = psuffix
                pax.set_title(ptitle)
                pax.legend()

            ax.scatter(*left_centers_to_plot.T, label=labels[k],
                color=colors[k], alpha=scatter_alpha,
                marker=scatter_marker
            )
            # TODO factor out scatter + opt numbers (internal fn?)
            if roi_numbers:
                for i, (x, y) in enumerate(left_centers_to_plot):
                    ax.text(x + text_x_offset, y, str(i),
                        color=colors[k], fontsize=fontsize
                    )

            # Because generally this loop only scatterplots the left_centers,
            # so without this, the last set of centers would not get a
            # scatterplot.
            if (k + 1) == (len(sequence_of_centers) - 1):
                last_centers = right_centers_to_plot

                ax.scatter(*last_centers.T, label=labels[-1],
                    color=colors[-1], alpha=scatter_alpha,
                    marker=scatter_marker
                )
                if roi_numbers:
                    for i, (x, y) in enumerate(last_centers):
                        ax.text(x + text_x_offset, y, str(i),
                            color=colors[-1], fontsize=fontsize
                        )

            if connect_centers:
                n_not_drawn = 0
                for li, ri in zip(left_idx, right_idx):
                    if costs[li,ri] >= max_cost:
                        n_not_drawn += 1
                        continue
                        #linestyle = '--'
                    else:
                        linestyle = '-'

                    lc = left_centers_to_plot[li]
                    rc = right_centers_to_plot[ri]
                    correspondence_line = ([lc[0], rc[0]], [lc[1], rc[1]])

                    ax.plot(*correspondence_line, linestyle=linestyle,
                        color=color, alpha=0.7)

                    if pairwise_plots:
                        pax.plot(*correspondence_line, linestyle=linestyle,
                            color=color, alpha=0.7)

                # TODO didn't i have some fn for getting filenames from things
                # like titles? use that if so
                # TODO plot format + flag to control saving + save to some
                # better dir
                # TODO separate dir for these figs? or at least place where some
                # of other figs currently go?
                if pairwise_plots and write_plots:
                    fname = util.to_filename(ptitle) + plot_format
                    print(f'writing to {fname}')
                    pfig.savefig(fname)

        k_unmatched_left = set(range(len(left_centers))) - set(left_idx)
        k_unmatched_right = set(range(len(right_centers))) - set(right_idx)

        # TODO why is costs.min() actually 0? that seems unlikely?
        match_costs = costs[left_idx, right_idx]
        total_cost = match_costs.sum()

        to_unmatch = match_costs >= max_cost
        # For checking consistent w/ draw output above
        if verbose or n_not_drawn is not None:
            n_unmatched = to_unmatch.sum()
            if n_not_drawn is not None:
                assert n_not_drawn == n_unmatched, \
                    f'{n_not_drawn} != {n_unmatched}'
            if verbose:
                print(f'(iteration={ci}) unmatched {n_unmatched} for exceeding'
                    ' max_cost'
                )

        if debug_points and ci in debug_points:
            l_idxs = []
            r_idxs = []
            for pt_info in debug_points[ci]:
                name = pt_info['name']
                # swap_xy etc b/c of flip earlier
                xy0 = pt_info['xy0']
                xy1 = pt_info['xy1']
                print(f'name: {name}, xy0: {xy0}, xy1: {xy1}')

                lidx = point_idx(left_centers, xy0, swap_xy=True)
                assert left_idx.max() <= len(left_centers)

                midx0 = np.where(left_idx == lidx)[0]
                if len(midx0) > 0:
                    assert len(midx0) == 1
                    midx0 = midx0[0]
                    assert left_idx[midx0] == lidx
                    lpt = left_centers[left_idx[midx0]]
                    assert tuple(lpt)[::-1] == xy0
                    # since by the time debug_points are generated, point
                    # matching seems off, rpt will not necessarily be
                    # equal to lpt.
                    rpt_idx = right_idx[midx0]
                    rpt = right_centers[rpt_idx]
                    mcost0 = match_costs[midx0]
                    print(f'xy0 matched ({lidx}:{lpt} -> {rpt_idx}:{rpt}) '
                        f'at cost {mcost0:.3f}'
                    )
                    if to_unmatch[midx0]:
                        print('xy0 will be unmatched for cost >= max_cost!')
                    else:
                        l_idxs.append((name, lidx))
                        # For use debugging downstream of this function.
                        pt_info['xy0_lidx'] = lidx
                        pt_info['xy0_ridx'] = rpt_idx
                        pt_info['xy0_lpt'] = lpt[::-1]
                        pt_info['xy0_rpt'] = rpt[::-1]
                else:
                    print(f'xy0 not matched!')

                ridx = point_idx(right_centers, xy1, swap_xy=True)
                assert right_idx.max() <= len(right_centers)
                midx1 = np.where(right_idx == ridx)[0]
                if len(midx1) > 0:
                    assert len(midx1) == 1
                    midx1 = midx1[0]
                    assert right_idx[midx1] == ridx
                    rpt = right_centers[right_idx[midx1]]
                    assert tuple(rpt)[::-1] == xy1
                    # likewise, this is not necessarily equal to xy0, by the
                    # time downstream functions screw up propagating the matches
                    lpt_idx = left_idx[midx1]
                    lpt = left_centers[lpt_idx]
                    mcost1 = match_costs[midx1]
                    print(f'xy1 matched ({ridx}:{rpt} <- {lpt_idx}:{lpt}) '
                        f'at cost {mcost1:.3f}'
                    )
                    if to_unmatch[midx1]:
                        print('xy1 will be unmatched for cost >= max_cost!')
                    else:
                        r_idxs.append((name, ridx))
                        # For use debugging downstream of this function.
                        pt_info['xy1_lidx'] = lpt_idx
                        pt_info['xy1_ridx'] = ridx
                        pt_info['xy1_lpt'] = lpt[::-1]
                        pt_info['xy1_rpt'] = rpt[::-1]
                else:
                    print(f'xy1 not matched!')
                print('')

        k_unmatched_left.update(left_idx[to_unmatch])
        k_unmatched_right.update(right_idx[to_unmatch])
        left_idx = left_idx[~ to_unmatch]
        right_idx = right_idx[~ to_unmatch]

        n_unassigned = abs(len(left_centers) - len(right_centers))

        total_cost += max_cost * n_unassigned
        # TODO better way to normalize error?
        total_cost = total_cost / min(len(left_centers), len(right_centers))

        # TODO maybe compute costs for all unmatched w/ pdist, and check
        # nothing is < max_cost

        unmatched_left.append(np.array(list(k_unmatched_left)))
        unmatched_right.append(np.array(list(k_unmatched_right)))
        cost_totals.append(total_cost)
        lr_matches.append(np.stack([left_idx, right_idx], axis=-1))

        # These just need to be consistent w/ numbers printed before colons
        # above (and they are).
        if debug_points and ci in debug_points:
            lrm = lr_matches[-1]
            for name, li in l_idxs:
                midx = np.argwhere(lrm[:, 0] == li)[0]
                assert len(midx) == 1
                midx = midx[0]
                print(f'name: {name}, xy0 match row {midx}:', lrm[midx, :])
            for name, ri in r_idxs:
                midx = np.argwhere(lrm[:, 1] == ri)[0]
                assert len(midx) == 1
                midx = midx[0]
                print(f'name: {name}, xy1 match row {midx}:', lrm[midx, :])
            print('')

    if show:
        ax.legend()
        ax.set_title(title)

        if write_plots:
            # TODO and delete this extra hack
            if len(sequence_of_centers) > 2:
                extra = '_acrossblocks'
            else:
                extra = ''
            fname = util.to_filename(title + extra) + plot_format
            #
            print(f'writing to {fname}')
            fig.savefig(fname)
            #

    # TODO TODO change all parts that require squeeze=True to squeeze=False?
    if squeeze and len(sequence_of_centers) == 2:
        lr_matches = lr_matches[0]
        unmatched_left = unmatched_left[0]
        unmatched_right = unmatched_right[0]
        cost_totals = cost_totals[0]

    # TODO maybe stop returning unmatched_* . not sure it's useful.

    return lr_matches, unmatched_left, unmatched_right, cost_totals, fig


def stable_rois(lr_matches, verbose=False):
    """
    Takes a list of n_cells x 2 matrices, with each row taking an integer ROI
    label from one set of labels to the other.

    Input is as first output of correspond_rois.

    Returns:
    stable_cells: a n_stable_cells x (len(lr_matches) + 1) matrix, where rows
        represent different labels for the same real cells. Columns have the
        set of stable cells IDs, labelled as the inputs are.

    new_lost: a (len(lr_matches) - 1) length list of IDs lost when matching
        lr_matches[i] to lr_matches[i + 1]. only considers IDs that had
        been stable across all previous pairs of matchings.
    """
    # TODO TODO also test in cases where lr_matches is greater than len 2
    # (at least len 3)

    # TODO TODO also test when lr_matches is len 1, to support that case
    if len(lr_matches) == 1 or type(lr_matches) is not list:
        raise NotImplementedError

    orig_matches = lr_matches
    # Just since it gets written to in the loop.
    lr_matches = [m.copy() for m in lr_matches]

    stable = lr_matches[0][:,0]
    UNLABELLED = -1
    new_lost = []
    for i in range(len(lr_matches) - 1):
        matches1 = lr_matches[i]
        matches2 = lr_matches[i + 1]

        # These two columns should have the ROI / center numbers
        # represent the same real ROI / point coordinates.
        stable_1to2, m1_idx, m2_idx = np.intersect1d(
            matches1[:,1], matches2[:,0], return_indices=True)

        assert np.array_equal(matches1[m1_idx, 1], matches2[m2_idx, 0])

        curr_stable_prior_labels = matches1[m1_idx, 0]

        matches2[m2_idx, 0] = curr_stable_prior_labels

        # To avoid confusion / errors related too using old, now meaningless
        # labels.
        not_in_m2_idx = np.setdiff1d(np.arange(len(matches2)), m2_idx)
        assert (lr_matches[i + 1] == UNLABELLED).sum() == 0
        matches2[not_in_m2_idx] = UNLABELLED 
        assert (lr_matches[i + 1] == UNLABELLED).sum() == 2 * len(not_in_m2_idx)

        ids_lost_at_i = np.setdiff1d(stable, curr_stable_prior_labels)
        stable = np.setdiff1d(stable, ids_lost_at_i)
        new_lost.append(ids_lost_at_i)

        n_lost_at_i = len(ids_lost_at_i)
        if verbose and n_lost_at_i > 0:
            print(f'Lost {n_lost_at_i} ROI(s) between blocks {i} and {i + 1}')

    # TODO make a test case where the total number of *matched* rois is
    # conserved at each time step, but the matching makes the length of the
    # ultimate stable set reduce
    n_matched = [len(m) - ((m == UNLABELLED).sum() / 2) for m in lr_matches]
    assert len(stable) <= min(n_matched)

    stable_cells = []
    for i, matches in enumerate(lr_matches):
        # Because each of these columns will have been edited in the loop
        # above, to have labels matching the first set of center labels.
        _, _, stable_indices_i = np.intersect1d(stable, matches[:,0],
            return_indices=True)

        assert not UNLABELLED in matches[stable_indices_i, 0]
        orig_labels_stable_i = orig_matches[i][stable_indices_i, 0]
        stable_cells.append(orig_labels_stable_i)

    # This last column in the last element in the last of matches
    # was the only column that did NOT get painted over with the new labels.
    stable_cells.append(matches[stable_indices_i, 1])
    stable_cells = np.stack(stable_cells, axis=1)

    # might be redundant...
    stable_cells = stable_cells[np.argsort(stable_cells[:,0]), :]
    assert np.array_equal(stable_cells[:,0], stable)
    return stable_cells, new_lost


# TODO try to numba this
def renumber_rois2(matches_list, centers_list):
    id2frame_bounds = dict()
    id2indices = dict()
    next_id = 0
    seen_at_i = dict()
    for i in range(len(matches_list) + 1):
        if i not in seen_at_i:
            seen_at_i[i] = set()

        m = matches_list[min(i, len(matches_list) - 1)]
        for left, right in m:
            if i < len(matches_list):
                if left in seen_at_i[i]:
                    continue
                seen_at_i[i].add(left)
                roi_indices_across_frames = [left]
            else:
                if right in seen_at_i[i]:
                    continue
                roi_indices_across_frames = []

            first_frame = i
            # So that the frame counter increments as soon as we have one
            # "right" element (every match row must correspond to at least
            # two timepoints).
            j = i + 1
            while j <= len(matches_list):
                roi_indices_across_frames.append(right)
                last_frame = j

                if j in seen_at_i:
                    seen_at_i[j].add(right)
                else:
                    seen_at_i[j] = set()

                if j == len(matches_list):
                    break

                next_matches = matches_list[j]
                next_row_idx = np.argwhere(next_matches[:, 0] == right)
                if len(next_row_idx) == 0:
                    break

                next_row_idx = next_row_idx[0][0]
                left, right = next_matches[next_row_idx]
                j += 1

            assert (last_frame - first_frame + 1 ==
                len(roi_indices_across_frames)
            )
            id2frame_bounds[next_id] = (first_frame, last_frame)
            id2indices[next_id] = roi_indices_across_frames
            next_id += 1

        if i < len(matches_list):
            unmatched = np.setdiff1d(np.arange(len(centers_list[i])), m[:,0])
        else:
            unmatched = np.setdiff1d(np.arange(len(centers_list[i])), m[:,1])

        for u in unmatched:
            # TODO never need to check whether this is in seen, do i?
            id2frame_bounds[next_id] = (i, i)
            id2indices[next_id] = [u]
            next_id += 1

    assert set(id2frame_bounds.keys()) == set(id2indices.keys())
    centers_array = np.empty((len(centers_list), next_id,
        centers_list[0].shape[1])) * np.nan

    for roi_id in id2frame_bounds.keys():
        start, end = id2frame_bounds[roi_id]
        indices = id2indices[roi_id]
        centers_array[start:end+1, roi_id, :] = \
            [c[i] for c, i in zip(centers_list[start:end+1], indices)]

    # TODO assert min / max non-nan cover full frame for reasonable test data

    return centers_array


# TODO TODO should either this fn or correspond_rois try to handle the case
# where a cell drifts out of plane and then back into plane???
# possible? some kind of filtering?
def renumber_rois(matches_list, centers_list, debug_points=None, max_cost=None):
    """
    Each sequence of matched ROIs gets an increasing integer identifier
    (including length-1 sequences, i.e. unmatched stuff).

    Returns lists of IDs in each element of input list and centers,
    re-indexed with new IDs.
    """
    # TODO use this function inside stable_rois / delete that function
    # altogether (?)

    if type(matches_list) is not list or type(centers_list) is not list:
        raise ValueError('both input arguments must be lists')

    assert len(centers_list) == len(matches_list) + 1

    # Since they get written to in the loop.
    matches_list = [m.copy() for m in matches_list]
    centers_list = [c.copy() for c in centers_list]

    # TODO test case where input is not == np.arange(input.max())
    # (both just missing some less and w/ ids beyond len(centers) - 1)
    ids_list = []
    first_ids = np.arange(len(centers_list[0]))
    assert len(np.setdiff1d(matches_list[0][:,0], first_ids)) == 0
    ids_list.append(first_ids)
    next_new_id = first_ids.max() + 1
    print('next_new_id (after making first_ids):', next_new_id)
    ##next_new_id = matches_list[0][:,0].max() + 1

    #if len(centers_list[0]) > len(matches_list[0]):
    #    import ipdb; ipdb.set_trace()

    # TODO delete / put behind something like a `checks` flag
    assert max_cost is not None
    id2last_xy = {i: c for i, c in zip(first_ids, centers_list[0][:,:2])}
    id2src_history = {i:
        ['first_match' if i in matches_list[0][:,0] else 'new_first']
        for i in first_ids
    }
    id2idx_history = dict()
    for i in first_ids:
        try:
            idx = matches_list[0][i,0]
        except IndexError:
            idx = None
        id2idx_history[i] = [idx]
    assert set(id2src_history.keys()) == set(id2idx_history.keys())
    nonshared_m2_idx_list = []
    #

    for i in range(len(matches_list)):
        # These centers are referred to by the IDs in matches_list[i + 1][:, 1],
        # and (if it exists) matches_list[i + 2][:, 1]
        centers = centers_list[i + 1]
        matches1 = matches_list[i]

        '''
        # This includes stuff shared and stuff lost by m2.
        # The only thing this should not include is stuff that should get
        # a new ID in m2.
        centers_in_m1 = matches1[:, 1]

        # These include both things in matches2 (those not shared with matches1)
        # and things we need to generate new IDs for.
        only_new_centers_idx = np.setdiff1d(
            np.arange(len(centers)),
            centers_in_m1
        )
        # This should be of the same length as centers and should index each
        # value, just in a different order.
        new_center_idx = np.concatenate((
            centers_in_m1,
            only_new_centers_idx
        ))
        assert np.array_equal(
            np.arange(len(centers)),
            np.unique(new_center_idx)
        )

        # We are re-ordering the centers, so that they are in the same order
        # as the IDs (both propagated and new) at this timestep (curr_ids).
        centers_list[i + 1] = centers[new_center_idx]
        '''

        # TODO TODO TODO i think this is the heart of the problem
        # (b/c all problem indices were in the new_ids that got cut off
        # when trying to fit into smaller space of nonshared_m2_idx
        existing_ids = matches1[:, 0]
        #next_new_id = existing_ids.max() + 1
        ###n_new_ids = len(only_new_centers_idx)
        ##assert len(centers) - len(matches1) == n_new_ids
        n_new_ids = len(centers) - len(matches1)
        # Not + 1 because arange does not include the endpoint.
        stop = next_new_id + n_new_ids
        new_ids = np.arange(next_new_id, stop)
        for i, idl in enumerate(ids_list[::-1]):
            print(- (i + 1))
            overlap = set(new_ids) & set(idl)
            if len(overlap) > 0:
                print('overlap:', overlap)
                import ipdb; ipdb.set_trace()
        #
        print('i:', i)
        print('n_new_ids:', n_new_ids)
        print('stop:', stop)
        print('next_new_id:', next_new_id)
        print('next_new_id - existing_ids.max():',
            next_new_id - existing_ids.max()
        )
        next_new_id = stop

        curr_ids = np.concatenate((existing_ids, new_ids))
        assert len(curr_ids) == len(centers)
        assert len(curr_ids) == len(np.unique(curr_ids))

        # TODO this is the necessary condition for having current centers not
        # get mis-ordered, right?
        #assert np.array_equal(np.argsort(curr_ids), np.argsort(new_center_idx))
        #
        #import ipdb; ipdb.set_trace()

        #'''
        for j, (_id, cxy) in enumerate(zip(curr_ids, centers_list[i+1][:,:2])):
            if _id in id2last_xy:
                last_xy = id2last_xy[_id]
                dist = util.euclidean_dist(cxy, last_xy)
                try:
                    assert dist < max_cost
                except AssertionError:
                    print('')
                    #print(max_cost)
                    #print(dist)
                    print('id:', _id)
                    #print(last_xy)
                    #print(cxy)
                    if _id in new_ids:
                        fr = 'new'
                    elif _id in existing_ids:
                        fr = 'old'
                    else:
                        assert False
                    print(fr)

                    print(id2src_history[_id])
                    prev_idx = id2idx_history[_id]
                    print(prev_idx)
                    if len(prev_idx) > 0:
                        prev_idx = prev_idx[-1]
                        if prev_idx is not None:
                            # (previous entry in ids_list)
                            assert (np.argwhere(ids_list[i] == _id)[0][0] ==
                                prev_idx
                            )

                    import ipdb; ipdb.set_trace()

            id2last_xy[_id] = cxy
            # TODO delete these after debugging
            assert (_id in id2src_history) == (_id in id2idx_history)
            src_hist = 'new' if _id in new_ids else 'old'
            if _id in id2src_history:
                id2src_history[_id].append(src_hist)
                id2idx_history[_id].append(j)
            else:
                id2src_history[_id] = [src_hist]
                id2idx_history[_id] = [j]
            #
        #'''

        ids_list.append(curr_ids)

        # TODO TODO TODO some assertion that re-ordered centers are still
        # fully equiv to old centers, when indexing as they get indexed below?
        # though ordering across centers is what really matters...

        # TODO `i` as well?
        '''
        if debug_points and i + 1 in debug_points:
            print(f'I + 1 = {i + 1}')
            for pt_info in debug_points[i + 1]:
                roi_id = int(pt_info['name'])
                xy0 = pt_info['xy0']
                xy1 = pt_info['xy1']
                print('roi_id:', roi_id)
                print('xy0:', xy0)

                # TODO turn into assertion
                # shouldn't happen?
                if roi_id not in curr_ids:
                    print('not in curr_ids')
                    import ipdb; ipdb.set_trace()
                #

                if roi_id in matches1[:,0]:
                    print('in matches1[:,0] (old IDs)')
                elif roi_id in new_ids:
                    print('in new_ids!')
                else:
                    assert False, 'neither in old nor new ids'

                id_idx = np.argmax(curr_ids == roi_id)
                cxy = tuple(centers_list[i + 1][id_idx][:2])
                assert cxy == xy0
                lidx = pt_info.get('xy0_lidx')
                if lidx is not None:
                    xy0_was_matched = True
                    lpt = pt_info.get('xy0_lpt')
                    # so we can still index in to the non-re-ordered centers
                    assert tuple(centers[lidx, :2]) == xy0
                    print('xy0_lidx:', lidx)
                else:
                    xy0_was_matched = False

                #if xy0_was_matched:
                #    assert
                #import ipdb; ipdb.set_trace()

            #import ipdb; ipdb.set_trace()
        '''

        if i + 1 < len(matches_list):
            matches2 = matches_list[i + 1]
            assert len(matches2) <= len(centers)

            # These two columns should have the ROI / center numbers
            # represent the same real ROI / point coordinates.
            _, shared_m1_idx, shared_m2_idx = np.intersect1d(
                matches1[:,1], matches2[:,0], return_indices=True
            )
            assert np.array_equal(
                matches1[shared_m1_idx, 1],
                matches2[shared_m2_idx, 0]
            )
            prior_ids_of_shared = matches1[shared_m1_idx, 0]
            print(len(np.unique(matches2[:,0])) == len(matches2[:,0]))
            matches2[shared_m2_idx, 0] = prior_ids_of_shared
            print(len(np.unique(matches2[:,0])) == len(matches2[:,0]))

            nonshared_m2_idx = np.setdiff1d(np.arange(len(matches2)),
                shared_m2_idx
            )
            # ROIs unmatched in matches2 get any remaining higher IDs in new_ids
            # It is possible for there to be new_ids without any
            # nonshared_m2_idx.
            # TODO TODO TODO will we ever need to map from these new_ids that
            # run off the end to specific centers later?
            print('new_ids:', new_ids)
            print('new_ids[:len(nonshared_m2_idx)]:',
                new_ids[:len(nonshared_m2_idx)]
            )
            print('nonshared_m2_idx:', nonshared_m2_idx)
            print('matches2[nonshared_m2_idx, 0]:',
                matches2[nonshared_m2_idx, 0]
            )
            import ipdb; ipdb.set_trace()
            matches2[nonshared_m2_idx, 0] = new_ids[:len(nonshared_m2_idx)]
            assert len(np.unique(matches2[:,0])) == len(matches2[:,0])

    for i, (ids, cs) in enumerate(zip(ids_list, centers_list)):
        assert len(ids) == len(cs), f'(i={i}) {len(ids)} != {len(cs)}'

    centers_array = np.empty((len(centers_list), next_new_id,
        centers_list[0].shape[1])) * np.nan

    for i, (ids, centers) in enumerate(zip(ids_list, centers_list)):
        centers_array[i, ids, :] = centers

        if debug_points:
            if i in debug_points:
                for pt_info in debug_points[i]:
                    roi_id = int(pt_info['name'])
                    xy0 = pt_info['xy0']
                    cidx = point_idx(centers_array[i], xy0)
                    assert cidx == roi_id

    return centers_array


def roi_jumps(roi_xyd, max_cost):
    """
    Returns dict of first_frame -> list of (x, y, str(point idx)) for each
    time an ROI jumps by >= max_cost on consecutive frames.

    correspond_rois should have not matched these points.

    Output suitable for debug_points kwarg to correspond_rois
    """
    diffs = np.diff(roi_xyd[:, :, :2], axis=0)
    dists = np.sqrt((np.diff(roi_xyd[:, :, :2], axis=0) ** 2).sum(axis=2))
    # to avoid NaN comparison warning on >= (dists must be positive anyway)
    dists[np.isnan(dists)] = -1
    jumps = np.argwhere(dists >= max_cost)
    dists[dists == -1] = np.nan

    first_frames = set(jumps[:,0])
    debug_points = dict()
    for ff in first_frames:
        ff_rois = jumps[jumps[:,0] == ff, 1]
        # switching frame and roi axes so iteration is over rois
        # (zippable w/ ff_rois below)
        xys = np.swapaxes(np.round(roi_xyd[ff:ff+2, ff_rois, :2]
            ).astype(np.uint16), 0, 1
        )
        ff_info = []
        for roi, roi_xys in zip(ff_rois, xys):
            xy0, xy1 = roi_xys
            pt_info = {'name': str(roi), 'xy0': tuple(xy0), 'xy1': tuple(xy1)}
            ff_info.append(pt_info)
        debug_points[ff] = ff_info

    return debug_points


# TODO TODO use in unit tests of roi tracking w/ some real data as input
def check_no_roi_jumps(roi_xyd, max_cost):
    assert len(roi_jumps(roi_xyd, max_cost)) == 0


# TODO TODO TODO re-enable checks!!!
def correspond_and_renumber_rois(roi_xyd_sequence, debug=False, checks=False,
    use_renumber_rois2=True, **kwargs):

    max_cost = kwargs.get('max_cost')
    if max_cost is None:
        # TODO maybe switch to max / check current approach yields results
        # just as reasonable as those w/ larger max_cost
        min_diam = min([xyd[:, 2].min() for xyd in roi_xyd_sequence])
        # + 1 b/c cost == max_cost is thrown out
        max_cost = min_diam / 2 + 1
        kwargs['max_cost'] = max_cost

    # TODO fix what seems to be making correspond_rois fail in case where
    # diameter info is also passed in (so it can be used here and elsewhere
    # w/o having to toss that data first)
    roi_xy_seq = [xyd[:, :2] for xyd in roi_xyd_sequence]

    lr_matches, _, _, _, _ = correspond_rois(roi_xy_seq, squeeze=False,
    #    verbose=debug, show=debug, write_plots=False, **kwargs
        verbose=debug, show=False, write_plots=False, **kwargs
    )
    '''
    if debug:
        # For stuff plotted in correspond_rois
        plt.show()
    '''

    debug_points = kwargs.get('debug_points')
    if use_renumber_rois2:
        new_roi_xyd = renumber_rois2(lr_matches, roi_xyd_sequence)
    else:
        new_roi_xyd = renumber_rois(lr_matches, roi_xyd_sequence,
            debug_points=debug_points, max_cost=max_cost
        )
    if checks:
        check_no_roi_jumps(new_roi_xyd, max_cost)

    return new_roi_xyd


# TODO add nonoverlap constraint? somehow make closer to real data?
# TODO use this to test gui/fitting/tracking
def make_test_centers(initial_n=20, nt=100, frame_shape=(256, 256), sigma=3,
    exlusion_radius=None, p=0.05, max_n=None, round_=False, diam_px=20,
    add_diameters=True, verbose=False):
    # TODO maybe adapt p so it's the p over the course of the
    # nt steps, and derivce single timestep p from that?

    if exlusion_radius is not None:
        raise NotImplementedError

    # So that we can pre-allocate the center coordinates over time
    # (rather than having to figure out how many were added by the end,
    # and then pad all the preceding arrays of centers w/ NaN)
    if p:
        max_n = 2 * initial_n
    else:
        # Don't need to allocate extra space if the number of ROIs is
        # deterministic.
        max_n = initial_n

    assert len(frame_shape) == 2
    assert frame_shape[0] == frame_shape[1]
    d = frame_shape[0]
    max_coord = d - 1

    # Also using this for new centers gained while iterating.
    initial_centers = np.random.randint(d, size=(max_n, 2))

    # TODO more idiomatic numpy way to generate cumulative noise?
    # (if so, just repeat initial_centers to generate centers, and add the
    # two) (maybe not, with my constraints...)
    # TODO TODor generate inside the loop (only as many as non-NaN, and only
    # apply to non NaN)
    xy_steps = np.random.randn(nt - 1, max_n, 2) * sigma

    next_trajectory_idx = initial_n
    centers = np.empty((nt, max_n, 2)) * np.nan
    centers[0, :initial_n] = initial_centers[:initial_n]
    # TODO should i be generating the noise differently, so that the x and y
    # components are not independent (so that if deviation is high in one,
    # it's more likely to be lower in other coordinate, to more directly
    # constrain the distance? maybe it's just a scaling thing though...)
    for t in range(1, nt):
        # TODO maybe handle this differently...
        if p and next_trajectory_idx == max_n:
            raise RuntimeError(f'reached max_n ({max_n}) on step {t} '
                f'(before {nt} requested steps'
            )
            #break

        centers[t] = centers[t - 1] + xy_steps[t - 1]

        # TODO make sure NaN stuff handled correctly here
        # The centers should stay within the imaginary frame bounds.
        centers[t][centers[t] > max_coord] = max_coord
        centers[t][centers[t] < 0] = 0

        if not p:
            continue

        lose = np.random.binomial(1, p, size=max_n).astype(np.bool)
        if verbose:
            nonnan = ~ np.isnan(centers[t,:,0])
            print('# non-nan:', nonnan.sum())
            n_lost = (nonnan & lose).sum()
            if n_lost > 0:
                print(f't={t}, losing {n_lost}')
        centers[t][lose] = np.nan

        # TODO TODO note: if not allowed to fill NaN that come from losing
        # stuff, then max_n might more often limit # unique rather than #
        # concurrent tracks... (and that would prob make a format more close to
        # what i was already implementing in association code...)
        # maybe this all means i could benefit from a different
        # representation...
        # one more like id -> (start frame, end frame, coordinates)

        # Currently, giving any new trajectories different indices (IDs)
        # from any previous trajectories, by putting them in ranges that
        # had so far only had NaN. As association code may be, this also
        # groups new ones in the next-unused-integer-indices, rather
        # than giving each remaining index a chance.
        # To justify first arg (n), imagine case where initial_n=0 and
        # max_n=1.
        n_to_gain = np.random.binomial(max_n - initial_n, p)
        if n_to_gain > 0:
            if verbose:
                print(f't={t}, gaining {n_to_gain}')

            first_ic_idx = next_trajectory_idx - initial_n
            centers[t][next_trajectory_idx:next_trajectory_idx + n_to_gain] = \
                initial_centers[first_ic_idx:first_ic_idx + n_to_gain]
            next_trajectory_idx += n_to_gain

    assert len(centers) == nt

    # This seems to convert NaN to zero...
    if round_:
        centers = np.round(centers).astype(np.uint16)

    if add_diameters:
        roi_diams = np.expand_dims(np.ones(centers.shape[:2]) * diam_px, -1)
        centers = np.concatenate((centers, roi_diams), axis=-1)

    # TODO check output is in same kind of format as output of my matching fns

    return centers
