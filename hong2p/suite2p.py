"""
Functions for working with suite2p as part of analysis.
"""

from hong2p import thor, util


def suite2p_params(thorimage_dir):
    # From: https://suite2p.readthedocs.io/en/latest/settings.html

    single_plane_fps, xy, z, c, n_flyback, _ = thor.load_thorimage_metadata(
        thorimage_dir
    )

    # "(int, default: 1) each tiff has this many planes in sequence"
    # wait, is this =z, or =z*t or just =t?
    nplanes = z

    # "(int, default: 1) each tiff has this many channels per plane"
    #nchannels = 

    # "(int, default: 1) this channel is used to extract functional ROIs (1-based, so 1
    # means first channel, and 2 means second channel)"
    #functional_chan = 

    # "(float, default: 1.0) The timescale of the sensor (in seconds), used for
    # deconvolution kernel. The kernel is fixed to have this decay and is not fit to the
    # data. We recommend: 0.7 for GCaMP6f, 1.0 for GCaMP6m, 1.25-1.5 for GCaMP6s"
    #tau = 

    # "(float, default: 10.0) Sampling rate (per plane). For instance, if you have a 10
    # plane recording acquired at 30Hz, then the sampling rate per plane is 3Hz,
    # so set ops[‘fs’] = 3 "
    fs = single_plane_fps / (z + n_flyback)

    # TODO TODO how much of a problem is it to ignore the fact that there are flyback
    # frames contributing to time before a plane is returned to? not sure i'm
    # understanding 'fs' correctly to begin with...

    # TODO TODO maybe use ops['ignore_flyback'] (== indices of each flyback frame?
    # or is there a more concise option?) to make fs more meaningful?

    # TODO TODO can we pick a better spatial_scale than suite2p does automatically?
    # does its optimal value even differ across the types of data we have in the lab?

    ops = {
        'fs': fs,
        'nplanes': nplanes,
    }
    return ops


def print_suite2p_params(thorimage_dir):
    ops = suite2p_params(thorimage_dir)
    print('nplanes:', ops['nplanes'])
    print(f'fs: {ops["fs"]:.2f}')
    print('tau: 0.7 (recommended for GCaMP6f)')


def invert_combined_view_offset(combined_ops, roi):

    # TODO TODO TODO probably assert that all pixels (after this) are within bounds of
    # [0, Lx) and [0, Ly), UNLESS i modify this function to handle stuff merged across
    # multiple panes of combined view
    # TODO TODO TODO does merging across planes in combined view mangle some of the
    # information i am currently using here (e.g. iplane)? i suppose i could determine
    # which of the tiled plane panes the various pixels for a given (merged) roi
    # occupies, and assign that way rather than using iplane? perhaps just for the stuff
    # marked as merged?
    # TODO TODO TODO actually probably just use imerge to find the stat entries for the
    # input cells and use the iplanes from them (because yes, 'iplane' is essentially
    # meaningless for the merged cells)

    def evendiv(x, y):
        div, mod = divmod(x, y)
        assert mod == 0
        return div

    orig_coords_roi = deepcopy(roi)

    orig_frame_x_shape = combined_ops['Lxc']
    orig_frame_y_shape = combined_ops['Lyc']

    nx = evendiv(combined_ops['Lx'], orig_frame_x_shape)
    ny = evendiv(combined_ops['Ly'], orig_frame_y_shape)

    cy, cx = divmod(roi['iplane'], nx)
    assert cx < nx
    assert cy < ny

    dx = cx * orig_frame_x_shape
    dy = cy * orig_frame_y_shape

    orig_coords_roi['ypix'] -= dy
    orig_coords_roi['xpix'] -= dx

    # This coordinate order is consistent w/ suite2p code in suite2p/io/save.py:combined
    orig_coords_roi['med'][0] -= dy
    orig_coords_roi['med'][1] -= dx

    return orig_coords_roi


# TODO TODO TODO handle stuff in suite2p that is merged cross planes in the combined
# view, such that each part goes on the appropriate plane and isn't mangled by
# `invert_combined_view_offset`
def suite2p_roi_stat2roi(roi_stat, ops, xy_only=False, fillna_0=True,
    invert_combined=True):
    """Takes a single suite2p ROI to a dense numpy representation

    roi_stat and ops are both assumed to come from the 'combined' folder outputs
    """
    # TODO add args for additional conversion to xarray?
    # TODO args for using sparse representations instead?
    if invert_combined:
        ny = ops['Lyc']
        nx = ops['Lxc']
        roi_stat = invert_combined_view_offset(ops, roi_stat)
    else:
        ny = ops['Ly']
        nx = ops['Lx']

    xy_roi = np.zeros((ny, nx))
    if not fillna_0:
        xy_roi *= np.nan

    xpix = roi_stat['xpix']
    ypix = roi_stat['ypix']
    # TODO TODO TODO split into appropriate grid coordinates (if more than one),
    # subtract off the appropriate offsets, and put each into the approprite z-slice
    xy_roi[ypix, xpix] = roi_stat['lam']

    if xy_only:
        return xy_roi

    roi = np.zeros((ops['nplanes'],) + xy_roi.shape)
    if not fillna_0:
        roi *= np.nan

    roi[roi_stat['iplane']] = xy_roi
    return roi


def suite2p_stat2rois(stat, ops, as_xarray=True, **kwargs):
    """Takes suite2p 'stat' array / dict (roi # -> roi stat) to array / xarray ROIs.
    """
    # TODO TODO TODO first get a list of all the merged indices, and then only load the
    # non-merged stuff (or maybe do that in the load fn, though i might need the
    # original ROIs there... i suppose i could modify 'iplane' on load for the merged
    # ROIs to be a sequence containing all the respective ones? not sure that would
    # solve all my possible needs for non-merged-ROI-data...)

    # TODO TODO implement support for array input (output of unprocessed load of suite2p
    # stat.npy)

    roi_dict = {i: suite2p_roi_stat2roi(s, ops, **kwargs) for i, s in stat.items()}

    roi_array = np.stack(list(roi_dict.values()), axis=-1)

    if as_xarray:
        roi_nums = roi_dict.keys()
        # TODO maybe overwrite default sequential 'roi_num'?
        rois = util.numpy2xarray_rois(roi_array,
            roi_indices={'s2p_roi_num': roi_nums}
        )
        return rois
    else:
        return roi_array


# TODO TODO may need to adapt depending on how i handle rois merged across planes...
def plot_roi(roi_stat, ops, ax=None):
    if ax is None:
        ax = plt.gca()

    xy_roi = suite2p_roi_stat2roi(roi_stat, ops, xy_only=True)

    # TODO in future, might be nice to convert xpix / ypix and only ever make roi_img of
    # the shape of the cropped ROI (change crop_to_nonzero and fn it calls)

    cropped, ((x_min, x_max), (y_min, y_max)) = util.crop_to_nonzero(xy_roi, margin=0)

    ax.imshow(cropped)
    ax.axis('off')

    #from scipy.ndimage import binary_closing
    #closed = binary_closing(roi_img_tmp > 0)
    #ax.contour(closed > 0, [0.5])


def load_s2p_pickle(npy_path):
    return np.load(npy_path, allow_pickle=True)


# TODO switch to using analysis_dir as input or at least by output
def get_suite2p_dir(analysis_dir):
    return join(analysis_dir, 'suite2p')


def get_suite2p_combined_dir(analysis_dir):
    return join(get_suite2p_dir(analysis_dir), 'combined')


def get_iscell_path(s2p_out_dir):
    return join(s2p_out_dir, 'iscell.npy')


def load_iscell(s2p_out_dir):
    return np.load(get_iscell_path(s2p_out_dir))


def set_suite2p_iscell_label(s2p_out_dir, roi_num, is_good):
    """
    Args:
        roi_num (int): index of ROI to change label of, according to suite2p
        s2p_out_dir (str): must directly contain a single iscell.npy file
    """
    assert is_good in (True, False)

    iscell_npy = get_iscell_path(s2p_out_dir)
    iscell = np.load(iscell_npy)

    import ipdb; ipdb.set_trace()
    dtype_before = iscell.dtype
    # TODO TODO need to index the correct column (1st of 2)
    iscell[roi_num] = float(is_good)
    assert iscell.dtype == dtype_before

    # TODO only write if diff
    # TODO TODO see how suite2p actually writes this file, in case there is anything
    # special
    import ipdb; ipdb.set_trace()
    # NOTE: if redcell.npy is used, may also need to modify that? see suite2p/io:save.py
    np.save(iscell_npy, iscell)
    import ipdb; ipdb.set_trace()


# TODO maybe wrap call that checks this (and opens suite2p if not) with something that
# also saves an empty hidden file in the directory to mark good if no modifications
# necessary? or unlikely enough? / just prompt before opening suite2p each time?
# TODO maybe just use mtime?
def is_iscell_modified(s2p_out_dir, warn=True):
    iscell_path = get_iscell_path(s2p_out_dir)
    iscell = np.load(iscell_path)

    # Defined in suite2p/suite2p/classification/classifier.py, in kwarg to `run`.
    # `run` is called in classify.py in the same directory, which doesn't pass this
    # kwarg (so it should always be this default value).
    p_threshold = 0.5

    iscell_bool = iscell[:, 0].astype(np.bool_)

    if warn and iscell_bool.all():
        # Since this is probably the result of just setting the threshold to 0 in the
        # GUI and not further marking the ROIs as cell/not-cell from there.
        warnings.warn(f'all suite2p ROIs in {s2p_out_dir} marked as good. check this '
            'is intentional.'
        )

    # The GUI (in suite2p/gui/merge.py:apply), also does not include equality.
    return not np.array_equal(iscell_bool, iscell[:, 1] > p_threshold)


def mark_all_suite2p_rois_good(s2p_out_dir):
    """Modify iscell.npy to set all labels (first column) 1.

    This is to undo the automatic classification step that I do not want and can not
    seem to disable. It should be equivalent to entering 0 for the probability threshold
    in the GUI.
    """
    assert not is_iscell_modified(s2p_out_dir, warn=False)

    iscell_path = get_iscell_path(s2p_out_dir)
    iscell = np.load(iscell_path)

    iscell[:, 0] = 1
    np.save(iscell_path, iscell)

    assert is_iscell_modified(s2p_out_dir, warn=False)


def modify_iscell_in_suite2p(stat_path):
    # TODO TODO maybe show a plot for the relevant df/f image(s) alongside this, to
    # see if i'm getting *those* glomeruli? would probably need to couple with main loop
    # more though... (or modify my suite2p fork to have optional additional views)

    # This would block until the process finishes, as opposed to Popen call below.
    #subprocess.run(f'suite2p --statfile {stat_path}'.split(), check=True)

    # TODO some way to have ctrl-c in main program also kill this opened suite2p window?

    # This will wait for suite2p to be closed before it returns.
    # NOTE: the --statfile argument is something I added in my fork of suite2p
    proc = subprocess.Popen(f'suite2p --statfile {stat_path}'.split())

    # TODO maybe refactor so closing suite2p will automatically close any still-open
    # matplotlib figures?
    plt.show()

    proc.wait()
    print('SUITE2P CLOSED', flush=True)

    # TODO warn if not modified after closing?


# TODO delete?
def suite2p_footprint2bool_mask(roi_stat, ops):
    from scipy.ndimage import binary_closing

    full_roi = np.zeros((ops['Ly'], ops['Lx'])) * np.nan
    xpix = roi_stat['xpix']
    ypix = roi_stat['ypix']

    full_roi[ypix, xpix] = roi_stat['lam']

    # np.nan > 0 is False (as is np.nan < 0), so the nan background is fine
    closed = binary_closing(full_roi > 0)

    return closed


class ROIsNotLabeledError(Exception):
    pass

class LabelsNotModifiedError(ROIsNotLabeledError):
    pass

class LabelsNotSelectiveError(ROIsNotLabeledError):
    pass


def load_s2p_outputs(plane_or_combined_dir, good_only=True, merge_inputs=False,
    merge_outputs=False, subset_stat=True):

    traces_path = join(plane_or_combined_dir, 'F.npy')
    if not exists(traces_path):
        raise IOError(f'suite2p needs to be run on this data ({traces_path} '
            'did not exist)'
        )

    # TODO TODO are traces output by suite2p already delta F / F, or just F?
    # (seems like just F, though not i'm pretty sure there were some options for using
    # some percentile as a baseline, so need to check again)

    if good_only and not is_iscell_modified(plane_or_combined_dir):
        raise LabelsNotModifiedError('suite2p ROI labels not modified')

    iscell = load_iscell(plane_or_combined_dir)
    traces = load_s2p_pickle(traces_path)

    # TODO regarding single entries in this array (each a dict):
    # - what is 'soma_crop'? it's a boolean array but what's it mean?
    # - what is 'med'? (len 2 list, but is it a centroid or what?)
    # - where are the weights for the ROI? (expecting something of same length as xpix
    #   and ypix)? it's not 'lam', is it? and if it's 'lam', should i normalized it
    #   before using? why isn't it already normalized?
    stat_path = join(plane_or_combined_dir, 'stat.npy')
    stat = load_s2p_pickle(stat_path)

    ops = load_s2p_pickle(join(plane_or_combined_dir, 'ops.npy')).item()

    good_rois = iscell[:, 0].astype(np.bool_)

    if good_only and len(good_rois) == good_rois.sum():
        # (and that probably means probability threshold was set to ~0 and no ROIs were
        # marked bad since then, though it would almost certainly be necessary)
        raise LabelsNotSelectiveError('*all* ROIs marked good')

    # Transposing because original is of shape (# ROIs, # timepoints in movie),
    # but compute_trial_stats expects first dimension to be of size # timepoints in
    # movie (so it can be used directly on movie as well).
    traces = traces.T

    traces = pd.DataFrame(data=traces)
    traces.index.name = 'frame'
    traces.columns.name = 'roi'

    merges = {
        i: list(s['imerge']) for i, s in enumerate(stat) if len(s.get('imerge', [])) > 0
    }

    if not merge_inputs:
        merge_input_indices = [x for xs in merges.values() for x in xs]
        assert len(set(merge_input_indices)) == len(merge_input_indices)

        # TODO test in case where there are no merged rois
        traces = traces.loc[:, ~ traces.columns.isin(merge_input_indices)]

        # TODO TODO maybe still leave them in stat dict / return separately tho, for
        # display of rois? or maybe the new stat entry is already appropriate for those
        # purposes anyway?

    if not merge_outputs:
        traces = traces.loc[:, [x for x in traces.columns if x not in merges]]

    if good_only:
        good_roi_nums = [x for x in np.where(good_rois)[0] if x in traces.columns]

        # TODO assuming they maintain their pre-merge iscell labels and thus some are
        # marked 'good', test in case where some of merge inputs have already been
        # dropped above
        traces = traces.loc[:, good_roi_nums]

    # Converting type so it can be sensibly indexed after subsetting
    if subset_stat:
        stat_dict = {r: stat[r] for r in traces.columns}
    else:
        stat_dict = {r: stat[r] for r in range(len(stat))}

    # TODO TODO TODO maybe return a 4th value that is a dict: <merge output roi #> ->
    # <set of merge input roi #>, for use with manually recomputing "merged" ROIs
    # (including just picking the best plane)
    return traces, stat_dict, ops, merges


def load_s2p_combined_outputs(analysis_dir, **kwargs):
    """
    Raises IOError if corresponding suite2p output not found (i.e. suite2p needs to be
    run), LabelsNotModifiedError, or LabelsNotSelectiveError.
    """
    combined_dir = get_suite2p_combined_dir(analysis_dir)
    return load_s2p_outputs(combined_dir, **kwargs)

