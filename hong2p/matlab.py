"""
Functions for dealing with MATLAB analysis outputs, either via
`scipy.io.loadmat` or the MATLAB engine.
"""

import os
import sys
import atexit
import signal
import warnings

import numpy as np

# Only currently using `is_array_sorted` from here, but trying to import as:
# `from hong2p.util import is_array_sorted` led to a ImportError with a message
# indicating it was likely because of a circular import.
import hong2p.util


# TODO delete after refactoring to not require this engine.
# flag so i can revert to old matlab-engine behavior while i'm still
# implementing support via non-matlab-engine means
USE_MATLAB_ENGINE = False
#

# Holding on to old excepthook, rather than using always-available default at
# sys.__excepthook__, so global modifications to sys.excepthook in Python
# startup (like with a PYTHONSTARTUP script) are maintained.
old_excepthook = sys.excepthook
def matlab_exit_except_hook(exctype, value, traceback):
    if exctype == TypeError:
        args = value.args
        # This message is what MATLAB says in this case.
        if (len(args) == 1 and
            args[0] == 'exit expected at most 1 arguments, got 2'):
            return

    old_excepthook(exctype, value, traceback)

    # TODO delete this after confirming holding onto old excepthook works
    #sys.__excepthook__(exctype, value, traceback)


# TODO maybe rename to init_matlab and return nothing, to be more clear that
# other fns here are using it behind the scenes?
evil = None
def matlab_engine(force=False):
    """
    Gets an instance of MATLAB engine w/ correct paths for Remy's single plane
    code (my version in `ejhonglab/matlab_kc_plane`).

    Args:
    force (bool): If True, will load engine even if `USE_MATLAB_ENGINE=False`.
    
    Tries to undo Ctrl-C capturing that MATLAB seems to do, by modifying
    `sys.excepthook`.
    """
    global evil

    if USE_MATLAB_ENGINE and not force:
        warnings.warn('not loading MATLAB engine because USE_MATLAB_ENGINE=False')
        return None

    import matlab.engine

    evil = matlab.engine.start_matlab()
    # TODO TODO this doesn't seem to kill parallel workers... it should
    # (happened in case where there was a memory error. visible in top output.)
    # TODO work inside a fn?
    atexit.register(evil.quit)

    exclude_from_matlab_path = {
        'CaImAn-MATLAB',
        'CaImAn-MATLAB_hong',
        'matlab_helper_functions'
    }
    userpath = evil.userpath()
    for root, dirs, _ in os.walk(userpath, topdown=True):
        dirs[:] = [d for d in dirs if (not d.startswith('.') and
            not d.startswith('@') and not d.startswith('+') and
            d not in exclude_from_matlab_path and 
            d != 'private')]

        evil.addpath(root)

    # Since exiting without letting MATLAB handle it seems to yield a TypeError
    # We will register a global handler that hides that non-useful error
    # message, below.
    signal.signal(signal.SIGINT, sys.exit)
    sys.excepthook = matlab_exit_except_hook

    return evil


# TODO don't expose this if i can refactor other stuff to not use it
# otherwise use this rather than their own separate definitions
# (only currently used in kc_natural_mixes/populate_db.py)
# Subdirectory within each fly's analysis directory that contains the .mat file
# output by the matlab_kc_plane code.
rel_to_cnmf_mat = 'cnmf'

def matfile(date, fly_num, thorimage_id):
    """Returns filename of Remy's metadata [+ CNMF output] .mat file.
    """
    return join(analysis_fly_dir(date, fly_num), rel_to_cnmf_mat,
        thorimage_id + '_cnmf.mat'
    )


# TODO not currently used. maybe delete.
def tiff_matfile(tif):
    """Returns filename of Remy's metadata [+ CNMF output] .mat file.
    """
    keys = tiff_filename2keys(tif)
    return matfile(*keys)


# TODO TODO need to acquire a lock to use the matlab instance safely?
# (if i'm sure gui is enforcing only one call at a time anyway, probably
# don't need to worry about it)
def get_matfile_var(matfile, varname, require=True):
    """Returns length-one list with variable contents, or empty list.

    Raises KeyError if require is True and variable not found.
    """
    if USE_MATLAB_ENGINE:
        global evil

        if evil is None:
            matlab_engine()

        try:
            # TODO maybe clear workspace either before or after?
            # or at least clear this specific variable after?
            load_output = evil.load(matfile, varname, nargout=1)
            var = load_output[varname]
            if type(var) is dict:
                return [var]
            return var
        except KeyError:
            # TODO maybe check for var presence some other way than just
            # catching this generic error?
            if require:
                raise
            else:
                return []
    else:
        raise NotImplementedError


# TODO maybe just wrap get_matfile_var?
def load_mat_timing_info(mat_file, use_matlab_engine=None):
    """Loads and returns timing information from .mat output of Remy's script.

    Args:
    mat_file (str): filename of a .mat file with timing information.
    use_matlab_engine (bool or None): If a bool, overrides `USE_MATLAB_ENGINE`.

    Returns a dict with the following keys, each pointing to a numpy array:
    - 'frame_times'
    - 'block_first_frames'
    - 'block_last_frames'
    - 'odor_onset_frames'
    - 'odor_offset_frames'

    All `*_frames` variables use 0 to refer to the first frame, and so on from
    there.

    Raises `AssertionError` if the data seems inconsistent with itself.
    Raises `matlab.engine.MatlabExecutionError` when MATLAB engine calls do.
    """
    from scipy import stats

    if use_matlab_engine is None:
        use_matlab_engine = USE_MATLAB_ENGINE

    if use_matlab_engine:
        import matlab.engine
        # TODO this sufficient w/ global above to get access to matlab engine in
        # here? (necessary? delete?)
        #global evil

        if evil is None:
            # `force=True` is just for case when `use_matlab_engine` is
            # overriding `USE_MATLAB_ENGINE`
            matlab_engine(force=True)

        try:
            # TODO probably switch to doing it this way
            '''
            evil.clear(nargout=0)
            load_output = evil.load(mat_file, 'ti', nargout=1)
            ti = load_output['ti']
            '''
            evil.evalc("clear; data = load('{}', 'ti');".format(mat_file))

        except matlab.engine.MatlabExecutionError as e:
            raise

        ti = evil.eval('data.ti')
        # If any of the types initially loaded as smaller types end up
        # overflowing, would probably need to set appropriate dtype here
        # (could do through a lookup table of names -> dtypes), rather than
        # just casting with `.astype(...)` below
        for k in ti.keys():
            # Converting from type `mlarray.double`
            ti[k] = np.array(ti[k])
    else:
        from scipy.io import loadmat

        data = loadmat(mat_file, variable_names=['ti'])['ti']
        # TODO check this is all still necessary w/ all combinations of loadmat
        # args + latest version
        varnames = data.dtype.names
        assert data.shape == (1, 1)
        assert data[0].shape == (1,)
        vardata = data[0][0]
        assert len(varnames) == len(vardata)
        # TODO is it even guaranteed that order of names is same as this
        # order? do some sanity checks (at least) until i can get an answer on
        # this... (inspection of one example in ipdb seemed to have everything
        # in order though)
        ti = {n: d for n, d in zip(varnames, vardata)}

    # TODO check on first element? subtract first element so it starts at t=0.0?
    frame_times = ti['frame_times'].astype('float64').flatten()
    assert len(frame_times.shape) == 1 and len(frame_times) > 1
    # TODO assert that all things that refer to frame indices do not have max
    # value above number of entries in frame_times (- 1)

    # TODO just fix all existing saved to new naming convention in a script
    # like kc_natural_mixes/populate_db? (for vars that produce
    # block_<first/last>_frames)

    # Frame indices for CNMF output.
    # Of length equal to number of blocks. Each element is the frame
    # index (from 1) in CNMF output that starts the block, where
    # block is defined as a period of continuous acquisition.
    try:
        block_first_frames = ti['block_start_frame']
    except KeyError:
        # This was the old name for it.
        block_first_frames = ti['trial_start']
    try:
        block_last_frames = ti['block_end_frame']
    except KeyError:
        # This was the old name for it.
        block_last_frames = ti['trial_end']

    block_last_frames = block_last_frames.astype(np.uint32).flatten() - 1
    block_first_frames = block_first_frames.astype(np.uint32).flatten() - 1

    odor_onset_frames = ti['stim_on'].astype(np.uint32).flatten() - 1
    odor_offset_frames = ti['stim_off'].astype(np.uint32).flatten() - 1

    # TODO maybe warn if there are keys in `ti` that will not be returned? opt?

    assert len(block_first_frames) == len(block_last_frames)
    assert len(odor_onset_frames) == len(odor_offset_frames)

    assert block_first_frames[0] >= 0
    # Without equality since there should be some frames before first of each.
    assert block_last_frames[0] > 0
    assert odor_onset_frames[0] > 0
    assert odor_offset_frames[0] > 0

    assert hong2p.util.is_array_sorted(frame_times)

    for frame_index_arr in [block_first_frames, block_last_frames,
        odor_onset_frames, odor_offset_frames]:
        assert is_array_sorted(frame_index_arr)
        assert frame_index_arr.max() < len(frame_times)

    for o_start, o_end in zip(odor_onset_frames, odor_offset_frames):
        # TODO could also check this without equality if we can tell from
        # other info that the frame rate should have been high enough
        # to get two separate frames for the start and end
        # (would probably need odor pulse length to be manually entered)
        assert o_start <= o_end

    # TODO break odor_<onset/offset>_frames into blocks and check they are
    # all within the bounds specified by the corresponding elements of
    # block_<first/last>_frames?

    for i, (b_start, b_end) in enumerate(
        zip(block_first_frames, block_last_frames)):

        assert b_start < b_end
        if i != 0:
            last_b_end = block_last_frames[i - 1]
            assert last_b_end == (b_start - 1)

        block_frametimes = frame_times[b_start:b_end]
        dts = np.diff(block_frametimes)
        # np.max(np.abs(dts - np.mean(dts))) / np.mean(dts)
        # was 0.000148... in one case I tested w/ data from the older
        # system, so the check below w/ rtol=1e-4 would fail.
        mode = stats.mode(dts)[0]

        # TODO see whether triggering in 2019-05-03/6/fn_0002 case is fault of
        # data or loading code?
        # (seems to trigger in sub-recording cases, but those may no longer
        # be relevant...)
        # (above case also triggers when not loading a sub-recording, so that's
        # not a complete explanation...)

        # If this fails, method for extracting block frames as likely failed,
        # as blocks are defined as continuous periods of acquisition, so
        # adjacent frames should be close in time.
        # Had to increase rtol from 3e-4 to 3e-3 for 2020-03-09/1/fn_007, and
        # presumably other volumetric data too.
        assert np.allclose(dts, mode, rtol=3e-3), \
            'block first/last frames likely wrong'

    return {
        'frame_times': frame_times,
        'block_first_frames': block_first_frames,
        'block_last_frames': block_last_frames,
        'odor_onset_frames': odor_onset_frames,
        'odor_offset_frames': odor_offset_frames,
    }


# TODO use in unit test on a representative subset of my data so far
# (comparing matlab engine and scipy.io.loadmat ways to load the data)
def _check_timing_info_equal(ti0, ti1):
    """Raises `AssertionError` if timing info in `ti0` and `ti1` is not equal.
    
    Assumes all values are numpy array-like.
    """
    assert ti0.keys() == ti1.keys()
    for k in ti0.keys():
        assert np.array_equal(ti0[k], ti1[k])


def check_movie_timing_info(movie, frame_times, block_first_frames,
    block_last_frames):
    """Checks that `movie` and `ti` refer to the same number of frames.

    Raises `AssertionError` if the check fails.
    """
    total_block_frames = sum([e - s + 1 for s, e in
        zip(block_first_frames, block_last_frames)
    ])
    n_frames = movie.shape[0]
    # TODO may need to replace this with a warning (at least optionally), given
    # comment in gui.py that did so because failure (described in cthulhu:190520
    # bug text file)
    assert len(frame_times) == n_frames
    # TODO may need to remove this assert to handle cases where there is a
    # partial block (stopped early). leave assert after slicing tho.
    # (warn instead, probably)
    assert total_block_frames == n_frames, \
        '{} != {}'.format(total_block_frames, n_frames)

