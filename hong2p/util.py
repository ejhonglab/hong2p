"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
from os import listdir
from os.path import join, split, exists, sep, isdir, normpath, getmtime
import socket
import pickle
import atexit
import signal
import sys
import xml.etree.ElementTree as etree
from types import ModuleType
from datetime import datetime
import warnings
from pprint import pprint
import glob
import re
import hashlib
import time
# TODO delete if custom Unpickler doesn't work
import io

import numpy as np
from numpy.ma import MaskedArray
import pandas as pd
# TODO delete if custom Unpickler doesn't work
from pandas.compat import pickle_compat

import matplotlib as mpl
try:
    # TODO TODO TODO will i want to explicitly check sys.modules to see whether
    # any code has imported pyplot, or will matplotlib fail / warn appropriately
    # if we try to `mpl.use(...)` after pyplot has already been imported.
    # what i'm trying to avoid is it just silently failing, such that the
    # backend is not actually changed

    # see https://stackoverflow.com/questions/30483246 if need to check
    # sys.modules ourselves

    # TODO maybe only hardcode it if current default backend happens to be
    # "non-gui" as in this error:
    # TODO some mpl fn to check if it is a "gui" backend?
    # UserWarning: Matplotlib is currently using agg, which is a non-GUI
    # backend, so cannot show the figure
    # TODO re: above, does mpl.get_backend() interfere w/ future .use calls?
    mpl.use('Qt5Agg')
except ImportError:
    print('All possible (not necessarily installed) matplotlib backends:')
    pprint(mpl.rcsetup.all_backends)

# Having all matplotlib-related imports come after `hong2p.util` import,
# so that I can let `hong2p.util` set the backend, which it seems must be set
# before the first import of `matplotlib.pyplot`
import matplotlib.patches as patches
# is just importing this potentially going to interfere w/ gui?
# put import behind paths that use it?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Note: many imports were pushed down into the beginnings of the functions that
# use them, to reduce the number of hard dependencies.


# TODO delete after refactoring to not require this engine.
# flag so i can revert to old matlab-engine behavior while i'm still
# implementing support via non-matlab-engine means
NO_MATLAB_ENGINE = True
#

np.set_printoptions(precision=2)

recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
trial_only_cols = [
    'comparison',
    'name1',
    'name2',
    'repeat_num'
]
trial_cols = recording_cols + trial_only_cols

date_fmt_str = '%Y-%m-%d'
dff_latex = r'$\frac{\Delta F}{F}$'

db_hostname = os.getenv('HONG_POSTGRES_HOST', 'atlas')

# TODO TODO probably just move all stuff that uses db conn into it's own module
# under this package, and then just get the global conn upon that module import
conn = None
def get_db_conn():
    global conn
    global meta
    if conn is not None:
        return conn
    else:
        from sqlalchemy import create_engine, MetaData

        our_hostname = socket.gethostname()
        if our_hostname == db_hostname:
            url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
        else:
            url = ('postgresql+psycopg2://tracedb:tracedb@{}' +
                ':5432/tracedb').format(db_hostname)

        conn = create_engine(url)

        # TODO this necessary? was it just for to_sql_with_duplicates or
        # something? why else?
        meta = MetaData()
        meta.reflect(bind=conn)

        return conn


# was too much trouble upgrading my python 3.6 caiman conda env to 3.7
'''
# This is a Python >=3.7 feature only.
def __getattr__(name):
    if name == 'conn':
        return get_db_conn()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
'''

data_root_name = 'mb_team'

# Module level cache.
_data_root = None
def data_root():
    global _data_root
    if _data_root is None:
        # TODO separate env var for local one? or have that be the default?
        data_root_key = 'HONG_2P_DATA'
        nas_prefix_key = 'HONG_NAS'
        fallback_data_root_key = 'DATA_DIR'

        prefix = None
        if data_root_key in os.environ:
            data_root = os.environ[data_root_key]
            source = data_root_key

        elif nas_prefix_key in os.environ:
            prefix = os.environ[nas_prefix_key]
            source = nas_prefix_key

        elif fallback_data_root_key in os.environ:
            data_root = os.environ[fallback_data_root_key]
            source = fallback_data_root_key

        else:
            prefix = '/mnt/nas'
            warnings.warn('None of environment variables specifying data path '
                f'found!\nUsing default data path of : {prefix}\n'
                f'Set one of the following: {data_root_key}, {nas_prefix_key}, '
                f'{fallback_data_root_key}'
            )
            source = None

        if prefix is not None:
            data_root = join(prefix, data_root_name)
        _data_root = data_root

        if not isdir(_data_root):
            emsg = (f'data root expected at {_data_root}, but no directory '
                'exists there!'
            )
            if source is not None:
                emsg += f'\nDirectory chosen from environment variable {source}'
            raise IOError(emsg)

    # TODO err if nothing in data_root, saying which env var to set and how
    return _data_root


# TODO (for both below) support a local and a remote one ([optional] local copy
# for faster repeat analysis)?
# TODO use env var like kc_analysis currently does for prefix after refactoring
# (include mb_team in that part and rename from data_root?)
def raw_data_root():
    return join(data_root(), 'raw_data')


def analysis_output_root():
    return join(data_root(), 'analysis_output')


def stimfile_root():
    return join(data_root(), 'stimulus_data_files')


def format_date(date):
    """
    Takes a pandas Timestamp or something that can be used to construct one
    and returns a str with the formatted date.

    Used to name directories by date, etc.
    """
    return pd.Timestamp(date).strftime(date_fmt_str)


def _fly_dir(date, fly):
    if not type(date) is str:
        date = format_date(date)

    if not type(fly) is str:
        fly = str(int(fly))

    return join(date, fly)


def raw_fly_dir(date, fly):
    return join(raw_data_root(), _fly_dir(date, fly))


def thorimage_dir(date, fly, thorimage_id):
    return join(raw_fly_dir(date, fly), thorimage_id)


def thorsync_dir(date, fly, base_thorsync_dir):
    return join(raw_fly_dir(date, fly), base_thorsync_dir)


def analysis_fly_dir(date, fly):
    return join(analysis_output_root(), _fly_dir(date, fly))


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
    force (bool): If True, will load engine even if `NO_MATLAB_ENGINE=True`.
    
    Tries to undo Ctrl-C capturing that MATLAB seems to do, by modifying
    `sys.excepthook`.
    """
    global evil

    if NO_MATLAB_ENGINE and not force:
        warnings.warn('not loading MATLAB engine because NO_MATLAB_ENGINE set')
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


# TODO TODO need to acquire a lock to use the matlab instance safely?
# (if i'm sure gui is enforcing only one call at a time anyway, probably
# don't need to worry about it)
def get_matfile_var(matfile, varname, require=True):
    """Returns length-one list with variable contents, or empty list.

    Raises KeyError if require is True and variable not found.
    """
    if not NO_MATLAB_ENGINE:
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


# TODO unit test
def is_array_sorted(array):
    """Returns whether 1-dimensional np.ndarray is sorted."""
    # could implement an `axis` kwarg if i wanted to support multidimensional
    # arrays
    assert len(array.shape) == 1, 'only 1-dimensional arrays supported'
    # https://stackoverflow.com/questions/47004506
    return np.all(array[:-1] <= array[1:])


# TODO maybe just wrap get_matfile_var?
def load_mat_timing_info(mat_file, use_matlab_engine=None):
    """Loads and returns timing information from .mat output of Remy's script.

    Args:
    mat_file (str): filename of a .mat file with timing information.
    use_matlab_engine (bool or None): If a bool, overrides `NO_MATLAB_ENGINE`.

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
        use_matlab_engine = not NO_MATLAB_ENGINE

    if use_matlab_engine:
        import matlab.engine
        # TODO this sufficient w/ global above to get access to matlab engine in
        # here? (necessary? delete?)
        #global evil

        if evil is None:
            # `force=True` is just for case when `use_matlab_engine` is
            # overriding `NO_MATLAB_ENGINE`
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
    # like populate_db? (for vars that produce block_<first/last>_frames)

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

    assert is_array_sorted(frame_times)

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


def print_block_frames(block_first_frames, block_last_frames):
    """Prints block numbers and the corresponding start / stop frames.

    For subsetting TIFF in ImageJ / other manual analysis.

    Prints frame numbers 1-indexed, but takes 0-indexed frames.
    """
    print('Block frames:')
    for i, (b_first, b_last) in enumerate(zip(block_first_frames,
        block_last_frames)):
        # Adding one to index frames as in ImageJ.
        print('{}: {} - {}'.format(i, b_first + 1, b_last + 1))
    print('')


def md5(fname):
    """Calculates MD5 hash on file with name `fname`.
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_timestamp(timestamp):
    """Returns human-readable str for timestamp accepted by `pd.Timestamp`.
    """
    return str(pd.Timestamp(timestamp))[:16]


# TODO TODO can to_sql with pg_upsert replace this? what extra features did this
# provide?
def to_sql_with_duplicates(new_df, table_name, index=False, verbose=False):
    from sqlalchemy import MetaData, Table

    # TODO TODO document what index means / delete

    # TODO TODO if this fails and time won't be saved on reinsertion, any rows
    # that have been inserted already should be deleted to avoid confusion
    # (mainly, for the case where the program is interrupted while this is
    # running)
    # TODO TODO maybe have some cleaning step that checks everything in database
    # has the correct number of rows? and maybe prompts to delete?

    global conn
    if conn is None:
        conn = get_db_conn()

    # Other columns should be generated by database anyway.
    cols = list(new_df.columns)
    if index:
        cols += list(new_df.index.names)
    table_cols = ', '.join(cols)

    md = MetaData()
    table = Table(table_name, md, autoload_with=conn)
    dtypes = {c.name: c.type for c in table.c}

    if verbose:
        print('SQL column types:')
        pprint(dtypes)
   
    df_types = new_df.dtypes.to_dict()
    if index:
        df_types.update({n: new_df.index.get_level_values(n).dtype
            for n in new_df.index.names})

    if verbose:
        print('\nOld dataframe column types:')
        pprint(df_types)

    sqlalchemy2pd_type = {
        'INTEGER()': np.dtype('int32'),
        'SMALLINT()': np.dtype('int16'),
        'REAL()': np.dtype('float32'),
        'DOUBLE_PRECISION(precision=53)': np.dtype('float64'),
        'DATE()': np.dtype('<M8[ns]')
    }
    if verbose:
        print('\nSQL types to cast:')
        pprint(sqlalchemy2pd_type)

    new_df_types = {n: sqlalchemy2pd_type[repr(t)] for n, t in dtypes.items()
        if repr(t) in sqlalchemy2pd_type}

    if verbose:
        print('\nNew dataframe column types:')
        pprint(new_df_types)

    # TODO how to get around converting things to int if they have NaN.
    # possible to not convert?
    new_column_types = dict()
    new_index_types = dict()
    for k, t in new_df_types.items():
        if k in new_df.columns and not new_df[k].isnull().any():
            new_column_types[k] = t

        # TODO or is it always true that index level can't be NaN anyway?
        elif (k in new_df.index.names and
            not new_df.index.get_level_values(k).isnull().any()):

            new_index_types[k] = t

        # TODO print types being skipped b/c nan?

    new_df = new_df.astype(new_column_types, copy=False)
    if index:
        # TODO need to handle case where conversion dict is empty
        # (seems to fail?)
        #pprint(new_index_types)

        # MultiIndex astype method seems to not work the same way?
        new_df.index = pd.MultiIndex.from_frame(
            new_df.index.to_frame().astype(new_index_types, copy=False))

    # TODO print the type of any sql types not convertible?
    # TODO assert all dtypes can be converted w/ this dict?

    if index:
        print('writing to temporary table temp_{}...'.format(table_name))

    # TODO figure out how to profile
    new_df.to_sql('temp_' + table_name, conn, if_exists='replace', index=index,
        dtype=dtypes)

    # TODO change to just get column names?
    query = '''
    SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS data_type
    FROM   pg_index i
    JOIN   pg_attribute a ON a.attrelid = i.indrelid
        AND a.attnum = ANY(i.indkey)
    WHERE  i.indrelid = '{}'::regclass
    AND    i.indisprimary;
    '''.format(table_name)
    result = conn.execute(query)
    pk_cols = ', '.join([n for n, _ in result])

    # TODO TODO TODO modify so on conflict the new row replaces the old one!
    # (for updates to analysis, if exact code version w/ uncommited changes and
    # everything is not going to be part of primary key...)
    # (want updates to non-PK rows)

    # TODO TODO should i just delete rows w/ our combination(s) of pk_cols?
    # (rather than other upsert strategies)
    # TODO (flag to) check deletion was successful
    # TODO factor deletion into another fn (?) and expose separately in gui


    # TODO prefix w/ ANALYZE EXAMINE and look at results
    query = ('INSERT INTO {0} ({1}) SELECT {1} FROM temp_{0} ' +
        'ON CONFLICT ({2}) DO NOTHING').format(table_name, table_cols, pk_cols)
    # TODO maybe a merge is better for this kind of upsert, in postgres?
    if index:
        # TODO need to stdout flush or something?
        print('inserting into {} from temporary table... '.format(table_name),
            end='')

    # TODO let this happen async in the background? (don't need result)
    conn.execute(query)

    # TODO flag to read back and check insertion stored correct data?

    if index:
        print('done')

    # TODO drop staging table


def pg_upsert(table, conn, keys, data_iter):
    from sqlalchemy.dialects import postgresql
    # https://github.com/pandas-dev/pandas/issues/14553
    for row in data_iter:
        row_dict = dict(zip(keys, row))
        sqlalchemy_table = meta.tables[table.name]
        stmt = postgresql.insert(sqlalchemy_table).values(**row_dict)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=table.index,
            set_=row_dict)
        conn.execute(upsert_stmt)


def odorset_name(df_or_odornames):
    """Returns name for set of odors in DataFrame.

    Looks at odors in original_name1 column. Name used to lookup desired
    plotting order for the odors in the set.
    """
    try:
        if 'original_name1' in df_or_odornames.columns:
            unique_odornames = df_or_odornames.original_name1.unique()
            abbreviated = False
        else:
            assert 'name1' in df_or_odornames.columns, \
                'need either original_name1 or name1 in df columns'
            # Assuming abbreviated names now.
            unique_odornames = df_or_odornames.name1.unique()
            # maybe don't assume abbreviated just b/c name1?
            # (particularly if supporting abbrev/not in iterable input
            # case, could also check name1 contents)
            abbreviated = True

    except AttributeError:
        unique_odornames = set(df_or_odornames)
        # TODO maybe also support abbreviated names in this case?
        abbreviated = False
        
    odor_set = None
    # TODO TODO derive these diagnostic odors from odor_set2order? would that
    # still be redundant w/ something else i hardcoded (if so, further
    # de-dupe)?
    # TODO at least lookup abbreviations from full names?
    if not abbreviated:
        if 'ethyl butyrate' in unique_odornames:
            odor_set = 'kiwi'
        elif 'acetoin' in unique_odornames:
            odor_set = 'flyfood'
        elif '1-octen-3-ol' in unique_odornames:
            odor_set = 'control'
    else:
        if 'eb' in unique_odornames:
            odor_set = 'kiwi'
        elif 'atoin' in unique_odornames:
            odor_set = 'flyfood'
        elif '1o3ol' in unique_odornames:
            odor_set = 'control'

    if odor_set is None:
        raise ValueError('none of diagnostic odors in odor column')

    # TODO probably just find single odor that satisfies is_mix and derive from
    # that, for more generality (would only work in original name case)
    return odor_set


def load_stimfile(stimfile_path):
    """Loads odor metadata stored in a pickle.

    These metadata files are generated by scripts under
    `ejhonglab/cutpast_arduino_stimuli`.
    """
    # TODO better check for path already containing stimfile_root?
    # string prefix check might be too fragile...
    if not stimfile_path.startswith(stimfile_root()):
        stimfile_path = join(stimfile_root(), stimfile_path)

    # TODO also err if not readable / valid
    if not exists(stimfile_path):
        stimfile_just_fname = split(stimfile_path)[1]
        raise ValueError('copy missing stimfile {} to {}'.format(
            stimfile_just_fname, stimfile_root
        ))

    with open(stimfile_path, 'rb') as f:
        data = pickle.load(f)
    return data


# TODO delete (subsuming contents into load_experiment) if i'd never want to
# call this in other circumstances
def load_odor_metadata(stimfile_path):
    """Returns odor metadata loaded from pickle and additional computed values.

    Additional values are added into the dictionary loaded from the pickle.
    In some cases, this can overwrite the loaded values.
    """
    data = load_stimfile(stimfile_path)
    # TODO infer from data if no stimfile and not specified in
    # metadata (is there actually any value in this? maybe if we have
    # a sufficient amount of other data about the odor order in metadata
    # yaml (though this is not supported now...)?)

    # TODO delete this hack (which is currently just using new pickle
    # format as a proxy for the experiment being a supermixture experiment)
    if 'odor_lists' not in data:
        pair_case = True

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors odor-pair experiments.
        presentations_per_repeat = 3
        odor_list = data['odor_pair_list']

        # could delete eventually. b/c originally i was casting to int,
        # though it's likely it was always int anyway...
        assert type(data['n_repeats']) is int
    else:
        pair_case = False

        n_expected_real_blocks = 3
        odor_list = data['odor_lists']
        # because of "block" def in arduino / get_stiminfo code
        # not matching def in randomizer / stimfile code
        # (scopePin pulses vs. randomization units, depending on settings)
        presentations_per_repeat = len(odor_list) // n_expected_real_blocks
        assert len(odor_list) % n_expected_real_blocks == 0

        # Hardcode to break up into more blocks, to align defs of blocks.
        # TODO (maybe just for experiments on 2019-07-25 ?) or change block
        # handling in here? make more flexible?

        # Will overwrite existing value.
        assert 'n_repeats' in data.keys()
        data['n_repeats'] = 1

        # TODO check that overwriting of presentations_per_block with newest
        # data in this case is still accurate (post fixing some of the values
        # in pickle data) (also check similar values)

    presentations_per_block = data['n_repeats'] * presentations_per_repeat

    # Overwriting exisiting value.
    assert 'presentations_per_block' in data.keys()
    data['presentations_per_block'] = presentations_per_block

    # NOT overwriting an existing value.
    assert 'pair_case' not in data.keys()
    data['pair_case'] = pair_case

    # NOT overwriting existing value.
    assert 'presentations_per_repeat' not in data.keys()
    data['presentations_per_repeat'] = presentations_per_repeat

    # NOT overwriting an existing value.
    assert 'odor_list' not in data.keys()
    data['odor_list'] = odor_list

    return data


def print_trial_odors(data, odor_onset_frames=None):
    """
    data should be as the output of `load_odor_metadata`
    """
    import chemutils as cu

    n_repeats = data['n_repeats']
    odor_list = data['odor_list']
    presentations_per_block = data['presentations_per_block']
    pair_case = data['pair_case']

    n_blocks, remainder = divmod(len(odor_list), presentations_per_block)
    assert remainder == 0

    # TODO add extra input data / add extra values to `data` in fn that already
    # augments that as necessary, for all values used below to still be defined
    if pair_case:
        print(('{} comparisons ({{A, B, A+B}} in random order x ' +
            '{} repeats)').format(n_blocks, n_repeats)
        )
    else:
        mix_names = {x for x in odor_list if '@' not in x}
        if len(mix_names) == 1:
            mix_name = mix_names.pop()
            print('{} randomized blocks of "{}" and its components'.format(
                n_blocks, mix_name
            ))
        else:
            assert len(mix_names) == 0
            print('No mixtures, so presumably this is a calibration experiment')
            print(f'{n_blocks} randomized blocks')

    # TODO maybe print this in tabular form?
    trial = 0
    for i in range(n_blocks):
        p_start = presentations_per_block * i
        p_end = presentations_per_block * (i + 1)
        cline = '{}: '.format(i)

        odor_strings = []
        for o in odor_list[p_start:p_end]:
            # TODO maybe always have odor_list hold str repr?
            # or unify str repr generation -> don't handle use odor_lists
            # for str representation in supermixture case?
            # would also be a good time to unify name + *concentration*
            # handling
            if pair_case:
                # TODO odor2abbrev here too probably... be more uniform
                if o[1] == 'paraffin':
                    odor_string = o[0]
                else:
                    odor_string = ' + '.join(o)
            else:
                assert type(o) is str

                parts = o.split('@')
                odor_name = parts[0].strip()
                abbrev = None
                try:
                    abbrev = cu.odor2abbrev(odor_name)
                # For a chemutils conversion failure.
                except ValueError:
                    pass

                if abbrev is None:
                    abbrev = odor_name

                odor_string = abbrev
                # TODO also don't append stuff if conc is @ 0.0 (log)
                if len(parts) > 1:
                    assert len(parts) == 2
                    odor_string += ' @' + parts[1]

            # Adding one to index frames as in ImageJ.
            if odor_onset_frames is not None:
                odor_string += ' ({})'.format(odor_onset_frames[trial] + 1)

            trial += 1
            odor_strings.append(odor_string)

        print(cline + ', '.join(odor_strings))


def assign_frames_to_trials(movie, presentations_per_block, block_first_frames,
    odor_onset_frames):
    """Returns arrays trial_start_frames, trial_stop_frames
    """
    n_frames = movie.shape[0]
    # TODO maybe just add metadata['drop_first_n_frames'] to this?
    # (otherwise, that variable screws things up, right?)
    #onset_frame_offset = \
    #    odor_onset_frames[0] - block_first_frames[0]

    # TODO delete this hack, after implementing more robust frame-to-trial
    # assignment described below
    b2o_offsets = sorted([o - b for b, o in zip(block_first_frames,
        odor_onset_frames[::presentations_per_block])
    ])
    assert len(b2o_offsets) >= 3
    # TODO TODO TODO re-enable after fixing frame_times based issues w/
    # volumetric data
    # TODO might need to allow for some error here...? frame or two?
    # (in resonant scanner case, w/ frame averaging maybe)
    #assert b2o_offsets[-1] == b2o_offsets[-2]
    onset_frame_offset = b2o_offsets[-1]
    #
    
    # TODO TODO TODO instead of this frame # strategy for assigning frames
    # to trials, maybe do this:
    # 1) find ~max # frames from block start to onset, as above
    # TODO but maybe still warn if some offset deviates from max by more
    # than a frame or two...
    # 2) paint all frames before odor onsets up to this max # frames / time
    #    (if frames have a time discontinuity between them indicating
    #     acquisition did not proceed continuously between them, do not
    #     paint across that boundary)
    # 3) paint still-unassigned frames following odor onset in the same
    #    fashion (again stopping at boundaries of > certain dt)
    # [4)] if not using max in #1 (but something like rounded mean)
    #      may still have unassigned frames at block starts. assign those to
    #      trials.
    # TODO could just assert everything within block regions i'm painting
    # does not have time discontinuities, and then i could just deal w/
    # frames

    trial_start_frames = np.append(0,
        odor_onset_frames[1:] - onset_frame_offset
    )
    trial_stop_frames = np.append(
        odor_onset_frames[1:] - onset_frame_offset - 1, n_frames - 1
    )

    # TODO same checks are made for blocks, so factor out?
    total_trial_frames = 0
    for i, (t_start, t_end) in enumerate(
        zip(trial_start_frames, trial_stop_frames)):

        if i != 0:
            last_t_end = trial_stop_frames[i - 1]
            assert last_t_end == (t_start - 1)

        total_trial_frames += t_end - t_start + 1

    assert total_trial_frames == n_frames, \
        '{} != {}'.format(total_trial_frames, n_frames)
    #

    # TODO warn if all block/trial lens are not the same? (by more than some
    # threshold probably)

    return trial_start_frames, trial_stop_frames


def tiff_ijroi_filename(tiff, confirm=None,
    gui_confirm=False, gui_fallback=False, gui='qt5'):
    """
    Takes a tiff path to corresponding ImageJ ROI file.

    Automatic search for ROI will only check same folder as the TIFF passed in.

    Options for fallback to manual selection / confirmation of appropriate
    ROI file.
    """
    if gui_confirm and confirm:
        raise ValueError('only specify either gui_confirm or confirm')

    if gui_confirm or gui_fallback:
        if gui == 'qt5':
            from PyQt5.QtWidgets import QMessageBox, QFileDialog

        # TODO maybe implement some version w/ a builtin python gui like
        # tkinter?
        else:
            raise NotImplementedError(f'gui {gui} not supported. see function.')
    else:
        if confirm is None:
            confirm = True

    curr_tiff_dir = split(tiff)[0]
    # TODO check that *.zip glob still matches in case where it is the empty
    # string (fix if it doesn't)
    thorimage_id = tiff_thorimage_id(tiff)
    possible_ijroi_files = glob.glob(join(curr_tiff_dir,
        thorimage_id + '*.zip'
    ))

    ijroiset_filename = None
    # TODO fix automatic first choice in _NNN naming convention case
    # (seemed to not work on 2019-07-25/2/_008)
    # but actually it did work in */_007 case... so idk what's happening
    if len(possible_ijroi_files) == 1:
        ijroiset_filename = possible_ijroi_files[0]

        if confirm:
            # TODO factor into a fn to always get a Yy/Nn answer?
            prompt = (format_keys(*tiff_filename2keys(tiff) +
                f': use ImageJ ROIs in {ijroiset_filename}? [y/n] '
            ))
            response = None
            while response not in ('y', 'n'):
                response = input(prompt).lower()

            if response == 'n':
                ijroiset_filename = None
                # TODO manual text entry of appropriate filename in this case?
                # ...or maybe just totally unsupport terminal interaction?

        elif gui_confirm:
            confirmation_choice = QMessageBox.question(self, 'Confirm ROI file',
                f'Use ImageJ ROIs in {ijroiset_filename}?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if confirmation_choice != QMessageBox.Yes:
                ijroiset_filename = None

    elif not gui_fallback and len(possible_ijroi_files) > 1:
        raise IOError('too many candidate ImageJ ROI files')

    elif not gui_fallback and len(possible_ijroi_files) == 0:
        raise IOError('no candidate ImageJ ROI files')

    if gui_fallback and ijroiset_filename is None:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO restrict *.zip files shown to those also following some
        # naming convention (to indicate it's for the currently loaded TIFF,
        # and not a TIFF from some other experiment on the same fly)?
        # maybe checkbox / diff option to show all?

        # TODO need to pass in parent widget (what `self` was) or leave null if
        # that works? / define in here?
        raise NotImplementedError('see comment above')
        ijroiset_filename, _ = QFileDialog.getOpenFileName(self,
            'Select ImageJ ROI zip...', curr_tiff_dir,
            'ImageJ ROIs (*.zip)', options=options
        )
        # TODO (opt?) to not allow no selection? (so downstream code can assume
        # it's defined...) (should also probably change confirmation behavior
        # above)
        if len(ijroiset_filename) == 0:
            print('No ImageJ ROI zipfile selected.')
            ijroiset_filename = None

    return ijroiset_filename


# TODO TODO TODO may want to change how this fn operates (so it operates on
# blocks rather than all of them concatenated + to provide different baselining
# options)
def calculate_df_over_f(raw_f, trial_start_frames, odor_onset_frames,
    trial_stop_frames):
    # TODO TODO maybe factor this into some kind of util fn that applies
    # another fn (perhaps inplace, perhaps onto new array) to each
    # (cell, block) (or maybe just each block, if smooth can be vectorized,
    # so it could also apply in frame-shape-preserved case?)
    '''
    for b_start, b_end in zip(block_first_frames, block_last_frames):

        for c in range(n_footprints):
            # TODO TODO TODO TODO need to be (b_end + 1) since not
            # inclusive? (<-fixed) other problems like this elsewhere?????
            # TODO maybe smooth less now that df/f is being calculated more
            # sensibly...
            raw_f[b_start:(b_end + 1), c] = smooth(
                raw_f[b_start:(b_end + 1), c], window_len=11
            )
    '''
    df_over_f = np.empty_like(raw_f) * np.nan
    for t_start, odor_onset, t_end in zip(trial_start_frames, odor_onset_frames,
        trial_stop_frames):

        # TODO TODO maybe use diff way of calculating baseline
        # (include stuff at end of response? just some percentile over a big
        # window or something?)
        # TODO kwargs to control method of calculating baseline

        # TODO maybe display baseline period on plots for debugging?
        # maybe frame numbers got shifted?
        baselines = np.mean(raw_f[t_start:(odor_onset + 1), :], axis=0)
        trial_f = raw_f[t_start:(t_end + 1), :]
        df_over_f[t_start:(t_end + 1), :] = (trial_f - baselines) / baselines

    # TODO some check that no value in df_over_f are currently NaN?

    return df_over_f


def load_recording(tiff, allow_gsheet_to_restrict_blocks=True,
    allow_missing_odor_presentations=False, verbose=True):
    # TODO summarize the various errors this could possibly raise
    # probably at least valueerror, assertionerror, ioerror (?), & tifffile
    # memory error
    # TODO TODO try to add a flag to force/allow this fn to operate
    # independently of the postgres database
    """
    May raise errors if some part of the loading fails.
    """
    import tifffile
    import ijroi
    from scipy.sparse import coo_matrix

    import chemutils as cu

    # TODO test that all of this works w/ both raw & motion corrected tiffs
    # (named and placed according to convention)
    keys = tiff_filename2keys(tiff)
    recording_title = format_keys(*keys)
    if verbose:
        print(recording_title)

    start = time.time()

    mat = matfile(*keys)
    # For some of the older data, need to either modify scipy loadmat call or
    # revert to use_matlab_engine=True call.
    ti = load_mat_timing_info(mat)
    frame_times = ti['frame_times']
    block_first_frames = ti['block_first_frames']
    block_last_frames = ti['block_last_frames']
    odor_onset_frames = ti['odor_onset_frames']
    odor_offset_frames = ti['odor_offset_frames']
    del ti
    # Copying so we can subset while still checking the original values
    # against the freshly loaded movie.
    orig_frame_times = frame_times.copy()
    orig_block_first_frames = block_first_frames.copy()
    orig_block_last_frames = block_last_frames.copy()

    # TODO TODO also store (the contents of this) in db?
    # This will return defaults if the YAML file is not found.
    meta = metadata(*keys)
    drop_first_n_frames = meta['drop_first_n_frames']
    # TODO TODO err if this is past first odor onset (or probably even too
    # close)
    del meta

    df = mb_team_gsheet()
    recordings = df.loc[
        (df.date == keys.date) &
        (df.fly_num == keys.fly_num) &
        (df.thorimage_dir == keys.thorimage_id)
    ]
    del df
    recording = recordings.iloc[0]
    del recordings
    if recording.project != 'natural_odors':
        raise ValueError('project type {} not supported'.format(
            recording.project
        ))

    stimfile = recording['stimulus_data_file']
    first_block = recording['first_block']
    last_block = recording['last_block']
    sync_dir = thorsync_dir(*keys[:2], recording['thorsync_dir'])
    image_dir = thorimage_dir(*keys[:2], recording['thorimage_dir'])
    del recording

    data = load_odor_metadata(stimfile)
    odor_list = data['odor_list']
    n_repeats = data['n_repeats']
    presentations_per_repeat = data['presentations_per_repeat']
    presentations_per_block = data['presentations_per_block']
    pair_case = data['pair_case']

    # TODO TODO maybe `allow_gsheet_to_restrict` block should also influence
    # whether the values in `recording` are used? (unless i want to just
    # unsupport the False case...)
    if pd.isnull(first_block):
        first_block = 0
    else:
        first_block = int(first_block) - 1

    if pd.isnull(last_block):
        n_full_panel_blocks = \
            int(len(odor_list) / presentations_per_block)
        last_block = n_full_panel_blocks - 1
    else:
        last_block = int(last_block) - 1

    xml = get_thorimage_xmlroot(image_dir)
    started_at = get_thorimage_time_xml(xml)

    # TODO upload full_frame_avg_trace like in populate_db?
    recording_df = pd.DataFrame({
        'started_at': [started_at],
        'thorsync_path': [sync_dir],
        'thorimage_path': [image_dir],
        'stimulus_data_path': [stimfile],
        'first_block': [first_block],
        'last_block': [last_block],
        'n_repeats': [n_repeats],
        'presentations_per_repeat': [presentations_per_repeat]
        # TODO but do i want the trace of the full movie or slice?
        # if full movie, probably should just calculate once, then
        # slice that for the trace i use here
        #'full_frame_avg_trace': 
    })
    # TODO at least put behind ACTUALLY_UPLOAD?
    # TODO maybe defer this to accepting?
    # TODO TODO also replace other cases that might need to be
    # updated w/ pg_upsert based solution
    # TODO just completely delete this fn at this point?
    ##to_sql_with_duplicates(recording_df, 'recordings')
    recording_df.set_index('started_at').to_sql('recordings', conn,
        if_exists='append', method=pg_upsert
    )
    db_recording = pd.read_sql_query('SELECT * FROM recordings WHERE ' +
        "started_at = '{}'".format(pd.Timestamp(started_at)), conn
    )
    db_recording = db_recording[recording_df.columns]
    assert recording_df.equals(db_recording)
    del db_recording

    # TODO maybe use subset here too, to be consistent w/ which mixtures get
    # entered... (according to the blocks that are ultimately used, right?
    # something else?)
    if pair_case:
        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [0 if x == 'paraffin' else
                natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })
    else:
        # TODO fix db to represent arbitrary mixtures more generally,
        # so this hack isn't necessary
        # TODO TODO fix entries already in db w/ trailing / leading
        # whitespace (split_odor_w_conc didn't used to strip returned name)
        # TODO TODO TODO fix what generates broken data['odors'] in at least
        # some of the fly food cases (missing fly food b)
        '''
        odors = pd.DataFrame([split_odor_w_conc(x) for x in
            (data['odors'] + ['no_second_odor'])
        ])
        '''
        odors = pd.DataFrame([split_odor_w_conc(x) for x in
            (list(set(data['odor_lists'])) + ['no_second_odor'])
        ])

    to_sql_with_duplicates(odors, 'odors')

    # TODO make unique id before insertion? some way that wouldn't require
    # the IDs, but would create similar tables?

    first_presentation = first_block * presentations_per_block
    last_presentation = (last_block + 1) * presentations_per_block - 1

    odor_list = odor_list[first_presentation:(last_presentation + 1)]
    assert (len(odor_list) % (presentations_per_repeat * n_repeats) == 0)

    # TODO set gui segmentation widget self.db_odors w/ this value at return /
    # recompute out there
    db_odors = pd.read_sql('odors', conn)
    db_odors.set_index(['name', 'log10_conc_vv'],
        verify_integrity=True, inplace=True
    )

    # TODO invert to check
    # TODO is this sql table worth anything if both keys actually need to be
    # referenced later anyway? (?)

    # TODO TODO TODO modify to work w/ any output of cutpaste generator
    # (arbitrary len lists in odor_lists) (see cutpaste code for similar
    # problem?)
    # TODO only add as many as there were blocks from thorsync timing info?
    if pair_case:
        # TODO this rounding to 5 decimal places always work?
        o2c = odors.set_index('name', verify_integrity=True
            ).log10_conc_vv.round(decimals=5)

        odor1_ids = [db_odors.at[(o1, o2c[o1]), 'odor_id']
            for o1, _ in odor_list
        ]
        odor2_ids = [db_odors.at[(o2, o2c[o2]), 'odor_id']
            for _, o2 in odor_list
        ]
        del o2c
    else:
        odor1_ids = [db_odors.at[tuple(split_odor_w_conc(o)),
            'odor_id'] for o in odor_list
        ]

        # TODO fix db to represent arbitrary mixtures more generally,
        # so this hack isn't necessary
        no_second_odor_id = db_odors.at[
            ('no_second_odor', 0.0), 'odor_id'
        ]
        odor2_ids = [no_second_odor_id] * len(odor1_ids)

    # TODO make unique first. only need order for filling in the
    # values in responses. (?)
    # TODO wait, how is this associated w/ anything else in this run?
    # is this table even used?
    mixtures = pd.DataFrame({
        'odor1': odor1_ids,
        'odor2': odor2_ids
    })
    # TODO maybe defer this to accepting...
    to_sql_with_duplicates(mixtures, 'mixtures')

    # TODO rename to indicate it's for each presentation / stimulus / trial?
    odor_ids = list(zip(odor1_ids, odor2_ids))

    n_blocks_from_gsheet = last_block - first_block + 1
    assert len(odor_list) == n_blocks_from_gsheet * presentations_per_block

    n_blocks_from_thorsync = len(block_first_frames)
    err_msg = ('{} blocks ({} to {}, inclusive) in Google sheet {{}} {} ' +
        'blocks from ThorSync.').format(n_blocks_from_gsheet,
        first_block + 1, last_block + 1, n_blocks_from_thorsync
    )
    fail_msg = (' Fix in Google sheet, turn off '
        'cache if necessary, and rerun.'
    )

    if n_blocks_from_gsheet > n_blocks_from_thorsync:
        raise ValueError(err_msg.format('>') + fail_msg)

    elif n_blocks_from_gsheet < n_blocks_from_thorsync:
        if allow_gsheet_to_restrict_blocks:
            warnings.warn(err_msg.format('<') + (' This is ONLY ok if you '+
                'intend to exclude the LAST {} blocks in the Thor output.'
                ).format(n_blocks_from_thorsync - n_blocks_from_gsheet)
            )
        else:
            raise ValueError(err_msg.format('<') + fail_msg)

    if allow_gsheet_to_restrict_blocks:
        # TODO unit test for case where first_block != 0 and == 0
        # w/ last_block == first_block and > first_block
        # TODO TODO doesn't this only support dropping blocks at end?
        # do i assert that first_block is 0 then? probably should...
        # TODO TODO TODO shouldnt it be first_block:last_block+1?
        block_first_frames = block_first_frames[
            :(last_block - first_block + 1)
        ]
        block_last_frames = block_last_frames[
            :(last_block - first_block + 1)
        ]
        assert len(block_first_frames) == n_blocks_from_gsheet
        assert len(block_last_frames) == n_blocks_from_gsheet

        odor_onset_frames = odor_onset_frames[
            :(last_presentation - first_presentation + 1)
        ]
        odor_offset_frames = odor_offset_frames[
            :(last_presentation - first_presentation + 1)
        ]
        del first_presentation, last_presentation
        frame_times = frame_times[:(block_last_frames[-1] + 1)]

    # TODO TODO TODO need to adjust odor_onset_frames to exclude last
    # presentations missing at end of each block, if not same len as odor
    # list
    n_missing_presentations = len(odor_list) - len(odor_onset_frames)
    assert n_missing_presentations >= 0

    if allow_missing_odor_presentations and n_missing_presentations > 0:
        # MATLAB code also assumes equal number missing in each block.
        assert n_missing_presentations % n_blocks_from_gsheet == 0
        n_missing_per_block = \
            n_missing_presentations // n_blocks_from_gsheet

        warnings.warn('{} missing presentations per block!'.format(
            n_missing_per_block
        ))
        n_deleted = 0
        for i in range(n_blocks_from_gsheet):
            end_plus_one = presentations_per_block * (i + 1) - n_deleted
            del_start = end_plus_one - n_missing_per_block
            del_stop = end_plus_one - 1

            to_delete = odor_list[del_start:(del_stop + 1)]
            warnings.warn('presentations {} to {} ({}) were missing'.format(
                del_start + 1 + n_deleted, del_stop + 1 + n_deleted,
                to_delete
            ))
            n_deleted += len(to_delete)
            del odor_list[del_start:(del_stop + 1)]
            del odor_ids[del_start:(del_stop + 1)]

        presentations_per_block -= n_missing_per_block

    # TODO move these three checks to a fn that checks timing info against
    # stimfile
    n_presentations = n_blocks_from_gsheet * presentations_per_block
    assert (len(odor_onset_frames) ==
        (n_presentations - n_missing_presentations)
    )
    assert (len(odor_offset_frames) ==
        (n_presentations - n_missing_presentations)
    )
    del n_presentations, n_missing_presentations

    assert len(odor_onset_frames) == len(odor_list)

    assert len(odor_ids) == len(odor_list)

    if verbose:
        print_trial_odors(data, odor_onset_frames)
        print('')

    end = time.time()
    print('Loading metadata took {:.3f} seconds'.format(end - start))

    # TODO TODO any way to only del existing movie if required to have
    # enough memory to load the new one (referring to how this code worked when
    # it was still a part of gui.py)?
    print('Loading TIFF {}...'.format(tiff), end='', flush=True)
    start = time.time()
    # TODO maybe just load a range of movie (if not all blocks/frames used)?
    # TODO is cnmf expecting float to be in range [0,1], like skimage?
    movie = tifffile.imread(tiff).astype('float32')
    end = time.time()
    print(' done')
    print('Loading TIFF took {:.3f} seconds'.format(end - start))

    # TODO TODO TODO fix what is causing more elements in frame_times than i
    # expect (in matlab/matlab_kc_plane/get_stiminfo.m) and delete this hack 
    n_flyback_frames = get_thorimage_n_flyback_xml(xml)
    del xml
    if n_flyback_frames > 0:
        assert len(movie.shape) == 4
        z_total = movie.shape[1] + n_flyback_frames

        # this should be effectively taking the min within each stride
        frame_times = frame_times[::z_total]
        
        # these are the bigger opportunity for error
        frame_times = frame_times[:movie.shape[0]]
        # assuming frame_times was not modified earlier. if keeping this hack,
        # would want to at least move it before earlier possible modifications,
        # and then delete this line.
        orig_frame_times = frame_times.copy()

        step = int(len(frame_times) / 3)
        block_first_frames = np.arange(len(frame_times) - step + 1, step=step)
        block_last_frames = np.arange(step - 1, len(frame_times), step=step)

        orig_block_first_frames = block_first_frames.copy()
        orig_block_last_frames = block_last_frames.copy()

        odor_onset_frames = np.round(odor_onset_frames / z_total
            ).astype(np.uint16)
        odor_offset_frames = np.round(odor_offset_frames / z_total
            ).astype(np.uint16)

    # TODO may need to remove this assert to handle cases where there is a
    # partial block (stopped early). still check after slicing tho.
    # (warn instead, probably) (add a flag to just warn?)
    check_movie_timing_info(movie, orig_frame_times,
        orig_block_first_frames, orig_block_last_frames
    )
    del orig_frame_times, orig_block_first_frames, orig_block_last_frames

    # TODO probably delete after i come up w/ a better way to handle splitting
    # movies and analyzing subsets of them.  this is just to get the frame #s to
    # subset tiff in imagej
    # Printing before `drop_first_n_frames` is subtracted, otherwise frame
    # numbers would not be correct.
    # TODO shouldn't i move this before movie loading if possible, as some of
    # the other prints? (flag to loading fn to do this?)
    if verbose:
        print_block_frames(block_first_frames, block_last_frames)

    last_frame = block_last_frames[-1]
    # TODO TODO should they really not be considered part of the last block
    # in this case...?
    n_tossed_frames = movie.shape[0] - (last_frame + 1)
    if n_tossed_frames != 0:
        warnings.warn(('Tossing trailing {} of {} frames of movie, which'
            ' did not belong to any used block.\n').format(
            n_tossed_frames, movie.shape[0]
        ))
    del n_tossed_frames

    odor_onset_frames = [n - drop_first_n_frames
        for n in odor_onset_frames
    ]
    odor_offset_frames = [n - drop_first_n_frames
        for n in odor_offset_frames
    ]
    block_first_frames = [n - drop_first_n_frames
        for n in block_first_frames
    ]
    # TODO TODO TODO why was i doing this? after subtracting one, is this
    # still not true??? (fix!)
    # i feel like this might mean the rest of my handling of this case might
    # be incorrect...
    #block_first_frames[0] = 0
    # TODO delete after addressing the above. maybe move a check like this
    # to `load_mat_timing_info`
    assert block_first_frames[0] == 0
    #

    block_last_frames = [n - drop_first_n_frames
        for n in block_last_frames
    ]

    frame_times = frame_times[drop_first_n_frames:]
    # TODO TODO TODO is it correct that we were using the last_frame defined
    # before drop_first_n_frames wasa subtracted from everything???
    # TODO want / need to do more than just slice to free up memory from
    # other pixels? is that operation worth it?
    movie = movie[drop_first_n_frames:(last_frame + 1)]

    # This check is now an assert in check_movie_timing_info call below.
    # May need to allow that to be switched to a warning, if this failure
    # mode still exists.
    '''
    if movie.shape[0] != len(frame_times):
        warnings.warn('{} != {}'.format(movie.shape[0], len(frame_times)))
    '''
    check_movie_timing_info(movie, frame_times, block_first_frames,
        block_last_frames
    )

    trial_start_frames, trial_stop_frames = assign_frames_to_trials(
        movie, presentations_per_block, block_first_frames, odor_onset_frames
    )
    # TODO probably do want a fn that can return movie and metadata, so that
    # other segmentation functions can be connected to that...
    # (not clear on what best representation would be, however)
    ############################################################################
    # End what originally happened in gui.py/Segmentation.open_recording
    ############################################################################

    ############################################################################
    # Copied from gui load_ijois
    ############################################################################

    # TODO delete this hardcode hack
    if len(movie.shape) == 4:
        assert tiff.startswith(raw_data_root())
        ijroiset_filename = join(image_dir, 'rois.zip')
        assert exists(ijroiset_filename)
    #
    else:
        # TODO maybe factor this getting roi filname, reading rois, making masks
        # [, extracting traces] into a fn?
        ijroiset_filename = tiff_ijroi_filename(tiff)
        if ijroiset_filename is None:
            raise IOError('tiff_ijroi_filename returned None')

    # TODO may need to use this mtime later (particularly if we enter into
    # db as before in gui)
    # (was set into self.run_at)
    # (also set parameter_json and run_len_seconds to None)
    ijroiset_mtime = datetime.fromtimestamp(getmtime(ijroiset_filename))

    ijrois = ijroi.read_roi_zip(ijroiset_filename)

    frame_shape = movie.shape[1:]
    footprints = ijrois2masks(ijrois, frame_shape)

    raw_f = extract_traces_boolean_footprints(movie, footprints)
    #n_footprints = raw_f.shape[1]

    df_over_f = calculate_df_over_f(raw_f, trial_start_frames,
        odor_onset_frames, trial_stop_frames
    )

    ############################################################################
    # What was originally in gui.py/Segmentation.get_recording_dfs
    ############################################################################
    n_frames, n_cells = df_over_f.shape
    # would have to pass footprints back / read from sql / read # from sql
    #assert n_cells == n_footprints
    # TODO bring back after fixing this indexing issue,
    # whatever it is. as with other check in open_recording
    # (mostly redundant w/ assert comparing movie frames and frame_times in
    # end of open_recording...)
    #assert frame_times.shape[0] == n_frames

    presentation_dfs = []
    comparison_dfs = []
    comparison_num = -1

    # TODO consider deleting this conditional if i'm not actually going to
    # support else case (not used now, see where repeat_num is set in loop)
    if pair_case:
        repeats_across_real_blocks = False
    else:
        repeats_across_real_blocks = True
        repeat_nums = {id_group: 0 for id_group in odor_ids}

    print('processing presentations...', end='', flush=True)
    for i in range(len(trial_start_frames)):
        if i % presentations_per_block == 0:
            comparison_num += 1
            if not repeats_across_real_blocks:
                repeat_nums = {id_group: 0 for id_group in odor_ids}

        start_frame = trial_start_frames[i]
        stop_frame = trial_stop_frames[i]
        onset_frame = odor_onset_frames[i]
        offset_frame = odor_offset_frames[i]

        assert start_frame < onset_frame
        assert onset_frame < offset_frame
        assert offset_frame < stop_frame

        # If either of these is out of bounds, presentation_frametimes will
        # just be shorter than it should be, but it would not immediately
        # make itself apparent as an error.
        assert start_frame < len(frame_times)
        assert stop_frame < len(frame_times)

        onset_time = frame_times[onset_frame]
        # TODO TODO check these don't jump around b/c discontinuities
        # TODO TODO TODO honestly, i forget now, have i ever had acquisition
        # stop any time other than between "blocks"? do i want to stick to
        # that definition?
        # if it did only ever stop between blocks, i suppose i'm gonna have
        # to paint frames between trials within a block as belonging to one
        # trial or the other, for purposes here...
        presentation_frametimes = \
            frame_times[start_frame:stop_frame] - onset_time

        curr_odor_ids = odor_ids[i]
        # TODO update if odor ids are ever actually allowed to be arbitrary
        # len list (and not just forced to be length-2 as they are now, b/c
        # of the db mixture table design)
        odor1, odor2 = curr_odor_ids
        #

        if pair_case:
            repeat_num = repeat_nums[curr_odor_ids]
            repeat_nums[curr_odor_ids] = repeat_num + 1

        # See note in missing odor handling portion of
        # process_segmentation_output to see reasoning behind this choice.
        else:
            repeat_num = comparison_num

        # TODO check that all frames go somewhere and that frames aren't
        # given to two presentations. check they stay w/in block boundaries.
        # (they don't right now. fix!)

        date, fly_num = keys[:2]

        # TODO share more of this w/ dataframe creation below, unless that
        # table is changed to just reference presentation table
        presentation = pd.DataFrame({
            # TODO fix hack (what was this for again? it really a problem?)
            'temp_presentation_id': [i],
            'prep_date': [date],
            'fly_num': fly_num,
            'recording_from': started_at,
            'analysis': ijroiset_mtime, #run_at,
            # TODO get rid of this hack after fixing earlier association of
            # blocks / repeats (or fixing block structure for future
            # recordings)
            'comparison': comparison_num if pair_case else 0,
            'real_block': comparison_num,
            'odor1': odor1,
            'odor2': odor2,
            #'repeat_num': repeat_num if pair_case else comparison_num,
            'repeat_num': repeat_num,
            'odor_onset_frame': onset_frame,
            'odor_offset_frame': offset_frame,
            'from_onset': [[float(x) for x in presentation_frametimes]],
            # TODO now that this isn't in the gui, probably make this start
            # NULL / False?
            'presentation_accepted': True
        })

        # TODO TODO assert that len(presentation_frametimes)
        # == stop_frame - start_frame (off-by-one?)
        # TODO (it would fail now) fix!!
        # maybe this is a failure to merge correctly later???
        # b/c presentation frametimes seems to be defined to be same length
        # above... same indices...
        # (unless maybe frame_times is sometimes shorter than df_over_f, etc)

        '''
        presentation_dff = df_over_f[start_frame:stop_frame, :]
        presentation_raw_f = raw_f[start_frame:stop_frame, :]
        '''
        # TODO TODO fix / delete hack!!
        # TODO probably just need to more correctly calculate stop_frame?
        # (or could also try expanding frametimes to include that...)
        actual_frametimes_slice_len = len(presentation_frametimes)
        stop_frame = start_frame + actual_frametimes_slice_len
        presentation_dff = df_over_f[start_frame:stop_frame, :]
        presentation_raw_f = raw_f[start_frame:stop_frame, :]

        # Assumes that cells are indexed same here as in footprints.
        cell_dfs = []
        for cell_num in range(n_cells):

            cell_dff = presentation_dff[:, cell_num].astype('float32')
            cell_raw_f = presentation_raw_f[:, cell_num].astype('float32')

            cell_dfs.append(pd.DataFrame({
                # TODO maybe rename / do in a less hacky way
                'temp_presentation_id': [i],
                ###'presentation_id': [presentation_id],
                'recording_from': [started_at],
                'segmentation_run': [ijroiset_mtime], #run_at],
                'cell': [cell_num],
                'df_over_f': [[float(x) for x in cell_dff]],
                'raw_f': [[float(x) for x in cell_raw_f]]
            }))
        response_df = pd.concat(cell_dfs, ignore_index=True)

        # TODO maybe draw correlations from each of these, as i go?
        # (would still need to do block by block, not per trial)

        presentation_dfs.append(presentation)
        # TODO rename...
        comparison_dfs.append(response_df)
    print(' done', flush=True)

    # TODO would need to fix coo_matrix handling in 3d+t case
    # (may not be possible w/ coo_matrix...)
    ''''
    n_footprints = footprints.shape[-1]
    footprint_dfs = []
    for cell_num in range(n_footprints):
        # TODO could use tuple of slice objects to accomodate arbitrary dims
        # here (x,y,Z). change all places like this.
        sparse = coo_matrix(footprints[:,:,cell_num])
        footprint_dfs.append(pd.DataFrame({
            'recording_from': [started_at],
            'segmentation_run':  [ijroiset_mtime], #run_at],
            'cell': [cell_num],
            # Can be converted from lists of Python types, but apparently
            # not from numpy arrays or lists of numpy scalar types.
            # TODO check this doesn't transpose things
            # TODO just move appropriate casting to my to_sql function,
            # and allow having numpy arrays (get type info from combination
            # of that and the database, like in other cases)
            # TODO TODO TODO TODO was sparse.col for x_* and sparse.row for
            # y_*. I think this was why I needed to tranpose footprints
            # sometimes. fix everywhere.
            'x_coords': [[int(x) for x in sparse.row.astype('int16')]],
            'y_coords': [[int(x) for x in sparse.col.astype('int16')]],
            'weights': [[float(x) for x in sparse.data.astype('float32')]]
        }))
    footprint_df = pd.concat(footprint_dfs, ignore_index=True)
    '''

    ############################################################################
    # From gui.py/process_segmentation_output
    ############################################################################
    presentations_df = pd.concat(presentation_dfs, ignore_index=True)

    # TODO TODO TODO do i really need to recalculate these?

    # TODO TODO TODO probably just fix self.n_blocks earlier
    # in supermixture case
    # (so there is only one button for accepting and stuff...)
    if pair_case:
        n_blocks = n_blocks
        presentations_per_block = presentations_per_block
    else:
        # TODO delete (though check commented and new are equiv on all
        # non pair_case experiments)
        '''
        n_blocks = presentations_df.comparison.max() + 1
        n_repeats = n_expected_repeats(presentations_df)
        n_stim = len(presentations_df[['odor1','odor2']].drop_duplicates())
        presentations_per_block = n_stim * n_repeats
        '''
        #
        # TODO TODO TODO is this really what i want?
        n_blocks = 1
        presentations_per_block = len(odor_ids)

    presentations_df = merge_odors(presentations_df, db_odors.reset_index())

    # TODO maybe adapt to case where name2 might have only occurence of 
    # an odor, or name1 might be paraffin.
    # TODO TODO check this is actually in the order i want across blocks
    # (idk if name1,name2 are sorted / re-ordered somewhere)
    name1_unique = presentations_df.name1.unique()
    name2_unique = presentations_df.name2.unique()
    # TODO should fail earlier (rather than having to wait for cnmf
    # to finish)
    assert (set(name2_unique) == {'no_second_odor'} or 
        set(name2_unique) - set(name1_unique) == {'paraffin'}
    )
    # TODO TODO TODO factor all abbreviation into its own function
    # (which transforms dataframe w/ full odor names / ids maybe to
    # df w/ the additional abbreviated col (or renamed col))
    single_letter_abbrevs = False
    abbrev_in_presentation_order = True
    if single_letter_abbrevs:
        if not abbrev_in_presentation_order:
            # TODO would (again) need to look up fixed desired order, as
            # in kc_mix_analysis, to support this
            raise NotImplementedError
    else:
        if abbrev_in_presentation_order:
            warnings.warn('abbrev_in_presentation_order can only '
                'be False if not using single_letter_abbrevs'
            )

    if not abbrev_in_presentation_order:
        # TODO would (again) need to look up fixed desired order, as
        # in kc_mix_analysis, to support this
        raise NotImplementedError
        # (could implement other order by just reorder name1_unique)

    # TODO probably move this fn from chemutils to hong2p.utils
    odor2abbrev = cu.odor2abbrev_dict(name1_unique,
        single_letter_abbrevs=single_letter_abbrevs
    )

    # TODO rewrite later stuff to avoid need for this.
    # it just adds a bit of confusion at this point.
    # TODO need to deal w/ no_second_odor in here?
    # So that code detecting which combinations of name1+name2 are
    # monomolecular does not need to change.
    # TODO TODO doesn't chemutils do this at this point? test
    odor2abbrev['paraffin'] = 'paraffin'
    # just so name2 isn't all NaN for now...
    odor2abbrev['no_second_odor'] = 'no_second_odor'

    block_iter = list(range(n_blocks))

    for i in block_iter:
        # TODO maybe concat and only set whole df as instance variable in
        # get_recording_df? then use just as in kc_analysis all throughout
        # here? (i.e. subset from presentationS_df above...)
        presentation_dfs = presentation_dfs[
            (presentations_per_block * i):
            (presentations_per_block * (i + 1))
        ]
        presentation_df = pd.concat(presentation_dfs,
            ignore_index=True
        )
        comparison_dfs = comparison_dfs[
            (presentations_per_block * i):
            (presentations_per_block * (i + 1))
        ]

        # Using this in place of NaN, so frame nums will still always have
        # int dtype. maybe NaN would be better though...
        # TODO maybe i want something unlikely to be system dependent
        # though... if i'm ever going to serialize something containing
        # this value...
        INT_NO_REAL_FRAME = sys.maxsize
        last_real_temp_id = presentation_df.temp_presentation_id.max()

        # Not supporting filling in missing odor presentations in pair case
        # (it hasn't happened yet). (and would need to consider within
        # comparisons, since odors be be shared across)
        if not pair_case:
            # TODO maybe add an 'actual_block' column or something in
            # paircase? or in both?
            assert presentation_df.comparison.nunique() == 1
            n_full_repeats = presentation_df.odor1.value_counts().max()
            assert len(presentation_df) % n_full_repeats == 0

            odor1_set = set(presentation_df.odor1)
            n_odors = len(odor1_set)

            # n_blocks and n_pres_per_actual_block currently have different
            # meanings in pair_case and not here, w/ blocks in this case not
            # matching actual "scopePin high" blocks.
            # In this case, "actual blocks" have each odor once.
            n_pres_per_actual_block = len(presentation_df) // n_full_repeats

            n_missed_per_block = n_odors - n_pres_per_actual_block

            # TODO TODO add a flag for whether we should fill in missing
            # data like this, and maybe fail if the flag is false and we
            # have missing data (b/c plot labels get screwed up)
            # (don't i already have a flag in open_recording? make more
            # global (or an instance variable)?)

            # TODO may want to assert all odor id lookups work in merge
            # (if it doesn't already functionally do that), because
            # technically it's possible that it just so happens the last
            # odor (the missed one) is always the same

            if n_missed_per_block > 0:
                # Could modify loop below to iterate over missed odors if
                # want to support this.
                assert n_missed_per_block == 1, 'for simplicity'

                # from_onset is not hashable so nunique on everything fails
                const_cols = presentation_df.columns[[
                    (False if c == 'from_onset' else
                    presentation_df[c].nunique() == 1)
                    for c in presentation_df.columns
                ]]
                const_vals = presentation_df[const_cols].iloc[0].to_dict()

                # TODO if i were to support where n_cells being different in
                # each block, would need to subset comparison df to block
                # and get unique values from there (in loop below)
                cells = comparison_dfs[0].cell.unique()
                rec_from = const_vals['recording_from']
                filler_seg_run = pd.NaT

                pdf_in_order = \
                    presentation_df.sort_values('odor_onset_frame')

                next_filler_temp_id = last_real_temp_id + 1
                for b in range(n_full_repeats):
                    start = b * n_pres_per_actual_block
                    stop = (b + 1) * n_pres_per_actual_block
                    bdf = pdf_in_order[start:stop]

                    row_data = dict(const_vals)
                    row_data['from_onset'] = [np.nan]
                    # Careful! This should be cleared after frame2order def.
                    row_data['odor_onset_frame'] = \
                        bdf.odor_onset_frame.max() + 1
                    row_data['odor_offset_frame'] = INT_NO_REAL_FRAME

                    real_block_nums = bdf.real_block.unique()
                    assert len(real_block_nums) == 1
                    real_block_num = real_block_nums[0]
                    row_data['real_block'] = real_block_num

                    # The question here is whether I want to start the
                    # repeat numbering with presentations that actually have
                    # frames, or whether I want to keep the numbering as it
                    # would have been...

                    # Since in !pair_case, real_block num should be
                    # equal to the intended repeat_num.
                    row_data['repeat_num'] = real_block_num
                    # TODO would need to fix this case to handle multiple
                    # missing of one odor, if i did want to have repeat_num
                    # numbering start with presentations that actually have
                    # frames
                    # (- 1 since 0 indexed)
                    #row_data['repeat_num'] = n_full_repeats - 1

                    missing_odor1s = list(odor1_set - set(bdf.odor1))
                    assert len(missing_odor1s) == 1
                    missing_odor1 = missing_odor1s.pop()
                    row_data['odor1'] = missing_odor1

                    row_data['temp_presentation_id'] = next_filler_temp_id

                    presentation_df = \
                        presentation_df.append(row_data, ignore_index=True)

                    # TODO what's the np.nan stuff here for?
                    # why not left to get_recording_dfs?
                    comparison_dfs.append(pd.DataFrame({
                        'temp_presentation_id': next_filler_temp_id,
                        'recording_from': rec_from,
                        'segmentation_run': filler_seg_run,
                        'cell': cells,
                        'raw_f': [[np.nan] for _ in range(len(cells))],
                        'df_over_f': [[np.nan] for _ in range(len(cells))]
                    }))
                    next_filler_temp_id += 1

    frame2order = {f: o for o, f in
        enumerate(sorted(presentation_df.odor_onset_frame.unique()))
    }
    presentation_df['order'] = \
        presentation_df.odor_onset_frame.map(frame2order)
    del frame2order

    # This does nothing if there were no missing odor presentations.
    presentation_df.loc[
        presentation_df.temp_presentation_id > last_real_temp_id,
        'odor_onset_frame'] = INT_NO_REAL_FRAME

    comparison_df = pd.concat(comparison_dfs, ignore_index=True,
        sort=False
    )

    # TODO don't have separate instance variables for presentation_dfs
    # and comparison_dfs if i'm always going to merge here.
    # just merge before and then put in one instance variable.
    # (probably just keep name comparison_dfs)
    presentation_df['from_onset'] = presentation_df['from_onset'].apply(
        lambda x: np.array(x)
    )
    presentation_df = merge_odors(presentation_df, db_odors.reset_index())

    # TODO maybe only abbreviate at end? this approach break upload to
    # database? maybe redo so abbrev only happens before plot?
    # (may want a consistent order across experiments anyway)
    presentation_df['original_name1'] = presentation_df.name1.copy()
    presentation_df['original_name2'] = presentation_df.name2.copy()

    presentation_df['name1'] = presentation_df.name1.map(odor2abbrev)
    presentation_df['name2'] = presentation_df.name2.map(odor2abbrev)

    presentation_df = merge_recordings(presentation_df, recording_df)

    # TODO TODO TODO assert here, and earlier if necessary, that
    # each odor has all repeat_num + ordering of repeat_num matches
    # that of 'order' column
    #comparison_df[['name1','repeat_num','order']
    #].drop_duplicates().sort_values(['name1','repeat_num','order'])

    # Just including recording_from so it doesn't get duplicated in
    # output (w/ '_x' and '_y' suffixes). This checks recording_from
    # values are all equal, rather than just dropping one.
    # No other columns should be common.
    comparison_df = comparison_df.merge(presentation_df,
        left_on=['recording_from', 'temp_presentation_id'],
        right_on=['recording_from', 'temp_presentation_id']
    )
    comparison_df.drop(columns='temp_presentation_id', inplace=True)
    del presentation_df

    comparison_df = expand_array_cols(comparison_df)

    # TODO TODO make this optional
    # (and probably move to upload where fig gets saved.
    # just need to hold onto a ref to comparison_df)
    #df_filename = (run_at.strftime('%Y%m%d_%H%M_') +
    df_filename = (ijroiset_mtime.strftime('%Y%m%d_%H%M_') +
        recording_title.replace('/','_') + '.p'
    )
    df_filename = join(analysis_output_root(), 'trace_pickles',
        df_filename
    )

    print('writing dataframe to {}...'.format(df_filename), end='',
        flush=True
    )
    # TODO TODO write a dict pointing to this, to also include PID
    # information in another variable?? or at least stuff to index
    # the PID information?
    comparison_df.to_pickle(df_filename)
    print(' done', flush=True)

    # TODO TODO TODO only return dataframes?
    return comparison_df


def stimfile_odorset(stimfile_path, strict=True):
    data = load_stimfile(stimfile_path)

    # TODO did i use some other indicator elsewhere? anything more robust than
    # this?
    if 'odor_pair_list' in data.keys():
        # Just because I don't believe the pair experiment analysis
        # made any use of something like an odorset, so trying to
        # get one from those stimfiles is likely a mistake.
        if strict:
            # TODO maybe this err should happen whether or not strict is
            # true...?
            raise ValueError('trying to get complex mixture odor set '
                'from old pair experiment stimfile'
            )
        else:
            return None

    odors = set(data['odor_lists'])
    # TODO TODO TODO what caused this error where some fields were empty?
    # this corrupt or test data / not intended to be saved?
    if len(odors) == 0:
        if strict:
            raise ValueError(f'empty odor lists in {stimfile_path}')
        else:
            return None
    assert type(data['odor_lists'][0]) is str

    # TODO TODO TODO fix how stimfile generation stuff doesn't save
    # hardcoded real stuff into odors (+ maybe other vars?)
    '''
    print(len(odors))
    pprint(odors)
    print(len(set(data['odors2pins'].keys())))
    pprint(set(data['odors2pins'].keys()))
    print(len(set(data['pins2odors'].values())))
    pprint(set(data['pins2odors'].values()))
    '''
    #assert odors == set(data['odors'])
    assert odors == set(data['odors2pins'].keys())
    assert odors == set(data['pins2odors'].values())

    # Not this accessor syntax, because .name is a property of all pandas
    # objects.
    odor_names = [split_odor_w_conc(oc)['name'] for oc in odors]

    return odorset_name(odor_names)


def print_all_stimfile_odorsets() -> None:
    stimfiles = sorted(glob.glob(join(stimfile_root(), '*.p')))
    stimfile_odorsets = [stimfile_odorset(sf, strict=False)
        for sf in stimfiles
    ]
    # TODO maybe print grouped by day
    pprint([(split(f)[1], s)
        for f, s in zip(stimfiles, stimfile_odorsets) if s
    ])


solvents = ('pfo', 'water')
natural = ('kiwi', 'fly food')
# TODO maybe load (on demand) + cache the abbreviated versions of these, if
# chemutils is available?
odor_set2order = {
    'kiwi': [
        'pfo',
        'ethyl butyrate',
        'ethyl acetate',
        'isoamyl acetate',
        'isoamyl alcohol',
        'ethanol',
        # TODO check that changing the order of these last two hasn't broken
        # stuff...
        'kiwi approx.',
        'd3 kiwi'
    ],
    'control': [
        'pfo',
        '1-octen-3-ol',
        'furfural',
        'valeric acid',
        'methyl salicylate',
        '2-heptanone',
        # Only one of these will actually be present, they just take the same
        # place in the order.
        'control mix 1',
        'control mix 2'
    ],
    'flyfood': [
        'water',
        'propanoic acid',
        'isobutyric acid',
        'acetic acid',
        'acetoin',
        'ethanol',
        'fly food approx.',
        'fly food b'
    ]
}
def df_to_odor_order(df, observed=True, return_name1=False):
    """Takes a complex-mixture DataFrame to odor names in desired plot order.

    Args:
    df (pd.DataFrame): should have a 'original_name1' column, with names of
        odors from complex mixture experiments we have pre-defined odor orders
        for.

    observed (bool): (optional, default=True) If True, only return odor names
        in `df`.

    return_name1 (bool): (optional, default=False) If True, corresponding
        values in 'name1' will be returned for each value in 'original_name1'.
    """
    # TODO might need to use name1 if original_name1 not there...
    # (for gui case)
    odor_set = odorset_name(df)
    order = odor_set2order[odor_set]
    observed_odors = df.original_name1.unique()
    if observed:
        order = [o for o in order if o in observed_odors]
    else:
        # TODO maybe just handle this externally (force all data w/in some
        # analysis to only have one or the other control mix) and then delete
        # this special casing
        cm1 = 'control mix 1'
        cm2 = 'control mix 2'
        have_cm1 = cm1 in observed_odors
        have_cm2 = cm2 in observed_odors
        order = [o for o in order if o not in (cm1, cm2)]
        if have_cm1:
            assert not have_cm2, 'df should only have either cm1 or cm2'
            order.append(cm1)

        elif have_cm2:
            order.append(cm2)

    if return_name1:
        o2n = df[['original_name1','name1']].drop_duplicates(
            ).set_index('original_name1').name1
        order = list(o2n[order])

    return order


def old_fmt_thorimage_num(x):
    if pd.isnull(x) or not (x[0] == '_' and len(x) == 4):
        return np.nan
    try:
        n = int(x[1:])
        return n
    except ValueError:
        return np.nan


def new_fmt_thorimage_num(x):
    parts = x.split('_')
    if len(parts) == 1:
        return 0
    else:
        return int(x[-1])


def thorsync_num(x):
    prefix = 'SyncData'
    return int(x[len(prefix):])


# TODO rethink gid kwarg(s)
def gsheet_csv_export_link(file_with_edit_link): #, add_default_gid=True):
    """
    Takes a gsheet link copied from browser while editing it, and returns a
    URL suitable for reading it as a CSV into a DataFrame.

    Must append appropriate GID to what is returned.
    """
    # TODO make expectations on URL consistent whether from file or not
    if file_with_edit_link.startswith('http'):
        base_url = file_with_edit_link
    else:
        pkg_data_dir = split(split(__file__)[0])[0]
        with open(join(pkg_data_dir, file_with_edit_link), 'r') as f:
            base_url = f.readline().split('/edit')[0]

    gsheet_link = base_url + '/export?format=csv&gid='
    return gsheet_link


# TODO TODO for this and other stuff that depends on network access (if not
# cached), fallback to cache (unless explicitly prevented?), and warn
# that we are doing so (unless cached version explicitly requested)
_mb_team_gsheet = None
def mb_team_gsheet(use_cache=False, natural_odors_only=False,
    drop_nonexistant_dirs=True, show_inferred_paths=False,
    print_excluded_on_disk=True, verbose=False):
    '''Returns a pandas.DataFrame with data on flies and MB team recordings.
    '''
    global _mb_team_gsheet
    if _mb_team_gsheet is not None:
        return _mb_team_gsheet

    gsheet_cache_file = '.gsheet_cache.p'
    if use_cache and exists(gsheet_cache_file):
        print('Loading MB team sheet data from cache at {}'.format(
            gsheet_cache_file))

        with open(gsheet_cache_file, 'rb') as f:
            sheets = pickle.load(f)

    else:
        # TODO TODO maybe env var pointing to this? or w/ link itself?
        # TODO maybe just get relative path from __file__ w/ /.. or something?
        # TODO TODO TODO give this an [add_]default_gid=True (set to False here)
        # so other code of mine can use this function
        gsheet_link = gsheet_csv_export_link('mb_team_sheet_link.txt')

        # If you want to add more sheets, when you select the new sheet in your
        # browser, the GID will be at the end of the URL in the address bar.
        sheet_gids = {
            'fly_preps': '269082112',
            'recordings': '0',
            'daily_settings': '229338960'
        }

        sheets = dict()
        for df_name, gid in sheet_gids.items():
            df = pd.read_csv(gsheet_link + gid)

            # TODO convert any other dtypes?
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            df.drop(columns=[c for c in df.columns
                if c.startswith('Unnamed: ')], inplace=True)

            if 'fly_num' in df.columns:
                last_with_fly_num = df.fly_num.notnull()[::-1].idxmax()
                df.drop(df.iloc[(last_with_fly_num + 1):].index, inplace=True)

            sheets[df_name] = df

        boolean_columns = {
            'attempt_analysis',
            'raw_data_discarded',
            'raw_data_lost'
        }
        na_cols = list(set(sheets['recordings'].columns) - boolean_columns)
        sheets['recordings'].dropna(how='all', subset=na_cols, inplace=True)

        with open(gsheet_cache_file, 'wb') as f:
            pickle.dump(sheets, f) 

    # TODO maybe make df some merge of the three sheets?
    df = sheets['recordings']

    # TODO TODO maybe flag to disable path inference / rethink how it should
    # interact w/ timestamp based correspondence between thorsync/image and
    # mapping that to the recordings in the gsheet
    # TODO should inference that reads the metadata happen in this fn?
    # maybe yes, but still factor it out and just call here?

    # TODO maybe start by not filling in fully-empty groups / flagging
    # them for later -> preferring to infer those from local files ->
    # then inferring fully-empty groups from default numbering as before

    keys = ['date', 'fly_num']
    # These should happen before rows start being dropped, because the dropped
    # rows might have the information needed to ffill.
    # This should NOT ffill a fly_past a change in date.

    # Assuming that if date changes, even if fly_nums keep going up, that was
    # intentional.
    df.date = df.date.fillna(method='ffill')
    df.fly_num = df.groupby('date')['fly_num'].apply(
        lambda x: x.ffill().bfill()
    )
    # This will only apply to groups (dates) where there are ONLY missing
    # fly_nums, given filling logic above.
    df.dropna(subset=['fly_num'], inplace=True)
    assert not df.date.isnull().any()

    df['stimulus_data_file'] = df['stimulus_data_file'].fillna(method='ffill')

    df.raw_data_discarded = df.raw_data_discarded.fillna(False)
    # TODO say when this happens?
    df.drop(df[df.raw_data_discarded].index, inplace=True)

    # TODO TODO warn if 'attempt_analysis' and either discard / lost is checked

    # Not sure where there were any NaN here anyway...
    df.raw_data_lost = df.raw_data_lost.fillna(False)
    df.drop(df[df.raw_data_lost].index, inplace=True)

    # TODO as per note below, any thorimage/thorsync dirs entered in spreadsheet
    # should probably cause warning/err if either of above rejection reason
    # is checked

    # This happens after data is dropped for the above two reasons, because
    # generally those mistakes do not consume any of our sequential filenames.
    # They should not have files associated with them, and the Google sheet
    # information on them is just for tracking problems / efficiency.
    df['recording_num'] = df.groupby(keys).cumcount() + 1

    if show_inferred_paths:
        missing_thorimage = pd.isnull(df.thorimage_dir)
        missing_thorsync = pd.isnull(df.thorsync_dir)

    my_project = 'natural_odors'

    # TODO TODO fix current behavior where key groups that have nothing filled
    # in on gsheet (for dirs) will default to old format.  misbehaving for
    # 8-27/1 and all of 11-21, for example. (fixed?)

    check_and_set = []
    for gn, gdf in df.groupby(keys):
        if not (gdf.project == my_project).any():
            continue

        if gdf[['thorimage_dir','thorsync_dir']].isnull().all(axis=None):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            if verbose:
                print('\n' + fly_dir)

            try:
                # Since we are disabling check that ThorImage nums (from naming
                # convention) are unique, we must check this before
                # mb_team_gsheet returns.
                image_and_sync_pairs = pair_thor_subdirs(fly_dir,
                     check_unique_thorimage_nums=False
                )
                if verbose:
                    print('pairs:')
                    pprint(image_and_sync_pairs)

            # TODO TODO should ValueError actually be caught?
            # (from comments in other fns) it seems it will only be raised
            # when they are not 1:1, which should maybe cause failure in the way
            # AssertionErrors do now...
            except ValueError as e:
                gn_str = format_keys(*gn)
                print(f'For {gn_str}:')
                print('could not pair thor dirs automatically!')
                print(f'({e})\n')
                continue

            # could maybe try to sort things into "prep checking" / real
            # experiment based on time length or something (and maybe try
            # to fall back to just pairing w/ real experiments? and extending
            # condition below to # real experiments in gdf)
            n_matched = len(image_and_sync_pairs)
            ng = len(gdf)
            # TODO should this be an error (was previously just a
            # print + continue)?
            if n_matched < ng:
                msg = ('more rows for (date, fly) pair than matched outputs'
                    f' ({n_matched} < {ng})'
                )
                raise AssertionError(msg)
                #print(msg)
                #continue

            all_group_in_old_dir_fmt = True
            group_tids = []
            group_tsds = []
            for tid, tsd in image_and_sync_pairs:
                tid = split(tid)[-1]
                if pd.isnull(old_fmt_thorimage_num(tid)):
                    if verbose:
                        print(f'{tid} not in old format')
                    all_group_in_old_dir_fmt = False

                group_tids.append(tid)
                group_tsds.append(split(tsd)[-1])

            # Not immediately setting df in this case, so that I can check
            # these results against the old way of doing things.
            if all_group_in_old_dir_fmt:
                if verbose:
                    print('all in old dir format')

                check_and_set.append((gn, gdf.index, group_tids, group_tsds))
            else:
                if verbose:
                    print('filling in b/c not (all) in old dir format')

                # TODO is it ok to modify df used to create groupby while
                # iterating over groupby?
                df.loc[gdf.index, 'thorimage_dir'] = group_tids
                df.loc[gdf.index, 'thorsync_dir'] = group_tsds

    if print_excluded_on_disk:
        # So that we can exclude these directories when printing stuff on disk
        # (but not in df) later, to reduce noise (because of course other
        # project stuff is dropped and will not be mentioned in df).
        ti_from_other_projects = set()
        ts_from_other_projects = set()
        for gn, gdf in df[df.project != my_project].groupby(keys):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            gdf_ti = set(gdf.thorimage_dir.dropna().apply(
                lambda d: join(fly_dir, d))
            )
            gdf_ts = set(gdf.thorsync_dir.dropna().apply(
                lambda d: join(fly_dir, d))
            )

            ti_from_other_projects |= gdf_ti
            ts_from_other_projects |= gdf_ts

    df.drop(df[df.project != my_project].index, inplace=True)

    # TODO TODO implement option to (at least) also keep prep checking that
    # preceded natural_odors (or maybe just that was on the same day)
    # (so that i can get all that ethyl acetate data for use as a reference
    # odor)

    # TODO display stuff inferred from files separately from stuff inferred
    # from combination of gsheet info and convention

    df['thorimage_num'] = df.thorimage_dir.apply(old_fmt_thorimage_num)
    # TODO TODO should definition of consistency be changed to just check that
    # the ranking of the two are the same?
    # (maybe just if the group is all new format (which will have first real
    # experiments start w/ thorimage_num zero more often, b/c fn / fn_0000
    # thing)
    df['numbering_consistent'] = \
        pd.isnull(df.thorimage_num) | (df.thorimage_num == df.recording_num)

    # TODO unit test this
    # TODO TODO check that, if there are mismatches here, that they *never*
    # happen when recording num will be used for inference in rows in the group
    # *after* the mismatch (?)
    gkeys = keys + [
        'thorimage_dir',
        'thorsync_dir',
        'thorimage_num',
        'recording_num',
        'numbering_consistent'
    ]
    for name, group_df in df.groupby(keys):
        # TODO maybe refactor above so case 3 collapses into case 1?
        '''
        Case 1: all consistent
        Case 2: not all consistent, but all thorimage_dir filled in
        Case 3: not all consistent, but just because thorimage_dir was null
        '''
        #print(group_df[gkeys])

        # TODO check that first_mismatch based approach includes this case
        #if pd.notnull(group_df.thorimage_dir).all():
        #    continue

        mismatches = np.argwhere(~ group_df.numbering_consistent)
        if len(mismatches) == 0:
            continue

        first_mismatch_idx = mismatches[0][0]
        #print('first_mismatch:\n', group_df[gkeys].iloc[first_mismatch_idx])

        # TODO test case where the first mismatch is last
        following_thorimage_dirs = \
            group_df.thorimage_dir.iloc[first_mismatch_idx:]
        #print('checking these are not null:\n', following_thorimage_dirs)
        assert pd.notnull(following_thorimage_dirs).all()

    df.thorsync_dir.fillna(df.thorimage_num.apply(lambda x:
        np.nan if pd.isnull(x) else 'SyncData{:03d}'.format(int(x))),
        inplace=True
    )

    # Leaving recording_num because it might be prettier to use that for
    # IDs in figure than whatever Thor output directory naming convention.
    df.drop(columns=['thorimage_num','numbering_consistent'], inplace=True)

    # TODO TODO check for conditions in which we might need to renumber
    # recording num? (dupes / any entered numbers along the way that are
    # inconsistent w/ recording_num results)
    # TODO update to handle case where thorimage dir does not start w/
    # _ and is not just 3 digits after that?
    # (see what format other stuff from day is?)
    df.thorimage_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else '_{:03d}'.format(int(x))), inplace=True
    )
    df.thorsync_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else 'SyncData{:03d}'.format(int(x))),
        inplace=True
    )

    for gn, gidx, gtids, gtsds in check_and_set:
        # Since some stuff may have been dropped (prep checking stuff, etc).
        still_in_idx = gidx.isin(df.index)
        # No group w/ files on NAS should have been dropped completely.
        assert still_in_idx.sum() > 0, f'group {gn} dropped completely'

        gidx = gidx[still_in_idx]
        gtids = np.array(gtids)[still_in_idx]
        gtsds = np.array(gtsds)[still_in_idx]

        from_gsheet = df.loc[gidx, ['thorimage_dir', 'thorsync_dir']]
        from_thor = [gtids, gtsds]
        consistent = (from_gsheet == from_thor).all(axis=None)
        if not consistent:
            print('Inconsistency between path infererence methods!')
            print(dict(zip(keys, gn)))
            print('Derived from Google sheet:')
            print(from_gsheet.T.to_string(header=False))
            print('From matching Thor output files:')
            print(pd.DataFrame(dict(zip(from_gsheet.columns, from_thor))
                ).T.to_string(header=False))
            print('')
            raise AssertionError('inconsistent rankings w/ old format')

    assert df.fly_num.notnull().all()
    df = df.astype({'fly_num': np.int64})

    cols = keys + ['thorimage_dir','thorsync_dir','attempt_analysis']
    # TODO flag to do this only for stuff marked attempt_analysis
    if show_inferred_paths:
        # TODO only do this if any actually *were* inferred
        print('Inferred ThorImage directories:')
        print(df.loc[missing_thorimage, cols].to_string(index=False))
        print('\nInferred ThorSync directories:')
        print(df.loc[missing_thorsync, cols].to_string(index=False))
        print('')

    duped_thorimage = df.duplicated(subset=keys + ['thorimage_dir'], keep=False)
    duped_thorsync = df.duplicated(subset=keys + ['thorsync_dir'], keep=False)
    try:
        assert not duped_thorimage.any()
        assert not duped_thorsync.any()
    except AssertionError:
        print('Duplicated ThorImage directories after path inference:')
        print(df[duped_thorimage])
        print('\nDuplicated ThorSync directories after path inference:')
        print(df[duped_thorsync])
        raise

    flies = sheets['fly_preps']
    flies['date'] = flies['date'].fillna(method='ffill')
    flies.dropna(subset=['date','fly_num'], inplace=True)

    # TODO maybe flag to not update database? or just don't?
    # TODO groups all inserts into transactions across tables, and as few as
    # possible (i.e. only do this later)?
    to_sql_with_duplicates(flies.rename(
        columns={'date': 'prep_date'}), 'flies'
    )

    # For manual sanity checking that important data isn't being excluded
    # inappropriately.
    if print_excluded_on_disk:
        ti_ondisk_not_in_df = set()
        ts_ondisk_not_in_df = set()
        for gn, gdf in df.groupby(keys):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            # Need them somewhat absolute (w/ date + fly info at least), so that
            # set operations on directories across (date, fly) combinations are
            # meaningful.
            gdf_ti = set(gdf.thorimage_dir.apply(lambda d: join(fly_dir, d)))
            gdf_ts = set(gdf.thorsync_dir.apply(lambda d: join(fly_dir, d)))

            thorimage_dirs, thorsync_dirs = thor_subdirs(fly_dir)
            ti_ondisk_not_in_df |= set(thorimage_dirs) - gdf_ti
            ts_ondisk_not_in_df |= set(thorsync_dirs) - gdf_ts

        # Excluding other-project stuff that was dropped from df earlier.
        ti_ondisk_not_in_df -= ti_from_other_projects
        ts_ondisk_not_in_df -= ts_from_other_projects

        msg = '{} directories on disk but not in DataFrame (from gsheet):'
        if len(ti_ondisk_not_in_df) > 0:
            print(msg.format('ThorImage'))
            pprint(ti_ondisk_not_in_df)
            print('')

        if len(ts_ondisk_not_in_df) > 0:
            print(msg.format('ThorSync'))
            pprint(ts_ondisk_not_in_df)
            print('')

    fly_dirs = df.apply(lambda r: raw_fly_dir(r.date, r.fly_num), axis=1)
    abs_thorimage_dirs = fly_dirs.str.cat(others=df.thorimage_dir, sep='/')
    abs_thorsync_dirs = fly_dirs.str.cat(others=df.thorsync_dir, sep='/')
    thorimage_exists = abs_thorimage_dirs.apply(isdir)
    thorsync_exists = abs_thorsync_dirs.apply(isdir)
    any_dir_missing = ~ (thorimage_exists & thorsync_exists)

    any_missing_marked_attempt = (any_dir_missing & df.attempt_analysis)
    # TODO maybe an option to just warn here, rather than failing
    if any_missing_marked_attempt.any():
        print('Directories marked attempt analysis with missing data:')
        print(df.loc[any_missing_marked_attempt, cols[:-1]].to_string())
        print('')
        raise AssertionError('some experiments marked attempt_analysis '
            'had some data directories missing')

    if drop_nonexistant_dirs:
        n_to_drop = any_dir_missing.sum()
        if n_to_drop > 0:
            print(
                f'Dropping {n_to_drop} rows because directories did not exist.'
            )
        df.drop(df[any_dir_missing].index, inplace=True)

    # TODO TODO is 2019-08-27 fn_0000 stuff inferred correctly?
    # (will have same thorimage num as fn) (?)
    # (not critical apart from value as test case, b/c all stuff used from
    # that day has explicit paths in gsheet)

    _mb_team_gsheet = df

    # TODO handle case where database is empty but gsheet cache still exists
    # (all inserts will probably fail, for lack of being able to reference fly
    # table)
    return df


def merge_gsheet(df, *args, use_cache=False):
    """
    df must have a column named either 'recording_from' or 'started_at'

    gsheet rows get this information by finding the right ThorImage
    Experiment.xml files on the NAS and loading them for this timestamp.
    """
    if len(args) == 0:
        gsdf = mb_team_gsheet(use_cache=use_cache)
    elif len(args) == 1:
        # TODO maybe copy in this case?
        gsdf = args[0]
    else:
        raise ValueError('incorrect number of arguments')

    if 'recording_from' in df.columns:
        # TODO maybe just merge_recordings w/ df in advance in this case?
        df = df.rename(columns={'recording_from': 'started_at'})
    elif 'started_at' not in df.columns:
        raise ValueError("df needs 'recording_from'/'started_at' in columns")

    gsdf['recording_from'] = pd.NaT
    for i, row in gsdf.iterrows():
        date_dir = row.date.strftime(date_fmt_str)
        fly_num = str(int(row.fly_num))
        image_dir = join(raw_data_root(),
            date_dir, fly_num, row.thorimage_dir
        )
        thorimage_xml_path = join(image_dir, 'Experiment.xml')

        try:
            xml_root = _xmlroot(thorimage_xml_path)
        except FileNotFoundError as e:
            continue

        gsdf.loc[i, 'recording_from'] = get_thorimage_time_xml(xml_root)

    # TODO fail if stuff marked attempt_analysis has missing xml files?
    # or if nothing was found?

    gsdf = gsdf.rename(columns={'date': 'prep_date'})

    return merge_recordings(gsdf, df, verbose=False)


def merge_odors(df, *args):
    global conn
    if conn is None:
        conn = get_db_conn()

    if len(args) == 0:
        odors = pd.read_sql('odors', conn)
    elif len(args) == 1:
        odors = args[0]
    else:
        raise ValueError('incorrect number of arguments')

    print('merging with odors table...', end='', flush=True)
    # TODO way to do w/o resetting index? merge failing to find odor1 or just
    # drop?
    # TODO TODO TODO do i want drop=True? (it means cols in index won't be
    # inserted into dataframe...) check use of merge_odors and change to
    # drop=False (default) if it won't break anything
    df = df.reset_index(drop=True)

    df = pd.merge(df, odors, left_on='odor1', right_on='odor_id',
                  suffixes=(False, False))

    df.drop(columns=['odor_id','odor1'], inplace=True)
    df.rename(columns={'name': 'name1',
        'log10_conc_vv': 'log10_conc_vv1'}, inplace=True)

    df = pd.merge(df, odors, left_on='odor2', right_on='odor_id',
                  suffixes=(False, False))

    df.drop(columns=['odor_id','odor2'], inplace=True)
    df.rename(columns={'name': 'name2',
        'log10_conc_vv': 'log10_conc_vv2'}, inplace=True)

    print(' done')

    # TODO refactor merge fns to share some stuff? (progress, length checking,
    # arg unpacking, etc)?
    return df


def merge_recordings(df, *args, verbose=True):
    global conn
    if conn is None:
        conn = get_db_conn()

    if len(args) == 0:
        recordings = pd.read_sql('recordings', conn)
    elif len(args) == 1:
        recordings = args[0]
    else:
        raise ValueError('incorrect number of arguments')

    print('merging with recordings table...', end='', flush=True)
    len_before = len(df)
    # TODO TODO TODO do i want drop=True? (it means cols in index won't be
    # inserted into dataframe...) check use of this fn and change to
    # drop=False (default) if it won't break anything
    df = df.reset_index(drop=True)

    df = pd.merge(df, recordings, how='left', left_on='recording_from',
        right_on='started_at', suffixes=(False, False))

    df.drop(columns=['started_at'], inplace=True)

    # TODO TODO see notes in kc_analysis about sub-recordings and how that
    # will now break this in the recordings table
    # (multiple dirs -> one start time)
    df['thorimage_id'] = df.thorimage_path.apply(lambda x: split(x)[-1])
    assert len_before == len(df), 'merging changed input df length'
    print(' done')
    return df


def arraylike_cols(df):
    """Returns a list of columns that have only lists or arrays as elements.
    """
    df = df.select_dtypes(include='object')
    return df.columns[df.applymap(lambda o:
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

    orig_dtypes = df.dtypes.to_dict()
    for ac in array_cols:
        df[ac] = df[ac].apply(lambda x: np.array(x))
        assert len(df[ac]) > 0 and len(df[ac][0]) > 0
        orig_dtypes[ac] = df[ac][0][0].dtype

    non_array_cols = df.columns.difference(array_cols)

    # TODO true vectorized way to do this?
    # is str.len (on either rows/columns) faster (+equiv)?
    array_lengths = df[array_cols].applymap(len)
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


def diff_dataframes(df1, df2):
    """Returns a DataFrame summarizing input differences.
    """
    # TODO do i want df1 and df2 to be allowed to be series?
    # (is that what they are now? need to modify anything?)
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
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

        other_cols = set(df1.columns) - set(floats1.columns) - set(arr_cols)
        other_diff_mask = df1[other_cols] != df2[other_cols]

        diff_mask = pd.concat([
            diff_mask_floats,
            diff_mask_arr,
            other_diff_mask], axis=1)

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
                            index=changed.index)


def first_group(df, group_cols):
    """Returns key tuple and df of first group, grouping df on group_cols.

    Just for ease of interactively testing out functions on DataFrames of a
    groupby.
    """
    gb = df.groupby(group_cols)
    first_group_tuple = list(gb.groups.keys())[0]
    gdf = gb.get_group(first_group_tuple)
    return first_group_tuple, gdf


def git_hash(repo_file):
    """Takes any file in a git directory and returns current hash.
    """
    import git
    repo = git.Repo(repo_file, search_parent_directories=True)
    current_hash = repo.head.object.hexsha
    return current_hash


# TODO TODO maybe check that remote seems to be valid, and fail if not.
# don't want to assume we have an online (backed up) record of git repo when we
# don't...
def version_info(*args, used_for='', force_git=False):
    """Takes module or string path to file in Git repo to a dict with version
    information (with keys and values the database will accept).
    """
    import git
    import pkg_resources

    if len(args) == 1:
        module_or_path = args[0]
    elif len(args) == 0:
        module_or_path = __file__
        force_git = True
    else:
        raise ValueError('too many arguments')

    if isinstance(module_or_path, ModuleType):
        module = module_or_path
        pkg_path = module.__file__
        name = module.__name__
    else:
        if type(module_or_path) != str:
            raise ValueError('must path either a Python module or str path')
        pkg_path = module_or_path
        module = None

    try:
        repo = git.Repo(pkg_path, search_parent_directories=True)
        name = split(repo.working_tree_dir)[-1]
        remote_urls = list(repo.remotes.origin.urls)
        assert len(remote_urls) == 1
        remote_url = remote_urls[0]

        current_hash = repo.head.object.hexsha

        index = repo.index
        diff = index.diff(None, create_patch=True)
        changes = ''
        for d in diff:
            changes += str(d)

        return {
            'name': name,
            'used_for': used_for,
            'git_remote': remote_url,
            'git_hash': current_hash,
            'git_uncommitted_changes': changes
        }

    except git.exc.InvalidGitRepositoryError:
        if force_git:
            raise

        if module is None:
            # TODO try to find module from str
            raise NotImplementedError(
                'pass module for non-source installations')

        # There may be circumstances in which module name isn't the right name
        # to use here, but assuming we won't encounter that for now.
        version = pkg_resources.get_distribution(module.__name__).version

        return {'name': name, 'used_for': used_for, 'version': version}


def upload_code_info(code_versions):
    """Returns a list of integer IDs for inserted code version rows.

    code_versions should be a list of dicts, each dict representing a row in the
    corresponding table.
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    if len(code_versions) == 0:
        raise ValueError('code versions can not be empty')

    code_versions_df = pd.DataFrame(code_versions)
    # TODO delete try/except
    try:
        code_versions_df.to_sql('code_versions', conn, if_exists='append',
            index=False)
    except:
        print(code_versions_df)
        import ipdb; ipdb.set_trace()

    # TODO maybe only read most recent few / restrict to some other key if i
    # make one?
    db_code_versions = pd.read_sql('code_versions', conn)

    our_version_cols = code_versions_df.columns
    version_ids = list()
    for _, row in code_versions_df.iterrows():
        # This should take the *first* row that is equal.
        idx = (db_code_versions[code_versions_df.columns] == row).all(
            axis=1).idxmax()
        version_id = db_code_versions['version_id'].iat[idx]
        assert version_id not in version_ids
        version_ids.append(version_id)

    return version_ids


def upload_analysis_info(*args) -> None:
    """
    Requires that corresponding row in analysis_runs table already exists,
    if only two args are passed.
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    have_ids = False
    if len(args) == 2:
        analysis_started_at, code_versions = args
    elif len(args) == 3:
        recording_started_at, analysis_started_at, code_versions = args

        if len(code_versions) == 0:
            raise ValueError('code_versions can not be empty')

        if type(code_versions) == list and np.issubdtype(
            type(code_versions[0]), np.integer):

            version_ids = code_versions
            have_ids = True

        pd.DataFrame({
            'run_at': [analysis_started_at],
            'recording_from': recording_started_at
        }).set_index('run_at').to_sql('analysis_runs', conn,
            if_exists='append', method=pg_upsert)

    else:
        raise ValueError('incorrect number of arguments')

    if not have_ids:
        version_ids = upload_code_info(code_versions)

    analysis_code = pd.DataFrame({
        'run_at': analysis_started_at,
        'version_id': version_ids
    })
    to_sql_with_duplicates(analysis_code, 'analysis_code')


def motion_corrected_tiff_filename(date, fly_num, thorimage_id):
    """Takes vars identifying recording to the name of a motion corrected TIFF
    for it. Non-rigid preferred over rigid. Relies on naming convention.
    """
    tif_dir = join(analysis_fly_dir(date, fly_num), 'tif_stacks')
    nr_tif = join(tif_dir, '{}_nr.tif'.format(thorimage_id))
    rig_tif = join(tif_dir, '{}_rig.tif'.format(thorimage_id))
    tif = None
    if exists(nr_tif):
        tif = nr_tif
    elif exists(rig_tif):
        tif = rig_tif

    if tif is None:
        raise IOError('No motion corrected TIFs found in {}'.format(tif_dir))

    return tif


# TODO use this in other places that normalize to thorimage_ids
def tiff_thorimage_id(tiff_filename):
    """
    Takes a path to a TIFF and returns ID to identify recording within
    (date, fly). Relies on naming convention.

    Works for input that is either a raw TIFF or a motion corrected TIFF,
    the latter of which should have a conventional suffix indicating the
    type of motion correction ('_nr' / '_rig').
    """
    # Behavior of os.path.split makes this work even if tiff_filename does not
    # have any directories in it.
    parts = split(tiff_filename[:-len('.tif')])[1].split('_')

    # Last part of the filename, which I use to indicate the type of motion
    # correction applied. Should only apply in TIFFs under analysis directory.
    if parts[-1] in ('nr', 'rig'):
        parts = parts[:-1]

    return '_'.join(parts)


# TODO don't expose this if i can refactor other stuff to not use it
# otherwise use this rather than their own separate definitions
# (like in populate_db, etc)
rel_to_cnmf_mat = 'cnmf'
def matfile(date, fly_num, thorimage_id):
    """Returns filename of Remy's metadata [+ CNMF output] .mat file.
    """
    return join(analysis_fly_dir(date, fly_num), rel_to_cnmf_mat,
        thorimage_id + '_cnmf.mat'
    )


def tiff_matfile(tif):
    """Returns filename of Remy's metadata [+ CNMF output] .mat file.
    """
    keys = tiff_filename2keys(tif)
    return matfile(*keys)


def metadata_filename(date, fly_num, thorimage_id):
    """Returns filename of YAML for extra metadata.
    """
    return join(raw_fly_dir(date, fly_num), thorimage_id + '_metadata.yaml')


# TODO maybe something to indicate various warnings
# (like mb team not being able to pair things) should be suppressed?
def metadata(date, fly_num, thorimage_id):
    """Returns metadata from YAML, with defaults added.
    """
    import yaml

    metadata_file = metadata_filename(date, fly_num, thorimage_id)

    # TODO another var specifying number of frames that has *already* been
    # cropped out of raw tiff (start/end), to resolve any descrepencies wrt 
    # thorsync data
    metadata = {
        'drop_first_n_frames': 0
    }
    if exists(metadata_file):
        # TODO TODO TODO also load single odors (or maybe other trial
        # structures) from stuff like this, so analysis does not need my own
        # pickle based stim format
        with open(metadata_file, 'r') as mdf:
            yaml_metadata = yaml.load(mdf)

        for k in metadata.keys():
            if k in yaml_metadata:
                metadata[k] = yaml_metadata[k]

    return metadata


def tiff_filename2keys(tiff_filename):
    """Takes TIFF filename to pd.Series w/ 'date','fly_num','thorimage_id' keys.

    TIFF must be placed and named according to convention, because the
    date and fly_num are taken from names of some of the containing directories.

    Works with TIFFs either under `raw_data_root` or `analysis_output_root`.
    """
    parts = tiff_filename.split(sep)[-5:]
    date = None
    for i, p in enumerate(parts):
        try:
            date = pd.Timestamp(datetime.strptime(p, date_fmt_str))
            fly_num_idx = i + 1
            break
        except ValueError:
            pass

    if date is None:
        raise ValueError('no date directory found in TIFF path')

    fly_num = int(parts[fly_num_idx])
    thorimage_id = tiff_thorimage_id(tiff_filename)
    return pd.Series({
        'date': date, 'fly_num': fly_num, 'thorimage_id': thorimage_id
    })


def recording_df2keys(df):
    dupes = df[recording_cols].drop_duplicates()
    assert len(dupes) == 1
    return tuple(dupes.iloc[0])


def list_motion_corrected_tifs(include_rigid=False, attempt_analysis_only=True):
    """List motion corrected TIFFs in conventional directory structure on NAS.
    """
    motion_corrected_tifs = []
    df = mb_team_gsheet()
    for full_date_dir in sorted(glob.glob(join(analysis_output_root(), '**'))):
        for full_fly_dir in sorted(glob.glob(join(full_date_dir, '**'))):
            date_dir = split(full_date_dir)[-1]
            try:
                fly_num = int(split(full_fly_dir)[-1])

                fly_used = df.loc[df.attempt_analysis &
                    (df.date == date_dir) & (df.fly_num == fly_num)]

                used_thorimage_dirs = set(fly_used.thorimage_dir)

                tif_dir = join(full_fly_dir, 'tif_stacks')
                if exists(tif_dir):
                    tif_glob = '*.tif' if include_rigid else '*_nr.tif'
                    fly_tifs = sorted(glob.glob(join(tif_dir, tif_glob)))

                    used_tifs = [x for x in fly_tifs if '_'.join(
                        split(x)[-1].split('_')[:-1]) in used_thorimage_dirs]

                    motion_corrected_tifs += used_tifs

            except ValueError:
                continue

    return motion_corrected_tifs


def list_segmentations(tif_path):
    """Returns a DataFrame of segmentation_runs for given motion corrected TIFF.
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    # TODO could maybe turn these two queries into one (WITH semantics?)
    # TODO TODO should maybe trim all prefixes from input_filename before
    # uploading? unless i want to just figure out path from other variables each
    # time and use that to match (if NAS_PREFIX is diff, there will be no match)
    prefix = analysis_output_root()
    if tif_path.startswith(prefix):
        tif_path = tif_path[len(prefix):]

    # TODO test this strpos stuff is equivalent to where input_filename = x
    # in case where prefixes are the same
    analysis_runs = pd.read_sql_query('SELECT * FROM analysis_runs WHERE ' +
        "strpos(input_filename, '{}') > 0".format(tif_path), conn)

    if len(analysis_runs) == 0:
        return None

    # TODO better way than looping over each of these? move to sql query?
    analysis_start_times = analysis_runs.run_at.unique()
    seg_runs = []
    for run_at in analysis_start_times:
        seg_runs.append(pd.read_sql_query('SELECT * FROM segmentation_runs ' +
            "WHERE run_at = '{}'".format(pd.Timestamp(run_at)), conn
        ))

        # TODO maybe merge w/ analysis_code (would have to yield multiple rows
        # per segmentation run when multiple code versions referenced)

    seg_runs = pd.concat(seg_runs, ignore_index=True)
    if len(seg_runs) == 0:
        return None

    seg_runs = seg_runs.merge(analysis_runs)
    seg_runs.sort_values('run_at', inplace=True)
    return seg_runs


def is_thorsync_dir(d, verbose=False):
    """True if dir has expected ThorSync outputs, False otherwise.
    """
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_settings = False
    have_h5 = False
    for f in files:
        # checking for substring
        if 'ThorRealTimeDataSettings.xml' in f:
            have_settings = True
        if '.h5':
            have_h5 = True

    if verbose:
        print('have_settings:', have_settings)
        print('have_h5:', have_h5)

    return have_h5 and have_settings


def is_thorimage_dir(d, verbose=False):
    """True if dir has expected ThorImage outputs, False otherwise.

    Looks for .raw not any TIFFs now.
    """
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_xml = False
    have_raw = False
    # TODO support tif output case(s) as well
    #have_processed_tiff = False
    for f in files:
        if f == 'Experiment.xml':
            have_xml = True
        # Needs to match at least 'Image_0001_0001.raw' and 'Image_001_001.raw'
        elif f.startswith('Image_00') and f.endswith('001.raw'):
            have_raw = True
        #elif f == split(d)[-1] + '_ChanA.tif':
        #    have_processed_tiff = True

    if verbose:
        print('have_xml:', have_xml)
        print('have_raw:', have_raw)
        if have_xml and not have_raw:
            print('all dir contents:')
            pprint(files)

    if have_xml and have_raw:
        return True
    else:
        return False


def _filtered_subdirs(parent_dir, filter_funcs, exclusive=True, verbose=False):
    """Takes dir and indicator func(s) to subdirs satisfying them.

    Output is a flat list of directories if filter_funcs is a function.

    If it is a list of funcs, output has the same length, with each element
    a list of satisfying directories.
    """
    parent_dir = normpath(parent_dir)

    try:
        _ = iter(filter_funcs)
    except TypeError:
        filter_funcs = [filter_funcs]

    # [[]] * len(filter_funcs) was the inital way I tried this, but the inner
    # lists all end up referring to the same object.
    all_filtered_subdirs = []
    for _ in range(len(filter_funcs)):
        all_filtered_subdirs.append([])

    for d in glob.glob(f'{parent_dir}{sep}*{sep}'):
        if verbose:
            print(d)

        for fn, filtered_subdirs in zip(filter_funcs, all_filtered_subdirs):
            if verbose:
                print(fn.__name__)

            if verbose:
                try:
                    val = fn(d, verbose=True)
                except TypeError:
                    val = fn(d)
            else:
                val = fn(d)

            if verbose:
                print(val)

            if val:
                filtered_subdirs.append(d[:-1])
                if exclusive:
                    break

        if verbose:
            print('')

    if len(filter_funcs) == 1:
        all_filtered_subdirs = all_filtered_subdirs[0]

    return all_filtered_subdirs


def thorimage_subdirs(parent_dir):
    """
    Returns a list of any immediate child directories of `parent_dir` that have
    all expected ThorImage outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorimage_dir)


def thorsync_subdirs(parent_dir):
    """Returns a list of any immediate child directories of `parent_dir`
    that have all expected ThorSync outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorsync_dir)


def thor_subdirs(parent_dir, absolute_paths=True):
    """
    Returns a length-2 tuple, where the first element is all ThorImage children
    and the second element is all ThorSync children (of `parent_dir`).
    """
    thorimage_dirs, thorsync_dirs = _filtered_subdirs(parent_dir,
        (is_thorimage_dir, is_thorsync_dir)
    )
    if not absolute_paths:
        thorimage_dirs = [split(d)[-1] for d in thorimage_dirs]
        thorsync_dirs = [split(d)[-1] for d in thorsync_dirs]

    return (thorimage_dirs, thorsync_dirs)


def pair_thor_dirs(thorimage_dirs, thorsync_dirs, use_mtime=False,
    use_ranking=True, check_against_naming_conv=True,
    check_unique_thorimage_nums=True, verbose=False):
    """
    Takes lists (not necessarily same len) of dirs, and returns a list of
    lists of matching (ThorImage, ThorSync) dirs (sorted by experiment time).

    Args:
    check_against_naming_conv (bool): (default=True) If True, check ordering
        from pairing is consistent with ordering derived from our naming
        conventions for Thor software output.

    Raises ValueError if two dirs of one type match to the same one of the
    other, but just returns shorter list of pairs if some matches can not be
    made. These errors currently just cause skipping of pairing for the
    particular (date, fly) pair above (though maybe this should change?).

    Raises AssertionError when assumptions are violated in a way that should
    trigger re-evaluating the code.
    """
    if use_ranking:
        if len(thorimage_dirs) != len(thorsync_dirs):
            raise ValueError('can only pair with ranking when equal # dirs')

    thorimage_times = {d: get_thorimage_time(d, use_mtime=use_mtime)
        for d in thorimage_dirs
    }
    thorsync_times = {d: get_thorsync_time(d) for d in thorsync_dirs}

    thorimage_dirs = np.array(
        sorted(thorimage_dirs, key=lambda x: thorimage_times[x])
    )
    thorsync_dirs = np.array(
        sorted(thorsync_dirs, key=lambda x: thorsync_times[x])
    )

    if use_ranking:
        pairs = list(zip(thorimage_dirs, thorsync_dirs))
    else:
        from scipy.optimize import linear_sum_assignment

        # TODO maybe call scipy func on pandas obj w/ dirs as labels?
        costs = np.empty((len(thorimage_dirs), len(thorsync_dirs))) * np.nan
        for i, tid in enumerate(thorimage_dirs):
            ti_time = thorimage_times[tid]
            if verbose:
                print('tid:', tid)
                print('ti_time:', ti_time)

            for j, tsd in enumerate(thorsync_dirs):
                ts_time = thorsync_times[tsd]

                cost = (ts_time - ti_time).total_seconds()

                if verbose:
                    print(' tsd:', tsd)
                    print('  ts_time:', ts_time)
                    print('  cost (ts - ti):', cost)

                # Since ts time should be larger, but only if comparing XML TI
                # time w/ TS mtime (which gets changed as XML seems to be
                # written as experiment is finishing / in progress).
                if use_mtime:
                    cost = abs(cost)

                elif cost < 0:
                    # TODO will probably just need to make this a large const
                    # inf seems to make the scipy imp fail. some imp it works
                    # with?
                    #cost = np.inf
                    cost = 1e7

                costs[i,j] = cost

            if verbose:
                print('')

        ti_idx, ts_idx = linear_sum_assignment(costs)
        print(costs)
        print(ti_idx)
        print(ts_idx)
        pairs = list(zip(thorimage_dirs[ti_idx], thorsync_dirs[ts_idx]))

    if check_against_naming_conv:
        ti_last_parts = [split(tid)[-1] for tid, _ in pairs]

        thorimage_nums = []
        not_all_old_fmt = False
        for tp in ti_last_parts:
            num = old_fmt_thorimage_num(tp)
            if pd.isnull(num):
                not_all_old_fmt = True
                break
            thorimage_nums.append(num)

        if not_all_old_fmt:
            try:
                thorimage_nums = [new_fmt_thorimage_num(d)
                    for d in ti_last_parts
                ]
            # If ALL ThorImage directories are not in old naming convention,
            # then we assume they will ALL be named according to the new
            # convention.
            except ValueError as e:
                # (changing error type so it isn't caught, w/ other ValueErrors)
                raise AssertionError('Check against naming convention failed, '
                    'because a new_fmt_thorimage_num parse call failed with: ' +
                    str(e)
                )

        # TODO TODO need to stable (arg)sort if not going to check this, but
        # still checking ordering below??? (or somehow ordering by naming
        # convention, so that fn comes before fn_0000, etc?)

        # Call from mb_team_gsheet disables this, so that fn / fn_0000 don't
        # cause a failure even though both have ThorImage num of 0, because fn
        # should always be dropped after the pairing in this case (should be
        # checked in mb_team_gsheet after, since it will then not be checked
        # here).
        if check_unique_thorimage_nums:
            if len(thorimage_nums) > len(set(thorimage_nums)):
                print('Directories where pairing failed:')
                print('ThorImage:')
                pprint(list(thorimage_dirs))
                print('Extracted thorimage_nums:')
                pprint(thorimage_nums)
                print('ThorSync:')
                pprint(list(thorsync_dirs))
                print('')
                raise AssertionError('thorimage nums were not unique')

        thorsync_nums = [thorsync_num(split(tsd)[-1]) for _, tsd in pairs]

        # Ranking rather than straight comparison in case there is an offset.
        ti_rankings = np.argsort(thorimage_nums)
        ts_rankings = np.argsort(thorsync_nums)
        if not np.array_equal(ti_rankings, ts_rankings):
            raise AssertionError('time based rankings inconsistent w/ '
                'file name convention rankings')
        # TODO maybe also re-order pairs by these rankings? or by their own,
        # to also include case where not check_against... ?

        return pairs

    """
    thorimage_times = {d: get_thorimage_time(d) for d in thorimage_dirs}
    thorsync_times = {d: get_thorsync_time(d) for d in thorsync_dirs}

    image_and_sync_pairs = []
    matched_dirs = set()
    # TODO make sure this order is going the way i want
    for tid in sorted(thorimage_dirs, key=lambda x: thorimage_times[x]):
        ti_time = thorimage_times[tid]
        if verbose:
            print('tid:', tid)
            print('ti_time:', ti_time)

        # Seems ThorImage time (from TI XML) is always before ThorSync time
        # (from mtime of TS XML), so going to look for closest mtime.
        # TODO could also warn / fail if closest ti mtime to ts mtime
        # is inconsistent? or just use that?
        # TODO or just use numbers in names? or default to that / warn/fail if
        # not consistent?

        # TODO TODO would need to modify this alg to handle many cases
        # where there are mismatched #'s of recordings
        # (first tid will get the tsd, even if another tid is closer)
        # scipy.optimize.linear_sum_assignment looks interesting, but
        # not sure it can handle 

        min_positive_td = None
        closest_tsd = None
        for tsd in thorsync_dirs:
            ts_time = thorsync_times[tsd]
            td = (ts_time - ti_time).total_seconds()

            if verbose:
                print(' tsd:', tsd)
                print('  ts_time:', ts_time)
                print('  td (ts - ti):', td)

            # Since ts_time should be larger.
            if td < 0:
                continue

            if min_positive_td is None or td < min_positive_td:
                min_positive_td = td
                closest_tsd = tsd

            '''
            # didn't seem to work at all for newer output ~10/2019
            if abs(td) < time_mismatch_cutoff_s:
                if tid in matched_dirs or tsd in matched_dirs:
                    raise ValueError(f'either {tid} or {tsd} was already '
                        f'matched. existing pairs:\n{matched_dirs}')

                image_and_sync_pairs.append((tid, tsd))
                matched_dirs.add(tid)
                matched_dirs.add(tsd)
            '''

            matched_dirs.add(tid)
            matched_dirs.add(tsd)

        if verbose:
            print('')

    return image_and_sync_pairs
    """


def pair_thor_subdirs(parent_dir, verbose=False, **kwargs):
    """
    Raises ValueError/AssertionError when pair_thor_dirs does.

    Above, the former causes skipping of automatic pairing, whereas the latter
    is not handled and will intentionally cause failure, to prevent incorrect
    assumptions from leading to incorrect results.
    """
    # TODO TODO need to handle case where maybe one thorimage/sync dir doesn't
    # have all output, and then that would maybe offset the pairing? test!
    # (change filter fns to include a minimal set of data, s.t. all such cases
    # still are counted?)
    thorimage_dirs, thorsync_dirs = thor_subdirs(parent_dir)
    if verbose:
        print('thorimage_dirs:')
        pprint(thorimage_dirs)
        print('thorsync_dirs:')
        pprint(thorsync_dirs)

    return pair_thor_dirs(thorimage_dirs, thorsync_dirs, verbose=True, **kwargs)


# TODO still work w/ parens added around initial .+ ? i want to match the parent
# id...
shared_subrecording_regex = r'(.+)_\db\d_from_(nr|rig)'
def is_subrecording(thorimage_id):
    """
    Returns whether a recording id matches my GUIs naming convention for the
    "sub-recordings" it can create.
    """
    if re.search(shared_subrecording_regex + '$', thorimage_id):
        return True
    else:
        return False


def is_subrecording_tiff(tiff_filename):
    """
    Takes a TIFF filename to whether it matches the GUI's naming convention for
    the "sub-recordings" it can create.
    """
    # TODO technically, nr|rig should be same across two...
    if re.search(shared_subrecording_regex + '_(nr|rig).tif$', tiff_filename):
        return True
    else:
        return False


def subrecording_tiff_blocks(tiff_filename):
    """Returns tuple of int (start, stop) block numbers subrecording contains.

    Block numbers start at 0.

    Requires that is_subrecording_tiff(tiff_filename) would return True.
    """
    parts = tiff_filename.split('_')[-4].split('b')

    first_block = int(parts[0]) - 1
    last_block = int(parts[1]) - 1

    return first_block, last_block


def subrecording_tiff_blocks_df(series):
    """Takes a series w/ TIFF name in series.name to (start, stop) block nums.

    (series.name must be a TIFF path)

    Same behavior as `subrecording_tiff_blocks`.
    """
    # TODO maybe fail in this case?
    if not series.is_subrecording:
        return None, None

    tiff_filename = series.name
    first_block, last_block = subrecording_tiff_blocks(tiff_filename)
    return first_block, last_block
    '''
    return {
        'first_block': first_block,
        'last_block': last_block
    }
    '''


def parent_recording_id(tiffname_or_thorimage_id):
    """Returns recording id for recording subrecording was derived from.

    Input can be a TIFF filename or recording id.
    """
    last_part = split(tiffname_or_thorimage_id)[1]
    match = re.search(shared_subrecording_regex, last_part)
    if match is None:
        raise ValueError('not a subrecording')
    return match.group(1)
        

def accepted_blocks(analysis_run_at, verbose=False):
    """
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    if verbose:
        print('entering accepted_blocks')

    analysis_run_at = pd.Timestamp(analysis_run_at)
    presentations = pd.read_sql_query('SELECT presentation_id, ' +
        'comparison, presentation_accepted FROM presentations WHERE ' +
        "analysis = '{}'".format(analysis_run_at), conn,
        index_col='comparison')
    # TODO any of stuff below behave differently if index is comparison
    # (vs. default range index)? groupby('comparison')?

    analysis_run = pd.read_sql_query('SELECT accepted, input_filename, ' +
        "recording_from FROM analysis_runs WHERE run_at = '{}'".format(
        analysis_run_at), conn)
    assert len(analysis_run) == 1
    analysis_run = analysis_run.iloc[0]
    recording_from = analysis_run.recording_from
    input_filename = analysis_run.input_filename
    all_blocks_accepted = analysis_run.accepted

    # TODO TODO make sure block bounds are loaded into db from gui first, if
    # they changed in the gsheet. otherwise, will be stuck using old values, and
    # this function will not behave correctly
    # TODO TODO this has exactly the same problem canonical_segmentation
    # currently has: only one of each *_block per recording start time =>
    # sub-recordings will clobber each other. fix!
    # (currently just working around w/ subrecording tif filename hack)
    recording = pd.read_sql_query('SELECT thorimage_path, first_block, ' +
        "last_block FROM recordings WHERE started_at = '{}'".format(
        recording_from), conn)

    assert len(recording) == 1
    recording = recording.iloc[0]

    # TODO delete this if not going to use it to calculate uploaded_block_info
    if len(presentations) > 0:
        presentations_with_responses = pd.read_sql_query('SELECT ' +
            'presentation_id FROM responses WHERE segmentation_run = ' +
            "'{}'".format(analysis_run_at), conn)
        # TODO faster to check isin if this is a set?
    #

    # TODO TODO implement some kind of handling of sub-recordings in db
    # and get rid of this hack
    #print(input_filename)
    if is_subrecording_tiff(input_filename):
        first_block, last_block = subrecording_tiff_blocks(input_filename)

        if verbose:
            print(input_filename, 'belonged to a sub-recording')

    else:
        if recording.last_block is None or recording.first_block is None:
            # TODO maybe generate it in this case?
            raise ValueError(('no block info in db for recording_from = {} ({})'
                ).format(recording_from, recording.thorimage_path))

        first_block = recording.first_block
        last_block = recording.last_block

    n_blocks = last_block - first_block + 1
    expected_comparisons = list(range(n_blocks))

    # TODO delete these prints. for debugging.
    if verbose:
        print('presentations:', presentations)
        print('expected_comparisons:', expected_comparisons)
        print('all_blocks_accepted:', all_blocks_accepted)
    #
    # TODO TODO TODO check that upload will keep comparison numbered as blocks
    # are, so that missing comparisons numbers can be imputed with False here
    # (well, comparison numbering should probably start w/ 0 at first_block)

    # TODO TODO test cases where not all blocks were uploaded at all, where some
    # where not uploaded and some are explicitly marked not accepted, and where
    # all blocks rejected are explicitly so

    if pd.notnull(all_blocks_accepted):
        fill_value = all_blocks_accepted
    else:
        fill_value = False

    def block_accepted(presentation_df):
        null = pd.isnull(presentation_df.presentation_accepted)
        if null.any():
            assert null.all()
            return fill_value

        accepted = presentation_df.presentation_accepted
        if accepted.any():
            assert accepted.all()
            return True
        else:
            return False

    '''
    null_presentation_accepted = \
        pd.isnull(presentations.presentation_accepted)
    if null_presentation_accepted.any():
        if verbose:
            print('at least one presentation was null')
        # TODO fix db w/ a script or just always check for old way of doing it?
        # fixing db would mean translating all analysis_runs.accepted into
        # presentations.presentation_accepted and then deleting
        # analysis_runs.accepted column

        assert null_presentation_accepted.all(), 'not all null'
        assert not pd.isnull(all_blocks_accepted),'all_blocks_accepted null'

        if all_blocks_accepted:
            accepted = [True] * n_blocks
        else:
            accepted = [False] * n_blocks
    else:
        if verbose:
            print('no presentations were null')
    '''
    # TODO make sure sorted by comparison #. groupby ensure that?
    accepted = presentations.groupby('comparison'
        ).agg(block_accepted).presentation_accepted
    accepted.name = 'comparison_accepted'
    assert len(accepted.shape) == 1, 'accepted was not a Series'

    if verbose:
        print('accepted before filling missing values:', accepted)

    if (((accepted == True).any() and all_blocks_accepted == False) or
        ((accepted == False).any() and all_blocks_accepted)):
        # TODO maybe just correct db in this case?
        # (set analysis_run.accepted to null and keep presentation_accepted
        # if inconsistent / fill them from analysis_run.accepted if missing)
        #raise ValueError('inconsistent accept labels')
        warnings.warn('inconsistent accept labels. ' +
            'nulling analysis_runs.accepted in corresponding row.')

        # TODO TODO test this!
        sql = ('UPDATE presentations SET presentation_accepted = {} WHERE ' +
            "analysis = '{}' AND presentation_accepted IS NULL").format(
            fill_value, analysis_run_at)
        ret = conn.execute(sql)
        # TODO if i'm gonna call this multiple times, maybe just factor it into
        # a fn
        presentations_after_update = pd.read_sql_query(
            'SELECT presentation_id, ' +
            'comparison, presentation_accepted FROM presentations WHERE ' +
            "analysis = '{}'".format(analysis_run_at), conn,
            index_col='comparison')
        if verbose:
            print('Presentations after filling w/ all_blocks_accepted:')
            print(presentations_after_update)

        sql = ('UPDATE analysis_runs SET accepted = NULL WHERE run_at = ' +
            "'{}'").format(analysis_run_at)
        ret = conn.execute(sql)

    # TODO TODO TODO are this case + all_blocks_accepted=False case in if
    # above the only two instances where the block info is not uploaded (or
    # should be, assuming no accept of non-uploaded experiment)
    for c in expected_comparisons:
        if c not in accepted.index:
            accepted.loc[c] = fill_value

    accepted = accepted.to_list()

    # TODO TODO TODO also calculate and return uploaded_block_info
    # based on whether a given block has (all) of it's presentations and
    # responses entries (whether accepted or not)
    if verbose:
        print('leaving accepted_blocks\n')
    return accepted


def print_all_accepted_blocks():
    """Just for testing behavior of accepted_blocks fn.
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    analysis_runs = pd.read_sql_query('SELECT run_at FROM segmentation_runs',
        conn).run_at

    for r in analysis_runs:
        try:
            print('{}: {}'.format(r, accepted_blocks(r)))
        except ValueError as e:
            print(e)
            continue
        #import ipdb; ipdb.set_trace()

    import ipdb; ipdb.set_trace()


def _xmlroot(xml_path):
    return etree.parse(xml_path).getroot()


# TODO maybe rename to exclude get_ prefix, to be consistent w/
# thorimage_dir(...) and others above?
def get_thorimage_xml_path(thorimage_dir):
    """Takes ThorImage output dir to path to its XML output.
    """
    return join(thorimage_dir, 'Experiment.xml')


def get_thorimage_xmlroot(thorimage_dir):
    """Takes ThorImage output dir to object w/ XML data.
    """
    xml_path = get_thorimage_xml_path(thorimage_dir)
    return _xmlroot(xml_path)


def get_thorimage_time_xml(xml):
    """Takes etree XML root object to recording start time.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    date_ele = xml.find('Date')
    from_date = datetime.strptime(date_ele.attrib['date'], '%m/%d/%Y %H:%M:%S')
    from_utime = datetime.fromtimestamp(float(date_ele.attrib['uTime']))
    assert (from_date - from_utime).total_seconds() < 1
    return from_utime


def get_thorimage_time(thorimage_dir, use_mtime=False):
    """Takes ThorImage directory to recording start time (from XML).
    """
    xml_path = get_thorimage_xml_path(thorimage_dir)

    # TODO delete. for debugging matching.
    '''
    xml = _xmlroot(xml_path)
    print(thorimage_dir)
    print(get_thorimage_time_xml(xml))
    print(datetime.fromtimestamp(getmtime(xml_path)))
    print('')
    '''
    #
    if not use_mtime:
        xml = _xmlroot(xml_path)
        return get_thorimage_time_xml(xml)
    else:
        return datetime.fromtimestamp(getmtime(xml_path))


def get_thorsync_time(thorsync_dir):
    """Returns modification time of ThorSync XML.

    Not perfect, but it doesn't seem any ThorSync outputs have timestamps.
    """
    syncxml = join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return datetime.fromtimestamp(getmtime(syncxml))


def get_thorimage_dims_xml(xml):
    """Takes etree XML root object to (xy, z, c) dimensions of movie.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm_attribs = xml.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x, y)

    # what is Streaming -> flybackLines? (we already have flybackFrames...)

    # TODO maybe either subtract flyback frames here or return that separately?
    # (for now, just leaving it out of this fn)
    z = int(xml.find('ZStage').attrib['steps'])
    # may break on my existing single plane data if value of z not meaningful
    # there

    if z != 1:
        streaming = xml.find('Streaming')
        assert streaming.attrib['enable'] == '1'
        # Not true, see: 2020-03-09/1/fn (though another run of populate_db.py
        # seemed to indicate this dir was passed over for tiff creation
        # anyway??? i'm confused...)
        #assert streaming.attrib['zFastEnable'] == '1'
        if streaming.attrib['zFastEnable'] != '1':
            z = 1

    # still not sure how this is encoded in the XML...
    c = None

    # may want to add ZStage -> stepSizeUM to TIFF metadata?

    return xy, z, c


def get_thorimage_pixelsize_xml(xml):
    """Takes etree XML root object to XY pixel size in um.

    Pixel size in X is the same as pixel size in Y.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    # TODO does thorimage (and their xml) even support unequal x and y?
    # TODO support z here?
    return float(xml.find('LSM').attrib['pixelSizeUM'])


def get_thorimage_fps_xml(xml):
    """Takes etree XML root object to (after-any-averaging) fps of recording.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm_attribs = xml.find('LSM').attrib
    raw_fps = float(lsm_attribs['frameRate'])
    # TODO is this correct handling of averageMode?
    average_mode = int(lsm_attribs['averageMode'])
    if average_mode == 0:
        n_averaged_frames = 1
    else:
        # TODO TODO TODO does this really not matter in the volumetric streaming
        # case, as Remy said? (it's still displayed in the software, and still
        # seems enabled in that averageMode=1...)
        # (yes it does seem to matter TAKE INTO ACCOUNT!)
        n_averaged_frames = int(lsm_attribs['averageNum'])
    saved_fps = raw_fps / n_averaged_frames
    return saved_fps


def get_thorimage_fps(thorimage_directory):
    """Takes ThorImage dir to (after-any-averaging) fps of recording.
    """
    xml = get_thorimage_xmlroot(thorimage_directory)
    return get_thorimage_fps_xml(xml)


# TODO maybe delete / refactor to use fns above
def tif2xml_root(filename):
    """Returns etree root of ThorImage XML settings from TIFF filename.

    Path can be to analysis output directory, as long as raw data directory
    exists.
    """
    if filename.startswith(analysis_output_root()):
        filename = filename.replace(analysis_output_root(), raw_data_root())

    parts = filename.split(sep)
    thorimage_id = '_'.join(parts[-1].split('_')[:-1])

    xml_fname = sep.join(parts[:-2] + [thorimage_id, 'Experiment.xml'])
    return _xmlroot(xml_fname)


# TODO TODO rename this one to make it clear why it's diff from above
# + how to use it (or just delete one...)
def fps_from_thor(df):
    """Takes a DataFrame and returns fps from ThorImage XML.
    
    df must have a thorimage_dir column (that can be either a relative or
    absolute path, as long as it's under raw_data_root)

    Only the path in the first row is used.
    """
    # TODO assert unique first?
    thorimage_dir = df['thorimage_path'].iat[0]
    # TODO maybe factor into something that ensures path has a certain prefix
    # that maybe also validates right # parts?
    thorimage_dir = join(raw_data_root(), *thorimage_dir.split('/')[-3:])
    fps = get_thorimage_fps(thorimage_dir)
    return fps


def cnmf_metadata_from_thor(filename):
    """Takes TIF filename to key settings from XML needed for CNMF.
    """
    xml_root = tif2xml_root(filename)
    fps = get_thorimage_fps_xml(xml_root)
    # "spatial resolution of FOV in pixels per um" "(float, float)"
    # TODO do they really mean pixel/um, not um/pixel?
    pixels_per_um = 1 / get_thorimage_pixelsize_xml(xml_root)
    dxy = (pixels_per_um, pixels_per_um)
    # TODO maybe load dims anyway?
    return {'fr': fps, 'dxy': dxy}


def get_thorimage_n_flyback_xml(xml):
    streaming = xml.find('Streaming')
    assert streaming.attrib['enable'] == '1'
    if streaming.attrib['zFastEnable'] == '1':
        n_flyback_frames = int(streaming.attrib['flybackFrames'])
    else:
        # may fail in non-streaming volume case? though might then also not want
        # to assert first streaming object is enabled?
        assert z == 1
        n_flyback_frames = 0

    return n_flyback_frames


def load_thorimage_metadata(thorimage_directory, return_xml=False):
    """Returns (fps, xy, z, c, n_flyback, raw_output_path) for ThorImage dir.

    Returns xml as an additional final return value if `return_xml` is True.
    """
    xml = get_thorimage_xmlroot(thorimage_directory)

    fps = get_thorimage_fps_xml(xml)
    xy, z, c = get_thorimage_dims_xml(xml)

    n_flyback_frames = get_thorimage_n_flyback_xml(xml)

    # So far, I have seen this be one of:
    # - Image_0001_0001.raw
    # - Image_001_001.raw
    # ...but not sure if there any meaning behind the differences.
    imaging_files = glob.glob(join(thorimage_directory, 'Image_*.raw'))
    assert len(imaging_files) == 1, 'multiple possible imaging files'
    imaging_file = imaging_files[0]

    if not return_xml:
        return fps, xy, z, c, n_flyback_frames, imaging_file
    else:
        return fps, xy, z, c, n_flyback_frames, imaging_file, xml


# TODO rename to indicate a thor (+raw?) format
def read_movie(thorimage_dir, discard_flyback=True):
    """Returns (t,[z,]x,y) indexed timeseries as a numpy array.
    """
    fps, xy, z, c, n_flyback, imaging_file, xml = \
        load_thorimage_metadata(thorimage_dir, return_xml=True)

    x, y = xy

    # From ThorImage manual: "unsigned, 16-bit, with little-endian byte-order"
    dtype = np.dtype('<u2')

    with open(imaging_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)

    # TODO maybe just don't read the data known to be flyback frames?

    n_frame_pixels = x * y
    n_frames = len(data) // n_frame_pixels
    assert len(data) % n_frame_pixels == 0, 'apparent incomplete frames'

    # This does not fail in the volumetric case, because 'frames' here
    # refers to XY frames there too.
    assert n_frames == int(xml.find('Streaming').attrib['frames'])

    # TODO how to reshape if there are also multiple channels?

    if z > 0:
        # TODO TODO some way to set pockel power to zero during flyback frames?
        # not sure why that data is even wasting space in the file...

        z_total = z + n_flyback
        n_frames, remainder = divmod(n_frames, z_total)
        assert remainder == 0
        # TODO check this against method by reshaping as before and slicing w/
        # appropriate strides [+ concatenating?]
        data = np.reshape(data, (n_frames, z_total, x, y))

        if discard_flyback:
            data = data[:, :z, :, :]
    else:
        data = np.reshape(data, (n_frames, x, y))

    return data


def write_tiff(tiff_filename, movie):
    """Write a TIFF loading the same as the TIFFs we create with ImageJ.

    TIFFs are written in big-endian byte order to be readable by `imread_big`
    from MATLAB file exchange.

    Dimensions of input should be (t,[z,],x,y).

    Metadata may not be correct.
    """
    import tifffile

    dtype = movie.dtype
    if not (dtype.itemsize == 2 and
        np.issubdtype(dtype, np.unsignedinteger)):

        raise ValueError('movie must have uint16 dtype')

    if dtype.byteorder == '|':
        raise ValueError('movie must have explicit endianness')

    # If little-endian, convert to big-endian before saving TIFF, almost
    # exclusively for the benefit of MATLAB imread_big, which doesn't seem
    # able to discern the byteorder.
    if (dtype.byteorder == '<' or
        (dtype.byteorder == '=' and sys.byteorder == 'little')):
        movie = movie.byteswap().newbyteorder()
    else:
        assert dtype.byteorder == '>'

    # TODO TODO maybe change so ImageJ considers appropriate dimension the time
    # dimension (both in 2d x T and 3d x T cases)
    
    # TODO actually make sure any metadata we use is the same
    # TODO maybe just always do test from test_readraw here?
    # (or w/ flag to disable the check)
    tifffile.imsave(tiff_filename, movie, imagej=True)


def full_frame_avg_trace(movie):
    """Takes a (t,[z,]x,y) movie to t-length vector of frame averages.
    """
    # Averages all dims but first, which is assumed to be time.
    return np.mean(movie, axis=tuple(range(1, movie.ndim)))


def crop_to_coord_bbox(matrix, coords, margin=0):
    """Returns matrix cropped to bbox of coords and bounds.
    """
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    assert x_min >= 0 and y_min >= 0, \
        f'mins must be >= 0 (x_min={x_min}, y_min={y_min})'

    # TODO might need to fix this / fns that use this such a that 
    # coord limits are actually < matrix dims, rather than <=
    '''
    assert x_max < matrix.shape[0] and y_max < matrix.shape[1], \
        (f'maxes must be < matrix shape = {matrix.shape} (x_max={x_max}' +
        f', y_max={y_max}')
    '''
    assert x_max <= matrix.shape[0] and y_max <= matrix.shape[1], \
        (f'maxes must be <= matrix shape = {matrix.shape} (x_max={x_max}' +
        f', y_max={y_max}')

    # Keeping min at 0 to prevent slicing error in that case
    # (I think it will be empty, w/ -1:2, for instance)
    # Capping max not necessary to prevent err, but to make behavior of bounds
    # consistent on both edges.
    x_min = max(0, x_min - margin)
    x_max = min(x_max + margin, matrix.shape[0] - 1)
    y_min = max(0, y_min - margin)
    y_max = min(y_max + margin, matrix.shape[1] - 1)

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


def crop_to_nonzero(matrix, margin=0):
    """
    Returns a matrix just large enough to contain the non-zero elements of the
    input, and the bounding box coordinates to embed this matrix in a matrix
    with indices from (0,0) to the max coordinates in the input matrix.
    """
    coords = np.argwhere(matrix > 0)
    return crop_to_coord_bbox(matrix, coords, margin=margin)


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


def db_footprints2array(df, shape):
    """Returns footprints in an array of dims (shape + (n_footprints,)).
    """
    return np.stack([db_row2footprint(r, shape) for _, r in df.iterrows()],
        axis=-1)


# TODO test w/ mpl / cv2 contours that never see ij to see if transpose is
# necessary!
def contour2mask(contour, shape):
    """Returns a boolean mask True inside contour and False outside.
    """
    import cv2
    # TODO any checking of contour necessary for it to be well behaved in
    # opencv?
    mask = np.zeros(shape, np.uint8)
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


def ijrois2masks(ijrois, shape):
    """
    Transforms ROIs loaded from my ijroi fork to an array full of boolean masks,
    of dimensions (shape + (n_rois,)).
    """
    # TODO maybe index final pandas thing by ijroi name (before .roi prefix)
    # (or just return np array indexed as CNMF "A" is) (?)
    if len(shape) == 2:
        # TODO TODO why was i reversing the shape again...? maybe i shouldn't?
        masks = [imagej2py_coords(contour2mask(c, shape[::-1]))
            for _, c in ijrois
        ]
        # This concatenates along the last element of the new shape
        masks = np.stack(masks, axis=-1)

    # TODO TODO TODO actually make sure this works in the len(shape) == 3 case
    elif len(shape) == 3:
        # NOTE: only supporting case of one volumetric mask for now
        # (probably make dimension that used to correspond to n_footprints
        # singleton here)
        mask = np.zeros(shape, np.uint8).astype('bool')
        xy_shape = shape[1:]
        # to not have to worry about reversing this part of shape for now
        assert xy_shape[0] == xy_shape[1]
        index_set = set()
        for name, contour in ijrois:
            # Uses automatic ROI naming convention from TIFFs created by my
            # write_tiff (where what is really Z seems to be treated as C by
            # ImageJ)
            z_index = int(name.split('-')[0]) - 1
            index_set.add(z_index)

            # TODO may also need to reverse part of shape here, if really was
            # necessary above (test would probably need to be in asymmetric
            # case...)
            z_mask = imagej2py_coords(contour2mask(contour, xy_shape))
            mask[z_index, :, :] = z_mask

        assert index_set == set(range(shape[0])), 'some z slices missing ijrois'
        masks = np.stack([mask], axis=-1)
    else:
        raise ValueError('shape must have length 2 or 3')

    # TODO maybe normalize here?
    # (and if normalizing, might as well change dtype to match cnmf output?)
    # and worth casting type to bool, rather than keeping 0/1 uint8 array?
    return masks


# TODO rename these two to indicate it only works on images (not coordinates)
# TODO and make + use fns that operate on coordinates?
def imagej2py_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
    # TODO how does this behave in the 3d case...?
    # still what i want, or should i exclude the z dimension somehow?
    return array.T


def py2imagej_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
    return array.T


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


def extract_traces_boolean_footprints(movie, footprints,
    footprint_framenums=None, verbose=True):
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


def exp_decay(t, scale, tau, offset):
    # TODO is this the usual definition of tau (as in RC time constant?)
    return scale * np.exp(-t / tau) + offset


# TODO call for each odor onset (after fixed onset period?)
# est onset period? est rise kinetics jointly? how does cnmf do it?
def fit_exp_decay(signal, sampling_rate=None, times=None, numerical_scale=1.0):
    """Returns fit parameters for an exponential decay in the input signal.

    Args:
        signal (1 dimensional np.ndarray): time series, beginning at decay onset
        sampling_rate (float): sampling rate in Hz
    """
    from scipy.optimize import curve_fit

    if sampling_rate is None and times is None:
        raise ValueError('pass either sampling_rate or times as keyword arg')

    if sampling_rate is not None:
        sampling_interval = 1 / sampling_rate
        n_samples = len(signal)
        end_time = n_samples * sampling_interval
        times = np.linspace(0, end_time, num=n_samples, endpoint=True)

    # TODO make sure input is not modified here. copy?
    signal = signal * numerical_scale

    # TODO constrain params somehow? for example, so scale stays positive
    popt, pcov = curve_fit(exp_decay, times, signal,
        p0=(1.8 * numerical_scale, 5.0, 0.0 * numerical_scale))

    # TODO is this correct to scale after converting variance to stddev?
    sigmas = np.sqrt(np.diag(pcov))
    sigmas[0] = sigmas[0] / numerical_scale
    # skipping tau, which shouldn't need to change (?)
    sigmas[2] = sigmas[2] / numerical_scale

    # TODO only keep this if signal is modified s.t. it affects calling fn.
    # in this case, maybe still just copy above?
    signal = signal / numerical_scale

    scale, tau, offset = popt
    return (scale / numerical_scale, tau, offset / numerical_scale), sigmas


def latest_analysis(verbose=False):
    global conn
    if conn is None:
        conn = get_db_conn()

    # TODO sql based command to get analysis info for stuff that has its
    # timestamp in segmentation_runs, to condense these calls to one?
    seg_runs = pd.read_sql_query('SELECT run_at FROM segmentation_runs',
        conn)
    analysis_runs = pd.read_sql('analysis_runs', conn)
    seg_runs = seg_runs.merge(analysis_runs)

    seg_runs.input_filename = seg_runs.input_filename.apply(lambda t:
        t.split('mb_team/analysis_output/')[1])

    # TODO TODO change all path handling to be relative to NAS_PREFIX.
    # don't store absolute paths (or if i must, also store what prefix is?)
    input_tifs = seg_runs.input_filename.unique()
    has_subrecordings = set()
    key2tiff = dict()
    # TODO decide whether i want this to be parent_id or thorimage_id
    # maybe make a kwarg flag to this fn to switch between them
    tiff2parent_id = dict()
    tiff_is_subrecording = dict()
    for tif in input_tifs:
        if verbose:
            print(tif, end='')

        date_fly_keypart = '/'.join(tif.split('/')[:2])
        thorimage_id = tiff_thorimage_id(tif)
        key = '{}/{}'.format(date_fly_keypart, thorimage_id)
        # Assuming this is 1:1 for now.
        key2tiff[key] = tif
        try:
            parent_id = parent_recording_id(tif)
            tiff_is_subrecording[tif] = True
            parent_key = '{}/{}'.format(date_fly_keypart, parent_id)
            has_subrecordings.add(parent_key)
            tiff2parent_id[tif] = parent_id

            if verbose:
                print(' (sub-recording of {})'.format(parent_key))

        # This is triggered if tif is not a sub-recording.
        except ValueError:
            tiff_is_subrecording[tif] = False
            tiff2parent_id[tif] = thorimage_id
            if verbose:
                print('')

    nonoverlapping_input_tifs = set(t for k, t in key2tiff.items()
                                 if k not in has_subrecordings)
    # set(input_tifs) - nonoverlapping_input_tifs

    # TODO if verbose, maybe also print stuff tossed for having subrecordings
    # as well as # rows tossed for not being accepted / stuff w/o analysis /
    # stuff w/o anything accepted

    # TODO between this and the above, make sure to handle (ignore) stuff that
    # doesn't have any analysis done.
    seg_runs = seg_runs[seg_runs.input_filename.isin(nonoverlapping_input_tifs)]

    # TODO TODO and if there are disjoint sets of accepted blocks, would ideally
    # return something indicating which analysis to get which block from?  would
    # effectively have to search per block/comparison, right?
    # TODO would ideally find the latest analysis that has the maximal
    # number of blocks accepted (per input file) (assuming just going to return
    # one analysis version per tif, rather than potentially a different one for
    # each block)
    seg_runs['n_accepted_blocks'] = seg_runs.run_at.apply(lambda r:
        sum(accepted_blocks(r)))
    accepted_runs = seg_runs[seg_runs.n_accepted_blocks > 0]

    latest_tif_analyses = accepted_runs.groupby('input_filename'
        ).run_at.max().to_frame()
    latest_tif_analyses['is_subrecording'] = \
        latest_tif_analyses.index.map(tiff_is_subrecording)

    subrec_blocks = latest_tif_analyses.apply(subrecording_tiff_blocks_df,
        axis=1, result_type='expand')
    latest_tif_analyses[['first_block','last_block']] = subrec_blocks

    latest_tif_analyses['thorimage_id'] = \
        latest_tif_analyses.index.map(tiff2parent_id)

    # TODO what format would make the most sense for the output?
    # which index? just trimmed input_filename? recording_from (+ block /
    # comparison)? (fly, date, [thorimage_id? recording_from?] (+ block...)
    # ?
    return latest_tif_analyses


def sql_timestamp_list(df):
    """
    df must have a column run_at, that is a pandas Timestamp type
    """
    timestamp_list = '({})'.format(', '.join(
        ["'{}'".format(x) for x in df.run_at]
    ))
    return timestamp_list


# TODO w/ this or a separate fn using this, print what we have formatted
# roughly like in data_tree in gui, so that i can check it against the gui
def latest_analysis_presentations(analysis_run_df):
    global conn
    if conn is None:
        conn = get_db_conn()

    # TODO maybe compare time of this to getting all and filtering locally
    # TODO at least once, compare the results of this to filtering locally
    # IS NOT DISTINCT FROM should also 
    presentations = pd.read_sql_query('SELECT * FROM presentations WHERE ' +
        '(presentation_accepted = TRUE OR presentation_accepted IS NULL) ' +
        'AND analysis IN ' + sql_timestamp_list(analysis_run_df), conn)

    # TODO TODO maybe just do a migration on the db to fix all comparisons
    # to not have to be renumbered, and fix gui(+populate_db?) so they don't
    # restart numbering across sub-recordings that come from same recording?

    # TODO TODO TODO test that this is behaving as expected
    # - is there only one place where presentatinos.analysis == row.run_at?
    #   assert that?
    # - might the sample things ever get incremented twice?
    for row in analysis_run_df[analysis_run_df.is_subrecording].itertuples():
        run_presentations = (presentations.analysis == row.run_at)
        presentations.loc[run_presentations, 'comparison'] = \
            presentations[run_presentations].comparison + int(row.first_block)

        # TODO check that these rows are also complete / valid

    # TODO use those check fns on these presentations, to make sure they are
    # full blocks and stuff

    # TODO ultimately, i want all of these latest_* functions to return a
    # dataframe without an analysis column (still return it, just in case it
    # becomes necessary later?)
    # (or at least i want to make sure that the other index columns can uniquely
    # refer to something, s.t. adding analysis to a drop_duplicates call does
    # not change the total # of returned de-duplicated rows)
    # TODO which index cols again?

    return presentations


def latest_analysis_footprints(analysis_run_df):
    global conn
    if conn is None:
        conn = get_db_conn()

    footprints = pd.read_sql_query(
        'SELECT * FROM cells WHERE segmentation_run IN ' +
        sql_timestamp_list(analysis_run_df), conn)
    return footprints


def latest_analysis_traces(df):
    """
    Input DataFrame must have a presentation_id column matching that in the db.
    This way, presentations already filtered to be the latest just get their
    responses assigned too them.
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    responses = pd.read_sql_query(
        'SELECT * FROM responses WHERE presentation_id IN ' +
        '({})'.format(','.join([str(x) for x in df.presentation_id])), conn)
    # responses should by larger by a factor of # cells within each analysis run
    assert len(df) == len(responses.presentation_id.unique())
    return responses
    

response_stat_cols = [
    'exp_scale',
    'exp_tau',
    'exp_offset',
    'exp_scale_sigma',
    'exp_tau_sigma',
    'exp_offset_sigma',
    'avg_dff_5s',
    'avg_zchange_5s'
]
def latest_response_stats(*args):
    """
    """
    global conn
    if conn is None:
        conn = get_db_conn()

    index_cols = [
        'prep_date',
        'fly_num',
        'recording_from',
        'analysis',
        'comparison',
        'odor1',
        'odor2',
        'repeat_num'
    ]
    # TODO maybe just get cols db has and exclude from_onset or something?
    # just get all?
    presentation_cols_to_get = index_cols + response_stat_cols
    if len(args) == 0:
        db_presentations = pd.read_sql('presentations', conn,
            columns=(presentation_cols_to_get + ['presentation_id']))

    elif len(args) == 1:
        db_presentations = args[0]

    else:
        raise ValueError('too many arguments. expected 0 or 1')

    referenced_recordings = set(db_presentations['recording_from'].unique())

    if len(referenced_recordings) == 0:
        return

    db_analysis_runs = pd.read_sql('analysis_runs', conn,
        columns=['run_at', 'recording_from', 'accepted'])
    db_analysis_runs.set_index(['recording_from', 'run_at'],
        inplace=True)

    # Making sure not to get multiple presentation entries referencing the same
    # real presentation in any single recording.
    presentation_stats = []
    for r in referenced_recordings:
        # TODO are presentation->recording and presentation->
        # analysis_runs->recording inconsistent somehow?
        # TODO or is this an insertion order thing? rounding err?
        # maybe set_index is squashing stuff?
        # TODO maybe just stuff i'm skipping now somehow?

        # TODO TODO merge db_analysis_runs w/ recordings to get
        # thorimage_dir / id for troubleshooting?
        # TODO fix and delete try / except
        try:
            rec_analysis_runs = db_analysis_runs.loc[(r,)]
        except KeyError:
            # TODO should this maybe be an error?
            '''
            print(db_analysis_runs)
            print(referenced_recordings)
            print(r)
            import ipdb; ipdb.set_trace()
            '''
            warnings.warn('referenced recording not in analysis_runs!')
            continue

        # TODO TODO TODO switch to using presentations.presentation_accepted
        raise NotImplementedError
        rec_usable = rec_analysis_runs.accepted.any()

        rec_presentations = db_presentations[
            db_presentations.recording_from == r]

        # TODO maybe use other fns here to check it has all repeats / full
        # comparisons?

        for g, gdf in rec_presentations.groupby(
            ['comparison', 'odor1', 'odor2', 'repeat_num']):

            # TODO rename (maybe just check all response stats at this point...)
            # maybe just get most recent row that has *any* of them?
            # (otherwise would have to combine across rows...)
            has_exp_fit = gdf[gdf.exp_scale.notnull()]

            # TODO compute if no response stats?
            if len(has_exp_fit) == 0:
                continue

            most_recent_fit_idx = has_exp_fit.analysis.idxmax()
            most_recent_fit = has_exp_fit.loc[most_recent_fit_idx].copy()

            assert len(most_recent_fit.shape) == 1

            # TODO TODO TODO switch to using presentations.presentation_accepted
            raise NotImplementedError
            most_recent_fit['accepted'] = rec_usable

            # TODO TODO TODO clean up older fits on same data?
            # (delete from database)
            # (if no dependent objects...)
            # probably somewhere else...?

            presentation_stats.append(most_recent_fit.to_frame().T)

    if len(presentation_stats) == 0:
        return

    presentation_stats_df = pd.concat(presentation_stats, ignore_index=True)

    # TODO just convert all things that look like floats?

    for c in response_stat_cols:
        presentation_stats_df[c] = presentation_stats_df[c].astype('float64')

    for date_col in ('prep_date', 'recording_from'):
        presentation_stats_df[date_col] = \
            pd.to_datetime(presentation_stats_df[date_col])

    return presentation_stats_df


def n_expected_repeats(df):
    """Returns expected # repeats given DataFrame w/ repeat_num col.
    """
    max_repeat = df.repeat_num.max()
    return max_repeat + 1


# TODO TODO could now probably switch to using block metadata in recording table
# (n_repeats should be in there)
def missing_repeats(df, n_repeats=None):
    """
    Requires at least recording_from, comparison, name1, name2, and repeat_num
    columns. Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO n_repeats default to 3 or None?
    if n_repeats is None:
        # TODO or should i require input is merged w/ recordings for stimuli
        # data file paths and then just load for n_repeats and stuff?
        n_repeats = n_expected_repeats(df)

    # Expect repeats to include {0,1,2} for 3 repeat experiments.
    expected_repeats = set(range(n_repeats))

    repeat_cols = []
    opt_repeat_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_repeat_cols:
        if oc in df.columns:
            repeat_cols.append(oc)

    repeat_cols += [
        'recording_from',
        'comparison',
        'name1',
        'name2'#,
        #'log10_conc_vv1',
        #'log10_conc_vv2'
    ]
    # TODO some issue created by using float concs as a key?
    # TODO use odor ids instead?
    missing_repeat_dfs = []
    for g, gdf in df.groupby(repeat_cols):
        comparison_n_repeats = gdf.repeat_num.unique()

        no_extra_repeats = (gdf.repeat_num.value_counts() == 1).all()
        assert no_extra_repeats

        missing_repeats = [r for r in expected_repeats
            if r not in comparison_n_repeats]

        if len(missing_repeats) > 0:
            gmeta = gdf[repeat_cols].drop_duplicates().reset_index(drop=True)

        for r in missing_repeats:
            new_row = gmeta.copy()
            new_row['repeat_num'] = r
            missing_repeat_dfs.append(new_row)

    if len(missing_repeat_dfs) == 0:
        missing_repeats_df = \
            pd.DataFrame({r: [] for r in repeat_cols + ['repeat_num']})
    else:
        # TODO maybe merge w/ odor info so caller doesn't have to, if thats the
        # most useful for troubleshooting?
        missing_repeats_df = pd.concat(missing_repeat_dfs, ignore_index=True)

    missing_repeats_df.recording_from = \
        pd.to_datetime(missing_repeats_df.recording_from)

    # TODO should expected # blocks be passed in?

    return missing_repeats_df


def have_all_repeats(df, n_repeats=None):
    """
    Returns True if a recording has all blocks gsheet says it has, w/ full
    number of repeats for each. False otherwise.

    Requires at least recording_from, comparison, name1, name2, and repeat_num
    columns. Can also take prep_date, fly_num, thorimage_id.
    """
    missing_repeats_df = missing_repeats(df, n_repeats=n_repeats)
    if len(missing_repeats_df) == 0:
        return True
    else:
        return False


def missing_odor_pairs(df):
    """
    Requires at least recording_from, comparison, name1, name2 columns.
    Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO check that for each comparison, both A, B, and A+B are there
    # (3 combos of name1, name2, or whichever other odor ids)
    comp_cols = []
    opt_rec_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_rec_cols:
        if oc in df.columns:
            comp_cols.append(oc)

    comp_cols += [
        'recording_from',
        'comparison'
    ]

    odor_cols = [
        'name1',
        'name2'
    ]

    incomplete_comparison_dfs = []
    for g, gdf in df.groupby(comp_cols):
        comp_odor_pairs = gdf[odor_cols].drop_duplicates()
        if len(comp_odor_pairs) != 3:
            incomplete_comparison_dfs.append(gdf[comp_cols].drop_duplicates(
                ).reset_index(drop=True))

        # TODO generate expected combinations of name1,name2
        # TODO possible either odor not in db, in which case, would need extra
        # information to say which odor is actually missing... (would need
        # stimulus data)
        '''
        if len(missing_odor_pairs) > 0:
            gmeta = gdf[comp_cols].drop_duplicates().reset_index(drop=True)

        for r in missing_odor_pairs:
            new_row = gmeta.copy()
            new_row['repeat_num'] = r
            missing_odor_pair_dfs.append(new_row)
        '''

    if len(incomplete_comparison_dfs) == 0:
        incomplete_comparison_df = pd.DataFrame({r: [] for r in comp_cols})
    else:
        incomplete_comparison_df = \
            pd.concat(incomplete_comparison_dfs, ignore_index=True)

    incomplete_comparison_df.recording_from = \
        pd.to_datetime(incomplete_comparison_df.recording_from)

    return incomplete_comparison_df


def have_full_comparisons(df):
    """
    Requires at least recording_from, comparison, name1, name2 columns.
    Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO docstring
    if len(missing_odor_pairs(df)) == 0:
        return True
    else:
        return False


def skipped_comparison_nums(df):
    # TODO doc
    """
    Requires at least recording_from and comparison columns.
    Can also take prep_date, fly_num, and thorimage_id.
    """
    rec_cols = []
    opt_rec_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_rec_cols:
        if oc in df.columns:
            rec_cols.append(oc)

    rec_cols.append('recording_from')

    skipped_comparison_dfs = []
    for g, gdf in df.groupby(rec_cols):
        max_comp_num = gdf.comparison.max()
        min_comp_num = gdf.comparison.min()
        skipped_comp_nums = [x for x in range(min_comp_num, max_comp_num + 1)
            if x not in gdf.comparison]

        if len(skipped_comp_nums) > 0:
            gmeta = gdf[rec_cols].drop_duplicates().reset_index(drop=True)

        for c in skipped_comp_nums:
            new_row = gmeta.copy()
            new_row['comparison'] = c
            skipped_comparison_dfs.append(new_row)

    if len(skipped_comparison_dfs) == 0:
        skipped_comparison_df = pd.DataFrame({r: [] for r in
            rec_cols + ['comparison']})
    else:
        skipped_comparison_df = \
            pd.concat(skipped_comparison_dfs, ignore_index=True)

    # TODO move this out of each of these check fns, and put wherever this
    # columns is generated (in the way that required this cast...)
    skipped_comparison_df.recording_from = \
        pd.to_datetime(skipped_comparison_df.recording_from)

    return skipped_comparison_df


def no_skipped_comparisons(df):
    # TODO doc
    """
    Requires at least recording_from and comparison columns.
    Can also take prep_date, fly_num, and thorimage_id.
    """
    if len(skipped_comparison_nums(df)) == 0:
        return True
    else:
        return False


# TODO do i actually need this, or just call drop_orphaned/missing_...?
'''
def db_has_all_repeats():
    # TODO just read db and call have_all_repeats
    # TODO may need to merge stuff?
    raise NotImplementedError
'''


# TODO also check recording has as many blocks (in df / in db) as it's supposed
# to, given what the metadata + gsheet say


def drop_orphaned_presentations():
    # TODO only stuff that isn't also most recent response params?
    # TODO find presentation rows that don't have response row referring to them
    raise NotImplementedError


# TODO TODO maybe implement check fns above as wrappers around another fn that
# finds inomplete stuff? (check if len is 0), so that these fns can just wrap
# the same thing...
def drop_incomplete_presentations():
    raise NotImplementedError


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd
            integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman' flat window will produce a moving average
            smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of
    a string
    NOTE: length(output) != length(input), to correct this: return
    y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', " +
            "'hamming', 'bartlett', 'blackman'")


    # is this necessary?
    #s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    #y = np.convolve(w/w.sum(), s, mode='valid')
    # not sure what to change above to get this to work...

    y = np.convolve(w/w.sum(), x, mode='same')
    return y


# TODO finish translating. was directly translating matlab registration script
# to python.
"""
def motion_correct_to_tiffs(thorimage_dir, output_dir):
    # TODO only read this if at least one motion correction would be run
    movie = read_movie(thorimage_dir)

    # TODO do i really want to basically just copy the matlab version?
    # opportunity for some refactoring?

    output_subdir = 'tif_stacks'

    _, thorimage_id = split(thorimage_dir)

    rig_tif = join(output_dir, output_subdir, thorimage_id + '_rig.tif')
    avg_rig_tif = join(output_dir, output_subdir, 'AVG', 'rigid',
        'AVG{}_rig.tif'.format(thorimage_id))

    nr_tif = join(output_dir, output_subdir, thorimage_id + '_nr.tif')
    avg_nr_tif = join(output_dir, output_subdir, 'AVG', 'nonrigid',
        'AVG{}_nr.tif'.format(thorimage_id))

    need_rig_tif = not exist(rig_tif)
    need_avg_rig_tif = not exist(avg_rig_tif)
    need_nr_tif = not exist(nr_tif)
    need_avg_nr_tif = not exist(avg_nr_tif)

    if not (need_rig_tif or need_avg_rig_tif or need_nr_tif or need_avg_nr_tif):
        print('All registration already done.')
        return

    # Remy: this seems like it might just be reading in the first frame?
    ###Y = input_tif_path
    # TODO maybe can just directly use filename for python version though? raw
    # even?

    # rigid moco (normcorre)
    # TODO just pass filename instead of Y, and compute dimensions or whatever
    # separately, so that normcorre can (hopefully?) take up less memory
    if need_rig_tif:
        MC_rigid = MotionCorrection(Y)

        options_rigid = NoRMCorreSetParms('d1',MC_rigid.dims(1),
            'd2',MC_rigid.dims(2),
            'bin_width',50,
            'max_shift',15,
            'phase_flag', 1,
            'us_fac', 50,
            'init_batch', 100,
            'plot_flag', false,
            'iter', 2) 

        # TODO so is nothing actually happening in parallel?
        ## rigid moco
        MC_rigid.motionCorrectSerial(options_rigid)  # can also try parallel
        # TODO which (if any) of these do i still want?
        MC_rigid.computeMean()
        MC_rigid.correlationMean()
        #####MC_rigid.crispness()
        print('normcorre done')

        ## plot shifts
        #plt.plot(MC_rigid.shifts_x)
        #plt.plot(MC_rigid.shifts_y)

        # save .tif
        M = MC_rigid.M
        M = uint16(M) 
        tiffoptions.overwrite = true

        print(['saving tiff to ' rig_tif])
        saveastiff(M, rig_tif, tiffoptions)

    if need_avg_rig_tif:
        ##
        # save average image
        #AVG = single(mean(MC_rigid.M,3))
        AVG = single(MC_rigid.template)
        tiffoptions.overwrite = true

        print(['saving tiff to ' avg_rig_tif])
        saveastiff(AVG, avg_rig_tif, tiffoptions)

    if need_nr_tif:
        MC_nonrigid = MotionCorrection(Y)
        options_nonrigid = NoRMCorreSetParms('d1',MC_nonrigid.dims(1),
            'd2',MC_nonrigid.dims(2),
            'grid_size',[64,64],
            'mot_uf',4,
            'bin_width',50,
            'max_shift',[15 15],
            'max_dev',3,
            'us_fac',50,
            'init_batch',200,
            'iter', 2)

        MC_nonrigid.motionCorrectParallel(options_nonrigid)
        MC_nonrigid.computeMean()
        MC_nonrigid.correlationMean()
        MC_nonrigid.crispness()
        print('non-rigid normcorre done')

        # save .tif
        M = uint16(MC_nonrigid.M)
        tiffoptions.overwrite  = true
        print(['saving tiff to ' nr_tif])
        saveastiff(M, nr_tif, tiffoptions)

    if need_avg_nr_tif:
        # TODO flag to disable saving this average
        #AVG = single(mean(MC_nonrigid.M,3))
        AVG = single(MC_nonrigid.template)
        tiffoptions.overwrite = true
        print(['saving tiff to ' avg_nr_tif])
        saveastiff(AVG, avg_nr_tif, tiffoptions)

    raise NotImplementedError
"""

def cell_ids(df):
    """Takes a DataFrame with 'cell' in MultiIndex or columns to unique values.
    """
    if 'cell' in df.index.names:
        return df.index.get_level_values('cell').unique().to_series()
    elif 'cell' in df.columns:
        cids = pd.Series(data=df['cell'].unique(), name='cell')
        cids.index.name = 'cell'
        return cids
    else:
        raise ValueError("'cell' not in index or columns of DataFrame")


def matlabels(df, rowlabel_fn):
    """
    Takes DataFrame and function that takes one row of index to a label.

    `rowlabel_fn` should take a DataFrame row (w/ columns from index) to a str.
    """
    return df.index.to_frame().apply(rowlabel_fn, axis=1)


def format_odor_conc(name, log10_conc):
    """Takes `str` odor name and log10 concentration to a formatted `str`.
    """
    if log10_conc is None:
        return name
    else:
        # TODO tex formatting for exponent
        #return r'{} @ $10^{{'.format(name) + '{:.2f}}}$'.format(log10_conc)
        return '{} @ $10^{{{:.2f}}}$'.format(name, log10_conc)


def format_mixture(*args):
    """Returns `str` representing 2-component odor mixture.

    Input can be any of:
    - 2 `str` names
    - 2 names and concs (n1, n2, c1, c2)
    - a pandas.Series / dict with keys `name1`, `name2`, and (optionally)
      `log10_concvv<1/2>`
    """
    log10_c1 = None
    log10_c2 = None
    if len(args) == 2:
        n1, n2 = args
    elif len(args) == 4:
        n1, n2, log10_c1, log10_c2 = args
    elif len(args) == 1:
        row = args[0]

        # TODO maybe refactor to use this fn in plot_odor_corr fn too
        def single_var_with_prefix(prefix):
            single_var = None
            for v in row.keys():
                if type(v) is str and v.startswith(prefix):
                    if single_var is not None:
                        raise ValueError(f'multiple vars w/ prefix {prefix}')
                    single_var = v

            if single_var is None:
                raise KeyError(f'no vars w/ prefix {prefix}')

            return single_var
        #

        n1 = row[single_var_with_prefix('name1')]
        try:
            n2 = row[single_var_with_prefix('name2')]
        except KeyError:
            n2 = None

        # TODO maybe also use prefix fn here?
        if 'log10_conc_vv1' in row:
            log10_c1 = row['log10_conc_vv1']
            if n2 is not None:
                log10_c2 = row['log10_conc_vv2']
    else:
        raise ValueError('incorrect number of args')

    if n1 == 'paraffin':
        title = format_odor_conc(n2, log10_c2)
    elif n2 == 'paraffin' or n2 == 'no_second_odor' or n2 is None:
        title = format_odor_conc(n1, log10_c1)
    else:
        title = '{} + {}'.format(
            format_odor_conc(n1, log10_c1),
            format_odor_conc(n2, log10_c2)
        )

    return title


def split_odor_w_conc(row_or_str):
    try:
        odor_w_conc = row_or_str.odor_w_conc
        include_other_row_data = True
    except AttributeError:
        assert type(row_or_str) is str
        odor_w_conc = row_or_str
        include_other_row_data = False

    parts = odor_w_conc.split('@')
    assert len(parts) == 1 or len(parts) == 2
    if len(parts) == 1:
        log10_conc = 0.0
    else:
        log10_conc = float(parts[1])

    ret = {'name': parts[0].strip(), 'log10_conc_vv': log10_conc}
    if include_other_row_data:
        ret.update(row_or_str.to_dict())

    # TODO maybe only return series if include_other_row_data (rename if),
    # tuple/dict otherwise?
    return pd.Series(ret)


def format_keys(date, fly, *other_keys):
    date = date.strftime(date_fmt_str)
    fly = str(int(fly))
    others = [str(k) for k in other_keys]
    return '/'.join([date] + [fly] + others)


# TODO rename to be inclusive of cases other than pairs
def pair_ordering(comparison_df):
    """Takes a df w/ name1 & name2 to a dict of their tuples to order int.

    Order integers start at 0 and do not skip any numbers.
    """
    # TODO maybe assert only 3 combinations of name1/name2
    pairs = [(x.name1, x.name2) for x in
        comparison_df[['name1','name2']].drop_duplicates().itertuples()]

    # Will define the order in which odor pairs will appear, left-to-right,
    # in subplots.
    ordering = dict()

    # TODO maybe check that it's the second element specifically, since right
    # now, it's only cause paraffin is abbreviated to pfo (for name1 col)
    # that complex-mixture experiments go into first branch...
    has_paraffin = [p for p in pairs if 'paraffin' in p]
    if len(has_paraffin) == 0:
        import chemutils as cu
        assert {x[1] for x in pairs} == {'no_second_odor'}
        odors = [p[0] for p in pairs]

        # TODO change how odorset is identified so it can fail if none should be
        # detected / return None or something, then call back to just sorting
        # the odor names here, if no odor set name can be identified
        # (do we also want to support some case where original_name1 is defined
        # but the odorset name isn't necessarily?)
        if 'original_name1' in comparison_df.columns:
            original_name_order = df_to_odor_order(comparison_df)
            o2n = comparison_df[['original_name1','name1']].drop_duplicates(
                ).set_index('original_name1').name1
            # TODO maybe don't assume 'no_second_odor' like this (& below)?
            ordering = {(v, 'no_second_odor'): i for i, v in
                enumerate(o2n[original_name_order])}
        else:
            # TODO also support case where there isn't something we want to
            # stick at the end like this, for Matt's case
            last = None
            for o in odors:
                if cu.odor_is_mix(o):
                    if last is None:
                        last = o
                    else:
                        raise ValueError('multiple mixtures in odors to order')
            assert last is not None, 'expected a mix'
            ordering[(last, 'no_second_odor')] = len(odors) - 1
            
            i = 0
            for o in sorted(odors):
                if o == last:
                    continue
                ordering[(o, 'no_second_odor')] = i
                i += 1
    else:
        no_pfo = [p for p in pairs if 'paraffin' not in p]
        if len(no_pfo) < 1:
            raise ValueError('All pairs for this comparison had paraffin.' +
                ' Analysis error? Incomplete recording?')

        assert len(no_pfo) == 1
        last = no_pfo[0]
        ordering[last] = 2

        for i, p in enumerate(sorted(has_paraffin,
            key=lambda x: x[0] if x[1] == 'paraffin' else x[1])):

            ordering[p] = i

    # Checks that we order integers start at zero and don't skip anything.
    # Important for some ways of using them (e.g. to index axes array).
    assert {x for x in ordering.values()} == {x for x in range(len(ordering))}

    return ordering


def matshow(df, title=None, ticklabels=None, xticklabels=None,
    yticklabels=None, xtickrotation=None, colorbar_label=None,
    group_ticklabels=False, ax=None, fontsize=None, fontweight=None):
    # TODO TODO w/ all ticklabels kwargs, also support them being functions,
    # which operate on (what type exactly?) index rows
    # TODO shouldn't this get ticklabels from matrix if nothing else?
    # maybe at least in the case when both columns and row indices are all just
    # one level of strings?

    made_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        made_fig = True

    def one_level_str_index(index):
        return (len(index.shape) == 1 and
            all(index.map(lambda x: type(x) is str)))

    if (xticklabels is None) and (yticklabels is None):
        if ticklabels is None:
            if one_level_str_index(df.columns):
                xticklabels = df.columns
            else:
                xticklabels = None
            if one_level_str_index(df.index):
                yticklabels = df.index
            else:
                yticklabels = None
        else:
            # TODO maybe also assert indices are actually equal?
            assert df.shape[0] == df.shape[1]

            if callable(ticklabels):
                ticklabels = matlabels(df, ticklabels)

            xticklabels = ticklabels
            yticklabels = ticklabels
    else:
        # TODO fix. probably need to specify axes of df or something.
        # (maybe first modifying matlabels to accept that...)
        '''
        if callable(xticklabels):
            xticklabels = matlabels(df, xticklabels)

        if callable(yticklabels):
            yticklabels = matlabels(df, yticklabels)
        '''
        pass


    # TODO update this formula to work w/ gui corrs (too big now)
    if fontsize is None:
        fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))

    cax = ax.matshow(df)

    # just doing it in this case now to support kc_analysis use case
    # TODO put behind flag or something
    if made_fig:
        cbar = fig.colorbar(cax)

        if colorbar_label is not None:
            # rotation=270?
            cbar.ax.set_ylabel(colorbar_label)

        # TODO possible to provide facilities for colorbar in case when ax is
        # passed in? pass in another ax for colorbar? or just as easy to handle
        # outside in that case (probably)?

    def grouped_labels_info(labels):
        if not group_ticklabels or labels is None:
            return labels, 1, 0

        labels = pd.Series(labels)
        n_repeats = int(len(labels) / len(labels.unique()))

        # TODO TODO assert same # things from each unique element.
        # that's what this whole tickstep thing seems to assume.

        # Assumes order is preserved if labels are grouped at input.
        # May need to calculate some other way if not always true.
        labels = labels.unique()
        tick_step = n_repeats
        offset = n_repeats / 2 - 0.5
        return labels, tick_step, offset

    # TODO automatically only group labels in case where all repeats are
    # adjacent?
    # TODO make fontsize / weight more in group_ticklabels case?
    xticklabels, xstep, xoffset = grouped_labels_info(xticklabels)
    yticklabels, ystep, yoffset = grouped_labels_info(yticklabels)

    if xticklabels is not None:
        # TODO nan / None value aren't supported in ticklabels are they?
        # (couldn't assume len is defined if so)
        if xtickrotation is None:
            if all([len(x) == 1 for x in xticklabels]):
                xtickrotation = 'horizontal'
            else:
                xtickrotation = 'vertical'

        ax.set_xticklabels(xticklabels, fontsize=fontsize,
            fontweight=fontweight, rotation=xtickrotation)
        #    rotation='horizontal' if group_ticklabels else 'vertical')
        ax.set_xticks(np.arange(0, len(df.columns), xstep) + xoffset)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=fontsize,
            fontweight=fontweight, rotation='horizontal')
        #    rotation='vertical' if group_ticklabels else 'horizontal')
        ax.set_yticks(np.arange(0, len(df), ystep) + yoffset)

    # TODO test this doesn't change rotation if we just set rotation above

    # this doesn't seem like it will work, since it seems to clear the default
    # ticklabels that there actually were...
    #ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize,
    #    fontweight=fontweight)

    # didn't seem to do what i was expecting
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False)

    if title is not None:
        ax.set_xlabel(title, fontsize=(fontsize + 1.5), labelpad=12)

    if made_fig:
        plt.tight_layout()
        return fig
    else:
        return cax


# TODO TODO call this in gui / factor into plot_odor_corrs (though it would
# require accesss to df...) and call that there
def add_missing_odor_cols(df, missing_df):
    """
    """
    # TODO maybe check cols are indeed describing odors in missing_df?

    # TODO delete / change note to be relevant here. copied from original
    # implementation in gui
    # This + pivot_table w/ dropna=False won't work until this bug:
    # https://github.com/pandas-dev/pandas/issues/18030 is fixed.
    '''
    window_trial_means = pd.concat([window_trial_means,
        missing_dff.set_index(window_trial_means.index.names
        ).df_over_f
    ])
    '''
    missing_dff = df[df.df_over_f.isnull()][
        missing_df.columns.names + ['cell', 'df_over_f']
    ]
    # Hack to workaround pivot NaN behavior bug mentioned above.
    assert missing_dff.df_over_f.isnull().all()
    missing_dff.df_over_f = missing_dff.df_over_f.fillna(0)
    extra_cols = missing_dff.pivot_table(
        index='cell', values='df_over_f',
        columns=['name1','name2','repeat_num','order']
    )
    extra_cols.iloc[:] = np.nan

    assert (len(missing_df.columns.drop_duplicates()) ==
        len(missing_df.columns))

    missing_df = pd.concat([missing_df, extra_cols], axis=1)

    assert (len(missing_df.columns.drop_duplicates()) ==
        len(missing_df.columns))

    missing_df.sort_index(axis='columns', inplace=True)
    # end of the hack to workaround pivot NaN behavior

    return missing_df


# TODO should i actually compute correlations in here too? check input, and
# compute if input wasn't correlations (/ symmetric?)?
# if so, probably return them as well.
def plot_odor_corrs(corr_df, odor_order=False, odors_in_order=None,
    trial_stat='mean', title_suffix='', **kwargs):
    """Takes a symmetric DataFrame with odor x odor correlations and plots it.
    """
    # TODO TODO TODO test this fn w/ possible missing data case.
    # bring guis support for that in here?
    if odors_in_order is not None:
        odor_order = True

    if odor_order:
        # 'name2' is just 'no_second_odor' for a lot of my data
        # (the non-pair stuff)
        name_prefix = 'name1'

        # TODO probably refactor the two duped things below
        odor_name_rows = [c for c in corr_df.index.names
            if c.startswith(name_prefix)
        ]
        if len(odor_name_rows) != 1:
            raise ValueError('expected the name of exactly one index level to '
                f'start with {name_prefix}'
            )
        odor_name_row = odor_name_rows[0]

        odor_name_cols = [c for c in corr_df.columns.names
            if c.startswith(name_prefix)
        ]
        if len(odor_name_cols) != 1:
            raise ValueError('expected the name of exactly one column level to '
                f'start with {name_prefix}'
            )
        odor_name_col = odor_name_cols[0]
        #

        if len(corr_df.index.names) == 1:
            assert len(corr_df.columns.names) == 1
            # Necessary to avoid this error:
            # KeyError: 'Requested level...does not match index name (None)'
            odor_name_row = None
            odor_name_col = None

        corr_df = corr_df.reindex(odors_in_order, level=odor_name_row,
            axis='index').reindex(odors_in_order, level=odor_name_col,
            axis='columns'
        )
        if odors_in_order is None:
            # TODO 
            raise NotImplementedError

        if 'group_ticklabels' not in kwargs:
            kwargs['group_ticklabels'] = True
    else:
        corr_df = corr_df.sort_index(
            axis=0, level='order', sort_remaining=False).sort_index(
            axis=1, level='order', sort_remaining=False
        )

    if 'title' not in kwargs:
        kwargs['title'] = ('Odor' if odor_order else 'Presentation') + ' order'
        kwargs['title'] += title_suffix

    if 'ticklabels' not in kwargs:
        kwargs['ticklabels'] = format_mixture

    if 'colorbar_label' not in kwargs:
        kwargs['colorbar_label'] = \
            trial_stat.title() + r' response $\frac{\Delta F}{F}$ correlation'

    return matshow(corr_df, **kwargs)


# TODO maybe one fn that puts in matrix format and another in table
# (since matrix may be sparse...)
def plot_pair_n(df, *args):
    """Plots a matrix of odor1 X odor2 w/ counts of flies as entries.

    Args:
    df (pd.DataFrame): DataFrame with columns:
        - prep_date
        - fly_num
        - thorimage_id
        - name1
        - name2
        Data already collected w/ odor pairs.

    odor_panel (pd.DataFrame): (optional) DataFrame with columns:
        - odor_1
        - odor_2
        - reason (maybe make this optional?)
        The odor pairs experiments are supposed to collect data for.
    """
    import imgkit
    import seaborn as sns

    odor_panel = None
    if len(args) == 1:
        odor_panel = args[0]
    elif len(args) != 0:
        raise ValueError('incorrect number of arguments')
    # TODO maybe make df optional and read from database if it's not passed?
    # TODO a flag to show all stuff marked attempt analysis in gsheet?

    # TODO borrow more of this / call this in part of kc_analysis that made that
    # table w/ these counts for repeats?

    # TODO also handle no_second_odor
    df = df.drop(
        index=df[(df.name1 == 'paraffin') | (df.name2 == 'paraffin')].index)

    # TODO possible to do at least a partial check w/ n_accepted_blocks sum?
    # (would have to do outside of this fn. presentations here doesn't have it.
    # whatever latest_analysis returns might.)

    replicates = df[
        ['prep_date','fly_num','recording_from','name1','name2']
    ].drop_duplicates()

    # TODO do i actually want margins? (would currently count single odors twice
    # if in multiple comparison... may at least not want that?)
    # hide margins for now.
    pair_n = pd.crosstab(replicates.name1, replicates.name2) #, margins=True)

    # Making the rectangular matrix pair_n square
    # (same indexes on row and column axes)

    if odor_panel is None:
        # This is basically equivalent to the logic in the branch below,
        # although the index is not defined separately here.
        full_pair_n = pair_n.combine_first(pair_n.T).fillna(0.0)
    else:
        # TODO [change odor<n> to / handle] name<n>, to be consistent w/ above
        # TODO TODO TODO also make this triangular / symmetric
        odor_panel = odor_panel.pivot_table(index='odor_1', columns='odor_2',
            aggfunc=lambda x: True, values='reason')

        full_panel_index = odor_panel.index.union(odor_panel.columns)
        full_data_index = pair_n.index.union(pair_n.columns)
        assert full_data_index.isin(full_panel_index).all()
        # TODO also check no pairs occur in data that are not in panel
        # TODO isin-like check for tuples (or other combinations of rows)?
        # just iterate over both after drop_duplicates?

        full_pair_n = pair_n.reindex(index=full_panel_index
            ).reindex(columns=full_panel_index)
        # TODO maybe making symmetric is a matter of setting 0 to nan here?
        # and maybe setting back to nan at the end if still 0?
        full_pair_n.update(full_pair_n.T)
        # TODO full_pair_n.fillna(0, inplace=True)?

    # TODO TODO delete this hack once i find a nicer way to make the
    # output of crosstab symmetric
    for i in range(full_pair_n.shape[0]):
        for j in range(full_pair_n.shape[1]):
            a = full_pair_n.iat[i,j]
            b = full_pair_n.iat[j,i]
            if a > 0 and (pd.isnull(b) or b == 0):
                full_pair_n.iat[j,i] = a
            elif b > 0 and (pd.isnull(a) or a == 0):
                full_pair_n.iat[i,j] = b
    # TODO also delete this hack. this assumes that anything set to 0
    # is not actually a pair in the panel (which should be true right now
    # but will not always be)
    full_pair_n.replace(0, np.nan, inplace=True)
    #

    # TODO TODO TODO make crosstab output actually symmetric, not just square
    # (or is it always one diagonal that's filled in? if so, really just need
    # that)
    assert full_pair_n.equals(full_pair_n.T)

    # TODO TODO TODO how to indicate which of the pairs we are actually
    # interested in? grey out the others? white the others? (just set to nan?)
    # (maybe only use to grey / white out if passed in?)
    # (+ margins for now)

    # TODO TODO TODO color code text labels by pair selection reason + key
    # TODO what to do when one thing falls under two reasons though...?
    # just like a key (or things alongside ticklabels) that has each color
    # separately? just symbols in text, if that's easier?

    # TODO TODO display actual counts in squares in matshow
    # maybe make colorbar have discrete steps?

    full_pair_n.fillna('', inplace=True)
    cm = sns.light_palette('seagreen', as_cmap=True)
    # TODO TODO if i'm going to continue using styler + imgkit
    # at least figure out how to get the cmap to actually work
    # need some css or something?
    html = full_pair_n.style.background_gradient(cmap=cm).render()
    imgkit.from_string(html, 'natural_odors_pair_n.png')


# TODO get x / y from whether they were declared share<x/y> in facetgrid
# creation?
def fix_facetgrid_axis_labels(facet_grid, shared_in_center=False,
    x=True, y=True) -> None:
    """Modifies a FacetGrid to not duplicate X and Y axis text labels.
    """
    # regarding the choice of shared_in_center: WWMDD?
    if shared_in_center:
        # TODO maybe add a axes over / under the FacetGrid axes, with the same
        # shape, and label that one (i think i did this in my gui or one of the
        # plotting fns. maybe plot_traces?)
        raise NotImplementedError
    else:
        for ax in facet_grid.axes.flat:
            if not (ax.is_first_col() and ax.is_last_row()):
                if x:
                    ax.set_xlabel('')
                if y:
                    ax.set_ylabel('')


def set_facetgrid_legend(facet_grid, **kwargs) -> None:
    """
    In cases where different axes have different subsets of the hue levels,
    the legend may not contain the artists for the union of hue levels across
    all axes. This sets a legend from the hue artists across all axes.
    """
    #from matplotlib.collections import PathCollection
    legend_data = dict()
    for ax in facet_grid.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for label, h in zip(labels, handles):
            #if type(h) is PathCollection:
            # From inspecting facet_grid._legend_data in cases where some labels
            # pointed to empty lines (the phenotype in the case where things
            # weren't behaving as I wanted), the empty lines had this empty
            # facecolor.
            facecolor = h.get_facecolor()
            if len(facecolor) == 0:
                continue
            #else:
            #    print(type(h))
            #    import ipdb; ipdb.set_trace()

            if label in legend_data:
                # TODO maybe assert a wide variety of properties of the
                # matplotlib.collections.PathCollection objects are the same
                # (line width, dash, etc)
                past_facecolor = legend_data[label].get_facecolor()
                # TODO TODO TODO fix! this is failing again 2020-08-25
                # (after re-installing requirements.txt, when running
                # kc_mix_analysis.py w/ no just -j arg)
                assert np.array_equal(facecolor, past_facecolor), \
                    f'{facecolor} != {past_facecolor}'
            else:
                legend_data[label] = h

    facet_grid.add_legend(legend_data, **kwargs)


# TODO test when ax actually is passed in now that I made it a kwarg
# (also works as optional positional arg, right?)
def closed_mpl_contours(footprint, ax=None, if_multiple='err', **kwargs):
    """
    Args:
        if_multiple (str): 'take_largest'|'join'|'err'
    """
    dims = footprint.shape
    padded_footprint = np.zeros(tuple(d + 2 for d in dims))
    padded_footprint[tuple(slice(1,-1) for _ in dims)] = footprint
    
    # TODO delete
    #fig = plt.figure()
    #
    if ax is None:
        ax = plt.gca()

    mpl_contour = ax.contour(padded_footprint > 0, [0.5], **kwargs)
    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1

    paths = mpl_contour.collections[0].get_paths()

    if len(paths) != 1:
        if if_multiple == 'err':
            raise RuntimeError('multiple disconnected paths in one footprint')

        elif if_multiple == 'take_largest':
            largest_sum = 0
            largest_idx = 0
            total_sum = 0
            for p in range(len(paths)):
                path = paths[p]

                # TODO TODO TODO maybe replace mpl stuff w/ cv2 drawContours?
                # (or related...) (fn now in here as contour2mask)
                mask = np.ones_like(footprint, dtype=bool)
                for x, y in np.ndindex(footprint.shape):
                    # TODO TODO not sure why this seems to be transposed, but it
                    # does (make sure i'm not doing something wrong?)
                    if path.contains_point((x, y)):
                        mask[x, y] = False
                # Places where the mask is False are included in the sum.
                path_sum = MaskedArray(footprint, mask=mask).sum()
                # TODO maybe check that sum of all path_sums == footprint.sum()?
                # seemed there were some paths w/ 0 sum... cnmf err?
                '''
                print('mask_sum:', (~ mask).sum())
                print('path_sum:', path_sum)
                print('regularly masked sum:', footprint[(~ mask)].sum())
                plt.figure()
                plt.imshow(mask)
                plt.figure()
                plt.imshow(footprint)
                plt.show()
                import ipdb; ipdb.set_trace()
                '''
                if path_sum > largest_sum:
                    largest_sum = path_sum
                    largest_idx = p

                total_sum += path_sum
            footprint_sum = footprint.sum()
            # TODO float formatting / some explanation as to what this is
            print('footprint_sum:', footprint_sum)
            print('total_sum:', total_sum)
            print('largest_sum:', largest_sum)
            # TODO is this only failing when stuff is overlapping?
            # just merge in that case? (wouldn't even need to dilate or
            # anything...) (though i guess then the inequality would go the
            # other way... is it border pixels? just ~dilate by one?)
            # TODO fix + uncomment
            ######assert np.isclose(total_sum, footprint_sum)
            path = paths[largest_idx]

        elif if_multiple == 'join':
            raise NotImplementedError
    else:
        path = paths[0]

    # TODO delete
    #plt.close(fig)
    #

    contour = path.vertices
    # Correct index change caused by padding.
    return contour - 1


def plot_traces(*args, footprints=None, order_by='odors', scale_within='cell',
    n=20, random=False, title=None, response_calls=None, raw=False,
    smoothed=True, show_footprints=True, show_footprints_alone=False,
    show_cell_ids=True, show_footprint_with_mask=False, gridspec=None,
    linewidth=0.5, verbose=True):
    # TODO TODO be clear on requirements of df and cell_ids in docstring
    """
    n (int): (default=20) Number of cells to plot traces for if cell_ids not
        passed as second positional argument.
    random (bool): (default=False) Whether the `n` cell ids should be selected
        randomly. If False, the first `n` cells are used.
    order_by (str): 'odors' or 'presentation_order'
    scale_within (str): 'none', 'cell', or 'trial'
    gridspec (None or matplotlib.gridspec.*): region of a parent figure
        to draw this plot on.
    linewidth (float): 0.25 seemed ok on CNMF data, but too small w/ clean
    traces.
    """
    import tifffile
    import cv2
    # TODO maybe use cv2 and get rid of this dep?
    from skimage import color

    # TODO make text size and the spacing of everything more invariant to figure
    # size. i think the default size of this figure ended up being bigger when i
    # was using it in kc_analysis than it is now in the gui, so it isn't display
    # well in the gui, but fixing it here might break it in the kc_analysis case
    if verbose:
        print('Entering plot_traces...')

    if len(args) == 1:
        df = args[0]
        # TODO flag to also subset to responders first?
        all_cells = cell_ids(df)
        n = min(n, len(all_cells))
        if random:
            # TODO maybe flag to disable seed?
            cells = all_cells.sample(n=n, random_state=1)
        else:
            cells = all_cells[:n]

    elif len(args) == 2:
        df, cells = args

    else:
        raise ValueError('must call with either df or df and cells')

    if show_footprints:
        # or maybe just download (just the required!) footprints from sql?
        if footprints is None:
            raise ValueError('must pass footprints kwarg if show_footprints')
        # decide whether this should be in the preconditions or just done here
        # (any harm to just do here anyway?)
        #else:
        #    footprints = footprints.set_index(recording_cols + ['cell'])

    # TODO TODO TODO fix odor labels as in matrix (this already done?)
    # (either rotate or use abbreviations so they don't overlap!)

    # TODO check order_by and scale_within are correct
    assert raw or smoothed

    # TODO maybe automatically show_cells if show_footprints is true,
    # otherwise don't?
    # TODO TODO maybe indicate somehow the multiple response criteria
    # when it is a list (add border & color each half accordingly?)

    extra_cols = 0
    # TODO which of these cases do i want to support here?
    if show_footprints:
        if show_footprints_alone:
            extra_cols = 2
        else:
            extra_cols = 1
    elif show_footprints_alone:
        raise NotImplementedError

    # TODO possibility of other column for avg + roi overlays
    # possible to make it larger, or should i use a layout other than
    # gridspec? just give it more grid elements?
    # TODO for combinatorial combinations of flags enabling cols on
    # right, maybe set index for each of those flags up here

    # TODO could also just could # trials w/ drop_duplicates, for more
    # generality
    n_repeats = n_expected_repeats(df)
    n_trials = n_repeats * len(df[['name1','name2']].drop_duplicates())

    if gridspec is None:
        # This seems to hang... not sure if it's usable w/ some changes.
        #fig = plt.figure(constrained_layout=True)
        fig = plt.figure()
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
        made_fig = True
    else:
        fig = gridspec.get_topmost_subplotspec().get_gridspec().figure
        gs = gridspec.subgridspec(4, 6, hspace=0.4, wspace=0.3)
        made_fig = False

    if show_footprints:
        trace_gs_slice = gs[:3,:4]
    else:
        trace_gs_slice = gs[:,:]

    # For common X/Y labels
    bax = fig.add_subplot(trace_gs_slice, frameon=False)
    # hide tick and tick label of the big axes
    bax.tick_params(top=False, bottom=False, left=False, right=False,
        labelcolor='none')
    bax.grid(False)

    trace_gs = trace_gs_slice.subgridspec(len(cells),
        n_trials + extra_cols, hspace=0.15, wspace=0.06)

    axs = []
    for ti in range(trace_gs._nrows):
        axs.append([])
        for tj in range(trace_gs._ncols):
            ax = fig.add_subplot(trace_gs[ti,tj])
            axs[-1].append(ax)
    axs = np.array(axs)

    # TODO want all of these behind show_footprints?
    if show_footprints:
        # TODO use 2/3 for widgets?
        # TODO or just text saying which keys to press? (if only
        # selection mechanism is going to be hover, mouse clicks
        # wouldn't make sense...)

        avg_ax = fig.add_subplot(gs[:, -2:])
        # TODO TODO maybe show trial movie beneath this?
        # (also on hover/click like (trial,cell) data)

    if title is not None:
        #pad = 40
        pad = 15
        # was also using default fontsize here in kc_analysis use case
        # increment pad by 5 for each newline in title?
        bax.set_title(title, pad=pad, fontsize=9)

    bax.set_ylabel('Cell')

    # This pad is to make it not overlap w/ time label on example plot.
    # Was left to default value for kc_analysis.
    # TODO negative labelpad work? might get drawn over by axes?
    labelpad = -10
    if order_by == 'odors':
        bax.set_xlabel('Trials ordered by odor', labelpad=labelpad)
    elif order_by == 'presentation_order':
        bax.set_xlabel('Trials in presentation order', labelpad=labelpad)

    ordering = pair_ordering(df)

    '''
    display_start_time = -3.0
    display_stop_time = 10
    display_window = df[
        (comparison_df.from_onset >= display_start_time) &
        (comparison_df.from_onset <= display_stop_time)]
    '''
    display_window = df

    smoothing_window_secs = 1.0
    fps = fps_from_thor(df)
    window_size = int(np.round(smoothing_window_secs * fps))

    group_cols = trial_cols + ['order']

    xmargin = 1
    xmin = display_window.from_onset.min() - xmargin
    xmax = display_window.from_onset.max() + xmargin

    response_rgb = (0.0, 1.0, 0.2)
    nonresponse_rgb = (1.0, 0.0, 0.0)
    response_call_alpha = 0.2

    if scale_within == 'none':
        ymin = None
        ymax = None

    cell2contour = dict()
    cell2rect = dict()
    cell2text_and_rect = dict()

    seen_ij = set()
    avg = None
    for i, cell_id in enumerate(cells):
        if verbose:
            print('Plotting cell {}/{}...'.format(i + 1, len(cells)))

        cell_data = display_window[display_window.cell == cell_id]
        cell_trials = cell_data.groupby(group_cols, sort=False)[
            ['from_onset','df_over_f']]

        prep_date = pd.Timestamp(cell_data.prep_date.unique()[0])
        date_dir = prep_date.strftime(date_fmt_str)
        fly_num = cell_data.fly_num.unique()[0]
        thorimage_id = cell_data.thorimage_id.unique()[0]

        #assert len(cell_trials) == axs.shape[1]

        if show_footprints:
            if avg is None:
                # only uncomment to support dff images and other stuff like that
                '''
                try:
                    # TODO either put in docstring that datetime.datetime is
                    # required, or cast input date as appropriate
                    # (does pandas date type support strftime?)
                    # or just pass date_dir?
                    # TODO TODO should not use nr if going to end up using the
                    # rig avg... but maybe lean towards computing the avg in
                    # that case rather than deferring to rigid?
                    tif = motion_corrected_tiff_filename(
                        prep_date, fly_num, thorimage_id)
                except IOError as e:
                    if verbose:
                        print(e)
                    continue

                # TODO maybe show progress bar / notify on this step
                if verbose:
                    print('Loading full movie from {} ...'.format(tif),
                        end='', flush=True)
                movie = tifffile.imread(tif)
                if verbose:
                    print(' done.')
                '''

                # TODO modify motion_corrected_tiff_filename to work in this
                # case too?
                tif_dir = join(analysis_output_root(), date_dir, str(fly_num),
                    'tif_stacks')
                avg_nr_tif = join(tif_dir, 'AVG', 'nonrigid',
                    'AVG{}_nr.tif'.format(thorimage_id))
                avg_rig_tif = join(tif_dir, 'AVG', 'rigid',
                    'AVG{}_rig.tif'.format(thorimage_id))

                avg_tif = None
                if exists(avg_nr_tif):
                    avg_tif = avg_nr_tif
                elif exists(avg_rig_tif):
                    avg_tif = avg_rig_tif

                if avg_tif is None:
                    raise IOError(('No average motion corrected TIFs ' +
                        'found in {}').format(tif_dir))

                avg = tifffile.imread(avg_tif)
                '''
                avg = cv2.normalize(avg, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                '''
                # TODO find a way to histogram equalize w/o converting
                # to 8 bit?
                avg = cv2.normalize(avg, None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                better_constrast = cv2.equalizeHist(avg)

                rgb_avg = color.gray2rgb(better_constrast)

            cell_row = (prep_date, fly_num, thorimage_id, cell_id)
            footprint_row = footprints.loc[cell_row]

            # TODO TODO TODO probably need to tranpose how footprint is handled
            # downstream (would prefer not to transpose footprint though)
            # (as i had to switch x_coords and y_coords in db as they were
            # initially entered swapped)
            footprint = db_row2footprint(footprint_row, shape=avg.shape)

            # TODO maybe some percentile / fixed size about maximum
            # density?
            cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
                crop_to_nonzero(footprint, margin=6)
            cell2rect[cell_id] = (x_min, x_max, y_min, y_max)

            cropped_avg = \
                better_constrast[x_min:x_max + 1, y_min:y_max + 1]

            if show_footprint_with_mask:
                # TODO figure out how to suppress clipping warning in the case
                # when it's just because of float imprecision (e.g. 1.0000001
                # being clipped to 1) maybe just normalize to [0 + epsilon, 1 -
                # epsilon]?
                # TODO TODO or just set one channel to be this
                # footprint?  scale first?
                cropped_footprint_rgb = \
                    color.gray2rgb(cropped_footprint)
                for c in (1,2):
                    cropped_footprint_rgb[:,:,c] = 0
                # TODO plot w/ value == 1 to test?

                cropped_footprint_hsv = \
                    color.rgb2hsv(cropped_footprint_rgb)

                cropped_avg_hsv = \
                    color.rgb2hsv(color.gray2rgb(cropped_avg))

                # TODO hue already seems to be constant at 0.0 (red?)
                # so maybe just directly set to red to avoid confusion?
                cropped_avg_hsv[..., 0] = cropped_footprint_hsv[..., 0]

                alpha = 0.3
                cropped_avg_hsv[..., 1] = cropped_footprint_hsv[..., 1] * alpha

                composite = color.hsv2rgb(cropped_avg_hsv)

                # TODO TODO not sure this is preserving hue/sat range to
                # indicate how strong part of filter is
                # TODO figure out / find some way that would
                # TODO TODO maybe don't normalize within each ROI? might
                # screw up stuff relative to histogram equalized
                # version...
                # TODO TODO TODO still normalize w/in crop in contour
                # case?
                composite = cv2.normalize(composite, None, alpha=0.0,
                    beta=1.0, norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F)

            else:
                # TODO could also use something more than this
                # TODO TODO fix bug here. see 20190402_bug1.txt
                # TODO TODO where are all zero footprints coming from?
                cropped_footprint_nonzero = cropped_footprint > 0
                if not np.any(cropped_footprint_nonzero):
                    continue

                level = \
                    cropped_footprint[cropped_footprint_nonzero].min()

            if show_footprints_alone:
                ax = axs[i,-2]
                f_ax = axs[i,-1]
                f_ax.imshow(cropped_footprint, cmap='gray')
                f_ax.axis('off')
            else:
                ax = axs[i,-1]

            if show_footprint_with_mask:
                ax.imshow(composite)
            else:
                ax.imshow(cropped_avg, cmap='gray')
                # TODO TODO also show any other contours in this rectangular ROI
                # in a diff color! (copy how gui does this)
                cell2contour[cell_id] = \
                    closed_mpl_contours(cropped_footprint, ax, colors='red')

            ax.axis('off')

            text = str(cell_id + 1)
            h = y_max - y_min
            w = x_max - x_min
            rect = patches.Rectangle((y_min, x_min), h, w,
                linewidth=1.5, edgecolor='b', facecolor='none')
            cell2text_and_rect[cell_id] = (text, rect)

        if scale_within == 'cell':
            ymin = None
            ymax = None

        for n, cell_trial in cell_trials:
            #(prep_date, fly_num, thorimage_id,
            (_, _, _, comp, o1, o2, repeat_num, order) = n

            # TODO TODO also support a 'fixed' order that B wanted
            # (which should also include missing stuff[, again in gray,]
            # ideally)
            if order_by == 'odors':
                j = n_repeats * ordering[(o1, o2)] + repeat_num

            elif order_by == 'presentation_order':
                j = order

            else:
                raise ValueError("supported orderings are 'odors' and "+
                    "'presentation_order'")

            if scale_within == 'trial':
                ymin = None
                ymax = None

            assert (i,j) not in seen_ij
            seen_ij.add((i,j))
            ax = axs[i,j]

            # So that events that get the axes can translate to cell /
            # trial information.
            ax.cell_id = cell_id
            ax.trial_info = n

            # X and Y axis major tick label fontsizes.
            # Was left to default for kc_analysis.
            ax.tick_params(labelsize=6)

            trial_times = cell_trial['from_onset']

            # TODO TODO why is *first* ea trial the one not shown, and
            # apparently the middle pfo trial
            # (was it not actually ordered by 'order'/frame_num outside of
            # odor order???)
            # TODO TODO TODO why did this not seem to work? (or only for
            # 1/3.  the middle one. iaa.)
            # (and actually title is still hidden for ea and pfo trials
            # mentioned above, but numbers / ticks / box still there)
            # (above notes only apply to odor order case. presentation order
            # worked)
            # TODO and why is gray title over correct axes in odor order case,
            # but axes not displaying data are in wrong place?
            # TODO is cell_trial messed up?

            # Supports at least the case when there are missing odor
            # presentations at the end of the ~block.
            missing_this_presentation = \
                trial_times.shape == (1,) and pd.isnull(trial_times.iat[0])

            if i == 0:
                # TODO group in odors case as w/ matshow?
                if order_by == 'odors':
                    trial_title = format_mixture({
                        'name1': o1,
                        'name2': o2,
                    })
                elif order_by == 'presentation_order':
                    trial_title = format_mixture({
                        'name1': o1,
                        'name2': o2
                    })

                if missing_this_presentation:
                    tc = 'gray'
                else:
                    tc = 'black'

                ax.set_title(trial_title, fontsize=6, color=tc)
                # TODO may also need to do tight_layout here...
                # it apply to these kinds of titles?

            if missing_this_presentation:
                ax.axis('off')
                continue

            trial_dff = cell_trial['df_over_f']

            if raw:
                if ymax is None:
                    ymax = trial_dff.max()
                    ymin = trial_dff.min()
                else:
                    ymax = max(ymax, trial_dff.max())
                    ymin = min(ymin, trial_dff.min())

                ax.plot(trial_times, trial_dff, linewidth=linewidth)

            if smoothed:
                # TODO kwarg(s) to control smoothing?
                sdff = smooth(trial_dff, window_len=window_size)

                if ymax is None:
                    ymax = sdff.max()
                    ymin = sdff.min()
                else:
                    ymax = max(ymax, sdff.max())
                    ymin = min(ymin, sdff.min())

                # TODO TODO have plot_traces take kwargs to be passed to
                # plotting fn + delete separate linewidth
                ax.plot(trial_times, sdff, color='black', linewidth=linewidth)

            # TODO also / separately subsample?

            if response_calls is not None:
                was_a_response = \
                    response_calls.loc[(o1, o2, repeat_num, cell_id)]

                if was_a_response:
                    ax.set_facecolor(response_rgb +
                        (response_call_alpha,))
                else:
                    ax.set_facecolor(nonresponse_rgb +
                        (response_call_alpha,))

            if i == axs.shape[0] - 1 and j == 0:
                # want these centered on example plot or across all?

                # I had not specified fontsize for kc_analysis case, so whatever
                # the default value was probably worked OK there.
                ax.set_xlabel('Seconds from odor onset', fontsize=6)

                if scale_within == 'none':
                    scaletext = ''
                elif scale_within == 'cell':
                    scaletext = '\nScaled within each cell'
                elif scale_within == 'trial':
                    scaletext = '\nScaled within each trial'

                # TODO just change to "% maximum w/in <x>" or something?
                # Was 70 for kc_analysis case. That's much too high here.
                #labelpad = 70
                labelpad = 10
                ax.set_ylabel(r'$\frac{\Delta F}{F}$' + scaletext,
                    rotation='horizontal', labelpad=labelpad)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            else:
                if show_cell_ids and j == len(cell_trials) - 1:
                    # Indexes as they would be from one. For comparison
                    # w/ Remy's MATLAB analysis.
                    # This and default fontsize worked for kc_analysis case,
                    # not for GUI.
                    #labelpad = 18
                    labelpad = 25
                    ax.set_ylabel(str(cell_id + 1),
                        rotation='horizontal', labelpad=labelpad, fontsize=5)
                    ax.yaxis.set_label_position('right')
                    # TODO put a label somewhere on the plot indicating
                    # these are cell IDs

                for d in ('top', 'right', 'bottom', 'left'):
                    ax.spines[d].set_visible(False)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        # TODO change units in this case on ylabel?
        # (to reflect how it was scaled)
        if scale_within == 'cell':
            for r in range(len(cell_trials)):
                ax = axs[i,r]
                ax.set_ylim(ymin, ymax)

    if scale_within == 'none':
        for i in range(len(cells)):
            for j in range(len(cell_trials)):
                ax = axs[i,j]
                ax.set_ylim(ymin, ymax)
    
    if show_footprints:
        fly_title = '{}, fly {}, {}'.format(
            date_dir, fly_num, thorimage_id)

        # title like 'Recording average fluorescence'?
        #avg_ax.set_title(fly_title)
        avg_ax.imshow(rgb_avg)
        avg_ax.axis('off')

        cell2rect_artists = dict()
        for cell_id in cells:
            # TODO TODO fix bug that required this (zero nonzero pixel
            # in cropped footprint thing...)
            if cell_id not in cell2text_and_rect:
                continue

            (text, rect) = cell2text_and_rect[cell_id]

            box = rect.get_bbox()
            # TODO appropriate font properties? placement still good?
            # This seemed to work be for (larger?) figures in kc_analysis,
            # too large + too close to boxes in gui (w/ ~8"x5" gridspec,dpi 100)
            # TODO set in relation to actual fig size (+ dpi?)
            #boxlabel_fontsize = 9
            boxlabel_fontsize = 6
            text_artist = avg_ax.text(box.xmin, box.ymin - 2, text,
                color='b', size=boxlabel_fontsize, fontweight='bold')
            # TODO jitter somehow (w/ arrow pointing to box?) to ensure no
            # overlap? (this would be ideal, but probably hard to implement)
            avg_ax.add_patch(rect)

            cell2rect_artists[cell_id] = (text_artist, rect)

    for i in range(len(cells)):
        for j in range(len(cell_trials)):
            ax = axs[i,j]
            ax.set_xlim(xmin, xmax)

    if made_fig:
        fig.tight_layout()
        return fig


def imshow(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    return fig


def image_grid(image_list):
    n = int(np.ceil(np.sqrt(len(image_list))))
    fig, axs = plt.subplots(n,n)
    for ax, img in zip(axs.flat, image_list):
        ax.imshow(img, cmap='gray')

    for ax in axs.flat:
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.05)
    return fig


# TODO worth having min/max as inputs, so that maybe can use vals from
# either scene or template for the other? i guess point of baselining
# is to avoid need for stuff like that...
def baselined_normed_u8(img):
    u8_max = 255
    # TODO maybe convert to float64 or something first before some operations,
    # to minimize rounding errs?
    baselined = img - img.min()
    normed = baselined / baselined.max()
    return (u8_max * normed).astype(np.uint8)


def template_match(scene, template, method_str='cv2.TM_CCOEFF', hist=False,
    debug=False):

    import cv2

    vscaled_scene = baselined_normed_u8(scene)
    # TODO TODO maybe template should only be scaled to it's usual fraction of
    # max of the scene? like scaled both wrt orig_scene.max() / max across all
    # images?
    vscaled_template = baselined_normed_u8(template)

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


def euclidean_dist(v1, v2):
    # Without the conversions to float 64 (or at least something else signed),
    # uint inputs lead to wraparound -> big distances occasionally.
    return np.linalg.norm(np.array(v1).astype(np.float64) -
        np.array(v2).astype(np.float64)
    )


def u8_color(draw_on):
    # TODO figure out why background looks lighter here than in other 
    # imshows of same input (w/o converting manually)
    draw_on = draw_on - np.min(draw_on)
    draw_on = draw_on / np.max(draw_on)
    cmap = plt.get_cmap('gray') #, lut=256)
    # (throwing away alpha coord w/ last slice)
    draw_on = np.round((cmap(draw_on)[:, :, :3] * 255)).astype(np.uint8)
    return draw_on



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

        draw_on = u8_color(draw_on)
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

                    '''
                    if per_scale_last_idx[scale_idx] == len(matches):
                        # TODO 
                        import ipdb; ipdb.set_trace()
                    '''

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

    '''
    if debug and _show_packing_constraints:
        title = 'greedy_roi_packing overlap exlusion mask'
        imshow(claimed, title)
    '''

    if debug and draw_on is not None and _show_fit:
        imshow(draw_on, 'greedy_roi_packing fit')

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

                dist = euclidean_dist(center, other_center)
                if dist <= min_dist2neighbor_px:
                    n_neighbors += 1

                if n_neighbors >= min_neighbors:
                    filtered_roi_centers.append(center)
                    filtered_roi_radii.append(radius)
                    break

    assert len(filtered_roi_centers) == len(filtered_roi_radii)
    return np.array(filtered_roi_centers), np.array(filtered_roi_radii)


def autoroi_metadata_filename(ijroi_file):
    path, fname = split(ijroi_file)
    return join(path, '.{}.meta.p'.format(fname))


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
        keys = tiff_filename2keys(tif)
        ti_dir = thorimage_dir(*tuple(keys))
        xmlroot = get_thorimage_xmlroot(ti_dir)
        um_per_pixel_xy = get_thorimage_pixelsize_xml(xmlroot)
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
        # analysis_output_root (or just change input in populate_db).
        # TODO or at least err if not subdir of it
        # see: https://stackoverflow.com/questions/3812849

        if _force_write_to is not None:
            if _force_write_to == True:
                fname = join(path, tiff_parts[0] + '_auto_rois.zip')
            else:
                fname = _force_write_to

        # TODO TODO TODO also check for modifications before overwriting (mtime 
        # in that hidden file)
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
            title = tiff_title(tif)
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
    return join(analysis_output_root(), template_cache)


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


# TODO TODO TODO after refactoring much of the stuff that was under
# open_recording and some of its downstream fns from gui.py, also refactor this
# to use the new fns
def movie_blocks(tif, movie=None, allow_gsheet_to_restrict_blocks=True,
    stimfile=None, first_block=None, last_block=None):
    """Returns list of arrays, one per continuous acquisition.

    Total length along time dimension should be preserved from input TIFF.
    """
    from scipy import stats

    if movie is None:
        import tifffile
        movie = tifffile.imread(tif)

    keys = tiff_filename2keys(tif)
    mat = matfile(*keys)
    # TODO TODO TODO refactor all stuff that uses this to new output format
    # (and remove factored checks, etc)
    ti = load_mat_timing_info(mat)

    if stimfile is None:
        df = mb_team_gsheet()
        recordings = df.loc[
            (df.date == keys.date) &
            (df.fly_num == keys.fly_num) &
            (df.thorimage_dir == keys.thorimage_id)
        ]
        del df
        recording = recordings.iloc[0]
        del recordings
        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.'.format(
                recording.project))
            return

        stimfile = recording['stimulus_data_file']
        first_block = recording['first_block']
        last_block = recording['last_block']
        del recording

        stimfile_path = join(stimfile_root(), stimfile)
    else:
        warnings.warn('using hardcoded stimulus file, rather than using value '
            'from MB team gsheet'
        )
        if exists(stimfile):
            stimfile_path = stimfile
        else:
            stimfile_path = join(stimfile_root(), stimfile)
            assert exists(stimfile_path), (f'stimfile {stimfile} not found '
                f'alone or under {stimfile_root()}'
            )

    # TODO also err if not readable / valid
    if not exists(stimfile_path):
        raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
            stimfile_root)
        )

    with open(stimfile_path, 'rb') as f:
        data = pickle.load(f)

    # TODO just infer from data if no stimfile and not specified in
    # metadata_file
    n_repeats = int(data['n_repeats'])

    # TODO delete this hack (which is currently just using new pickle
    # format as a proxy for the experiment being a supermixture experiment)
    if 'odor_lists' not in data:
        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3
        odor_list = data['odor_pair_list']
    else:
        n_expected_real_blocks = 3
        odor_list = data['odor_lists']
        # because of "block" def in arduino / get_stiminfo code
        # not matching def in randomizer / stimfile code
        # (scopePin pulses vs. randomization units, depending on settings)
        presentations_per_repeat = len(odor_list) // n_expected_real_blocks
        assert len(odor_list) % n_expected_real_blocks == 0

        # Hardcode to break up into more blocks, to align defs of blocks.
        # TODO (maybe just for experiments on 2019-07-25 ?) or change block
        # handling in here? make more flexible?
        n_repeats = 1

    presentations_per_block = n_repeats * presentations_per_repeat

    if pd.isnull(first_block):
        first_block = 0
    else:
        first_block = int(first_block) - 1

    if pd.isnull(last_block):
        n_full_panel_blocks = \
            int(len(odor_list) / presentations_per_block)
        last_block = n_full_panel_blocks - 1
    else:
        last_block = int(last_block) - 1

    first_presentation = first_block * presentations_per_block
    last_presentation = (last_block + 1) * presentations_per_block - 1

    odor_list = odor_list[first_presentation:(last_presentation + 1)]
    assert (len(odor_list) % (presentations_per_repeat * n_repeats) == 0)

    # TODO TODO delete odor frame stuff after using them to check blocks frames
    # are actually blocks and not trials
    # TODO or if keeping odor stuff, re-add asserts involving odor_list,
    # since how i have that here

    odor_onset_frames = np.array(ti['stim_on'], dtype=np.uint32
        ).flatten() - 1
    odor_offset_frames = np.array(ti['stim_off'], dtype=np.uint32).flatten() - 1
    assert len(odor_onset_frames) == len(odor_offset_frames)

    # Of length equal to number of blocks. Each element is the frame
    # index (from 1) in CNMF output that starts the block, where
    # block is defined as a period of continuous acquisition.
    block_first_frames = np.array(ti['block_start_frame'], dtype=np.uint32
        ).flatten() - 1
    block_last_frames = np.array(ti['block_end_frame'], dtype=np.uint32
        ).flatten() - 1

    n_blocks_from_gsheet = last_block - first_block + 1
    n_blocks_from_thorsync = len(block_first_frames)

    assert (len(odor_list) == (last_block - first_block + 1) *
        presentations_per_block)

    n_presentations = n_blocks_from_gsheet * presentations_per_block

    err_msg = ('{} blocks ({} to {}, inclusive) in Google sheet {{}} {} ' +
        'blocks from ThorSync.').format(n_blocks_from_gsheet,
        first_block + 1, last_block + 1, n_blocks_from_thorsync)
    fail_msg = (' Fix in Google sheet, turn off ' +
        'cache if necessary, and rerun.')

    if n_blocks_from_gsheet > n_blocks_from_thorsync:
        raise ValueError(err_msg.format('>') + fail_msg)

    elif n_blocks_from_gsheet < n_blocks_from_thorsync:
        if allow_gsheet_to_restrict_blocks:
            warnings.warn(err_msg.format('<') + (' This is ONLY ok if you '+
                'intend to exclude the LAST {} blocks in the Thor output.'
                ).format(n_blocks_from_thorsync - n_blocks_from_gsheet))
        else:
            raise ValueError(err_msg.format('<') + fail_msg)

    frame_times = np.array(ti['frame_times']).flatten()

    # TODO replace this w/ factored check fn
    total_block_frames = 0
    for i, (b_start, b_end) in enumerate(
        zip(block_first_frames, block_last_frames)):

        if i != 0:
            last_b_end = block_last_frames[i - 1]
            assert last_b_end == (b_start - 1)

        assert (b_start < len(frame_times)) and (b_end < len(frame_times))
        block_frametimes = frame_times[b_start:b_end]
        dts = np.diff(block_frametimes)
        # np.max(np.abs(dts - np.mean(dts))) / np.mean(dts)
        # was 0.000148... in one case I tested w/ data from the older
        # system, so the check below w/ rtol=1e-4 would fail.
        mode = stats.mode(dts)[0]
        assert np.allclose(dts, mode, rtol=3e-4)

        total_block_frames += b_end - b_start + 1

    orig_n_frames = movie.shape[0]
    # TODO may need to remove this assert to handle cases where there is a
    # partial block (stopped early). leave assert after slicing tho.
    # (warn instead, probably)
    assert total_block_frames == orig_n_frames, \
        '{} != {}'.format(total_block_frames, orig_n_frames)

    if allow_gsheet_to_restrict_blocks:
        # TODO unit test for case where first_block != 0 and == 0
        # w/ last_block == first_block and > first_block
        # TODO TODO doesn't this only support dropping blocks at end?
        # do i assert that first_block is 0 then? probably should...
        # TODO TODO TODO shouldnt it be first_block:last_block+1?
        block_first_frames = block_first_frames[
            :(last_block - first_block + 1)]
        block_last_frames = block_last_frames[
            :(last_block - first_block + 1)]

        assert len(block_first_frames) == n_blocks_from_gsheet
        assert len(block_last_frames) == n_blocks_from_gsheet

        # TODO also delete this odor frame stuff when done
        odor_onset_frames = odor_onset_frames[
            :(last_presentation - first_presentation + 1)]
        odor_offset_frames = odor_offset_frames[
            :(last_presentation - first_presentation + 1)]

        assert len(odor_onset_frames) == n_presentations
        assert len(odor_offset_frames) == n_presentations
        #

        frame_times = frame_times[:(block_last_frames[-1] + 1)]

    last_frame = block_last_frames[-1]

    n_tossed_frames = movie.shape[0] - (last_frame + 1)
    if n_tossed_frames != 0:
        print(('Tossing trailing {} of {} frames of movie, which did not ' +
            'belong to any used block.\n').format(
            n_tossed_frames, movie.shape[0]))

    # TODO factor this metadata handling out. fns for load / set?
    # combine w/ remy's .mat metadata (+ my stimfile?)

    # This will return defaults if the YAML file is not found.
    meta = metadata(*keys)

    # TODO want / need to do more than just slice to free up memory from
    # other pixels? is that operation worth it?
    drop_first_n_frames = meta['drop_first_n_frames']
    # TODO TODO err if this is past first odor onset (or probably even too
    # close)
    del meta

    odor_onset_frames = [n - drop_first_n_frames for n in odor_onset_frames]
    odor_offset_frames = [n - drop_first_n_frames for n in odor_offset_frames]

    block_first_frames = [n - drop_first_n_frames for n in block_first_frames]
    block_first_frames[0] = 0
    block_last_frames = [n - drop_first_n_frames for n in block_last_frames]

    assert odor_onset_frames[0] > 0

    frame_times = frame_times[drop_first_n_frames:]
    movie = movie[drop_first_n_frames:(last_frame + 1)]

    # TODO TODO fix bug referenced in cthulhu:190520...
    # and re-enable assert
    assert movie.shape[0] == len(frame_times), \
        '{} != {}'.format(movie.shape[0], len(frame_times))
    #

    if movie.shape[0] != len(frame_times):
        warnings.warn('{} != {}'.format(movie.shape[0], len(frame_times)))

    # TODO maybe move this and the above checks on block start/end frames
    # + frametimes into assign_frames_to_trials
    n_frames = movie.shape[0]
    total_block_frames = sum([e - s + 1 for s, e in
        zip(block_first_frames, block_last_frames)
    ])

    assert total_block_frames == n_frames, \
        '{} != {}'.format(total_block_frames, n_frames)


    # TODO any time / space diff returning slices to slice array and only
    # slicing inside loop vs. returning list of (presumably views) by slicing
    # matrix?
    blocks = [movie[start:(stop + 1)] for start, stop in
        zip(block_first_frames, block_last_frames)
    ]
    assert sum([b.shape[0] for b in blocks]) == movie.shape[0]
    return blocks


def downsample_movie(movie, target_fps, current_fps, allow_overshoot=True,
    allow_uneven_division=False, relative_fps_err=True, debug=False):
    """Returns downsampled movie by averaging consecutive groups of frames.

    Groups of frames averaged do not overlap.
    """
    if allow_uneven_division:
        raise NotImplementedError

    # TODO maybe kwarg for max acceptable (rel/abs?) factor error,
    # and err / return None if it can't be achieved

    target_factor = current_fps / target_fps
    if debug:
        print(f'allow_overshoot: {allow_overshoot}')
        print(f'allow_uneven_division: {allow_uneven_division}')
        print(f'relative_fps_err: {relative_fps_err}')
        print(f'target_fps: {target_fps:.2f}\n')
        print(f'target_factor: {target_factor:.2f}\n')

    n_frames = movie.shape[0]

    # TODO TODO also support uneven # of frames per bin (toss last probably)
    # (skip loop checking for even divisors in that case)

    # Find the largest/closest downsampling we can do, with equal numbers of
    # frames for each average.
    best_divisor = None
    for i in range(1, n_frames):
        if n_frames % i != 0:
            continue

        decimated_n_frames = n_frames // i
        # (will always be float(i) in even division case, so could get rid of
        # this if that's all i'll support)
        factor = n_frames / decimated_n_frames
        if debug:
            print(f'factor: {factor:.2f}')

        if factor > target_factor and not allow_overshoot:
            if debug:
                print('breaking because of overshoot')
            break

        downsampled_fps = current_fps / factor
        fps_error = downsampled_fps - target_fps
        if relative_fps_err:
            fps_error = fps_error / target_fps

        if debug:
            print(f'downsampled_fps: {downsampled_fps:.2f}')
            print(f'fps_error: {fps_error:.2f}')

        if best_divisor is None or abs(fps_error) < abs(best_fps_error):
            best_divisor = i
            best_downsampled_fps = downsampled_fps
            best_fps_error = fps_error
            best_factor = factor

            if debug:
                print(f'best_downsampled_fps: {best_downsampled_fps:.2f}')
                print('new best factor')

        elif (best_divisor is not None and
            abs(fps_error) > abs(best_fps_error)):

            assert allow_overshoot
            if debug:
                print('breaking because past best factor')
            break

        if debug:
            print('')

    assert best_divisor is not None

    # TODO unit test for this case
    if best_divisor == 1:
        raise ValueError('best downsampling with this flags at factor of 1')

    if debug:
        print(f'best_divisor: {best_divisor}')
        print(f'best_factor: {best_factor:.2f}')
        print(f'best_fps_error: {best_fps_error:.2f}')
        print(f'best_downsampled_fps: {best_downsampled_fps:.2f}')

    frame_shape = movie.shape[1:]
    new_n_frames = n_frames // best_divisor

    # see: stackoverflow.com/questions/15956309 for how to adapt this
    # to uneven division case
    downsampled = movie.reshape((new_n_frames, best_divisor) + frame_shape
        ).mean(axis=1)

    # TODO maybe it's obvious, but is there any kind of guarantee dimensions in
    # frame_shape will not be screwed up in a way relevant to the average
    # when reshaping?
    # well, at least this looks reasonable:
    # image_grid(downsampled[:64])

    return downsampled, best_downsampled_fps


# TODO maybe move to ijroi
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


def tiff_title(tif):
    """Returns abbreviation of TIFF filename for use in titles.
    """
    parts = [x for x in tif.split('/')[-4:] if x != 'tif_stacks']
    ext = '.tif'
    if parts[-1].endswith(ext):
        parts[-1] = parts[-1][:-len(ext)]
    return '/'.join(parts)


# TODO didn't i have some other fn for this? delete one if so
# (or was it just in natural_odors?)
def to_filename(title):
    return title.replace('/','_').replace(' ','_').replace(',','').replace(
        '.','') + '.'


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


def correspond_rois(left_centers_or_seq, *right_centers, cost_fn=euclidean_dist,
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
                    fname = to_filename(ptitle) + plot_format
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
            fname = to_filename(title + extra) + plot_format
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
                dist = euclidean_dist(cxy, last_xy)
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


# Adapted from Vishal's answer at https://stackoverflow.com/questions/287871
_color_codes = {
    'red': '31',
    'green': '32',
    'yellow': '33',
    'blue': '34',
    'cyan': '36'
}
def start_color(color_name):
    try:
        color_code = _color_codes[color_name]
    except KeyError as err:
        print('Available colors are:')
        pprint(list(_color_codes.keys()))
        raise
    print('\033[{}m'.format(color_code), end='')


def stop_color():
    print('\033[0m', end='')


def print_color(color_name, *args, **kwargs):
    start_color(color_name)
    print(*args, **kwargs, end='')
    stop_color()


def latest_trace_pickles():
    """Returns (date, fly, id) indexed DataFrame w/ filename and timestamp cols.

    Only returns rows for filenames that had the latest timestamp for the
    combination of index values.
    """
    def vars_from_filename(tp_path):
        final_part = split(tp_path)[1][:-2]

        # Note that we have lost any more precise time resolution, so an 
        # exact search for this timestamp in database would fail.
        n_time_chars = len('YYYYMMDD_HHMM')
        run_at = pd.Timestamp(datetime.strptime(final_part[:n_time_chars],
            '%Y%m%d_%H%M'
        ))

        parts = final_part.split('_')[2:]
        date = pd.Timestamp(datetime.strptime(parts[0], date_fmt_str))
        fly_num = int(parts[1])
        thorimage_id = '_'.join(parts[2:])
        return date, fly_num, thorimage_id, run_at, tp_path

    keys = ['date', 'fly_num', 'thorimage_id']
    tp_root = join(analysis_output_root(), 'trace_pickles')
    tp_data = [vars_from_filename(f) for f in glob.glob(join(tp_root, '*.p'))]
    if len(tp_data) == 0:
        raise IOError(f'no trace pickles found under {tp_root}')

    df = pd.DataFrame(columns=keys + ['run_at', 'trace_pickle_path'],
        data=tp_data
    )

    unique_len_before = len(df[keys].drop_duplicates())
    latest = df.groupby(keys).run_at.idxmax()
    df.drop(index=df.index.difference(latest), inplace=True)
    assert len(df[keys].drop_duplicates()) == unique_len_before

    df.set_index(keys, inplace=True)
    return df


def add_group_id(df, group_keys, name=None, start_at_one=True):
    """Adds integer column to df to identify unique combinations of group_keys.
    """
    if name is None:
        name = '_'.join(group_keys) + '_id'
    assert name not in df.columns
    df[name] = df.groupby(group_keys).ngroup()

    if start_at_one:
        df[name] = df[name] + 1

    return df


def add_fly_id(df):
    name = 'fly_id'
    return add_group_id(df, recording_cols[:2], name=name)


def add_recording_id(df):
    name = 'recording_id'
    return add_group_id(df, recording_cols, name=name)


# My attempt at writing an Unpickler that loads all objects that wouldn't
# raise errors during unpickling (some pandas objects do if there is a different
# version), comes from: https://stackoverflow.com/questions/46857615
class UnpickleableObject:
    pass


# TODO TODO any way to figure out which pandas version it is?
# or any way to load just the data that is err-ing w/ the pandas
# load(read?)_pickle fn, which claims to maintain compatibility across
# versions

# TODO TODO TODO maybe at the same time as i implement some of my own caching
# decorators / fns again, also implement something to maybe save all the pandas
# / numpy stuff each to their own pickles, in like a zip file format, so i have
# a better chance of loading those things in some compat mode if naive
# unpickling fails (or write fns to fix broken pickles)
# TODO maybe save numpy / pandas version strings as keys of dict at top level
# or in specially named variables that are not returned (if as part of some
# caching fns of mine), to help in fixing broken pickles
# TODO or just save all version / git info in place of just explicitly the numpy
# and pandas stuff... (in case some object from a module i didn't anticipate has
# similar problems)

unpickler_class = pickle_compat.Unpickler
orig_find_class = unpickler_class.find_class

def find_class(self, module, name):
    print('module:', module)
    print('name:', name)
    return orig_find_class(self, module, name)
    '''
    try:
        return super(Unpickler, self).find_class(module, name)
    except AttributeError as e:
        print(e)
        import ipdb; ipdb.set_trace()
        return UnpickleableObject
    print()
    '''
unpickler_class.find_class = find_class


'''
# TODO need to subclass pickle._Unpickler, like in SO link above?
# (don't think so)
class Unpickler(pickle.Unpickler):
    debugger = True
    def find_class(self, module, name):
        print('module:', module)
        print('name:', name)
        try:
            return super(Unpickler, self).find_class(module, name)
        except AttributeError as e:
            print(e)
            if self.debugger:
                import ipdb; ipdb.set_trace()
            return UnpickleableObject
        print()
unpickler_class = Unpickler
'''


def unpickler_load(file_obj):
    return unpickler_class(file_obj).load()


