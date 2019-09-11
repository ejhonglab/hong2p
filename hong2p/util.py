"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
from os import listdir
from os.path import join, split, exists, sep, isdir
import socket
import pickle
import atexit
import signal
import sys
import xml.etree.ElementTree as etree
from types import ModuleType
from datetime import datetime
import warnings
import pprint
import glob
import re

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import sqltypes
from sqlalchemy.dialects import postgresql
import numpy as np
from numpy.ma import MaskedArray
import pandas as pd
import git
import pkg_resources
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
import tifffile
import cv2
# TODO maybe use cv2 and get rid of this dep?
from skimage import color
import matplotlib.patches as patches
# is just importing this potentially going to interfere w/ gui?
# put import behind paths that use it?
import matplotlib.pyplot as plt
import matlab.engine
import seaborn as sns


recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
# TODO delete / generalize to more than just name1/name2 case (or hack bolted on
# top of that)? (want to support more than pair experiments)
# TODO might need to add analysis here (although if i'm using it before upload,
# it should always be most recent anyway...)
trial_only_cols = [
    'comparison',
    'name1',
    'name2',
    'repeat_num'
]

trial_cols = recording_cols + trial_only_cols

odor2abbrev = {
    'ethyl acetate': 'eta',
    'ethyl butyrate': 'etb',
    'isoamyl alcohol': 'iaol',
    'ethanol': 'etol',
    'isoamyl acetate': 'iaa'
}

db_hostname = 'atlas'
our_hostname = socket.gethostname()
if our_hostname == db_hostname:
    url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
else:
    url = ('postgresql+psycopg2://tracedb:tracedb@{}' +
        ':5432/tracedb').format(db_hostname)

conn = create_engine(url)

meta = MetaData()
meta.reflect(bind=conn)

# Module level cache.
_nas_prefix = None
def nas_prefix():
    global _nas_prefix
    if _nas_prefix is None:
        # TODO separate env var for local one? or have that be the default?
        nas_prefix_key = 'HONG_NAS'
        if nas_prefix_key in os.environ:
            prefix = os.environ[nas_prefix_key]
        else:
            prefix = '/mnt/nas'
        _nas_prefix = prefix
    # TODO TODO err if nothing in nas_prefix, saying which env var to set and
    # how
    return _nas_prefix


# TODO (for both below) support a local and a remote one ([optional] local copy
# for faster repeat analysis)?
# TODO use env var like kc_analysis currently does for prefix after refactoring
# (include mb_team in that part and rename from nas_prefix?)
def raw_data_root():
    return join(nas_prefix(), 'mb_team/raw_data')


def analysis_output_root():
    return join(nas_prefix(), 'mb_team/analysis_output')


def stimfile_root():
    return join(nas_prefix(), 'mb_team/stimulus_data_files')


def matlab_exit_except_hook(exctype, value, traceback):
    if exctype == TypeError:
        args = value.args
        # This message is what MATLAB says in this case.
        if (len(args) == 1 and
            args[0] == 'exit expected at most 1 arguments, got 2'):
            return
    sys.__excepthook__(exctype, value, traceback)


# TODO maybe rename to init_matlab and return nothing, to be more clear that
# other fns here are using it behind the scenes?
def matlab_engine():
    """
    Gets an instance of MATLAB engine w/ correct paths for Remy's single plane
    code.
    
    Tries to undo Ctrl-C capturing that MATLAB seems to do.
    """
    import matlab.engine
    global evil

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
            d not in exclude_from_matlab_path)]

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
    global evil
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


# TODO maybe just wrap get_matfile_var?
def load_mat_timing_information(mat_file):
    """Loads and returns timing information from .mat output of Remy's script.

    Raises matlab.engine.MatlabExecutionError
    """
    # TODO this sufficient w/ global above to get access to matlab engine in
    # here?
    global evil
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
    return evil.eval('data.ti')


# TODO TODO can to_sql with pg_upsert replace this? what extra features did this
# provide?
def to_sql_with_duplicates(new_df, table_name, index=False, verbose=False):
    # TODO TODO document what index means / delete

    # TODO TODO if this fails and time won't be saved on reinsertion, any rows
    # that have been inserted already should be deleted to avoid confusion
    # (mainly, for the case where the program is interrupted while this is
    # running)
    # TODO TODO maybe have some cleaning step that checks everything in database
    # has the correct number of rows? and maybe prompts to delete?

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
        pprint.pprint(dtypes)
   
    df_types = new_df.dtypes.to_dict()
    if index:
        df_types.update({n: new_df.index.get_level_values(n).dtype
            for n in new_df.index.names})

    if verbose:
        print('\nOld dataframe column types:')
        pprint.pprint(df_types)

    sqlalchemy2pd_type = {
        'INTEGER()': np.dtype('int32'),
        'SMALLINT()': np.dtype('int16'),
        'REAL()': np.dtype('float32'),
        'DOUBLE_PRECISION(precision=53)': np.dtype('float64'),
        'DATE()': np.dtype('<M8[ns]')
    }
    if verbose:
        print('\nSQL types to cast:')
        pprint.pprint(sqlalchemy2pd_type)

    new_df_types = {n: sqlalchemy2pd_type[repr(t)] for n, t in dtypes.items()
        if repr(t) in sqlalchemy2pd_type}

    if verbose:
        print('\nNew dataframe column types:')
        pprint.pprint(new_df_types)

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
        #pprint.pprint(new_index_types)

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
    # https://github.com/pandas-dev/pandas/issues/14553
    for row in data_iter:
        row_dict = dict(zip(keys, row))
        sqlalchemy_table = meta.tables[table.name]
        stmt = postgresql.insert(sqlalchemy_table).values(**row_dict)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=table.index,
            set_=row_dict)
        conn.execute(upsert_stmt)


_mb_team_gsheet = None
def mb_team_gsheet(use_cache=False, show_inferred_paths=False,
    natural_odors_only=False):
    '''Returns a pandas.DataFrame with data on flies and MB team recordings.
    '''
    global _mb_team_gsheet
    if _mb_team_gsheet is not None:
        return _mb_team_gsheet

    gsheet_cache_file = '.gsheet_cache.p'
    if use_cache and exists(gsheet_cache_file):
        print('Loading Google sheet data from cache at {}'.format(
            gsheet_cache_file))

        with open(gsheet_cache_file, 'rb') as f:
            sheets = pickle.load(f)

    else:
        # TODO TODO maybe env var pointing to this? or w/ link itself?
        # TODO maybe just get relative path from __file__ w/ /.. or something?
        pkg_data_dir = split(split(__file__)[0])[0]
        with open(join(pkg_data_dir, 'google_sheet_link.txt'), 'r') as f:
            gsheet_link = \
                f.readline().split('/edit')[0] + '/export?format=csv&gid='

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

            # TODO complain if there are any missing fly_nums

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

    df[['date','fly_num']] = df[['date','fly_num']].fillna(method='ffill')

    df.raw_data_discarded = df.raw_data_discarded.fillna(False)
    # TODO say when this happens?
    df.drop(df[df.raw_data_discarded].index, inplace=True)

    # TODO TODO warn / fail if 'attempt_analysis' and either discard / lost is
    # checked

    # Not sure where there were any NaN here anyway...
    df.raw_data_lost = df.raw_data_lost.fillna(False)
    df.drop(df[df.raw_data_lost].index, inplace=True)

    keys = ['date', 'fly_num']
    df['recording_num'] = df.groupby(keys).cumcount() + 1
    # Since otherwise cumcount seems to be zero for stuff without a group...
    # (i.e. w/ one of the keys null)
    df.loc[pd.isnull(df[keys]).any(axis=1), 'recording_num'] = np.nan

    df['stimulus_data_file'] = df['stimulus_data_file'].fillna(method='ffill')
    # TODO delete hack after dealing w/ remy's conventions (some of which were
    # breaking the code assuming my conventions)
    df.drop(df[df.project != 'natural_odors'].index, inplace=True)

    # TODO TODO implement option to (at least) also keep prep checking that
    # preceded natural_odors (or maybe just that was on the same day)
    # (so that i can get all that ethyl acetate data for use as a reference
    # odor)

    if show_inferred_paths:
        missing_thorimage = pd.isnull(df.thorimage_dir)
        missing_thorsync = pd.isnull(df.thorsync_dir)

    def thorimage_num(x):
        if pd.isnull(x) or not (x[0] == '_' and len(x) == 4):
            return np.nan
        try:
            n = int(x[1:])
            return n
        except ValueError:
            return np.nan
        
    df['thorimage_num'] = df.thorimage_dir.apply(thorimage_num)
    df['numbering_consistent'] = \
        pd.isnull(df.thorimage_num) | (df.thorimage_num == df.recording_num)

    # TODO unit test this
    # TODO TODO check that, if there are mismatches here, that they *never*
    # happen when recording num will be used for inference in rows in the group
    # *after* the mismatch
    gkeys = keys + ['thorimage_dir','thorsync_dir','thorimage_num',
                    'recording_num','numbering_consistent']
    for name, group_df in df.groupby(keys):
        '''
        # case 1: all consistent
        # case 2: not all consistent, but all thorimage_dir filled in
        # case 3: not all consistent, but just because thorimage_dir was null
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
        inplace=True)

    df.drop(columns=['thorimage_num','numbering_consistent'], inplace=True)

    # TODO TODO check for conditions in which we might need to renumber
    # recording num? (dupes / any entered numbers along the way that are
    # inconsistent w/ recording_num results)
    # TODO update to handle case where thorimage dir does not start w/
    # _ and is not just 3 digits after that?
    # (see what format other stuff from day is?)
    df.thorimage_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else '_{:03d}'.format(int(x))), inplace=True)

    df.thorsync_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else 'SyncData{:03d}'.format(int(x))),
        inplace=True)

    if show_inferred_paths:
        cols = ['date','fly_num','thorimage_dir','thorsync_dir']
        print('Inferred ThorImage directories:')
        print(df.loc[missing_thorimage, cols])
        print('\nInferred ThorSync directories:')
        print(df.loc[missing_thorsync, cols])
        print('')

    keys = ['date','fly_num']
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

    # TODO drop any 'Unnamed: X' columns

    # TODO maybe flag to not update database? or just don't?
    # TODO groups all inserts into transactions across tables, and as few as
    # possible (i.e. only do this later)?
    to_sql_with_duplicates(sheets['fly_preps'].rename(
        columns={'date': 'prep_date'}), 'flies')

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
        date_dir = row.date.strftime('%Y-%m-%d')
        fly_num = str(int(row.fly_num))
        thorimage_dir = join(raw_data_root(),
            date_dir, fly_num, row.thorimage_dir)
        thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')

        try:
            xml_root = xmlroot(thorimage_xml_path)
        except FileNotFoundError as e:
            continue

        gsdf.loc[i, 'recording_from'] = \
            datetime.fromtimestamp(float(xml_root.find('Date').attrib['uTime']))

    # TODO fail if stuff marked attempt_analysis has missing xml files?
    # or if nothing was found?

    gsdf = gsdf.rename(columns={'date': 'prep_date'})

    return merge_recordings(gsdf, df, verbose=False)


def merge_odors(df, *args):
    if len(args) == 0:
        odors = pd.read_sql('odors', conn)
    elif len(args) == 1:
        odors = args[0]
    else:
        raise ValueError('incorrect number of arguments')

    print('merging with odors table...', end='')
    # TODO way to do w/o resetting index? merge failing to find odor1 or just
    # drop?
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
    return df


def merge_recordings(df, *args, verbose=True):
    if len(args) == 0:
        recordings = pd.read_sql('recordings', conn)
    elif len(args) == 1:
        recordings = args[0]
    else:
        raise ValueError('incorrect number of arguments')

    print('merging with recordings table...', end='')
    len_before = len(df)
    df = df.reset_index(drop=True)

    # TODO explicitly make this a left merge? (want len(df) preserved)
    df = pd.merge(df, recordings,
                  left_on='recording_from', right_on='started_at')

    df.drop(columns=['started_at'], inplace=True)

    # TODO TODO see notes in kc_analysis about sub-recordings and how that
    # will now break this in the recordings table
    # (multiple dirs -> one start time)
    df['thorimage_id'] = df.thorimage_path.apply(lambda x: split(x)[-1])
    assert len_before == len(df), 'merging changed input df length'
    print(' done')
    return df


def arraylike_cols(df):
    """Returns a list of column names that were lists or arrays.
    """
    df = df.select_dtypes(include='object')
    return df.columns[df.applymap(lambda o:
        type(o) is list or isinstance(o, np.ndarray)).all()]


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
    repo = git.Repo(repo_file, search_parent_directories=True)
    current_hash = repo.head.object.hexsha
    return current_hash


# TODO TODO maybe check that remote seems to be valid, and fail if not.
# don't want to assume we have an online (backed up) record of git repo when we
# don't...
def version_info(module_or_path, used_for=''):
    """Takes module or string path to file in Git repo.
    """
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
    date_dir = date.strftime('%Y-%m-%d')
    fly_num = str(fly_num)

    tif_dir = join(analysis_output_root(), date_dir, fly_num, 'tif_stacks')

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
            "WHERE run_at = '{}'".format(pd.Timestamp(run_at)), conn))

        # TODO maybe merge w/ analysis_code (would have to yield multiple rows
        # per segmentation run when multiple code versions referenced)

    seg_runs = pd.concat(seg_runs, ignore_index=True)
    if len(seg_runs) == 0:
        return None

    seg_runs = seg_runs.merge(analysis_runs)
    seg_runs.sort_values('run_at', inplace=True)
    return seg_runs


# TODO use this in other places that normalize to thorimage_ids
def tiff_thorimage_id(tiff_filename):
    # Behavior of os.path.split makes this work even if tiff_filename does not
    # have any directories in it.
    return '_'.join(split(tiff_filename)[1].split('_')[:-1])


# warn if has SyncData in name but fails this?
def is_thorsync_dir(d):
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

    return have_h5 and have_settings


def is_thorimage_dir(d):
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
        elif f == 'Image_0001_0001.raw':
            have_raw = True
        #elif f == split(d)[-1] + '_ChanA.tif':
        #    have_processed_tiff = True

    if have_xml and have_raw:
        return True
    else:
        return False


# TODO still work w/ parens added around initial .+ ? i want to match the parent
# id...
shared_subrecording_regex = '(.+)_\db\d_from_(nr|rig)'
def is_subrecording(thorimage_id):
    if re.search(shared_subrecording_regex + '$', thorimage_id):
        return True
    else:
        return False


def is_subrecording_tiff(tiff_filename):
    # TODO technically, nr|rig should be same across two...
    if re.search(shared_subrecording_regex + '_(nr|rig).tif$', tiff_filename):
        return True
    else:
        return False


def subrecording_tiff_blocks(tiff_filename):
    """Requires that is_subrecording_tiff(tiff_filename) would return True.
    """
    parts = tiff_filename.split('_')[-4].split('b')

    first_block = int(parts[0]) - 1
    last_block = int(parts[1]) - 1

    return first_block, last_block


def subrecording_tiff_blocks_df(series):
    """
    series.name must be a TIFF path
    """
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
    last_part = split(tiffname_or_thorimage_id)[1]
    match = re.search(shared_subrecording_regex, last_part)
    if match is None:
        raise ValueError('not a subrecording')
    return match.group(1)
        

def accepted_blocks(analysis_run_at, verbose=False):
    """
    """
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

    # TODO TODO TODO TODO also calculate and return uploaded_block_info
    # based on whether a given block has (all) of it's presentations and
    # responses entries (whether accepted or not)
    if verbose:
        print('leaving accepted_blocks\n')
    return accepted


def print_all_accepted_blocks():
    """Just for testing behavior of accepted_blocks fn.
    """
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


def get_experiment_xmlroot(directory):
    xml_path = join(directory, 'Experiment.xml')
    return etree.parse(xml_path).getroot()


def get_thorimage_dims_xml(xmlroot):
    """
    """
    lsm_attribs = xmlroot.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x,y)

    # TODO make this None unless z-stepping seems to be enabled
    # + check this variable actually indicates output steps
    #int(xml.find('ZStage').attrib['steps'])
    z = None
    c = None

    return xy, z, c


def get_thorimage_fps_xml(xmlroot):
    """
    """
    lsm_attribs = xmlroot.find('LSM').attrib
    raw_fps = float(lsm_attribs['frameRate'])
    # TODO is this correct handling of averageMode?
    average_mode = int(lsm_attribs['averageMode'])
    if average_mode == 0:
        n_averaged_frames = 1
    else:
        n_averaged_frames = int(lsm_attribs['averageNum'])
    saved_fps = raw_fps / n_averaged_frames
    return saved_fps


def get_thorimage_fps(thorimage_directory):
    """
    """
    xmlroot = get_experiment_xmlroot(thorimage_directory)
    return get_thorimage_fps_xml(xmlroot)


def xmlroot(xml_filename):
    return etree.parse(xml_filename).getroot()


def tif2xml_root(filename):
    """Returns etree root of ThorImage XML settings from TIFF filename.
    """
    if filename.startswith(analysis_output_root()):
        filename = filename.replace(analysis_output_root(), raw_data_root())

    parts = filename.split(sep)
    thorimage_id = '_'.join(parts[-1].split('_')[:-1])

    xml_fname = sep.join(parts[:-2] + [thorimage_id, 'Experiment.xml'])
    return xmlroot(xml_fname)


# TODO TODO rename this one to make it clear why it's diff from above
# + how to use it (or just delete one...)
def fps_from_thor(df):
    # TODO assert unique first?
    thorimage_dir = df['thorimage_path'].iat[0]
    # TODO TODO TODO why is it [3:] again?? does this depend on my current
    # particular nas_prefix (make it so it does not, if so!)?
    thorimage_dir = join(nas_prefix(), *thorimage_dir.split('/')[3:])
    fps = get_thorimage_fps(thorimage_dir)
    return fps


def cnmf_metadata_from_thor(filename):
    """Takes TIF filename to key settings from XML at fixed relative position.
    """
    xml_root = tif2xml_root(filename)
    fps = get_thorimage_fps_xml(xml_root)
    # "spatial resolution of FOV in pixels per um" "(float, float)"
    # TODO do they really mean pixel/um, not um/pixel?
    pixels_per_um = 1 / float(xml_root.find('LSM').attrib['pixelSizeUM'])
    dxy = (pixels_per_um, pixels_per_um)
    # TODO maybe load dims anyway?
    return {'fr': fps, 'dxy': dxy}


def load_thorimage_metadata(thorimage_directory):
    """
    """
    xml = get_experiment_xmlroot(thorimage_directory)

    fps = get_thorimage_fps_xml(xml)
    xy, z, c = get_thorimage_dims_xml(xml)
    imaging_file = join(thorimage_directory, 'Image_0001_0001.raw')

    return fps, xy, z, c, imaging_file


# TODO TODO use this to convert to tifs, which will otherwise read the same as
# those saved w/ imagej
def read_movie(thorimage_dir):
    """Returns (t,x,y) indexed timeseries.
    """
    fps, xy, z, c, imaging_file = load_thorimage_metadata(thorimage_dir)
    x, y = xy

    # From ThorImage manual: "unsigned, 16-bit, with little-endian byte-order"
    dtype = np.dtype('<u2')

    with open(imaging_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)

    n_frame_pixels = x * y
    n_frames = len(data) // n_frame_pixels
    assert len(data) % n_frame_pixels == 0, 'apparent incomplete frames'

    data = np.reshape(data, (n_frames, x, y))
    return data


def write_tiff(tiff_filename, movie):
    """Write a tiff loading the same as the TIFFs we create with ImageJ.
    """
    # TODO actually make sure any metadata we use is the same
    # TODO maybe just always do test from test_readraw here?
    # (or w/ flag to disable the check)
    tifffile.imsave(tiff_filename, movie)


def full_frame_avg_trace(movie):
    # TODO handle 2d+t or 3d+t data as well
    # (axis=(1,2) just works for 2d+t data)
    return np.mean(movie, axis=(1,2))


def crop_to_nonzero(matrix, margin=0):
    """
    Returns a matrix just large enough to contain the non-zero elements of the
    input, and the bounding box coordinates to embed this matrix in a matrix
    with indices from (0,0) to the max coordinates in the input matrix.
    """
    coords = np.argwhere(matrix > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    x_min = x_min - margin
    x_max = x_max + margin
    y_min = y_min - margin
    y_max = y_max + margin

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


# TODO better name?
def db_row2footprint(db_row, shape=None):
    """Returns dense array w/ footprint from row in cells table.
    """
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


def ijrois2masks(ijrois, shape, dims_as_cnmf=False):
    """
    Transforms ROIs loaded from my ijroi fork to an array full of boolean masks,
    of dimensions (shape + (n_rois,)).
    """
    # TODO maybe index final pandas thing by ijroi name (before .roi prefix)
    # (or just return np array indexed as CNMF "A" is)
    masks = [imagej2py_coords(contour2mask(c, shape[::-1])) for _, c in ijrois]
    masks = np.stack(masks, axis=-1)
    # (actually, putting off the below for now. just gonna not also reshape this
    # output as we currently reshape CNMF A before using it for other stuff)
    if dims_as_cnmf:
        # TODO check that reshaping is not breaking association to components
        # (that it is equivalent to repeating reshape w/in each component and
        # then stacking)
        # TODO TODO conform shape to cnmf output shape (what's that dim order?)
        # n_pixels x n_components, w/ n_pixels reshaped from ixj image "in F
        # order"
        #import ipdb; ipdb.set_trace()
        raise NotImplementedError
    # TODO maybe normalize here?
    # (and if normalizing, might as well change dtype to match cnmf output?)
    # and worth casting type to bool, rather than keeping 0/1 uint8 array?
    return masks


def imagej2py_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
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


def extract_traces_boolean_footprints(movie, footprints):
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
    # TODO vectorized way to do this?
    n_footprints = footprints.shape[-1]
    traces = np.empty((n_frames, n_footprints)) * np.nan
    print('extracting traces from boolean masks...', end='', flush=True)
    for i in range(n_footprints):
        mask = footprints[slices + (i,)]
        # TODO compare time of this to sparse matrix dot product?
        # + time of MaskedArray->mean w/ mask expanded by n_frames?

        # TODO TODO is this correct? check
        # axis=1 because movie[:, mask] only has two dims (frames x pixels)
        trace = np.mean(movie[:, mask], axis=1)
        assert len(trace.shape) == 1 and len(trace) == n_frames
        traces[:, i] = trace
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
    if 'cell' in df.index.names:
        return df.index.get_level_values('cell').unique().to_series()
    elif 'cell' in df.columns:
        cids = pd.Series(data=df['cell'].unique(), name='cell')
        cids.index.name = 'cell'
        return cids
    else:
        raise ValueError("'cell' not in index or columns of DataFrame")


def matlabels(df, rowlabel_fn):
    return df.index.to_frame().apply(rowlabel_fn, axis=1)


def format_odor_conc(name, log10_conc):
    if log10_conc is None:
        return name
    else:
        # TODO tex formatting for exponent
        #return r'{} @ $10^{{'.format(name) + '{:.2f}}}$'.format(log10_conc)
        return '{} @ $10^{{{:.2f}}}$'.format(name, log10_conc)


def format_mixture(*args):
    log10_c1 = None
    log10_c2 = None
    if len(args) == 2:
        n1, n2 = args
    elif len(args) == 4:
        n1, n2, log10_c1, log10_c2 = args
    elif len(args) == 1:
        row = args[0]
        n1 = row['name1']
        try:
            n2 = row['name2']
        except KeyError:
            n2 = None
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


# TODO rename to be inclusive of cases other than pairs
def pair_ordering(comparison_df):
    """Takes a df w/ name1 & name2 to a dict of their tuples to order int.
    """
    # TODO maybe assert only 3 combinations of name1/name2
    pairs = [(x.name1, x.name2) for x in
        comparison_df[['name1','name2']].drop_duplicates().itertuples()]

    # Will define the order in which odor pairs will appear, left-to-right,
    # in subplots.
    ordering = dict()

    has_paraffin = [p for p in pairs if 'paraffin' in p]
    if len(has_paraffin) == 0:
        assert {x[1] for x in pairs} == {'no_second_odor'}
        odors = [p[0] for p in pairs]

        # TODO TODO also support case where there isn't something we want to
        # stick at the end like this, for Matt's case
        last = None
        for o in odors:
            lo = o.lower()
            if 'approx' in lo or 'mix' in lo:
                if last is None:
                    last = o
                else:
                    raise ValueError('multiple mixtures in odors to order')
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

    return ordering


def matshow(df, title=None, ticklabels=None, xticklabels=None,
    yticklabels=None, xtickrotation=None, colorbar_label=None,
    group_ticklabels=False, ax=None, fontsize=None, fontweight=None):
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
            assert df.shape[0] == df.shape[1]
            # TODO maybe also assert indices are actually equal?
            xticklabels = ticklabels
            yticklabels = ticklabels
    else:
        # TODO delete this hack
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

    avg = None
    for i, cell_id in enumerate(cells):
        if verbose:
            print('Plotting cell {}/{}...'.format(i + 1, len(cells)))

        cell_data = display_window[display_window.cell == cell_id]
        cell_trials = cell_data.groupby(group_cols, sort=False)[
            ['from_onset','df_over_f']]

        prep_date = pd.Timestamp(cell_data.prep_date.unique()[0])
        date_dir = prep_date.strftime('%Y-%m-%d')
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

            ax = axs[i,j]

            # So that events that get the axes can translate to cell /
            # trial information.
            ax.cell_id = cell_id
            ax.trial_info = n

            # X and Y axis major tick label fontsizes.
            # Was left to default for kc_analysis.
            ax.tick_params(labelsize=6)

            trial_times = cell_trial['from_onset']
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

            if i == 0:
                # TODO TODO group in odors case as w/ matshow
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

                ax.set_title(trial_title, fontsize=6) #, rotation=90)
                # TODO may also need to do tight_layout here...
                # it apply to these kinds of titles?

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

