"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
from os.path import join, split, exists, sep
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

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import sqltypes
from sqlalchemy.dialects import postgresql
import numpy as np
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


recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
# TODO might need to add analysis here (although if i'm using it before upload,
# it should always be most recent anyway...)
trial_only_cols = [
    'comparison',
    'name1',
    'name2',
    'repeat_num'
]

trial_cols = recording_cols + trial_only_cols
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


def mb_team_gsheet(use_cache=False, show_inferred_paths=False,
    natural_odors_only=False):
    '''Returns a pandas.DataFrame with data on flies and MB team recordings.
    '''
    gsheet_cache_file = '.gsheet_cache.p'
    if use_cache and exists(gsheet_cache_file):
        print('Loading Google sheet data from cache at {}'.format(
            gsheet_cache_file))

        with open(gsheet_cache_file, 'rb') as f:
            sheets = pickle.load(f)

    else:
        with open('google_sheet_link.txt', 'r') as f:
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

    # TODO handle case where database is empty but gsheet cache still exists
    # (all inserts will probably fail, for lack of being able to reference fly
    # table)

    return df


def merge_gsheet(df, *args, use_cache=False):
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

    df = df.reset_index(drop=True)

    df = pd.merge(df, recordings,
                  left_on='recording_from', right_on='started_at')

    df.drop(columns=['started_at'], inplace=True)

    df['thorimage_id'] = df.thorimage_path.apply(lambda x: split(x)[-1])

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
    # TODO what does averageMode = 1 mean? always like that?
    # 
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
    thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')
    xml_root = etree.parse(thorimage_xml_path).getroot()
    lsm = xml_root.find('LSM').attrib
    fps = float(lsm['frameRate']) / float(lsm['averageNum'])
    return fps


def cnmf_metadata_from_thor(filename):
    """Takes TIF filename to key settings from XML at fixed relative position.
    """
    xml_root = tif2xml_root(filename)
    lsm = xml_root.find('LSM').attrib
    fps = float(lsm['frameRate']) / float(lsm['averageNum'])
    # "spatial resolution of FOV in pixels per um" "(float, float)"
    # TODO do they really mean pixel/um, not um/pixel?
    pixels_per_um = 1 / float(lsm['pixelSizeUM'])
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
        # TODO maybe check we actually get enough bytes for n_frames?
        data = np.fromfile(f, dtype=dtype)

    data = np.reshape(data, (n_frames, x, y))
    return data


def crop_to_nonzero(matrix, margin=0):
    coords = np.argwhere(matrix > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    x_min = x_min - margin
    x_max = x_max + margin
    y_min = y_min - margin
    y_max = y_max + margin

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


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


def missing_repeats(df, n_repeats=None):
    # TODO n_repeats defalut to 3 or None?
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

    # TODO TODO TODO are comparisons numbered correctly when i break the full
    # panel across recordings within a fly flies??????
    # (shouldn't they not start at 0 on the next recording?)
    return missing_repeats_df


def have_all_repeats(df, n_repeats=None):
    """
    Returns True if a recording has all blocks gsheet says it has, w/ full
    number of repeats for each. False otherwise.
    """
    missing_repeats_df = missing_repeats(df, n_repeats=n_repeats)
    if len(missing_repeats_df) == 0:
        return True
    else:
        return False


# TODO TODO TODO separately, also check correct # blocks / comparisons per fly
# (including that comparisons from diff recordings don't overwrite each other)
def missing_odor_pairs(df):
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
    if len(missing_odor_pairs(df)) == 0:
        return True
    else:
        return False


def skipped_comparison_nums(df):
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
        n2 = row['name2']
        if 'log10_conc_vv1' in row:
            log10_c1 = row['log10_conc_vv1']
            log10_c2 = row['log10_conc_vv2']
    else:
        raise ValueError('incorrect number of args')

    if n1 == 'paraffin':
        title = format_odor_conc(n2, log10_c2)
    elif n2 == 'paraffin':
        title = format_odor_conc(n1, log10_c1)
    else:
        title = '{} + {}'.format(
            format_odor_conc(n1, log10_c1),
            format_odor_conc(n2, log10_c2)
        )

    return title


def pair_ordering(comparison_df):
    """Takes a df w/ name1 & name2 to a dict of their tuples to order int.
    """
    # TODO maybe assert only 3 combinations of name1/name2
    pairs = [(x.name1, x.name2) for x in
        comparison_df[['name1','name2']].drop_duplicates().itertuples()]

    has_paraffin = [p for p in pairs if 'paraffin' in p]
    no_pfo = [p for p in pairs if 'paraffin' not in p]

    if len(no_pfo) < 1:
        raise ValueError('All pairs for this comparison had paraffin.' +
            ' Analysis error? Incomplete recording?')

    assert len(no_pfo) == 1
    last = no_pfo[0]

    # Will define the order in which odor pairs will appear, left-to-right,
    # in subplots.
    ordering = dict()
    ordering[last] = 2

    for i, p in enumerate(sorted(has_paraffin,
        key=lambda x: x[0] if x[1] == 'paraffin' else x[1])):

        ordering[p] = i

    return ordering


def matshow(df, title=None, ticklabels=None, colorbar_label=None,
    group_ticklabels=False, ax=None, fontsize=None):
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
        if all([len(x) == 1 for x in xticklabels]):
            xtickrotation = 'horizontal'
        else:
            xtickrotation = 'vertical'

        ax.set_xticklabels(xticklabels, fontsize=fontsize,
            rotation=xtickrotation)
        #    rotation='horizontal' if group_ticklabels else 'vertical')
        ax.set_xticks(np.arange(0, len(df.columns), xstep) + xoffset)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=fontsize,
            rotation='horizontal')
        #    rotation='vertical' if group_ticklabels else 'horizontal')
        ax.set_yticks(np.arange(0, len(df), ystep) + yoffset)

    if title is not None:
        ax.set_xlabel(title)

    if made_fig:
        plt.tight_layout()
        return fig


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
    odor_panel = None
    if len(args) == 1:
        odor_panel = args[0]
    elif len(args) != 0:
        raise ValueError('incorrect number of arguments')
    # TODO maybe make df optional and read from database if it's not passed?
    # TODO a flag to show all stuff marked attempt analysis in gsheet?

    # TODO borrow more of this / call this in part of kc_analysis that made that
    # table w/ these counts for repeats?

    df = df.drop(
        index=df[(df.name1 == 'paraffin') | (df.name2 == 'paraffin')].index)

    replicates = df[[
        'prep_date',
        'fly_num',
        'thorimage_id',
        'comparison',
        'name1',
        'name2']].drop_duplicates()

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
        odor_panel = odor_panel.pivot_table(index='odor_1', columns='odor_2',
            aggfunc=lambda x: True, values='reason')

        full_panel_index = odor_panel.index.union(odor_panel.columns)
        full_data_index = pair_n.index.union(pair_n.columns)
        assert full_data_index.isin(full_panel_index).all()
        # TODO also check no pairs occur in data that are not in panel

        import ipdb; ipdb.set_trace()

    # TODO TODO TODO maybe use this combine_first thing on odor_panel
    # and then use those INDICES (but not values, as w/ combine_first)
    # as the indices for full_pair_n!
    # (for case when odor panel has pairs not included here)

    # maybe make colorbar have discrete steps?

    # TODO TODO TODO how to indicate which of the pairs we are actually
    # interested in? grey out the others? white the others? (just set to nan?)
    # (maybe only use to grey / white out if passed in?)
    # (+ margins for now)

    # TODO TODO TODO color code text labels by pair selection reason + key
    # TODO what to do when one thing falls under two reasons though...?
    # just like a key (or things alongside ticklabels) that has each color
    # separately? just symbols in text, if that's easier?

    # TODO TODO display actual counts in squares in matshow

    import ipdb; ipdb.set_trace()



def closed_mpl_contours(footprint, ax, err_on_multiple_comps=True, **kwargs):
    """
    """
    dims = footprint.shape
    padded_footprint = np.zeros(tuple(d + 2 for d in dims))
    padded_footprint[tuple(slice(1,-1) for _ in dims)] = footprint
    
    mpl_contour = ax.contour(padded_footprint > 0, [0.5], **kwargs)
    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1
    paths = mpl_contour.collections[0].get_paths()
    assert len(paths) == 1
    contour = paths[0].vertices

    # Correct index change caused by padding.
    return contour - 1


def plot_traces(*args, footprints=None, order_by='odors', scale_within='none',
    n=20, random=False, title=None, response_calls=None, raw=False,
    smoothed=True, show_footprints=True, show_footprints_alone=False,
    show_cell_ids=True, show_footprint_with_mask=False, gridspec=None,
    verbose=True):
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
    n_trials = n_repeats * 3

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

            weights, x_coords, y_coords = \
                footprint_row[['weights','x_coords','y_coords']]

            footprint = np.array(coo_matrix((weights,
                (x_coords, y_coords)), shape=avg.shape).todense())

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

                ax.plot(trial_times, trial_dff)

            if smoothed:
                # TODO kwarg(s) to control smoothing?
                sdff = smooth(trial_dff, window_len=window_size)

                if ymax is None:
                    ymax = sdff.max()
                    ymin = sdff.min()
                else:
                    ymax = max(ymax, sdff.max())
                    ymin = min(ymin, sdff.min())

                ax.plot(trial_times, sdff, color='black')

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

