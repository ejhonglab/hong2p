"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
import socket
import pickle
import atexit
import signal
import sys

# TODO or just use sqlalchemy.types?
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import sqltypes
from sqlalchemy.dialects import postgresql
import numpy as np
import pandas as pd
import git
import pkg_resources


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

    evil = matlab.engine.start_matlab()
    # TODO TODO this doesn't seem to kill parallel workers... it should
    # (happened in case where there was a memory error. visible in top output.)
    # TODO work inside a fn?
    atexit.register(evil.quit)

    exclude_from_matlab_path = {'CaImAn-MATLAB','matlab_helper_functions'}
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


def mb_team_gsheet(use_cache=False, show_inferred_paths=False):
    '''Returns a pandas.DataFrame with data on flies and MB team recordings.
    '''
    gsheet_cache_file = '.gsheet_cache.p'
    if use_cache and os.path.exists(gsheet_cache_file):
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
            'used_for_analysis',
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

    # TODO TODO warn / fail if 'used_for_analysis' and either discard / lost is
    # checked

    # Not sure where there were any NaN here anyway...
    df.raw_data_lost = df.raw_data_lost.fillna(False)
    df.drop(df[df.raw_data_lost].index, inplace=True)

    keys = ['date', 'fly_num']
    df['recording_num'] = df.groupby(keys).cumcount() + 1
    # Since otherwise cumcount seems to be zero for stuff without a group...
    # (i.e. w/ one of the keys null)
    df.loc[pd.isnull(df[keys]).any(axis=1), 'recording_num'] = np.nan

    # TODO delete hack after dealing w/ remy's conventions (some of which were
    # breaking the code assuming my conventions)
    df.drop(df[df.project != 'natural_odors'].index, inplace=True)

    if show_inferred_paths:
        missing_thorimage = pd.isnull(df.thorimage_dir)
        missing_thorsync = pd.isnull(df.thorsync_dir)


    df['thorimage_num'] = df.thorimage_dir.apply(lambda x:
        np.nan if pd.isnull(x) else int(x[1:]))
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
    df.thorimage_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else'_{:03d}'.format(int(x))), inplace=True)

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

    # TODO maybe flag to not update database? or just don't?
    sheets['fly_preps'].dropna(subset=['date','fly_num'], inplace=True)
    to_sql_with_duplicates(sheets['fly_preps'].rename(
        columns={'date': 'prep_date'}), 'flies')

    return df


def git_hash(repo_file):
    repo = git.Repo(repo_file, search_parent_directories=True)
    current_hash = repo.head.object.hexsha
    return current_hash


def caiman_version_info():
    # TODO generalize to version_info / take module / module name?
    try:
        import caiman
        pkg_path = caiman.__file__
        repo = git.Repo(pkg_path, search_parent_directories=True)
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
            'git_remote': remote_url,
            'git_hash': current_hash,
            'git_uncommitted_changes': changes
        }

    except git.exc.InvalidGitRepositoryError:
        # TODO this the right name? try in conda? how does this error?
        version = pkg_resources.get_distribution('caiman').version

        return {'version': version}

