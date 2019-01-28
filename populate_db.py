#!/usr/bin/env python3

"""
Traverses analysis output and loads traces and odor information into database.
"""

import os
from os.path import join, split
import glob
from datetime import datetime
import pickle
import warnings
import atexit
import time
import pprint

from sqlalchemy import create_engine
import h5py
import numpy as np
import pandas as pd
import matlab.engine
import git


use_cached_gsheet = True
show_inferred_paths = True
convert_h5 = True
calc_timing_info = False
motion_correct = True
only_motion_correct_for_analysis = True


analysis_started_at = time.time()

# TODO need to add stuff to path? what all?
# TODO future have a bug generally, or only if stopped w/ ctrl-d from ipdb like
# i had?
#future = matlab.engine.start_matlab(async=True)
#evil = None
evil = matlab.engine.start_matlab()
atexit.register(evil.quit)

# The latter is here so I can tell which files within it are actually needed
# in the current version of the analysis.
exclude_from_matlab_path = {'CaImAn-MATLAB','matlab_helper_functions'}
#
userpath = evil.userpath()
for root, dirs, _ in os.walk(userpath, topdown=True):
    dirs[:] = [d for d in dirs if (not d.startswith('.') and
        not d.startswith('@') and not d.startswith('+') and
        d not in exclude_from_matlab_path)]

    evil.addpath(root)

# To get Git version information to have a record of what analysis was
# performed.
matlab_repo_name = 'matlab_kc_plane'
matlab_code_path = join(userpath, matlab_repo_name)
#

url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
conn = create_engine(url)

#matlab_output_file = 'test_data/struct_no_sparsearray.mat'
####matlab_output_file = 'test_data/_007_cnmf.mat'

gsheet_cache_file = '.gsheet_cache.p'
if use_cached_gsheet and os.path.exists(gsheet_cache_file):
    print('Loading Google sheet data from cache at {}'.format(
        gsheet_cache_file))

    with open(gsheet_cache_file, 'rb') as f:
        sheets = pickle.load(f)

else:
    with open('google_sheet_link.txt', 'r') as f:
        gsheet_link = f.readline().split('/edit')[0] + '/export?format=csv&gid='

    # If you want to add more sheets, when you select the new sheet in your
    # browser, the GID will be at the end of the URL in the address bar.
    sheet_gids = {
        'fly_preps': '269082112',
        'recordings': '0',
        'daily_settings': '229338960'
    }

    # TODO flag to cache these to just be nice to google?
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

# TODO say when this happens?
df.drop(df[df.raw_data_discarded].index, inplace=True)
df.drop(df[df.raw_data_lost].index, inplace=True)

keys = ['date', 'fly_num']
df['recording_num'] = df.groupby(keys).cumcount() + 1
# Since otherwise cumcount seems to be zero for stuff without a group...
# (i.e. w/ one of the keys null)
df.loc[pd.isnull(df[keys]).any(axis=1), 'recording_num'] = np.nan

if show_inferred_paths:
    missing_thorimage = pd.isnull(df.thorimage_dir)
    missing_thorsync = pd.isnull(df.thorsync_dir)


df['thorimage_num'] = df.thorimage_dir.apply(lambda x: np.nan if pd.isnull(x)
    else int(x[1:]))
df['numbering_consistent'] = \
    pd.isnull(df.thorimage_num) | (df.thorimage_num == df.recording_num)

# TODO unit test this
# TODO TODO check that, if there are mismatches here, that they *never* happen
# when recording num will be used for inference in rows in the group *after*
# the mismatch
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
    following_thorimage_dirs = group_df.thorimage_dir.iloc[first_mismatch_idx:]
    #print('checking these are not null:\n', following_thorimage_dirs)
    assert pd.notnull(following_thorimage_dirs).all()

df.thorsync_dir.fillna(df.thorimage_num.apply(lambda x: np.nan if pd.isnull(x)
    else 'SyncData{:03d}'.format(int(x))), inplace=True)

df.drop(columns=['thorimage_num','numbering_consistent'], inplace=True)


# TODO TODO check for conditions in which we might need to renumber recording
# num? (dupes / any entered numbers along the way that are inconsistent w/
# recording_num results)
df.thorimage_dir.fillna(df.recording_num.apply(lambda x: np.nan if pd.isnull(x)
    else'_{:03d}'.format(int(x))), inplace=True)

df.thorsync_dir.fillna(df.recording_num.apply(lambda x: np.nan if pd.isnull(x)
    else 'SyncData{:03d}'.format(int(x))), inplace=True)

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

# TODO TODO warn if any raw data is not present on NAS / report which
# (could indicate problems w/ path inference)

# TODO add bool col to recordings to indicate whether stimulus order in python
# file was adhered to?
# TODO or just only use it for natural_odors project for now?
# (probably just want to ignore order for project="n/a (prep checking)" columns
# anyway)

# TODO + summarize those that still need analysis run on them (and run it?)
# TODO print stuff w/ used_for_analysis checked w/o either data in database or
# analysis output on disk (or just latter)

# TODO move these paths to config file...
raw_data_root = '/mnt/nas/mb_team/raw_data'
analysis_output_root = '/mnt/nas/mb_team/analysis_output'

rel_to_cnmf_mat = 'cnmf'

stimfile_root = '/mnt/nas/mb_team/stimulus_data_files' 

natural_odors_concentrations = pd.read_csv('natural_odor_panel_vial_concs.csv')
natural_odors_concentrations.set_index('name', inplace=True)

# TODO TODO loop over more stuff than just natural_odors / used_for_analysis
# to load all PID stuff in (will need pin info for prep checking, etc, exps)

# TODO complain if there are flies w/ used_for_analysis not checked w/o
# rejection reason

# TODO maybe don't err in this case (w/ option to only run on analysis?)
# and symmetric option for analysis root?
if not os.path.isdir(raw_data_root):
    raise IOError('raw_data_root {} does not exist'.format(
        raw_data_root))

if not os.path.isdir(analysis_output_root):
    raise IOError('analysis_output_root {} does not exist'.format(
        analysis_output_root))

if not os.path.isdir(stimfile_root):
    raise IOError('stimfile_root {} does not exist'.format(
        stimfile_root))

# TODO TODO also do a first pass over everything w/ default params so just
# manual correction remains

# TODO make fns that take date + fly_num + cwd then re-use iteration
# over date / fly_num (if mirrored data layout for raw / analysis)?

for full_fly_dir in glob.glob(raw_data_root + '/*/*/'):
    full_fly_dir = os.path.normpath(full_fly_dir)
    #print(full_fly_dir)
    prefix, fly_dir = split(full_fly_dir)
    _, date_dir = split(prefix)

    if fly_dir == 'unsorted':
        # TODO maybe make attempt to sort?
        continue

    try:
        fly_num = int(fly_dir)
    except ValueError:
        # TODO maybe warn if not in whitelist, like 'unsorted')
        continue

    try:
        date = datetime.strptime(date_dir, '%Y-%m-%d')
    except ValueError:
        continue

    #print(date)
    #print(fly_num)

    used = df.loc[df.used_for_analysis &
        (df.date == date) & (df.fly_num == fly_num)]

    # TODO maybe do this in analysis actually? (to not just make a bunch of
    # empty dirs...)
    analysis_fly_dir = join(analysis_output_root, date_dir, fly_dir)
    if not os.path.isdir(analysis_fly_dir):
        # Will also make any necessary parent (date) directories.
        os.makedirs(analysis_fly_dir)

    # TODO maybe use regexp to check syncdata / util fn to check for name +
    # stuff in it?
    if convert_h5:
        ####print('Converting ThorSync HDF5 files to .mat...')
        for syncdir in glob.glob(join(full_fly_dir, 'SyncData*')):
            print(syncdir)
            '''
            if evil is None:
                evil = future.result()
                userpath = evil.userpath()
                paths_before = set(evil.path().split(':'))
                for root, dirs, _ in os.walk(userpath, topdown=True):
                    dirs[:] = [d for d in dirs if (not d.startswith('.') and
                        not d.startswith('@') and not d.startswith('+') and
                        d not in exclude_from_matlab_path)]

                    evil.addpath(root)

                paths_after = set(evil.path().split(':'))
                pprint.pprint(paths_after - paths_before)
            '''

            # TODO make it so one ctrl-c closes whole program, rather than just
            # cancelling matlab function and continuing

            # Will immediately return if output already exists.
            evil.thorsync_h5_to_mat(syncdir, full_fly_dir, nargout=0)
            # TODO (check whether she is loosing information... file size really
            # shouldn't be that different. both hdf5...)

            # TODO do i want flags to disable *each* step separately for
            # unanalyzed stuff? just one flag? always ignore that stuff?

            # TODO remy could also convert xml here if she wanted
        ####print('Done converting ThorSync HDF5 to .mat\n')

    if calc_timing_info:
        for _, row in used[['thorimage_dir','thorsync_dir']].iterrows():
            thorimage_dir = join(full_fly_dir, row['thorimage_dir'])
            if not os.path.isdir(thorimage_dir):
                warnings.warn('thorimage_dir {} did not exist for recording ' +
                    'marked as used_for_analysis.')
                continue

            # If not always running h5->mat conversion first, will need to check
            # for the mat, rather than just thorsync_dir.
            thorsync_dir = join(full_fly_dir, row['thorsync_dir'])
            if not os.path.isdir(thorsync_dir):
                warnings.warn('thorsync_dir {} did not exist for recording ' +
                    'marked as used_for_analysis.')
                continue

            print('\nThorImage and ThorSync dirs for get_stiminfo:')
            print(thorimage_dir)
            print(thorsync_dir)
            print('')

            print(('getting stimulus timing information for {}, {}, {}...'
                ).format( date_dir, fly_num, row['thorimage_dir']), end='')

            '''
            # TODO delete
            print('\n')
            print(date_dir)
            print(date_dir == '2019-01-18')
            print(fly_num)
            print(fly_num == 2)
            print(row['thorsync_dir'])
            print(row['thorsync_dir'] == 'SyncData005')
            if not (date_dir == '2019-01-18' and fly_num == 2 and
                #print('SKIPPING KNOWN FAILING GET_STIMINFO CALL')
                print('SKIPPING LIKELY NON-FAILING GET_STIMINFO CALL')
                    row['thorsync_dir'] == 'SyncData005'):
                continue
            #
            '''

            # throwing everything into _<>_cnmf.mat, as we are, would need to
            # inspect it to check whether we already have the stiminfo...
            evil.get_stiminfo(thorimage_dir, row['thorsync_dir'],
                analysis_fly_dir, date_dir, fly_num, nargout=0)

            print(' done.')

    # TODO loop over thorimage dirs and make tifs from each if they don't exist
    # TODO TODO what all metadata is important in the tif? need to script imagej
    # to get most reliable tifs? some python / matlab fn work?
    '''
    for thorimage_dir in glob.glob(join(full_fly_dir, '_*/')):
        thorimage_id = split(os.path.normpath(thorimage_dir))[-1]
    '''

    # maybe avoid searching for thorimage dirs at all if there are no used 
    # rows for this (date,fly) combo, and only_motion_correct_for_analysis

    # TODO use multiple matlab instances to run normcore on different
    # directories in parallel?
    # TODO exclude stuff that indicates it's either already avg or motion
    # corrected? (or just always keep them separately?)
    if motion_correct:
        for input_tif_path in glob.glob(
            join(full_fly_dir, 'tif_stacks', '_*.tif')):

            if only_motion_correct_for_analysis:
                thorimage_id = split(input_tif_path)[-1][:-4]

                recordings = used[used.thorimage_dir == thorimage_id]
                if len(recordings) == 0:
                    continue

            # TODO only register one way by default? nonrigid? args to
            # configure?
            evil.normcorre(input_tif_path, analysis_fly_dir, nargout=0)

    # TODO and if remy wants, copy thorimage xmls

    print('')

    # TODO maybe delete empty folders under analysis?

import sys; sys.exit()

def git_hash(repo_file):
    repo = git.Repo(repo_file, search_parent_directories=True)
    current_hash = repo.head.object.hexsha
    return current_hash
# TODO maybe also return current remote url, if listed? whether pushed?
# TODO store unsaved changes too? sep column? see my metatools package
'''
diff = repo.index.diff(None, create_patch=True)
exactly = 'exactly ' if len(diff) == 0 else ''
'''

this_repo_file = os.path.realpath(__file__)
this_repo_path = split(this_repo_file)[0]
current_hash = git_hash(this_repo_file)
matlab_hash = git_hash(matlab_code_path)

# TODO just store all data separately?
analysis_description = '{}@{}\n{}@{}'.format(this_repo_path, current_hash,
    matlab_code_path, matlab_hash)

# TODO TODO try inserting w/o analysis_run and seeing if serial gets incremented
# otherwise either do it manually or drop that approach


# TODO diff between ** and */ ?
# TODO os.path.join + os invariant way of looping over dirs
for analysis_dir in glob.glob(analysis_output_root+ '/*/*/'):
    analysis_dir = os.path.normpath(analysis_dir)

    prefix, fly_dir = split(analysis_dir)
    _, date_dir = split(prefix)

    try:
        fly_num = int(fly_dir)
    except ValueError:
        continue

    try:
        date = datetime.strptime(date_dir, '%Y-%m-%d')
    except ValueError:
        continue

    print(analysis_dir)

    # TODO TODO complain if stuff marked as used for analysis is not found here
    for mat in glob.glob(join(analysis_dir, rel_to_cnmf_mat, '*_cnmf.mat')):
        prefix = split(mat)[-1].split('_')[:-1]

        print(mat)

        thorimage_dir = '_' + prefix[1]

        # TODO TODO need to infer missing thor dirs first, if doing it this
        # way...
        recordings = df.loc[(df.date == date) &
                            (df.thorimage_dir == thorimage_dir)]

        if prefix[0] == '':
            # TODO check there is only one fly w/ used_in_analysis checked for
            # this date, then associate that fly num w/ this thorimage id?
            if len(recordings) == 1:
                recording = recordings.iloc[0]
                fly_num = int(recording['fly_num'])

            else:
                # TODO flag to err instead?
                warnings.warn(('{} has no fly_num prefix and the spreadsheet ' +
                    'indicates multiple flies with this date ({}) and ' +
                    'ThorImage directory name ({}). Append proper fly_num' +
                    ' prefix to analysis output.').format(mat, date,
                    thorimage_dir))
                continue

        else:
            fly_num = int(prefix[0])
            recordings = recordings.loc[recordings.fly_num == fly_num]
            if len(recordings) > 1:
                # TODO TODO fix case where this can happen if same fly_num
                # has data saved to two different project directories
                # (maybe by just getting rid of project dirs...)
                raise ValueError(('multiple repeats of fly_num {} for ' +
                    '({}, {})').format(fly_num, date, thorimage_dir))

            elif len(recordings) == 0:
                raise ValueError(('missing expected fly_num {} for ' +
                    '({}, {})').format(fly_num, date, thorimage_dir))

            recording = recordings.iat[0]

        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.')
            continue

        stimfile = recording['stimulus_data_file']
        stimfile_path = join(stimfile_root, stimfile)
        print(stimfile)
        # TODO also err if not readable
        if not os.path.exists(stimfile_path):
            raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
                stimfile_root))

        with open(stimfile_path, 'rb') as f:
            # TODO TODO handle concentration info
            data = pickle.load(f)

            # TODO TODO subset odor order information by start/end block cols
            # (for natural_odors stuff)

        # TODO will need to augment w/ concentration info somehow...
        # maybe handle in a way specific to natural_odors project?

        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })

        odors.to_sql('odors', conn, if_exists='append', index=False)
        # TODO TODO TODO insert and then get ids -> then use that to merge other
        # tables? or what? make unique id before insertion? some way that
        # wouldn't require the IDs, but would create similar tables?

        # TODO calculate from thorsync odor trigger info

        pulse_length = 1.0
        # TODO get from daily_settings table
        odor_flow_slpm = 0.4
        carrier_flow_slpm = 1.6
        #
        volume_ml = 2.0

        # TODO TODO here, i think i probably do want to generate (some) IDs in
        # python (to calculate odors_in_mixtures...)
        # TODO TODO generate a number of IDs equal to the number of odor pairs
        # TODO TODO don't necessarily want to constantly append mixtures
        # though... (if odors_in_mixtures referring to them are the same, they
        # are the same. does 3 table design make figuring out how many rows I
        # need in mixtures complicated?)
        '''
        mixtures = 

        odors_in_mixtures = 
        '''

        import ipdb; ipdb.set_trace()

        # TODO TODO 
        # -> load stimulus file ->
        # combine w/ timing information Remy saves to .mat file to associate
        # traces w/ specific odors


        # TODO maybe use Remy's thorsync timing info to get num pulses for prep
        # checking trials and then assume it's ethyl acetate (for PID purposes,
        # at least)?

        with h5py.File(mat, 'r') as data:
            print(list(data.keys()))

            if 'sCNM' not in data.keys():
                # TODO normally err / flag to
                warnings.warn('no sCNM object in {}'.format(mat))
                continue

            pprint.pprint(list(data['sCNM'].items()))
            pprint.pprint(list(data['S'].items()))

            # block_{f/i}ct - i=initial f=final cross time (scope trigger?)
            # TODO get 'ti' for PID trials too?
            ti = data['ti']
            frame_times = np.array(ti['frame_times'])
            print(np.array(data['sCNM']['file']))
            #print(np.array(data['sCNM']['file']).tostring().decode())

            print(int(np.array(ti['num_stim'])[0,0]))
            print(int(np.array(ti['num_trials'])[0,0]))
            pprint.pprint(list(data['ti'].items()))
            print('')
            # stim_... are similar. of length = to num of stimuli. not grouped
            # by block.

            # spt / fpt = seconds and frames per trial
            # so far, w/ frame averaging, get exactly requested num frames per
            # block (scalar)
            

            # stim_on/off 

            # not sure what value of si is...

            # TODO list of sets / tuples (of names?) to table to sql?
            # TODO out-of-band concentration info?
            # TODO default volume / flows?

            #print(ti['pin_odors'])
            #print(type(ti['pin_odors']))
            
            '''
            print(ti['struct_pin_odors'])
            print(list(ti['struct_pin_odors'].items()))
            print(np.array(ti['struct_pin_odors']['odor']))
            print(np.array(ti['struct_pin_odors']['odor'][0]))
            '''
            '''
            print([dir(x) for x in ti['pin_odors']])
            print(type(ti['pin_odors'][0]))
            # TODO assert this is length 1 if indexing this way
            pin_odor_refs = ti['pin_odors'][0]
            r = pin_odor_refs[0]
            # TODO it seems i need to convert this back to strings...
            print('dereferenced:', [ti[r] for r in pin_odor_refs])
            '''
            # TODO assert x is actually always len 1?
            '''
            channel_A_pins = [int(x[0]) for x in ti['channelA']]
            channel_B_pins = [int(x[0]) for x in ti['channelB']]

            channel_A_odors = [pin2odors[p] for p in channel_A_pins]
            channel_B_odors = [pin2odors[p] for p in channel_B_pins]

            odors = [set(x) for x in zip(channel_A_odors, channel_B_odors)]
            '''

            # TODO how to get odor pid
            # A has footprints
            # dims=dimensions of image (256x256)
            # T is # of timestamps
            #print(data['sCNM'])
            #print('')

            # DFF - traces
            # sDFF - filtered traces (filtered how?)
            # (what she did w/ jon. see source)
            # k - # components (redundant)
            # F0 - background fluorescence (dims?)
            # TODO TODO subset to stuff just starting from onset
            traces = np.array(data['S']['sDFF'])
            #print(traces)
            # TODO get from_onset array of equal length
            # (seconds from odor valve on) (include negative values?)


# TODO print unused stimfiles / option to delete them

"""

with h5py.File(matlab_output_file, 'r') as data:
    print(list(data.keys()))

    pprint.pprint(list(data['sCNM'].items()))
    pprint.pprint(list(data['S'].items()))

    # block_{f/i}ct - i=initial f=final cross time (scope trigger?)
    # TODO get 'ti' for PID trials too?
    ti = data['ti']
    frame_times = np.array(ti['frame_times'])
    print(np.array(data['sCNM']['file']))
    #print(np.array(data['sCNM']['file']).tostring().decode())

    print(int(np.array(ti['num_stim'])[0,0]))
    print(int(np.array(ti['num_trials'])[0,0]))
    pprint.pprint(list(data['ti'].items()))
    print('')
    # stim_... are similar. of length = to num of stimuli. not grouped by
    # block.

    # spt / fpt = seconds and frames per trial
    # so far, w/ frame averaging, get exactly requested num frames per block
    # (scalar)

    # stim_on/off 

    # TODO index (adjust for diff indexing) into pin_odors w/ pin (- 1) to get
    # odor names

    # not sure what value of si is...

    # TODO how does remy enter odor information now?
    # reformat my pickles for her automated use? just convert that part of
    # analysis to python?
    pin2odors = dict()
    # TODO list of sets / tuples (of names?) to table to sql?
    # TODO out-of-band concentration info?
    # TODO default volume / flows?

    #print(ti['pin_odors'])
    #print(type(ti['pin_odors']))
    
    print(ti['struct_pin_odors'])
    print(list(ti['struct_pin_odors'].items()))
    print(np.array(ti['struct_pin_odors']['odor']))
    print(np.array(ti['struct_pin_odors']['odor'][0]))
    '''
    print([dir(x) for x in ti['pin_odors']])
    print(type(ti['pin_odors'][0]))
    # TODO assert this is length 1 if indexing this way
    pin_odor_refs = ti['pin_odors'][0]
    r = pin_odor_refs[0]
    # TODO it seems i need to convert this back to strings...
    print('dereferenced:', [ti[r] for r in pin_odor_refs])
    '''
    # TODO assert x is actually always len 1?
    '''
    channel_A_pins = [int(x[0]) for x in ti['channelA']]
    channel_B_pins = [int(x[0]) for x in ti['channelB']]

    channel_A_odors = [pin2odors[p] for p in channel_A_pins]
    channel_B_odors = [pin2odors[p] for p in channel_B_pins]

    odors = [set(x) for x in zip(channel_A_odors, channel_B_odors)]
    '''

    # TODO how to get odor pid
    # A has footprints
    # dims=dimensions of image (256x256)
    # T is # of timestamps
    #print(data['sCNM'])
    #print('')

    # DFF - traces
    # sDFF - filtered traces (filtered how?)
    # (what she did w/ jon. see source)
    # k - # components (redundant)
    # F0 - background fluorescence (dims?)
    # TODO TODO subset to stuff just starting from onset
    traces = np.array(data['S']['sDFF'])
    #print(traces)
    # TODO get from_onset array of equal length
    # (seconds from odor valve on) (include negative values?)

"""
