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
import pprint

from sqlalchemy import create_engine
import h5py
import numpy as np
import pandas as pd


use_cached_gsheet = True

'''
url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
conn = create_engine(url)
'''

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

    boolean_columns = {'used_for_analysis'}
    na_cols = list(set(sheets['recordings'].columns) - boolean_columns)
    sheets['recordings'].dropna(how='all', subset=na_cols, inplace=True)

    with open(gsheet_cache_file, 'wb') as f:
        pickle.dump(sheets, f)

# TODO maybe make df some merge of the three sheets?
df = sheets['recordings']

# TODO TODO make sure this does not add counts for pairs w/ either date or
# fly_num NaN (or zero them out manually afterwards)
df['recording_num'] = df.groupby(['date','fly_num']).cumcount() + 1

# TODO check values are reasonable. inspect those that were previously NaN.
df.thorimage_dir.fillna(df.recording_num.apply(lambda x: '_{:03d}'.format(x)),
    inplace=True)
df.thorsync_dir.fillna(
    df.recording_num.apply(lambda x: 'SyncData{:03d}'.format(x)), inplace=True)


# TODO add bool col to recordings to indicate whether stimulus order in python
# file was adhered to?
# TODO or just only use it for natural_odors project for now?
# (probably just want to ignore order for project="n/a (prep checking)" columns
# anyway)

# TODO settle on NAS / group dropbox data layout
# TODO iterate over that layout and find folders with analysis output
# TODO + summarize those that still need analysis run on them (and run it?)
# TODO print stuff w/ used_for_analysis checked w/o either data in database or
# analysis output on disk (or just latter)

# TODO change to dropbox / nas
analysis_output_root = '/media/tom/smhcr/2019 data analysis'
rel_to_cnmf_mat = 'cnmf'
stimfile_root = '/media/tom/smhcr/stimulus_data_files'

natural_odors_concentrations = pd.read_csv('natural_odor_panel_vial_concs.csv')
natural_odors_concentrations.set_index('name', inplace=True)

# TODO TODO loop over more stuff than just natural_odors / used_for_analysis
# to load all PID stuff in (will need pin info for prep checking, etc, exps)

# TODO complain if there are flies w/ used_for_analysis not checked w/o
# rejection reason

# TODO TODO err if root dirs aren't even there...
# TODO diff between ** and */ ?
# TODO os.path.join + os invariant way of looping over dirs
for analysis_dir in glob.glob(analysis_output_root + '/*/'):
    print(analysis_dir)

    date = datetime.strptime(split(analysis_dir[:-1])[-1].split('_')[0],
                                      '%Y-%m-%d')
    print(date)

    # TODO TODO complain if stuff marked as used for analysis is not found here
    for mat in glob.glob(join(analysis_dir, rel_to_cnmf_mat, '*_cnmf.mat')):
        prefix = split(mat)[-1].split('_')[:-1]

        print(mat)
        print(prefix)

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
