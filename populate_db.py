#!/usr/bin/env python3

"""
Traverses analysis output and loads traces and odor information into database.
"""

import os
from os.path import join, split
import sys
import glob
from datetime import datetime
import pickle
import warnings
import time
import xml.etree.ElementTree as etree
import pprint

from sqlalchemy import create_engine, MetaData, Table
# TODO or just use sqlalchemy.types?
from sqlalchemy.sql import sqltypes
from sqlalchemy.dialects import postgresql
import h5py
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import matlab.engine

import util as u


################################################################################
verbose = False

use_cached_gsheet = False
show_inferred_paths = True

only_do_anything_for_analysis = True

convert_h5 = True
calc_timing_info = True
# If timing info ("ti") already exists in .mat, should we recalculate it?
update_timing_info = True
motion_correct = True
# TODO fix. this seems to not be working correctly.
only_motion_correct_for_analysis = True

load_traces = False
# TODO make sure that incomplete entries are not preventing full
# analysis from being inserted, despite setting of this flag
# TODO maybe just use some kind of transaction to guarantee no incomplete
# entries?
#overwrite_older_analysis = True
overwrite_older_analysis = False

################################################################################


if only_do_anything_for_analysis:
    only_motion_correct_for_analysis = True

# TODO TODO implement some kind of queue (or just lock files on NAS?) so
# multiple instantiations can work in parallel

analysis_started_at = time.time()

# TODO just factor all calls to matlab fns into util and don't even expose
# engine?
# TODO future have a bug generally, or only if stopped w/ ctrl-d from ipdb like
# i had?
#future = matlab.engine.start_matlab(async=True)
#evil = None
evil = u.matlab_engine()

# To get Git version information to have a record of what analysis was
# performed.
matlab_repo_name = 'matlab_kc_plane'
userpath = evil.userpath()
matlab_code_path = join(userpath, matlab_repo_name)

matlab_caiman_folder = 'CaImAn-MATLAB_remy'
mc_on_path = [x for x in evil.path().split(':')
    if x.endswith(matlab_caiman_folder)]

if len(mc_on_path) > 1:
    raise ValueError('more than one CaImAn version on MATLAB path. add ' +
        'versions you do not wish to use to exclude_from_matlab_path, in util')

elif len(mc_on_path) == 0:
    raise ValueError('MATLAB CaImAn package not found. Put on path or update ' +
        'matlab_caiman_folder')

matlab_caiman_path = mc_on_path[0]
matlab_caiman_version = u.version_info(matlab_caiman_path)

this_repo_file = os.path.realpath(__file__)
# TODO just use util fn that gets this internally
this_repo_path = split(this_repo_file)[0]

#driver_version_info ?
matlab_code_version = u.version_info(matlab_code_path)

# TODO fn to convert raw output to tifs (that are compat w/ current ij tifs)

df = u.mb_team_gsheet(
    use_cache=use_cached_gsheet,
    show_inferred_paths=show_inferred_paths
)

# TODO TODO warn if any raw data is not present on NAS / report which
# (could indicate problems w/ path inference)

# TODO add bool col to recordings to indicate whether stimulus order in python
# file was adhered to?
# TODO or just only use it for natural_odors project for now?
# (probably just want to ignore order for project="n/a (prep checking)" columns
# anyway)

# TODO + summarize those that still need analysis run on them (and run it?)
# TODO print stuff w/ attempt_analysis checked w/o either data in database or
# analysis output on disk (or just latter)

# TODO move these paths to config file...
raw_data_root = '/mnt/nas/mb_team/raw_data'
analysis_output_root = '/mnt/nas/mb_team/analysis_output'

rel_to_cnmf_mat = 'cnmf'

stimfile_root = '/mnt/nas/mb_team/stimulus_data_files' 

natural_odors_concentrations = pd.read_csv('natural_odor_panel_vial_concs.csv')
natural_odors_concentrations.set_index('name', inplace=True)

# TODO TODO loop over more stuff than just natural_odors / attempt_analysis
# to load all PID stuff in (will need pin info for prep checking, etc, exps)

# TODO complain if there are flies w/ attempt_analysis not checked w/o
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
        # TODO maybe warn if not in whitelist, like 'unsorted'
        continue

    try:
        date = datetime.strptime(date_dir, '%Y-%m-%d')
    except ValueError:
        continue

    print('Date:', date_dir)
    print('Fly:', fly_num)

    used = df.loc[df.attempt_analysis &
        (df.date == date) & (df.fly_num == fly_num)]

    if len(used) > 0:
        print('Used ThorImage dirs:')
        for d in used.thorimage_dir:
            print(d)
    else:
        print('No used ThorImage dirs.')

        if only_do_anything_for_analysis:
            continue

    # TODO maybe do this in analysis actually? (to not just make a bunch of
    # empty dirs...)
    # TODO TODO or just clean up appropriately
    analysis_fly_dir = join(analysis_output_root, date_dir, fly_dir)
    if not os.path.isdir(analysis_fly_dir):
        # Will also make any necessary parent (date) directories.
        os.makedirs(analysis_fly_dir)

    # TODO maybe use regexp to check syncdata / util fn to check for name +
    # stuff in it?
    if convert_h5:
        ####print('Converting ThorSync HDF5 files to .mat...')
        for syncdir in glob.glob(join(full_fly_dir, 'SyncData*')):
            print('before calling matlab h5->mat conversion...')
            print('syncdir={}'.format(syncdir))

            # Will immediately return if output already exists.
            evil.thorsync_h5_to_mat(syncdir, full_fly_dir, nargout=0)
            # TODO (check whether she is losing information... file size really
            # shouldn't be that different. both hdf5...)

            print('after calling matlab h5-> mat conversion')

            # TODO do i want flags to disable *each* step separately for
            # unanalyzed stuff? just one flag? always ignore that stuff?

            # TODO remy could also convert xml here if she wanted
        ####print('Done converting ThorSync HDF5 to .mat\n')

    matfile_dir = join(analysis_fly_dir, 'cnmf')

    if calc_timing_info:
        for _, row in used[['thorimage_dir','thorsync_dir']].iterrows():
            thorimage_dir = join(full_fly_dir, row['thorimage_dir'])
            if not os.path.isdir(thorimage_dir):
                warnings.warn('thorimage_dir {} did not exist for recording ' +
                    'marked as attempt_analysis.')
                continue

            # If not always running h5->mat conversion first, will need to check
            # for the mat, rather than just thorsync_dir.
            thorsync_dir = join(full_fly_dir, row['thorsync_dir'])
            if not os.path.isdir(thorsync_dir):
                warnings.warn('thorsync_dir {} did not exist for recording ' +
                    'marked as attempt_analysis.')
                continue

            # TODO maybe check for existance of SyncData<nnn> first, to have
            # option to be less verbose for stuff that doesn't exist here

            print('\nThorImage and ThorSync dirs for call to get_stiminfo:')
            print(thorimage_dir)
            print(thorsync_dir)
            print('')

            print(('getting stimulus timing information for {}, {}, {}...'
                ).format(date_dir, fly_num, row['thorimage_dir']), end='',
                flush=True)

            matfile = join(matfile_dir, '{}_cnmf.mat'.format(
                row['thorimage_dir']))

            # TODO TODO check exit code -> save all applicable version info
            # into the same matfile, calling the matlab interface from here
            try:
                # TODO maybe determine whether to update_ti based on reading
                # version info (in update_timing_info == False case)?
                update_ti = update_timing_info

                # throwing everything into _<>_cnmf.mat, as we are, would need
                # to inspect it to check whether we already have the stiminfo...
                updated_ti = evil.get_stiminfo(thorimage_dir,
                    row['thorsync_dir'], analysis_fly_dir, date_dir, fly_num,
                    update_ti, nargout=1)

                if updated_ti:
                    evil.workspace['ti_code_version'] = matlab_code_version 
                    evil.save(matfile, 'ti_code_version', '-append', nargout=0)

                    # Testing version info is stored correctly.
                    evil.clear(nargout=0)
                    load_output = evil.load(matfile, 'ti_code_version',
                        nargout=1)

                    rt_matlab_code_version = load_output['ti_code_version']
                    assert matlab_code_version == rt_matlab_code_version
                    evil.clear(nargout=0)

            except matlab.engine.MatlabExecutionError:
                continue

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
        # TODO maybe also look w/o underscore, if that's remy's convention
        for input_tif_path in glob.glob(
            join(full_fly_dir, 'tif_stacks', '_*.tif')):

            thorimage_dir = split(input_tif_path)[-1][:-4]
            if only_motion_correct_for_analysis:
                recordings = used[used.thorimage_dir == thorimage_dir]
                if len(recordings) == 0:
                    continue

            matfile = join(matfile_dir, '{}_cnmf.mat'.format(thorimage_dir))

            print('\nRunning normcorre_tiff on', input_tif_path)
            # TODO only register one way by default? nonrigid? args to
            # configure?
            rig_updated, nr_updated = evil.normcorre_tiff(input_tif_path,
                analysis_fly_dir, nargout=2)

            mocorr_code_versions = [matlab_code_version, matlab_caiman_version]

            if rig_updated:
                evil.workspace['rig_code_versions'] = mocorr_code_versions
                evil.save(matfile, 'rig_code_versions', '-append', nargout=0)

                # Testing version info is stored correctly.
                evil.clear(nargout=0)
                load_output = evil.load(matfile, 'rig_code_versions',
                    nargout=1)

                rt_mocorr_code_versions = load_output['rig_code_versions']
                assert mocorr_code_versions == rt_mocorr_code_versions
                evil.clear(nargout=0)

            if nr_updated:
                evil.workspace['nr_code_versions'] = mocorr_code_versions
                evil.save(matfile, 'nr_code_versions', '-append', nargout=0)

                # Testing version info is stored correctly.
                evil.clear(nargout=0)
                load_output = evil.load(matfile, 'nr_code_versions',
                    nargout=1)

                rt_mocorr_code_versions = load_output['nr_code_versions']
                assert mocorr_code_versions == rt_mocorr_code_versions
                evil.clear(nargout=0)

    # TODO and if remy wants, copy thorimage xmls

    print('')

    # TODO maybe delete empty folders under analysis? (do in atexit handler)

if not load_traces:
    sys.exit()

# TODO delete all this stuff after saving full version info as appropriate
current_hash = u.git_hash(this_repo_file)
matlab_hash = u.git_hash(matlab_code_path)

# TODO just store all data separately?
# TODO TODO maybe just use matlab code repo + description in analysis
# description that gets checked? (because that's what actually generates cnmf,
# which is used for responses)
analysis_description = '{}@{}\n{}@{}'.format(this_repo_path, current_hash,
    matlab_code_path, matlab_hash)

analyzed_at = datetime.fromtimestamp(analysis_started_at)

# TODO TODO clean this of runs that don't have data in the database...
# (+ reindex serial id?)
analysis_runs = pd.DataFrame({
    'analysis_description': [analysis_description],
    'analyzed_at': [analyzed_at]
})
# TODO don't insert into this if dependent stuff won't be written? same for some
# of the other metadata tables?
u.to_sql_with_duplicates(analysis_runs, 'analysis_runs')

# Need to do this as long as the part of the key indicating the
# analysis, in the recordings table, is generated by the database.
db_analysis_runs = pd.read_sql('analysis_runs', u.conn).set_index(
    'analysis_description')
analysis_run = \
    db_analysis_runs.loc[analysis_description, 'analysis_run']


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


    mat_files = glob.glob(join(analysis_dir, rel_to_cnmf_mat, '*_cnmf.mat'))
    # TODO both in this case and w/ stuff above, maybe don't print anything in
    # case where no data is found
    if len(mat_files) == 0:
        if verbose:
            print(analysis_dir)
            print('no CNMF output MAT files')
        continue

    # TODO complain if stuff marked as used for analysis is not found here
    # TODO also implement only_do_anything_for_analysis here
    for mat in mat_files:
        print(mat)

        prefix = split(mat)[-1].split('_')[:-1]

        thorimage_id = '_' + prefix[1]
        # TODO TODO delete
        '''
        if not (date_dir == '2019-01-18' and fly_num == 2 and
                thorimage_id == '_003'):
            print('skipping')
            continue
        print('not skipping this one')
        '''
        if not date_dir == '2019-02-27':
            print('skipping')
            continue
        #

        recordings = df.loc[(df.date == date) & (df.fly_num == fly_num) &
                            (df.thorimage_dir == thorimage_id)]
        recording = recordings.iloc[0]

        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.')
            continue


        raw_fly_dir = join(raw_data_root, date_dir, fly_dir)
        thorsync_dir = join(raw_fly_dir, recording['thorsync_dir'])
        thorimage_dir = join(raw_fly_dir, recording['thorimage_dir'])
        stimulus_data_path = join(stimfile_root,
                                  recording['stimulus_data_file'])

        # TODO for recordings.started_at, load time from one of the thorlabs
        # files
        # TODO check that ThorImageExperiment/Date/uTime parses to date field,
        # w/ appropriate timezone settings (or insert raw and check database has
        # something like date)?
        # TODO TODO factor this into util, right? gui uses it?
        thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')
        xml_root = etree.parse(thorimage_xml_path).getroot()
        started_at = \
            datetime.fromtimestamp(float(xml_root.find('Date').attrib['uTime']))


        # TODO check this whole section after analysis refactoring...
        entered = pd.read_sql_query('SELECT DISTINCT prep_date, ' +
            'fly_num, recording_from, analysis FROM presentations', u.conn)
        # TODO TODO check that the right number of rows are in there, otherwise
        # drop and re-insert (optionally? since it might take a bit of time to
        # load CNMF output to check / for database to check)

        # TODO more elegant way to check for row w/ certain values?
        curr_entered = (
            (entered.prep_date == date) &
            (entered.fly_num == fly_num) &
            (entered.recording_from == started_at)
        )

        if overwrite_older_analysis:
            curr_entered = curr_entered & (entered.analysis == analysis_run)

        curr_entered = curr_entered.any()

        # TODO maybe replace analysis w/ same description but earlier version?
        # (where description is just combination of repo names, not w/ version
        # as now
        if curr_entered:
            print('{}, {}, {} already entered with current analysis'.format(
                date, fly_num, thorimage_id))
            continue

        recordings = pd.DataFrame({
            'started_at': [started_at],
            'thorsync_path': [thorsync_dir],
            'thorimage_path': [thorimage_dir],
            'stimulus_data_path': [stimulus_data_path]
        })
        u.to_sql_with_duplicates(recordings, 'recordings')


        stimfile = recording['stimulus_data_file']
        stimfile_path = join(stimfile_root, stimfile)
        # TODO also err if not readable
        if not os.path.exists(stimfile_path):
            raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
                stimfile_root))

        with open(stimfile_path, 'rb') as f:
            data = pickle.load(f)

        n_repeats = int(data['n_repeats'])

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3

        presentations_per_block = \
            n_repeats * presentations_per_repeat


        if pd.isnull(recording['first_block']):
            first_block = 0
        else:
            first_block = int(recording['first_block']) - 1

        if pd.isnull(recording['last_block']):
            n_full_panel_blocks = \
                int(len(data['odor_pair_list']) / presentations_per_block)

            last_block = n_full_panel_blocks - 1

        else:
            last_block = int(recording['last_block']) - 1

        first_presentation = first_block * presentations_per_block
        last_presentation = (last_block + 1) * presentations_per_block

        # TODO will need to augment w/ concentration info somehow...
        # maybe handle in a way specific to natural_odors project?

        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [0 if x == 'paraffin' else
                natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })

        u.to_sql_with_duplicates(odors, 'odors')

        # TODO make unique id before insertion? some way that wouldn't require
        # the IDs, but would create similar tables?

        db_odors = pd.read_sql('odors', u.conn)
        # TODO TODO in general, the name alone won't be unique, so use another
        # strategy
        db_odors.set_index('name', inplace=True)

        # TODO test slicing
        # TODO make sure if there are extra trials in matlab, these get assigned
        # to first
        # + if there are less in matlab, should error
        odor_pair_list = \
            data['odor_pair_list'][first_presentation:last_presentation]

        assert (len(odor_pair_list) %
            (presentations_per_repeat * n_repeats) == 0)

        # TODO invert to check
        # TODO is this sql table worth anything if both keys actually need to be
        # referenced later anyway?

        # TODO only add as many as there were blocks from thorsync timing info?
        odor1_ids = [db_odors.at[o1,'odor'] for o1, _ in odor_pair_list]
        odor2_ids = [db_odors.at[o2,'odor'] for _, o2 in odor_pair_list]

        # TODO TODO make unique first. only need order for filling in the values
        # in responses.
        mixtures = pd.DataFrame({
            'odor1': odor1_ids,
            'odor2': odor2_ids
        })

        u.to_sql_with_duplicates(mixtures, 'mixtures')
        # TODO merge w/ odors to check

        # TODO maybe use Remy's thorsync timing info to get num pulses for prep
        # checking trials and then assume it's ethyl acetate (for PID purposes,
        # at least)?
        # TODO would need to make sure all types are compat to load this way
        #print('loading MAT file in MATLAB...', end='')

        evil.evalc("clear; data = load('{}', 'S');".format(mat))

        # sDFF - filtered traces (filtered how?)
        # F0 - background fluorescence (dims?)
        #print(' done')
        try:
            S = evil.eval('data.S')
        except matlab.engine.MatlabExecutionError:
            print('CNMF still needs to be run on this data')
            continue

        # TODO rename to indicate this is not just a filtered version of C?
        # (and how exactly is it different again?)
        # Loading w/ h5py lead to transposed indices wrt loading w/ MATLAB
        # engine.
        #filtered_df_over_f = np.array(S['sDFF']).T
        # TODO possible to just load a subfield of the CNM object / S w/ load
        # semantics?
        df_over_f = np.array(S['DFF']).T

        # TODO quantitatively compare sDFF / DFF. (s=smoothed)
        # maybe just upload non-smoothed and let people smooth downstream if
        # they want?

        # TODO could check for sCNM, to load that in cases where we can't make a
        # cnmf object? (but we should able able to here...)
        try:
            evil.evalc("clear; data = load('{}', 'CNM', 'ti');".format(mat))
        except matlab.engine.MatlabExecutionError as e:
            # TODO inspect error somehow to see if it's a memory error?
            # -> continue if so
            # TODO print to stderr
            print(e)
            continue

        raw_f = np.array(evil.eval('data.CNM.C')).T

        ti = evil.eval('data.ti')
        # TODO dtype appropriate?
        frame_times = np.array(ti['frame_times']).flatten()

        # Frame indices for CNMF output.
        # Of length equal to number of blocks. Each element is the frame
        # index (from 1) in CNMF output that starts the block, where
        # block is defined as a period of continuous acquisition.
        block_first_frames = np.array(ti['trial_start'], dtype=np.uint32
            ).flatten() - 1

        # stim_on is a number as above, but for the frame of the odor
        # onset.
        # TODO how does rounding work here? closest frame? first after?
        # TODO TODO did Remy change these variables? (i mean, it worked w/ some
        # videos?)
        odor_onset_frames = np.array(ti['stim_on'], dtype=np.uint32
            ).flatten() - 1
        odor_offset_frames = np.array(ti['stim_off'], dtype=np.uint32
            ).flatten() - 1
        # TODO TODO TODO if these are 1d, should be sorted... is Remy doing
        # something else weird?
        # (address after candidacy)
        

        # TODO how to get odor pid

        # A has footprints
        # from caiman docs:
        # "A: ... (d x K)"
        # "K: ... # of neurons to extract"
        # so what is d? (looks like 256 x 256, the # of pixels)

        # TODO TODO maybe get some representation of the sparse matrix that i
        # can use to create one from the scipy constructors, to minimize amount
        # of data sent from matlab
        print('loading footprints...', end='')
        footprints = np.array(evil.eval('full(data.CNM.A)'))
        # Assuming equal number along both dimensions.
        pixels_per_side = int(np.sqrt(footprints.shape[0]))
        n_footprints = footprints.shape[1]

        # TODO C order? (check against image to make sure things don't seem
        # transposed...)
        footprints = np.reshape(footprints,
            (pixels_per_side, pixels_per_side, n_footprints))
        print(' done')

        # Just to try to free up some memory.
        evil.evalc('clear;')

        # TODO TODO TODO make sure these cell IDs match up with the ones from
        # below!!!

        footprint_dfs = []
        for cell_num in range(n_footprints):
            sparse = coo_matrix(footprints[:,:,cell_num])
            footprint_dfs.append(pd.DataFrame({
                'recording_from': [started_at],
                'cell': [cell_num],
                # Can be converted from lists of Python types, but apparently
                # not from numpy arrays or lists of numpy scalar types.
                # TODO check this doesn't transpose things
                # TODO TODO just move appropriate casting to my to_sql function,
                # and allow having numpy arrays (get type info from combination
                # of that and the database, like in other cases)
                'x_coords': [[int(x) for x in sparse.col.astype('int16')]],
                'y_coords': [[int(x) for x in sparse.row.astype('int16')]],
                'weights': [[float(x) for x in sparse.data.astype('float32')]],
                'analysis': [analysis_run]
            }))

        footprint_df = pd.concat(footprint_dfs, ignore_index=True)
        # TODO filter out footprints less than a certain # of pixels in cnmf?
        # (is 3 pixels really reasonable?)
        u.to_sql_with_duplicates(footprint_df, 'cells', verbose=True)

        # TODO and what would be a good db representation of footprint?
        # TODO TODO generalize to_sql casting to array types too?
        

        # TODO store image w/ footprint overlayed?
        # TODO TODO maybe store an average frame of registered TIF, and then
        # indexes around that per footprint? (explicitly try to avoid responses
        # when doing so, for easier interpretation as a background?)

        # dims=dimensions of image (256x256)
        # T is # of timestamps

        # TODO why 474 x 4 + 548 in one case? i thought frame numbers were
        # supposed to be more similar... (w/ np.diff(odor_onset_frames))
        first_onset_frame_offset = odor_onset_frames[0] - block_first_frames[0]

        n_frames, n_cells = df_over_f.shape
        assert n_cells == n_footprints

        start_frames = np.append(0,
            odor_onset_frames[1:] - first_onset_frame_offset)
        stop_frames = np.append(
            odor_onset_frames[1:] - first_onset_frame_offset - 1, n_frames)
        lens = [stop - start for start, stop in zip(start_frames, stop_frames)]

        # TODO delete version w/ int cast after checking they give same answers
        assert int(frame_times.shape[0]) == int(n_frames)
        assert frame_times.shape[0] == n_frames

        print(start_frames)
        print(stop_frames)
        # TODO find where the discrepancies are!
        print(sum(lens))
        print(n_frames)

        # TODO assert here that all frames add up / approx

        # TODO TODO either warn or err if len(start_frames) is !=
        # len(odor_pair_list)

        odor_id_pairs = [(o1,o2) for o1,o2 in zip(odor1_ids, odor2_ids)]

        comparison_num = -1

        for i in range(len(start_frames)):
            if i % (presentations_per_repeat * n_repeats) == 0:
                comparison_num += 1
                repeat_nums = {id_pair: 0 for id_pair in odor_id_pairs}

            # TODO TODO also save to csv/flat binary/hdf5 per (date, fly,
            # thorimage)
            print('Processing presentation {}'.format(i))

            start_frame = start_frames[i]
            stop_frame = stop_frames[i]
            # TODO off by one?? check
            # TODO check against frames calculated directly from odor offset...
            # may not be const # frames between these "starts" and odor onset?
            onset_frame = start_frame + first_onset_frame_offset

            # TODO check again that these are always equal and delete
            # "direct_onset_frame" bit
            print('onset_frame:', onset_frame)
            direct_onset_frame = odor_onset_frames[i]
            print('direct_onset_frame:', direct_onset_frame)

            # TODO TODO why was i not using direct_onset_frame for this before?
            onset_time = frame_times[direct_onset_frame]
            assert start_frame < stop_frame
            # TODO check these don't jump around b/c discontinuities
            presentation_frametimes = \
                frame_times[start_frame:stop_frame] - onset_time
            # TODO delete try/except after fixing
            try:
                assert len(presentation_frametimes) > 1
            except AssertionError:
                print(frame_times)
                print(start_frame)
                print(stop_frame)
                import ipdb; ipdb.set_trace()

            odor_pair = odor_id_pairs[i]
            odor1, odor2 = odor_pair
            repeat_num = repeat_nums[odor_pair]
            repeat_nums[odor_pair] = repeat_num + 1

            offset_frame = odor_offset_frames[i]
            print('offset_frame:', offset_frame)
            assert offset_frame > direct_onset_frame
            # TODO share more of this w/ dataframe creation below, unless that
            # table is changed to just reference presentation table
            presentation = pd.DataFrame({
                'prep_date': [date],
                'fly_num': fly_num,
                'recording_from': started_at,
                'comparison': comparison_num,
                'odor1': odor1,
                'odor2': odor2,
                'repeat_num': repeat_num,
                'odor_onset_frame': direct_onset_frame,
                'odor_offset_frame': offset_frame,
                'from_onset': [[float(x) for x in presentation_frametimes]],
                'analysis': analysis_run
            })
            u.to_sql_with_duplicates(presentation, 'presentations')


            # maybe share w/ code that checks distinct to decide whether to
            # load / analyze?
            key_cols = [
                'prep_date',
                'fly_num',
                'recording_from',
                'comparison',
                'odor1',
                'odor2',
                'repeat_num'
            ]
            db_presentations = pd.read_sql('presentations', u.conn,
                columns=(key_cols + ['presentation_id']))

            presentation_ids = (db_presentations[key_cols] ==
                                presentation[key_cols].iloc[0]).all(axis=1)
            assert presentation_ids.sum() == 1
            presentation_id = db_presentations.loc[presentation_ids,
                'presentation_id'].iat[0]

            # TODO get remy to save it w/ less than 64 bits of precision?
            presentation_dff = df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = raw_f[start_frame:stop_frame, :]

            # Assumes that cells are indexed same here as in footprints.
            cell_dfs = []
            for cell_num in range(n_cells):

                cell_dff = presentation_dff[:, cell_num].astype('float32')
                cell_raw_f = presentation_raw_f[:, cell_num].astype('float32')

                cell_dfs.append(pd.DataFrame({
                    'presentation_id': [presentation_id],
                    'recording_from': [started_at],
                    'cell': [cell_num],
                    'df_over_f': [[float(x) for x in cell_dff]],
                    'raw_f': [[float(x) for x in cell_raw_f]]
                }))
            response_df = pd.concat(cell_dfs, ignore_index=True)

            u.to_sql_with_duplicates(response_df, 'responses')

            # TODO put behind flag
            db_presentations = pd.read_sql_query('SELECT DISTINCT prep_date, ' +
                'fly_num, recording_from, comparison FROM presentations',
                u.conn)

            print(db_presentations)
            print(len(db_presentations))
            #


            print('Done processing presentation {}'.format(i))

        # TODO check that all frames go somewhere and that frames aren't
        # given to two presentations. check they stay w/in block boundaries.
        # (they don't right now. fix!)

# TODO print unused stimfiles / option to delete them

