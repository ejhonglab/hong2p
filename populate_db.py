#!/usr/bin/env python3

"""
Traverses analysis output and loads traces and odor information into database.
"""

import os
from os.path import join, split, exists
import sys
import glob
from datetime import datetime
import pickle
import warnings
import time
import xml.etree.ElementTree as etree
import copy
import pprint

import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import matlab.engine
import tifffile
import matplotlib.pyplot as plt

# TODO maybe just move these fns into a module hong2p, rather than hong2p util?
# or maybe __init__ stuff to get them importable under hong2p?
import hong2p.util as u


################################################################################
verbose = True

# False or positive integers.
# If not False, analysis will only run on the most recent n dates in the
# mb_team_flies Google sheet metadata.
only_last_n_days = 3

use_cached_gsheet = False
show_inferred_paths = True
allow_gsheet_to_restrict_blocks = True

fail_on_missing_dir_to_attempt = True
only_do_anything_for_analysis = True

convert_h5 = True
calc_timing_info = True
# If timing info ("ti") already exists in .mat, should we recalculate it?
# TODO if this is False, still check that ti_code_version is there?
update_timing_info = False
convert_raw_to_tiffs = True
motion_correct = True
only_motion_correct_for_analysis = True
fit_rois = True

process_time_averages = False
upload_matlab_cnmf_output = False
ACTUALLY_UPLOAD = True

# TODO make sure that incomplete entries are not preventing full
# analysis from being inserted, despite setting of this flag
# TODO maybe just use some kind of (across to_sql calls) transaction to
# guarantee no incomplete entries?
#overwrite_older_analysis = True
overwrite_older_analysis = False

# TODO delete
# this one had a MATLAB function cannot be evaluated error
# (but maybe context was important?)
# (this error didn't seem to repeat when calling just on this...)
#test_recording = ('2019-04-11', 2, '_002')
test_recording = None
#
################################################################################

if only_do_anything_for_analysis:
    only_motion_correct_for_analysis = True

conn = u.get_db_conn()

# TODO TODO implement some kind of queue (or just lock files on NAS?) so
# multiple instantiations can work in parallel

analyzed_at = datetime.fromtimestamp(time.time())

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
matlab_caiman_version = u.version_info(matlab_caiman_path,
                                       used_for='motion correction')

this_repo_file = os.path.realpath(__file__)
# TODO just use util fn that gets this internally
this_repo_path = split(this_repo_file)[0]

#driver_version_info ?
matlab_code_version = u.version_info(matlab_code_path)
curr_ti_code_version = copy.deepcopy(matlab_code_version)
curr_ti_code_version['used_for'] = 'calculating timing information'
matlab_code_version['used_for'] = 'driving motion correction'

df = u.mb_team_gsheet(
    use_cache=use_cached_gsheet,
    show_inferred_paths=show_inferred_paths
)
if only_last_n_days:
    dates_to_consider = sorted(df.date.unique())[-only_last_n_days:]
    date_dirs_to_consider = {pd.Timestamp(d).strftime(u.date_fmt_str)
        for d in dates_to_consider}
    print('B/c only_last_n_days setting, only considering date directories:',
        date_dirs_to_consider)

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

# TODO move these paths to config file / envvar (+ defaults in util?)
raw_data_root = u.raw_data_root()
analysis_output_root = u.analysis_output_root()

rel_to_cnmf_mat = 'cnmf'

stimfile_root = u.stimfile_root()

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

    if only_last_n_days and date_dir not in date_dirs_to_consider:
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

    # TODO maybe try to address this issue (get_stiminfo), probably don't
    if date_dir == '2019-01-17':
        continue

    # TODO delete
    if test_recording is not None:
        if date_dir != test_recording[0] or fly_num != test_recording[1]:
            continue
    #

    print('Date:', date_dir)
    print('Fly:', fly_num)

    fly_df = df.loc[(df.date == date) & (df.fly_num == fly_num)]
    used = fly_df.loc[fly_df.attempt_analysis]

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
        if only_do_anything_for_analysis:
            get_ti_df = used
        else:
            get_ti_df = fly_df

        for _, row in get_ti_df[['thorimage_dir','thorsync_dir']].iterrows():
            # TODO delete. for debugging.
            if test_recording is not None:
                if row['thorimage_dir'] != test_recording[2]:
                    continue
            #
            if u.is_subrecording(row['thorimage_dir']):
                continue

            thorimage_dir = join(full_fly_dir, row['thorimage_dir'])
            if not os.path.isdir(thorimage_dir):
                err_msg = ('thorimage_dir {} did not exist for recording ' +
                    'marked as attempt_analysis.').format(thorimage_dir)
                if fail_on_missing_dir_to_attempt:
                    raise IOError(err_msg)
                else:
                    warnings.warn(err_msg)
                    continue

            # If not always running h5->mat conversion first, will need to check
            # for the mat, rather than just thorsync_dir.
            thorsync_dir = join(full_fly_dir, row['thorsync_dir'])
            if not os.path.isdir(thorsync_dir):
                err_msg = ('thorsync_dir {} did not exist for recording ' +
                    'marked as attempt_analysis.').format(thorsync_dir)
                if fail_on_missing_dir_to_attempt:
                    raise IOError(err_msg)
                else:
                    warnings.warn(err_msg)
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

            # wasn't actually changing matlab err print color (cause stderr?)
            # even if i did get it to work, might also color warnings and
            # verbose prints, which i don't want
            #u.start_color('red')
            try:
                # TODO maybe determine whether to update_ti based on reading
                # version info (in update_timing_info == False case)?
                update_ti = update_timing_info

                # throwing everything into _<>_cnmf.mat, as we are, would need
                # to inspect it to check whether we already have the stiminfo...
                updated_ti = evil.get_stiminfo(thorimage_dir,
                    row['thorsync_dir'], analysis_fly_dir, update_ti, nargout=1)

                if exists(matfile) and updated_ti:
                    evil.workspace['ti_code_version'] = curr_ti_code_version 
                    evil.save(matfile, 'ti_code_version', '-append', nargout=0)

                    # Testing version info is stored correctly.
                    evil.clear(nargout=0)
                    load_output = evil.load(matfile, 'ti_code_version',
                        nargout=1)

                    rt_matlab_code_version = load_output['ti_code_version']
                    assert curr_ti_code_version == rt_matlab_code_version
                    evil.clear(nargout=0)

            except matlab.engine.MatlabExecutionError as err:
                u.print_color('red', err)
                print('')
                continue
            #finally:
            #    u.stop_color()

            print(' done.')

    tiff_dir = join(full_fly_dir, 'tif_stacks')
    if convert_raw_to_tiffs:
        # TODO only do this for stuff we are going to actually motion correct?
        # or at least respect only_do_anything_for_analysis...
        if not exists(tiff_dir):
            os.mkdir(tiff_dir)

        for thorimage_dir in glob.glob(join(full_fly_dir, '*/')):
            if not u.is_thorimage_dir(thorimage_dir):
                continue

            thorimage_id = split(os.path.normpath(thorimage_dir))[-1]
            tiff_filename = join(tiff_dir, thorimage_id + '.tif')
            if exists(tiff_filename):
                continue

            from_raw = u.read_movie(thorimage_dir)
            print('Writing TIFF to {}... '.format(tiff_filename), end='',
                flush=True)
            u.write_tiff(tiff_filename, from_raw)
            print('done.')

        # TODO at least delete dir if empty (only if we made it?)
        try:
            os.rmdir(tiff_dir)
        # This catches case where directory is not empty, without deleting.
        except OSError:
            pass

    # maybe avoid searching for thorimage dirs at all if there are no used 
    # rows for this (date,fly) combo, and only_motion_correct_for_analysis

    # TODO use multiple matlab instances to run normcore on different
    # directories in parallel?
    # TODO exclude stuff that indicates it's either already avg or motion
    # corrected? (or just always keep them separately?)
    # TODO maybe also look w/o underscore, if that's remy's convention
    for input_tif_path in glob.glob(join(tiff_dir, '*.tif')):
        if motion_correct:
            thorimage_dir = split(input_tif_path)[-1][:-4]
            if only_motion_correct_for_analysis:
                recordings = used[used.thorimage_dir == thorimage_dir]
                if len(recordings) == 0:
                    print(('{}: skipping motion correction b/c ' +
                        'only_motion_correct_for_analysis').format(
                        thorimage_dir))
                    continue

            matfile = join(matfile_dir, '{}_cnmf.mat'.format(thorimage_dir))

            print('\nRunning normcorre_tiff on', input_tif_path)
            # TODO only register one way by default? nonrigid? args to
            # configure?
            try:
                rig_updated, nr_updated = evil.normcorre_tiff(
                    input_tif_path, analysis_fly_dir, nargout=2)

            except matlab.engine.MatlabExecutionError as e:
                continue

            mocorr_code_versions = [matlab_code_version, matlab_caiman_version]

            # TODO any reason i'm not just/also directly uploading these...?
            if rig_updated:
                evil.workspace['rig_code_versions'] = mocorr_code_versions
                # TODO only append if not exists
                # (timing info calculation could fail but we still want to
                # mocorr)
                if exists(matfile):
                    evil.save(matfile, 'rig_code_versions', '-append',
                        nargout=0)
                else:
                    evil.save(matfile, 'rig_code_versions', nargout=0)

                # Testing version info is stored correctly.
                evil.clear(nargout=0)
                load_output = evil.load(matfile, 'rig_code_versions',
                    nargout=1)

                rt_mocorr_code_versions = load_output['rig_code_versions']
                assert mocorr_code_versions == rt_mocorr_code_versions
                evil.clear(nargout=0)

            if nr_updated:
                evil.workspace['nr_code_versions'] = mocorr_code_versions
                if exists(matfile):
                    evil.save(matfile, 'nr_code_versions', '-append', nargout=0)
                else:
                    evil.save(matfile, 'nr_code_versions', nargout=0)

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


if fit_rois:
    print('Fitting ROIs...')
    template_data = u.load_template_data()
    if template_data is None:
        warnings.warn('template data not found, so can not fit_rois')
    else:
        # TODO make generator fns or something in util that yield
        # raw / analysis dirs / tifs / whatever
        for analysis_dir in glob.glob(analysis_output_root + '/*/*/'):
            analysis_dir = os.path.normpath(analysis_dir)

            prefix, fly_dir = split(analysis_dir)
            _, date_dir = split(prefix)

            if only_last_n_days and date_dir not in date_dirs_to_consider:
                continue

            try:
                fly_num = int(fly_dir)
            except ValueError:
                continue

            try:
                date = datetime.strptime(date_dir, '%Y-%m-%d')
            except ValueError:
                continue

            thorimage_ids = [split(td)[1] for td in
                u.thorimage_subdirs(u.raw_fly_dir(date, fly_num))
            ]
            for thorimage_id in thorimage_ids:
                try:
                    tif = u.motion_corrected_tiff_filename(date, fly_num,
                        thorimage_id
                    )
                # TODO is this except gonna work? OSError count?
                except IOError as e:
                    print(thorimage_id, end=': ')
                    print(e)
                    continue

                # TODO TODO check if analysis is ticked in df (gsheet)

                try:
                    u.fit_circle_rois(tif, template_data, write_ijrois=True,
                        overwrite=True, min_neighbors=None, multiscale=True,
                        threshold=0.3, roi_diams_from_kmeans_k=2,
                        exclude_dark_regions=True
                    )
                except RuntimeError as e:
                    print(e)
                    plt.show()
                    continue

# TODO TODO why had i commented this? some reason it should not be this way?
if not (upload_matlab_cnmf_output or process_time_averages):
    sys.exit()
#

# TODO delete all this stuff after saving full version info as appropriate
'''
# TODO TODO clean this of runs that don't have data in the database...
# (+ reindex serial id?)
analysis_runs = pd.DataFrame({
    'analysis_description': [analysis_description],
    'analyzed_at': [analyzed_at]
})
# TODO don't insert into this if dependent stuff won't be written? same for some
# of the other metadata tables?
u.to_sql_with_duplicates(analysis_runs, 'analysis_runs')
'''

recording_outcomes = []
# TODO diff between ** and */ ?
# TODO os.path.join + os invariant way of looping over dirs
for analysis_dir in glob.glob(analysis_output_root+ '/*/*/'):
    analysis_dir = os.path.normpath(analysis_dir)

    prefix, fly_dir = split(analysis_dir)
    _, date_dir = split(prefix)

    if only_last_n_days and date_dir not in date_dirs_to_consider:
        continue

    try:
        fly_num = int(fly_dir)
    except ValueError:
        continue

    try:
        date = datetime.strptime(date_dir, '%Y-%m-%d')
    except ValueError:
        continue

    recording_outcome = {
        'date': date_dir,
        'fly_num': fly_num,
        'thorimage_dir': '*',
        'skipped': False,
        'no_movie': False,
        'no_mat': False,
        'unsupported_project': False
    }

    # It's clear the timing information for these experiments is incorrect,
    # though they had a slightly different trial structure.
    # TODO TODO fix get_stiminfo for this case, just in case it might affect
    # future recordings
    skip = False
    if date_dir == '2019-01-18' or date_dir == '2019-01-17':
        skip = True
    #
    if test_recording is not None:
        if date_dir != test_recording[0] or fly_num != test_recording[1]:
            skip = True
    #

    if skip:
        recording_outcome['skipped'] = True
        recording_outcomes.append(recording_outcome)
        continue

    print('')
    print('#' * 80)
    print('#' * 80)
    print('date_dir:', date_dir)
    print('fly_num:', fly_num)
    print('')

    # TODO TODO loop over thorimage dirs and then just find mat files if we are
    # going to be supporting a trace analysis only path here
    # (only would really matter for whole trace storage right... still need odor
    # info to get the rejection criteria from that...)
    # TODO make dataframe saying whether each entry in mb_team_gsheet df
    # has cnmf output / timing information / tif / motion correction?
    mat_files = glob.glob(join(analysis_dir, rel_to_cnmf_mat, '*_cnmf.mat'))
    # TODO both in this case and w/ stuff above, maybe don't print anything in
    # case where no data is found
    if len(mat_files) == 0:
        if verbose:
            print(analysis_dir)
            print('no CNMF output MAT files')
        recording_outcome['no_mat'] = True
        recording_outcomes.append(recording_outcome)
        continue

    # TODO complain if stuff marked as used for analysis is not found here
    # TODO also implement only_do_anything_for_analysis here
    for mat in mat_files:
        print(mat)
        prefix = split(mat)[-1].split('_')[:-1]
        thorimage_id = '_'.join(prefix)
        recording_outcome['thorimage_dir'] = thorimage_id

        skip = False
        # TODO maybe come up w/ a recording_outcome for these
        # TODO TODO TODO maybe proceed, but keep track s.t. it isn't done twice
        # (b/c i might mark parent as not attempt_analysis, so it wouldn't get
        # reached here...)
        if u.is_subrecording(thorimage_id):
            print('skipping sub-recording')
            skip = True

        # get_stiminfo also currently fails on this recording, with an assertion
        # error (4/28)
        # TODO also fix get_stiminfo in this case
        '''
        if date_dir == '2019-02-27' and fly_num == 4 and thorimage_id == '_003':
            skip = True
        '''

        # TODO delete. for debugging.
        if test_recording is not None:
            if thorimage_id != test_recording[2]:
                skip = True
        #

        if skip:
            recording_outcome['skipped'] = True
            recording_outcomes.append(recording_outcome)
            continue

        recordings = df.loc[(df.date == date) & (df.fly_num == fly_num) &
                            (df.thorimage_dir == thorimage_id)]
        recording = recordings.iloc[0]

        # TODO TODO does util function that gets everything drop non-natural
        # odors stuff anyway? might not want that now...
        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.')
            recording_outcome['unsupported_project'] = True
            recording_outcomes.append(recording_outcome)
            continue

        raw_fly_dir = join(raw_data_root, date_dir, fly_dir)
        thorsync_dir = join(raw_fly_dir, recording['thorsync_dir'])
        thorimage_dir = join(raw_fly_dir, recording['thorimage_dir'])
        stimulus_data_path = join(stimfile_root,
                                  recording['stimulus_data_file'])

        # TODO TODO factor this into util, right? gui uses it?
        thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')
        xml_root = etree.parse(thorimage_xml_path).getroot()
        started_at = \
            datetime.fromtimestamp(float(xml_root.find('Date').attrib['uTime']))

        if upload_matlab_cnmf_output:
            # TODO check this whole section after analysis refactoring...
            entered = pd.read_sql_query('SELECT DISTINCT prep_date, ' +
                'fly_num, recording_from, analysis FROM presentations', conn)
            # TODO TODO check that the right number of rows are in there,
            # otherwise drop and re-insert (optionally? since it might take a
            # bit of time to load CNMF output to check / for database to check)

            # TODO more elegant way to check for row w/ certain values?
            curr_entered = (
                (entered.prep_date == date) &
                (entered.fly_num == fly_num) &
                (entered.recording_from == started_at)
            )

            raise NotImplementedError
            if overwrite_older_analysis:
                curr_entered = curr_entered & (entered.analysis == analysis_run)

            curr_entered = curr_entered.any()

            # TODO maybe replace analysis w/ same description but earlier
            # version? (where description is just combination of repo names,
            # not w/ version as now)
            if curr_entered:
                print('{}, {}, {} already entered with current analysis'.format(
                    date, fly_num, thorimage_id))
                # TODO if still going to support this path, add a flag for
                # recording_outcomes here
                continue

        stimfile = recording['stimulus_data_file']
        stimfile_path = join(stimfile_root, stimfile)
        # TODO also err if not readable (validate after read)
        if not os.path.exists(stimfile_path):
            raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
                stimfile_root))

        with open(stimfile_path, 'rb') as f:
            data = pickle.load(f)

        n_repeats = int(data['n_repeats'])

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3

        presentations_per_block = n_repeats * presentations_per_repeat

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

        recordings = pd.DataFrame({
            'started_at': [started_at],
            'thorsync_path': [thorsync_dir],
            'thorimage_path': [thorimage_dir],
            'stimulus_data_path': [stimulus_data_path],
            'first_block': [first_block],
            'last_block': [last_block],
            'n_repeats': [n_repeats],
            'presentations_per_repeat': [presentations_per_repeat]
        })

        if process_time_averages:
            # TODO TODO TODO just load .raw? (or .tif in case of accidentally
            # having saved an ome tiff)
            # just for .raw/.tif where raw would be/.tif in separate tif dir
            tif_path = join(raw_fly_dir, 'tif_stacks', thorimage_id  + '.tif')
            if not exists(tif_path):
                print('Raw TIFF {} did not exist!'.format(tif_path))
                recording_outcome['no_movie'] = True
                recording_outcomes.append(recording_outcome)
                continue

            print('Loading TIFF from {}...'.format(tif_path), flush=True,
                end='')
            movie = tifffile.imread(tif_path)
            print(' done.')

            # TODO but maybe want to check consistent wrt both movie and matlab
            # cnmf output in case where we load both?? (-> don't name either
            # just n_frames)
            n_frames = movie.shape[0]

            full_frame_avg_trace = u.full_frame_avg_trace(movie)
            recordings['full_frame_avg_trace'] = [[float(x) for x in
                full_frame_avg_trace.astype('float32')]]

        if ACTUALLY_UPLOAD:
            # TODO just completely delete this fn at this point?
            # TODO TODO TODO also replace other cases that might need to be
            # updated w/ pg_upsert based solution. presentations table
            # especially.
            ####u.to_sql_with_duplicates(recordings, 'recordings')
            recordings.set_index('started_at', inplace=True)
            recordings.to_sql('recordings', conn, if_exists='append',
                method=u.pg_upsert)

            db_recording = pd.read_sql_query('SELECT * FROM recordings WHERE ' +
                "started_at = '{}'".format(pd.Timestamp(started_at)), conn,
                index_col='started_at')
            db_recording = db_recording[recordings.columns]

            assert recordings.equals(db_recording)

        first_presentation = first_block * presentations_per_block
        last_presentation = (last_block + 1) * presentations_per_block - 1

        # TODO will need to augment w/ concentration info somehow...
        # maybe handle in a way specific to natural_odors project?

        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [0 if x == 'paraffin' else
                natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })

        odor_pair_list = data['odor_pair_list'][
            first_presentation:(last_presentation + 1)]

        assert len(odor_pair_list) % presentations_per_block == 0

        if ACTUALLY_UPLOAD:
            u.to_sql_with_duplicates(odors, 'odors')

            # TODO make unique id before insertion? some way that wouldn't
            # require the IDs, but would create similar tables?

            db_odors = pd.read_sql('odors', conn)
            # TODO TODO in general, the name alone won't be unique, so use
            # another strategy
            db_odors.set_index('name', inplace=True)

            # TODO test slicing
            # TODO make sure if there are extra trials in matlab, these get
            # assigned to first
            # + if there are less in matlab, should error
            # TODO invert to check
            # TODO is this sql table worth anything if both keys actually need
            # to be referenced later anyway?

            # TODO only add as many as there were blocks from thorsync timing
            # info?
            odor1_ids = [db_odors.at[o1,'odor_id'] for o1, _ in odor_pair_list]
            odor2_ids = [db_odors.at[o2,'odor_id'] for _, o2 in odor_pair_list]

            # TODO TODO make unique first. only need order for filling in the
            # values in responses.
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

        if upload_matlab_cnmf_output:
            evil.evalc("clear; data = load('{}', 'S');".format(mat))

            # sDFF - filtered traces (filtered how?)
            # F0 - background fluorescence (dims?)
            #print(' done')
            try:
                S = evil.eval('data.S')
            except matlab.engine.MatlabExecutionError:
                # TODO inspect error somehow to see if it's a memory error?
                print('CNMF still needs to be run on this data')
                # TODO TODO since process_time_averages is in same loop, maybe
                # don't continue / otherwise restructure
                # TODO maybe support recording outcome flag here?
                continue

            # TODO rename to indicate this is not just a filtered version of C?
            # (and how exactly is it different again?)
            # Loading w/ h5py lead to transposed indices wrt loading w/ MATLAB
            # engine.
            #filtered_df_over_f = np.array(S['sDFF']).T
            # TODO possible to just load a subfield of the CNM object / S w/
            # load semantics?
            df_over_f = np.array(S['DFF']).T

            # TODO quantitatively compare sDFF / DFF. (s=smoothed)
            # maybe just upload non-smoothed and let people smooth downstream if
            # they want?

            # TODO could check for sCNM, to load that in cases where we can't
            # make a cnmf object? (but we should able able to here...)
            try:
                evil.evalc("clear; data = load('{}', 'CNM');".format(mat))
            except matlab.engine.MatlabExecutionError as e:
                # TODO inspect error somehow to see if it's a memory error?
                # TODO print to stderr
                print(e)
                continue

            raw_f = np.array(evil.eval('data.CNM.C')).T

        try:
            ti = u.load_mat_timing_information(mat)
        except matlab.engine.MatlabExecutionError as e:
            print(e)
            # TODO recording outcome? or just fail here?
            continue

        # TODO maybe change to return None/dict rather than deal with lists
        # and then just check not None in a list comprehension aggregating them?
        ti_code_version = u.get_matfile_var(mat, 'ti_code_version')

        # TODO maybe it was not possible to calc timing info because input has
        # moved / was corrupted (either the thorsync file or the .mat it gets
        # converted to)?
        if calc_timing_info and update_timing_info:
            # TODO probably just print error message and continue
            assert ti_code_version != curr_ti_code_version, \
                'timing information should have been regenerated but was not'

        # TODO TODO would have to check whether motion correction was run for
        # this particular experiment in order to determine whether or not to
        # add the two motion correction code repos to this.
        # May ultimately make the most sense to have the outer loop be over
        # keys to recordings, and have trace process just follow motion
        # correction?
        if ACTUALLY_UPLOAD:
            u.upload_analysis_info(started_at, analyzed_at, ti_code_version)

        # TODO dtype appropriate?
        frame_times = np.array(ti['frame_times']).flatten()

        # Frame indices for CNMF output.
        # Of length equal to number of blocks. Each element is the frame
        # index (from 1) in CNMF output that starts the block, where
        # block is defined as a period of continuous acquisition.
        block_first_frames = np.array(ti['block_start_frame'], dtype=np.uint32
            ).flatten() - 1

        # TODO delete / use something else to get num blocks, if just using this
        # list of sample times to count num blocks... (change matlab code)
        # TODO after better understanding where block_start_frame comes from,
        # could get rid of this check if it's just tautological
        block_ic_thorsync_idx = np.array(ti['block_start_sample']).flatten()
        assert len(block_ic_thorsync_idx) == len(block_first_frames), \
            'variables in MATLAB ti have inconsistent # of blocks'
        #

        # TODO unit tests for block handling code
        n_blocks_from_gsheet = last_block - first_block + 1
        n_blocks_from_thorsync = len(block_first_frames)

        assert (len(odor_pair_list) == (last_block - first_block + 1) *
            presentations_per_block)

        n_presentations = n_blocks_from_gsheet * presentations_per_block

        err_msg = ('{} blocks ({} to {}, inclusive) in Google sheet {{}} {} ' +
            'blocks from ThorSync.').format(n_blocks_from_gsheet,
            first_block + 1, last_block + 1, n_blocks_from_thorsync)
        fail_msg = (' Fix in Google sheet, turn off ' +
            'cache if necessary, and rerun.')

        # TODO factor all this code out, but especially these checks, so that
        # populate_db would catch this as well
        if n_blocks_from_gsheet > n_blocks_from_thorsync:
            raise ValueError(err_msg.format('>') + fail_msg)

        elif n_blocks_from_gsheet < n_blocks_from_thorsync:
            if allow_gsheet_to_restrict_blocks:
                warnings.warn(err_msg.format('<') + (' This is ONLY ok if you '+
                    'intend to exclude the LAST {} blocks in the Thor output.'
                    ).format(n_blocks_from_thorsync - n_blocks_from_gsheet))
            else:
                raise ValueError(err_msg.format('<') + fail_msg)

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
        # TODO just assert equal to sorted version
        # TODO some fn for checking sorted? (i mean it's linear vs n*log(n)...)

        block_last_frames = np.array(ti['block_end_frame'], dtype=np.uint32
            ).flatten() - 1

        if allow_gsheet_to_restrict_blocks:
            # TODO unit test for case where first_block != 0 and == 0
            # w/ last_block == first_block and > first_block
            block_first_frames = block_first_frames[
                :(last_block - first_block + 1)]
            block_last_frames = block_last_frames[
                :(last_block - first_block + 1)]

            assert len(block_first_frames) == n_blocks_from_gsheet
            assert len(block_last_frames) == n_blocks_from_gsheet

            odor_onset_frames = odor_onset_frames[
                :(last_presentation - first_presentation + 1)]

            odor_offset_frames = odor_offset_frames[
                :(last_presentation - first_presentation + 1)]

            assert len(odor_onset_frames) == n_presentations
            assert len(odor_offset_frames) == n_presentations

            frame_times = frame_times[:(block_last_frames[-1] + 1)]

        # Since trace is computed on full movie, not the subset we define above.
        last_frame = block_last_frames[-1]

        assert len(odor_onset_frames) == len(odor_pair_list)

        # TODO delete try / except
        try:
            assert len(odor_onset_frames) == len(odor_offset_frames)
            print('past first assertion')
            assert (len(block_first_frames) * presentations_per_block == 
                    len(odor_onset_frames)), ('timing information wrong. ' +
                '# presentations inconsistent with # blocks.')

        except AssertionError:
            print(len(block_first_frames))
            print(len(odor_onset_frames))
            print(ti_code_version[0])
            import ipdb; ipdb.set_trace()

        # TODO how to get odor pid

        if upload_matlab_cnmf_output:
            raise NotImplementedError('not currently getting new required ' +
                'segmentation_run info from matlab output')

            # A has footprints
            # from caiman docs:
            # "A: ... (d x K)"
            # "K: ... # of neurons to extract"
            # so what is d? (looks like 256 x 256, the # of pixels)
            # dims=dimensions of image (256x256)
            # T is # of timestamps

            # TODO TODO maybe get some representation of the sparse matrix that
            # i can use to create one from the scipy constructors, to minimize
            # amount of data sent from matlab
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

            # TODO TODO TODO make sure these cell IDs match up with the ones
            # from below!!!

            footprint_dfs = []
            for cell_num in range(n_footprints):
                sparse = coo_matrix(footprints[:,:,cell_num])
                raise NotImplementedError('check row/col below not broken')
                footprint_dfs.append(pd.DataFrame({
                    'recording_from': [started_at],
                    'cell': [cell_num],
                    # Can be converted from lists of Python types, but
                    # apparently not from numpy arrays or lists of numpy scalar
                    # types.
                    # TODO TODO just move appropriate casting to my to_sql
                    # function, and allow having numpy arrays (get type info
                    # from combination of that and the database, like in other
                    # cases)
                    # TODO TODO TODO TODO i swapped row and col here 8/13 to fix
                    # some what i think was source of need for extra tranposes.
                    # check that inputting matlab stuff did not require old
                    # order.
                    'x_coords': [[int(x) for x in sparse.row.astype('int16')]],
                    'y_coords': [[int(x) for x in sparse.col.astype('int16')]],
                    'weights': [[float(x) for x in
                                 sparse.data.astype('float32')]],
                    # TODO TODO generate all required info
                    # (have everything needed in matlab output, or would i have
                    # to relax some other requirements?)
                    'segmentation_run': None
                }))

            footprint_df = pd.concat(footprint_dfs, ignore_index=True)
            # TODO filter out footprints less than a certain # of pixels in
            # cnmf?  (is 3 pixels really reasonable?)
            if ACTUALLY_UPLOAD:
                u.to_sql_with_duplicates(footprint_df, 'cells', verbose=True)

            n_frames, n_cells = df_over_f.shape
            assert n_cells == n_footprints

        recording_outcomes.append(recording_outcome)
        # TODO if i want to keep this, remove checks for process_time_averages
        # after this
        if not (ACTUALLY_UPLOAD and process_time_averages):
            continue

        # TODO TODO assertion to detect whatever this was talking about...
        # TODO why 474 x 4 + 548 in one case? i thought frame numbers were
        # supposed to be more similar... (w/ np.diff(odor_onset_frames))
        first_onset_frame_offset = odor_onset_frames[0] - block_first_frames[0]

        # TODO why is odor_onset_frames[0] not used? always 0?
        # should start_frames just be (odor_onset_frames -
        # first_onset_frame_offset)?
        start_frames = np.append(0,
            odor_onset_frames[1:] - first_onset_frame_offset)
        stop_frames = np.append(
            odor_onset_frames[1:] - first_onset_frame_offset - 1, n_frames)
        lens = [stop - start for start, stop in zip(start_frames, stop_frames)]

        # This will fail in case of unsliced movie now...
        # would have to do slice the movie as the gui does and then get # frames
        if upload_matlab_cnmf_output:
            assert frame_times.shape[0] == n_frames

        print(start_frames)
        print(stop_frames)
        # TODO TODO which discrepancies? put back whichever assertion would fail
        # TODO find where the discrepancies are!
        # seems n_frames could be sum(lens) + len(lens) - 1 in some cases?
        print(sum(lens))
        print(n_frames)

        # TODO assert here that all frames add up / approx

        # TODO TODO either warn or err if len(start_frames) is !=
        # len(odor_pair_list)

        if process_time_averages:
            full_frame_avg_trace = full_frame_avg_trace[:(last_frame + 1)]
            assert full_frame_avg_trace.shape[0] == len(frame_times)

        # TODO make this work in not ACTUALLY_UPLOAD case w/o odor1_ids
        # or odor2_ids
        odor_id_pairs = [(o1,o2) for o1,o2 in zip(odor1_ids, odor2_ids)]

        comparison_num = -1

        for i in range(len(start_frames)):
            if i % presentations_per_block == 0:
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
            # (below line copied from above just to see what onset_frame is)
            #first_onset_frame_offset = odor_onset_frames[0] - block_first_frames[0]
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
            assert len(presentation_frametimes) > 1

            # TODO delete try/except
            try:
                odor_pair = odor_id_pairs[i]
            except IndexError:
                print('odor_id_pairs:', odor_id_pairs)
                print('len(odor_id_pairs):', len(odor_id_pairs))
                print('len(start_frames):', len(start_frames))
                print('i:', i)
                import ipdb; ipdb.set_trace()
            #
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
                'analysis': analyzed_at,
                'comparison': comparison_num,
                'odor1': odor1,
                'odor2': odor2,
                'repeat_num': repeat_num,
                'odor_onset_frame': direct_onset_frame,
                'odor_offset_frame': offset_frame,
                'from_onset': [[float(x) for x in presentation_frametimes]]
            })

            # TODO run this over all of the prep checking stuff?  for
            # natural_odors at least (where it should all have been ~same
            # concentration of ethyl acetate?) (which would i guess mean any
            # prep checking immediately followed by natural_odors?)
            if process_time_averages:
                fps = u.get_thorimage_fps(thorimage_dir)
                # TODO does it really take this long in most cases?
                rise_time_s = 2.0 #1.5
                rise_time_frames = int(round(rise_time_s * fps))
                decay_start_frame = direct_onset_frame + rise_time_frames

                # Relationship between presentation indices and times will
                # differ from presentation_frametimes, to the extent of rise
                # time.

                # TODO subtract something added to onset_time?
                decay_frametimes = (frame_times[decay_start_frame:stop_frame] -
                                    onset_time)
                decay_trace = full_frame_avg_trace[
                    decay_start_frame:stop_frame]
                assert decay_trace.shape == decay_frametimes.shape

                # This still only goes to odor onset, in case rise time is
                # slightly misspecified.
                baseline = full_frame_avg_trace[start_frame:direct_onset_frame]
                avg_baseline = np.mean(baseline)
                stddev_baseline = np.std(baseline)

                avg_df_over_f = (decay_trace - avg_baseline) / avg_baseline
                # TODO TODO should this be avg_df_over_f - new baseline computed
                # over df_over_f?
                zchange_dff = (decay_trace - avg_baseline) / stddev_baseline

                # TODO principled way of setting this? to maximize some kind of
                # separability / robustness?
                resp_time_s = 5.0
                resp_time_frames = int(round(resp_time_s * fps))

                avg_dff_5s = np.mean(avg_df_over_f[:(resp_time_frames + 1)])
                avg_zchange_5s = np.mean(zchange_dff[:(resp_time_frames + 1)])

                presentation['avg_dff_5s'] = avg_dff_5s
                presentation['avg_zchange_5s'] = avg_zchange_5s

                # TODO TODO catch runtimewarning overflow and turn into error
                (scale, tau, offset), sigmas = u.fit_exp_decay(
                    avg_df_over_f,
                    times=decay_frametimes,
                    numerical_scale=200
                )
                # TODO TODO test that w/ n * numerical_scale,
                # scale and offset are scaled by n, and tau is the same
                # (so that the scaling can be inverted before entering the
                # parametrs in the database)

                presentation['calc_exp_from'] = rise_time_s
                presentation['exp_scale'] = scale
                presentation['exp_tau'] = tau
                presentation['exp_offset'] = offset
                presentation['exp_scale_sigma'] = sigmas[0]
                presentation['exp_tau_sigma'] = sigmas[1]
                presentation['exp_offset_sigma'] = sigmas[2]
                # TODO maybe do store (all) pcov components?
                # might help evaluate goodness of fit, and that might be more
                # useful than just means of params, since some params can be
                # huge / small when fit is poor...
                # (see notes in fit_exp_decay)

                # TODO are these sufficient, or would it also make sense to
                # include average dffs for various numbers of seconds after odor
                # onset? try it?

                print('avg_dff_5s:', avg_dff_5s)
                print('avg_zchange_5s:', avg_zchange_5s)

                # TODO reformat?
                print('scale:', scale)
                print('tau:', tau)
                print('offset:', offset)
                #

                #
                # TODO TODO still plot times before hardcoded rise time,
                # just so that i can see i'm not throwing out some earlier peaks
                # TODO or just smooth and find maxima for everything ->
                # use that as rise time
                '''
                model_trace = u.exp_decay(decay_frametimes, scale, tau, offset)
                import matplotlib.pyplot as plt
                plt.plot(decay_frametimes, avg_df_over_f, label='data')
                plt.plot(decay_frametimes, model_trace, label='fit')
                plt.show()
                '''
                #

            if ACTUALLY_UPLOAD:
                # TODO TODO TODO is this insertion method causing some
                # parameters to not actually get updated?
                # use to_sql w/ pg_upsert?
                u.to_sql_with_duplicates(presentation, 'presentations')

            db_presentations = pd.read_sql('presentations', conn)

            # maybe share w/ code that checks distinct to decide whether to
            # load / analyze?
            key_cols = [
                'prep_date',
                'fly_num',
                'recording_from',
                'analysis',
                'comparison',
                'odor1',
                'odor2',
                'repeat_num'
            ]

            presentation_ids = (db_presentations[key_cols] ==
                                presentation[key_cols].iloc[0]).all(axis=1)
            assert presentation_ids.sum() == 1
            presentation_id = db_presentations.loc[presentation_ids,
                'presentation_id'].iat[0]

            # Check correct insertion
            db_presentation = db_presentations.loc[presentation_ids,
                presentation.columns].reset_index(drop=True)
            diff = u.diff_dataframes(presentation, db_presentation)
            if diff is not None:
                print(diff)
                raise IOError('SQL insertion failed')

            if upload_matlab_cnmf_output:
                # TODO get remy to save it w/ less than 64 bits of precision?
                # (or just change matlab code myself)
                presentation_dff = df_over_f[start_frame:stop_frame, :]
                presentation_raw_f = raw_f[start_frame:stop_frame, :]

                # Assumes that cells are indexed same here as in footprints.
                cell_dfs = []
                for cell_num in range(n_cells):

                    cell_dff = presentation_dff[:, cell_num].astype('float32')
                    cell_raw_f = presentation_raw_f[:, cell_num].astype(
                        'float32')

                    cell_dfs.append(pd.DataFrame({
                        'presentation_id': [presentation_id],
                        'recording_from': [started_at],
                        'cell': [cell_num],
                        'df_over_f': [[float(x) for x in cell_dff]],
                        'raw_f': [[float(x) for x in cell_raw_f]]
                    }))
                response_df = pd.concat(cell_dfs, ignore_index=True)

                if ACTUALLY_UPLOAD:
                    u.to_sql_with_duplicates(response_df, 'responses')

                # TODO put behind flag?
                '''
                new_db_presentations = pd.read_sql_query('SELECT DISTINCT ' +
                    'prep_date, fly_num, recording_from, comparison FROM' +
                    'presentations', conn)

                print(new_db_presentations)
                print(len(new_db_presentations))
                '''

            # TODO move out of this populate_db script?
            # Case for keeping it in here, is that all the dependencies will
            # still need to be calculated to evaluate a given experiment,
            # although maybe the input would need to be rewritten a little to
            # just select one specific (the most recent) experiment.
            # TODO factor out and still call it here?
            if process_time_averages:
                # TODO unit test this bit (w/ or w/o multiple versions of
                # analysis, etc)

                # TODO maybe don't exclude current presentation_id?
                # TODO TODO TODO uncomment
                db_curr_odor_presentations = db_presentations[
                    ~presentation_ids & 
                    (db_presentations.odor1 == odor1) &
                    (db_presentations.odor2 == odor2)
                ]
                # TODO TODO print which odors / concs this is too

                # TODO maybe just return empty df as base case?
                response_stats = \
                    u.latest_response_stats(db_curr_odor_presentations)

                if response_stats is None:
                    continue

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
                # TODO compute percentile of current set of response_stat_cols
                # to all past response_stat_cols. idiomatic way to do that in
                # pandas?

                good_bad_means = response_stats.groupby('accepted'
                    )[response_stat_cols].mean()

                # TODO maybe print # accepted vs not, for context.
                # might just be a lot not labelled one way or the other?
                # TODO list what was accepted and what wasn't too
                print(good_bad_means)

                # TODO maybe also take repeat_num into account?

                # TODO define cutoff relative to good / not good and then print
                # which side of that cutoff we are on
                # TODO maybe cluster all stuff, good or not?
                # TODO good or not, print how this compares to other recordings,
                # maybe using a percentile?

            print('Done processing presentation {}'.format(i))

        # TODO check that all frames go somewhere and that frames aren't
        # given to two presentations. check they stay w/in block boundaries.
        # (they don't right now. fix!)

# TODO print recording_outcomes stuff
recording_outcomes_df = pd.DataFrame(recording_outcomes)
import ipdb; ipdb.set_trace()


# TODO print unused stimfiles / option to delete them

