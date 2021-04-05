#!/usr/bin/env python3

import sys
import time
import traceback
from os.path import exists
from pprint import pprint

import numpy as np
import pandas as pd

from hong2p import thor, util, viz


open_with_showsync = True

def analyze(df, image_and_sync_pairs=None, hdf5_corrupted_paths=None):

    by_occurence = df.groupby(['file', 'lineno']).size().sort_values(ascending=False)

    if image_and_sync_pairs is not None:
        n_tried = len(image_and_sync_pairs)

        if hdf5_corrupted_paths is not None:
            n_corrupted = len(hdf5_corrupted_paths)
            print(f'{len(hdf5_corrupted_paths)} corrupted HDF5 files')

            n_tried = n_tried - n_corrupted

        n_failed = len(df)
        print(f'Failed on {n_failed}/{n_tried} ({n_failed/n_tried:.3f})')

    print('Errors from most frequent to least:')
    for (fname, lineno), count in by_occurence.iteritems():

        gdf = df[(df['file'] == fname) & (df.lineno == lineno)]

        assert gdf.function.nunique() == 1
        function = gdf.iloc[0].function

        print(f'{fname}:{lineno} in {function}')
        print('count:', count)

        # TODO TODO why is this not true? (see command by 'lineno' inside loop in fn
        # below)
        #assert gdf.exception_type.nunique() == 1
        print('unique exception types:', gdf.exception_type.unique())

        # TODO reason some are np.nan? how could line be empty?
        print('unique lines:')
        for x in gdf.line.unique():
            print(x)
        print()

        print('affected data:')
        for row in gdf.itertuples():
            print(row.thorsync_dir)

        if open_with_showsync:
            # TODO also allow labelling of good / bad (saved to a yaml / text file,
            # as the known_garbage_thorsync_data.yaml i made manually)
            # TODO also change loop below to read these if exists and skip
            # (or maybe change util fns that generate these directory [pairs]?)
            for d in gdf.thorsync_dir:
                viz.showsync(thorsync_dir=d)

        print('\n')


def main():
    use_failing_pairs = True
    #use_failing_pairs = False

    analyze_csv_only = True

    test_paths_csv = 'test_thor_failing.csv'

    if use_failing_pairs:
        df = pd.read_csv(test_paths_csv)

        if analyze_csv_only:
            analyze(df)
            sys.exit()

        image_and_sync_pairs = [
            (x.thorimage_dir, x.thorsync_dir) for x in df.itertuples()
        ]

    else:
        image_and_sync_pairs = util._all_paired_thor_dirs()
        #pprint(image_and_sync_pairs)
        # TODO maybe also print which dirs couldn't be paired (via verbose flag to
        # this util fn perhaps?)

    failing_pairs2err_info = dict()
    hdf5_corrupted_paths = []

    for thorimage_dir, thorsync_dir in image_and_sync_pairs:
        print('thorimage_dir:', thorimage_dir)
        print('thorsync_dir:', thorsync_dir)

        # TODO possible to parse / print thor software versions + which computer
        # it was acquired on?

        before = time.time()
        print('loading HDF5...', flush=True, end='')

        try:
            verbose = False
            df = thor.load_thorsync_hdf5(thorsync_dir, verbose=verbose)

        # Intended to catch errors like:
        # "RuntimeError: Object visitation failed (wrong B-tree signature)"
        # ...raised by (presumably corrupted) data in 2019-01-23/2/SyncData004
        except RuntimeError as e:
            hdf5_corrupted_paths.append(thorsync_dir)
            print('HDF5 corrupted!\n')
            continue

        took_s = time.time() - before
        print(f'done ({took_s:.1f}s)')

        # TODO print aggregate list of which fail at end?

        # This function and several it calls have a bunch of assertions, so if
        # they work on a lot of data, it does help argue the functions are
        # working correctly.
        try:
            _ = thor.assign_frames_to_odor_presentations(df,
                thorimage_dir
            )
        except Exception as e:
            etype, eobj, etb = sys.exc_info()
            traceback.print_exception(etype, eobj, etb)

            stack_summary = traceback.extract_tb(etb)
            deepest_frame = stack_summary[-1]

            failing_pairs2err_info[(thorimage_dir, thorsync_dir)] = {
                'thorimage_dir': thorimage_dir,
                'thorsync_dir': thorsync_dir,

                'file': deepest_frame.filename,
                # TODO TODO why do some values of (filename, <this>) have multiple
                # values (e.g. in exception_type / msg cols)??? am i using the wrong
                # line number???
                'lineno': deepest_frame.lineno,
                'function': deepest_frame.name,

                # The contents of the line (as would be printed in traceback)
                'line': deepest_frame.line,

                # Can't figure out an easy way to remove <class '<x>'>
                # surrounding part of string.
                'exception_type': str(type(e)),

                # Many of these would not be useful to group on, but can still
                # use this to print a representative message at the end.
                'msg': str(e),
            }

            print()
            continue

        print()

    df = pd.DataFrame(failing_pairs2err_info.values()).replace('', np.nan)
    if exists(test_paths_csv):
        print(f"{test_paths_csv} existed, saving with 'new_' prefix")
        test_paths_csv = 'new_' + test_paths_csv

        if exists(test_paths_csv):
            print('overwriting!')

    df.to_csv(test_paths_csv, index=False)

    analyze(df, image_and_sync_pairs=image_and_sync_pairs,
        hdf5_corrupted_paths=hdf5_corrupted_paths
    )


if __name__ == '__main__':
    main()

