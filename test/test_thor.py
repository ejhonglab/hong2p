#!/usr/bin/env python3

import time
from pprint import pprint

from hong2p import thor, util


# TODO probably should make these all not returned by default (factor this list
# into thor.py)
hdf5_exclude_datasets = [
    'piezo_monitor',
    'pockels1_monitor',
    'frame_in',
    'light_path_shutter',
    'flipper_mirror',
    'pid',
    'frame_counter',
]

def main():
    image_and_sync_pairs = util._all_paired_thor_dirs()
    #pprint(image_and_sync_pairs)


    for thorimage_dir, thorsync_dir in image_and_sync_pairs:
        print('thorimage_dir:', thorimage_dir)
        print('thorsync_dir:', thorsync_dir)

        # TODO TODO possible to parse / print thor software versions + which
        # computer it was acquired on?

        before = time.time()
        print('loading HDF5...', flush=True, end='')

        df = thor.load_thorsync_hdf5(thorsync_dir,
            exclude_datasets=hdf5_exclude_datasets #, verbose=True
        )

        took_s = time.time() - before
        print(f'done ({took_s:.1f}s)')

        # TODO probably try / catch + log which fail, etc

        # This function and several it calls have a bunch of assertions, so if
        # they work on a lot of data, it does help argue the functions are
        # working correctly.
        _ = thor.assign_frames_to_odor_presentations(df,
            thorimage_dir
        )
        print()


if __name__ == '__main__':
    main()

