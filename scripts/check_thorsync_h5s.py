#!/usr/bin/env python3
"""
Loop over all avaible ThorSync HDF5 files and check whether "Frame In"/"FrameIn"
is ever non-zero. Also notifies about corrupted HDF5 files.
"""

from os.path import join, split
import glob

from hong2p import thor, util


def main():
    # TODO could exclude everything besides frame_in for purposes here, so
    # switch to white-list if that kwarg to load_thorsync_hdf5 is working
    datasets = [
        # TODO if i end up defaulting to err-ing if there are missing dataset
        # names, will either need to implement+enable warn-only or something
        # (because only expecting one of these, but named seems to not always
        # have been the same)
        'frame_in',
        'framein',
    ]

    data_dir = util.raw_data_root()
    h5s = glob.glob(join(data_dir, '*/*/*/Episode001.h5'))
    thorsync_dirs = [split(x)[0] for x in h5s]
    # TODO tqdm, or will prints bork that?
    for d in thorsync_dirs:
        print(d)
        try:
            df = thor.load_thorsync_hdf5(d, datasets=datasets)

        # Catches various types of HDF5 file corruption
        except RuntimeError:
            print('HDF5 corrupted!\n')
            continue

        # TODO TODO TODO also catch + diagnose cause of the OSError i
        # encountered in at least one case

        if 'frame_in' in df.columns:
            frame_in = df.frame_in
        elif 'framein' in df.columns:
            frame_in = df.framein

        print(frame_in.value_counts())
        print()

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

