#!/usr/bin/env python3
"""
Loop over all avaible ThorSync HDF5 files and check whether "Frame In"/"FrameIn"
is ever non-zero. Also notifies about corrupted HDF5 files.
"""

from os.path import join, split
import glob
from pprint import pprint

from hong2p import thor, util


def main():
    check_frame_in = False

    # TODO could exclude everything besides frame_in for purposes here, so
    # switch to white-list if that kwarg to load_thorsync_hdf5 is working
    cfi_datasets = [
        # TODO if i end up defaulting to err-ing if there are missing dataset
        # names, will either need to implement+enable warn-only or something
        # (because only expecting one of these, but named seems to not always
        # have been the same)
        'frame_in',
        'framein',
    ]


    if check_frame_in:
        load_kwargs = dict(datasets=cfi_datasets)
    else:
        load_kwargs = dict(return_dataset_names_only=True)
        thorsync_dir2dataset_names = dict()
        dataset_name_set = set()

    data_dir = util.raw_data_root()
    h5s = glob.glob(join(data_dir, '*/*/*/Episode001.h5'))
    thorsync_dirs = [split(x)[0] for x in h5s]
    # TODO tqdm, or will prints bork that?
    for d in thorsync_dirs:
        print(d)
        try:
            ret = thor.load_thorsync_hdf5(d, **load_kwargs)

        # Catches various types of HDF5 file corruption
        except RuntimeError:
            print('HDF5 corrupted!\n')
            continue

        # TODO TODO TODO also catch + diagnose cause of the OSError i
        # encountered in at least one case
        except OSError as e:
            print(e)
            print()
            continue

        if check_frame_in:
            df = ret
            if 'frame_in' in df.columns:
                frame_in = df.frame_in
            elif 'framein' in df.columns:
                frame_in = df.framein

            print(frame_in.value_counts())
            print()

        else:
            names = ret
            thorsync_dir2dataset_names[d] = names
            dataset_name_set.update(names)

    if not check_frame_in:
        print('\nThorsync Directories to HDF5 dataset names:')
        pprint(thorsync_dir2dataset_names)

        print('\nUnique dataset names:')
        pprint(dataset_name_set)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

