#!/usr/bin/env python3

import argparse
from os.path import isdir, exists, join

from hong2p import util
from hong2p.thor import read_movie
from hong2p.viz import showsync


# NOTE: to add additional endpoints:
# 1) Implement a function in here.
#
# 2) Edit __init__.py to also import that function by name.
#
# 3) Edit ../setup.py to add another entry in the list behind 'console_scripts',
#    in the `entry_points` keyword argument to `setup`.


def thor2tiff_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('thor_raw_dir',
        help='path containing .raw and metadata created by ThorImage'
    )
    # TODO .tif or .tiff?
    tiff_ext = '.tif'
    # TODO default to just changing extention of input raw? or something like
    # that (or make output name required...)
    parser.add_argument('-o', '--output-name', default=None,
        help=f'name of {tiff_ext} to create'
    )
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
        help='otherwise, will fail if output already exists'
    )
    args = parser.parse_args()
    raw_dir = args.thor_raw_dir

    output_name = args.output_name
    if output_name is None:
        output_name = join(raw_dir, 'converted' + tiff_ext)

    if not args.overwrite:
        assert not exists(output_name)

    # TODO maybe also load metadata like fps (especially stuff, as w/ fps, that isn't
    # already baked into the TIFF, assuming the TIFF is saved correctly. so not
    # including stuff like z, c, xy), and print w/ -v flag?

    print('Reading RAW movie...', flush=True, end='')
    from_raw = read_movie(raw_dir)
    print(' done', flush=True)

    # TODO TODO TODO try to figure out if anything can be done about tifffile
    # using so much memory on writing (says "Killed" and exits in the middle of
    # writing when trying to write what should be a ~5.5GB movie when i have
    # close to 20GB of RAM free...). maybe memory profile my own code to see if
    # i'm doing something stupid.
    # TODO test read_movie on all thorimage .raw outputs i have to check which can
    # currently reproduce this issue

    print(f'Writing TIFF to {output_name}...', flush=True, end='')
    util.write_tiff(output_name, from_raw)
    print(' done', flush=True)


def showsync_cli():
    parser = argparse.ArgumentParser()
    # TODO maybe also accept fly / date / thorsync basename instead of this?
    # or also check for the existence of this downstream of `util.raw_data_root`?
    parser.add_argument('thorsync_dir',
        help='path containing output of a ThorSync recording'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
        help='will print all column names as they are in .h5 file and any renaming'
        'that occurs inside thor.load_thorsync_hdf5'
    )
    args = parser.parse_args()
    thorsync_dir = args.thorsync_dir
    verbose = args.verbose

    showsync(thorsync_dir, verbose=verbose)

