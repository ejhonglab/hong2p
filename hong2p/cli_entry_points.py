#!/usr/bin/env python3

import argparse
from os.path import isdir, exists, join

import hong2p.util as u
from hong2p.thor import read_movie


def thor2tiff():
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
    assert isdir(raw_dir)

    output_name = args.output_name
    if output_name is None:
        output_name = join(raw_dir, 'converted' + tiff_ext)

    if not args.overwrite:
        assert not exists(output_name)

    print('Reading RAW movie...', flush=True, end='')
    from_raw = read_movie(raw_dir)
    print(' done', flush=True)

    # TODO TODO TODO try to figure out if anything can be done about tifffile
    # using so much memory on writing (says "Killed" and exits in the middle of
    # writing when trying to write what should be a ~5.5GB movie when i have
    # close to 20GB of RAM free...). maybe memory profile my own code to see if
    # i'm doing something stupid.

    print(f'Writing TIFF to {output_name}...', flush=True, end='')
    u.write_tiff(output_name, from_raw)
    print(' done', flush=True)

