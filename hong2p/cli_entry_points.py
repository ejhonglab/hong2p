
import argparse
import os
from os.path import isdir, exists, join

import pandas as pd

from hong2p import util, thor
from hong2p.viz import showsync
from hong2p.suite2p import print_suite2p_params


# NOTE: to add additional endpoints:
# 1) Implement a function in here.
#
# 2) Edit __init__.py to also import that function by name.
#
# 3) Edit ../setup.py to add another entry in the list behind 'console_scripts',
#    in the `entry_points` keyword argument to `setup`.


# TODO TODO add entrypoint for diffing odor set between two flies (reporting any odors
# in one but not the other, any concentrations that changed, any solvents that changed)

def thor2tiff_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('thor_raw_dir',
        help='path containing .raw and metadata created by ThorImage'
    )
    # TODO default to just changing extention of input raw? or something like
    # that (or make output name required...)
    parser.add_argument('-o', '--output-name',
        help='full path of .tif to create. raw.tif in same directory by default'
    )
    parser.add_argument('-w', '--overwrite', action='store_true',
        help='otherwise, will fail if output already exists'
    )
    # TODO also expose flip_lr? name can be handled more consistently in code...
    parser.add_argument('-c', '--check-round-trip', action='store_true',
        help='reads created TIFF and checks it is equal to data from ThorImage raw'
    )
    args = parser.parse_args()
    raw_dir = args.thor_raw_dir
    output_name = args.output_name

    # Options are 'err', 'overwrite', 'ignore', or 'load'
    if_exists = 'overwrite' if args.overwrite else 'err'

    check_round_trip = args.check_round_trip

    util.thor2tiff(raw_dir, output_name=output_name, if_exists=if_exists,
        check_round_trip=check_round_trip
    )


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
    # TODO say what is show by default (or excluded...)
    parser.add_argument('-a', '--all', action='store_true',
        help='will display all data in HDF5 (except frame counter)'
    )
    parser.add_argument('-d', '--datasets', action='store',
        help='comma separated list of (normalized) names of traces to plot'
    )
    args = parser.parse_args()
    thorsync_dir = args.thorsync_dir
    verbose = args.verbose
    exclude_datasets = False if args.all else None
    datasets = None if not args.datasets else ['gctr'] + args.datasets.split(',')

    showsync(thorsync_dir, verbose=verbose, exclude_datasets=exclude_datasets,
        datasets=datasets
    )


def suite2p_params_cli():
    """Prints data specific parameters so they can be set in suite2p GUI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('thorimage_dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    parser.add_argument('-s', '--shape', action='store_true', help='also print movie '
        'shape (for picking registration block size, batch size, etc)'
    )
    args = parser.parse_args()
    print_suite2p_params(args.thorimage_dir, print_movie_shape=args.shape)


def print_dir_fn_cli_wrapper(fn):

    def cli_wrapper():
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', action='store_true')
        args = parser.parse_args()
        verbose = args.verbose

        fn(verbose=verbose)

    return cli_wrapper


# TODO factor my ~/.bash_aliases commands using these (e.g. 2p, 2pr, 2pa, print_2pa)
# into a script in hong2p + call that script to install them in my ~/.bash_aliases
# (+ provide script / instructions for other people to install them)
@print_dir_fn_cli_wrapper
def print_data_root(verbose=False):
    print(util.data_root(verbose=verbose))


@print_dir_fn_cli_wrapper
def print_raw_data_root(verbose=False):
    print(util.raw_data_root(verbose=verbose))


def print_analysis_intermediates_root():
    # This one doesn't take have a `verbose` kwarg like the others
    print(util.analysis_intermediates_root())


def print_paired_thor_subdirs():
    # TODO clarify in doc what "experiment time" is. is it when thorimage started?
    # ended?
    """Prints pairs of (ThorImage, ThorSync) dirs that are direct descendents of input.

    Printed in order of ThorImage experiment times.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    args = parser.parse_args()
    parent = args.dir

    # TODO TODO expose some kwargs to CLI? may need to if we have e.g. 'anat' w/o
    # corresponding ThorSync dir
    paired_dirs = thor.pair_thor_subdirs(parent)

    # TODO opt to include get_thorimage_time in nice format?

    df = pd.DataFrame.from_records(columns=['ThorImage', 'ThorSync'], data=[
        (i.name, s.name) for (i, s) in sorted(paired_dirs, key=lambda p:
        thor.get_thorimage_time(p[0]))
    ])
    print(df.to_string(index=False))


def print_thorimage_subdir_notes():
    """Prints the note field of ThorImage subdirectories.

    Printed in order of ThorImage experiment times.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    args = parser.parse_args()
    parent = args.dir

    thorimage_subdirs = thor.thorimage_subdirs(parent)
    for thorimage_dir in sorted(thorimage_subdirs, key=lambda d:
        thor.get_thorimage_time(d)):

        print(util.shorten_path(thorimage_dir))
        print(thor.get_thorimage_notes(thorimage_dir))
        print('')

