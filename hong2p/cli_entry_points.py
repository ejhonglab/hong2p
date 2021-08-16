
import argparse
from os.path import isdir, exists, join

from hong2p import util
from hong2p.viz import showsync
from hong2p.suite2p import print_suite2p_params


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
    # TODO default to just changing extention of input raw? or something like
    # that (or make output name required...)
    parser.add_argument('-o', '--output-name', default=None,
        help='full path of .tif to create. converted.tif in same directory by default'
    )
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
        help='otherwise, will fail if output already exists'
    )
    args = parser.parse_args()
    raw_dir = args.thor_raw_dir
    output_name = args.output_name

    # Options are 'err', 'overwrite', or 'ignore'
    if_exists = 'overwrite' if  args.overwrite else 'err'

    util.thor2tiff(raw_dir, output_name=output_name, if_exists=if_exists)


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
    parser.add_argument('-a', '--all', action='store_true',
        help='will display all data in HDF5 (except frame counter)'
    )
    args = parser.parse_args()
    thorsync_dir = args.thorsync_dir
    verbose = args.verbose
    exclude_datasets = False if args.all else None

    showsync(thorsync_dir, verbose=verbose, exclude_datasets=exclude_datasets)


def suite2p_params_cli():
    """Prints data specific parameters so they can be set in suite2p GUI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('thorimage_dir',
        help='path containing .raw and metadata created by ThorImage'
    )
    args = parser.parse_args()
    thorimage_dir = args.thorimage_dir

    print_suite2p_params(thorimage_dir)


def print_data_root():
    print(util.data_root())


def print_raw_data_root():
    print(util.raw_data_root())


def print_analysis_intermediates_root():
    print(util.analysis_intermediates_root())

