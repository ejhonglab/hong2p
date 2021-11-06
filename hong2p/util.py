"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
from os.path import join, split, exists, sep, isdir, isfile, getmtime, splitext
import pickle
import sys
from types import ModuleType
from datetime import datetime
import warnings
from pprint import pprint
import glob
import re
import hashlib
import functools

import numpy as np
from numpy.ma import MaskedArray
import pandas as pd
import xarray as xr
import yaml

# TODO now that i'm not trying to force a particular backend, move all the
# imports here and in a few other place back to a typical order

# Importing all matplotlib stuff besides mpl itself here to allow the above code
# to change backend. Might just need to defer import pyplot though...
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hong2p import matlab, db, thor, viz, olf

# Note: many imports were pushed down into the beginnings of the functions that
# use them, to reduce the number of hard dependencies.


# These three environment variables are in priority order (if first defined, it will be
# the one used).
DATA_ROOT_ENV_VAR = 'HONG2P_DATA'
NAS_PREFIX_ENV_VAR = 'HONG_NAS'

# If NAS_PREFIX_ENV_VAR is selected (i.e. DATA_ROOT_ENV_VAR is not defined), this is
# used to find a path on the NAS that would be suiteable as a value for
# DATA_ROOT_ENV_VAR (it's where I put my data on the NAS).
NAS_PATH_TO_HONG2P_DATA = 'mb_team'

# Sets optional faster-storage directory that is checked first (currently just in
# `raw_fly_dir`).
FAST_DATA_ROOT_ENV_VAR = 'HONG2P_FAST_DATA'

_fast_data_root = os.environ.get(FAST_DATA_ROOT_ENV_VAR)
if _fast_data_root is not None and not isdir(_fast_data_root):
    raise IOError(f'{FAST_DATA_ROOT_ENV_VAR} set but is not a directory')

np.set_printoptions(precision=2)

# TODO maybe move all of these to __init__.py, or at least expose them there?
# or maybe to a hong2p.py module (maybe importing all of its contents in
# __init__.py ?)
# TODO migrate all 'prep_date' -> 'date'? (seems i already use 'date' in a lot
# of places...)
recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
trial_only_cols = [
    'comparison',
    'name1',
    'name2',
    'repeat_num'
]
trial_cols = recording_cols + trial_only_cols

date_fmt_str = '%Y-%m-%d'
dff_latex = r'$\frac{\Delta F}{F}$'


# Module level cache.
_data_root = None
# TODO add _fast_data_root setting as kwarg here?
def set_data_root(new_data_root):
    """Sets data root, so future calls to `data_root` will return the input.

    You may either use this function or set one of the environment variables
    that the `data_root` function checks.
    """
    global _data_root

    if not isdir(new_data_root):
        raise IOError(f'{new_data_root} is not a directory!')

    if _data_root is not None:
        warnings.warn('data_root was already defined. usually set_data_root '
            'should only need to be called once.'
        )

    _data_root = new_data_root


def data_root(verbose=False):
    global _data_root

    if _data_root is None:
        # TODO print priority order of these env vars in any failure below
        # TODO TODO refactor (to loop, w/ break, probably) to also check if directories
        # exist before picking one to use?
        prefix = None

        if DATA_ROOT_ENV_VAR in os.environ:
            root = os.environ[DATA_ROOT_ENV_VAR]
            source = DATA_ROOT_ENV_VAR

            if verbose:
                print(f'found {DATA_ROOT_ENV_VAR}')

        elif NAS_PREFIX_ENV_VAR in os.environ:
            root = join(os.environ[NAS_PREFIX_ENV_VAR], NAS_PATH_TO_HONG2P_DATA)
            source = NAS_PREFIX_ENV_VAR

            if verbose:
                print(f'did not find {DATA_ROOT_ENV_VAR}')
                print(f'found {NAS_PREFIX_ENV_VAR}')

        else:
            raise IOError('either set one of the environment variables '
                f'({DATA_ROOT_ENV_VAR} or {NAS_PREFIX_ENV_VAR}) or call '
                'hong2p.util.set_data_root(<data root>) before this data_root call'
            )

        _data_root = root

        if not isdir(_data_root):
            raise IOError(f'data root expected at {_data_root}, but no directory exists'
                f' there!\nDirectory chosen from environment variable {source}'
            )

    # TODO err if nothing in data_root, saying which env var to set and how
    return _data_root


def check_dir_exists(fn_returning_dir):

    @functools.wraps(fn_returning_dir)
    def optionally_checked_fn_returning_dir(*args, check=True, **kwargs):

        directory = fn_returning_dir(*args, **kwargs)

        if check and not isdir(directory):
            raise IOError('directory {directory} does not exist!')

        return directory

    return optionally_checked_fn_returning_dir


# TODO (for both below) support a local and a remote one ([optional] local copy
# for faster repeat analysis)?
# TODO use env var like kc_analysis currently does for prefix after refactoring
# (include mb_team in that part and rename from data_root?)
@check_dir_exists
def raw_data_root(root=None, **kwargs):

    if root is None:
        root = data_root(**kwargs)

    return join(root, 'raw_data')


# TODO kwarg / default to makeing dir if not exist (and for similar fns above)?
@check_dir_exists
def analysis_intermediates_root():
    # TODO probably prefer using $HONG2P_DATA over os.getcwd() (assuming it's not on NAS
    # and it therefore acceptably fast if not instead using $HONG_NAS)
    if FAST_DATA_ROOT_ENV_VAR in os.environ:
        intermediates_root_parent = os.environ[FAST_DATA_ROOT_ENV_VAR]
    else:
        warnings.warn(f'environment variable {FAST_DATA_ROOT_ENV_VAR} not set, so '
            'storing analysis intermediates under current directory'
        )
        intermediates_root_parent = os.getcwd()

    intermediates_root = join(intermediates_root_parent, 'analysis_intermediates')
    return intermediates_root


@check_dir_exists
def stimfile_root(**kwargs):
    return join(data_root(**kwargs), 'stimulus_data_files')


# TODO replace this w/ above (need to change kc_natural_mixes / natural_odors, or at
# least pin an older version of hong2p for them)
@check_dir_exists
def analysis_output_root():
    return join(data_root(), 'analysis_output')


def format_date(date):
    """
    Takes a pandas Timestamp or something that can be used to construct one
    and returns a str with the formatted date.

    Used to name directories by date, etc.
    """
    return pd.Timestamp(date).strftime(date_fmt_str)


def format_timestamp(timestamp):
    # TODO example of when this should be used. maybe explicitly say use
    # `format_date` for dates
    """Returns human-readable str for timestamp accepted by `pd.Timestamp`.
    """
    return str(pd.Timestamp(timestamp))[:16]


# TODO maybe rename to [get_]fly_basedir?
def get_fly_dir(date, fly):
    """Returns str path fragment as YYYY-MM-DD/<n> for variety of input types
    """
    if not type(date) is str:
        date = format_date(date)

    if not type(fly) is str:
        fly = str(int(fly))

    return join(date, fly)


def raw_fly_dir(date, fly, warn=True, short=False):
    """
    Args:
        short (bool): (default=False) If True, returns in format
            YYYY-MM-DD/<fly #>/<ThorImage dir>, without the prefix specifying the full
            path. Intended for creating more readable paths, where absolute paths are
            not required.
    """
    raw_fly_basedir = get_fly_dir(date, fly)

    # TODO TODO maybe refactor for more granularity (might need to change a lot of usage
    # of data_root() and stuff that uses it though... perhaps also functions that
    # operate on directories like the fn to pair thor dirs)
    if _fast_data_root is not None:
        fast_raw_fly_dir = join(raw_data_root(root=_fast_data_root), raw_fly_basedir)
        # TODO warn if not using this despite env var being set
        if isdir(fast_raw_fly_dir):
            return fast_raw_fly_dir
        else:
            if warn:
                warnings.warn(f'{FAST_DATA_ROOT_ENV_VAR} set ({_fast_data_root}) but '
                    f'raw data directory for fly ({date}, {fly}) did not exist there'
                )

    return join(raw_data_root(), raw_fly_basedir)


def thorimage_dir(date, fly, thorimage_id, **kwargs):
    return join(raw_fly_dir(date, fly, **kwargs), thorimage_id)


def thorsync_dir(date, fly, base_thorsync_dir, **kwargs):
    return join(raw_fly_dir(date, fly, **kwargs), base_thorsync_dir)


# TODO use new name in al_pair_grids + also handle fast data dir here.
# (maybe always returning directories under fast? or kwarg to behave that way?)
def analysis_fly_dir(date, fly):
    return join(analysis_output_root(), get_fly_dir(date, fly))


def shorten_path(full_path, n_parts=3):
    """Returns a string containing just the last n_parts (default=3) of input path.

    For making IDs / easier-to-read paths, when the full path isn't required.
    """
    return '/'.join(full_path.split(sep)[-n_parts:])


# TODO maybe rename suffix here / thor.pair_thor_subdirs(->_dirs) for
# consistency. i think i was already thinking consolidating/renaming those two
# thor functions
# TODO maybe it would be better to have this return a data frame? maybe add
# another fn that converts output of this to that?
# TODO maybe also allow specification of optional third/additional keys to
# restrict to only some thorimage / thorsync dirs for a subset? or maybe it'd
# make more sense to add other functions for blacklisting/whitelisting stuff?
# TODO TODO function like this but that returns everything, with kwargs for only getting
# stuff between a start and end date (w/ end date not specified as well, for analyzing
# ongoing experiments)
def date_fly_list2paired_thor_dirs(date_fly_list, n_first=None, verbose=False,
    **pair_kwargs):
    # TODO add code example to doc
    """Takes list of (date, fly_num) tuples to pairs of their Thor outputs.

    Args:
        date_fly_list (list of (date-like, int)): (date, fly number) tuples

        n_first (None | int): If passed, only up to this many of pairs are enumerated.
            Intended for testing on subsets of data.

        verbose (bool): (default=False) If True, prints the fly/ThorImage/ThorSync
            directories as they are being iterated over.

        **pair_kwargs: Passed through to `thor.pair_thor_subdirs`. See arguments to
            `thor.pair_thor_dirs` (called by `thor.pair_thor_subdirs`) for most of the
            useful options.

    Each output is of the form:
    ((date, fly_num), (thorimage_dir<i>, thorsync_dir<i>))
    """
    if n_first is not None:
        warnings.warn(f'only returning first {n_first} paired Thor[Image/Sync] outputs')

    n = 0
    for date, fly_num in date_fly_list:
        fly_dir = raw_fly_dir(date, fly_num)

        # TODO if verbose and ignore is in pair_kwargs, maybe thread some other
        # arguments through such that we can have the inner function print just which
        # pairs it is ignoring? (or [opt to] return them from pair_thor_subdirs and then
        # print here?)

        paired_thor_dirs = thor.pair_thor_subdirs(fly_dir, **pair_kwargs)

        for image_dir, sync_dir in paired_thor_dirs:

            if n_first is not None and n >= n_first:
                return

            if verbose:
                print('thorimage_dir:', image_dir)
                print('thorsync_dir:', sync_dir)

            yield (date, fly_num), (image_dir, sync_dir)
            n += 1


# TODO TODO merge date_fly_list2paired_thor_dirs into this or just delete that and add
# kwarg here to replace above (too similar)
def paired_thor_dirs(start_date=None, end_date=None, n_first=None, skip_redone=True,
    verbose=False, print_skips=True, print_fast=True, print_full_paths=True,
    **pair_kwargs):
    # TODO add code example to doc
    """

    Args:
        n_first (None | int): If passed, only up to this many of pairs are enumerated.
            Intended for testing on subsets of data.

        verbose (bool): (default=False) If True, prints the fly/ThorImage/ThorSync
            directories as they are being iterated over.

        **pair_kwargs: Passed through to `thor.pair_thor_subdirs`. See arguments to
            `thor.pair_thor_dirs` (called by `thor.pair_thor_subdirs`) for most of the
            useful options.

    Each output is of the form:
    ((date, fly_num), (thorimage_dir<i>, thorsync_dir<i>))
    """
    if n_first is not None:
        warnings.warn(f'only returning first {n_first} paired Thor[Image/Sync] outputs')

    if start_date is not None:
        start_date = pd.Timestamp(start_date)

    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    n = 0

    def grandchildren(d):
        # Returns without the trailing '/' glob would normally add using this syntax.
        return [split(x)[0] for x in glob.glob(join(d, '*/*/'))]

    def date_fly_parts(d):
        rest, fly_part = split(d)
        _, date_part = split(rest)
        return date_part, fly_part

    if _fast_data_root is not None:
        candidate_grandchildren = grandchildren(raw_data_root(root=_fast_data_root))
        fast_parts = {date_fly_parts(d) for d in candidate_grandchildren}
    else:
        candidate_grandchildren = []
        # Set of tuples representing deepest-level ("leaf") directories under
        # `_fast_data_root`.
        fast_parts = {}

    gs = grandchildren(raw_data_root())
    for g in gs:
        date_fly = date_fly_parts(g)
        if date_fly in fast_parts:
            if verbose and print_fast:
                # TODO maybe just warn for any we aren't using fast for (assuming we are
                # using fast for *any*), and generally don't print
                print(f'not using {g} because had equivalent fast dir under '
                    f'{_fast_data_root}'
                )

            continue
        candidate_grandchildren.append(g)

    # Sorting on (date, fly) parts (to the extent that's what they are)
    for d in sorted(candidate_grandchildren, key=lambda g: g.split(sep)[-2:]):
        date_part, fly_part = date_fly_parts(d)

        try:
            fly_num = int(fly_part)
        except ValueError:
            if verbose and print_skips:
                print(f'skipping {d} because could not parse fly_num from {fly_part}')

            continue

        try:
            date = pd.Timestamp(datetime.strptime(date_part, date_fmt_str))
        except ValueError:
            if verbose and print_skips:
                print(f'skipping {d} because could not parse date from {date_part}')

            continue

        if start_date is not None and date < start_date:
            if verbose and print_skips:
                print(f'skipping {d} because earlier than {format_date(start_date)}')

            continue

        if end_date is not None and end_date < date:
            if verbose and print_skips:
                print(f'skipping {d} because later than {format_date(end_date)}')

            continue

        # TODO if verbose and ignore is in pair_kwargs, maybe thread some other
        # arguments through such that we can have the inner function print just which
        # pairs it is ignoring? (or [opt to] return them from pair_thor_subdirs and then
        # print here?)

        fly_dir = raw_fly_dir(date, fly_num)

        paired_dirs = thor.pair_thor_subdirs(fly_dir, **pair_kwargs)

        # TODO generalize to also match and similarly handle ThorImage-autogenerated
        # number suffixes that typically mean the same thing (though check the behavior
        # of diff ThorImage versions is supported)
        if skip_redone:
            redo_suffix = '_redo'
            prefixes_of_thorimage_redos = {
                ti[:-len(redo_suffix)] for ti, td in paired_dirs
                if ti.endswith(redo_suffix)
            }

        # TODO test that the mtime of the directories reliably gives me the order i want
        for image_dir, sync_dir in sorted(paired_dirs, key=lambda p: getmtime(p[0])):

            if skip_redone and image_dir in prefixes_of_thorimage_redos:
                if verbose and print_skips:
                    print(f'skipping {image_dir} because matching redo exists\n')

                continue

            if n_first is not None and n >= n_first:
                return

            if verbose:
                if print_full_paths:
                    image_dir_toprint = image_dir
                    sync_dir_toprint = sync_dir
                else:
                    image_dir_toprint = shorten_path(image_dir)
                    sync_dir_toprint = shorten_path(sync_dir)

                print('thorimage_dir:', image_dir_toprint)
                print('thorsync_dir:', sync_dir_toprint)

            yield (date, fly_num), (image_dir, sync_dir)
            n += 1


def _raw_data_root_grandchildren():
    return glob.glob(join(raw_data_root(), '*/*/'))


def _all_thorimage_dirs():
    """
    Returns list of all ThorImage directories two levels under data root (where
    they should be given my folder structure conventions).

    For testing functions on all of the data under the root.
    """
    # TODO fix so it's actually all children of these directories
    raise NotImplementedError
    return [d for d in _raw_data_root_grandchildren()
        if thor.is_thorimage_dir(d)
    ]


def _all_thorsync_dirs():
    """
    Returns list of all ThorSync directories two levels under data root (where
    they should be given my folder structure conventions).

    For testing functions on all of the data under the root.
    """
    # TODO fix so it's actually all children of these directories
    raise NotImplementedError
    return [d for d in _raw_data_root_grandchildren()
        if thor.is_thorsync_dir(d)
    ]


def _all_paired_thor_dirs(skip_errors=True, **kwargs):
    """
    Returns a list of all (ThorImage, ThorSync) directories that can be paired
    (i.e. determined to come from the same exerpiment) and that are both
    immediate children of (the same) one of the directories returned by
    `_raw_data_root_grandchildren()`.

    skip_errors (bool): (default=True) if False, will raise any caught
        `ValueError` rather than skipping results and continuing

    `kwargs` are passed to `thor.pair_thor_subdirs`
    """
    all_pairs = []
    for d in _raw_data_root_grandchildren():
        try:
            d_pairs = thor.pair_thor_subdirs(d, **kwargs)
        except ValueError:
            if skip_errors:
                continue
            else:
                raise

        all_pairs.extend(d_pairs)

    return all_pairs


def stimulus_yaml_from_thorimage(thorimage_dir_or_xml, stimfile_dir=None):
    """Returns absolute path to stimulus YAML file from note field in ThorImage XML.

    Args:
        thorimage_dir_or_xml: path to ThorImage output directory or XML Element
            containing parsed contents of the corresponding Experiment.xml file.

        stimfile_dir (str): (optional) directory containing stimulus .yaml files.
            If not passed, `stimfile_root()` is used.

    Raises:
        IOError if stimulus file directory does not exist
        ValueError if multiple or no substrings of note field end with .yaml

    XML should contain a manually-entered path relative to where the olfactometer code
    that generated it was run, but assuming it was copied to the appropriate location
    (directly under `stimfile_dir` if passed or `stimfile_root()` otherwise), this
    absolute path should exist.
    """
    if stimfile_dir is None:
        stimfile_dir = stimfile_root()

    elif not isdir(stimfile_dir):
        raise IOError(f'passed stimfile_dir={stimfile_dir} is not a directory!')

    notes = thor.get_thorimage_notes(thorimage_dir_or_xml)

    yaml_path = None
    parts = notes.split()
    for p in parts:
        p = p.strip()
        if p.endswith('.yaml'):
            if yaml_path is not None:
                raise ValueError('encountered multiple *.yaml substrings!')

            yaml_path = p

    if yaml_path is None:
        raise ValueError('no string ending in .yaml found in ThorImage note field')

    assert yaml_path is not None

    # TODO change data that has this to expand paths + delete this hack
    if '""' in yaml_path:
        date_str = '_'.join(yaml_path.split('_')[:2])
        old_yaml_path = yaml_path
        yaml_path = yaml_path.replace('""', date_str)

        warnings.warn(f'replacing of stimulus YAML path of {old_yaml_path} with '
            f'{yaml_path}'
        )
    #

    # Since paths copied/pasted within Windows may have '\' as a file
    # separator character.
    yaml_path = yaml_path.replace('\\', '/')

    if not exists(join(stimfile_dir, yaml_path)):
        prefix, ext = splitext(yaml_path)
        yaml_dir = '_'.join(prefix.split('_')[:3])
        subdir_path = join(stimfile_dir, yaml_dir, yaml_path)
        if exists(subdir_path):
            yaml_path = subdir_path

    yaml_path = join(stimfile_dir, yaml_path)
    assert exists(yaml_path), f'{yaml_path}'

    return yaml_path


def thorimage2yaml_info_and_odor_lists(thorimage_dir_or_xml, stimfile_dir=None):
    """Returns yaml_path, yaml_data, odor_lists

    Args:
        thorimage_dir_or_xml: path to ThorImage output directory or XML Element object
            parsed from corresponding Experiment.xml file

        stimfile_dir (str): (optional) directory containing stimulus .yaml files.
            If not passed, `stimfile_root()` is used.

    Returns:
        yaml_path (str): path to YAML

        yaml_data (dict): loaded contents of `yaml_path`

        odor_lists (list-of-lists-of-dicts): each list this contains is a representation
            of all the odors presented together on a given trial
    """
    yaml_path = stimulus_yaml_from_thorimage(thorimage_dir_or_xml,
        stimfile_dir=stimfile_dir
    )

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    odor_lists = olf.yaml_data2odor_lists(yaml_data)

    return yaml_path, yaml_data, odor_lists


def is_array_sorted(array):
    """Returns whether 1-dimensional np.ndarray is sorted."""
    # could implement an `axis` kwarg if i wanted to support multidimensional
    # arrays
    assert len(array.shape) == 1, 'only 1-dimensional arrays supported'
    # https://stackoverflow.com/questions/47004506
    return np.all(array[:-1] <= array[1:])


def print_block_frames(block_first_frames, block_last_frames):
    """Prints block numbers and the corresponding start / stop frames.

    For subsetting TIFF in ImageJ / other manual analysis.

    Prints frame numbers 1-indexed, but takes 0-indexed frames.
    """
    print('Block frames:')
    for i, (b_first, b_last) in enumerate(zip(block_first_frames,
        block_last_frames)):
        # Adding one to index frames as in ImageJ.
        print('{}: {} - {}'.format(i, b_first + 1, b_last + 1))
    print('')


def md5(fname):
    """Calculates MD5 hash on file with name `fname`.
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# TODO move to project specific repo
def odorset_name(df_or_odornames):
    """Returns name for set of odors in DataFrame.

    Looks at odors in original_name1 column. Name used to lookup desired
    plotting order for the odors in the set.
    """
    try:
        if 'original_name1' in df_or_odornames.columns:
            unique_odornames = df_or_odornames.original_name1.unique()
            abbreviated = False
        else:
            assert 'name1' in df_or_odornames.columns, \
                'need either original_name1 or name1 in df columns'
            # Assuming abbreviated names now.
            unique_odornames = df_or_odornames.name1.unique()
            # maybe don't assume abbreviated just b/c name1?
            # (particularly if supporting abbrev/not in iterable input
            # case, could also check name1 contents)
            abbreviated = True

    except AttributeError:
        unique_odornames = set(df_or_odornames)
        # TODO maybe also support abbreviated names in this case?
        abbreviated = False

    odor_set = None
    # TODO TODO derive these diagnostic odors from odor_set2order? would that
    # still be redundant w/ something else i hardcoded (if so, further
    # de-dupe)?
    # TODO at least lookup abbreviations from full names?
    if not abbreviated:
        if 'ethyl butyrate' in unique_odornames:
            odor_set = 'kiwi'
        elif 'acetoin' in unique_odornames:
            odor_set = 'flyfood'
        elif '1-octen-3-ol' in unique_odornames:
            odor_set = 'control'
    else:
        if 'eb' in unique_odornames:
            odor_set = 'kiwi'
        elif 'atoin' in unique_odornames:
            odor_set = 'flyfood'
        elif '1o3ol' in unique_odornames:
            odor_set = 'control'

    if odor_set is None:
        raise ValueError('none of diagnostic odors in odor column')

    # TODO probably just find single odor that satisfies is_mix and derive from
    # that, for more generality (would only work in original name case)
    return odor_set


def load_stimfile(stimfile_path):
    """Loads odor metadata stored in a pickle.

    These metadata files are generated by scripts under
    `ejhonglab/cutpast_arduino_stimuli`.
    """
    # TODO better check for path already containing stimfile_root?
    # string prefix check might be too fragile...
    if not stimfile_path.startswith(stimfile_root()):
        stimfile_path = join(stimfile_root(), stimfile_path)

    # TODO also err if not readable / valid
    if not exists(stimfile_path):
        stimfile_just_fname = split(stimfile_path)[1]
        raise ValueError('copy missing stimfile {} to {}'.format(
            stimfile_just_fname, stimfile_root
        ))

    with open(stimfile_path, 'rb') as f:
        data = pickle.load(f)
    return data


# TODO either generalize / allow passing additional functions / keys or move to
# project specific repo
# TODO delete (subsuming contents into load_experiment) if i'd never want to
# call this in other circumstances
# TODO can this be generalized nicely to load YAML files output by my newer
# olfactometer code?
def load_odor_metadata(stimfile_path):
    """Returns odor metadata loaded from pickle and additional computed values.

    Additional values are added into the dictionary loaded from the pickle.
    In some cases, this can overwrite the loaded values.
    """
    data = load_stimfile(stimfile_path)
    # TODO infer from data if no stimfile and not specified in
    # metadata (is there actually any value in this? maybe if we have
    # a sufficient amount of other data about the odor order in metadata
    # yaml (though this is not supported now...)?)

    # TODO delete this hack (which is currently just using new pickle
    # format as a proxy for the experiment being a supermixture experiment)
    if 'odor_lists' not in data:
        pair_case = True

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors odor-pair experiments.
        presentations_per_repeat = 3
        odor_list = data['odor_pair_list']

        # could delete eventually. b/c originally i was casting to int,
        # though it's likely it was always int anyway...
        assert type(data['n_repeats']) is int
    else:
        pair_case = False

        n_expected_real_blocks = 3
        odor_list = data['odor_lists']
        # because of "block" def in arduino / get_stiminfo code
        # not matching def in randomizer / stimfile code
        # (scopePin pulses vs. randomization units, depending on settings)
        presentations_per_repeat = len(odor_list) // n_expected_real_blocks
        assert len(odor_list) % n_expected_real_blocks == 0

        # Hardcode to break up into more blocks, to align defs of blocks.
        # TODO (maybe just for experiments on 2019-07-25 ?) or change block
        # handling in here? make more flexible?

        # Will overwrite existing value.
        assert 'n_repeats' in data.keys()
        data['n_repeats'] = 1

        # TODO check that overwriting of presentations_per_block with newest
        # data in this case is still accurate (post fixing some of the values
        # in pickle data) (also check similar values)

    presentations_per_block = data['n_repeats'] * presentations_per_repeat

    # Overwriting exisiting value.
    assert 'presentations_per_block' in data.keys()
    data['presentations_per_block'] = presentations_per_block

    # NOT overwriting an existing value.
    assert 'pair_case' not in data.keys()
    data['pair_case'] = pair_case

    # NOT overwriting existing value.
    assert 'presentations_per_repeat' not in data.keys()
    data['presentations_per_repeat'] = presentations_per_repeat

    # NOT overwriting an existing value.
    assert 'odor_list' not in data.keys()
    data['odor_list'] = odor_list

    return data


# TODO maybe break into a kc_mix_analysis repo (or something else appropriately
# project specific)
def print_trial_odors(data, odor_onset_frames=None):
    """
    data should be as the output of `load_odor_metadata`
    """
    import chemutils as cu

    n_repeats = data['n_repeats']
    odor_list = data['odor_list']
    presentations_per_block = data['presentations_per_block']
    pair_case = data['pair_case']

    n_blocks, remainder = divmod(len(odor_list), presentations_per_block)
    assert remainder == 0

    # TODO add extra input data / add extra values to `data` in fn that already
    # augments that as necessary, for all values used below to still be defined
    if pair_case:
        print(('{} comparisons ({{A, B, A+B}} in random order x ' +
            '{} repeats)').format(n_blocks, n_repeats)
        )
    else:
        mix_names = {x for x in odor_list if '@' not in x}
        if len(mix_names) == 1:
            mix_name = mix_names.pop()
            print('{} randomized blocks of "{}" and its components'.format(
                n_blocks, mix_name
            ))
        else:
            assert len(mix_names) == 0
            print('No mixtures, so presumably this is a calibration experiment')
            print(f'{n_blocks} randomized blocks')

    # TODO maybe print this in tabular form?
    trial = 0
    for i in range(n_blocks):
        p_start = presentations_per_block * i
        p_end = presentations_per_block * (i + 1)
        cline = '{}: '.format(i)

        odor_strings = []
        for o in odor_list[p_start:p_end]:
            # TODO maybe always have odor_list hold str repr?
            # or unify str repr generation -> don't handle use odor_lists
            # for str representation in supermixture case?
            # would also be a good time to unify name + *concentration*
            # handling
            if pair_case:
                # TODO odor2abbrev here too probably... be more uniform
                if o[1] == 'paraffin':
                    odor_string = o[0]
                else:
                    odor_string = ' + '.join(o)
            else:
                assert type(o) is str

                parts = o.split('@')
                odor_name = parts[0].strip()
                abbrev = None
                try:
                    abbrev = cu.odor2abbrev(odor_name)
                # For a chemutils conversion failure.
                except ValueError:
                    pass

                if abbrev is None:
                    abbrev = odor_name

                odor_string = abbrev
                # TODO also don't append stuff if conc is @ 0.0 (log)
                if len(parts) > 1:
                    assert len(parts) == 2
                    odor_string += ' @' + parts[1]

            # Adding one to index frames as in ImageJ.
            if odor_onset_frames is not None:
                odor_string += ' ({})'.format(odor_onset_frames[trial] + 1)

            trial += 1
            odor_strings.append(odor_string)

        print(cline + ', '.join(odor_strings))


def tiff_ijroi_filename(tiff, confirm=None,
    gui_confirm=False, gui_fallback=False, gui='qt5'):
    """
    Takes a tiff path to corresponding ImageJ ROI file, assuming a certain
    naming convention and folder structure.

    Automatic search for ROI will only check same folder as the TIFF passed in.

    Options for fallback to manual selection / confirmation of appropriate
    ROI file.
    """
    if gui_confirm and confirm:
        raise ValueError('only specify either gui_confirm or confirm')

    if gui_confirm or gui_fallback:
        if gui == 'qt5':
            from PyQt5.QtWidgets import QMessageBox, QFileDialog

        # TODO maybe implement some version w/ a builtin python gui like
        # tkinter?
        else:
            raise NotImplementedError(f'gui {gui} not supported. see function.')
    else:
        if confirm is None:
            confirm = True

    curr_tiff_dir = split(tiff)[0]
    # TODO check that *.zip glob still matches in case where it is the empty
    # string (fix if it doesn't)
    thorimage_id = tiff_thorimage_id(tiff)
    possible_ijroi_files = glob.glob(join(curr_tiff_dir,
        thorimage_id + '*.zip'
    ))

    ijroiset_filename = None
    # TODO fix automatic first choice in _NNN naming convention case
    # (seemed to not work on 2019-07-25/2/_008)
    # but actually it did work in */_007 case... so idk what's happening
    if len(possible_ijroi_files) == 1:
        ijroiset_filename = possible_ijroi_files[0]

        if confirm:
            # TODO factor into a fn to always get a Yy/Nn answer?
            prompt = (format_keys(*tiff_filename2keys(tiff) +
                f': use ImageJ ROIs in {ijroiset_filename}? [y/n] '
            ))
            response = None
            while response not in ('y', 'n'):
                response = input(prompt).lower()

            if response == 'n':
                ijroiset_filename = None
                # TODO manual text entry of appropriate filename in this case?
                # ...or maybe just totally unsupport terminal interaction?

        elif gui_confirm:
            confirmation_choice = QMessageBox.question(self, 'Confirm ROI file',
                f'Use ImageJ ROIs in {ijroiset_filename}?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if confirmation_choice != QMessageBox.Yes:
                ijroiset_filename = None

    elif not gui_fallback and len(possible_ijroi_files) > 1:
        raise IOError('too many candidate ImageJ ROI files')

    elif not gui_fallback and len(possible_ijroi_files) == 0:
        raise IOError('no candidate ImageJ ROI files')

    if gui_fallback and ijroiset_filename is None:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # TODO restrict *.zip files shown to those also following some
        # naming convention (to indicate it's for the currently loaded TIFF,
        # and not a TIFF from some other experiment on the same fly)?
        # maybe checkbox / diff option to show all?

        # TODO need to pass in parent widget (what `self` was) or leave null if
        # that works? / define in here?
        raise NotImplementedError('see comment above')
        ijroiset_filename, _ = QFileDialog.getOpenFileName(self,
            'Select ImageJ ROI zip...', curr_tiff_dir,
            'ImageJ ROIs (*.zip)', options=options
        )
        # TODO (opt?) to not allow no selection? (so downstream code can assume
        # it's defined...) (should also probably change confirmation behavior
        # above)
        if len(ijroiset_filename) == 0:
            print('No ImageJ ROI zipfile selected.')
            ijroiset_filename = None

    return ijroiset_filename


# TODO TODO TODO may want to change how this fn operates (so it operates on
# blocks rather than all of them concatenated + to provide different baselining
# options)
def calculate_df_over_f(raw_f, trial_start_frames, odor_onset_frames,
    trial_stop_frames):
    # TODO TODO maybe factor this into some kind of util fn that applies
    # another fn (perhaps inplace, perhaps onto new array) to each
    # (cell, block) (or maybe just each block, if smooth_1d can be vectorized,
    # so it could also apply in frame-shape-preserved case?)
    '''
    for b_start, b_end in zip(block_first_frames, block_last_frames):

        for c in range(n_footprints):
            # TODO TODO TODO TODO need to be (b_end + 1) since not
            # inclusive? (<-fixed) other problems like this elsewhere?????
            # TODO maybe smooth less now that df/f is being calculated more
            # sensibly...
            raw_f[b_start:(b_end + 1), c] = smooth_1d(
                raw_f[b_start:(b_end + 1), c], window_len=11
            )
    '''
    df_over_f = np.empty_like(raw_f) * np.nan
    for t_start, odor_onset, t_end in zip(trial_start_frames, odor_onset_frames,
        trial_stop_frames):

        # TODO TODO maybe use diff way of calculating baseline
        # (include stuff at end of response? just some percentile over a big
        # window or something?)
        # TODO kwargs to control method of calculating baseline

        # TODO maybe display baseline period on plots for debugging?
        # maybe frame numbers got shifted?
        baselines = np.mean(raw_f[t_start:(odor_onset + 1), :], axis=0)
        trial_f = raw_f[t_start:(t_end + 1), :]
        df_over_f[t_start:(t_end + 1), :] = (trial_f - baselines) / baselines

    # TODO some check that no value in df_over_f are currently NaN?

    return df_over_f


# TODO move to project specific repo
def stimfile_odorset(stimfile_path, strict=True):
    data = load_stimfile(stimfile_path)

    # TODO did i use some other indicator elsewhere? anything more robust than
    # this?
    if 'odor_pair_list' in data.keys():
        # Just because I don't believe the pair experiment analysis
        # made any use of something like an odorset, so trying to
        # get one from those stimfiles is likely a mistake.
        if strict:
            # TODO maybe this err should happen whether or not strict is
            # true...?
            raise ValueError('trying to get complex mixture odor set '
                'from old pair experiment stimfile'
            )
        else:
            return None

    odors = set(data['odor_lists'])
    # TODO TODO TODO what caused this error where some fields were empty?
    # this corrupt or test data / not intended to be saved?
    if len(odors) == 0:
        if strict:
            raise ValueError(f'empty odor lists in {stimfile_path}')
        else:
            return None
    assert type(data['odor_lists'][0]) is str

    # TODO TODO TODO fix how stimfile generation stuff doesn't save
    # hardcoded real stuff into odors (+ maybe other vars?)
    '''
    print(len(odors))
    pprint(odors)
    print(len(set(data['odors2pins'].keys())))
    pprint(set(data['odors2pins'].keys()))
    print(len(set(data['pins2odors'].values())))
    pprint(set(data['pins2odors'].values()))
    '''
    #assert odors == set(data['odors'])
    assert odors == set(data['odors2pins'].keys())
    assert odors == set(data['pins2odors'].values())

    # Not this accessor syntax, because .name is a property of all pandas
    # objects.
    odor_names = [split_odor_w_conc(oc)['name'] for oc in odors]

    return odorset_name(odor_names)


def print_all_stimfile_odorsets() -> None:
    stimfiles = sorted(glob.glob(join(stimfile_root(), '*.p')))
    stimfile_odorsets = [stimfile_odorset(sf, strict=False)
        for sf in stimfiles
    ]
    # TODO maybe print grouped by day
    pprint([(split(f)[1], s)
        for f, s in zip(stimfiles, stimfile_odorsets) if s
    ])


# TODO TODO move all of this kc mix stuff to project specific repo
solvents = ('pfo', 'water')
natural = ('kiwi', 'fly food')
# TODO maybe load (on demand) + cache the abbreviated versions of these, if
# chemutils is available?
odor_set2order = {
    'kiwi': [
        'pfo',
        'ethyl butyrate',
        'ethyl acetate',
        'isoamyl acetate',
        'isoamyl alcohol',
        'ethanol',
        # TODO check that changing the order of these last two hasn't broken
        # stuff...
        'kiwi approx.',
        'd3 kiwi'
    ],
    'control': [
        'pfo',
        '1-octen-3-ol',
        'furfural',
        'valeric acid',
        'methyl salicylate',
        '2-heptanone',
        # Only one of these will actually be present, they just take the same
        # place in the order.
        'control mix 1',
        'control mix 2'
    ],
    'flyfood': [
        'water',
        'propanoic acid',
        'isobutyric acid',
        'acetic acid',
        'acetoin',
        'ethanol',
        'fly food approx.',
        'fly food b'
    ]
}
def df_to_odor_order(df, observed=True, return_name1=False):
    """Takes a complex-mixture DataFrame to odor names in desired plot order.

    Args:
    df (pd.DataFrame): should have a 'original_name1' column, with names of
        odors from complex mixture experiments we have pre-defined odor orders
        for.

    observed (bool): (optional, default=True) If True, only return odor names
        in `df`.

    return_name1 (bool): (optional, default=False) If True, corresponding
        values in 'name1' will be returned for each value in 'original_name1'.
    """
    # TODO might need to use name1 if original_name1 not there...
    # (for gui case)
    odor_set = odorset_name(df)
    order = odor_set2order[odor_set]
    observed_odors = df.original_name1.unique()
    if observed:
        order = [o for o in order if o in observed_odors]
    else:
        # TODO maybe just handle this externally (force all data w/in some
        # analysis to only have one or the other control mix) and then delete
        # this special casing
        cm1 = 'control mix 1'
        cm2 = 'control mix 2'
        have_cm1 = cm1 in observed_odors
        have_cm2 = cm2 in observed_odors
        order = [o for o in order if o not in (cm1, cm2)]
        if have_cm1:
            assert not have_cm2, 'df should only have either cm1 or cm2'
            order.append(cm1)

        elif have_cm2:
            order.append(cm2)

    if return_name1:
        o2n = df[['original_name1','name1']].drop_duplicates(
            ).set_index('original_name1').name1
        order = list(o2n[order])

    return order


# TODO maybe also allow using GID from file?
def gsheet_csv_export_link(file_with_edit_link, gid=0, no_append_gid=False):
    """
    Takes a gsheet link copied from browser while editing it, and returns a
    URL suitable for reading it as a CSV into a DataFrame.

    GID seems to default to 0 for the first sheet, but seems unpredictable for further
    sheets in the same document, though you can extract it from the URL in those cases.
    """
    # TODO make expectations on URL consistent whether from file or not
    if file_with_edit_link.startswith('http'):
        base_url = file_with_edit_link
    else:
        pkg_data_dir = split(split(__file__)[0])[0]

        dirs_to_try = (os.getcwd(), pkg_data_dir)
        # .txt file containing link
        link_filename = None
        for d in dirs_to_try:
            fname = join(d, file_with_edit_link)
            if exists(fname):
                link_filename = fname
                break

        if link_filename is None:
            raise IOError(f'{file_with_edit_link} not found in any of {dirs_to_try}')

        with open(link_filename, 'r') as f:
            base_url = f.readline().split('/edit')[0]

    gsheet_link = base_url + '/export?format=csv&gid='

    if not no_append_gid:
        gsheet_link += str(gid)

    return gsheet_link


def gsheet_to_frame(file_with_edit_link, *, gid=0, bool_fillna_false=True,
    convert_date_col=True, drop_trailing_bools=True, restore_ints=True,
    normalize_col_names=False):
    """
    Args:
        bool_fillna_false (bool): whether to replace missing values in columns that
            otherwise only contain True/False with False. will convert column dtype to
            'bool' as well.

        convert_date_col (bool): whether to convert the contents of any columns named
            'date' (case insensitive) to `pd.Timestamp`

        drop_trailing_bools (bool): whether to drop blocks of False in bool columns
            beyond the last row where all non-bool columns have any non-NaN values.

            If a column has data validation for a boolean, the frame will have values
            (False as I've seen it so far) through to the end of the validation range,
            despite the fact that no data has been entered.

        restore_ints (bool): whether to convert columns parsed as floats (because
            missing data in rows where only default values for bool cols are present)
            to an integer type. Requires that drop_trailing_bools actually gets rid of
            all the NaN values in the columns to be converted to ints (float columns
            with only whole number / NaN values).

        normalize_col_names (bool): (default=False) whether to rename columns using the
            `hong2p.util.to_filename` (with `period=False` to that function) as well as
            lowercasing.
    """

    gsheet_link = gsheet_csv_export_link(file_with_edit_link, gid=gid)

    df = pd.read_csv(gsheet_link)

    bool_col_unique_vals = {True, False, np.nan}
    bool_cols = [c for c in df.columns if df[c].dtype == 'bool' or
        (df[c].dtype == 'object' and set(df[c].unique()) == bool_col_unique_vals)
    ]

    if bool_fillna_false:
        for c in bool_cols:
            df[c] = df[c].fillna(False).astype('bool')

    # Could consider replacing this w/ just parse_dates [+ infer_datetime_format] kwargs
    # to pd.read_csv
    if convert_date_col:
        date_cols = [c for c in df.columns if c.lower() == 'date']
        for c in date_cols:
           df[c] = pd.to_datetime(df[c])

    if drop_trailing_bools:
        nonbool_cols = [c for c in df.columns if c not in bool_cols]
        nonbool_cols_some_data = ~ df[nonbool_cols].isna().all(axis='columns')

        last_row_with_data_idx = nonbool_cols_some_data.where(nonbool_cols_some_data
            ).last_valid_index()

        will_be_dropped = df.iloc[(last_row_with_data_idx + 1):]

        # We expect all bool_cols beyond last data in non-bool cols to be False
        # (default value as I currently have the data validation for those columns in
        # most / all Gsheets where I use them)
        assert not will_be_dropped[bool_cols].any(axis=None)

        df = df.iloc[:last_row_with_data_idx].copy()

    if restore_ints:
        # (works for 'float64' at least, presumably all float types)
        float_cols = [c for c in df.columns if df[c].dtype == 'float']

        for c in float_cols:
            col = df[c]

            # If dropping trailing NaN values didn't get rid of all the NaN, we can't
            # change the dtype of the column to a numpy integer type.
            if col.isna().any():
                continue

            mod1 = np.mod(col, 1)

            # TODO actually a risk of floats not exactly having mod 1 of 0 if input is
            # indeed an integer for all of them? assuming no for now.
            if (mod1 == 0).all():
                df[c] = col.astype('int')

    if normalize_col_names:
        df.rename(columns=lambda x: to_filename(x, period=False).lower(), inplace=True)

    return df


# TODO TODO for this and other stuff that depends on network access (if not
# cached), fallback to cache (unless explicitly prevented?), and warn
# that we are doing so (unless cached version explicitly requested)
_mb_team_gsheet = None
def mb_team_gsheet(use_cache=False, natural_odors_only=False,
    drop_nonexistant_dirs=True, show_inferred_paths=False,
    print_excluded_on_disk=True, verbose=False):
    '''Returns a pandas.DataFrame with data on flies and MB team recordings.
    '''
    global _mb_team_gsheet
    if _mb_team_gsheet is not None:
        return _mb_team_gsheet

    gsheet_cache_file = '.gsheet_cache.p'
    if use_cache and exists(gsheet_cache_file):
        print(f'Loading MB team sheet data from cache at {gsheet_cache_file}')

        with open(gsheet_cache_file, 'rb') as f:
            sheets = pickle.load(f)

    else:
        # TODO TODO maybe env var pointing to this? or w/ link itself?
        # TODO maybe just get relative path from __file__ w/ /.. or something?
        # TODO TODO TODO give this an [add_]default_gid=True (set to False here)
        # so other code of mine can use this function
        gsheet_link = gsheet_csv_export_link('mb_team_sheet_link.txt',
            no_append_gid=True
        )

        # If you want to add more sheets, when you select the new sheet in your
        # browser, the GID will be at the end of the URL in the address bar.
        sheet_gids = {
            'fly_preps': '269082112',
            'recordings': '0',
            'daily_settings': '229338960'
        }

        sheets = dict()
        for df_name, gid in sheet_gids.items():
            df = pd.read_csv(gsheet_link + gid)

            # TODO convert any other dtypes?
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            df.drop(columns=[c for c in df.columns
                if c.startswith('Unnamed: ')], inplace=True)

            if 'fly_num' in df.columns:
                last_with_fly_num = df.fly_num.notnull()[::-1].idxmax()
                df.drop(df.iloc[(last_with_fly_num + 1):].index, inplace=True)

            sheets[df_name] = df

        boolean_columns = {
            'attempt_analysis',
            'raw_data_discarded',
            'raw_data_lost'
        }
        na_cols = list(set(sheets['recordings'].columns) - boolean_columns)
        sheets['recordings'].dropna(how='all', subset=na_cols, inplace=True)

        with open(gsheet_cache_file, 'wb') as f:
            pickle.dump(sheets, f)

    # TODO maybe make df some merge of the three sheets?
    df = sheets['recordings']

    # TODO TODO maybe flag to disable path inference / rethink how it should
    # interact w/ timestamp based correspondence between thorsync/image and
    # mapping that to the recordings in the gsheet
    # TODO should inference that reads the metadata happen in this fn?
    # maybe yes, but still factor it out and just call here?

    # TODO maybe start by not filling in fully-empty groups / flagging
    # them for later -> preferring to infer those from local files ->
    # then inferring fully-empty groups from default numbering as before

    # TODO try to replace w/ central key definition (module level)
    keys = ['date', 'fly_num']
    # These should happen before rows start being dropped, because the dropped
    # rows might have the information needed to ffill.
    # This should NOT ffill a fly_past a change in date.

    # Assuming that if date changes, even if fly_nums keep going up, that was
    # intentional.
    df.date = df.date.fillna(method='ffill')
    df.fly_num = df.groupby('date')['fly_num'].apply(
        lambda x: x.ffill().bfill()
    )
    # This will only apply to groups (dates) where there are ONLY missing
    # fly_nums, given filling logic above.
    df.dropna(subset=['fly_num'], inplace=True)
    assert not df.date.isnull().any()

    df['stimulus_data_file'] = df['stimulus_data_file'].fillna(method='ffill')

    df.raw_data_discarded = df.raw_data_discarded.fillna(False)
    # TODO say when this happens?
    df.drop(df[df.raw_data_discarded].index, inplace=True)

    # TODO TODO warn if 'attempt_analysis' and either discard / lost is checked

    # Not sure where there were any NaN here anyway...
    df.raw_data_lost = df.raw_data_lost.fillna(False)
    df.drop(df[df.raw_data_lost].index, inplace=True)

    # TODO as per note below, any thorimage/thorsync dirs entered in spreadsheet
    # should probably cause warning/err if either of above rejection reason
    # is checked

    # This happens after data is dropped for the above two reasons, because
    # generally those mistakes do not consume any of our sequential filenames.
    # They should not have files associated with them, and the Google sheet
    # information on them is just for tracking problems / efficiency.
    df['recording_num'] = df.groupby(keys).cumcount() + 1

    if show_inferred_paths:
        missing_thorimage = pd.isnull(df.thorimage_dir)
        missing_thorsync = pd.isnull(df.thorsync_dir)

    my_project = 'natural_odors'

    # TODO TODO fix current behavior where key groups that have nothing filled
    # in on gsheet (for dirs) will default to old format.  misbehaving for
    # 8-27/1 and all of 11-21, for example. (fixed?)

    check_and_set = []
    for gn, gdf in df.groupby(keys):
        if not (gdf.project == my_project).any():
            continue

        if gdf[['thorimage_dir','thorsync_dir']].isnull().all(axis=None):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            if verbose:
                print('\n' + fly_dir)

            try:
                # Since we are disabling check that ThorImage nums (from naming
                # convention) are unique, we must check this before
                # mb_team_gsheet returns.
                image_and_sync_pairs = thor.pair_thor_subdirs(fly_dir,
                     check_against_naming_conv=True,
                     check_unique_thorimage_nums=False
                )
                if verbose:
                    print('pairs:')
                    pprint(image_and_sync_pairs)

            # TODO TODO should ValueError actually be caught?
            # (from comments in other fns) it seems it will only be raised
            # when they are not 1:1, which should maybe cause failure in the way
            # AssertionErrors do now...
            except ValueError as e:
                gn_str = format_keys(*gn)
                print(f'For {gn_str}:')
                print('could not pair thor dirs automatically!')
                print(f'({e})\n')
                continue

            # could maybe try to sort things into "prep checking" / real
            # experiment based on time length or something (and maybe try
            # to fall back to just pairing w/ real experiments? and extending
            # condition below to # real experiments in gdf)
            n_matched = len(image_and_sync_pairs)
            ng = len(gdf)
            # TODO should this be an error (was previously just a
            # print + continue)?
            if n_matched < ng:
                msg = ('more rows for (date, fly) pair than matched outputs'
                    f' ({n_matched} < {ng})'
                )
                raise AssertionError(msg)
                #print(msg)
                #continue

            all_group_in_old_dir_fmt = True
            group_tids = []
            group_tsds = []
            for tid, tsd in image_and_sync_pairs:
                tid = split(tid)[-1]
                if pd.isnull(thor.old_fmt_thorimage_num(tid)):
                    if verbose:
                        print(f'{tid} not in old format')
                    all_group_in_old_dir_fmt = False

                group_tids.append(tid)
                group_tsds.append(split(tsd)[-1])

            # Not immediately setting df in this case, so that I can check
            # these results against the old way of doing things.
            if all_group_in_old_dir_fmt:
                if verbose:
                    print('all in old dir format')

                check_and_set.append((gn, gdf.index, group_tids, group_tsds))
            else:
                if verbose:
                    print('filling in b/c not (all) in old dir format')

                # TODO is it ok to modify df used to create groupby while
                # iterating over groupby?
                df.loc[gdf.index, 'thorimage_dir'] = group_tids
                df.loc[gdf.index, 'thorsync_dir'] = group_tsds

    if print_excluded_on_disk:
        # So that we can exclude these directories when printing stuff on disk
        # (but not in df) later, to reduce noise (because of course other
        # project stuff is dropped and will not be mentioned in df).
        ti_from_other_projects = set()
        ts_from_other_projects = set()
        for gn, gdf in df[df.project != my_project].groupby(keys):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            gdf_ti = set(gdf.thorimage_dir.dropna().apply(
                lambda d: join(fly_dir, d))
            )
            gdf_ts = set(gdf.thorsync_dir.dropna().apply(
                lambda d: join(fly_dir, d))
            )

            ti_from_other_projects |= gdf_ti
            ts_from_other_projects |= gdf_ts

    df.drop(df[df.project != my_project].index, inplace=True)

    # TODO TODO implement option to (at least) also keep prep checking that
    # preceded natural_odors (or maybe just that was on the same day)
    # (so that i can get all that ethyl acetate data for use as a reference
    # odor)

    # TODO display stuff inferred from files separately from stuff inferred
    # from combination of gsheet info and convention

    df['thorimage_num'] = df.thorimage_dir.apply(thor.old_fmt_thorimage_num)
    # TODO TODO should definition of consistency be changed to just check that
    # the ranking of the two are the same?
    # (maybe just if the group is all new format (which will have first real
    # experiments start w/ thorimage_num zero more often, b/c fn / fn_0000
    # thing)
    df['numbering_consistent'] = \
        pd.isnull(df.thorimage_num) | (df.thorimage_num == df.recording_num)

    # TODO unit test this
    # TODO TODO check that, if there are mismatches here, that they *never*
    # happen when recording num will be used for inference in rows in the group
    # *after* the mismatch (?)
    gkeys = keys + [
        'thorimage_dir',
        'thorsync_dir',
        'thorimage_num',
        'recording_num',
        'numbering_consistent'
    ]
    for name, group_df in df.groupby(keys):
        # TODO maybe refactor above so case 3 collapses into case 1?
        '''
        Case 1: all consistent
        Case 2: not all consistent, but all thorimage_dir filled in
        Case 3: not all consistent, but just because thorimage_dir was null
        '''
        #print(group_df[gkeys])

        # TODO check that first_mismatch based approach includes this case
        #if pd.notnull(group_df.thorimage_dir).all():
        #    continue

        mismatches = np.argwhere(~ group_df.numbering_consistent)
        if len(mismatches) == 0:
            continue

        first_mismatch_idx = mismatches[0][0]
        #print('first_mismatch:\n', group_df[gkeys].iloc[first_mismatch_idx])

        # TODO test case where the first mismatch is last
        following_thorimage_dirs = \
            group_df.thorimage_dir.iloc[first_mismatch_idx:]
        #print('checking these are not null:\n', following_thorimage_dirs)
        assert pd.notnull(following_thorimage_dirs).all()

    df.thorsync_dir.fillna(df.thorimage_num.apply(lambda x:
        np.nan if pd.isnull(x) else 'SyncData{:03d}'.format(int(x))),
        inplace=True
    )

    # Leaving recording_num because it might be prettier to use that for
    # IDs in figure than whatever Thor output directory naming convention.
    df.drop(columns=['thorimage_num','numbering_consistent'], inplace=True)

    # TODO TODO check for conditions in which we might need to renumber
    # recording num? (dupes / any entered numbers along the way that are
    # inconsistent w/ recording_num results)
    # TODO update to handle case where thorimage dir does not start w/
    # _ and is not just 3 digits after that?
    # (see what format other stuff from day is?)
    df.thorimage_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else '_{:03d}'.format(int(x))), inplace=True
    )
    df.thorsync_dir.fillna(df.recording_num.apply(lambda x:
        np.nan if pd.isnull(x) else 'SyncData{:03d}'.format(int(x))),
        inplace=True
    )

    for gn, gidx, gtids, gtsds in check_and_set:
        # Since some stuff may have been dropped (prep checking stuff, etc).
        still_in_idx = gidx.isin(df.index)
        # No group w/ files on NAS should have been dropped completely.
        assert still_in_idx.sum() > 0, f'group {gn} dropped completely'

        gidx = gidx[still_in_idx]
        gtids = np.array(gtids)[still_in_idx]
        gtsds = np.array(gtsds)[still_in_idx]

        from_gsheet = df.loc[gidx, ['thorimage_dir', 'thorsync_dir']]
        from_thor = [gtids, gtsds]
        consistent = (from_gsheet == from_thor).all(axis=None)
        if not consistent:
            print('Inconsistency between path infererence methods!')
            print(dict(zip(keys, gn)))
            print('Derived from Google sheet:')
            print(from_gsheet.T.to_string(header=False))
            print('From matching Thor output files:')
            print(pd.DataFrame(dict(zip(from_gsheet.columns, from_thor))
                ).T.to_string(header=False))
            print('')
            raise AssertionError('inconsistent rankings w/ old format')

    assert df.fly_num.notnull().all()
    df = df.astype({'fly_num': np.int64})

    cols = keys + ['thorimage_dir', 'thorsync_dir', 'attempt_analysis']
    # TODO flag to do this only for stuff marked attempt_analysis
    if show_inferred_paths:
        # TODO only do this if any actually *were* inferred
        print('Inferred ThorImage directories:')
        print(df.loc[missing_thorimage, cols].to_string(index=False))
        print('\nInferred ThorSync directories:')
        print(df.loc[missing_thorsync, cols].to_string(index=False))
        print('')

    duped_thorimage = df.duplicated(subset=keys + ['thorimage_dir'], keep=False)
    duped_thorsync = df.duplicated(subset=keys + ['thorsync_dir'], keep=False)
    try:
        assert not duped_thorimage.any()
        assert not duped_thorsync.any()
    except AssertionError:
        print('Duplicated ThorImage directories after path inference:')
        print(df[duped_thorimage])
        print('\nDuplicated ThorSync directories after path inference:')
        print(df[duped_thorsync])
        raise

    flies = sheets['fly_preps']
    flies['date'] = flies['date'].fillna(method='ffill')
    flies.dropna(subset=['date','fly_num'], inplace=True)

    # TODO maybe flag to not update database? or just don't?
    # TODO groups all inserts into transactions across tables, and as few as
    # possible (i.e. only do this later)?
    db.to_sql_with_duplicates(flies.rename(
        columns={'date': 'prep_date'}), 'flies'
    )

    # For manual sanity checking that important data isn't being excluded
    # inappropriately.
    if print_excluded_on_disk:
        ti_ondisk_not_in_df = set()
        ts_ondisk_not_in_df = set()
        for gn, gdf in df.groupby(keys):
            fly_dir = raw_fly_dir(*gn)
            if not exists(fly_dir):
                continue

            # Need them somewhat absolute (w/ date + fly info at least), so that
            # set operations on directories across (date, fly) combinations are
            # meaningful.
            gdf_ti = set(gdf.thorimage_dir.apply(lambda d: join(fly_dir, d)))
            gdf_ts = set(gdf.thorsync_dir.apply(lambda d: join(fly_dir, d)))

            thorimage_dirs, thorsync_dirs = thor.thor_subdirs(fly_dir)
            ti_ondisk_not_in_df |= set(thorimage_dirs) - gdf_ti
            ts_ondisk_not_in_df |= set(thorsync_dirs) - gdf_ts

        # Excluding other-project stuff that was dropped from df earlier.
        ti_ondisk_not_in_df -= ti_from_other_projects
        ts_ondisk_not_in_df -= ts_from_other_projects

        msg = '{} directories on disk but not in DataFrame (from gsheet):'
        if len(ti_ondisk_not_in_df) > 0:
            print(msg.format('ThorImage'))
            pprint(ti_ondisk_not_in_df)
            print('')

        if len(ts_ondisk_not_in_df) > 0:
            print(msg.format('ThorSync'))
            pprint(ts_ondisk_not_in_df)
            print('')

    fly_dirs = df.apply(lambda r: raw_fly_dir(r.date, r.fly_num), axis=1)
    abs_thorimage_dirs = fly_dirs.str.cat(others=df.thorimage_dir, sep='/')
    abs_thorsync_dirs = fly_dirs.str.cat(others=df.thorsync_dir, sep='/')
    thorimage_exists = abs_thorimage_dirs.apply(isdir)
    thorsync_exists = abs_thorsync_dirs.apply(isdir)
    any_dir_missing = ~ (thorimage_exists & thorsync_exists)

    any_missing_marked_attempt = (any_dir_missing & df.attempt_analysis)
    # TODO maybe an option to just warn here, rather than failing
    if any_missing_marked_attempt.any():
        print('Directories marked attempt analysis with missing data:')
        print(df.loc[any_missing_marked_attempt, cols[:-1]].to_string())
        print('')
        raise AssertionError('some experiments marked attempt_analysis '
            'had some data directories missing')

    if drop_nonexistant_dirs:
        n_to_drop = any_dir_missing.sum()
        if n_to_drop > 0:
            print(
                f'Dropping {n_to_drop} rows because directories did not exist.'
            )
        df.drop(df[any_dir_missing].index, inplace=True)

    # TODO TODO is 2019-08-27 fn_0000 stuff inferred correctly?
    # (will have same thorimage num as fn) (?)
    # (not critical apart from value as test case, b/c all stuff used from
    # that day has explicit paths in gsheet)

    _mb_team_gsheet = df

    # TODO handle case where database is empty but gsheet cache still exists
    # (all inserts will probably fail, for lack of being able to reference fly
    # table)
    return df


def arraylike_cols(df):
    """Returns a list of columns that have only lists or arrays as elements.
    """
    df = df.select_dtypes(include='object')
    return df.columns[df.applymap(lambda o:
        type(o) is list or isinstance(o, np.ndarray)).all()]


# TODO use in other places that duplicate this functionality
# (like in natural_odors/kc_analysis ?)
def expand_array_cols(df):
    """Expands any list/array entries, with new rows for each entry.

    For any columns in `df` that have all list/array elements (at each row),
    the column in `out_df` will have the type of single elements from those
    arrays.

    The length of `out_df` will be the length of the input `df`, multiplied by
    the length (should be common in each input row) of each set of list/array
    elements.

    Other columns have their values duplicated, to match the lengths of the
    expanded array values.

    Args:
    `df` (pd.DataFrame)

    Returns:
    `out_df` (pd.DataFrame)
    """
    if len(df.index.names) > 1 or df.index.names[0] is not None:
        raise NotImplementedError('numpy repeating may not handle index. '
            'reset_index first.')

    # Will be ['raw_f', 'df_over_f', 'from_onset'] in the main way I'm using
    # this function.
    array_cols = arraylike_cols(df)

    if len(array_cols) == 0:
        raise ValueError('df did not appear to have any columns with all '
            'arraylike elements')

    orig_dtypes = df.dtypes.to_dict()
    for ac in array_cols:
        df[ac] = df[ac].apply(lambda x: np.array(x))
        assert len(df[ac]) > 0 and len(df[ac][0]) > 0
        orig_dtypes[ac] = df[ac][0][0].dtype

    non_array_cols = df.columns.difference(array_cols)

    # TODO true vectorized way to do this?
    # is str.len (on either rows/columns) faster (+equiv)?
    array_lengths = df[array_cols].applymap(len)
    c0 = array_lengths[array_cols[0]]
    for c in array_cols[1:]:
        assert np.array_equal(c0, array_lengths[c])
    array_lengths = c0

    # TODO more idiomatic / faster way to do what this loop is doing?
    n_non_array_cols = len(non_array_cols)
    expanded_rows_list = []
    for row, n_repeats in zip(df[non_array_cols].values, array_lengths):
        # could try subok=True if want to use pandas obj as input rather than
        # stuff from .values?
        expanded_rows = np.broadcast_to(row, (n_repeats, n_non_array_cols))
        expanded_rows_list.append(expanded_rows)
    nac_data = np.concatenate(expanded_rows_list, axis=0)

    ac_data = df[array_cols].apply(np.concatenate)
    assert nac_data.shape[0] == ac_data.shape[0]
    data = np.concatenate((nac_data, ac_data), axis=1)
    assert data.shape[1] == df.shape[1]

    new_cols = list(non_array_cols) + list(array_cols)
    # TODO copy=False is fine here, right? measure the time difference?
    out_df = pd.DataFrame(columns=new_cols, data=data).astype(orig_dtypes,
        copy=False)

    return out_df


def diff_dataframes(df1, df2):
    """Returns a DataFrame summarizing input differences.
    """
    # TODO do i want df1 and df2 to be allowed to be series?
    # (is that what they are now? need to modify anything?)
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    # TODO is this really necessary? not an empty df in this case anyway?
    if df1.equals(df2):
        return None
    else:
        # TODO unit test w/ descrepencies in each of the cases.
        # TODO also test w/ nan in list / nan in float column (one / both nan)
        floats1 = df1.select_dtypes(include='float')
        floats2 = df2.select_dtypes(include='float')
        assert set(floats1.columns) == set(floats2.columns)
        diff_mask_floats = ~pd.DataFrame(
            columns=floats1.columns,
            index=df1.index,
            # TODO TODO does this already deal w/ nan correctly?
            # otherwise, this part needs to handle possibility of nan
            # (it does not. need to handle.)
            data=np.isclose(floats1, floats2)
        )
        diff_mask_floats = (diff_mask_floats &
            ~(floats1.isnull() & floats2.isnull()))

        # Just assuming, for now, that array-like cols are same across two dfs.
        arr_cols = arraylike_cols(df1)
        # Also assuming, for now, that no elements of these lists / arrays will
        # be nan (which is currently true).
        diff_mask_arr = ~pd.DataFrame(
            columns=arr_cols,
            index=df1.index,
            data=np.vectorize(np.allclose)(df1[arr_cols], df2[arr_cols])
        )

        other_cols = set(df1.columns) - set(floats1.columns) - set(arr_cols)
        other_diff_mask = df1[other_cols] != df2[other_cols]

        diff_mask = pd.concat([
            diff_mask_floats,
            diff_mask_arr,
            other_diff_mask], axis=1)

        if diff_mask.sum().sum() == 0:
            return None

        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        # TODO are these what i want? prob change id (basically just to index?)?
        # TODO get id from index name of input dfs? and assert only one index
        # (assuming this wouldn't work w/ multiindex w/o modification)?
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)


def first_group(df, group_cols):
    """Returns key tuple and df of first group, grouping df on group_cols.

    Just for ease of interactively testing out functions on DataFrames of a
    groupby.
    """
    gb = df.groupby(group_cols)
    first_group_tuple = list(gb.groups.keys())[0]
    gdf = gb.get_group(first_group_tuple)
    return first_group_tuple, gdf


def git_hash(repo_file):
    """Takes any file in a git directory and returns current hash.
    """
    import git
    repo = git.Repo(repo_file, search_parent_directories=True)
    current_hash = repo.head.object.hexsha
    return current_hash


# TODO TODO maybe check that remote seems to be valid, and fail if not.
# don't want to assume we have an online (backed up) record of git repo when we
# don't...
def version_info(*args, used_for='', force_git=False):
    """Takes module or string path to file in Git repo to a dict with version
    information (with keys and values the database will accept).
    """
    import git
    import pkg_resources

    if len(args) == 1:
        module_or_path = args[0]
    elif len(args) == 0:
        module_or_path = __file__
        force_git = True
    else:
        raise ValueError('too many arguments')

    if isinstance(module_or_path, ModuleType):
        module = module_or_path
        pkg_path = module.__file__
        name = module.__name__
    else:
        if type(module_or_path) != str:
            raise ValueError('must path either a Python module or str path')
        pkg_path = module_or_path
        module = None

    try:
        repo = git.Repo(pkg_path, search_parent_directories=True)
        name = split(repo.working_tree_dir)[-1]
        remote_urls = list(repo.remotes.origin.urls)
        assert len(remote_urls) == 1
        remote_url = remote_urls[0]

        current_hash = repo.head.object.hexsha

        index = repo.index
        diff = index.diff(None, create_patch=True)
        changes = ''
        for d in diff:
            changes += str(d)

        return {
            'name': name,
            'used_for': used_for,
            'git_remote': remote_url,
            'git_hash': current_hash,
            'git_uncommitted_changes': changes
        }

    except git.exc.InvalidGitRepositoryError:
        if force_git:
            raise

        if module is None:
            # TODO try to find module from str
            raise NotImplementedError(
                'pass module for non-source installations')

        # There may be circumstances in which module name isn't the right name
        # to use here, but assuming we won't encounter that for now.
        version = pkg_resources.get_distribution(module.__name__).version

        return {'name': name, 'used_for': used_for, 'version': version}


def motion_corrected_tiff_filename(date, fly_num, thorimage_id):
    """Takes vars identifying recording to the name of a motion corrected TIFF
    for it. Non-rigid preferred over rigid. Relies on naming convention.
    """
    tif_dir = join(analysis_fly_dir(date, fly_num), 'tif_stacks')
    nr_tif = join(tif_dir, '{}_nr.tif'.format(thorimage_id))
    rig_tif = join(tif_dir, '{}_rig.tif'.format(thorimage_id))
    tif = None
    if exists(nr_tif):
        tif = nr_tif
    elif exists(rig_tif):
        tif = rig_tif

    if tif is None:
        raise IOError('No motion corrected TIFs found in {}'.format(tif_dir))

    return tif


# TODO use this in other places that normalize to thorimage_ids
def tiff_thorimage_id(tiff_filename):
    """
    Takes a path to a TIFF and returns ID to identify recording within
    (date, fly). Relies on naming convention.

    Works for input that is either a raw TIFF or a motion corrected TIFF,
    the latter of which should have a conventional suffix indicating the
    type of motion correction ('_nr' / '_rig').
    """
    # Behavior of os.path.split makes this work even if tiff_filename does not
    # have any directories in it.
    parts = split(tiff_filename[:-len('.tif')])[1].split('_')

    # Last part of the filename, which I use to indicate the type of motion
    # correction applied. Should only apply in TIFFs under analysis directory.
    if parts[-1] in ('nr', 'rig'):
        parts = parts[:-1]

    return '_'.join(parts)


def metadata_filename(date, fly_num, thorimage_id):
    """Returns filename of YAML for extra metadata.
    """
    return join(raw_fly_dir(date, fly_num), thorimage_id + '_metadata.yaml')


# TODO maybe something to indicate various warnings
# (like mb team not being able to pair things) should be suppressed?
# TODO wrap this + read_movie into loading that can also flip according to a key in the
# yaml (L/R, for more easily comparing data from diff AL sides in flies in the same
# orientation, for example)
def metadata(date, fly_num, thorimage_id):
    """Returns metadata from YAML, with defaults added.
    """
    import yaml

    metadata_file = metadata_filename(date, fly_num, thorimage_id)

    # TODO another var specifying number of frames that has *already* been
    # cropped out of raw tiff (start/end), to resolve any descrepencies wrt
    # thorsync data
    metadata = {
        'drop_first_n_frames': 0
    }
    if exists(metadata_file):
        # TODO TODO TODO also load single odors (or maybe other trial
        # structures) from stuff like this, so analysis does not need my own
        # pickle based stim format
        with open(metadata_file, 'r') as mdf:
            yaml_metadata = yaml.load(mdf)

        for k in metadata.keys():
            if k in yaml_metadata:
                metadata[k] = yaml_metadata[k]

    return metadata


def tiff_filename2keys(tiff_filename):
    """Takes TIFF filename to pd.Series w/ 'date','fly_num','thorimage_id' keys.

    TIFF must be placed and named according to convention, because the
    date and fly_num are taken from names of some of the containing directories.

    Works with TIFFs either under `raw_data_root` or `analysis_output_root`.
    """
    parts = tiff_filename.split(sep)[-5:]
    date = None
    for i, p in enumerate(parts):
        try:
            date = pd.Timestamp(datetime.strptime(p, date_fmt_str))
            fly_num_idx = i + 1
            break
        except ValueError:
            pass

    if date is None:
        raise ValueError('no date directory found in TIFF path')

    fly_num = int(parts[fly_num_idx])
    thorimage_id = tiff_thorimage_id(tiff_filename)
    return pd.Series({
        'date': date, 'fly_num': fly_num, 'thorimage_id': thorimage_id
    })


def recording_df2keys(df):
    dupes = df[recording_cols].drop_duplicates()
    assert len(dupes) == 1
    return tuple(dupes.iloc[0])


def list_motion_corrected_tifs(include_rigid=False, attempt_analysis_only=True):
    """List motion corrected TIFFs in conventional directory structure on NAS.
    """
    motion_corrected_tifs = []
    df = mb_team_gsheet()
    for full_date_dir in sorted(glob.glob(join(analysis_output_root(), '**'))):
        for full_fly_dir in sorted(glob.glob(join(full_date_dir, '**'))):
            date_dir = split(full_date_dir)[-1]
            try:
                fly_num = int(split(full_fly_dir)[-1])

                fly_used = df.loc[df.attempt_analysis &
                    (df.date == date_dir) & (df.fly_num == fly_num)]

                used_thorimage_dirs = set(fly_used.thorimage_dir)

                tif_dir = join(full_fly_dir, 'tif_stacks')
                if exists(tif_dir):
                    tif_glob = '*.tif' if include_rigid else '*_nr.tif'
                    fly_tifs = sorted(glob.glob(join(tif_dir, tif_glob)))

                    used_tifs = [x for x in fly_tifs if '_'.join(
                        split(x)[-1].split('_')[:-1]) in used_thorimage_dirs]

                    motion_corrected_tifs += used_tifs

            except ValueError:
                continue

    return motion_corrected_tifs


# TODO still work w/ parens added around initial .+ ? i want to match the parent
# id...
shared_subrecording_regex = r'(.+)_\db\d_from_(nr|rig)'
def is_subrecording(thorimage_id):
    """
    Returns whether a recording id matches my GUIs naming convention for the
    "sub-recordings" it can create.
    """
    if re.search(shared_subrecording_regex + '$', thorimage_id):
        return True
    else:
        return False


def is_subrecording_tiff(tiff_filename):
    """
    Takes a TIFF filename to whether it matches the GUI's naming convention for
    the "sub-recordings" it can create.
    """
    # TODO technically, nr|rig should be same across two...
    if re.search(shared_subrecording_regex + '_(nr|rig).tif$', tiff_filename):
        return True
    else:
        return False


def subrecording_tiff_blocks(tiff_filename):
    """Returns tuple of int (start, stop) block numbers subrecording contains.

    Block numbers start at 0.

    Requires that is_subrecording_tiff(tiff_filename) would return True.
    """
    parts = tiff_filename.split('_')[-4].split('b')

    first_block = int(parts[0]) - 1
    last_block = int(parts[1]) - 1

    return first_block, last_block


def subrecording_tiff_blocks_df(series):
    """Takes a series w/ TIFF name in series.name to (start, stop) block nums.

    (series.name must be a TIFF path)

    Same behavior as `subrecording_tiff_blocks`.
    """
    # TODO maybe fail in this case?
    if not series.is_subrecording:
        return None, None

    tiff_filename = series.name
    first_block, last_block = subrecording_tiff_blocks(tiff_filename)
    return first_block, last_block
    '''
    return {
        'first_block': first_block,
        'last_block': last_block
    }
    '''


def parent_recording_id(tiffname_or_thorimage_id):
    # TODO provide example of naming convention / manipulation in doc
    """Returns recording id for recording subrecording was derived from.

    Input can be a TIFF filename or recording id.
    """
    last_part = split(tiffname_or_thorimage_id)[1]
    match = re.search(shared_subrecording_regex, last_part)
    if match is None:
        raise ValueError('not a subrecording')
    return match.group(1)


def write_tiff(tiff_filename, movie, strict_dtype=True):
    # TODO also handle diff color channels
    """Write a TIFF loading the same as the TIFFs we create with ImageJ.

    TIFFs are written in big-endian byte order to be readable by `imread_big`
    from MATLAB file exchange.

    Dimensions of input should be (t,[z,],y,x).

    Metadata may not be correct.
    """
    import tifffile

    if strict_dtype:
        dtype = movie.dtype
        if not (dtype.itemsize == 2 and
            np.issubdtype(dtype, np.unsignedinteger)):

            # TODO TODO TODO handle casting from float (for df/f images, for example)
            # (how does imagej do this type of casting? i would think it would also need
            # to do something like that?) (at least if not strict_dtype)
            raise ValueError('movie must have uint16 dtype')

        if dtype.byteorder == '|':
            raise ValueError('movie must have explicit endianness')

        # If little-endian, convert to big-endian before saving TIFF, almost
        # exclusively for the benefit of MATLAB imread_big, which doesn't seem
        # able to discern the byteorder.
        if (dtype.byteorder == '<' or
            (dtype.byteorder == '=' and sys.byteorder == 'little')):
            movie = movie.byteswap().newbyteorder()
        else:
            assert dtype.byteorder == '>'

    # TODO TODO maybe change so ImageJ considers appropriate dimension the time
    # dimension (both in 2d x T and 3d x T cases)
    # TODO TODO TODO convert from thor data to appropriate dimension order (w/
    # singleton dimensions as necessary) (or keep dimensions + dimension order
    # of array, and pass metadata={'axes': 'TCXY'}, w/ the value constructed
    # appropriately? that work w/ imagej=True?)

    # TODO TODO TODO since scipy docs say [their version] of tifffile expects
    # channels in TZCYXS order
    # https://scikit-image.org/docs/0.14.x/api/skimage.external.tifffile.html

    if len(movie.shape) == 3:
        #axes = 'TXY'
        # Z and C
        new_dim_indices = (1, 2)
    elif len(movie.shape) == 4:
        #axes = 'TZXY'
        # C
        new_dim_indices = (2,)
    else:
        raise ValueError('unexpected number of dimensions to movie. have '
            f'{len(movie.shape)}. expected 3 (TXY) or 4 (TZXY).'
        )

    # I believe in some installation this line failed in (2d x t) case where
    # (415, 1, 192, 192) was initial movie.shape (a single plane expt), but now
    # I can't reproduce it in a python3.8 venv.
    movie = np.expand_dims(movie, new_dim_indices)

    # not doing this still seems to produce output consistent w/ what i see on
    # the scope, in constrast to what i see if i *do* swap these
    # Switches from last two dimensions being XY to YX
    #movie = np.swapaxes(movie, -2, -1)

    # TODO TODO is "UserWarning: TiffWriter: truncating ImageJ file" actually
    # something to mind? for example, w/ 2020-04-01/2/fn as input, the .raw is
    # 8.3GB and the .tif is 5.5GB (w/ 3 flyback frames for each 6 non-flyback
    # frames -> 8.3 * (2/3) = ~5.53  (~ 5.5...). some docs say bigtiff is not
    # supported w/ imagej=True, so maybe that wouldn't be a quick fix if the
    # warning actually does matter. if not, maybe suppress it somehow?

    # TODO actually make sure any metadata we use is the same
    # TODO maybe just always do test from test_readraw here?
    # (or w/ flag to disable the check)
    tifffile.imsave(tiff_filename, movie, imagej=True)
    # doesn't seem to work (reading it into imagej still has a slider for C at
    # the bottom, and it jumps across Z planes when scrolling either slider)
    #tifffile.imsave(tiff_filename, movie, metadata={'axes': axes})


def full_frame_avg_trace(movie):
    """Takes a (t,[z,]x,y) movie to t-length vector of frame averages.
    """
    # Averages all dims but first, which is assumed to be time.
    return np.mean(movie, axis=tuple(range(1, movie.ndim)))


# TODO TODO switch order of args, and allow passing just coords. if just coords are
# passed, shift all towards 0 (+ margin). use for e.g. xpix/ypix stats elements in
# suite2p stat output. corresponding pixel weights in lam output would not need to be
# modified.
def crop_to_coord_bbox(matrix, coords, margin=0):
    """Returns matrix cropped to bbox of coords and bounds.
    """
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    assert x_min >= 0 and y_min >= 0, \
        f'mins must be >= 0 (x_min={x_min}, y_min={y_min})'

    # NOTE: apparently i had to use <= or else some cases in old use of this function
    # (e.g. using certain CNMF outputs) would violate this. best to just fix that code
    # if it ever comes up again though.
    assert x_max < matrix.shape[0] and y_max < matrix.shape[1], (
        f'maxes must be < matrix shape = {matrix.shape} (x_max={x_max}' +
        f', y_max={y_max}'
    )

    # Keeping min at 0 to prevent slicing error in that case
    # (I think it will be empty, w/ -1:2, for instance)
    # Capping max not necessary to prevent err, but to make behavior of bounds
    # consistent on both edges.
    # TODO flag to err if margin would take it past edge? / warn?
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(x_max + margin, matrix.shape[0] - 1)
    y_max = min(y_max + margin, matrix.shape[1] - 1)

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


def crop_to_nonzero(matrix, margin=0):
    """
    Returns a matrix just large enough to contain the non-zero elements of the
    input, and the bounding box coordinates to embed this matrix in a matrix
    with indices from (0,0) to the max coordinates in the input matrix.
    """
    # nan_to_num will replace nan w/ 0 by default. infinities also converted but not
    # expected to be in input.
    coords = np.argwhere(np.nan_to_num(matrix) > 0)
    return crop_to_coord_bbox(matrix, coords, margin=margin)


# TODO if these 'db_row2*' fns are really just db related, move to db.py, but
# probably just rename...
# TODO better name?
def db_row2footprint(db_row, shape=None):
    """Returns dense array w/ footprint from row in cells table.
    """
    from scipy.sparse import coo_matrix
    weights, x_coords, y_coords = db_row[['weights','x_coords','y_coords']]
    # TODO maybe read shape from db / metadata on disk? / merging w/ other
    # tables (possible?)?
    footprint = np.array(coo_matrix((weights, (x_coords, y_coords)),
        shape=shape).todense()).T
    return footprint


def db_footprints2array(df, shape):
    """Returns footprints in an array of dims (shape + (n_footprints,)).
    """
    return np.stack([db_row2footprint(r, shape) for _, r in df.iterrows()],
        axis=-1)


# TODO maybe refactor so there is a function does this for single arrays, then concat
# using xarray functions in here? or if i still want both functions, how to dedupe code?
# allow this to accept single rois too (without that component of shape)?
def numpy2xarray_rois(rois, roi_indices=None):
    """Takes numpy array of shape ([z,]y,x,roi) to labelled xarray.

    Args:
        roi_indices (None | dict): values must be iterables of length equal to number of
            ROIs. 'roi_num' will be included as an additional ROI index regardless.
    """
    shape = rois.shape
    # TODO check that the fact that i swapped y and x now didn't break how i was using
    # this w/ actual ijrois / anything else. wanted to be more consistent w/ how
    # suite2p, ImageJ, etc seemed to do things.
    if len(shape) == 3:
        dims = ['y', 'x', 'roi']
    elif len(shape) == 4:
        dims = ['z', 'y', 'x', 'roi']
    else:
        raise ValueError('shape must have length 3 or 4')

    # NOTE: 'roi_num' can't be replaced w/ 'roi' b/c conflict w/ name of 'roi' dim
    roi_num_name = 'roi_num'
    roi_index_names = [roi_num_name]
    roi_index_levels = [np.arange(rois.shape[-1])]

    if roi_indices is not None:
        # If a 'roi_num' level is passed in, it will replace the one that would be added
        # automatically.
        if roi_num_name in roi_indices:
            roi_index_names = []
            roi_index_levels = []

        n_rois = shape[-1]
        for ns, xs in roi_indices.items():
            assert len(xs) == n_rois
            roi_index_names.append(ns)
            roi_index_levels.append(xs)

    roi_index = pd.MultiIndex.from_arrays(roi_index_levels, names=roi_index_names)
    return xr.DataArray(rois, dims=dims, coords={'roi': roi_index})


# TODO TODO rename / delete one-or-the-other of this and contour2mask etc
# (+ accept ijroi[set] filename or something if actually gonna call it this)
# (ALSO include ijrois2masks in consideration for refactoring. this fn might not be
# necessary)
def ijroi2mask(roi, shape, z=None):
    """
        z (None | int): (optional) z-index ROI was drawn on
    """
    # This mask creation was taken from Yusuke N.'s answer here:
    # https://stackoverflow.com/questions/3654289
    from matplotlib.path import Path

    if z is None:
        if len(shape) != 2:
            raise ValueError(f'len(shape) == {len(shape)}. must be 2 if z keyword '
                'argument not passed'
            )

        # TODO check transpose isn't necessary...
        nx, ny = shape

    else:
        if len(shape) != 3:
            raise ValueError(f'shape must be (z, x, y) if z is passed. shape == {shape}'
                ', which has the wrong length'
            )

        # TODO check transpose (of x and y) isn't necessary...
        nz, nx, ny = shape
        if z >= nz:
            raise ValueError(f'z ({z}) out of bounds with z size ({nz}) from shape[0]')

    # TODO test + delete
    #assert nx == ny, 'need to check code shoulnt be tranposing these'
    #

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(roi)
    grid = path.contains_points(points)
    # Transpose makes this correct in testing on some of YZ's data
    mask = grid.reshape((ny, nx)).T

    if z is None:
        return mask
    else:
        vol_mask = np.zeros(shape, dtype='bool')
        vol_mask[z, :, :] = mask
        return vol_mask


# TODO TODO add option to translate ijroi labels to pandas index values?
# (and check trace extraction downstream preserves those!)
# TODO TODO TODO document type / structure expecations of `ijrois` arg!
# TODO TODO accept either the input or output of ijroi.read_roi[_zip] for ijrois?
# read_roi takes file object and read_roi_zip takes filename
# TODO can ijroi lib be modified to read path to tif things were drawn over (is that
# data there?), and can that be used to get shape? or can i also accept a path to tiff
# / thorimage dir / etc for that?
# TODO TODO option to use one of those scipy sparse arrays for the masks instead?
# TODO TODO maybe update all of my roi stuff that currently has the roi index as the
# last index so that it is the first index instead? feels more intuitive...
# TODO TODO make as_xarray default behavior and remove if other places that use this
# output don't break / change them
# TODO TODO TODO fn to convert suite2p representation of masks to the same [xarray]
# representation of masks this spits out
def ijrois2masks(ijrois, shape, as_xarray=False):
    # TODO be clear on where shape is coming from (just shape of the data in the TIFF
    # the ROIs were draw in, right?)
    """
    Transforms ROIs loaded from my ijroi fork to an array full of boolean masks,
    of dimensions (shape + (n_rois,)).
    """
    import ijroi

    # TODO delete depending on how refactoring the below into xarray fn goes / whether
    # the non-xarray parts of this fn have the same requirements (which they probably
    # do...)
    if len(shape) not in (2, 3):
        raise ValueError('shape must have length 2 or 3')

    masks = []
    names = []
    if len(shape) == 3:
        roi_z_indices = []
        prefixes = []
        suffixes = []

    for name, roi in ijrois:

        if len(shape) == 3:
            # Otherwise, it should simply be a numpy array with points.
            # `points_only=False` to either `read_roi_zip` or `read_roi` should produce
            # input suitable for this branch.
            if hasattr(roi, 'z'):
                z_index = roi.z

                # This should be the same as the `name` this shadows, just without the
                # '.roi' suffix.
                name = roi.name

                try:
                    _, (prefix, suffix) = ijroi.parse_z_from_name(name,
                        return_unmatched=True
                    )
                except ValueError:
                    prefix = name
                    suffix = ''

            else:
                warnings.warn('trying to parse Z from ROI name. pass points_only=False'
                    ' to ijroi loading function to read Z directly.'
                )
                z_index, (prefix, suffix) = ijroi.parse_z_from_name(name,
                    return_unmatched=True
                )
        else:
            z_index = None

        if hasattr(roi, 'points'):
            points = roi.points
        else:
            points = roi

        # TODO may also need to reverse part of shape here, if really was
        # necessary above (test would probably need to be in asymmetric
        # case...)
        masks.append(ijroi2mask(points, shape, z=z_index))
        names.append(name)

        if len(shape) == 3:
            roi_z_indices.append(z_index)

            if len(prefix) == 0:
                prefix = np.nan

            if len(suffix) == 0:
                suffix = np.nan

            prefixes.append(prefix)
            suffixes.append(suffix)

    # This concatenates along the last element of the new shape
    masks = np.stack(masks, axis=-1)

    if not as_xarray:
        return masks

    roi_index_names = ['roi_name']
    roi_index_levels = [names]
    if len(shape) == 3:
        roi_index_names += ['roi_z', 'ijroi_prefix', 'ijroi_suffix']
        roi_index_levels += [roi_z_indices, prefixes, suffixes]

    return numpy2xarray_rois(masks,
        roi_indices=dict(zip(roi_index_names, roi_index_levels))
    )


# TODO maybe add a fn to plot single xarray masks for debugging?
# TODO TODO change `on` default to something like `roi`
def merge_rois(rois, on='ijroi_prefix', merge_fn=None, label_fn=None,
    check_no_overlap=False):
    """
    Args:
        rois (xarray.DataArray): must have at least dims 'x', 'y', and 'roi'.
            'roi' should be indexed by a MultiIndex and one of the levels should have
            the name of the `on` argument. Currently expect the dtype to be 'bool'.

        on (str): name of other ROI metadata dimension to merge on

        label_fn (callable): function mapping the values of the `on` column to labels
            for the ROIs. Only ROIs created via merging will be given these labels,
            while unmerged ROIs will recieve unique number labels. Defaults to the
            identity function.

        check_no_overlap (bool): (optional, default=False) If True, checks that no
            merged rois shared any pixels before being merged. If merged ROIs are all
            on different planes, this should be True because ImageJ ROIs are defined on
            single planes.
    """
    # TODO assert bool before this / rename to something like 'total_weight' that would
    # apply in non-boolean-mask case too
    total_weight_before = rois.sum().item()
    n_rois_before = len(rois.roi)

    def get_nonroi_shape(arr):
        return {k: n for k, n in zip(arr.dims, arr.shape) if k != 'roi'}

    nonroi_shape_before = get_nonroi_shape(rois)

    # TODO maybe i should use sum instead, if i'm not going to make the aggregation
    # function configurable?
    # If `on` contains NaN values, the groupby will not include groups for the NaN
    # values, and there is no argument to configure this (as in pandas). Therefore,
    # we need to drop things that were merged and then add these to what remains.
    # TODO do i need to check that nothing else conflicts w/ what i plan on renaming
    # `on` to ('roi')?
    if merge_fn is None:
        merged = rois.groupby(on).max()
    else:
        #raise NotImplementedError
        # TODO maybe try reduce? might need other inputs tho...
        merged = rois.groupby(on).map(merge_fn)

    # TODO some kind of inplace version of this? or does xarray not really do that?
    merged = merged.rename({on: 'roi'})

    if label_fn is None:
        label_fn = lambda x: x

    merged_roi_labels = [label_fn(x) for x in merged.roi.values]
    # Trying to pass label_fn here instead of calling before didn't work b/c it seems to
    # expect a fn that takes a DataArray (and it's not passing scalar DataArrays either)
    merged = merged.assign_coords(roi=merged_roi_labels)

    not_merged = rois[on].isnull()
    unmerged = rois.where(not_merged, drop=True).reset_index('roi', drop=True)

    n_orig_rois_merged = (~ not_merged).values.sum()
    n_rois_after = n_rois_before - n_orig_rois_merged + len(merged.roi)

    available_labels = [x for x in range(n_rois_after) if x not in merged_roi_labels]
    unmerged_roi_labels = available_labels[:len(unmerged.roi)]
    assert len(unmerged_roi_labels) == len(unmerged.roi)
    unmerged = unmerged.assign_coords(roi=unmerged_roi_labels)

    # The .groupby (seemingly with any function application, as .first() also does it)
    # and .where both change the dtype to float64 from bool
    was_bool = rois.dtype == 'bool'

    rois = xr.concat([merged, unmerged], 'roi')

    if was_bool:
        rois = rois.astype('bool')

    assert n_rois_after == len(rois.roi)
    assert len(set(rois.roi.values)) == len(rois.roi)

    total_weight_after = rois.sum().item()
    if check_no_overlap:
        assert total_weight_before == total_weight_after
    else:
        assert total_weight_before >= total_weight_after

    nonroi_shape_after = get_nonroi_shape(rois)
    assert nonroi_shape_before == nonroi_shape_after

    return rois


def merge_ijroi_masks(masks, **kwargs):
    """
    Args:
        masks (xarray.DataArray): must have at least dims 'x', 'y', and 'roi'.
            'roi' should be indexed by a MultiIndex and one of the levels should have
            the name of the `on` argument. Currently expect the dtype to be 'bool'.

        label_fn (callable): function mapping the values of the `on` column to labels
            for the ROIs. Only ROIs created via merging will be given these labels,
            while unmerged ROIs will recieve unique number labels. Defaults to a
            function that takes strings, removes trailing/leading underscores, and
            parses an int from what remains.
    """
    # TODO probably assert bool
    # TODO assert ijroi_prefix in here / accept kwarg on (defaulting to same), and
    # assert that's here

    return merge_rois(masks, on='ijroi_prefix', **kwargs)


# TODO TODO TODO make another function that groups rois based on spatial overlap
# (params to include [variable-number-of?] dilation steps, fraction of pixels[/weight?]
# that need to overlap, and correlation of responses required) -> generate appropriate
# input to [refactored + renamed] merge_ijroi_masks fn below, particularly the `masks`
# and `on` arguments, and have label_fn be identity

# TODO TODO how to handle a correlation threshold? pass correlations of some kind in
# or something to compute them from (probably the former, or make the correlation
# thresholding a separate step)?
def merge_single_plane_rois(rois, min_overlap_frac=0.3, n_dilations=1):
    """
    For handling single plane ROIs that are on adjacent Z planes, and correspond to the
    same biological feature. This is to merge the single plane ROIs that suite2p
    outputs.
    """
    raise NotImplementedError
    import ipdb; ipdb.set_trace()


# TODO TODO TODO refactor this + hong2p.suite2p.remerge_suite2p_merged to share core
# code here!!! (this initially copied from other fn and then adapted)
def rois2best_planes_only(rois, roi_quality):
    """
    Currently assumes input only has non-zero values in a single plane for a given
    unique combination of ROI identifier variables.
    """

    verbose = True

    # TODO TODO implement another strategy where as long as the roi_quality are
    # within some tolerance of the best, they are averaged? or weighted according to
    # response stat? weight according to variance of response stat (see examples of
    # weighted averages using something derived from variance for weights online)
    # TODO maybe also use 2/3rd highest lowest frame / percentile rather than actual min
    # / max (for picking 'best' plane), to gaurd against spiking noise

    '''
    merge_output_roi_nums = np.empty(len(roi_quality.columns)) * np.nan
    # TODO maybe build up set of seen merge input indices and check they are all seen in
    # columns of traces by end (that set seen in traces columns is same as set from
    # unioning all values in merges)
    for merge_output_roi_num, merge_input_roi_nums in merges.items():

        #merge_output_roi_nums[traces.columns.isin(merge_input_roi_nums)] = \
        merge_output_roi_nums[roi_quality.columns.isin(merge_input_roi_nums)] = \
            merge_output_roi_num
    '''

    roi_quality = roi_quality.to_frame(name='roi_quality')
    mo_key = 'name'
    #roi_quality[mo_key] = merge_output_roi_nums
    roi_quality[mo_key] = rois.roi.roi_name.to_numpy()
    gb = roi_quality.groupby(mo_key)
    best_per_merge_output = gb.idxmax()

    # Selecting the only column this DataFrame has (i.e. shape (n, 1))
    best_inputs = best_per_merge_output.iloc[:, 0]
    # The groupby -> idxmax() otherwise would have left this column named
    # 'roi_quality', which is what we were picking an index to maximize, but the
    # indices themselves are not response statistics.
    best_inputs.name = 'roi'

    best = roi_quality.loc[best_inputs]

    # TODO delete eventually
    assert np.array_equal(
       roi_quality.loc[best_inputs.values],
       roi_quality.loc[best_inputs]
    )
    #
    assert np.array_equal(
        best.roi_quality.values,
        gb.max().roi_quality.values
    )

    if verbose:
        by_response = roi_quality.dropna().set_index('name',
            append=True).swaplevel()

    notbest_to_drop = []
    for row in best.itertuples():
        merged_name = row.name
        curr_best = row.Index
        curr_notbest = list(rois.roi[
            (rois.roi_name == merged_name) & (rois.roi_num != curr_best)
        ].roi_num.values)

        notbest_to_drop.extend(curr_notbest)

        if verbose:
            print(f'merging ROI {merged_name}')
            print(f'selecting input ROI {curr_best} as best plane')
            print(f'dropping other input ROIs {curr_notbest}')
            print(by_response.loc[merged_name])
            print()

    # TODO maybe combine in to one step by just passing subsetted s2p_roi_num to
    # assign_coords?
    rois = rois.sel(roi= ~ rois.roi_num.isin(notbest_to_drop))

    roi_nums = rois.roi_num.values

    # TODO should i also include roi_z? would want to also do / use in s2p case for
    # consistency... also, how to modify this call to accomplish that?
    rois = rois.assign_coords(roi=rois.roi_name)

    # TODO TODO if i can figure out how to keep multiple levels for the roi dimension,
    # do that rather than return multiple things
    return roi_nums, rois


def ijroi_masks(ijroiset_dir_or_fname, thorimage_dir, as_xarray=True, **kwargs):

    # This must be my fork at https://github.com/tom-f-oconnell/ijroi
    import ijroi

    if isdir(ijroiset_dir_or_fname):

        ijroiset_basename = 'RoiSet.zip'

        # TODO if i standardize path to analysis intermediates, update this to look for
        # RoiSet.zip there?
        ijroiset_fname = join(ijroiset_dir_or_fname, ijroiset_basename)

        if not isfile(ijroiset_fname):
            raise IOError('directory passed for ijroiset_dir_or_fname, but '
                f'{ijroiset_fname} did not exist'
            )

    name_and_roi_list = ijroi.read_roi_zip(ijroiset_fname, points_only=False)

    _, (x, y), z, c, _, _ =  thor.load_thorimage_metadata(thorimage_dir)

    assert x == y, 'not tested in case x != y'

    # From what `thor.read_movie` says the output dimensions are (except the first
    # dimension, which is time).
    if z == 1:
        movie_shape_without_time = (y, x)
    else:
        movie_shape_without_time = (z, y, x)

    masks = ijrois2masks(name_and_roi_list, movie_shape_without_time,
        as_xarray=as_xarray
    )
    return masks

    ## TODO modify check_no_overlap to make sure it's also erring if two things that
    ## would be merged (by having same name / whatever) are not in the same z-plane
    ## (assuming the intention was to have one per plane, to make a single volumetric
    ## ROI)
    #merged = merge_ijroi_masks(masks, check_no_overlap=True)
    #
    #import ipdb; ipdb.set_trace()
    #
    #return merged


# TODO test / document requirements for type / properties of contour. it's just a
# numpy array of points, right? doesn't need start = end or anything, does it?
def contour2mask(contour, shape):
    """Returns a boolean mask True inside contour and False outside.
    """
    # TODO TODO TODO appropriate checking of contour input. i.e. any requirements on
    # first/last point / order (do some orders imply overlapping edge segments, and if
    # so, check there are none of those)
    import cv2
    # TODO any checking of contour necessary for it to be well behaved in
    # opencv?
    mask = np.zeros(shape, np.uint8)

    # NOTE: at some point i think i needed convexHull in ijroi2mask to get that + this
    # to work as I expected. AS I THINK CONVEXHULL MIGHT RESULT IN SOME UNEXPECTED
    # MODIFICATIONS TO CONTOURS, i need to change that code, and that might break some
    # of this code too
    # TODO TODO TODO if drawContours truly does need convex hull inputs, need to change
    # this function to no longer use drawContours
    # TODO TODO TODO see strategy i recommended to yang recently and consider using it
    # here instead
    # TODO draw into a sparse array maybe? or convert after?
    cv2.drawContours(mask, [contour.astype(np.int32)], 0, 1, -1)

    # TODO TODO TODO investigate need for this transpose
    # (imagej contour repr specific? maybe load to contours w/ dims swapped them
    # call this fn w/o transpose?)
    # (was it somehow still a product of x_coords / y_coords being swapped in
    # db?)
    # not just b/c reshaping to something expecting F order CNMF stuff?
    # didn't correct time averaging w/in roi also require this?
    return mask.astype('bool')


# TODO rename these two to indicate it only works on images (not coordinates)
# TODO and make + use fns that operate on coordinates?
def imagej2py_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
    # TODO how does this behave in the 3d case...?
    # still what i want, or should i exclude the z dimension somehow?
    # TODO TODO TODO probably just delete any code that actually relied on this?
    # assuming it doesn't still make sense...
    #return array.T
    return array


def py2imagej_coords(array):
    """
    Since ijroi source seems to have Y as first coord and X as second.
    """
    # TODO TODO TODO probably just delete any code that actually relied on this?
    # assuming it doesn't still make sense...
    #return array.T
    return array


# TODO maybe move to a submodule for interfacing w/ cnmf?
# TODO TODO probably make a corresponding fn to do the inverse
# (or is one of these not necessary? in one dir, is order='C' and order
def footprints_to_flat_cnmf_dims(footprints):
    """Takes array of (x, y[, z], n_footprints) to (n_pixels, n_footprints).

    There is more than one way this reshaping can be done, and this produces
    output as CNMF expects it.
    """
    frame_pixels = np.prod(footprints.shape[:-1])
    n_footprints = footprints.shape[-1]
    # TODO TODO is this supposed to be order='F' or order='C' matter?
    # wrong setting equivalent to transpose?
    # what's the appropriate test (make unit?)?
    return np.reshape(footprints, (frame_pixels, n_footprints), order='F')


def extract_traces_bool_masks(movie, footprints,
    footprint_framenums=None, verbose=True):
    """
    Averages the movie within each boolean mask in footprints
    to make a matrix of traces (n_frames x n_footprints).
    """
    assert footprints.dtype.kind != 'f', 'float footprints are not boolean'
    assert footprints.max() == 1, 'footprints not boolean'
    assert footprints.min() == 0, 'footprints not boolean'
    n_spatial_dims = len(footprints.shape) - 1
    spatial_dims = tuple(range(n_spatial_dims))
    assert np.any(footprints, axis=spatial_dims).all(), 'some zero footprints'
    slices = (slice(None),) * n_spatial_dims
    n_frames = movie.shape[0]
    n_footprints = footprints.shape[-1]
    traces = np.empty((n_frames, n_footprints)) * np.nan

    if verbose:
        print('extracting traces from boolean masks...', end='', flush=True)

    # TODO vectorized way to do this?
    for i in range(n_footprints):
        mask = footprints[slices + (i,)]
        # TODO compare time of this to sparse matrix dot product?
        # + time of MaskedArray->mean w/ mask expanded by n_frames?

        # TODO TODO is this correct? check
        # axis=1 because movie[:, mask] only has two dims (frames x pixels)
        trace = np.mean(movie[:, mask], axis=1)
        assert len(trace.shape) == 1 and len(trace) == n_frames
        traces[:, i] = trace

    if verbose:
        print(' done')

    return traces


def exp_decay(t, scale, tau, offset):
    # TODO is this the usual definition of tau (as in RC time constant?)
    return scale * np.exp(-t / tau) + offset


# TODO call for each odor onset (after fixed onset period?)
# est onset period? est rise kinetics jointly? how does cnmf do it?
def fit_exp_decay(signal, sampling_rate=None, times=None, numerical_scale=1.0):
    """Returns fit parameters for an exponential decay in the input signal.

    Args:
        signal (1 dimensional np.ndarray): time series, beginning at decay onset
        sampling_rate (float): sampling rate in Hz
    """
    from scipy.optimize import curve_fit

    if sampling_rate is None and times is None:
        raise ValueError('pass either sampling_rate or times as keyword arg')

    if sampling_rate is not None:
        sampling_interval = 1 / sampling_rate
        n_samples = len(signal)
        end_time = n_samples * sampling_interval
        times = np.linspace(0, end_time, num=n_samples, endpoint=True)

    # TODO make sure input is not modified here. copy?
    signal = signal * numerical_scale

    # TODO constrain params somehow? for example, so scale stays positive
    popt, pcov = curve_fit(exp_decay, times, signal,
        p0=(1.8 * numerical_scale, 5.0, 0.0 * numerical_scale))

    # TODO is this correct to scale after converting variance to stddev?
    sigmas = np.sqrt(np.diag(pcov))
    sigmas[0] = sigmas[0] / numerical_scale
    # skipping tau, which shouldn't need to change (?)
    sigmas[2] = sigmas[2] / numerical_scale

    # TODO only keep this if signal is modified s.t. it affects calling fn.
    # in this case, maybe still just copy above?
    signal = signal / numerical_scale

    scale, tau, offset = popt
    return (scale / numerical_scale, tau, offset / numerical_scale), sigmas


def n_expected_repeats(df):
    """Returns expected # repeats given DataFrame w/ repeat_num col.
    """
    max_repeat = df.repeat_num.max()
    return max_repeat + 1


# TODO TODO could now probably switch to using block metadata in recording table
# (n_repeats should be in there)
# TODO move to project specific repo unless stimulus metadata can be
# meaningfully generalized
def missing_repeats(df, n_repeats=None):
    """
    Requires at least recording_from, comparison, name1, name2, and repeat_num
    columns. Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO n_repeats default to 3 or None?
    if n_repeats is None:
        # TODO or should i require input is merged w/ recordings for stimuli
        # data file paths and then just load for n_repeats and stuff?
        n_repeats = n_expected_repeats(df)

    # Expect repeats to include {0,1,2} for 3 repeat experiments.
    expected_repeats = set(range(n_repeats))

    repeat_cols = []
    opt_repeat_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_repeat_cols:
        if oc in df.columns:
            repeat_cols.append(oc)

    repeat_cols += [
        'recording_from',
        'comparison',
        'name1',
        'name2'#,
        #'log10_conc_vv1',
        #'log10_conc_vv2'
    ]
    # TODO some issue created by using float concs as a key?
    # TODO use odor ids instead?
    missing_repeat_dfs = []
    for g, gdf in df.groupby(repeat_cols):
        comparison_n_repeats = gdf.repeat_num.unique()

        no_extra_repeats = (gdf.repeat_num.value_counts() == 1).all()
        assert no_extra_repeats

        missing_repeats = [r for r in expected_repeats
            if r not in comparison_n_repeats]

        if len(missing_repeats) > 0:
            gmeta = gdf[repeat_cols].drop_duplicates().reset_index(drop=True)

        for r in missing_repeats:
            new_row = gmeta.copy()
            new_row['repeat_num'] = r
            missing_repeat_dfs.append(new_row)

    if len(missing_repeat_dfs) == 0:
        missing_repeats_df = \
            pd.DataFrame({r: [] for r in repeat_cols + ['repeat_num']})
    else:
        # TODO maybe merge w/ odor info so caller doesn't have to, if thats the
        # most useful for troubleshooting?
        missing_repeats_df = pd.concat(missing_repeat_dfs, ignore_index=True)

    missing_repeats_df.recording_from = \
        pd.to_datetime(missing_repeats_df.recording_from)

    # TODO should expected # blocks be passed in?

    return missing_repeats_df


def have_all_repeats(df, n_repeats=None):
    """
    Returns True if a recording has all blocks gsheet says it has, w/ full
    number of repeats for each. False otherwise.

    Requires at least recording_from, comparison, name1, name2, and repeat_num
    columns. Can also take prep_date, fly_num, thorimage_id.
    """
    missing_repeats_df = missing_repeats(df, n_repeats=n_repeats)
    if len(missing_repeats_df) == 0:
        return True
    else:
        return False


def missing_odor_pairs(df):
    """
    Requires at least recording_from, comparison, name1, name2 columns.
    Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO check that for each comparison, both A, B, and A+B are there
    # (3 combos of name1, name2, or whichever other odor ids)
    comp_cols = []
    opt_rec_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_rec_cols:
        if oc in df.columns:
            comp_cols.append(oc)

    comp_cols += [
        'recording_from',
        'comparison'
    ]

    odor_cols = [
        'name1',
        'name2'
    ]

    incomplete_comparison_dfs = []
    for g, gdf in df.groupby(comp_cols):
        comp_odor_pairs = gdf[odor_cols].drop_duplicates()
        if len(comp_odor_pairs) != 3:
            incomplete_comparison_dfs.append(gdf[comp_cols].drop_duplicates(
                ).reset_index(drop=True))

        # TODO generate expected combinations of name1,name2
        # TODO possible either odor not in db, in which case, would need extra
        # information to say which odor is actually missing... (would need
        # stimulus data)
        '''
        if len(missing_odor_pairs) > 0:
            gmeta = gdf[comp_cols].drop_duplicates().reset_index(drop=True)

        for r in missing_odor_pairs:
            new_row = gmeta.copy()
            new_row['repeat_num'] = r
            missing_odor_pair_dfs.append(new_row)
        '''

    if len(incomplete_comparison_dfs) == 0:
        incomplete_comparison_df = pd.DataFrame({r: [] for r in comp_cols})
    else:
        incomplete_comparison_df = \
            pd.concat(incomplete_comparison_dfs, ignore_index=True)

    incomplete_comparison_df.recording_from = \
        pd.to_datetime(incomplete_comparison_df.recording_from)

    return incomplete_comparison_df


def have_full_comparisons(df):
    """
    Requires at least recording_from, comparison, name1, name2 columns.
    Can also take prep_date, fly_num, thorimage_id.
    """
    # TODO docstring
    if len(missing_odor_pairs(df)) == 0:
        return True
    else:
        return False


def skipped_comparison_nums(df):
    # TODO doc
    """
    Requires at least recording_from and comparison columns.
    Can also take prep_date, fly_num, and thorimage_id.
    """
    rec_cols = []
    opt_rec_cols = [
        'prep_date',
        'fly_num',
        'thorimage_id'
    ]
    for oc in opt_rec_cols:
        if oc in df.columns:
            rec_cols.append(oc)

    rec_cols.append('recording_from')

    skipped_comparison_dfs = []
    for g, gdf in df.groupby(rec_cols):
        max_comp_num = gdf.comparison.max()
        min_comp_num = gdf.comparison.min()
        skipped_comp_nums = [x for x in range(min_comp_num, max_comp_num + 1)
            if x not in gdf.comparison]

        if len(skipped_comp_nums) > 0:
            gmeta = gdf[rec_cols].drop_duplicates().reset_index(drop=True)

        for c in skipped_comp_nums:
            new_row = gmeta.copy()
            new_row['comparison'] = c
            skipped_comparison_dfs.append(new_row)

    if len(skipped_comparison_dfs) == 0:
        skipped_comparison_df = pd.DataFrame({r: [] for r in
            rec_cols + ['comparison']})
    else:
        skipped_comparison_df = \
            pd.concat(skipped_comparison_dfs, ignore_index=True)

    # TODO move this out of each of these check fns, and put wherever this
    # columns is generated (in the way that required this cast...)
    skipped_comparison_df.recording_from = \
        pd.to_datetime(skipped_comparison_df.recording_from)

    return skipped_comparison_df


def no_skipped_comparisons(df):
    # TODO doc
    """
    Requires at least recording_from and comparison columns.
    Can also take prep_date, fly_num, and thorimage_id.
    """
    if len(skipped_comparison_nums(df)) == 0:
        return True
    else:
        return False


# TODO also check recording has as many blocks (in df / in db) as it's supposed
# to, given what the metadata + gsheet say


def drop_orphaned_presentations():
    # TODO only stuff that isn't also most recent response params?
    # TODO find presentation rows that don't have response row referring to them
    raise NotImplementedError


# TODO TODO maybe implement check fns above as wrappers around another fn that
# finds inomplete stuff? (check if len is 0), so that these fns can just wrap
# the same thing...
def drop_incomplete_presentations():
    raise NotImplementedError


def smooth_1d(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd
            integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman' flat window will produce a moving average
            smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth_1d(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of
    a string
    NOTE: length(output) != length(input), to correct this: return
    y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth_1d only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', " +
            "'hamming', 'bartlett', 'blackman'")

    # is this necessary?
    #s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    #y = np.convolve(w/w.sum(), s, mode='valid')
    # not sure what to change above to get this to work...

    y = np.convolve(w/w.sum(), x, mode='same')
    return y


# TODO finish translating. was directly translating matlab registration script
# to python.
"""
def motion_correct_to_tiffs(image_dir, output_dir):
    # TODO only read this if at least one motion correction would be run
    movie = thor.read_movie(image_dir)

    # TODO do i really want to basically just copy the matlab version?
    # opportunity for some refactoring?

    output_subdir = 'tif_stacks'

    _, thorimage_id = split(image_dir)

    rig_tif = join(output_dir, output_subdir, thorimage_id + '_rig.tif')
    avg_rig_tif = join(output_dir, output_subdir, 'AVG', 'rigid',
        'AVG{}_rig.tif'.format(thorimage_id))

    nr_tif = join(output_dir, output_subdir, thorimage_id + '_nr.tif')
    avg_nr_tif = join(output_dir, output_subdir, 'AVG', 'nonrigid',
        'AVG{}_nr.tif'.format(thorimage_id))

    need_rig_tif = not exist(rig_tif)
    need_avg_rig_tif = not exist(avg_rig_tif)
    need_nr_tif = not exist(nr_tif)
    need_avg_nr_tif = not exist(avg_nr_tif)

    if not (need_rig_tif or need_avg_rig_tif or need_nr_tif or need_avg_nr_tif):
        print('All registration already done.')
        return

    # Remy: this seems like it might just be reading in the first frame?
    ###Y = input_tif_path
    # TODO maybe can just directly use filename for python version though? raw
    # even?

    # rigid moco (normcorre)
    # TODO just pass filename instead of Y, and compute dimensions or whatever
    # separately, so that normcorre can (hopefully?) take up less memory
    if need_rig_tif:
        MC_rigid = MotionCorrection(Y)

        options_rigid = NoRMCorreSetParms('d1',MC_rigid.dims(1),
            'd2',MC_rigid.dims(2),
            'bin_width',50,
            'max_shift',15,
            'phase_flag', 1,
            'us_fac', 50,
            'init_batch', 100,
            'plot_flag', false,
            'iter', 2)

        # TODO so is nothing actually happening in parallel?
        ## rigid moco
        MC_rigid.motionCorrectSerial(options_rigid)  # can also try parallel
        # TODO which (if any) of these do i still want?
        MC_rigid.computeMean()
        MC_rigid.correlationMean()
        #####MC_rigid.crispness()
        print('normcorre done')

        ## plot shifts
        #plt.plot(MC_rigid.shifts_x)
        #plt.plot(MC_rigid.shifts_y)

        # save .tif
        M = MC_rigid.M
        M = uint16(M)
        tiffoptions.overwrite = true

        print(['saving tiff to ' rig_tif])
        saveastiff(M, rig_tif, tiffoptions)

    if need_avg_rig_tif:
        ##
        # save average image
        #AVG = single(mean(MC_rigid.M,3))
        AVG = single(MC_rigid.template)
        tiffoptions.overwrite = true

        print(['saving tiff to ' avg_rig_tif])
        saveastiff(AVG, avg_rig_tif, tiffoptions)

    if need_nr_tif:
        MC_nonrigid = MotionCorrection(Y)
        options_nonrigid = NoRMCorreSetParms('d1',MC_nonrigid.dims(1),
            'd2',MC_nonrigid.dims(2),
            'grid_size',[64,64],
            'mot_uf',4,
            'bin_width',50,
            'max_shift',[15 15],
            'max_dev',3,
            'us_fac',50,
            'init_batch',200,
            'iter', 2)

        MC_nonrigid.motionCorrectParallel(options_nonrigid)
        MC_nonrigid.computeMean()
        MC_nonrigid.correlationMean()
        MC_nonrigid.crispness()
        print('non-rigid normcorre done')

        # save .tif
        M = uint16(MC_nonrigid.M)
        tiffoptions.overwrite  = true
        print(['saving tiff to ' nr_tif])
        saveastiff(M, nr_tif, tiffoptions)

    if need_avg_nr_tif:
        # TODO flag to disable saving this average
        #AVG = single(mean(MC_nonrigid.M,3))
        AVG = single(MC_nonrigid.template)
        tiffoptions.overwrite = true
        print(['saving tiff to ' avg_nr_tif])
        saveastiff(AVG, avg_nr_tif, tiffoptions)

    raise NotImplementedError
"""

def cell_ids(df):
    """Takes a DataFrame with 'cell' in MultiIndex or columns to unique values.
    """
    if 'cell' in df.index.names:
        return df.index.get_level_values('cell').unique().to_series()
    elif 'cell' in df.columns:
        cids = pd.Series(data=df['cell'].unique(), name='cell')
        cids.index.name = 'cell'
        return cids
    else:
        raise ValueError("'cell' not in index or columns of DataFrame")


# TODO move to olf / delete
def format_odor_conc(name, log10_conc):
    """Takes `str` odor name and log10 concentration to a formatted `str`.
    """
    if log10_conc is None:
        return name
    else:
        # TODO tex formatting for exponent
        #return r'{} @ $10^{{'.format(name) + '{:.2f}}}$'.format(log10_conc)
        return '{} @ $10^{{{:.2f}}}$'.format(name, log10_conc)


# TODO move to olf / delete
def format_mixture(*args):
    """Returns `str` representing 2-component odor mixture.

    Input can be any of:
    - 2 `str` names
    - 2 names and concs (n1, n2, c1, c2)
    - a pandas.Series / dict with keys `name1`, `name2`, and (optionally)
      `log10_concvv<1/2>`
    """
    log10_c1 = None
    log10_c2 = None
    if len(args) == 2:
        n1, n2 = args
    elif len(args) == 4:
        n1, n2, log10_c1, log10_c2 = args
    elif len(args) == 1:
        row = args[0]

        # TODO maybe refactor to use this fn in viz.plot_odor_corrs fn too
        def single_var_with_prefix(prefix):
            single_var = None
            for v in row.keys():
                if type(v) is str and v.startswith(prefix):
                    if single_var is not None:
                        raise ValueError(f'multiple vars w/ prefix {prefix}')
                    single_var = v

            if single_var is None:
                raise KeyError(f'no vars w/ prefix {prefix}')

            return single_var
        #

        n1 = row[single_var_with_prefix('name1')]
        try:
            n2 = row[single_var_with_prefix('name2')]
        except KeyError:
            n2 = None

        # TODO maybe also use prefix fn here?
        if 'log10_conc_vv1' in row:
            log10_c1 = row['log10_conc_vv1']
            if n2 is not None:
                log10_c2 = row['log10_conc_vv2']
    else:
        raise ValueError('incorrect number of args')

    if n1 == 'paraffin':
        title = format_odor_conc(n2, log10_c2)
    elif n2 == 'paraffin' or n2 == 'no_second_odor' or n2 is None:
        title = format_odor_conc(n1, log10_c1)
    else:
        title = '{} + {}'.format(
            format_odor_conc(n1, log10_c1),
            format_odor_conc(n2, log10_c2)
        )

    return title


# TODO move to olf / delete
def split_odor_w_conc(row_or_str):
    try:
        odor_w_conc = row_or_str.odor_w_conc
        include_other_row_data = True
    except AttributeError:
        assert type(row_or_str) is str
        odor_w_conc = row_or_str
        include_other_row_data = False

    parts = odor_w_conc.split('@')
    assert len(parts) == 1 or len(parts) == 2
    if len(parts) == 1:
        log10_conc = 0.0
    else:
        log10_conc = float(parts[1])

    ret = {'name': parts[0].strip(), 'log10_conc_vv': log10_conc}
    if include_other_row_data:
        ret.update(row_or_str.to_dict())

    # TODO maybe only return series if include_other_row_data (rename if),
    # tuple/dict otherwise?
    return pd.Series(ret)


def format_keys(date, fly, *other_keys):
    date = format_date(date)
    fly = str(int(fly))
    others = [str(k) for k in other_keys]
    return '/'.join([date] + [fly] + others)


# TODO rename to be inclusive of cases other than pairs
def pair_ordering(comparison_df):
    """Takes a df w/ name1 & name2 to a dict of their tuples to order int.

    Order integers start at 0 and do not skip any numbers.
    """
    # TODO maybe assert only 3 combinations of name1/name2
    pairs = [(x.name1, x.name2) for x in
        comparison_df[['name1','name2']].drop_duplicates().itertuples()]

    # Will define the order in which odor pairs will appear, left-to-right,
    # in subplots.
    ordering = dict()

    # TODO maybe check that it's the second element specifically, since right
    # now, it's only cause paraffin is abbreviated to pfo (for name1 col)
    # that complex-mixture experiments go into first branch...
    has_paraffin = [p for p in pairs if 'paraffin' in p]
    if len(has_paraffin) == 0:
        import chemutils as cu
        assert {x[1] for x in pairs} == {'no_second_odor'}
        odors = [p[0] for p in pairs]

        # TODO change how odorset is identified so it can fail if none should be
        # detected / return None or something, then call back to just sorting
        # the odor names here, if no odor set name can be identified
        # (do we also want to support some case where original_name1 is defined
        # but the odorset name isn't necessarily?)
        if 'original_name1' in comparison_df.columns:
            original_name_order = df_to_odor_order(comparison_df)
            o2n = comparison_df[['original_name1','name1']].drop_duplicates(
                ).set_index('original_name1').name1
            # TODO maybe don't assume 'no_second_odor' like this (& below)?
            ordering = {(v, 'no_second_odor'): i for i, v in
                enumerate(o2n[original_name_order])}
        else:
            # TODO also support case where there isn't something we want to
            # stick at the end like this, for Matt's case
            last = None
            for o in odors:
                if cu.odor_is_mix(o):
                    if last is None:
                        last = o
                    else:
                        raise ValueError('multiple mixtures in odors to order')
            assert last is not None, 'expected a mix'
            ordering[(last, 'no_second_odor')] = len(odors) - 1

            i = 0
            for o in sorted(odors):
                if o == last:
                    continue
                ordering[(o, 'no_second_odor')] = i
                i += 1
    else:
        no_pfo = [p for p in pairs if 'paraffin' not in p]
        if len(no_pfo) < 1:
            raise ValueError('All pairs for this comparison had paraffin.' +
                ' Analysis error? Incomplete recording?')

        assert len(no_pfo) == 1
        last = no_pfo[0]
        ordering[last] = 2

        for i, p in enumerate(sorted(has_paraffin,
            key=lambda x: x[0] if x[1] == 'paraffin' else x[1])):

            ordering[p] = i

    # Checks that we order integers start at zero and don't skip anything.
    # Important for some ways of using them (e.g. to index axes array).
    assert {x for x in ordering.values()} == {x for x in range(len(ordering))}

    return ordering


# TODO TODO call this in gui / factor into viz.plot_odor_corrs (though it would
# require accesss to df...) and call that there
def add_missing_odor_cols(df, missing_df):
    """
    """
    # TODO maybe check cols are indeed describing odors in missing_df?

    # TODO delete / change note to be relevant here. copied from original
    # implementation in gui
    # This + pivot_table w/ dropna=False won't work until this bug:
    # https://github.com/pandas-dev/pandas/issues/18030 is fixed.
    '''
    window_trial_means = pd.concat([window_trial_means,
        missing_dff.set_index(window_trial_means.index.names
        ).df_over_f
    ])
    '''
    missing_dff = df[df.df_over_f.isnull()][
        missing_df.columns.names + ['cell', 'df_over_f']
    ]
    # Hack to workaround pivot NaN behavior bug mentioned above.
    assert missing_dff.df_over_f.isnull().all()
    missing_dff.df_over_f = missing_dff.df_over_f.fillna(0)
    extra_cols = missing_dff.pivot_table(
        index='cell', values='df_over_f',
        columns=['name1','name2','repeat_num','order']
    )
    extra_cols.iloc[:] = np.nan

    assert (len(missing_df.columns.drop_duplicates()) ==
        len(missing_df.columns))

    missing_df = pd.concat([missing_df, extra_cols], axis=1)

    assert (len(missing_df.columns.drop_duplicates()) ==
        len(missing_df.columns))

    missing_df.sort_index(axis='columns', inplace=True)
    # end of the hack to workaround pivot NaN behavior

    return missing_df


# TODO test when ax actually is passed in now that I made it a kwarg
# (also works as optional positional arg, right?)
# TODO rename to be agnostic to fact mpl is used to implement it + to be more
# descriptive of function
def closed_mpl_contours(footprint, ax=None, if_multiple='err', **kwargs):
    # TODO doc / delete
    """
    Args:
        if_multiple (str): 'take_largest'|'join'|'err'
        **kwargs: passed through to matplotlib `ax.contour` call
    """
    dims = footprint.shape
    padded_footprint = np.zeros(tuple(d + 2 for d in dims))
    padded_footprint[tuple(slice(1,-1) for _ in dims)] = footprint

    # TODO delete
    #fig = plt.figure()
    #
    if ax is None:
        ax = plt.gca()

    mpl_contour = ax.contour(padded_footprint > 0, [0.5], **kwargs)
    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1

    paths = mpl_contour.collections[0].get_paths()

    if len(paths) != 1:
        if if_multiple == 'err':
            raise RuntimeError('multiple disconnected paths in one footprint')

        elif if_multiple == 'take_largest':
            largest_sum = 0
            largest_idx = 0
            total_sum = 0
            for p in range(len(paths)):
                path = paths[p]

                # TODO TODO TODO maybe replace mpl stuff w/ cv2 drawContours?
                # (or related...) (fn now in here as contour2mask)
                mask = np.ones_like(footprint, dtype=bool)
                for x, y in np.ndindex(footprint.shape):
                    # TODO TODO not sure why this seems to be transposed, but it
                    # does (make sure i'm not doing something wrong?)
                    if path.contains_point((x, y)):
                        mask[x, y] = False
                # Places where the mask is False are included in the sum.
                path_sum = MaskedArray(footprint, mask=mask).sum()
                # TODO maybe check that sum of all path_sums == footprint.sum()?
                # seemed there were some paths w/ 0 sum... cnmf err?
                '''
                print('mask_sum:', (~ mask).sum())
                print('path_sum:', path_sum)
                print('regularly masked sum:', footprint[(~ mask)].sum())
                plt.figure()
                plt.imshow(mask)
                plt.figure()
                plt.imshow(footprint)
                plt.show()
                import ipdb; ipdb.set_trace()
                '''
                if path_sum > largest_sum:
                    largest_sum = path_sum
                    largest_idx = p

                total_sum += path_sum
            footprint_sum = footprint.sum()
            # TODO float formatting / some explanation as to what this is
            print('footprint_sum:', footprint_sum)
            print('total_sum:', total_sum)
            print('largest_sum:', largest_sum)
            # TODO is this only failing when stuff is overlapping?
            # just merge in that case? (wouldn't even need to dilate or
            # anything...) (though i guess then the inequality would go the
            # other way... is it border pixels? just ~dilate by one?)
            # TODO fix + uncomment
            ######assert np.isclose(total_sum, footprint_sum)
            path = paths[largest_idx]

        elif if_multiple == 'join':
            raise NotImplementedError
    else:
        path = paths[0]

    # TODO delete
    #plt.close(fig)
    #

    contour = path.vertices
    # Correct index change caused by padding.
    return contour - 1


# TODO worth having min/max as inputs, so that maybe can use vals from
# either scene or template for the other? i guess point of baselining
# is to avoid need for stuff like that...
def baselined_normed_u8(img):
    u8_max = 255
    # TODO maybe convert to float64 or something first before some operations,
    # to minimize rounding errs?
    baselined = img - img.min()
    normed = baselined / baselined.max()
    return (u8_max * normed).astype(np.uint8)


# TODO refactor this behind a color=True kwarg baselined_normed_u8 above?
def u8_color(draw_on):
    # TODO figure out why background looks lighter here than in other
    # imshows of same input (w/o converting manually)
    draw_on = draw_on - np.min(draw_on)
    draw_on = draw_on / np.max(draw_on)
    cmap = plt.get_cmap('gray') #, lut=256)
    # (throwing away alpha coord w/ last slice)
    draw_on = np.round((cmap(draw_on)[:, :, :3] * 255)).astype(np.uint8)
    return draw_on


def template_match(scene, template, method_str='cv2.TM_CCOEFF', hist=False,
    debug=False):

    import cv2

    vscaled_scene = baselined_normed_u8(scene)
    # TODO TODO maybe template should only be scaled to it's usual fraction of
    # max of the scene? like scaled both wrt orig_scene.max() / max across all
    # images?
    vscaled_template = baselined_normed_u8(template)

    if debug:
        # To check how much conversion to u8 (necessary for cv2 template
        # matching) has reduced the number of pixel levels.
        scene_levels = len(set(scene.flat))
        vs_scene_levels = len(set(vscaled_scene.flat))
        template_levels = len(set(template.flat))
        vs_template_levels = len(set(vscaled_template.flat))
        print(f'Number of scene levels BEFORE scaling: {scene_levels}')
        print(f'Number of scene levels AFTER scaling: {vs_scene_levels}')
        print(f'Number of template levels BEFORE scaling: {template_levels}')
        print(f'Number of template levels AFTER scaling: {vs_template_levels}')

        # So you can see that the relative dimensions and scales of each of
        # these seems reasonable.
        def compare_template_and_scene(template, scene, suptitle,
            same_scale=True):

            smin = scene.min()
            smax = scene.max()
            tmin = template.min()
            tmax = template.max()

            print(f'\n{suptitle}')
            print('scene shape:', scene.shape)
            print('template shape:', template.shape)
            print('scene min:', smin)
            print('scene max:', smax)
            print('template min:', tmin)
            print('template max:', tmax)

            # Default, for this fig at least seemed to be (6.4, 4.8)
            # This has the same aspect ratio.
            fh = 10
            fw = (1 + 1/3) * fh
            fig, axs = plt.subplots(ncols=3, figsize=(fw, fh))

            xlim = (0, max(scene.shape[0], template.shape[0]) - 1)
            ylim = (0, max(scene.shape[1], template.shape[1]) - 1)

            if same_scale:
                vmin = min(smin, tmin)
                vmax = max(smax, tmax)
            else:
                vmin = None
                vmax = None

            ax = axs[0]
            sim = ax.imshow(scene, vmin=vmin, vmax=vmax)
            ax.set_title('scene')

            ax = axs[1]
            tim = ax.imshow(template, vmin=vmin, vmax=vmax)
            ax.set_title('template (real scale)')

            ax = axs[2]
            btim = ax.imshow(template, vmin=vmin, vmax=vmax)
            ax.set_title('template (blown up)')

            # https://stackoverflow.com/questions/31006971
            plt.setp(axs, xlim=xlim, ylim=ylim)

            ax.set_xlim((0, template.shape[0] - 1))
            ax.set_ylim((0, template.shape[0] - 1))

            fig.suptitle(suptitle)

            if same_scale:
                # l, b, w, h
                cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cb = fig.colorbar(sim, cax=cax)
                cb.set_label('shared')
                fig.subplots_adjust(right=0.8)
            else:
                # l, b, w, h
                cax1 = fig.add_axes([0.75, 0.15, 0.025, 0.7])
                cb1 = fig.colorbar(sim, cax=cax1)
                cb1.set_label('scene')

                cax2 = fig.add_axes([0.85, 0.15, 0.025, 0.7])
                cb2 = fig.colorbar(tim, cax=cax2)
                cb2.set_label('template')

                fig.subplots_adjust(right=0.7)

            bins = 50
            fig, axs = plt.subplots(ncols=2, sharex=same_scale)
            ax = axs[0]
            shistvs, sbins, _ = ax.hist(scene.flat, bins=bins, log=True)
            ax.set_title('scene')
            ax.set_ylabel('Frequency (a.u.)')
            ax = axs[1]
            thitvs, tbins, _ = ax.hist(template.flat, bins=bins, log=True)
            ax.set_title('template')
            fig.suptitle(f'{suptitle}\npixel value distributions ({bins} bins)')
            fig.subplots_adjust(top=0.85)

        compare_template_and_scene(template, scene, 'original',
            same_scale=False
        )
        compare_template_and_scene(vscaled_template, vscaled_scene,
            'baselined + scaled'
        )
        print('')
        hist = True

    method = eval(method_str)
    res = cv2.matchTemplate(vscaled_scene, vscaled_template, method)

    # b/c for sqdiff[_normed], find minima. for others, maxima.
    if 'SQDIFF' in method_str:
        res = res * -1

    if hist:
        fh = plt.figure()
        plt.hist(res.flatten())
        plt.title('Matching output values ({})'.format(method_str))

    return res


def euclidean_dist(v1, v2):
    # Without the conversions to float 64 (or at least something else signed),
    # uint inputs lead to wraparound -> big distances occasionally.
    return np.linalg.norm(np.array(v1).astype(np.float64) -
        np.array(v2).astype(np.float64)
    )


# TODO TODO TODO try updating to take max of two diff match images,
# created w/ different template scales (try a smaller one + existing),
# and pack appropriate size at each maxima.
# TODO make sure match criteria is comparable across scales (one threshold
# ideally) (possible? using one of normalized metrics sufficient? test this
# on fake test data?)
def greedy_roi_packing(match_images, ds, radii_px, thresholds=None, ns=None,
    exclusion_radius_frac=0.7, min_dist2neighbor_px=15, min_neighbors=3,
    exclusion_mask=None,
    draw_on=None, debug=False, draw_bboxes=True, draw_circles=True,
    draw_nums=True, multiscale_strategy='one_order', match_value_weights=None,
    radii_px_ps=None, scale_order=None, subpixel=False, _src_img_shape=None,
    _show_match_images=False, _show_packing_constraints=False, _show_fit=True,
    _initial_single_threshold=None):
    """
    Args:
    match_images (np.ndarray / iterable of same): 2-dimensional array of match
    value higher means better match of that point to template.

        Shape is determined by the number of possible offsets of the template in
        the original image, so it is smaller than the original image in each
        dimension. As an example, for a 3x3 image and a 2x2 template, the
        template can occupy 4 possible positions in the 3x3 image, and the match
        image will be 2x2.

    ds (int / iterable of same): integer width (and height) of square template.
        related to radius, but differ by margin set outside.

    radii_px (int / iterable of same): radius of cell in pixels.

    exclusion_radius_frac (float): approximately 1 - the fraction of two ROI
        radii that are allowed to overlap.
    """
    # TODO move drawing fns for debug to mpl and remove this if not gonna
    # use for constraints here
    import cv2
    #
    # Use of either this or KDTree seem to cause pytest ufunc size changed
    # warning (w/ pytest at least), though it should be harmless.
    from scipy.spatial import cKDTree

    if subpixel is True:
        raise NotImplementedError

    if thresholds is None and ns is None:
        raise ValueError('specify either thresholds or ns')

    if not ((ns is None and thresholds is not None) or
            (ns is not None and thresholds is None)):
        raise ValueError('only specify either thresholds or ns')

    # For multiscale matching, we require (at lesat) multiple radii, so we test
    # whether it is iterable to determine if we should be using multiscale
    # matching.
    try:
        iter(radii_px)

        if len(radii_px) == 1:
            multiscale = False
        else:
            assert len(set(ds)) == len(ds)
            assert len(set(radii_px)) == len(radii_px)
            multiscale = True

    except TypeError:
        multiscale = False
        # also check most other things are NOT iterable in this case?

        match_images = [match_images]
        ds = [ds]
        radii_px = [radii_px]

    if ns is None:
        total_n = None
        # TODO maybe delete this test and force thresholds (if-specified)
        # to have same length. useless if one threshold is never gonna work.
        try:
            iter(thresholds)
            # If we have multiple thresholds, we must have as many
            # as the things above.
            assert len(thresholds) == len(radii_px)
        except TypeError:
            thresholds = [thresholds] * len(radii_px)

    elif thresholds is None:
        try:
            iter(ns)
            # TODO want this behavior ever? maybe delete try/except...
            # Here, we are specify how many of each size we are looking for.
            assert len(ns) == len(radii_px)
            if len(ns) == 1:
                total_n = ns[0]
                ns = None
            else:
                total_n = None
        except TypeError:
            # Here, we specify a target number of cells of any size to find.
            total_n = ns
            ns = None

    if multiscale:
        n_scales = len(radii_px)
        assert len(match_images) == n_scales
        assert len(ds) == n_scales

        if multiscale_strategy != 'one_order':
            assert match_value_weights is None, ('match_value_weights are only '
                "meaningful in multiscale_strategy='one_order' case, because "
                'they do not change match ordering within a single match scale.'
                ' They only help make one ordering across matching scales.'
            )

        if multiscale_strategy != 'random':
            assert radii_px_ps is None, ('radii_px_ps is only meaningful in '
                "multiscale_strategy='random' case"
            )

        if multiscale_strategy != 'fixed_scale_order':
            assert scale_order is None, ('scale_order is only meaningful in '
                "multiscale_strategy='fixed_scale_order' case"
            )

        if multiscale_strategy == 'one_order':
            # Can still be None here, that just implies that match values
            # at different scales will be sorted into one order with no
            # weighting.
            if match_value_weights is not None:
                assert len(match_value_weights) == n_scales

            # could also accept callable for each element, if a fn (rather than
            # linear scalar) would be more useful to make match values
            # comparable across scales (test for it later, at time-to-use)

        elif multiscale_strategy == 'random':
            assert radii_px_ps is not None
            assert np.isclose(np.sum(radii_px_ps), 1)
            assert all([r >= 0 and r <= 1 for r in radii_px_ps])
            if any([r == 0 or r == 1 for r in radii_px_ps]):
                warnings.warn('Some elements of radii_px_ps were 0 or 1. '
                    "This means using multiscale_strategy='random' may not make"
                    ' sense.'
                )

        elif multiscale_strategy == 'fixed_scale_order':
            # could just take elements from other iterables in order passed
            # in... just erring on side of being explicit
            assert scale_order is not None
            assert len(set(scale_order)) == len(scale_order)
            for i in scale_order:
                try:
                    radii_px[i]
                except IndexError:
                    raise ValueError('scale_order had elements not usable to '
                        'index scales'
                    )

        else:
            raise ValueError(f'multiscale_strategy {multiscale_strategy} not '
                'recognized'
            )

        # Can not assert all match_images have the same shape, because d
        # affects shape of match image (as you can see from line inverting
        # this dependence to calculate orig_shape, below)

    else:
        n_scales = 1
        assert match_value_weights is None
        assert radii_px_ps is None
        assert scale_order is None
        # somewhat tautological. could delete.
        if thresholds is None:
            assert total_n is not None

    # TODO optimal non-greedy alg for this problem? (maximize weight of
    # match_image summed across all assigned ROIs)

    # TODO do away with this copying if not necessary
    # (just don't want to mutate inputs without being clear about it)
    # (multiplication by match_value_weights below)
    match_images = [mi.copy() for mi in match_images]
    orig_shapes = set()
    for match_image, d in zip(match_images, ds):
        # Working through example w/ 3x3 src img and 2x2 template -> 2x2 match
        # image in docstring indicates necessity for - 1 here.
        orig_shape = tuple(x + d - 1 for x in match_image.shape)
        orig_shapes.add(orig_shape)

    assert len(orig_shapes) == 1
    orig_shape = orig_shapes.pop()
    if _src_img_shape is not None:
        assert orig_shape == _src_img_shape
        del _src_img_shape

    if draw_on is not None:
        # if this fails, just fix shape comparison in next assertion and
        # then delete this assert
        assert len(draw_on.shape) == 2
        assert draw_on.shape == orig_shape

        draw_on = u8_color(draw_on)
        # upsampling just so cv2 drawing functions look better
        ups = 4
        draw_on = cv2.resize(draw_on,
            tuple([ups * x for x in draw_on.shape[:2]])
        )

    if match_value_weights is not None:
        for i, w in enumerate(match_value_weights):
            match_images[i] = match_images[i] * w

    if debug and _show_match_images:
        # wanted these as subplots w/ colorbar besides each, but colorbars
        # seemed to want to go to the side w/ the simplest attempt
        ncols = 3
        nrows = n_scales % ncols + 1
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows)

        if not multiscale or multiscale_strategy == 'one_order':
            vmin = min([mi.min() for mi in match_images])
            vmax = max([mi.max() for mi in match_images])
            same_scale = True
        else:
            vmin = None
            vmax = None
            same_scale = False

        for i, (ax, td, match_image) in enumerate(zip(
            axs.flat, ds, match_images)):

            to_show = match_image.copy()
            if thresholds is not None:
                to_show[to_show < thresholds[i]] = np.nan

            im = ax.imshow(to_show)
            if not same_scale:
                # https://stackoverflow.com/questions/23876588
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax)

            title = f'({td}x{td} template)'
            if match_value_weights is not None:
                w = match_value_weights[i]
                title += f' (weight={w:.2f})'
            ax.set_title(title)

        if same_scale:
            # l, b, w, h
            cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('match value')
            fig.subplots_adjust(right=0.8)

        title = 'template matching metric at each template offset'
        if thresholds is not None:
            title += '\n(white space is pixels below corresponding threshold)'
        fig.suptitle(title)
        # TODO may want to decrease wspace if same_scale
        fig.subplots_adjust(wspace=0.7)

    all_flat_vals = [mi.flatten() for mi in match_images]

    if debug:
        print('thresholds for each scale:', thresholds)
        print('min (possibly scaled) match val at each scale:',
            np.array([vs.min() for vs in all_flat_vals])
        )
        print('max (possibly scaled) match val at each scale:',
            np.array([vs.max() for vs in all_flat_vals])
        )
        print('match_value_weights:', match_value_weights)

    if not multiscale or multiscale_strategy == 'one_order':
        # TODO TODO TODO need to sort flat_vals into one order, while
        # maintaining info about which match_image (index) a particular
        # value came from
        # how to do this while also thresholding each one?

        all_vals = []
        all_scale_and_flat_indices = []
        for i, (fv, thresh) in enumerate(zip(all_flat_vals, thresholds)):
            if thresholds is not None:
                flat_idx_at_least_thresh = np.argwhere(fv > thresh)[:,0]
                vals_at_least_thresh = fv[flat_idx_at_least_thresh]
            else:
                flat_idx_at_least_thresh = np.arange(len(fv))
                vals_at_least_thresh = fv

            if debug:
                thr_frac = len(flat_idx_at_least_thresh) / len(fv)
                print(f'scale {i} fraction of (scaled) match values above'
                    ' threshold:', thr_frac
                )

                '''
                # TODO delete after figuring out discrepancy
                thr_fracs = [0.001252, 0.0008177839, 0.00087937249]
                assert np.isclose(thr_frac, thr_fracs[i])

                #print(len(flat_idx_at_least_thresh))
                #import ipdb; ipdb.set_trace()
                #
                '''
                # TODO TODO maybe find range of weights that produce same
                # fraction above thresholds, and see if somewhere in that range
                # is a set of weights that also leads to a global ordering that
                # behaves as I want?

                # TODO delete if not gonna finish
                if _initial_single_threshold is not None:
                    t0 = _initial_single_threshold

                    '''
                    if match_value_weights is not None:
                        # Undoing previous multiplication by weight.
                        w = match_value_weights[i]
                        orig_match_image = match_images[i] / w
                    orig 

                    # TODO TODO TODO fit(?) to find match value weight that
                    # produces same fraction of pixels above threshold
                    import ipdb; ipdb.set_trace()
                    '''
                #

            # TODO maybe just store ranges of indices in concatenated
            # flat_idx... that correspond to each source img?
            src_img_idx = np.repeat(i, len(flat_idx_at_least_thresh))

            scale_and_flat_indices = np.stack(
                [src_img_idx, flat_idx_at_least_thresh]
            )
            all_scale_and_flat_indices.append(scale_and_flat_indices)
            all_vals.append(vals_at_least_thresh)

        all_scale_and_flat_indices = np.concatenate(all_scale_and_flat_indices,
            axis=1
        )

        all_vals = np.concatenate(all_vals)
        # Reversing order so indices corresponding to biggest element is first,
        # and so on, decreasing.
        one_order_indices = np.argsort(all_vals)[::-1]

        '''
        # TODO delete
        #np.set_printoptions(threshold=sys.maxsize)
        out = all_scale_and_flat_indices.T[one_order_indices]
        print('all_scale_and_flat_indices.shape:',
            all_scale_and_flat_indices.shape
        )
        print('one_order_indices.shape:', one_order_indices.shape)

        print('sorted match values:')
        print(all_vals[one_order_indices])

        nlines = 20
        head = out[:nlines]
        tail = out[-nlines:]
        print('head:')
        print(head)
        print('tail:')
        print(tail)

        chead = np.array([[     2, 120520],
               [     1, 108599],
               [     0, 125250],
               [     2, 120521],
               [     2, 120029],
               [     2, 120519],
               [     2, 121011],
               [     2, 120030],
               [     2, 121012],
               [     2, 120028],
               [     2, 121010],
               [     1, 108600],
               [     1, 109096],
               [     1, 108598],
               [     1, 108102],
               [     0, 125750],
               [     0, 125249],
               [     0, 124750],
               [     0, 125251],
               [     1, 124002]])

        ctail = np.array([[     0, 108759],
               [     0, 112252],
               [     0, 111259],
               [     0, 112257],
               [     0, 125723],
               [     0, 124223],
               [     0, 128231],
               [     0, 121728],
               [     0, 128228],
               [     0, 124236],
               [     0, 125736],
               [     0, 121731],
               [     0, 128227],
               [     0, 126236],
               [     0, 126223],
               [     0, 121732],
               [     0, 123723],
               [     0, 128232],
               [     0, 121727],
               [     0, 123736]])

        try:
            assert np.array_equal(chead, head)
            assert np.array_equal(ctail, tail)
        except AssertionError:
            print('arrays did not match')
            print('correct versions (from specific thresholds):')
            print('correct head:')
            print(chead)
            print('correct tail:')
            print(ctail)
            import ipdb; ipdb.set_trace()
        #
        '''

        def match_iter_fn():
            for scale_idx, match_img_flat_idx in all_scale_and_flat_indices.T[
                one_order_indices]:

                match_image = match_images[scale_idx]
                match_pt = np.unravel_index(match_img_flat_idx,
                    match_image.shape
                )
                yield scale_idx, match_pt

    else:
        all_matches = []
        for i, match_image in enumerate(match_images):
            flat_vals = all_flat_vals[i]
            sorted_flat_indices = np.argsort(flat_vals)
            if thresholds is not None:
                idx = np.searchsorted(flat_vals[sorted_flat_indices],
                    thresholds[i]
                )
                sorted_flat_indices = sorted_flat_indices[idx:]
                del idx

            # Reversing order so indices corresponding to biggest element is
            # first, and so on, decreasing.
            sorted_flat_indices = sorted_flat_indices[::-1]
            matches = np.unravel_index(sorted_flat_indices, match_image.shape)
            all_matches.append(matches)

        if multiscale_strategy == 'fixed_scale_order':
            def match_iter_fn():
                for scale_idx in scale_order:
                    matches = all_matches[scale_idx]
                    for match_pt in zip(*matches):
                        yield scale_idx, match_pt

        elif multiscale_strategy == 'random':
            def match_iter_fn():
                per_scale_last_idx = [0] * n_scales
                scale_ps = radii_px_ps
                while True:
                    scale_idx = np.random.choice(n_scales, p=scale_ps)
                    matches = all_matches[scale_idx]

                    if all([last >= len(matches[0]) for last, matches in
                        zip(per_scale_last_idx, all_matches)]):

                        # This should end the generator's iteration.
                        return

                    # Currently just opting to retry sampling when we
                    # got something for which we have already exhausted all
                    # matches, rather than changing probabilities and choices.
                    if per_scale_last_idx[scale_idx] >= len(matches[0]):
                        continue

                    match_idx = per_scale_last_idx[scale_idx]
                    match_pt = tuple(m[match_idx] for m in matches)

                    per_scale_last_idx[scale_idx] += 1

                    yield scale_idx, match_pt

    match_iter = match_iter_fn()

    # TODO and any point to having subpixel circles anyway?
    # i.e. will packing decisions ever differ from those w/ rounded int
    # circles (and then also given that my ijroi currently only supports
    # saving non-subpixel rois...)?

    claimed = []
    center2radius = dict()

    total_n_found = 0
    roi_centers = []
    # roi_ prefix here is to disambiguate this from radii_px input, which
    # describes radii of various template scales to use for matching, but
    # NOT the radii of the particular matched ROI outputs.
    roi_radii_px = []

    if ns is not None:
        n_found_per_scale = [0] * n_scales

    max_exclusion_radius_px = max(exclusion_radius_frac * r for r in radii_px)
    scale_info_printed = [False] * n_scales
    for scale_idx, pt in match_iter:
        if total_n is not None:
            if total_n_found >= total_n:
                break

        elif ns is not None:
            if all([n_found >= n for n_found, n in zip(n_found_per_scale, ns)]):
                break

            if n_found_per_scale[scale_idx] >= ns[scale_idx]:
                continue

        d = ds[scale_idx]
        offset = d / 2
        center = (pt[0] + offset, pt[1] + offset)
        del offset

        if exclusion_mask is not None:
            if not exclusion_mask[tuple(int(round(v)) for v in center)]:
                continue

        radius_px = radii_px[scale_idx]
        exclusion_radius_px = radius_px * exclusion_radius_frac
        if debug:
            if not scale_info_printed[scale_idx]:
                print('template d:', d)
                print('radius_px:', radius_px)
                print('exclusion_radius_frac:', exclusion_radius_frac)
                print('exclusion_radius_px:', exclusion_radius_px)
                scale_info_printed[scale_idx] = True

        # Ideally I'd probably use a data structure that doesn't need to
        # be rebuilt each time (and k-d trees in general don't, but
        # scipy's doesn't support that (perhaps b/c issues w/ accumulating
        # rebalancing costs?), nor do they seem to offer spatial
        # structures that do)
        if len(claimed) > 0:
            tree = cKDTree(claimed)
            # (would need to relax if supporting 3d)
            assert tree.m == 2
            # TODO tests to check whether this is right dist bound
            # ( / 2 ?)
            dists, locs = tree.query(center,
                distance_upper_bound=max_exclusion_radius_px * 2
            )
            # Docs say this indicates no neighbors found.
            if locs != tree.n:
                try:
                    len(dists)
                except:
                    dists = [dists]
                    locs = [locs]

                conflict = False
                for dist, neighbor_idx in zip(dists, locs):
                    # TODO TODO any way to add metadata to tree element to avoid
                    # this lookup? (+ dist bound above)
                    neighbor_r = center2radius[tuple(tree.data[neighbor_idx])]
                    # We already counted the radius about the tentative
                    # new ROI, but that assumes all neighbors are just points.
                    # This prevents small ROIs from being placed inside big
                    # ones.
                    # TODO check these two lines
                    dist -= neighbor_r * exclusion_radius_frac
                    if dist <= exclusion_radius_px:
                        conflict = True
                        break
                if conflict:
                    continue

        total_n_found += 1
        roi_centers.append(center)
        roi_radii_px.append(radius_px)

        if draw_on is not None:
            draw_pt = (ups * pt[0], ups * pt[1])[::-1]
            draw_c = (
                int(round(ups * center[0])),
                int(round(ups * center[1]))
            )[::-1]

            # TODO factor this stuff out into post-hoc drawing fn, so that
            # roi filters in here can exclude stuff? or maybe just factor out
            # the filtering stuff anyway?

            if draw_bboxes:
                cv2.rectangle(draw_on, draw_pt,
                    (draw_pt[0] + ups * d, draw_pt[1] + ups * d), (0,0,255), 2
                )

            # TODO maybe diff colors for diff scales? (random or from kwarg)
            if draw_circles:
                cv2.circle(draw_on, draw_c, int(round(ups * radius_px)),
                    (255,0,0), 2
                )

            if draw_nums:
                cv2.putText(draw_on, str(total_n_found), draw_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
                )

        claimed.append(center)
        center2radius[tuple(center)] = radius_px

    '''
    if debug and _show_packing_constraints:
        title = 'greedy_roi_packing overlap exlusion mask'
        viz.imshow(claimed, title)
    '''

    if debug and draw_on is not None and _show_fit:
        viz.imshow(draw_on, 'greedy_roi_packing fit')

    # TODO also use kdtree for this step
    if not min_neighbors:
        filtered_roi_centers = roi_centers
        filtered_roi_radii = roi_radii_px
    else:
        # TODO maybe extend this to requiring the nth closest be closer than a
        # certain amount (to exclude 2 (or n) cells off by themselves)
        filtered_roi_centers = []
        filtered_roi_radii = []
        for i, (center, radius) in enumerate(zip(roi_centers, roi_radii_px)):
            n_neighbors = 0
            for j, other_center in enumerate(roi_centers):
                if i == j:
                    continue

                dist = euclidean_dist(center, other_center)
                if dist <= min_dist2neighbor_px:
                    n_neighbors += 1

                if n_neighbors >= min_neighbors:
                    filtered_roi_centers.append(center)
                    filtered_roi_radii.append(radius)
                    break

    assert len(filtered_roi_centers) == len(filtered_roi_radii)
    return np.array(filtered_roi_centers), np.array(filtered_roi_radii)


# TODO what were these files for again?
def autoroi_metadata_filename(ijroi_file):
    path, fname = split(ijroi_file)
    return join(path, '.{}.meta.p'.format(fname))


def scale_template(template_data, um_per_pixel_xy, target_cell_diam_um=None,
    target_cell_diam_px=None, target_template_d=None, debug=False):
    import cv2

    if target_cell_diam_um is None:
        # TODO make either of other kwargs also work (any of the 3 should
        # be alone)
        raise NotImplementedError

    template = template_data['template']
    margin = template_data['margin']
    # We enforce both elements of shape are same at creation.
    d = template.shape[0]

    target_cell_diam_px = target_cell_diam_um / um_per_pixel_xy

    # TODO which of these is correct? both? assert one is w/in
    # rounding err of other?
    template_cell_diam_px = d - 2 * margin
    template_scale = target_cell_diam_px / template_cell_diam_px
    '''
    template_cell_diam_um = template_data['mean_cell_diam_um']
    print(f'template_cell_diam_um: {template_cell_diam_um}')
    template_scale = target_cell_diam_um / template_cell_diam_um
    '''
    new_template_d = int(round(template_scale * d))
    new_template_shape = tuple([new_template_d] * len(template.shape))

    if debug:
        print(f'\nscale_template:\nd: {d}\nmargin: {margin}')
        print(f'um_per_pixel_xy: {um_per_pixel_xy}')
        print(f'target_cell_diam_um: {target_cell_diam_um}')
        print(f'target_cell_diam_px: {target_cell_diam_px}')
        print(f'template_cell_diam_px: {template_cell_diam_px}')
        print(f'template_scale: {template_scale}')
        print(f'new_template_d: {new_template_d}')
        print('')

    if new_template_d != d:
        scaled_template = cv2.resize(template, new_template_shape)
        scaled_template_cell_diam_px = \
            template_cell_diam_px * new_template_d / d

        return scaled_template, scaled_template_cell_diam_px

    else:
        return template.copy(), template_cell_diam_px


def _get_template_roi_radius_px(template_data, if_template_d=None, _round=True):
    template = template_data['template']
    margin = template_data['margin']
    d = template.shape[0]
    template_cell_diam_px = d - 2 * margin
    template_cell_radius_px = template_cell_diam_px / 2

    radius_frac = template_cell_radius_px / d

    if if_template_d is None:
        if_template_d = d

    radius_px = radius_frac * if_template_d
    if _round:
        radius_px = int(round(radius_px))
    return radius_px


# TODO test this w/ n.5 centers / radii
def get_circle_ijroi_input(center_px, radius_px):
    """Returns appropriate first arg for my ijroi.write_roi
    """
    min_corner = [center_px[0] - radius_px, center_px[1] - radius_px]
    max_corner = [
        min_corner[0] + 2 * radius_px,
        min_corner[1] + 2 * radius_px
    ]
    bbox = np.array([min_corner, max_corner])
    return bbox


# TODO move to viz? or maybe move all roi stuff to a new module?
def plot_circles(draw_on, centers, radii):
    import cv2
    draw_on = cv2.normalize(draw_on, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    draw_on = cv2.equalizeHist(draw_on)

    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.imshow(draw_on, cmap='gray')
    for center, radius in zip(centers, radii):
        roi_circle = plt.Circle((center[1] - 0.5, center[0] - 0.5), radius,
            fill=False, color='r'
        )
        ax.add_artist(roi_circle)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def fit_circle_rois(tif, template_data=None, avg=None, movie=None,
    method_str='cv2.TM_CCOEFF_NORMED', thresholds=None,
    exclusion_radius_frac=0.8, min_neighbors=None,
    debug=False, _packing_debug=False, show_fit=None,
    write_ijrois=False, _force_write_to=None, overwrite=False,
    exclude_dark_regions=None, dark_fraction_beyond_dhist_min=0.6,
    max_n_rois=650, min_n_rois=150,
    per_scale_max_n_rois=None,
    per_scale_min_n_rois=None, threshold_update_factor=0.7,
    update_factor_shrink_factor=0.7, max_threshold_tries=4,
    _um_per_pixel_xy=None, multiscale=True, roi_diams_px=None,
    roi_diams_um=None, roi_diams_from_kmeans_k=None,
    multiscale_strategy='one_order', template_d2match_value_scale_fn=None,
    allow_duplicate_px_scales=False, _show_scaled_templates=False, 
    verbose=False, **kwargs):
    """
    Even if movie or avg is passed in, tif is used to find metadata and
    determine where to save ImageJ ROIs.

    _um_per_pixel_xy only used for testing. Normally, XML is found from `tif`,
    and that is loaded to get this value.

    Returns centers_px, radii_px
    (both w/ coordinates and conventions ijrois uses)
    """
    import tifffile
    import cv2
    import ijroi
    from scipy.cluster.vq import vq

    if debug and show_fit is None:
        show_fit = True

    # TODO update all kwargs to go through a dict (store existing defaults as
    # dict at module level?) + need to handle passing of remaining to greedy_...
    # appropriately (don't pass stuff it won't take / don't unwrap and modify
    # so it only uses relevant ones)
    method_str2defaults = {
        # Though this does not work at all scales
        # (especially sensitive since not normed)
        'cv2.TM_CCOEFF': {'threshold': 4000.0, 'exclude_dark_regions': False},
        'cv2.TM_CCOEFF_NORMED': {'threshold': 0.3, 'exclude_dark_regions': True}
    }
    if method_str in method_str2defaults:
        for k, v in method_str2defaults[method_str].items():
            if k not in kwargs or kwargs[k] is None:
                kwargs[k] = v

    threshold = kwargs.pop('threshold')
    exclude_dark_regions = kwargs.pop('exclude_dark_regions')

    # Will divide rather than multiply by this,
    # if we need to increase threshold.
    assert threshold_update_factor < 1 and threshold_update_factor > 0

    # TODO also provide fitting for this fn in extract_template?
    mvw_key = 'match_value_weights'
    if template_d2match_value_scale_fn is not None:
        assert multiscale and multiscale_strategy == 'one_scale'
        assert mvw_key not in kwargs
        match_value_weights = []
    else:
        try:
            match_value_weights = kwargs.pop(mvw_key)
        except KeyError:
            match_value_weights = None

    if template_data is None:
        # TODO maybe options to cache this data across calls?
        # might not matter...
        template_data = load_template_data(err_if_missing=True)

    template = template_data['template']
    margin = template_data['margin']
    mean_cell_diam_um = template_data['mean_cell_diam_um']
    frame_shape = template_data['frame_shape']

    if _um_per_pixel_xy is None:
        keys = tiff_filename2keys(tif)
        ti_dir = thorimage_dir(*tuple(keys))
        xmlroot = thor.get_thorimage_xmlroot(ti_dir)
        um_per_pixel_xy = thor.get_thorimage_pixelsize_xml(xmlroot)
        del keys, ti_dir, xmlroot
    else:
        um_per_pixel_xy = _um_per_pixel_xy

    if multiscale:
        # Centroids are scalars in units of um diam
        kmeans_k2cluster_cell_diams = \
            template_data['kmeans_k2cluster_cell_diams']

        if roi_diams_px is not None:
            assert roi_diams_um is None and roi_diams_from_kmeans_k is None
            roi_diams_um = [rd_px * um_per_pixel_xy for rd_px in roi_diams_px]

        if roi_diams_um is None and roi_diams_from_kmeans_k is None:
            roi_diams_from_kmeans_k = 2

        if roi_diams_um is None:
            roi_diams_um = kmeans_k2cluster_cell_diams[roi_diams_from_kmeans_k]

            if verbose:
                in_px = roi_diams_um / um_per_pixel_xy
                print(f'Using ROI diameters {roi_diams_um} um ({in_px} px) from'
                    f' K-means (k={roi_diams_from_kmeans_k}) on data used to '
                    'generate template.'
                )
                del in_px

            if multiscale_strategy == 'random':
                all_cell_diams_um = template_data['all_cell_diams_um']
                clusters, _ = vq(all_cell_diams_um, roi_diams_um)
                count_clusters, counts = np.unique(clusters, return_counts=True)
                # otherwise would need to reindex the counts
                assert np.array_equal(count_clusters, np.sort(count_clusters))
                radii_px_ps = counts / np.sum(counts)
                kwargs['radii_px_ps'] = radii_px_ps
                print('Calculated these probabilities from template data:',
                    radii_px_ps
                )
    else:
        assert roi_diams_px is None
        assert roi_diams_um is None
        assert roi_diams_from_kmeans_k is None
        roi_diams_um = [mean_cell_diam_um]

    n_scales = len(roi_diams_um)

    if thresholds is None:
        thresholds = [threshold] * n_scales
    else:
        # TODO better way to specify thresholds in kmeans case, where
        # user may not know # thresholds needed in advance?
        assert len(thresholds) == n_scales
    del threshold
    thresholds = np.array(thresholds)

    if write_ijrois or _force_write_to is not None:
        write_ijrois = True

        path, tiff_last_part = split(tif)
        tiff_parts = tiff_last_part.split('.tif')
        assert len(tiff_parts) == 2 and tiff_parts[1] == ''
        fname = join(path, tiff_parts[0] + '_rois.zip')

        # TODO TODO change. fname needs to always be under
        # analysis_output_root (or just change input in
        # kc_natural_mixes/populate_db.py).
        # TODO or at least err if not subdir of it
        # see: https://stackoverflow.com/questions/3812849

        if _force_write_to is not None:
            if _force_write_to == True:
                fname = join(path, tiff_parts[0] + '_auto_rois.zip')
            else:
                fname = _force_write_to

        # TODO also check for modifications before overwriting (mtime in that hidden
        # file)
        elif not overwrite and exists(fname):
            print(fname, 'already existed. returning.')
            return None, None, None, None

    if avg is None:
        if movie is None:
            movie = tifffile.imread(tif)
        avg = movie.mean(axis=0)
    assert avg.shape[0] == avg.shape[1]
    orig_frame_d = avg.shape[0]

    # It seemed to me that picking a new threshold on cv2.TM_CCOEFF_NORMED was
    # not sufficient to reproduce cv2.TM_CCOEFF performance, so even if the
    # normed version were useful to keep the same threshold across image scales,
    # it seems other problems prevent me from using that in my case, so I'm
    # rescaling the image to match against.

    frame_downscaling = 1.0
    if avg.shape != frame_shape:
        scaled_avg = cv2.resize(avg, frame_shape)

        new_frame_d = scaled_avg.shape[0]
        frame_downscaling = orig_frame_d / new_frame_d
        del new_frame_d
        um_per_pixel_xy *= frame_downscaling
    else:
        scaled_avg = avg

    if debug:
        print('frame downscaling:', frame_downscaling)
        print('scaled_avg.shape:', scaled_avg.shape)

    if exclude_dark_regions:
        histvals, bins = np.histogram(scaled_avg.flat, bins=100, density=True)
        hv_deltas = np.diff(histvals)
        # TODO get the + 3 from a parameter controller percentage beyond
        # count delta min
        # min from: histvals[idx + 1] - histvals[idx]
        idx = np.argmin(hv_deltas)

        # TODO if this method of calculating dark_thresh doesn't seem robust,
        # compare robustness to thresholds from percetile of overal image,
        # or fixed thresholds on image scaled to [0,1], or fixed fractional
        # adjustment from delta hist threshold

        # Originally, dark_thresh was from bins[idx + 4], and that seemed to
        # work OK, so on one image, I calculated initial value (~0.5 -> 0.5)
        # of this from: ((scaled_avg <= bins[idx + 4]).sum() -
        # (scaled_avg <= bins[idx]).sum()) / scaled_avg.size (=0.543...)
        #dark_thresh = bins[idx + 4]
        dh_min_fractile = (scaled_avg <= bins[idx]).sum() / scaled_avg.size
        dark_thresh = np.percentile(scaled_avg,
            100 * (dark_fraction_beyond_dhist_min + dh_min_fractile)
        )

        exclusion_mask = scaled_avg >= dark_thresh
        if debug:
            fig, axs = plt.subplots(ncols=2)
            axs[0].imshow(scaled_avg)
            axs[1].imshow(exclusion_mask)
            axs[1].set_title('exclusion mask')
    else:
        exclusion_mask = None

    # We enforce earlier that template must be symmetric.
    d, d2 = template.shape
    assert d == d2

    match_images = []
    template_ds = []
    per_scale_radii_px = []
    for i, roi_diam_um in enumerate(roi_diams_um):
        scaled_template, scaled_template_cell_diam_px = scale_template(
            template_data, um_per_pixel_xy, roi_diam_um, debug=debug
        )
        scaled_radius_px = scaled_template_cell_diam_px / 2
        if debug:
            print('scaled template shape:', scaled_template.shape)

        if debug and _show_scaled_templates:
            fig, ax = plt.subplots()
            ax.imshow(scaled_template)
            title = f'scaled template (roi_diam_um={roi_diam_um:.2f})'
            if roi_diams_px is not None:
                title += f'\n(roi_diam_px={roi_diams_px[i]:.1f})'
            ax.set_title(title)

        # see note below about what i'd need to do to continue using
        # a check like this
        '''
        if template.shape != scaled_template.shape:
            # Just for checking that conversion back to original coordinates
            # (just scale diff) seems to be working.
            radius_px_before_scaling = int(round((d - 2 * margin) / 2))
        '''

        match_image = template_match(scaled_avg, scaled_template,
            method_str=method_str
        )
        if debug:
            print(f'scaled_template_cell_diam_px: '
                f'{scaled_template_cell_diam_px}'
            )
            print(f'scaled_radius_px: {scaled_radius_px}')

        template_d = scaled_template.shape[0]
        if (match_value_weights is not None and
            template_d2match_value_scale_fn is not None):

            match_value_weights.append(
                template_d2match_value_scale_fn(template_d)
            )

        match_images.append(match_image)
        template_ds.append(template_d)
        per_scale_radii_px.append(scaled_radius_px)

    if debug:
        print('template_ds:', template_ds)

    if len(set(template_ds)) != len(template_ds):
        if not allow_duplicate_px_scales:
            raise ValueError(f'roi_diams_um: {roi_diams_um} led to duplicate '
                f'pixel template scales ({template_ds})'
            )
        else:
            # TODO would still probably have to de-duplicate before passing to
            # packing fn
            raise NotImplementedError

    # TODO one fn that just returns circles, another to draw?
    draw_on = scaled_avg

    if per_scale_max_n_rois is not None or per_scale_min_n_rois is not None:
        if per_scale_max_n_rois is not None:
            assert len(per_scale_max_n_rois) == n_scales, \
                f'{len(per_scale_max_n_rois)} != {n_scales}'

        if per_scale_min_n_rois is not None:
            assert len(per_scale_min_n_rois) == n_scales, \
                f'{len(per_scale_min_n_rois)} != {n_scales}'

        print('Per-scale bounds on number of ROIs overriding global bounds.')
        min_n_rois = None
        max_n_rois = None
        per_scale_n_roi_bounds = True
    else:
        per_scale_n_roi_bounds = False

    threshold_tries_remaining = max_threshold_tries
    while threshold_tries_remaining > 0:
        # Regarding exclusion_radius_frac: 0.3 allowed too much overlap, 0.5
        # borderline too much w/ non-normed method (0.7 OK there)
        # (r=4,er=4,6 respectively, in 0.5 and 0.7 cases)
        if debug:
            print('per_scale_radii_px:', per_scale_radii_px)

        centers_px, radii_px = greedy_roi_packing(match_images, template_ds,
            per_scale_radii_px, thresholds=thresholds,
            min_neighbors=min_neighbors, exclusion_mask=exclusion_mask,
            exclusion_radius_frac=exclusion_radius_frac, draw_on=draw_on,
            draw_bboxes=False, draw_nums=False,
            multiscale_strategy=multiscale_strategy, debug=_packing_debug,
            match_value_weights=match_value_weights,
            _src_img_shape=scaled_avg.shape, **kwargs
        )

        n_found_per_scale = {r_px: 0 for r_px in per_scale_radii_px}
        for r_px in radii_px:
            n_found_per_scale[r_px] += 1
        assert len(centers_px) == sum(n_found_per_scale.values())

        if debug:
            print('number of ROIs found at each pixel radius scale:')
            pprint(n_found_per_scale)

        if per_scale_n_roi_bounds:
            wrong_num = False
            for i in range(n_scales):
                r_px = per_scale_radii_px[i]
                thr = thresholds[i]
                n_found = n_found_per_scale[r_px]

                sstr = f' w/ radius={r_px}px @ thr={thr:.2f}'
                have_retries = threshold_tries_remaining > 1
                if have_retries:
                    sstr += f'\nthr:={{:.2f}}'

                if per_scale_max_n_rois is not None:
                    smax = per_scale_max_n_rois[i]
                    if smax < n_found:
                        thresholds[i] /= threshold_update_factor
                        print((f'too many ROIs ({n_found} > {smax}){sstr}'
                            ).format(thresholds[i] if have_retries else tuple()
                        ))
                        wrong_num = True

                if per_scale_min_n_rois is not None:
                    smin = per_scale_min_n_rois[i]
                    if n_found < smin:
                        thresholds[i] *= threshold_update_factor
                        print(f'too few ROIs ({n_found} < {smin}){sstr}'.format(
                            thresholds[i] if have_retries else tuple()
                        ))
                        wrong_num = True

            if not wrong_num:
                break
            elif debug:
                print('')

        n_rois_found = len(centers_px)
        if not per_scale_n_roi_bounds:
            if ((min_n_rois is None or min_n_rois <= n_rois_found) and
                (max_n_rois is None or n_rois_found <= max_n_rois)):
                break

        threshold_tries_remaining -= 1
        if threshold_tries_remaining == 0:
            if debug or _packing_debug:
                plt.show()

            raise RuntimeError(f'too many/few ({n_rois_found}) ROIs still '
                f'detected after {max_threshold_tries} attempts to modify '
                'threshold. try changing threshold(s)?'
            )

        if not per_scale_n_roi_bounds:
            # TODO maybe squeeze to threshold if just one
            fail_notice_suffix = f', with thresholds={thresholds}'
            if max_n_rois is not None and n_rois_found > max_n_rois:
                thresholds /= threshold_update_factor
                fail_notice = \
                    f'found too many ROIs ({n_rois_found} > {max_n_rois})'

            elif min_n_rois is not None and n_rois_found < min_n_rois:
                thresholds *= threshold_update_factor
                fail_notice = \
                    f'found too few ROIs ({n_rois_found} < {min_n_rois})'

            fail_notice += fail_notice_suffix
            print(f'{fail_notice}\n\nretrying with thresholds={thresholds}')

        if update_factor_shrink_factor is not None:
            threshold_update_factor = \
                1 - (1 - threshold_update_factor) * update_factor_shrink_factor

    if frame_downscaling != 1.0:
        # TODO if i want to keep doing this check, while also supporting
        # multiscale case, gonna need to check (the set of?) radii returned
        # (would i need more info for that?)
        '''
        # This is to invert any previous scaling into coordinates for matching
        radius_px = scaled_radius_px * frame_downscaling

        # always gonna be true? seems like if a frame were 7x7, converting size
        # down to say 2x2 and back up by same formula would yield same result
        # as a 6x6 input or something, no?
        assert radius_px == radius_px_before_scaling
        del radius_px_before_scaling
        '''
        centers_px = centers_px * frame_downscaling
        radii_px  = radii_px * frame_downscaling

    # TODO would some other (alternating?) rounding rule help?
    # TODO random seed then randomly choose between floor and ceil for stuff
    # at 0.5?
    # TODO TODO or is rounding wrong? do some tests to try to figure out
    centers_px = np.round(centers_px).astype(np.uint16)
    radii_px = np.round(radii_px).astype(np.uint16)
    # this work if centers is empty?
    assert np.all(centers_px >= 0) and np.all(centers_px < orig_frame_d)

    if show_fit:
        fig, ax = plot_circles(avg, centers_px, radii_px)
        if tif is None:
            title = 'fit circles'
        else:
            title = tiff_title(tif)
        ax.set_title(title)

        roi_plot_dir = 'auto_rois'
        if not exists(roi_plot_dir):
            print(f'making directory {roi_plot_dir} for plots of ROI fits')
            os.makedirs(roi_plot_dir)

        roi_plot_fname = join(roi_plot_dir, title.replace('/','_') + '.png')
        print(f'Writing image showing fit ROIs to {roi_plot_fname}')
        fig.savefig(roi_plot_fname)

    if write_ijrois:
        auto_md_fname = autoroi_metadata_filename(fname)

        name2bboxes = list()
        for i, (center_px, radius_px) in enumerate(zip(centers_px, radii_px)):
            # TODO TODO test that these radii are preserved across
            # round trip save / loads?
            bbox = get_circle_ijroi_input(center_px, radius_px)
            name2bboxes.append((str(i), bbox))

        print('Writing ImageJ ROIs to {} ...'.format(fname))
        # TODO TODO TODO uncomment
        '''
        ijroi.write_oval_roi_zip(name2bboxes, fname)

        with open(auto_md_fname, 'wb') as f:
            data = {
                'mtime': getmtime(fname)
            }
            pickle.dump(data, f)
        '''

    ns_found = [n_found_per_scale[rpx] for rpx in per_scale_radii_px]

    return centers_px, radii_px, thresholds, ns_found


def template_data_file():
    template_cache = 'template.p'
    return join(analysis_output_root(), template_cache)


def load_template_data(err_if_missing=False):
    template_cache = template_data_file()
    if exists(template_cache):
        with open(template_cache, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        if err_if_missing:
            raise IOError(f'template data not found at {template_cache}')
        return None


def assign_frames_to_trials(movie, presentations_per_block, block_first_frames,
    odor_onset_frames):
    """Returns arrays trial_start_frames, trial_stop_frames
    """
    n_frames = movie.shape[0]
    # TODO maybe just add metadata['drop_first_n_frames'] to this?
    # (otherwise, that variable screws things up, right?)
    #onset_frame_offset = \
    #    odor_onset_frames[0] - block_first_frames[0]

    # TODO delete this hack, after implementing more robust frame-to-trial
    # assignment described below
    b2o_offsets = sorted([o - b for b, o in zip(block_first_frames,
        odor_onset_frames[::presentations_per_block])
    ])
    assert len(b2o_offsets) >= 3
    # TODO TODO TODO re-enable after fixing frame_times based issues w/
    # volumetric data
    # TODO might need to allow for some error here...? frame or two?
    # (in resonant scanner case, w/ frame averaging maybe)
    #assert b2o_offsets[-1] == b2o_offsets[-2]
    onset_frame_offset = b2o_offsets[-1]
    #

    # TODO TODO TODO instead of this frame # strategy for assigning frames
    # to trials, maybe do this:
    # 1) find ~max # frames from block start to onset, as above
    # TODO but maybe still warn if some offset deviates from max by more
    # than a frame or two...
    # 2) paint all frames before odor onsets up to this max # frames / time
    #    (if frames have a time discontinuity between them indicating
    #     acquisition did not proceed continuously between them, do not
    #     paint across that boundary)
    # 3) paint still-unassigned frames following odor onset in the same
    #    fashion (again stopping at boundaries of > certain dt)
    # [4)] if not using max in #1 (but something like rounded mean)
    #      may still have unassigned frames at block starts. assign those to
    #      trials.
    # TODO could just assert everything within block regions i'm painting
    # does not have time discontinuities, and then i could just deal w/
    # frames

    trial_start_frames = np.append(0,
        odor_onset_frames[1:] - onset_frame_offset
    )
    trial_stop_frames = np.append(
        odor_onset_frames[1:] - onset_frame_offset - 1, n_frames - 1
    )

    # TODO same checks are made for blocks, so factor out?
    total_trial_frames = 0
    for i, (t_start, t_end) in enumerate(
        zip(trial_start_frames, trial_stop_frames)):

        if i != 0:
            last_t_end = trial_stop_frames[i - 1]
            assert last_t_end == (t_start - 1)

        total_trial_frames += t_end - t_start + 1

    assert total_trial_frames == n_frames, \
        '{} != {}'.format(total_trial_frames, n_frames)
    #

    # TODO warn if all block/trial lens are not the same? (by more than some
    # threshold probably)

    return trial_start_frames, trial_stop_frames


# TODO TODO TODO after refactoring much of the stuff that was under
# open_recording and some of its downstream fns from gui.py, also refactor this
# to use the new fns
# TODO and maybe move this to project/analysis specific specific repo /
# submodule as it uses matlab pipeline ouputs...
# TODO maybe move to project/analysis specific repo / submodule (same as other
# stuff that uses matlab_kc_plane outputs)
# TODO maybe move to matlab (this is only fn that uses either of fns imported
# from there)
def movie_blocks(tif, movie=None, allow_gsheet_to_restrict_blocks=True,
    stimfile=None, first_block=None, last_block=None):
    """Returns list of arrays, one per continuous acquisition.

    `tif` must be named and placed according to convention, and a .mat file
    created from `ejhonglab/matlab_kc_plane` (typically run via `populate_db.py`
    in what is now my `kc_natural_mixes` repo) must exist in the conventional
    path under the analysis root. This .mat file is used for the timing
    information (ultimately derived mainly from ThorSync data).

    Total length along time dimension should be preserved from input TIFF.
    """
    from scipy import stats

    if movie is None:
        import tifffile
        movie = tifffile.imread(tif)

    keys = tiff_filename2keys(tif)
    mat = matlab.matfile(*keys)
    #mat = matfile(*keys)

    # TODO TODO TODO refactor all stuff that uses this to new output format
    # (and remove factored checks, etc)
    ti = matlab.load_mat_timing_info(mat)
    #ti = load_mat_timing_info(mat)

    if stimfile is None:
        df = mb_team_gsheet()
        recordings = df.loc[
            (df.date == keys.date) &
            (df.fly_num == keys.fly_num) &
            (df.thorimage_dir == keys.thorimage_id)
        ]
        del df
        recording = recordings.iloc[0]
        del recordings
        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.'.format(
                recording.project))
            return

        stimfile = recording['stimulus_data_file']
        first_block = recording['first_block']
        last_block = recording['last_block']
        del recording

        stimfile_path = join(stimfile_root(), stimfile)
    else:
        warnings.warn('using hardcoded stimulus file, rather than using value '
            'from MB team gsheet'
        )
        if exists(stimfile):
            stimfile_path = stimfile
        else:
            stimfile_path = join(stimfile_root(), stimfile)
            assert exists(stimfile_path), (f'stimfile {stimfile} not found '
                f'alone or under {stimfile_root()}'
            )

    # TODO also err if not readable / valid
    if not exists(stimfile_path):
        raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
            stimfile_root)
        )

    with open(stimfile_path, 'rb') as f:
        data = pickle.load(f)

    # TODO just infer from data if no stimfile and not specified in
    # metadata_file
    n_repeats = int(data['n_repeats'])

    # TODO delete this hack (which is currently just using new pickle
    # format as a proxy for the experiment being a supermixture experiment)
    if 'odor_lists' not in data:
        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3
        odor_list = data['odor_pair_list']
    else:
        n_expected_real_blocks = 3
        odor_list = data['odor_lists']
        # because of "block" def in arduino / get_stiminfo code
        # not matching def in randomizer / stimfile code
        # (scopePin pulses vs. randomization units, depending on settings)
        presentations_per_repeat = len(odor_list) // n_expected_real_blocks
        assert len(odor_list) % n_expected_real_blocks == 0

        # Hardcode to break up into more blocks, to align defs of blocks.
        # TODO (maybe just for experiments on 2019-07-25 ?) or change block
        # handling in here? make more flexible?
        n_repeats = 1

    presentations_per_block = n_repeats * presentations_per_repeat

    if pd.isnull(first_block):
        first_block = 0
    else:
        first_block = int(first_block) - 1

    if pd.isnull(last_block):
        n_full_panel_blocks = \
            int(len(odor_list) / presentations_per_block)
        last_block = n_full_panel_blocks - 1
    else:
        last_block = int(last_block) - 1

    first_presentation = first_block * presentations_per_block
    last_presentation = (last_block + 1) * presentations_per_block - 1

    odor_list = odor_list[first_presentation:(last_presentation + 1)]
    assert (len(odor_list) % (presentations_per_repeat * n_repeats) == 0)

    # TODO TODO delete odor frame stuff after using them to check blocks frames
    # are actually blocks and not trials
    # TODO or if keeping odor stuff, re-add asserts involving odor_list,
    # since how i have that here

    odor_onset_frames = np.array(ti['stim_on'], dtype=np.uint32
        ).flatten() - 1
    odor_offset_frames = np.array(ti['stim_off'], dtype=np.uint32).flatten() - 1
    assert len(odor_onset_frames) == len(odor_offset_frames)

    # Of length equal to number of blocks. Each element is the frame
    # index (from 1) in CNMF output that starts the block, where
    # block is defined as a period of continuous acquisition.
    block_first_frames = np.array(ti['block_start_frame'], dtype=np.uint32
        ).flatten() - 1
    block_last_frames = np.array(ti['block_end_frame'], dtype=np.uint32
        ).flatten() - 1

    n_blocks_from_gsheet = last_block - first_block + 1
    n_blocks_from_thorsync = len(block_first_frames)

    assert (len(odor_list) == (last_block - first_block + 1) *
        presentations_per_block)

    n_presentations = n_blocks_from_gsheet * presentations_per_block

    err_msg = ('{} blocks ({} to {}, inclusive) in Google sheet {{}} {} ' +
        'blocks from ThorSync.').format(n_blocks_from_gsheet,
        first_block + 1, last_block + 1, n_blocks_from_thorsync)
    fail_msg = (' Fix in Google sheet, turn off ' +
        'cache if necessary, and rerun.')

    if n_blocks_from_gsheet > n_blocks_from_thorsync:
        raise ValueError(err_msg.format('>') + fail_msg)

    elif n_blocks_from_gsheet < n_blocks_from_thorsync:
        if allow_gsheet_to_restrict_blocks:
            warnings.warn(err_msg.format('<') + (' This is ONLY ok if you '+
                'intend to exclude the LAST {} blocks in the Thor output.'
                ).format(n_blocks_from_thorsync - n_blocks_from_gsheet))
        else:
            raise ValueError(err_msg.format('<') + fail_msg)

    frame_times = np.array(ti['frame_times']).flatten()

    # TODO replace this w/ factored check fn
    total_block_frames = 0
    for i, (b_start, b_end) in enumerate(
        zip(block_first_frames, block_last_frames)):

        if i != 0:
            last_b_end = block_last_frames[i - 1]
            assert last_b_end == (b_start - 1)

        assert (b_start < len(frame_times)) and (b_end < len(frame_times))
        block_frametimes = frame_times[b_start:b_end]
        dts = np.diff(block_frametimes)
        # np.max(np.abs(dts - np.mean(dts))) / np.mean(dts)
        # was 0.000148... in one case I tested w/ data from the older
        # system, so the check below w/ rtol=1e-4 would fail.
        mode = stats.mode(dts)[0]
        assert np.allclose(dts, mode, rtol=3e-4)

        total_block_frames += b_end - b_start + 1

    orig_n_frames = movie.shape[0]
    # TODO may need to remove this assert to handle cases where there is a
    # partial block (stopped early). leave assert after slicing tho.
    # (warn instead, probably)
    assert total_block_frames == orig_n_frames, \
        '{} != {}'.format(total_block_frames, orig_n_frames)

    if allow_gsheet_to_restrict_blocks:
        # TODO unit test for case where first_block != 0 and == 0
        # w/ last_block == first_block and > first_block
        # TODO TODO doesn't this only support dropping blocks at end?
        # do i assert that first_block is 0 then? probably should...
        # TODO TODO TODO shouldnt it be first_block:last_block+1?
        block_first_frames = block_first_frames[
            :(last_block - first_block + 1)]
        block_last_frames = block_last_frames[
            :(last_block - first_block + 1)]

        assert len(block_first_frames) == n_blocks_from_gsheet
        assert len(block_last_frames) == n_blocks_from_gsheet

        # TODO also delete this odor frame stuff when done
        odor_onset_frames = odor_onset_frames[
            :(last_presentation - first_presentation + 1)]
        odor_offset_frames = odor_offset_frames[
            :(last_presentation - first_presentation + 1)]

        assert len(odor_onset_frames) == n_presentations
        assert len(odor_offset_frames) == n_presentations
        #

        frame_times = frame_times[:(block_last_frames[-1] + 1)]

    last_frame = block_last_frames[-1]

    n_tossed_frames = movie.shape[0] - (last_frame + 1)
    if n_tossed_frames != 0:
        print(('Tossing trailing {} of {} frames of movie, which did not ' +
            'belong to any used block.\n').format(
            n_tossed_frames, movie.shape[0]))

    # TODO factor this metadata handling out. fns for load / set?
    # combine w/ remy's .mat metadata (+ my stimfile?)

    # This will return defaults if the YAML file is not found.
    meta = metadata(*keys)

    # TODO want / need to do more than just slice to free up memory from
    # other pixels? is that operation worth it?
    drop_first_n_frames = meta['drop_first_n_frames']
    # TODO TODO err if this is past first odor onset (or probably even too
    # close)
    del meta

    odor_onset_frames = [n - drop_first_n_frames for n in odor_onset_frames]
    odor_offset_frames = [n - drop_first_n_frames for n in odor_offset_frames]

    block_first_frames = [n - drop_first_n_frames for n in block_first_frames]
    block_first_frames[0] = 0
    block_last_frames = [n - drop_first_n_frames for n in block_last_frames]

    assert odor_onset_frames[0] > 0

    frame_times = frame_times[drop_first_n_frames:]
    movie = movie[drop_first_n_frames:(last_frame + 1)]

    # TODO TODO fix bug referenced in cthulhu:190520...
    # and re-enable assert
    assert movie.shape[0] == len(frame_times), \
        '{} != {}'.format(movie.shape[0], len(frame_times))
    #

    if movie.shape[0] != len(frame_times):
        warnings.warn('{} != {}'.format(movie.shape[0], len(frame_times)))

    # TODO maybe move this and the above checks on block start/end frames
    # + frametimes into assign_frames_to_trials
    n_frames = movie.shape[0]
    total_block_frames = sum([e - s + 1 for s, e in
        zip(block_first_frames, block_last_frames)
    ])

    assert total_block_frames == n_frames, \
        '{} != {}'.format(total_block_frames, n_frames)


    # TODO any time / space diff returning slices to slice array and only
    # slicing inside loop vs. returning list of (presumably views) by slicing
    # matrix?
    blocks = [movie[start:(stop + 1)] for start, stop in
        zip(block_first_frames, block_last_frames)
    ]
    assert sum([b.shape[0] for b in blocks]) == movie.shape[0]
    return blocks


def downsample_movie(movie, target_fps, current_fps, allow_overshoot=True,
    allow_uneven_division=False, relative_fps_err=True, debug=False):
    """Returns downsampled movie by averaging consecutive groups of frames.

    Groups of frames averaged do not overlap.
    """
    if allow_uneven_division:
        raise NotImplementedError

    # TODO maybe kwarg for max acceptable (rel/abs?) factor error,
    # and err / return None if it can't be achieved

    target_factor = current_fps / target_fps
    if debug:
        print(f'allow_overshoot: {allow_overshoot}')
        print(f'allow_uneven_division: {allow_uneven_division}')
        print(f'relative_fps_err: {relative_fps_err}')
        print(f'target_fps: {target_fps:.2f}\n')
        print(f'target_factor: {target_factor:.2f}\n')

    n_frames = movie.shape[0]

    # TODO TODO also support uneven # of frames per bin (toss last probably)
    # (skip loop checking for even divisors in that case)

    # Find the largest/closest downsampling we can do, with equal numbers of
    # frames for each average.
    best_divisor = None
    for i in range(1, n_frames):
        if n_frames % i != 0:
            continue

        decimated_n_frames = n_frames // i
        # (will always be float(i) in even division case, so could get rid of
        # this if that's all i'll support)
        factor = n_frames / decimated_n_frames
        if debug:
            print(f'factor: {factor:.2f}')

        if factor > target_factor and not allow_overshoot:
            if debug:
                print('breaking because of overshoot')
            break

        downsampled_fps = current_fps / factor
        fps_error = downsampled_fps - target_fps
        if relative_fps_err:
            fps_error = fps_error / target_fps

        if debug:
            print(f'downsampled_fps: {downsampled_fps:.2f}')
            print(f'fps_error: {fps_error:.2f}')

        if best_divisor is None or abs(fps_error) < abs(best_fps_error):
            best_divisor = i
            best_downsampled_fps = downsampled_fps
            best_fps_error = fps_error
            best_factor = factor

            if debug:
                print(f'best_downsampled_fps: {best_downsampled_fps:.2f}')
                print('new best factor')

        elif (best_divisor is not None and
            abs(fps_error) > abs(best_fps_error)):

            assert allow_overshoot
            if debug:
                print('breaking because past best factor')
            break

        if debug:
            print('')

    assert best_divisor is not None

    # TODO unit test for this case
    if best_divisor == 1:
        raise ValueError('best downsampling with this flags at factor of 1')

    if debug:
        print(f'best_divisor: {best_divisor}')
        print(f'best_factor: {best_factor:.2f}')
        print(f'best_fps_error: {best_fps_error:.2f}')
        print(f'best_downsampled_fps: {best_downsampled_fps:.2f}')

    frame_shape = movie.shape[1:]
    new_n_frames = n_frames // best_divisor

    # see: stackoverflow.com/questions/15956309 for how to adapt this
    # to uneven division case
    downsampled = movie.reshape((new_n_frames, best_divisor) + frame_shape
        ).mean(axis=1)

    # TODO maybe it's obvious, but is there any kind of guarantee dimensions in
    # frame_shape will not be screwed up in a way relevant to the average
    # when reshaping?
    # well, at least this looks reasonable:
    # viz.image_grid(downsampled[:64])

    return downsampled, best_downsampled_fps


# TODO delete (+ probably delete contour2mask too and replace use of both w/ ijroi2mask)
# don't like this convexHull based approach though...
# (because roi may be intentionally not a convex hull)
def ijroi2cv_contour(roi):
    import cv2

    ## cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## cnts[1][0].shape
    ## cnts[1][0].dtype
    # from inspecting output of findContours, as above:
    #cnt = np.expand_dims(ijroi, 1).astype(np.int32)
    # TODO fix so this isn't necessary. in case of rois that didn't start as
    # circles, the convexHull may occasionally not be equal to what i want
    cnt = cv2.convexHull(roi.astype(np.int32))
    # if only getting cnt from convexHull, this is probably a given...
    assert cv2.contourArea(cnt) > 0
    return cnt
#


def roi_center(roi):
    import cv2
    cnt = ijroi2cv_contour(roi)
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array((cx, cy))


def roi_centers(rois):
    centers = []
    for roi in rois:
        center = roi_center(roi)
        # pretty close to (w/in 0.5 in each dim) np.mean(roi, axis=0),
        # in at least one example i played with
        centers.append(center)
    return np.array(centers)


def tiff_title(tif):
    """Returns abbreviation of TIFF filename for use in titles.
    """
    parts = [x for x in tif.split('/')[-4:] if x != 'tif_stacks']
    ext = '.tif'
    if parts[-1].endswith(ext):
        parts[-1] = parts[-1][:-len(ext)]
    return '/'.join(parts)


# TODO didn't i have some other fn for this? delete one if so
# (or was it just in natural_odors?)
def to_filename(x, period=True):
    """Take a str and normalizes it a bit to make it a better filename prefix.

    E.g. taking a plot title and using it to derive a filename for saving the
    plot.
    """
    # To handle things like consecutive whitespace (e.g. 'x\n y')
    x = '_'.join(x.split())

    replace_dict = {
        '/': '_',
        '@': '_',
        ',': '',
        '.': '',
        '(': '',
        ')': '',
        '[': '',
        ']': '',
    }
    for k, v in replace_dict.items():
        x = x.replace(k, v)

    # Replace multiple consecutive '_' with a single '_'
    x = re.sub('_+', '_', x)

    # TODO delete this and refactor code that expects this behavior to add the period
    if period:
        x += '.'

    return x


def point_idx(xys_to_check, pt_xy, swap_xy=False):
    if not swap_xy:
        x, y = pt_xy
    else:
        y, x = pt_xy

    matching_pt = (
        (xys_to_check[:,0] == x) &
        (xys_to_check[:,1] == y)
    )
    assert matching_pt.sum() == 1
    return np.argwhere(matching_pt)[0][0]


def correspond_rois(left_centers_or_seq, *right_centers, cost_fn=euclidean_dist,
    max_cost=None, show=False, write_plots=True, left_name='Left',
    right_name='Right', name_prefix='', draw_on=None, title='', colors=None,
    connect_centers=True, pairwise_plots=True, pairwise_same_style=False,
    roi_numbers=False, jitter=True, progress=None, squeeze=True,
    verbose=False, debug_points=None):
    """
    Args:
    left_centers_or_seq (list): (length n_timepoints) list of (n_rois x 2)
        arrays of ROI center coordinates.

    Returns:
    lr_matches: list of arrays matching ROIs in one timepoint to ROIs in the
        next.

    left_unmatched: list of arrays with ROI labels at time t,
        without a match at time (t + 1)

    right_unmatched: same as left_unmatched, but for (t + 1) with respect to t.

    total_costs: array of sums of costs from matching.

    fig: matplotlib figure handle to the figure with all ROIs on it,
        for modification downstream.
    """
    # TODO doc support for ROI inputs / rewrite to expect them
    # (to use jaccard, etc)

    from scipy.optimize import linear_sum_assignment
    import seaborn as sns

    # TODO maybe unsupport two args case to be more concise
    if len(right_centers) == 0:
        sequence_of_centers = left_centers_or_seq

    elif len(right_centers) == 1:
        right_centers = right_centers[0]
        sequence_of_centers = [left_centers_or_seq, right_centers]

    else:
        raise ValueError('wrong number of arguments')

    if progress is None:
        progress = len(sequence_of_centers) >= 10
    if progress:
        from tqdm import tqdm

    if max_cost is None:
        raise ValueError('max_cost must not be None')

    if verbose:
        print(f'max_cost: {max_cost:.2f}')

    default_two_colors = ['red', 'blue']
    if len(sequence_of_centers) == 2:
        pairwise_plots = False
        scatter_alpha = 0.6
        scatter_marker = None
        labels = [n + ' centers' for n in (left_name, right_name)]
        if colors is None:
            colors = default_two_colors
    else:
        scatter_alpha = 0.8
        scatter_marker = 'x'
        labels = [name_prefix + str(i) for i in range(len(sequence_of_centers))]
        if colors is None:
            colors = sns.color_palette('hls', len(sequence_of_centers))

    # TODO don't copy after removing need for flip
    # Copying so that flip doesn't screw with input data.
    new_sequence_of_centers = []
    for i, centers in enumerate(sequence_of_centers):
        # Otherwise it should be an ndarray representing centers
        # TODO assertion on dims in ndarray case
        if type(centers) is list:
            centers = roi_centers(centers)

        # This is just to make them display right (not transposed).
        # Should not change any of the matching.
        # TODO remove need for this flip
        new_sequence_of_centers.append(np.flip(centers, axis=1))
    sequence_of_centers = new_sequence_of_centers

    fig = None
    if show:
        figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
        if draw_on is None:
            color = 'black'
        else:
            ax.imshow(draw_on, cmap='gray')
            ax.axis('off')
            color = 'yellow'
        fontsize = 8
        text_x_offset = 2
        plot_format = 'png'

        if jitter:
            np.random.seed(50)
            jl = -0.1
            jh = 0.1

    unmatched_left = []
    unmatched_right = []
    lr_matches = []
    cost_totals = []

    if progress:
        centers_iter = tqdm(range(len(sequence_of_centers) - 1))
        print('Matching ROIs across timepoints:')
    else:
        centers_iter = range(len(sequence_of_centers) - 1)

    for ci, k in enumerate(centers_iter):
        left_centers = sequence_of_centers[k]
        right_centers = sequence_of_centers[k + 1]

        # TODO TODO use pdist / something else under scipy.spatial.distance?
        # TODO other / better ways to generate cost matrix?
        # pairwise jacard (would have to not take centers then)?
        # TODO why was there a "RuntimeWarning: invalid valid encounterd in
        # multiply" here ocassionally? it still seems like we had some left and
        # right centers, so idk
        costs = np.empty((len(left_centers), len(right_centers))) * np.nan
        for i, cl in enumerate(left_centers):
            for j, cr in enumerate(right_centers):
                # TODO short circuit as appropriate? better way to loop over
                # coords we need?
                cost = cost_fn(cl, cr)
                costs[i,j] = cost

        '''
        if verbose:
            print(f'(iteration {ci}) fraction of costs >= max_cost:',
                '{:.3f}'.format((costs >= max_cost).sum() / costs.size)
            )
        '''

        # TODO delete. problem does not seem to be in this fn.
        '''
        if debug_points and ci in debug_points:
            print(f'iteration {ci}:')
            ln = 3
            for pt_info in debug_points[ci]:
                name = pt_info['name']
                xy0 = pt_info['xy0']
                # TODO print cost wrt this point
                xy1 = pt_info['xy1']

                # swap_xy etc b/c of flip earlier
                idx = point_idx(left_centers, xy0, swap_xy=True)
                print(f'lowest {ln} costs for point {name} in '
                    'left_centers:'
                )
                # TODO also print to which other points (x,y)
                # correspond to these ln lowest costs
                print(np.sort(costs[idx, :])[:ln])
        '''

        # TODO TODO TODO test that setting these to an arbitrarily large number
        # produces matching equivalent to setting them to max_cost here
        costs[costs >= max_cost] = max_cost

        # TODO was Kellan's method of matching points not equivalent to this?
        # or per-timestep maybe it was (or this was better), but he also
        # had a way to evolve points over time (+ a particular cost)?

        left_idx, right_idx = linear_sum_assignment(costs)
        # Just to double-check properties I assume about the assignment
        # procedure.
        assert len(left_idx) == len(np.unique(left_idx))
        assert len(right_idx) == len(np.unique(right_idx))

        n_not_drawn = None
        if show:
            if jitter:
                left_jitter = np.random.uniform(low=jl, high=jh,
                    size=left_centers.shape)
                right_jitter = np.random.uniform(low=jl, high=jh,
                    size=right_centers.shape)

                left_centers_to_plot = left_centers + left_jitter
                right_centers_to_plot = right_centers + right_jitter
            else:
                left_centers_to_plot = left_centers
                right_centers_to_plot = right_centers

            if pairwise_plots:
                # TODO maybe change multiple pairwise plots to be created as
                # axes within one the axes from one call to subplots
                pfig, pax = plt.subplots(figsize=figsize)
                if pairwise_same_style:
                    pmarker = scatter_marker
                    c1 = colors[k]
                    c2 = colors[k + 1]
                else:
                    pmarker = None
                    c1 = default_two_colors[0]
                    c2 = default_two_colors[1]

                if draw_on is not None:
                    pax.imshow(draw_on, cmap='gray')
                    pax.axis('off')

                pax.scatter(*left_centers_to_plot.T, label=labels[k],
                    color=c1, alpha=scatter_alpha,
                    marker=pmarker
                )
                pax.scatter(*right_centers_to_plot.T, label=labels[k + 1],
                    color=c2, alpha=scatter_alpha,
                    marker=pmarker
                )
                psuffix = f'{k} vs. {k+1}'
                if len(name_prefix) > 0:
                    psuffix = f'{name_prefix} ' + psuffix
                if len(title) > 0:
                    ptitle = f'{title}, ' + psuffix
                else:
                    ptitle = psuffix
                pax.set_title(ptitle)
                pax.legend()

            ax.scatter(*left_centers_to_plot.T, label=labels[k],
                color=colors[k], alpha=scatter_alpha,
                marker=scatter_marker
            )
            # TODO factor out scatter + opt numbers (internal fn?)
            if roi_numbers:
                for i, (x, y) in enumerate(left_centers_to_plot):
                    ax.text(x + text_x_offset, y, str(i),
                        color=colors[k], fontsize=fontsize
                    )

            # Because generally this loop only scatterplots the left_centers,
            # so without this, the last set of centers would not get a
            # scatterplot.
            if (k + 1) == (len(sequence_of_centers) - 1):
                last_centers = right_centers_to_plot

                ax.scatter(*last_centers.T, label=labels[-1],
                    color=colors[-1], alpha=scatter_alpha,
                    marker=scatter_marker
                )
                if roi_numbers:
                    for i, (x, y) in enumerate(last_centers):
                        ax.text(x + text_x_offset, y, str(i),
                            color=colors[-1], fontsize=fontsize
                        )

            if connect_centers:
                n_not_drawn = 0
                for li, ri in zip(left_idx, right_idx):
                    if costs[li,ri] >= max_cost:
                        n_not_drawn += 1
                        continue
                        #linestyle = '--'
                    else:
                        linestyle = '-'

                    lc = left_centers_to_plot[li]
                    rc = right_centers_to_plot[ri]
                    correspondence_line = ([lc[0], rc[0]], [lc[1], rc[1]])

                    ax.plot(*correspondence_line, linestyle=linestyle,
                        color=color, alpha=0.7)

                    if pairwise_plots:
                        pax.plot(*correspondence_line, linestyle=linestyle,
                            color=color, alpha=0.7)

                # TODO didn't i have some fn for getting filenames from things
                # like titles? use that if so
                # TODO plot format + flag to control saving + save to some
                # better dir
                # TODO separate dir for these figs? or at least place where some
                # of other figs currently go?
                if pairwise_plots and write_plots:
                    fname = to_filename(ptitle) + plot_format
                    print(f'writing to {fname}')
                    pfig.savefig(fname)

        k_unmatched_left = set(range(len(left_centers))) - set(left_idx)
        k_unmatched_right = set(range(len(right_centers))) - set(right_idx)

        # TODO why is costs.min() actually 0? that seems unlikely?
        match_costs = costs[left_idx, right_idx]
        total_cost = match_costs.sum()

        to_unmatch = match_costs >= max_cost
        # For checking consistent w/ draw output above
        if verbose or n_not_drawn is not None:
            n_unmatched = to_unmatch.sum()
            if n_not_drawn is not None:
                assert n_not_drawn == n_unmatched, \
                    f'{n_not_drawn} != {n_unmatched}'
            if verbose:
                print(f'(iteration={ci}) unmatched {n_unmatched} for exceeding'
                    ' max_cost'
                )

        if debug_points and ci in debug_points:
            l_idxs = []
            r_idxs = []
            for pt_info in debug_points[ci]:
                name = pt_info['name']
                # swap_xy etc b/c of flip earlier
                xy0 = pt_info['xy0']
                xy1 = pt_info['xy1']
                print(f'name: {name}, xy0: {xy0}, xy1: {xy1}')

                lidx = point_idx(left_centers, xy0, swap_xy=True)
                assert left_idx.max() <= len(left_centers)

                midx0 = np.where(left_idx == lidx)[0]
                if len(midx0) > 0:
                    assert len(midx0) == 1
                    midx0 = midx0[0]
                    assert left_idx[midx0] == lidx
                    lpt = left_centers[left_idx[midx0]]
                    assert tuple(lpt)[::-1] == xy0
                    # since by the time debug_points are generated, point
                    # matching seems off, rpt will not necessarily be
                    # equal to lpt.
                    rpt_idx = right_idx[midx0]
                    rpt = right_centers[rpt_idx]
                    mcost0 = match_costs[midx0]
                    print(f'xy0 matched ({lidx}:{lpt} -> {rpt_idx}:{rpt}) '
                        f'at cost {mcost0:.3f}'
                    )
                    if to_unmatch[midx0]:
                        print('xy0 will be unmatched for cost >= max_cost!')
                    else:
                        l_idxs.append((name, lidx))
                        # For use debugging downstream of this function.
                        pt_info['xy0_lidx'] = lidx
                        pt_info['xy0_ridx'] = rpt_idx
                        pt_info['xy0_lpt'] = lpt[::-1]
                        pt_info['xy0_rpt'] = rpt[::-1]
                else:
                    print(f'xy0 not matched!')

                ridx = point_idx(right_centers, xy1, swap_xy=True)
                assert right_idx.max() <= len(right_centers)
                midx1 = np.where(right_idx == ridx)[0]
                if len(midx1) > 0:
                    assert len(midx1) == 1
                    midx1 = midx1[0]
                    assert right_idx[midx1] == ridx
                    rpt = right_centers[right_idx[midx1]]
                    assert tuple(rpt)[::-1] == xy1
                    # likewise, this is not necessarily equal to xy0, by the
                    # time downstream functions screw up propagating the matches
                    lpt_idx = left_idx[midx1]
                    lpt = left_centers[lpt_idx]
                    mcost1 = match_costs[midx1]
                    print(f'xy1 matched ({ridx}:{rpt} <- {lpt_idx}:{lpt}) '
                        f'at cost {mcost1:.3f}'
                    )
                    if to_unmatch[midx1]:
                        print('xy1 will be unmatched for cost >= max_cost!')
                    else:
                        r_idxs.append((name, ridx))
                        # For use debugging downstream of this function.
                        pt_info['xy1_lidx'] = lpt_idx
                        pt_info['xy1_ridx'] = ridx
                        pt_info['xy1_lpt'] = lpt[::-1]
                        pt_info['xy1_rpt'] = rpt[::-1]
                else:
                    print(f'xy1 not matched!')
                print('')

        k_unmatched_left.update(left_idx[to_unmatch])
        k_unmatched_right.update(right_idx[to_unmatch])
        left_idx = left_idx[~ to_unmatch]
        right_idx = right_idx[~ to_unmatch]

        n_unassigned = abs(len(left_centers) - len(right_centers))

        total_cost += max_cost * n_unassigned
        # TODO better way to normalize error?
        total_cost = total_cost / min(len(left_centers), len(right_centers))

        # TODO maybe compute costs for all unmatched w/ pdist, and check
        # nothing is < max_cost

        unmatched_left.append(np.array(list(k_unmatched_left)))
        unmatched_right.append(np.array(list(k_unmatched_right)))
        cost_totals.append(total_cost)
        lr_matches.append(np.stack([left_idx, right_idx], axis=-1))

        # These just need to be consistent w/ numbers printed before colons
        # above (and they are).
        if debug_points and ci in debug_points:
            lrm = lr_matches[-1]
            for name, li in l_idxs:
                midx = np.argwhere(lrm[:, 0] == li)[0]
                assert len(midx) == 1
                midx = midx[0]
                print(f'name: {name}, xy0 match row {midx}:', lrm[midx, :])
            for name, ri in r_idxs:
                midx = np.argwhere(lrm[:, 1] == ri)[0]
                assert len(midx) == 1
                midx = midx[0]
                print(f'name: {name}, xy1 match row {midx}:', lrm[midx, :])
            print('')

    if show:
        ax.legend()
        ax.set_title(title)

        if write_plots:
            # TODO and delete this extra hack
            if len(sequence_of_centers) > 2:
                extra = '_acrossblocks'
            else:
                extra = ''
            fname = to_filename(title + extra) + plot_format
            #
            print(f'writing to {fname}')
            fig.savefig(fname)
            #

    # TODO TODO change all parts that require squeeze=True to squeeze=False?
    if squeeze and len(sequence_of_centers) == 2:
        lr_matches = lr_matches[0]
        unmatched_left = unmatched_left[0]
        unmatched_right = unmatched_right[0]
        cost_totals = cost_totals[0]

    # TODO maybe stop returning unmatched_* . not sure it's useful.

    return lr_matches, unmatched_left, unmatched_right, cost_totals, fig


def stable_rois(lr_matches, verbose=False):
    """
    Takes a list of n_cells x 2 matrices, with each row taking an integer ROI
    label from one set of labels to the other.

    Input is as first output of correspond_rois.

    Returns:
    stable_cells: a n_stable_cells x (len(lr_matches) + 1) matrix, where rows
        represent different labels for the same real cells. Columns have the
        set of stable cells IDs, labelled as the inputs are.

    new_lost: a (len(lr_matches) - 1) length list of IDs lost when matching
        lr_matches[i] to lr_matches[i + 1]. only considers IDs that had
        been stable across all previous pairs of matchings.
    """
    # TODO TODO also test in cases where lr_matches is greater than len 2
    # (at least len 3)

    # TODO TODO also test when lr_matches is len 1, to support that case
    if len(lr_matches) == 1 or type(lr_matches) is not list:
        raise NotImplementedError

    orig_matches = lr_matches
    # Just since it gets written to in the loop.
    lr_matches = [m.copy() for m in lr_matches]

    stable = lr_matches[0][:,0]
    UNLABELLED = -1
    new_lost = []
    for i in range(len(lr_matches) - 1):
        matches1 = lr_matches[i]
        matches2 = lr_matches[i + 1]

        # These two columns should have the ROI / center numbers
        # represent the same real ROI / point coordinates.
        stable_1to2, m1_idx, m2_idx = np.intersect1d(
            matches1[:,1], matches2[:,0], return_indices=True)

        assert np.array_equal(matches1[m1_idx, 1], matches2[m2_idx, 0])

        curr_stable_prior_labels = matches1[m1_idx, 0]

        matches2[m2_idx, 0] = curr_stable_prior_labels

        # To avoid confusion / errors related too using old, now meaningless
        # labels.
        not_in_m2_idx = np.setdiff1d(np.arange(len(matches2)), m2_idx)
        assert (lr_matches[i + 1] == UNLABELLED).sum() == 0
        matches2[not_in_m2_idx] = UNLABELLED 
        assert (lr_matches[i + 1] == UNLABELLED).sum() == 2 * len(not_in_m2_idx)

        ids_lost_at_i = np.setdiff1d(stable, curr_stable_prior_labels)
        stable = np.setdiff1d(stable, ids_lost_at_i)
        new_lost.append(ids_lost_at_i)

        n_lost_at_i = len(ids_lost_at_i)
        if verbose and n_lost_at_i > 0:
            print(f'Lost {n_lost_at_i} ROI(s) between blocks {i} and {i + 1}')

    # TODO make a test case where the total number of *matched* rois is
    # conserved at each time step, but the matching makes the length of the
    # ultimate stable set reduce
    n_matched = [len(m) - ((m == UNLABELLED).sum() / 2) for m in lr_matches]
    assert len(stable) <= min(n_matched)

    stable_cells = []
    for i, matches in enumerate(lr_matches):
        # Because each of these columns will have been edited in the loop
        # above, to have labels matching the first set of center labels.
        _, _, stable_indices_i = np.intersect1d(stable, matches[:,0],
            return_indices=True)

        assert not UNLABELLED in matches[stable_indices_i, 0]
        orig_labels_stable_i = orig_matches[i][stable_indices_i, 0]
        stable_cells.append(orig_labels_stable_i)

    # This last column in the last element in the last of matches
    # was the only column that did NOT get painted over with the new labels.
    stable_cells.append(matches[stable_indices_i, 1])
    stable_cells = np.stack(stable_cells, axis=1)

    # might be redundant...
    stable_cells = stable_cells[np.argsort(stable_cells[:,0]), :]
    assert np.array_equal(stable_cells[:,0], stable)
    return stable_cells, new_lost


# TODO try to numba this
def renumber_rois2(matches_list, centers_list):
    id2frame_bounds = dict()
    id2indices = dict()
    next_id = 0
    seen_at_i = dict()
    for i in range(len(matches_list) + 1):
        if i not in seen_at_i:
            seen_at_i[i] = set()

        m = matches_list[min(i, len(matches_list) - 1)]
        for left, right in m:
            if i < len(matches_list):
                if left in seen_at_i[i]:
                    continue
                seen_at_i[i].add(left)
                roi_indices_across_frames = [left]
            else:
                if right in seen_at_i[i]:
                    continue
                roi_indices_across_frames = []

            first_frame = i
            # So that the frame counter increments as soon as we have one
            # "right" element (every match row must correspond to at least
            # two timepoints).
            j = i + 1
            while j <= len(matches_list):
                roi_indices_across_frames.append(right)
                last_frame = j

                if j in seen_at_i:
                    seen_at_i[j].add(right)
                else:
                    seen_at_i[j] = set()

                if j == len(matches_list):
                    break

                next_matches = matches_list[j]
                next_row_idx = np.argwhere(next_matches[:, 0] == right)
                if len(next_row_idx) == 0:
                    break

                next_row_idx = next_row_idx[0][0]
                left, right = next_matches[next_row_idx]
                j += 1

            assert (last_frame - first_frame + 1 ==
                len(roi_indices_across_frames)
            )
            id2frame_bounds[next_id] = (first_frame, last_frame)
            id2indices[next_id] = roi_indices_across_frames
            next_id += 1

        if i < len(matches_list):
            unmatched = np.setdiff1d(np.arange(len(centers_list[i])), m[:,0])
        else:
            unmatched = np.setdiff1d(np.arange(len(centers_list[i])), m[:,1])

        for u in unmatched:
            # TODO never need to check whether this is in seen, do i?
            id2frame_bounds[next_id] = (i, i)
            id2indices[next_id] = [u]
            next_id += 1

    assert set(id2frame_bounds.keys()) == set(id2indices.keys())
    centers_array = np.empty((len(centers_list), next_id,
        centers_list[0].shape[1])) * np.nan

    for roi_id in id2frame_bounds.keys():
        start, end = id2frame_bounds[roi_id]
        indices = id2indices[roi_id]
        centers_array[start:end+1, roi_id, :] = \
            [c[i] for c, i in zip(centers_list[start:end+1], indices)]

    # TODO assert min / max non-nan cover full frame for reasonable test data

    return centers_array


# TODO TODO should either this fn or correspond_rois try to handle the case
# where a cell drifts out of plane and then back into plane???
# possible? some kind of filtering?
def renumber_rois(matches_list, centers_list, debug_points=None, max_cost=None):
    """
    Each sequence of matched ROIs gets an increasing integer identifier
    (including length-1 sequences, i.e. unmatched stuff).

    Returns lists of IDs in each element of input list and centers,
    re-indexed with new IDs.
    """
    # TODO use this function inside stable_rois / delete that function
    # altogether (?)

    if type(matches_list) is not list or type(centers_list) is not list:
        raise ValueError('both input arguments must be lists')

    assert len(centers_list) == len(matches_list) + 1

    # Since they get written to in the loop.
    matches_list = [m.copy() for m in matches_list]
    centers_list = [c.copy() for c in centers_list]

    # TODO test case where input is not == np.arange(input.max())
    # (both just missing some less and w/ ids beyond len(centers) - 1)
    ids_list = []
    first_ids = np.arange(len(centers_list[0]))
    assert len(np.setdiff1d(matches_list[0][:,0], first_ids)) == 0
    ids_list.append(first_ids)
    next_new_id = first_ids.max() + 1
    print('next_new_id (after making first_ids):', next_new_id)
    ##next_new_id = matches_list[0][:,0].max() + 1

    #if len(centers_list[0]) > len(matches_list[0]):
    #    import ipdb; ipdb.set_trace()

    # TODO delete / put behind something like a `checks` flag
    assert max_cost is not None
    id2last_xy = {i: c for i, c in zip(first_ids, centers_list[0][:,:2])}
    id2src_history = {i:
        ['first_match' if i in matches_list[0][:,0] else 'new_first']
        for i in first_ids
    }
    id2idx_history = dict()
    for i in first_ids:
        try:
            idx = matches_list[0][i,0]
        except IndexError:
            idx = None
        id2idx_history[i] = [idx]
    assert set(id2src_history.keys()) == set(id2idx_history.keys())
    nonshared_m2_idx_list = []
    #

    for i in range(len(matches_list)):
        # These centers are referred to by the IDs in matches_list[i + 1][:, 1],
        # and (if it exists) matches_list[i + 2][:, 1]
        centers = centers_list[i + 1]
        matches1 = matches_list[i]

        '''
        # This includes stuff shared and stuff lost by m2.
        # The only thing this should not include is stuff that should get
        # a new ID in m2.
        centers_in_m1 = matches1[:, 1]

        # These include both things in matches2 (those not shared with matches1)
        # and things we need to generate new IDs for.
        only_new_centers_idx = np.setdiff1d(
            np.arange(len(centers)),
            centers_in_m1
        )
        # This should be of the same length as centers and should index each
        # value, just in a different order.
        new_center_idx = np.concatenate((
            centers_in_m1,
            only_new_centers_idx
        ))
        assert np.array_equal(
            np.arange(len(centers)),
            np.unique(new_center_idx)
        )

        # We are re-ordering the centers, so that they are in the same order
        # as the IDs (both propagated and new) at this timestep (curr_ids).
        centers_list[i + 1] = centers[new_center_idx]
        '''

        # TODO TODO TODO i think this is the heart of the problem
        # (b/c all problem indices were in the new_ids that got cut off
        # when trying to fit into smaller space of nonshared_m2_idx
        existing_ids = matches1[:, 0]
        #next_new_id = existing_ids.max() + 1
        ###n_new_ids = len(only_new_centers_idx)
        ##assert len(centers) - len(matches1) == n_new_ids
        n_new_ids = len(centers) - len(matches1)
        # Not + 1 because arange does not include the endpoint.
        stop = next_new_id + n_new_ids
        new_ids = np.arange(next_new_id, stop)
        for i, idl in enumerate(ids_list[::-1]):
            print(- (i + 1))
            overlap = set(new_ids) & set(idl)
            if len(overlap) > 0:
                print('overlap:', overlap)
                import ipdb; ipdb.set_trace()
        #
        print('i:', i)
        print('n_new_ids:', n_new_ids)
        print('stop:', stop)
        print('next_new_id:', next_new_id)
        print('next_new_id - existing_ids.max():',
            next_new_id - existing_ids.max()
        )
        next_new_id = stop

        curr_ids = np.concatenate((existing_ids, new_ids))
        assert len(curr_ids) == len(centers)
        assert len(curr_ids) == len(np.unique(curr_ids))

        # TODO this is the necessary condition for having current centers not
        # get mis-ordered, right?
        #assert np.array_equal(np.argsort(curr_ids), np.argsort(new_center_idx))
        #
        #import ipdb; ipdb.set_trace()

        #'''
        for j, (_id, cxy) in enumerate(zip(curr_ids, centers_list[i+1][:,:2])):
            if _id in id2last_xy:
                last_xy = id2last_xy[_id]
                dist = euclidean_dist(cxy, last_xy)
                try:
                    assert dist < max_cost
                except AssertionError:
                    print('')
                    #print(max_cost)
                    #print(dist)
                    print('id:', _id)
                    #print(last_xy)
                    #print(cxy)
                    if _id in new_ids:
                        fr = 'new'
                    elif _id in existing_ids:
                        fr = 'old'
                    else:
                        assert False
                    print(fr)

                    print(id2src_history[_id])
                    prev_idx = id2idx_history[_id]
                    print(prev_idx)
                    if len(prev_idx) > 0:
                        prev_idx = prev_idx[-1]
                        if prev_idx is not None:
                            # (previous entry in ids_list)
                            assert (np.argwhere(ids_list[i] == _id)[0][0] ==
                                prev_idx
                            )

                    import ipdb; ipdb.set_trace()

            id2last_xy[_id] = cxy
            # TODO delete these after debugging
            assert (_id in id2src_history) == (_id in id2idx_history)
            src_hist = 'new' if _id in new_ids else 'old'
            if _id in id2src_history:
                id2src_history[_id].append(src_hist)
                id2idx_history[_id].append(j)
            else:
                id2src_history[_id] = [src_hist]
                id2idx_history[_id] = [j]
            #
        #'''

        ids_list.append(curr_ids)

        # TODO TODO TODO some assertion that re-ordered centers are still
        # fully equiv to old centers, when indexing as they get indexed below?
        # though ordering across centers is what really matters...

        # TODO `i` as well?
        '''
        if debug_points and i + 1 in debug_points:
            print(f'I + 1 = {i + 1}')
            for pt_info in debug_points[i + 1]:
                roi_id = int(pt_info['name'])
                xy0 = pt_info['xy0']
                xy1 = pt_info['xy1']
                print('roi_id:', roi_id)
                print('xy0:', xy0)

                # TODO turn into assertion
                # shouldn't happen?
                if roi_id not in curr_ids:
                    print('not in curr_ids')
                    import ipdb; ipdb.set_trace()
                #

                if roi_id in matches1[:,0]:
                    print('in matches1[:,0] (old IDs)')
                elif roi_id in new_ids:
                    print('in new_ids!')
                else:
                    assert False, 'neither in old nor new ids'

                id_idx = np.argmax(curr_ids == roi_id)
                cxy = tuple(centers_list[i + 1][id_idx][:2])
                assert cxy == xy0
                lidx = pt_info.get('xy0_lidx')
                if lidx is not None:
                    xy0_was_matched = True
                    lpt = pt_info.get('xy0_lpt')
                    # so we can still index in to the non-re-ordered centers
                    assert tuple(centers[lidx, :2]) == xy0
                    print('xy0_lidx:', lidx)
                else:
                    xy0_was_matched = False

                #if xy0_was_matched:
                #    assert
                #import ipdb; ipdb.set_trace()

            #import ipdb; ipdb.set_trace()
        '''

        if i + 1 < len(matches_list):
            matches2 = matches_list[i + 1]
            assert len(matches2) <= len(centers)

            # These two columns should have the ROI / center numbers
            # represent the same real ROI / point coordinates.
            _, shared_m1_idx, shared_m2_idx = np.intersect1d(
                matches1[:,1], matches2[:,0], return_indices=True
            )
            assert np.array_equal(
                matches1[shared_m1_idx, 1],
                matches2[shared_m2_idx, 0]
            )
            prior_ids_of_shared = matches1[shared_m1_idx, 0]
            print(len(np.unique(matches2[:,0])) == len(matches2[:,0]))
            matches2[shared_m2_idx, 0] = prior_ids_of_shared
            print(len(np.unique(matches2[:,0])) == len(matches2[:,0]))

            nonshared_m2_idx = np.setdiff1d(np.arange(len(matches2)),
                shared_m2_idx
            )
            # ROIs unmatched in matches2 get any remaining higher IDs in new_ids
            # It is possible for there to be new_ids without any
            # nonshared_m2_idx.
            # TODO TODO TODO will we ever need to map from these new_ids that
            # run off the end to specific centers later?
            print('new_ids:', new_ids)
            print('new_ids[:len(nonshared_m2_idx)]:',
                new_ids[:len(nonshared_m2_idx)]
            )
            print('nonshared_m2_idx:', nonshared_m2_idx)
            print('matches2[nonshared_m2_idx, 0]:',
                matches2[nonshared_m2_idx, 0]
            )
            import ipdb; ipdb.set_trace()
            matches2[nonshared_m2_idx, 0] = new_ids[:len(nonshared_m2_idx)]
            assert len(np.unique(matches2[:,0])) == len(matches2[:,0])

    for i, (ids, cs) in enumerate(zip(ids_list, centers_list)):
        assert len(ids) == len(cs), f'(i={i}) {len(ids)} != {len(cs)}'

    centers_array = np.empty((len(centers_list), next_new_id,
        centers_list[0].shape[1])) * np.nan

    for i, (ids, centers) in enumerate(zip(ids_list, centers_list)):
        centers_array[i, ids, :] = centers

        if debug_points:
            if i in debug_points:
                for pt_info in debug_points[i]:
                    roi_id = int(pt_info['name'])
                    xy0 = pt_info['xy0']
                    cidx = point_idx(centers_array[i], xy0)
                    assert cidx == roi_id

    return centers_array


def roi_jumps(roi_xyd, max_cost):
    """
    Returns dict of first_frame -> list of (x, y, str(point idx)) for each
    time an ROI jumps by >= max_cost on consecutive frames.

    correspond_rois should have not matched these points.

    Output suitable for debug_points kwarg to correspond_rois
    """
    diffs = np.diff(roi_xyd[:, :, :2], axis=0)
    dists = np.sqrt((np.diff(roi_xyd[:, :, :2], axis=0) ** 2).sum(axis=2))
    # to avoid NaN comparison warning on >= (dists must be positive anyway)
    dists[np.isnan(dists)] = -1
    jumps = np.argwhere(dists >= max_cost)
    dists[dists == -1] = np.nan

    first_frames = set(jumps[:,0])
    debug_points = dict()
    for ff in first_frames:
        ff_rois = jumps[jumps[:,0] == ff, 1]
        # switching frame and roi axes so iteration is over rois
        # (zippable w/ ff_rois below)
        xys = np.swapaxes(np.round(roi_xyd[ff:ff+2, ff_rois, :2]
            ).astype(np.uint16), 0, 1
        )
        ff_info = []
        for roi, roi_xys in zip(ff_rois, xys):
            xy0, xy1 = roi_xys
            pt_info = {'name': str(roi), 'xy0': tuple(xy0), 'xy1': tuple(xy1)}
            ff_info.append(pt_info)
        debug_points[ff] = ff_info

    return debug_points


# TODO TODO use in unit tests of roi tracking w/ some real data as input
def check_no_roi_jumps(roi_xyd, max_cost):
    assert len(roi_jumps(roi_xyd, max_cost)) == 0


# TODO TODO TODO re-enable checks!!!
def correspond_and_renumber_rois(roi_xyd_sequence, debug=False, checks=False,
    use_renumber_rois2=True, **kwargs):

    max_cost = kwargs.get('max_cost')
    if max_cost is None:
        # TODO maybe switch to max / check current approach yields results
        # just as reasonable as those w/ larger max_cost
        min_diam = min([xyd[:, 2].min() for xyd in roi_xyd_sequence])
        # + 1 b/c cost == max_cost is thrown out
        max_cost = min_diam / 2 + 1
        kwargs['max_cost'] = max_cost

    # TODO fix what seems to be making correspond_rois fail in case where
    # diameter info is also passed in (so it can be used here and elsewhere
    # w/o having to toss that data first)
    roi_xy_seq = [xyd[:, :2] for xyd in roi_xyd_sequence]

    lr_matches, _, _, _, _ = correspond_rois(roi_xy_seq, squeeze=False,
    #    verbose=debug, show=debug, write_plots=False, **kwargs
        verbose=debug, show=False, write_plots=False, **kwargs
    )
    '''
    if debug:
        # For stuff plotted in correspond_rois
        plt.show()
    '''

    debug_points = kwargs.get('debug_points')
    if use_renumber_rois2:
        new_roi_xyd = renumber_rois2(lr_matches, roi_xyd_sequence)
    else:
        new_roi_xyd = renumber_rois(lr_matches, roi_xyd_sequence,
            debug_points=debug_points, max_cost=max_cost
        )
    if checks:
        check_no_roi_jumps(new_roi_xyd, max_cost)

    return new_roi_xyd


# TODO add nonoverlap constraint? somehow make closer to real data?
# TODO use this to test gui/fitting/tracking
def make_test_centers(initial_n=20, nt=100, frame_shape=(256, 256), sigma=3,
    exlusion_radius=None, p=0.05, max_n=None, round_=False, diam_px=20,
    add_diameters=True, verbose=False):
    # TODO maybe adapt p so it's the p over the course of the
    # nt steps, and derivce single timestep p from that?

    if exlusion_radius is not None:
        raise NotImplementedError

    # So that we can pre-allocate the center coordinates over time
    # (rather than having to figure out how many were added by the end,
    # and then pad all the preceding arrays of centers w/ NaN)
    if p:
        max_n = 2 * initial_n
    else:
        # Don't need to allocate extra space if the number of ROIs is
        # deterministic.
        max_n = initial_n

    assert len(frame_shape) == 2
    assert frame_shape[0] == frame_shape[1]
    d = frame_shape[0]
    max_coord = d - 1

    # Also using this for new centers gained while iterating.
    initial_centers = np.random.randint(d, size=(max_n, 2))

    # TODO more idiomatic numpy way to generate cumulative noise?
    # (if so, just repeat initial_centers to generate centers, and add the
    # two) (maybe not, with my constraints...)
    # TODO TODor generate inside the loop (only as many as non-NaN, and only
    # apply to non NaN)
    xy_steps = np.random.randn(nt - 1, max_n, 2) * sigma

    next_trajectory_idx = initial_n
    centers = np.empty((nt, max_n, 2)) * np.nan
    centers[0, :initial_n] = initial_centers[:initial_n]
    # TODO should i be generating the noise differently, so that the x and y
    # components are not independent (so that if deviation is high in one,
    # it's more likely to be lower in other coordinate, to more directly
    # constrain the distance? maybe it's just a scaling thing though...)
    for t in range(1, nt):
        # TODO maybe handle this differently...
        if p and next_trajectory_idx == max_n:
            raise RuntimeError(f'reached max_n ({max_n}) on step {t} '
                f'(before {nt} requested steps'
            )
            #break

        centers[t] = centers[t - 1] + xy_steps[t - 1]

        # TODO make sure NaN stuff handled correctly here
        # The centers should stay within the imaginary frame bounds.
        centers[t][centers[t] > max_coord] = max_coord
        centers[t][centers[t] < 0] = 0

        if not p:
            continue

        lose = np.random.binomial(1, p, size=max_n).astype(np.bool)
        if verbose:
            nonnan = ~ np.isnan(centers[t,:,0])
            print('# non-nan:', nonnan.sum())
            n_lost = (nonnan & lose).sum()
            if n_lost > 0:
                print(f't={t}, losing {n_lost}')
        centers[t][lose] = np.nan

        # TODO TODO note: if not allowed to fill NaN that come from losing
        # stuff, then max_n might more often limit # unique rather than #
        # concurrent tracks... (and that would prob make a format more close to
        # what i was already implementing in association code...)
        # maybe this all means i could benefit from a different
        # representation...
        # one more like id -> (start frame, end frame, coordinates)

        # Currently, giving any new trajectories different indices (IDs)
        # from any previous trajectories, by putting them in ranges that
        # had so far only had NaN. As association code may be, this also
        # groups new ones in the next-unused-integer-indices, rather
        # than giving each remaining index a chance.
        # To justify first arg (n), imagine case where initial_n=0 and
        # max_n=1.
        n_to_gain = np.random.binomial(max_n - initial_n, p)
        if n_to_gain > 0:
            if verbose:
                print(f't={t}, gaining {n_to_gain}')

            first_ic_idx = next_trajectory_idx - initial_n
            centers[t][next_trajectory_idx:next_trajectory_idx + n_to_gain] = \
                initial_centers[first_ic_idx:first_ic_idx + n_to_gain]
            next_trajectory_idx += n_to_gain

    assert len(centers) == nt

    # This seems to convert NaN to zero...
    if round_:
        centers = np.round(centers).astype(np.uint16)

    if add_diameters:
        roi_diams = np.expand_dims(np.ones(centers.shape[:2]) * diam_px, -1)
        centers = np.concatenate((centers, roi_diams), axis=-1)

    # TODO check output is in same kind of format as output of my matching fns

    return centers


# Adapted from Vishal's answer at https://stackoverflow.com/questions/287871
_color_codes = {
    'red': '31',
    'green': '32',
    'yellow': '33',
    'blue': '34',
    'cyan': '36'
}
def start_color(color_name):
    try:
        color_code = _color_codes[color_name]
    except KeyError as err:
        print('Available colors are:')
        pprint(list(_color_codes.keys()))
        raise
    print('\033[{}m'.format(color_code), end='')


def stop_color():
    print('\033[0m', end='')


def print_color(color_name, *args, **kwargs):
    start_color(color_name)
    print(*args, **kwargs, end='')
    stop_color()


def latest_trace_pickles():
    # TODO say which data is searched/included in this fn
    """Returns (date, fly, id) indexed DataFrame w/ filename and timestamp cols.

    Only returns rows for filenames that had the latest timestamp for the
    combination of index values.
    """
    def vars_from_filename(tp_path):
        final_part = split(tp_path)[1][:-2]

        # Note that we have lost any more precise time resolution, so an
        # exact search for this timestamp in database would fail.
        n_time_chars = len('YYYYMMDD_HHMM')
        run_at = pd.Timestamp(datetime.strptime(final_part[:n_time_chars],
            '%Y%m%d_%H%M'
        ))

        parts = final_part.split('_')[2:]
        date = pd.Timestamp(datetime.strptime(parts[0], date_fmt_str))
        fly_num = int(parts[1])
        thorimage_id = '_'.join(parts[2:])
        return date, fly_num, thorimage_id, run_at, tp_path

    # TODO maybe replace w/ `recording_cols`, which is currently the same except
    # the first element is 'prep_date' (maybe generalize representation to use
    # either? or migrate all data to using just date, change `recording_cols`,
    # then use here?)
    keys = ['date', 'fly_num', 'thorimage_id']
    tp_root = join(analysis_output_root(), 'trace_pickles')
    tp_data = [vars_from_filename(f) for f in glob.glob(join(tp_root, '*.p'))]
    if len(tp_data) == 0:
        raise IOError(f'no trace pickles found under {tp_root}')

    df = pd.DataFrame(columns=keys + ['run_at', 'trace_pickle_path'],
        data=tp_data
    )

    unique_len_before = len(df[keys].drop_duplicates())
    latest = df.groupby(keys).run_at.idxmax()
    df.drop(index=df.index.difference(latest), inplace=True)
    assert len(df[keys].drop_duplicates()) == unique_len_before

    df.set_index(keys, inplace=True)
    return df


# TODO TODO kwarg to have this replace the multiindex levels / columns values it is
# derived from (and thread through add_fly_id/add_recording_id)
def add_group_id(df, group_keys, name=None, start_at_one=True):
    """Adds integer column to df to identify unique combinations of group_keys.
    """
    if name is None:
        name = '_'.join(group_keys) + '_id'
    assert name not in df.columns
    df[name] = df.groupby(group_keys).ngroup()

    if start_at_one:
        df[name] = df[name] + 1

    return df


# TODO replace hardcoded recording_cols[:2] w/ kwarg that defaults to None where None
# gets replaced by current hardcoded value
def add_fly_id(df):
    name = 'fly_id'
    return add_group_id(df, recording_cols[:2], name=name)


def add_recording_id(df):
    name = 'recording_id'
    return add_group_id(df, recording_cols, name=name)


def thor2tiff(image_dir, output_name=None, output_basename=None,
    output_dir=None, if_exists='err', verbose=True):
    """Converts ThorImage .raw file to .tif file in same directory

    Args:
        if_exists (str): 'err', 'overwrite', or 'ignore'
    """
    assert if_exists in ('err', 'overwrite', 'ignore')

    if all([x is not None for x in (output_name, output_basename)]):
        raise ValueError('only pass at most one of output_name or output_basename')

    # TODO .tif or .tiff?
    tiff_ext = '.tif'
    if output_name is None and output_basename is None:
        output_basename = f'raw{tiff_ext}'

    if output_name is None:
        if output_dir is None:
            output_dir = image_dir

        assert isdir(output_dir), f'output_dir={output_dir} was not a directory'
        output_name = join(output_dir, output_basename)

    # TODO maybe options to just ignore and NOOP if exists
    if exists(output_name):
        if if_exists == 'err':
            raise IOError(f'{output_name} exists (set if_exists to either '
                "'overwrite' or 'ignore' for other behavior"
            )

        elif if_exists == 'ignore':
            if verbose:
                print(f'{output_name} exists. not regenerating tiff.')
            return

    # TODO maybe also load metadata like fps (especially stuff, as w/ fps, that isn't
    # already baked into the TIFF, assuming the TIFF is saved correctly. so not
    # including stuff like z, c, xy), and print w/ -v flag?

    if verbose:
        print('Reading RAW movie...', flush=True, end='')

    from_raw = thor.read_movie(image_dir)

    if verbose:
        print(' done', flush=True)

    # TODO TODO TODO try to figure out if anything can be done about tifffile
    # using so much memory on writing (says "Killed" and exits in the middle of
    # writing when trying to write what should be a ~5.5GB movie when i have
    # close to 20GB of RAM free...). maybe memory profile my own code to see if
    # i'm doing something stupid.
    # TODO test read_movie on all thorimage .raw outputs i have to check which can
    # currently reproduce this issue

    if verbose:
        print(f'Writing TIFF to {output_name}...', flush=True, end='')

    write_tiff(output_name, from_raw)

    if verbose:
        print(' done', flush=True)

