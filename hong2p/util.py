"""
Common functions for dealing with Thorlabs software output / stimulus metadata /
our databases / movies / CNMF output.
"""

import os
from os.path import join, split, exists, sep, isdir, getmtime, splitext
from pathlib import Path
import pickle
import platform
import sys
from types import ModuleType
from datetime import datetime
import warnings
from pprint import pprint
import glob
import re
import hashlib
import functools
from typing import Optional, Tuple, List, Generator, Sequence, Union, Any
import xml.etree.ElementTree as etree
from urllib.error import URLError

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tifffile

from hong2p import matlab, db, thor, olf
from hong2p.err import NoStimulusFile, TooManyStimulusFiles
from hong2p.types import (Pathlike, PathPair, Datelike, FlyNum, DateAndFlyNum,
    DataFrameOrDataArray
)

# Note: many imports were pushed down into the beginnings of the functions that
# use them, to reduce the number of hard dependencies.


# 1 to indicate computer is an acquisition computer, 0/unset otherwise.
ACQUISITION_ENV_VAR = 'HONG2P_ACQUISITION'

def is_acquisition_host() -> bool:
    _is_acquisition_host = False

    if ACQUISITION_ENV_VAR in os.environ:
        val = os.environ[ACQUISITION_ENV_VAR]
        if val == '1':
            _is_acquisition_host = True
        elif val == '0':
            _is_acquisition_host = False
        else:
            raise ValueError(f'invalid value {val} for {ACQUISITION_ENV_VAR}. must be '
                '0 or 1.'
            )
    else:
        # Returns the system/OS name, such as 'Linux','Darwin','Windows'
        if platform.system() == 'Windows':
            warnings.warn('assuming this is an acquisition computer because it is '
                f'windows. set {ACQUISITION_ENV_VAR} to 0/1 to silence this.'
            )
            _is_acquisition_host = True

    return _is_acquisition_host


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

STIMFILE_DIR_ENV_VAR = 'HONG2P_STIMFILE_DIR'

_fast_data_root: Optional[Path] = (Path(os.environ.get(FAST_DATA_ROOT_ENV_VAR))
    if FAST_DATA_ROOT_ENV_VAR in os.environ else None
)
if _fast_data_root is not None and not _fast_data_root.is_dir():
    raise IOError(f'{FAST_DATA_ROOT_ENV_VAR} set but is not a directory')

np.set_printoptions(precision=2)

# TODO maybe move all of these to __init__.py, or at least expose them there?
# or maybe to a hong2p.py module (maybe importing all of its contents in
# __init__.py ?)
# TODO migrate all 'prep_date' -> 'date'? (seems i already use 'date' in a lot
# of places...)
recording_cols = [
    # TODO delete after refactoring all code that used this / add_fly_id
    #'prep_date',
    'date',
    'fly_num',
    'thorimage_id'
]
# TODO delete [/ update to ('panel', 'odor1', 'odor2', 'repeat')]
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
def set_data_root(new_data_root: Pathlike) -> None:
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


def data_root(verbose: bool = False) -> Path:
    global _data_root

    if _data_root is None:
        # TODO print priority order of these env vars in any failure below
        # TODO TODO refactor (to loop, w/ break, probably) to also check if directories
        # exist before picking one to use?
        prefix = None

        if DATA_ROOT_ENV_VAR in os.environ:
            data_root_val = os.environ[DATA_ROOT_ENV_VAR]
            root = Path(data_root_val)
            source = DATA_ROOT_ENV_VAR

            if verbose:
                print(f'found {DATA_ROOT_ENV_VAR}={data_root_val}')

        elif NAS_PREFIX_ENV_VAR in os.environ:
            nas_prefix = os.environ[NAS_PREFIX_ENV_VAR]
            root = Path(nas_prefix) / NAS_PATH_TO_HONG2P_DATA
            source = NAS_PREFIX_ENV_VAR

            # TODO set flag if we are using NAS and only raise IOPerformanceWarning if
            # we actually try to load anything other than stimfiles from it
            # (assuming that if set via any other variables, it's at least fast enough
            # to not need warning, whether from the *_FAST_* variable or not)

            if verbose:
                print(f'did not find {DATA_ROOT_ENV_VAR}')
                print(f'found {NAS_PREFIX_ENV_VAR}={nas_prefix}')

        else:
            raise IOError('either set one of the environment variables '
                f'({DATA_ROOT_ENV_VAR} or {NAS_PREFIX_ENV_VAR}) or call '
                'hong2p.util.set_data_root(<data root>) before this data_root call'
            )

        _data_root = root

        if not _data_root.is_dir():
            raise IOError(f'data root expected at {_data_root}, but no directory exists'
                f' there!\nDirectory chosen from environment variable {source}'
            )

    # TODO err if nothing in data_root, saying which env var to set and how
    return _data_root


def check_dir_exists(fn_returning_dir):

    @functools.wraps(fn_returning_dir)
    def optionally_checked_fn_returning_dir(*args, check=True, create=False, **kwargs):

        directory = fn_returning_dir(*args, **kwargs)

        if not isdir(directory):

            if create:
                print(f'Creating directory {directory}')
                # This will error if `directory` points to something that exists that
                # is NOT a directory (as intended).
                os.makedirs(directory)

            elif check:
                raise IOError(f'directory {directory} (returned by '
                    f'{fn_returning_dir.__name__}) does not exist! check the relevant '
                    'environment variables are set correctly.'
                )

        return directory

    return optionally_checked_fn_returning_dir


# TODO (for both below) support a local and a remote one ([optional] local copy
# for faster repeat analysis)?
# TODO use env var like kc_analysis currently does for prefix after refactoring
# (include mb_team in that part and rename from data_root?)
@check_dir_exists
def raw_data_root(root: Optional[Pathlike] = None, **kwargs) -> Path:

    # TODO TODO also default to _fast_data_root?

    if root is None:
        root = data_root(**kwargs)

    return root / 'raw_data'


# TODO kwarg / default to makeing dir if not exist (and for similar fns above)?
@check_dir_exists
def analysis_intermediates_root() -> Path:
    # TODO probably prefer using $HONG2P_DATA over os.getcwd() (assuming it's not on NAS
    # and it therefore acceptably fast if not instead using $HONG_NAS)
    if _fast_data_root is None:
        warnings.warn(f'environment variable {FAST_DATA_ROOT_ENV_VAR} not set, so '
            'storing analysis intermediates under current directory'
        )
        intermediates_root_parent = Path.cwd()
    else:
        intermediates_root_parent = _fast_data_root

    intermediates_root = intermediates_root_parent / 'analysis_intermediates'
    return intermediates_root


@check_dir_exists
def stimfile_root(**kwargs) -> Path:
    return Path(os.environ.get(STIMFILE_DIR_ENV_VAR,
        data_root(**kwargs) / 'stimulus_data_files'
    ))


# TODO replace this w/ above (need to change kc_natural_mixes / natural_odors, or at
# least pin an older version of hong2p for them)
@check_dir_exists
def analysis_output_root(**kwargs) -> Path:
    return data_root(**kwargs) / 'analysis_output'


class IOPerformanceWarning(Warning):
    """Warning that data does not seem to be read/written from fast storage
    """


def format_date(date: Datelike) -> str:
    """
    Takes a pandas Timestamp or something that can be used to construct one
    and returns a str with the formatted date.

    Used to name directories by date, etc.
    """
    return pd.Timestamp(date).strftime(date_fmt_str)


def format_timestamp(timestamp: Datelike) -> str:
    # TODO example of when this should be used. maybe explicitly say use
    # `format_date` for dates
    """Returns human-readable str for timestamp accepted by `pd.Timestamp`, to minute.

    Ex: '2022-04-07 16:53'
    """
    return str(pd.Timestamp(timestamp))[:16]


# TODO maybe rename to [get_]fly_basedir?
def get_fly_dir(date: Datelike, fly: FlyNum) -> Path:
    """Returns str path fragment as YYYY-MM-DD/<n> for variety of input types
    """
    if not type(date) is str:
        date = format_date(date)

    if not type(fly) is str:
        fly = str(int(fly))

    return Path(date, fly)


def raw_fly_dir(date: Datelike, fly: FlyNum, *, warn: bool = True, short: bool = False
    ) -> Path:
    """
    Args:
        short: If True, returns in format YYYY-MM-DD/<fly #>/<ThorImage dir>, without
            the prefix specifying the full path. Intended for creating more readable
            paths, where absolute paths are not required.
    """
    raw_fly_basedir = get_fly_dir(date, fly)

    # TODO TODO maybe refactor for more granularity (might need to change a lot of usage
    # of data_root() and stuff that uses it though... perhaps also functions that
    # operate on directories like the fn to pair thor dirs)
    # TODO TODO move this logic into raw_data_root?
    if _fast_data_root is not None:
        fast_raw_fly_dir = raw_data_root(root=_fast_data_root) / raw_fly_basedir
        if fast_raw_fly_dir.is_dir():
            return fast_raw_fly_dir
        else:
            if warn:
                warnings.warn(f'{FAST_DATA_ROOT_ENV_VAR} set ({_fast_data_root}) but '
                    f'raw data directory for fly ({date}, {fly}) did not exist there',
                    IOPerformanceWarning
                )

    return raw_data_root() / raw_fly_basedir


def thorimage_dir(date, fly, thorimage_id, **kwargs) -> Path:
    return raw_fly_dir(date, fly, **kwargs) / thorimage_id


def thorsync_dir(date, fly, base_thorsync_dir, **kwargs) -> Path:
    return raw_fly_dir(date, fly, **kwargs) / base_thorsync_dir


# TODO test (also test w/ some that should be under fast data root)
def thorimage_dir_input(fn):
    """Wraps functions taking ThorImage path and adds option to call via fly keys

    Fly keys are (date, fly_num, thorimage_id)
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):

        if len(args) == 3:
            date, fly, thorimage_id = args

            # TODO is this getting fast dir when it is available?
            image_dir = thorimage_dir(date, fly, thorimage_id)

        elif len(args) == 1:
            image_dir = args[0]

        else:
            # TODO maybe just check if dir args[0] exists / if args[0:3] are all
            # parseable, and then pass thru any remaining args? or would it get too
            # complicated for too little benefit?
            raise ValueError('functions wrapped with thorimage_dir_input must be '
                'passed either date, fly_num, thorimage_id OR thorimage_dir'
            )

        # TODO also convert to Path, to avoid need in any wrapped fns?

        return fn(image_dir, **kwargs)

    return wrapped_fn


# TODO use new name in al_pair_grids + also handle fast data dir here.
# (maybe always returning directories under fast? or kwarg to behave that way?)
def analysis_fly_dir(date, fly) -> Path:
    return analysis_output_root() / get_fly_dir(date, fly)


# TODO maybe this should stay returning a str? i'm assuming a lot of what i do with this
# is print it / format it? or change to Path to be consistent w/ other path fns now?
def shorten_path(full_path: Pathlike, n_parts=3) -> str:
    """Returns a string containing just the last n_parts (default=3) of input path.

    For making IDs / easier-to-read paths, when the full path isn't required.
    """
    return '/'.join(Path(full_path).parts[-n_parts:])


def print_thor_paths(image_dir: Pathlike, sync_dir: Pathlike, print_full_paths=True
    ) -> None:

    if print_full_paths:
        image_dir_toprint = image_dir
        sync_dir_toprint = sync_dir
    else:
        image_dir_toprint = shorten_path(image_dir)
        sync_dir_toprint = shorten_path(sync_dir)

    print('thorimage_dir:', image_dir_toprint)
    print('thorsync_dir:', sync_dir_toprint)


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
def date_fly_list2paired_thor_dirs(date_fly_list, n_first=None, print_full_paths=True,
    verbose=False, **pair_kwargs) -> Generator[Tuple[DateAndFlyNum, PathPair], None,
    None]:
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
                print_thor_paths(image_dir, sync_dir, print_full_paths=print_full_paths)

            yield (date, fly_num), (image_dir, sync_dir)
            n += 1


# TODO TODO merge date_fly_list2paired_thor_dirs into this or just delete that and add
# kwarg here to replace above (too similar)
def paired_thor_dirs(matching_substrs: Optional[Sequence[str]] = None,
    start_date: Optional[Datelike] = None, end_date: Optional[Datelike] = None,
    n_first: Optional[int] = None, skip_redone: bool = True, verbose: bool = False,
    print_skips: bool = True, print_fast: bool = True, print_full_paths: bool = True,
    **pair_kwargs) -> Generator[Tuple[DateAndFlyNum, PathPair], None, None]:
    # TODO add code example to doc
    """Generates tuples of fly metadata and ThorImage output paths, in acquisition order

    Args:
        matching_substrs: If passed, only experiments whose ThorImage path contains at
            least one of these substring will be included.

        n_first: If passed, only up to this many of pairs are enumerated.
            Intended for testing on subsets of data.

        verbose: If True, prints the fly/ThorImage/ThorSync directories as they are
            being iterated over.

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
        # TODO replace join w/ pathlib alternative
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
            redone_thorimage_dirs = {
                # TODO pathlib alternatives for these str ops?
                Path(str(ti)[:-len(redo_suffix)]) for ti, td in paired_dirs
                if str(ti).endswith(redo_suffix)
            }

        for image_dir, sync_dir in sorted(paired_dirs, key=lambda p:
            thor.get_thorimage_time(p[0])):

            if matching_substrs is not None and len(matching_substrs) > 0:
                if not any([s in str(image_dir) for s in matching_substrs]):
                    if verbose and print_skips:
                        print(f'skipping {image_dir} because did not contain >=1 of '
                            f'matching_substrs="{matching_substrs}"'
                        )
                    continue

            if skip_redone and image_dir in redone_thorimage_dirs:
                if verbose and print_skips:
                    print(f'skipping {image_dir} because matching redo exists\n')

                continue

            if n_first is not None and n >= n_first:
                return

            if verbose:
                print_thor_paths(image_dir, sync_dir, print_full_paths=print_full_paths)

            yield (date, fly_num), (image_dir, sync_dir)
            n += 1


def _raw_data_root_grandchildren() -> Generator[Path, None, None]:
    return raw_data_root().glob('*/*/')


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


def _all_paired_thor_dirs(skip_errors=True, **kwargs) -> List[PathPair]:
    """
    Returns a list of all (ThorImage, ThorSync) directories that can be paired
    (i.e. determined to come from the same experiment) and that are both
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


def _stimfile_dir(stimfile_dir: Optional[Pathlike] = None) -> Path:
    if stimfile_dir is None:
        stimfile_dir = stimfile_root()
    else:
        stimfile_dir = Path(stimfile_dir)
        if not stimfile_dir.is_dir(stimfile_dir):
            raise IOError(f'passed stimfile_dir={stimfile_dir} is not a directory!')

    return stimfile_dir


def shorten_stimfile_path(stimfile_path, stimfile_dir: Optional[Pathlike] = None):
    """Shortens absolute stimulus YAML path to one relative to stimfile_dir.
    """
    stimfile_dir = _stimfile_dir(stimfile_dir)
    # TODO convert to a pathlib call
    assert str(stimfile_path).startswith(str(stimfile_dir))

    # + 1 to also exclude the os.sep character separating parent dir and relative
    # stimfile path.
    # TODO convert to pathlib
    return stimfile_path[(len(str(stimfile_dir)) + 1):]


def stimulus_yaml_from_thorimage(thorimage_dir_or_xml, stimfile_dir=None):
    """Returns absolute path to stimulus YAML file from note field in ThorImage XML.

    Args:
        thorimage_dir_or_xml: path to ThorImage output directory or XML Element
            containing parsed contents of the corresponding Experiment.xml file.

        stimfile_dir (str): (optional) directory containing stimulus .yaml files.
            If not passed, `stimfile_root()` is used.

    Raises:
        IOError if stimulus file directory does not exist
        TooManyStimulusFiles if multiple substrings of note field end with .yaml
        NoStimulusFile if no substrings of note field end with .yaml

    XML should contain a manually-entered path relative to where the olfactometer code
    that generated it was run, but assuming it was copied to the appropriate location
    (directly under `stimfile_dir` if passed or `stimfile_root()` otherwise), this
    absolute path should exist.
    """
    stimfile_dir = _stimfile_dir(stimfile_dir)

    notes = thor.get_thorimage_notes(thorimage_dir_or_xml)

    if isinstance(thorimage_dir_or_xml, etree.Element):
        name = thor.get_thorimage_name(thorimage_dir_or_xml)
    else:
        name = thorimage_dir_or_xml

    yaml_path = None
    parts = notes.split()
    for p in parts:
        p = p.strip()
        if p.endswith('.yaml'):
            if yaml_path is not None:
                raise TooManyStimulusFiles(
                    f'{name}: encountered multiple *.yaml substrings!'
                )

            yaml_path = p

    if yaml_path is None:
        raise NoStimulusFile(f'{name}: no string ending in .yaml found in ThorImage '
            'note field'
        )

    assert yaml_path is not None

    # TODO change data that has this to expand paths + delete this hack
    if '""' in yaml_path:
        date_str = '_'.join(yaml_path.split('_')[:2])
        old_yaml_path = yaml_path
        yaml_path = yaml_path.replace('""', date_str)

        warnings.warn(f'{name}: replacing of stimulus YAML path of {old_yaml_path} '
            f'with {yaml_path}'
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

    yaml_abspath = join(stimfile_dir, yaml_path)

    if not exists(yaml_abspath):
        raise IOError(f'{name} references {yaml_path}, but it did not '
            f'exist under stimfile_dir={stimfile_dir}'
        )

    return yaml_abspath


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
    yaml_data, odor_lists = olf.load_stimulus_yaml(yaml_path)

    return yaml_path, yaml_data, odor_lists


def most_recent_contained_file_mtime(path: Pathlike, recurse: bool = True,
    verbose: bool = False) -> Optional[float]:
    """Recursively find the `os.path.getmtime` of the most recently modified file

    Args:
        path: directory within which to check mtime of files

        recurse: whether to check files under descendant directories of input

        verbose: prints which file had the most recent mtime (mostly for debugging)

    Returns None if there are no files in the directory.

    Testing on Ubuntu, this does not recurse into symlinks to directories, as I want for
    at least current use case.
    """
    # TODO maybe need to actively exclude mtime on symlinks (to directories at least?)
    # because it will still have an mtime, but i forget whether it tracks the mtime of
    # the referenced directory, or whether it is just when the link is created...
    if recurse:
        files = [x for x in Path(path).rglob('*') if x.is_file()]
    else:
        files = [x for x in Path(path).glob('*') if x.is_file()]

    if len(files) == 0:
        # TODO err if path doesn't exist (/ is not dir)? warn?
        # TODO also print if verbose and here
        return None

    if verbose:
        mtimes = [getmtime(f) for f in files]
        max_mtime = max(mtimes)
        most_recent_file = files[mtimes.index(max_mtime)]
        # TODO also print modification time, appropriately formatted?
        print(f'most recent modified file in {path} ({recurse=}): '
            f'{most_recent_file.relative_to(path)}'
        )
        return max_mtime

    return max(getmtime(f) for f in files)


# TODO support xarray.DataArrays? (isnull().sum().item(0) should work)
# TODO factor both to a pandas module
def num_null(df: pd.DataFrame) -> int:
    return df.isna().sum().sum()

# TODO an issue that this returns a float type (at least in case where input is empty)?
def num_notnull(df: pd.DataFrame) -> int:
    return df.notna().sum().sum()


# TODO maybe accept dict of names / values? which pd fn to copy the interfact of
# names/values from (DataFrame creation probably)?
# TODO TODO support xarray dataarrays?
# TODO test
# TODO TODO TODO add support for sequence of values (for a given name (/names entry)).
# at the moment, seems only scalar values supported. or at least rename to indicate it
# only supports scalars in values.
def addlevel(df: pd.DataFrame, names, values, *, axis='index'):
    """Add level(s) to pandas MultiIndex

    Intended to be an inverse to pandas.DataFrame.droplevel. pandas.DataFrame.set_index
    with `append=True` would work *except* that there is no `axis` kwarg to that
    function, so it does not work for the columns. pandas.DataFrame.unstack is almost
    what I would want, but it can seemingly arbitrarily change order of rows.

    Args:
        df: DataFrame to add MultiIndex levels to
        names: `str`/sequence-of-`str` name(s) for the new levels
        values: values for the new levels. If `names` is a sequence, this should be of
            the same length.
        axis: 0/'index' or 1/'columns', defauling to 'index' as in pandas

    Returns: DataFrame with MultiIndex containing names/levels from input
    """
    # https://stackoverflow.com/questions/14744068

    if isinstance(names, str):
        names = [names]
        values = [values]

    for name, value in list(zip(names, values))[::-1]:
        df = pd.concat([df], names=[name], keys=[value], axis=axis)

    return df


# TODO test w/ ndarray/DataFrame[/DataArray?] input + typehint
# TODO type hint so that output type is same as input type (TypeVar?)
# TODO already have something like this in diff_dataframes? (didn't seem like it, but
# could maybe use this here)
def nan_eq(arr1, arr2):
    """Elementwise equality of arrays, but True if both are NaN at a location.

    Normally, NaN != NaN, but this is not the behavior I want when trying to check which
    parts of an array/DataFrame were changed by an operation.
    """
    # https://stackoverflow.com/questions/19322506
    # Psuedocode: (x == y) or (isnan(x) and isnan(y))
    return (arr1 == arr2) | ( (arr1 != arr2) & (arr2 != arr2) )

    # This is ugly, but should let this work on very large arrays without causing any
    # memory issues (not sure I need that...).
    # https://stackoverflow.com/questions/10819715
    # TODO does numexpr work w/ DataFrame inputs? returns same type?
    #return numexpr.evaluate('(arr1 == arr2) | ( (arr1 != arr1) & (arr2 != arr2) )')


def is_array_sorted(array: np.ndarray) -> bool:
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


# TODO transition everytion to as if include_val==True?
def const_ranges(xs: Sequence, include_val=False) -> Union[
    List[Tuple[int, int]], List[Tuple[Any, int, int]]
    ]:
    """Returns tuples of indices for largest contiguous constant-value ranges in input.

    >>> const_ranges(['MCH', 'MCH', 'OCT', 'OCT'])
    [(0, 1), (2, 3)]
    """
    # Since I want to allow xs to contain None, the sentinel I would have otherwise used
    # https://python-patterns.guide/python/sentinel-object/
    sentinel = object()

    x_prev = sentinel
    ranges = []
    curr_start = 0
    for i, x in enumerate(xs):
        if x_prev is not sentinel and x != x_prev:
            if i > 0:
                if not include_val:
                    range_data = (curr_start, i - 1)
                else:
                    range_data = (x_prev, curr_start, i - 1)

                ranges.append(range_data)

            curr_start = i

        x_prev = x

    if len(xs) > 0:
        x = xs[-1]
        i = len(xs)
        if not include_val:
            range_data = (curr_start, i - 1)
        else:
            # Since elements are only added above at level *changes*, we will always
            # need to add one at the end of the list.
            range_data = (x, curr_start, i - 1)

        ranges.append(range_data)

    return ranges


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


# TODO move to olf/io (along w/ other similar fns)?
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


# TODO delete
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
# TODO maybe replace w/ (/ borrow some ideas from) al_analysis.delta_f_over_f
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


# TODO TODO delete no_append_gid code? still used (at least in mb_team_gsheet below)?
# (also seems like natural_odors/literature_data expects this, tho kwarg not passed.
# maybe default changed?)
# TODO get rid of gid kwarg and just require it to be in url?
def gsheet_csv_export_link(file_with_edit_link: Union[str, Pathlike],
    gid: Optional[int] = None, no_append_gid: bool = False) -> str:
    """
    Takes a gsheet link copied from browser while editing it, and returns a
    URL suitable for reading it as a CSV into a DataFrame.

    GID seems to default to 0 for the first sheet, but seems unpredictable for further
    sheets in the same document, though you can extract it from the URL in those cases.
    """
    # TODO convert to pathlib

    if str(file_with_edit_link).startswith('http'):
        url = str(file_with_edit_link)

    # If the input wasn't a link itself, then it should be a path to a file containing
    # the link.
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
            url = f.readline()

    base_url_and_param_seperator = '/edit'
    if base_url_and_param_seperator in url:
        base_url, http_params = url.split(base_url_and_param_seperator)

        # TODO test in places called w/ str input (where presumably '/edit' (and
        # following) isn't in URL (tho it probably should be now)
        # TODO use this code in other path too
        http_param_parts = http_params.strip('#').strip().split('&')
        gid_param_prefix = 'gid='
        for p in http_param_parts:
            if p.startswith(gid_param_prefix):
                if gid is not None:
                    raise ValueError('gid specified in both file_with_edit_link and gid'
                        ' args'
                    )

                gid = int(p[len(gid_param_prefix):])
                # Assuming gid=<x> not specified more than once in URL
                break
    else:
        base_url = url

    # TODO get proper error in gsheet_to_frame if we default to gid=0 and that gid
    # somehow doesn't exist on sheet (possible? maybe if first sheet deleted?)
    # TODO warn in this case?
    if gid is None:
        # Seems to be default for first sheet
        gid = 0

    gsheet_link = base_url + '/export?format=csv&gid='

    if not no_append_gid:
        gsheet_link += str(gid)

    return gsheet_link


# TODO add option to strip whitespace + replace whitespace-only cells with NaN
# (then maybe use in natural_odors/literature_data)
def gsheet_to_frame(file_with_edit_link: Pathlike, *, gid: Optional[int] = None,
    bool_fillna_false: bool = True, convert_date_col: bool = True,
    drop_trailing_bools: bool = True, restore_ints: bool = True,
    normalize_col_names: bool = False, use_cache: bool = False) -> pd.DataFrame:
    # TODO doc file_with_edit_link / gid (w/ expected format + how to get them)
    # TODO want to allow str url for file_with_edit_link too (allowed in called fn)?
    """
    Args:
        file_with_edit_link: 

        gid: 

        bool_fillna_false: whether to replace missing values in columns that otherwise
            only contain True/False with False. will convert column dtype to 'bool' as
            well.

        convert_date_col: whether to convert the contents of any columns named 'date'
            (case insensitive) to `pd.Timestamp`

        drop_trailing_bools: whether to drop blocks of False in bool columns beyond the
            last row where all non-bool columns have any non-NaN values.

            If a column has data validation for a boolean, the frame will have values
            (False as I've seen it so far) through to the end of the validation range,
            despite the fact that no data has been entered.

        restore_ints: whether to convert columns parsed as floats (because missing data
            in rows where only default values for bool cols are present) to an integer
            type. Requires that drop_trailing_bools actually gets rid of all the NaN
            values in the columns to be converted to ints (float columns with only whole
            number / NaN values).

        normalize_col_names: whether to rename columns using the
            `hong2p.util.to_filename` (with `period=False` to that function) as well as
            lowercasing.

        use_cache: whether to try loading cached Google sheet data, if there is a
            connection error when trying to load the sheet data from online. Each call
            will unconditionally write to this cache, saved as a hidden file in the same
            directory as `file_with_edit_link`.
    """
    file_with_edit_link = Path(file_with_edit_link)

    gsheet_link = gsheet_csv_export_link(file_with_edit_link, gid=gid)

    try:
        df = pd.read_csv(gsheet_link)

    # This might not always be the error, depending on the pandas version.
    # Tested with 1.3.1
    except URLError:
        use_cache = True

    # TODO support file_with_edit_link being url str too (+ change type hint), or maybe
    # unsupport that in other gsheet fn
    cache_path = file_with_edit_link.parent / f'.{file_with_edit_link.stem}_cache.p'

    if use_cache and cache_path.exists():
        warnings.warn(f'using Google sheet cache at {cache_path}. may be out-of-date!')

        # TODO factor this + writing below into [load|write]_pickle(Path, ...) fns?
        return pickle.loads(cache_path.read_bytes())

    bool_col_unique_vals = {True, False, np.nan}
    # TODO may want to change issubset call to exclude cols where there is somehow only
    # NaN with dtype is still being 'object' (shouldn't be possible though, at least as
    # long as this is the first step?)
    bool_cols = [c for c in df.columns if df[c].dtype == 'bool' or
        (df[c].dtype == 'object' and set(df[c].unique()).issubset(bool_col_unique_vals))
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

        len_before = len(df)

        # TODO still works if last row actually does have data, right?
        df = df.iloc[:(last_row_with_data_idx + 1)].copy()

        assert len_before - len(will_be_dropped) == len(df)

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

    cache_path.write_bytes(pickle.dumps(df))

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
            # TODO replace w/ gsheet_to_frame?
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


# TODO work with NaN in inputs? i don't want NaN != NaN being reported as a difference.
# use my new nan_eq if needed/helpful.
# TODO possible to replace (some of?) this w/ pandas.DataFrame.compare?
def diff_dataframes(df1, df2) -> Optional[pd.DataFrame]:
    """Returns a DataFrame summarizing input differences, or None if no differences.
    """
    # TODO do i want df1 and df2 to be allowed to be series?
    # (is that what they are now? need to modify anything?)
    assert (df1.columns == df2.columns).all(), 'DataFrame column names are different'

    if any(df1.dtypes != df2.dtypes):
        print('Data Types are different, trying to convert')
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

        diff_mask = pd.concat([diff_mask_floats, diff_mask_arr, other_diff_mask],
            axis=1
        )
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
            index=changed.index
        )


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


# TODO test that date, fly_num, thorimage_id args still work here after refactoring to
# use wrapper
# TODO option to keep this under analysis_dir type tree instead? or just fully switch to
# that maybe?
@thorimage_dir_input
def metadata_filename(thorimage_dir):
    """Returns filename of YAML for extra metadata.
    """
    # TODO port over any metadata yamls i have in the raw data tree in this old location
    #return join(raw_fly_dir(date, fly_num), thorimage_id + '_metadata.yaml')
    return join(thorimage_dir(date, fly, thorimage_id), 'metadata.yaml')


# TODO maybe something to indicate various warnings
# (like mb team not being able to pair things) should be suppressed?
# TODO wrap this + read_movie into loading that can also flip according to a key in the
# yaml (L/R, for more easily comparing data from diff AL sides in flies in the same
# orientation, for example)
#def load_metadata(date, fly_num, thorimage_id):
def load_metadata(*args):
    """Returns metadata from YAML, with defaults added.
    """
    metadata_file = metadata_filename(*args)

    # TODO another var specifying number of frames that has *already* been
    # cropped out of raw tiff (start/end), to resolve any descrepencies wrt
    # thorsync data
    metadata = {
        'flip_lr': False,
        'drop_first_n_frames': 0,
    }
    if exists(metadata_file):
        # TODO also load single odors (or maybe other trial structures) from stuff like
        # this, so analysis does not need my own pickle based stim format
        with open(metadata_file, 'r') as mdf:
            yaml_metadata = yaml.load(mdf)

        for k in metadata.keys():
            if k in yaml_metadata:
                metadata[k] = yaml_metadata[k]

    return metadata


# TODO move to an io module?
def load_movie(thorimage_dir, **kwargs):
    """Loads movie and pre-processes (e.g. flipping) if metadata requests it.
    """
    movie = thor.read_movie(thorimage_dir, **kwargs)
    raise NotImplementedError

    # TODO TODO TODO finish implementing
    #metadata = load_metadata(


def dir2keys(path: Pathlike) -> Tuple[pd.Timestamp, int, str]:
    """Returns (date, fly_num, thorimage_id) for dir with these as last three parts.
    """
    path = Path(path)
    date_str, fly_num_str, thorimage_dirname = path.parts[-3:]

    date = pd.Timestamp(date_str)
    fly_num = int(fly_num_str)
    return date, fly_num, thorimage_dirname


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


# TODO delete?
def recording_df2keys(df):
    dupes = df[recording_cols].drop_duplicates()
    assert len(dupes) == 1
    return tuple(dupes.iloc[0])


# TODO delete?
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


# TODO test this works w/ both Path and str input
def write_tiff(tiff_filename: Pathlike, movie: np.ndarray, strict_dtype=True,
    dims: Optional[str] = None) -> None:
    # TODO also handle diff color channels
    """Write a TIFF loading the same as the TIFFs we create with ImageJ.

    TIFFs are written in big-endian byte order to be readable by `imread_big`
    from MATLAB file exchange.

    Dimensions of input should be (t,[z,],y,x).

    Metadata may not be correct.

    Args:
        dims: may or may not have the same meaning as `tifffile.imsave` `axes` kwarg
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

    else:
        # To avoid `ValueError: ImageJ does not support data type 'd'` from tifffile.py
        if movie.dtype == 'float64':
            movie = movie.astype('float32')

    # TODO TODO maybe change so ImageJ considers appropriate dimension the time
    # dimension (both in 2d x T and 3d x T cases)
    # TODO TODO TODO convert from thor data to appropriate dimension order (w/
    # singleton dimensions as necessary) (or keep dimensions + dimension order
    # of array, and pass metadata={'axes': 'TCXY'}, w/ the value constructed
    # appropriately? that work w/ imagej=True?) (i dont think it did)

    # TODO TODO TODO since scipy docs say [their version] of tifffile expects
    # channels in TZCYXS order
    # https://scikit-image.org/docs/0.14.x/api/skimage.external.tifffile.html

    imagej_dims = 'TZCYX'

    if dims is not None:
        if len(dims) != len(movie.shape):
            raise ValueError('wrong number of dimensions')

        dims = dims.upper()
        for c in dims:
            if c not in imagej_dims:
                raise ValueError(f'dimension {c} not among {imagej_dims}')
    else:
        if len(movie.shape) == 3:
            dims = 'TYX'
            # Z and C
            #new_dim_indices = (1, 2)
        elif len(movie.shape) == 4:
            dims = 'TZYX'
            # C
            #new_dim_indices = (2,)
        else:
            raise ValueError('unexpected number of dimensions to movie. have '
                f'{len(movie.shape)}. expected 3 (TYX) or 4 (TZYX).'
            )

    n_dims_to_add = len(imagej_dims) - len(movie.shape)
    movie = np.expand_dims(movie, axis=tuple(range(n_dims_to_add)))

    new_dims = ''.join([c for c in imagej_dims if c not in dims])
    dims = new_dims + dims
    assert set(dims) == set(imagej_dims)

    movie = np.transpose(movie, axes=[dims.index(c) for c in imagej_dims])

    # TODO TODO is "UserWarning: TiffWriter: truncating ImageJ file" actually
    # something to mind? for example, w/ 2020-04-01/2/fn as input, the .raw is
    # 8.3GB and the .tif is 5.5GB (w/ 3 flyback frames for each 6 non-flyback
    # frames -> 8.3 * (2/3) = ~5.53  (~ 5.5...). some docs say bigtiff is not
    # supported w/ imagej=True, so maybe that wouldn't be a quick fix if the
    # warning actually does matter. if not, maybe suppress it somehow?

    # TODO maybe just always do test from test_readraw here?
    # (or w/ flag to disable the check)

    # TODO TODO maybe just don't save w/ imagej=True? suite2p docs (or maybe it was
    # caiman docs?) seemed to suggest imagej tiffs might have some potentially-relevant
    # limitations... super specific i know
    tifffile.imsave(tiff_filename, movie, imagej=True)


def full_frame_avg_trace(movie):
    """Takes a (t,[z,]x,y) movie to t-length vector of frame averages.
    """
    # Averages all dims but first, which is assumed to be time.
    return np.mean(movie, axis=tuple(range(1, movie.ndim)))


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


# TODO call this in gui / factor into viz.plot_odor_corrs (though it would
# require accesss to df...) and call that there
# TODO delete?
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


def euclidean_dist(v1, v2):
    # Without the conversions to float 64 (or at least something else signed),
    # uint inputs lead to wraparound -> big distances occasionally.
    return np.linalg.norm(np.array(v1).astype(np.float64) -
        np.array(v2).astype(np.float64)
    )


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
    meta = load_metadata(*keys)

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
def to_filename(x: str, period: bool = True,
    extra_remove_chars: Optional[Sequence[str]] = None) -> str:
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
        '?': '',
    }
    if extra_remove_chars is not None:
        for c in extra_remove_chars:
            replace_dict[c] = ''

    for k, v in replace_dict.items():
        x = x.replace(k, v)

    # Replace multiple consecutive '_' with a single '_'
    x = re.sub('_+', '_', x)

    # TODO delete this and refactor code that expects this behavior to add the period
    if period:
        x += '.'

    return x


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


# TODO delete
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


# TODO kwarg to have this replace the multiindex levels / columns values it is derived
# from (and thread through add_fly_id/add_recording_id)
# TODO TODO axis kwarg?
def add_group_id(data: DataFrameOrDataArray, group_keys, name=None, dim=None,
    start_at_one=True, sort=True, inplace=False):
    """Adds integer column to identify unique combinations of group_keys.

    Args:
        data: DataFrame or DataArray to add metadata to
    """
    if name is None:
        name = '_'.join(group_keys) + '_id'

    if isinstance(data, pd.DataFrame):
        assert name not in data.columns
        df = data
        # Just to make sure we don't get SetWithCopyWarnings when assigning below.
        if not inplace:
            df = df.copy()
    else:
        if dim is None or dim not in data.dims:
            # TODO maybe don't raise this if there are valid cases where the
            # assign_coords call wouldn't need the tuple RHS (with this dim value)
            raise ValueError('must pass dim=<dimension from data.dims to add group ID '
                'to> for xarray input'
            )

        # TODO TODO TODO should i add handling for input where some coordinates in
        # group_keys are not associated with a dimension (as long as there are no
        # conflicts as to which dimension the coordinates in group_keys correspond
        # to...)? (if so, returned DataArray should probably move unassigned variables
        # to selected dimension)
        # or just require input has all group_keys assigned to (same) dimension?

        assert name not in data.coords
        # Using data[n].values didn't make a difference in one test (same result).
        # (and didn't fix TypeError when some, but not all, group_keys are scalars, but
        # led to a diff TypeError. may or may not have been important those vars were
        # also not among coordinates)
        #df = pd.DataFrame({n: data[n].values for n in group_keys})
        #import ipdb; ipdb.set_trace()
        df = pd.DataFrame({n: data[n] for n in group_keys})

    group_numbers = df.groupby(group_keys, sort=sort).ngroup()

    if start_at_one:
        group_numbers = group_numbers + 1

    if isinstance(data, pd.DataFrame):
        df[name] = group_numbers
        return df
    else:
        if inplace:
            # Because assign_coords docs says it returns a new object
            # (maybe it doesn't actually copy sometimes, or there are equivalent calls
            # that wouldn't need to?)
            raise NotImplementedError('inplace=True not supported for DataArray input')

        return data.assign_coords({name: (dim, group_numbers)})


# TODO replace hardcoded recording_cols[:2] w/ kwarg that defaults to None where None
# gets replaced by current hardcoded value
def add_fly_id(df, **kwargs):
    name = 'fly_id'
    # TODO TODO replace prep_date w/ date in recording cols.
    return add_group_id(df, recording_cols[:2], name=name, **kwargs)


def add_recording_id(df, **kwargs):
    name = 'recording_id'
    return add_group_id(df, recording_cols, name=name, **kwargs)


@thorimage_dir_input
def thor2tiff(image_dir: Pathlike, *, output_name=None, output_basename=None,
    output_dir=None, if_exists: str = 'err', flip_lr: Optional[bool] = None,
    discard_channel_b: bool = False, check_round_trip=False, verbose=True, _debug=False
    ) -> Optional[np.ndarray]:
    """Converts ThorImage .raw file to .tif file in same directory

    Args:
        if_exists: 'load', 'ignore', 'overwrite', or 'err'

        flip_lr: If True, flip the raw movie along the left/right axis, to make
            experiments including both left/right side data more comparable.
            If True/False, default output basename will be 'flipped.tif'. If None,
            default output basename will remain 'raw.tif'.

        check_round_trip: If True, and a TIFF was written, read it and check it is equal
            to data loaded from ThorImage raw.

    Returns an np.ndarray movie if TIFF was created OR if if_exists='load' and
    the corresponding TIFF already exists. Returns None if if_exists='ignore' and the
    corresponding TIFF already exists.
    """
    if_exists_options = ('load', 'ignore', 'err', 'overwrite')
    if if_exists not in if_exists_options:
        raise ValueError(f'if_exists must be one of {if_exists_options}')

    if all([x is not None for x in (output_name, output_basename)]):
        raise ValueError('only pass at most one of output_name or output_basename')

    image_dir = Path(image_dir)

    # TODO .tif or .tiff?
    tiff_ext = '.tif'

    if flip_lr is None:
        default_output_basename = f'raw{tiff_ext}'
    else:
        # Naming it 'flipped' in both the cases where we do/don't flip (as long as
        # flip_lr is specified True/False, so other analysis can know that we at least
        # made the decision as to whether to flip this data, whether or not we actually
        # flipped it)
        default_output_basename = f'flipped{tiff_ext}'

    if output_name is None and output_basename is None:
        output_basename = default_output_basename

    if output_name is None:
        if output_dir is None:
            output_dir = image_dir

        # TODO maybe just make it?
        assert isdir(output_dir), f'output_dir={output_dir} was not a directory'
        output_name = join(output_dir, output_basename)

    if exists(output_name):
        if if_exists == 'ignore':
            if _debug:
                print(f'TIFF {output_name} already exists. Doing nothing.')

            return None

        elif if_exists == 'load':
            if _debug:
                print(f'TIFF {output_name} exists. Reading it (instead of raw)...',
                    flush=True, end=''
                )

            movie = tifffile.imread(output_name)

            if _debug:
                print(' done', flush=True)

            # NOTE that if this will not returned any flipped version that might exist
            # UNLESS 1) it already exists, AND 2) flip_lr=False/True (not None)
            # TODO return as xarray? w/ flag to disable?
            return movie

        elif if_exists == 'err':
            raise IOError(f'{output_name} exists (set if_exists to either '
                "'overwrite' or 'ignore' for other behavior"
            )

        elif if_exists == 'overwrite':
            if _debug:
                print(f'TIFF {output_name} existed. Overwriting.')

    # TODO maybe also load metadata like fps (especially stuff, as w/ fps, that isn't
    # already baked into the TIFF, assuming the TIFF is saved correctly. so not
    # including stuff like z, c, xy), and print w/ -v flag?

    if verbose:
        print('Reading raw movie...', flush=True, end='')

    movie = thor.read_movie(image_dir, discard_channel_b=discard_channel_b)

    if verbose:
        print(' done', flush=True)

    if flip_lr:
        if verbose:
            print('Flipping movie along left/right axis, as requested')

        # axis=-1 should be the X axis (in a ([z,], y, x) shape movie), and does
        # visually flip left/right when plotting frames.
        movie = np.flip(movie, axis=-1)

    # TODO TODO try to figure out if anything can be done about tifffile using so much
    # memory on writing (says "Killed" and exits in the middle of writing when trying to
    # write what should be a ~5.5GB movie when i have close to 20GB of RAM free...).
    # maybe memory profile my own code to see if i'm doing something stupid. related to
    # imagej=True kwarg?
    # TODO test read_movie on all thorimage .raw outputs i have to check which can
    # currently reproduce this issue

    if verbose:
        print(f'Writing TIFF to {output_name}...', flush=True, end='')

    write_tiff(output_name, movie)

    if verbose:
        print(' done', flush=True)

    if check_round_trip:
        # Leaving this on verbose rather than _debug, because it could take some time
        # and we don't want it to become a core pare of a pipeline w/o being aware of
        # it.
        if verbose:
            print('Reading written TIFF for round trip check...', flush=True, end='')

        round_tripped = tifffile.imread(output_name)
        assert np.array_equal(movie, round_tripped)
        if verbose:
            print(' passed', flush=True)

    # TODO return as xarray? w/ flag to disable? maybe build a decorator to
    # automatically handle that conversion + add kwarg to toggle (how to get metadata
    # though...)?
    return movie


# TODO rename from melt (if that's not the closest-functionality pandas fn)?
# TODO still want all these kwargs?
# TODO get to work w/ xarray input too
# TODO TODO and also support case where input DataArray has one dimension along
# which a series of symmetric matrices have been concatenated, and handle that too
# (e.g. shape (5, 27, 27) series of 5 correlation matrices, one for each of 5 flies)
# TODO test that it also works w/o MultiIndex indices
def melt_symmetric(symmetric_df, drop_constant_levels=True,
    suffixes=('_a', '_b'), name=None, keep_duplicate_values=False):
    """Takes a symmetric DataFrame to a tidy Series with unique values.

    Symmetric means the row and columns indices are equal, and values should
    be a symmetric matrix.
    """
    # This is actually not dependent on the level names / .name (if we wanted to check
    # those + the types as well, we'd use .identical(...)), so it's ok if the
    # columns/index are already e.g. called 'odor' & 'odor_b', respectively.
    assert symmetric_df.columns.equals(symmetric_df.index)

    # TODO put behind a flag at the minimum?
    symmetric_df = symmetric_df.copy()
    symmetric_df.dropna(how='all', axis=0, inplace=True)
    symmetric_df.dropna(how='all', axis=1, inplace=True)
    assert symmetric_df.notnull().all(axis=None), 'not tested w/ non-all NaN'

    multiindices = False
    # Assuming index/columns are the same type (either MultiIndex or not)
    if isinstance(symmetric_df.index, pd.MultiIndex):
        multiindices = True

    # If identical(...) fails, that should mean the .names/.name attributes differed
    # (formally the dtypes could have differed too), as the .equals(...) assertion above
    # passed.
    if not symmetric_df.columns.identical(symmetric_df.index):
        # TODO factor out?
        def get_index_names(index):
            return index.names if multiindices else [index.name]

        col_names =  get_index_names(symmetric_df.columns)
        index_names = get_index_names(symmetric_df.index)

        # TODO use ValueError rather than assertions
        assert all(
            x.endswith(suffixes[1]) for x in col_names
        )
        assert all(
            x.endswith(suffixes[0]) for x in index_names
        )
    else:
        # TODO make all of this index name/level names renaming conditional on kwarg?
        if multiindices:
            # TODO adapt to work in non-multiindex case too! (rename there?)
            symmetric_df.index.rename([n + suffixes[0] for n in
                symmetric_df.index.names], inplace=True
            )
            symmetric_df.columns.rename([n + suffixes[1] for n in
                symmetric_df.columns.names], inplace=True
            )
        else:
            # TODO TODO maybe i still want to do this in the multiindex case tho?
            # would it break my old code that used this in kc_natural_mixes?
            old_name = symmetric_df.index.name
            assert old_name == symmetric_df.columns.name
            symmetric_df.index.name = f'{old_name}{suffixes[0]}'
            symmetric_df.columns.name = f'{old_name}{suffixes[1]}'

    # TODO maybe always test that triu equals tril tho (or w/ a _checks=True set)

    # To de-clutter what would otherwise become a highly-nested index.
    if multiindices and drop_constant_levels:
        # TODO may need to call index.remove_unused_levels() first, if using
        # levels here... (see docs of that remove fn)
        constant_levels = [i for i, levels in enumerate(symmetric_df.index.levels)
            if len(levels) == 1
        ]
        symmetric_df = symmetric_df.droplevel(constant_levels, axis='index')
        symmetric_df = symmetric_df.droplevel(constant_levels, axis='columns')

    # TODO maybe an option to interleave the new index names
    # (so it's like name1_a, name1_b, ... rather than *_a, *_b)
    # or would that not ever really be useful?

    if keep_duplicate_values:
        tidy = symmetric_df.stack(level=symmetric_df.columns.names)
        assert tidy.shape == (np.prod(symmetric_df.shape),)
    else:
        # From: https://stackoverflow.com/questions/34417685
        keep = np.triu(np.ones(symmetric_df.shape)).astype('bool')
        masked = symmetric_df.where(keep)
        n_nonnull = masked.notnull().sum().sum()
        # We already know both elements of shape are the same from equality
        # check on indices above.
        n = symmetric_df.shape[0]
        # The right expression is the number of elements expected for the
        # triangular of a square matrix w/ side length n, if the diagonal
        # is INCLUDED.
        assert n_nonnull == (n * (n + 1) / 2)

        # TODO make sure this also still works in non-multiindex case!
        tidy = masked.stack(level=masked.columns.names)
        assert tidy.shape == (n_nonnull,)

    tidy.name = name
    return tidy


# TODO rename if it could make it more accurate
def invert_melt_symmetric(ser, suffixes=('_a', '_b')):
    """
    """
    assert len(ser.shape) == 1, 'not a series'
    assert len(ser.index.names) == len(set(ser.index.names)), \
        'index names should be unique'

    assert len(suffixes) == 2 and len(set(suffixes)) == 2
    s0, s1 = suffixes

    levels_to_drop = set(ser.index.names)
    col_prefixes = []
    for c in ser.index.names:
        if type(c) is not str:
            continue

        if c.endswith(s0):
            prefix = c[:-len(s0)]
            if (prefix + s1) in ser.index.names:
                col_prefixes.append(prefix)
                levels_to_drop.remove(prefix + s0)
                levels_to_drop.remove(prefix + s1)

    levels_to_drop = list(levels_to_drop)
    # This does also work in the case where `levels_to_drop` is empty.
    ser = ser.droplevel(levels_to_drop)
    return ser.unstack([p + s0 for p in col_prefixes])


# TODO move this (+ other appropriate stuff), to a pandas.py module like my xarray.py
def check_index_vals_unique(df: pd.DataFrame) -> None:
    """Raises AssertionError if any duplicates in column/index indices.
    """
    assert not (df.index.duplicated().any() or df.columns.duplicated().any())


# TODO also factor to a pandas module
# TODO already have something like this?
# TODO typehint subclass of Index? already good on that?
def get_axis_index(df: pd.DataFrame, axis: Union[int, str]) -> pd.Index:
    if axis == 0 or axis == 'index':
        return df.index
    elif axis == 1 or axis == 'columns':
        return df.columns
    else:
        raise ValueError(f'invalid axis: {axis}')


# TODO also factor to a pandas module
def suffix_index_names(df: pd.DataFrame, suffix: str = '_b', axis='columns') -> None:
    # TODO TODO also support non-multiindex w/ .name
    # TODO factor out helper to get df
    index = get_axis_index(df, axis)
    index.names = [f'{x}{suffix}' for x in index.names]
    # TODO make not inplace?

