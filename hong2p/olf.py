"""
Functions for loading YAML metadata created by my tom-f-oconnell/olfactometer repo, and
dealing with the resulting representations of odors delivered during an experiment.

Keeping these functions here rather than in the olfactometer repo because it has other
somewhat heavy dependencies that the analysis side of things will generally not need.
"""

from collections import Counter
import warnings
from typing import Union, Sequence, Optional

import numpy as np
import pandas as pd


solvent_str = 'solvent'

def odordict_sort_key(odor_dict):
    name = odor_dict['name']

    # If present, we expect this value to be a non-positive number.
    # Using 0 as default for lack of 'log10_conc' key because that case should indicate
    # some type of pure odor (or something where the concentration is specified in the
    # name / unknown). '5% cleaning ammonia in water' for example, where original
    # concentration of cleaning ammonia is unknown.
    log10_conc = odor_dict.get('log10_conc', 0)

    # 'log10_conc: null' in one of the YAMLs should map to None here.
    if log10_conc is None:
        log10_conc = float('-inf')

    assert type(log10_conc) in (int, float), f'type({log10_conc}) == {type(log10_conc)}'

    return (name, log10_conc)


def sort_odor_list(odor_list):
    """Returns a sorted list of dicts representing odors for one trial

    Name takes priority over concentration, so with the same set of odor names in each
    trial's odor_list, this should produce a consistent ordering (and same indexes can
    be used assuming equal length of all)
    """
    return sorted(odor_list, key=odordict_sort_key)


# TODO share (parts?) w/ similar key fn now in hong2p.olf?
# TODO document i/o types
def odor_index_sort_key(level, name_order=None):
    # The assignment below failed for some int dtype levels, even though the boolean
    # mask dictating where assignment should happen must have been all False...
    if level.dtype != np.dtype('O'):
        return level

    sort_key = level.values.copy()

    solvent_elements = sort_key == solvent_str
    conc_delimiter = '@'
    assert all([conc_delimiter in x for x in sort_key[~ solvent_elements]])

    # TODO share this + fn wrapping this that returns sort key w/ other place that
    # is using sorted(..., key=...)
    def parse_log10_conc(odor_str):
        assert conc_delimiter in odor_str
        parts = odor_str.split(conc_delimiter)
        assert len(parts) == 2
        return float(parts[1].strip())

    def parse_odor_name(odor_str):
        assert conc_delimiter in odor_str
        parts = odor_str.split(conc_delimiter)
        assert len(parts) == 2
        # TODO want to handle 'pfo'/'solvent' special?
        return parts[0].strip()

    if not all(solvent_elements):
        conc_keys = [parse_log10_conc(x) for x in sort_key[~ solvent_elements]]
        sort_key[~ solvent_elements] = conc_keys

        # Setting solvent to an unused log10 concentration just under lowest we have,
        # such that it gets sorted into overall order as I want (first, followed by
        # lowest concentration).
        # TODO TODO maybe just use float('-inf') or numpy equivalent here?
        # should give me the sorting i want.
        sort_key[solvent_elements] = min(conc_keys) - 1

        if name_order is not None:
            if solvent_elements.sum() > 0:
                raise NotImplementedError("not currently supporting 'solvent' when "
                    'using name_order keyword argument'
                )

            # TODO TODO TODO do i need to handle solvent_elements here? how?
            # possible, or will i just need to do outside of this within-level-only
            # function?
            names = [parse_odor_name(x) for x in level]

            if not all(n in name_order for n in names):
                # TODO only print the ones missing
                raise ValueError(
                    f'some of names={names} were not in name_order={name_order}'
                )

            name_orders = [name_order.index(x) for x in names]

            sort_key = list(zip(name_orders, sort_key))

            # tupleize_cols=False prevents a MultiIndex from being created
            index = pd.Index(sort_key, tupleize_cols=False, name=level.name)
            return index

    # Converting back to an index so that `level=<previous level name>` arg to
    # `DataFrame.sort_index` doesn't get broken. This key function is used to generate
    # an intermediate Index pandas uses to sort, and that intermediate needs to have the
    # same level names to be able to refer to them as if it was the input object.
    return pd.Index(sort_key, name=level.name)


# TODO TODO add some kind of lookup for odor panels (might just need to get the set of
# all (odor name, odor concentrations) used in experiment and compare that.  -> force
# consistent order for things like kiwi.yaml/control1.yaml experiments (anything not
# pair that we actually wanna see plots for actually. probably just don't wanna sort
# glomeruli diagnostics) (only really relevant if i actually start randomizing order in
# those experiments... for now, could just not sort)
# TODO TODO TODO allow passing in an odor order for the odor names, used in conjunction
# with the sorting on the concentrations
odor_cols = ['odor1', 'odor2', 'odor1_b', 'odor2_b']
def sort_odor_indices(df: pd.DataFrame, name_order=None) -> pd.DataFrame:

    def levels_to_sort(index):
        return [k for k in index.names if k in odor_cols]

    levels = levels_to_sort(df.index)
    for axis_name in ('index', 'columns'):
        levels = levels_to_sort(getattr(df, axis_name))
        if len(levels) > 0:
            # TODO check my level sort key fn works in both case of 1 level passed in as
            # well as 2
            df = df.sort_index(
                key=lambda x: odor_index_sort_key(x, name_order=name_order),
                axis=axis_name, level=levels,
                sort_remaining=False
            )

    return df


def yaml_data2pin_lists(yaml_data):
    """
    Pins used as balances can be part of these lists despite not having a corresponding
    odor in 'pins2odors'.
    """
    return [x['pins'] for x in yaml_data['pin_sequence']['pin_groups']]


def yaml_data2odor_lists(yaml_data, sort=True):
    """Returns a list-of-lists of dictionary representation of odors.

    Each dictionary will have at least the key 'name' and generally also 'log10_conc'.

    The i-th list contains all of the odors presented simultaneously on the i-th odor
    presentation.

    Args:
        yaml_data (dict): parsed contents of stimulus YAML file
        sort (bool): (default=True) whether to, within each trial, sort odors
    """
    pin_lists = yaml_data2pin_lists(yaml_data)
    # int pin -> dict representing odor (keys 'name', 'log10_conc', etc)
    pins2odors = yaml_data['pins2odors']

    odor_lists = []
    for pin_list in pin_lists:

        odor_list = []
        for p in pin_list:
            if p in pins2odors:
                odor_list.append(pins2odors[p])

        if sort:
            odor_list = sort_odor_list(odor_list)

        odor_lists.append(odor_list)

    return odor_lists


def remove_consecutive_repeats(odor_lists):
    """Returns a list-of-str without any consecutive repeats and int # of repeats.

    Wanted to also take a list-of-lists-of-dicts, where each dict represents one odor
    and each internal list represents all of the odors on one trial, but the internal
    lists (nor the dicts they contain) would not be hashable, and thus cannot work with
    Counter as-is.

    Assumed that all elements of `odor_lists` are repeated the same number of times, and
    all repeats are consecutive. Actually now as long as any repeats are to full # and
    consecutive, it is ok for a particular odor (e.g. solvent control) to be repeated
    `n_repeats` times in each of several different positions.
    """
    # In Python 3.7+, order should be guaranteed to be equal to order first encountered
    # in odor_lists.
    # TODO modify to also allow counting non-hashable stuff (i.e.  dictionaries), so i
    # can pass my (list of) lists-of-dicts representation directly
    counts = Counter(odor_lists)

    count_values = set(counts.values())
    n_repeats = min(count_values)
    without_consecutive_repeats = odor_lists[::n_repeats]

    # TODO possible to combine these two lines to one?
    # https://stackoverflow.com/questions/25674169
    nested = [[x] * n_repeats for x in without_consecutive_repeats]
    flat = [x for xs in nested for x in xs]
    assert flat == odor_lists, 'variable number or non-consecutive repeats'

    # TODO add something like (<n>) to subsequent n_repeats occurence of the same odor
    # (e.g. solvent control) (OK without as long as we are prefixing filenames with
    # presentation index, but not-OK if we ever wanted to stop that)

    return without_consecutive_repeats, n_repeats


def format_odor(odor_dict, conc=True, name_conc_delim=' @ ', conc_key='log10_conc'):
    """Takes a dict representation of an odor to a pretty str.

    Expected to have at least 'name' key, but will also use 'log10_conc' (or `conc_key`)
    if available, unless `conc=False`.
    """
    ostr = odor_dict['name']

    if conc_key in odor_dict:
        # TODO opt to choose between 'solvent' and no string (w/ no delim below used?)?
        # what do i do in hong2p.util fn now?
        if odor_dict[conc_key] is None:
            return solvent_str

        if conc:
            ostr += f'{name_conc_delim}{odor_dict[conc_key]}'

    return ostr


# TODO TODO decorator or some other place to store minimum set of keys (and types?) for
# these formatting functions (at least those that take a Series as one option)?
# (so that hong2p.viz.callable_ticklabels can automatically convert / make good error
# messages if they are missing)
def format_mix_from_strs(odor_strs: Union[Sequence[str], pd.Series],
    delim: Optional[str] = None):

    if isinstance(odor_strs, pd.Series):
        odor_keys = [x for x in odor_strs.keys() if x.startswith('odor')]

        if len(odor_keys) < len(odor_strs):
            nonodor_keys = [x for x in odor_strs.keys() if x not in odor_keys]
            warnings.warn('format_mix_from_strs: ignoring levels not starting with '
                f"'odor' ({nonodor_keys})"
            )

        odor_strs = [odor_strs[k] for k in odor_keys]

    if delim is None:
        delim = ' + '

    odor_strs = [x for x in odor_strs if x != solvent_str]
    if len(odor_strs) > 0:
        return delim.join(odor_strs)
    else:
        return solvent_str


def format_odor_list(odor_list, delim=None, **kwargs):
    """Takes list of dicts representing odors for one trial to pretty str.
    """
    odor_strs = [format_odor(x, **kwargs) for x in odor_list]
    return format_mix_from_strs(odor_strs, delim=delim)

