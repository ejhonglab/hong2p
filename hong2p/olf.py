"""
Functions for loading YAML metadata created by my tom-f-oconnell/olfactometer repo, and
dealing with the resulting representations of odors delivered during an experiment.

Keeping these functions here rather than in the olfactometer repo because it has other
somewhat heavy dependencies that the analysis side of things will generally not need.
"""

from collections import Counter
import warnings
from typing import Union, Sequence, Optional, Tuple, List, Dict, Hashable

import numpy as np
import pandas as pd


solvent_str = 'solvent'
conc_delimiter = '@'

def parse_log10_conc(odor_str: str) -> Optional[float]:
    """Takes formatted odor string to float log10 vol/vol concentration.

    >>> parse_log10_conc('ethyl acetate @ -2')
    -2.0
    """
    # If conc_delimiter is in the string, we are assuming that it should be followed by
    # parseable float concentration. Letting it err below if that is not the case.
    if conc_delimiter not in odor_str:
        return None

    parts = odor_str.split(conc_delimiter)
    assert len(parts) == 2
    return float(parts[1].strip())


def parse_odor_name(odor_str: str) -> str:
    """Takes formatted odor string to just the name of the odor.

    >>> parse_log10_conc('ethyl acetate @ -2')
    'ethyl acetate'
    """
    assert conc_delimiter in odor_str
    parts = odor_str.split(conc_delimiter)
    assert len(parts) == 2
    # TODO want to handle 'pfo'/'solvent' special?
    return parts[0].strip()


# TODO maybe take a name_order kwarg here and return index of name in list instead, if
# passed? then maybe unify w/ other place that currently does something similar?
# TODO TODO may want to sort by concentration THEN name, to be consistent w/
# odor_index_sortkey (or maybe change that one?)
def odordict_sort_key(odor_dict: dict) -> Tuple[str, float]:
    """Returns a hashable key for sorting odors by name, then concentration.
    """
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


def odor_index_sort_key(level: pd.Index, sort_names=True, names_first=True,
    name_order: Optional[List[str]] = None) -> pd.Index:
    """
    Args:
        level: a pd.Index containing one level of a MultiIndex with odor metadata.
            elements should be odor strings.

        sort_names: whether to use odor names as part of sort key. If False, only sorts
            on concentrations.

        names_first: if True, sorts on names primarily, otherwise sorts on
            concentrations primarily. Ignored if sort_names is False.

        name_order: list of odor names to use as a fixed order for the names.
            Concentrations will be sorted within each name.
    """
    if name_order is not None:
        assert sort_names == True, 'sort_names should be True if name_order passed'

    # The assignment below failed for some int dtype levels, even though the boolean
    # mask dictating where assignment should happen must have been all False...
    if level.dtype != np.dtype('O'):
        return level

    # prob don't need the .values most/all places anymore, but would need to test
    odor_strs = level.values

    # Will be overwritten with floats (either log10 concentration, or another float to
    # appropriately order solvent elements).
    conc_keys = np.empty(len(odor_strs)) * np.nan

    solvent_elements = odor_strs == solvent_str
    assert all([conc_delimiter in x for x in odor_strs[~ solvent_elements]])

    if not all(solvent_elements):
        nonsolvent_conc_keys = [
            parse_log10_conc(x) for x in odor_strs[~ solvent_elements]
        ]
        conc_keys[~ solvent_elements] = nonsolvent_conc_keys
        conc_keys[solvent_elements] = float('-inf')
        assert not pd.isnull(conc_keys).any()

        if sort_names:
            min_name_key = float('-inf')
            name_keys = [
                parse_odor_name(x) if x != solvent_str else min_name_key for x in level
            ]
            names = [n for n in name_keys if n != min_name_key]

            if name_order is None:
                # Making alphabetical. Need to still use the .index call to key name
                # keys because the -inf is not comparable to str.
                # NOTE: i didn't catch this error until a test w/ only 'odor' index, not
                # the earlier tests w/ usual multiindex. not sure why... bug?
                name_order = sorted(names)
            else:
                if not all(n in name_order for n in names):
                    # TODO only print the ones missing
                    raise ValueError(
                        f'some of names={names} were not in name_order={name_order}'
                    )

            name_keys = [
                name_order.index(x) if x != min_name_key else x for x in name_keys
            ]

            if names_first:
                sort_keys = (name_keys, conc_keys)
            else:
                sort_keys = (conc_keys, name_keys)

            sort_key = list(zip(*sort_keys))

            # tupleize_cols=False prevents a MultiIndex from being created
            index = pd.Index(sort_key, tupleize_cols=False, name=level.name)
            return index

    # Converting back to an index so that `level=<previous level name>` arg to
    # `DataFrame.sort_index` doesn't get broken. This key function is used to generate
    # an intermediate Index pandas uses to sort, and that intermediate needs to have the
    # same level names to be able to refer to them as if it was the input object.
    return pd.Index(conc_keys, name=level.name)


def is_odor_var(var_name: Optional[str]) -> bool:
    """Returns True if column/level name or Series-key is named to store odor metadata

    Variables behind matching names should store strings representing *one*, of
    potentially multiple, component odors presented on a given trial. My convention for
    representing multiple components presented together one one trial is to make
    multiple variables (e.g. columns), named such as ['odor1', 'odor2', ...], with a
    different sufffix number for each component.
    """
    # For index [level] names that are not defined.
    if var_name is None:
        return False

    return var_name.startswith('odor')


# TODO add some kind of lookup for odor panels (might just need to get the set of all
# (odor name, odor concentrations) used in experiment and compare that.  -> force
# consistent order for things like kiwi.yaml/control1.yaml experiments (anything not
# pair that we actually wanna see plots for actually. probably just don't wanna sort
# glomeruli diagnostics) (only really relevant if i actually start randomizing order in
# those experiments... for now, could just not sort)
# TODO TODO some way to make it easy to support is_pair (or other things we wanna sort
# on before the odors?) (add kwarg for columns (+ key fns, via a dict mapping
# cols->key fns) to sort before / after odors (as well as orders via another dict?)?)
# (or just add columns to sort odors w/in groups of? but prob wouldn't wanna use
# groupby, rather find existing consecutive groups and sort within...)
def sort_odors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # TODO add doctest examples clarifying how the two columns interact + what happens
    # to 'solvent' (+ clarify in docstring)
    # TODO doctest examples w/ and w/o name_order
    """Sorts DataFrame by odor index/columns.

    Args:
        df: DataFrame with columns/index-level names matching `is_odor_var`
        **kwargs: passed through to odor_index_sort_key

    Notes:
    Index will be checked first, and if it contains odor information, will sort on that.
    Otherwise, will check and sort on matching columns.

    Sorts by concentration, then name. 'solvent' is treated as less than all odors.

    >>> df = pd.DataFrame({
    ...     'odor1': ['B @ -2', 'A @ -2', 'A @ -3'],
    ...     'odor2': ['solvent'] * 3,
    ...     'delta_f': [1.1, 1.2, 0.9]
    ... }).set_index(['odor1', 'odor2'])

    >>> sort_odors(df)
                    delta_f
    odor1  odor2
    A @ -3 solvent      0.9
    B @ -2 solvent      1.1
    A @ -2 solvent      1.2

    >>> sort_odors(df, name_order=['B','A'])
                    delta_f
    odor1  odor2
    B @ -2 solvent      1.1
    A @ -3 solvent      0.9
    A @ -2 solvent      1.2
    """

    def levels_to_sort(index):
        return [k for k in index.names if is_odor_var(k)]

    found_odor_multiindex = False
    for axis_name in ('index', 'columns'):

        # Sorting so that if something were to accidentally re-order e.g. 'odor1',
        # 'odor2' levels, the sort order would be invariant to that, with 'odor1' always
        # taking precedence in the sort.
        levels = sorted(levels_to_sort(getattr(df, axis_name)))

        if len(levels) > 0:
            # TODO check my level sort key fn works in both case of 1 level passed in as
            # well as 2
            df = df.sort_index(
                key=lambda x: odor_index_sort_key(x, **kwargs),
                axis=axis_name,
                level=levels,
                sort_remaining=False,
                # So that the sort is "stable", meaning if stuff compares equal, it
                # preserves input order.
                kind='mergesort',
            )
            found_odor_multiindex = True

    if not found_odor_multiindex:
        odor_cols = [c for c in df.columns if is_odor_var(c)]

        if len(odor_cols) == 0:
            raise ValueError('df had no index levels or columns with hong2p.olf.'
                'is_odor_var(name) == True'
            )

        if isinstance(df.index, pd.MultiIndex):
            raise NotImplementedError('sorting odor columns not supported when '
                'there is an existing MultiIndex. call df.set_index(...) to include the'
                ' odor columns, then pass that as input.'
            )

        # TODO also try to keep order of columns same
        # (current approach moves odor columns to the start)
        temp_index_col = '_old_index'
        assert temp_index_col not in df.columns
        # Would have used reset_index(), but didn't see an argument to change the name
        # of the column it creates.
        df = df.copy()
        df[temp_index_col] = df.index
        old_index_name = df.index.name

        df = sort_odors(df.set_index(odor_cols), **kwargs).reset_index()

        df = df.set_index(temp_index_col)
        df.index.name = old_index_name
        return df

    return df


# TODO maybe move to viz.py, since this is mainly intended as as helper for
# viz.with_panel_orders plotting function wrapper?
def panel_odor_orders(df: pd.DataFrame,
    panel2name_order: Optional[Dict[str, List[str]]] = None, **kwargs):
    # TODO doctest example
    """Returns dict of panel names to ordered unique odor strs.

    Args:
        df: DataFrame with columns 'panel' and >=1 matching `is_odor_var`

        panel2name_order: dict mapping panels to lists of odor names, each in the
            desired order

        **kwargs: passed through to sort_odors
    """
    # TODO test w/ input that has odor info in multiindex (does groupby fuck it up?)
    assert 'name_order' not in kwargs
    name_order = None

    odor_cols = sorted([c for c in df.columns if is_odor_var(c)])
    if len(odor_cols) == 0:
        raise ValueError('must have >=1 columns matching hong2p.olf.is_odor_var(name)')

    panel2order = dict()
    for panel, panel_df in df.groupby('panel'):
        panel_df = panel_df.drop_duplicates(subset=odor_cols)

        if panel2name_order is not None:
            name_order = panel2name_order[panel]

        panel_df = sort_odors(panel_df, name_order=name_order, **kwargs)

        # TODO maybe factor out this (and preceding finding + sorting odor_cols)?
        mix_strs = [
            format_mix_from_strs(ser) for _, ser in panel_df[odor_cols].iterrows()
        ]
        panel2order[panel] = mix_strs

    return panel2order


def yaml_data2pin_lists(yaml_data: dict):
    """
    Pins used as balances can be part of these lists despite not having a corresponding
    odor in 'pins2odors'.
    """
    return [x['pins'] for x in yaml_data['pin_sequence']['pin_groups']]


def yaml_data2odor_lists(yaml_data: dict, sort=True):
    # TODO doctest example showing within-trial sorting
    """Returns a list-of-lists of dictionary representation of odors.

    Each dictionary will have at least the key 'name' and generally also 'log10_conc'.

    The i-th list contains all of the odors presented simultaneously on the i-th odor
    presentation.

    Args:
        yaml_data (dict): parsed contents of stimulus YAML file

        sort (bool): (default=True) whether to, within each trial, sort odors.
            Irrelevant if there are is only ever a single odor presented on each trial.
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


# TODO may want to move to util
# TODO make a union type that also accepts pd.Series and np.ndarray in addition to
# Sequence? just trying to require that it can be sliced w/ arbitrary stride.
# TODO how to indicate that the Hashable in the argument and return type should be of
# the same type? does Hashable take any type of arguments? make my own ~mixin of
# Hashable + Generic(?)
def remove_consecutive_repeats(odor_lists: Sequence[Hashable]
    ) -> Tuple[List[Hashable], int]:
    """Returns a list without any consecutive repeats and int # of consecutive repeats.

    Raises ValueError if there is a variable number of consecutive repeats.

    Wanted to also take a list-of-lists-of-dicts, where each dict represents one odor
    and each internal list represents all of the odors on one trial, but the internal
    lists (nor the dicts they contain) would not be hashable, and thus cannot work with
    Counter as-is.

    Assumed that all elements of `odor_lists` are repeated the same number of times,
    for each consecutive group of repeats. As long as any repeats are to full
    `n_repeats` and consecutive, it is ok for a particular odor (e.g. solvent control)
    to be repeated `n_repeats` times in each of several different positions.

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','a','b','b','b'])
    >>> without_repeats
    ['a','b']
    >>> n
    3

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','b','b','a','a'])
    >>> without_repeats
    ['a','b','a']
    >>> n
    2

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','a','b','b'])
    Traceback (most recent call last):
     ...
    ValueError: variable number of consecutive repeats
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
    if flat != odor_lists:
        raise ValueError('variable number of consecutive repeats')

    # TODO add something like (<n>) to subsequent n_repeats occurence of the same odor
    # (e.g. solvent control) (OK without as long as we are prefixing filenames with
    # presentation index, but not-OK if we ever wanted to stop that)

    return without_consecutive_repeats, n_repeats


def format_odor(odor_dict, conc=True, name_conc_delim=None, conc_key='log10_conc'):
    """Takes a dict representation of an odor to a pretty str.

    Expected to have at least 'name' key, but will also use 'log10_conc' (or `conc_key`)
    if available, unless `conc=False`.

    >>> odor = {'name': 'ethyl acetate', 'log10_conc': -2}
    >>> format_odor(odor)
    'ethyl acetate @ -2'
    """
    if name_conc_delim is None:
        name_conc_delim = f' {conc_delimiter} '

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
# TODO factor out this union type (+ probably add np.ndarray), and use in
# remove_consecutive_repeats as well (or maybe in this particular fn, i actually want
# Iterable[str] in the Union? not striding here..
def format_mix_from_strs(odor_strs: Union[Sequence[str], pd.Series],
    delim: Optional[str] = None):

    if isinstance(odor_strs, pd.Series):
        odor_keys = [x for x in odor_strs.keys() if is_odor_var(x)]

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


def format_odor_list(odor_list, delim: Optional[str] = None, **kwargs):
    """Takes list of dicts representing odors for one trial to pretty str.
    """
    odor_strs = [format_odor(x, **kwargs) for x in odor_list]
    return format_mix_from_strs(odor_strs, delim=delim)

