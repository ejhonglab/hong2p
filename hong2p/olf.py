"""
Functions for loading YAML metadata created by my tom-f-oconnell/olfactometer repo, and
dealing with the resulting representations of odors delivered during an experiment.

Keeping these functions here rather than in the olfactometer repo because it has other
somewhat heavy dependencies that the analysis side of things will generally not need.
"""

import atexit
from collections import Counter, defaultdict
from pprint import pprint, pformat
import pickle
import warnings
from typing import Union, Sequence, Optional, Tuple, List, Dict, Hashable, Any

import numpy as np
import pandas as pd
import yaml
from platformdirs import user_data_path

from hong2p import util
from hong2p.types import Pathlike, ExperimentOdors, SingleTrialOdors


solvent_str = 'solvent'
conc_delimiter = '@'
# NOTE: do need the whitespace here, if i want abbrev to be able to handle abritrary
# input strs (whether single odors / mixes / whatever) (w/o having to specify which type
# of input it is, at least), because of odors like '(1S)-(+)-carene'
# (and also to deal with in-vial mixtures like 'ea+eb @ 0', which I'm going to stick to
# formatting without the whitespace, so this can work)
component_delim = ' + '

# should just be for when no valve is actuated on the corresponding manifold
# TODO delete? not used anywhere... or replace (some?) usage of solvent_str w/ this
# (actually, it is used in viz,util[, and deprecated db + scripts/gui], but is it
# actually needed? at least clarify more here how it's use should actually differ from
# solvent_str) (and why is this not the fill value for the pad_odor_* fns below?)
# TODO TODO probably replace pad_odor_* fn use of solvent_str w/ this
NO_ODOR = 'no_odor'

# TODO add something for mapping from the standard-hong-odor-names to the hallem names

odor2abbrev = dict()

# TODO get package + author name from setup.py / package metadata?
# TODO log we are using this directory (at least at debug level)
#
# ensure_exists=True will make it if needed
_cache_dir = user_data_path('hong2p', 'tom-f-oconnell', ensure_exists=True)
odor2abbrev_cache = _cache_dir / 'odor2abbrev_cache.p'
if odor2abbrev_cache.exists():
    _odor2abbrev_from_cache = pickle.loads(odor2abbrev_cache.read_bytes())
    odor2abbrev.update(_odor2abbrev_from_cache)

# TODO log (info/debug?) when we override something from cache?
# TODO fns for adding to this / overriding
# TODO load/supplement from (union of?) abbrevs included in configs, if possible
_hardcoded_odor2abbrev = {
    'hexyl hexanoate': 'hh',
    'furfural': 'fur',

    '1-octen-3-ol': '1o3ol',
    # TODO delete
    #'1-octen-3-ol': 'oct',

    'acetone': 'ace',
    'butanal': 'but',
    'ethyl acetate': 'ea',

    'ethyl hexanoate': 'eh',
    'hexyl acetate': 'ha',
    'ethanol': 'EtOH',
    'isoamyl alcohol': 'IAol',

    # TODO change back when i'm ready to convert / recompute all old analysis
    # intermediates using old ('~kiwi', 'control mix') names
    # TODO delete?
    #'kiwi approx.': '~kiwi',
    #'control mix': 'control mix',
    'kiwi approx.': 'kmix',
    'control mix': 'cmix',

    'ethyl lactate': 'elac',
    'methyl acetate': 'ma',
    '2,3-butanedione': '2,3-b',

    'ethyl 3-hydroxybutyrate': 'e3hb',
    'trans-2-hexenal': 't2h',
    'ethyl crotonate': 'ecrot',
    'methyl octanoate': 'moct',
    # good one for acetoin (should be only current diag w/o one)?

    'linalool': 'Lin',
    'B-citronellol': 'B-cit',
    'hexanal': '6al',
    'benzaldehyde': 'benz',
    '1-pentanol': '1-5ol',
    '1-octanol': '1-8ol',
    'pentyl acetate': 'pa',
    'ethyl propionate': 'ep',
    'acetic acid': 'aa',

    # To match Remy's abbreviations exactly
    'methyl salicylate': 'ms',
    '2-heptanone': '2h',
    'ethyl butyrate': 'eb',
    '1-hexanol': '1-6ol',
    'isoamyl acetate': 'IaA',
    '2-butanone': '2-but',
    'valeric acid': 'va',
    # Another name for 'valeric acid', but the one Remy had used.
    'pentanoic acid': 'va',
}
odor2abbrev.update(_hardcoded_odor2abbrev)

_initial_odor2abbrev = dict(odor2abbrev)
def save_odor2abbrev_cache():
    # TODO log debug/info that we are writing this (/ that it was unchanged)
    if odor2abbrev != _initial_odor2abbrev:
        odor2abbrev_cache.write_bytes(pickle.dumps(odor2abbrev))

atexit.register(save_odor2abbrev_cache)


def abbrev(odor_str: str, abbrevs: Optional[Dict[str, str]] = None, *,
    component_delim: str = component_delim, conc_delim: str = conc_delimiter) -> str:
    """Abbreviates odor name in input, when an abbreviation is available.

    Args:
        odor_str: can optionally contain concentration information (followed by
        `olf.conc_delimiter`, if so).

        abbrevs: dict mapping from input names to the names (abbreviations) you want. if
            not passed, the dict `olf.odor2abbrev` is used
    """
    # TODO add tests for each of the 3 cases (doctest?)

    if component_delim in odor_str:
        # TODO refactor to use parse_odor_lists? (would need an abbrev fn that takes
        # dict input)
        odor_strs = [x.strip() for x in odor_str.split(component_delim)]

        # TODO need to pass abbrevs thru? why do i even have this branch, if it's not
        # gonna recursively call abbrev?
        return format_mix_from_strs(odor_strs, delim=component_delim)

    elif conc_delim in odor_str:
        odor = parse_odor(odor_str, require_conc=True)
        odor['name'] = abbrev(odor['name'], abbrevs=abbrevs)
        return format_odor(odor)

    if abbrevs is None:
        abbrevs = odor2abbrev

    return abbrevs.get(odor_str, odor_str)


def add_abbrevs_from_odor_lists(odor_lists: ExperimentOdors,
    name2abbrev: Optional[Dict[str, str]] = None, yaml_path: Optional[Pathlike] = None,
    *, if_abbrev_mismatch: str = 'warn', verbose: bool = False) -> None:
    """Adds name->abbreviation mappings in odor_lists to odor2abbrev input.

    Args:
        yaml_path: this is used included in some print/warning messages, but is not
            loaded.
    """
    if name2abbrev is None:
        # changing the global (module level) odor2abbrev by default
        name2abbrev = odor2abbrev

    assert if_abbrev_mismatch in ('warn', 'err')
    for odors in odor_lists:
        for odor in odors:
            try:
                abbrev = odor['abbrev']
            except KeyError:
                continue

            name = odor['name']
            if name in name2abbrev:
                prev_abbrev = name2abbrev[name]
                if abbrev != prev_abbrev:
                    # TODO print yaml_path instead of 'YAML', when available?
                    # TODO update '(hardcoded'... part of message (when it is probably
                    # more often from previous calls of this fn... esp in the
                    # al_analysis.preprocess_recording context)?
                    msg = (f'abbreviation {abbrev} (YAML) != {prev_abbrev} '
                        '(hardcoded, will be used)'
                    )
                    if if_abbrev_mismatch == 'err':
                        raise ValueError(msg)

                    elif if_abbrev_mismatch == 'warn':
                        # TODO replace w/ logging?
                        warnings.warn(msg)
            else:
                if name == abbrev:

                    msg = f'name and abbrev were both {name}'
                    if yaml_path is not None:
                        msg += f' in {yaml_path}'

                    # TODO replace w/ logging?
                    warnings.warn(msg)

                name2abbrev[name] = abbrev
                if verbose:
                    msg = f'adding {name=} -> {abbrev=}'
                    if yaml_path is not None:
                        msg += f' from {yaml_path}'

                    print(msg)


def parse_log10_conc(odor_str: str, *, require: bool = False) -> Optional[float]:
    """Takes formatted odor string to float log10 vol/vol concentration.

    Returns `None` if input does not contain `olf.conc_delimiter`.

    Args:
        odor_str: contains odor name, and generally also concentration

        require: if `True`, raises `ValueError` if `olf.conc_delimiter` is not in input

    >>> parse_log10_conc('ethyl acetate @ -2')
    -2
    """
    # If conc_delimiter is in the string, we are assuming that it should be followed by
    # parseable float concentration. Letting it err below if that is not the case.
    if conc_delimiter not in odor_str:
        if require:
            raise ValueError(f'{odor_str=} did not contain {conc_delimiter=}')

        return None

    parts = odor_str.split(conc_delimiter)
    assert len(parts) == 2
    conc_part = parts[1].strip()

    # TODO replace this try-int(...)-first strategy w/ some float formatting that
    # formats stuff w/o stuff after '.' when that component is 0 (or close enough)?
    # is there such a formatting option?
    try:
        # trying this first so that we can preserve formatting of input in round trip
        # cases, rather than adding '.0'
        #
        # int should be correct for return type of float still
        # (as far as type system goes)
        log10_conc = int(conc_part)

    # e.g. `ValueError: invalid literal for int() with base 10: '-2.'`
    except ValueError:
        # if this parsing fails, we want that error to raise, so no try/except in here
        log10_conc = float(conc_part)

    return log10_conc


def parse_odor_name(odor_str: str) -> Optional[str]:
    # TODO some way to get the generated docs to refer to the value for the constant
    # rather than having to hardcode it for reference? a plugin maybe?
    # TODO does the parse_odor_name(solvent_str) doctest run/work as a test, or need to
    # change settings / refer to solvent_str differently?
    """Takes formatted odor string to just the name of the odor.

    Returns `None` if input matches `olf.solvent_str`, but otherwise raises ValueError
    if `odor_str` does not contain `olf.conc_delimiter`.

    Args:
        odor_str: contains odor name and concentration.
            name and concentration must be separated by `olf.conc_delimiter` ('@'), with
            whitespace on either side of it.

    >>> parse_odor_name('ethyl acetate @ -2')
    'ethyl acetate'

    >>> parse_odor_name(solvent_str) is None
    True
    """
    if odor_str == solvent_str:
        return None

    # TODO take this as kwarg?
    if conc_delimiter not in odor_str:
        raise ValueError(f'{conc_delimiter=} not in {odor_str}')

    parts = odor_str.split(conc_delimiter)
    if len(parts) != 2:
        raise ValueError(f'unexpected number of {conc_delimiter=} in {odor_str}')

    return parts[0].strip()


# TODO is require_conc=True actually supported? cause currently parse_odor_name will err
# w/o that conc_delimiter... change it so that's not the case?
def parse_odor(odor_str: str, *, require_conc: bool = False) -> dict:
    return {
        'name': parse_odor_name(odor_str),
        'log10_conc': parse_log10_conc(odor_str, require=require_conc),
    }


def parse_odor_list(trial_odors_str: str, *, delim: str = component_delim,
    **parse_odor_kwargs) -> SingleTrialOdors:

    # NOTE: actually, don't think i can do this, considering i have some odors like
    # '(1S)-(+)-carene' (and in-vial mixtures, that I'm currently representing as e.g.
    # 'ea+eb @ 0', w/o the whitespace around '+')
    #
    # so that i work w/ condensed representations (e.g. 'A+B') if i want
    #delim = delim.strip()

    odor_strs = [x.strip() for x in trial_odors_str.split(delim)]

    return [parse_odor(s, **parse_odor_kwargs) for s in odor_strs]


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


# TODO how to get generated docs to show `pd.Index` instead of Index from this typehint?
def odor_index_sort_key(level: pd.Index, sort_names: bool = True,
    names_first: bool = True, name_order: Optional[List[str]] = None,
    require_in_name_order: bool = False, warn: bool = True, _debug: bool = False
    ) -> pd.Index:
    """
    Args:
        level: one level from a `pd.MultiIndex` with odor metadata.
            elements should be odor strings (as :func:`parse_odor_name` and
            :func:`parse_log10_conc`).

        sort_names: whether to use odor names as part of sort key. If False, only sorts
            on concentrations.

        names_first: if True, sorts on names primarily, otherwise sorts on
            concentrations primarily. Ignored if sort_names is False.

        name_order: list of odor names to use as a fixed order for the names.
            Concentrations will be sorted within each name.

        require_in_name_order: if True, raises ValueError if odors with not in
            name_order are present. Otherwise sorts such odors alphabetically after
            those in name_order.

        warn: if True and `require_in_name_order=False`, warns about which odors were
            not in name_order
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
    # TODO relax, so i can support '5% cleaning ammonia in water' type stuff? at least
    # a flag to relax? warn only option? (for now, decided to just change all the
    # relevant stimulus files, and try to avoid doing this in the future)
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
                # I might have liked to have an option to drop any odors like these, but
                # pd.sort_index expects the key kwarg to return Index of the same shape.

                # TODO unit test both of the paths in here
                not_in_name_order = {n for n in names if n not in name_order}
                if len(not_in_name_order) > 0:
                    err_msg = (f'{pformat(not_in_name_order)} were not in name_order='
                        f'{pformat(name_order)}'
                    )
                    if require_in_name_order:
                        raise ValueError(err_msg)

                    if warn:
                        warnings.warn(f'{err_msg}. appending to end of name_order in '
                            'alphabetical order.'
                        )
                    name_order = name_order + sorted(not_in_name_order)

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


component_level_prefix = 'odor'

def is_odor_var(var_name: Optional[str]) -> bool:
    """Returns True if column/level name or Series-key is named to store odor metadata

    Values for matching keys should store strings representing *one*, of potentially
    multiple, component odors presented (simultaneously) on a given trial. My convention
    for representing multiple components presented together one one trial is to make
    multiple variables (e.g. columns), named such as ['odor1', 'odor2', ...], with a
    different sufffix number for each component.
    """
    # For index [level] names that are not defined.
    if var_name is None:
        return False

    # TODO do i actually care to support stuff like 'odor_a'? 'odor' probably
    # (those are basically only 2 reasons this fn still exists, rather than having
    # replaced it w/ is_odor_component_level)
    return var_name.startswith(component_level_prefix)


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
# TODO also implement something for getting name order from one of the YAML configs?
# (do the loading in here? prob take either dict or YAML path)
def sort_odors(df: pd.DataFrame, *, panel_order: Optional[List[str]] = None,
    panel2name_order: Optional[Dict[str, List[str]]] = None,
    panel: Optional[str] = None, if_panel_missing='warn',
    axis: Optional[str] = None, _debug: bool = False, **kwargs) -> pd.DataFrame:
    # TODO add doctest examples clarifying how the two columns interact + what happens
    # to solvent_str (+ clarify in docstring)
    # TODO doctest examples using panel_order+panel2name_order
    # TODO TODO how to deal w/ odor2 != solvent_str?
    """Sorts DataFrame by odor index/columns.

    Args:
        df: should have columns/index-level names where `olf.is_odor_var(<col name>)`
            returns `True`

        panel_order: list of str panel names. If passed, must also provide
            panel2name_order. Will sort panels first, then odors within each panel.

        panel2name_order: maps str panel names to lists of odor name orders, for each.
            If passed, must also pass panel_order.

        panel: to specify panel for input data, if it does not have separate index
            level(s) / column indicating which panel each odor belongs to. must have a
            matching key in `panel2name_order`. all data will be assumed to belong to
            this panel.

        if_panel_missing: 'warn'|'err'|None

        axis: if None, detect which axes to sort (and may sort both). otherwise,
            expecting 'columns'|'index'

        **kwargs: passed through to :func:`odor_index_sort_key`.

    Notes:
    Index will be checked first, and if it contains odor information, will sort on that.
    Otherwise, will check and sort on matching columns.

    Sorts by concentration, then name. `solvent_str` is treated as less than all odors.

    >>> df = pd.DataFrame({
    ...     'odor1': ['B @ -2', 'A @ -2', 'A @ -3'],
    ...     'odor2': ['solvent'] * 3,
    ...     'delta_f': [1.1, 1.2, 0.9]
    ... }).set_index(['odor1', 'odor2'])

    Names are sorted alphabetically by default, then within each name they are sorted by
    concentration. Pass `names_only=False` to only sort on concentration, or
    `names_first=False` to sort on concentrations first.
    >>> sort_odors(df)
                    delta_f
    odor1  odor2
    A @ -3 solvent      0.9
    A @ -2 solvent      1.2
    B @ -2 solvent      1.1

    >>> sort_odors(df, name_order=['B','A'])
                    delta_f
    odor1  odor2
    B @ -2 solvent      1.1
    A @ -3 solvent      0.9
    A @ -2 solvent      1.2
    """
    # TODO probably delete, now that i'm allowing some panels to be missing from
    # panel2name_order
    if panel2name_order is None and panel_order is not None:
        raise ValueError('must pass panel2name_order if supplying panel_order')

    if panel2name_order is None and panel is not None:
        raise ValueError('must pass panel2name_order if supplying panel')

    # TODO want to support panel2name_order=None when panel_order is passed
    # (probably, now that i'm thinking i want [at least the option to] have things run
    # before i fill out panel2name_order for every new thing?)?
    # could just check panel2name_order not-None if not...
    panel_sort = panel_order is not None or panel2name_order is not None
    if panel_sort:
        if 'name_order' in kwargs:
            raise ValueError('when specifying panel_order, use panel2name_order '
                'instead of name_order'
            )

        if panel is not None:
            # TODO change handling of panel missing from panel2name_order, to be
            # consistent w/ if_panel_missing conditional below (+ update doc above)
            # (see comment above loop over panels below)
            assert panel in panel2name_order
            name_order = panel2name_order[panel]
            return sort_odors(df, name_order=name_order, axis=axis, **kwargs)

    kwargs['_debug'] = _debug

    def levels_to_sort(index):
        # Sorting so that if something were to accidentally re-order e.g. 'odor1',
        # 'odor2' levels, the sort order would be invariant to that, with 'odor1' always
        # taking precedence in the sort.
        return sorted([k for k in index.names if is_odor_var(k)])

    found_odor_multiindex = False
    # TODO factor out a fn for finding which axis/axes has/have odor info, so i can
    # use here and in e.g. al_analysis.sort_odors, where i also addlevel panel info (but
    # i would like it to automatically add on correct axis)
    for axis_name in ('index', 'columns'):

        if axis is not None:
            assert axis in ('index', 'columns')
            if axis != axis_name:
                continue

        index = getattr(df, axis_name)
        levels = levels_to_sort(index)

        # TODO maybe check that if len(levels) == 0 and panel_sort, we don't have
        # 'panel' in other indices?

        if len(levels) == 0:
            continue

        if _debug:
            print(f'found odor MultiIndex on axis={axis_name}')
            print(f'odor levels to sort: {levels}')

        found_odor_multiindex = True

        if not panel_sort:
            df = df.sort_index(
                key=lambda x: odor_index_sort_key(x, **kwargs),
                axis=axis_name,
                level=levels,
                sort_remaining=False,
                # So that the sort is "stable", meaning if stuff compares equal, it
                # preserves input order.
                kind='mergesort',
            )

        else:
            if 'panel' not in index.names:
                raise ValueError('panel sorting requested, but axis had odor levels '
                    "without a 'panel' level"
                )

            if panel_order is None:
                # NOTE: intentionally using order of input dict (which should be order
                # keys are added to dict)
                panel_order = list(panel2name_order.keys())

            # TODO for a case where no elements of panel2name_order value
            # iterables have overlapping names, test that this method of
            # splitting->sorting->recombining is equivalent to giving a simple key
            # for the panel + sorting w/ one name_order constructed from all of the
            # panel2name_order values

            # TODO TODO figure out what should happen if odors are shared between
            # two panels (e.g. MS|VA @ -3 in control + megamat panels) -> test

            sorted_panel_dfs = []
            # TODO do i need to do level='panel' rather than implicit by='panel'
            # TODO maybe just set sort=True if panel_order not specified
            # (and if i want to allow that...)
            panels = []

            # TODO factor out body of loop to helper fn -> call in panel=<not-None> case
            # above too (to make missing panel handling same as in here, rather than not
            # allowing it)
            #
            # NOTE: at least in pandas 1.2.4, using level= is required to not have null
            # panels dropped (despite dropna=False!), so we need to use levels= kwarg,
            # assuming we want to support null panels.
            #
            # Since we are already checking 'panel' is in index.names above, we can
            # always use level= in groupby (would need to specify as by= [or positional
            # arg 0] if 'panel' was a column).
            for panel, pdf in df.groupby(level='panel', axis=axis_name, sort=False,
                dropna=False):

                try:
                    name_order = panel2name_order[panel]

                except KeyError:
                    if if_panel_missing == 'err':
                        raise

                    elif if_panel_missing == 'warn':
                        # TODO if panel null, is there still at least some warning?
                        # if not, just warn w/ a diff message? separate warning flag to
                        # control that? or rename this one to be more general?
                        if not pd.isnull(panel):
                            # TODO want to consolidate w/ warning odor_index_sort_key
                            # will also trigger if not in name_order?
                            warnings.warn(f'{panel=} not in panel2name_order. odor '
                                'names alphabetical by default. to silence, add to '
                                'panel2name_order or set if_panel_missing=None to '
                                'silence.'
                            )

                    elif if_panel_missing is None:
                        pass
                    else:
                        raise ValueError(f"{if_panel_missing=} must be 'err'|'warn'|"
                            "None"
                        )

                    # odor names will be ordered alphanumerically by default
                    name_order = None

                if _debug:
                    print(f'{panel=}, name_order:')
                    pprint(name_order)

                # NOTE: name_order here is just getting passed to odor_index_sort_key.
                # it's not explcitly a kwarg of sort_odors.
                sorted_pdf = sort_odors(pdf, name_order=name_order, **kwargs)
                sorted_panel_dfs.append(sorted_pdf)
                panels.append(panel)

            def _panel_order_key(panel):
                if pd.isnull(panel):
                    # True > False, in case it wasn't obvious. null should go to end.
                    # doesn't matter what 2nd & 3rd components here are, as they should
                    # never vary within False 1st elements.
                    return (True, 0, '')

                try:
                    panel_index = panel_order.index(panel)

                # (e.g. panel was not in explicit panel_order / panel2name_order.keys())
                # ValueError: <panel> is not in list
                except ValueError:
                    # (to put all these at the end)
                    panel_index = float('inf')

                # secondarily sorting on panel itself, so all stuff not in panel_order
                # still have a defined order (alphanumeric)
                return (False, panel_index, panel)

            sorted_panel_dfs = [x for _, x in sorted(zip(panels, sorted_panel_dfs),
                key=lambda y: _panel_order_key(y[0])
            )]
            df = pd.concat(sorted_panel_dfs, axis=axis_name)
            continue

    # Not just returning in loop, because we may need to sort *both* the rows and the
    # columns (e.g. if the input is a correlation matrix).
    if found_odor_multiindex:
        return df

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

    if panel_sort:
        if 'panel' not in df.columns:
            raise ValueError("'panel' not in df.columns, though odors are and panel"
                ' sorting requested'
            )
        odor_cols = ['panel'] + odor_cols

    # TODO also try to keep order of columns same
    # (current approach moves odor columns to the start)
    temp_index_col = '_old_index'
    assert temp_index_col not in df.columns
    # Would have used reset_index(), but didn't see an argument to change the name
    # of the column it creates.
    df = df.copy()
    df[temp_index_col] = df.index
    old_index_name = df.index.name

    df = sort_odors(df.set_index(odor_cols),
        panel_order=panel_order, panel2name_order=panel2name_order, **kwargs
    ).reset_index()

    df = df.set_index(temp_index_col)
    df.index.name = old_index_name
    return df


# TODO maybe move to viz.py, since this is mainly intended as as helper for
# viz.with_panel_orders plotting function wrapper?
# TODO alias for the panel2name_order type (+ use in new kwarg to sort_odors too)
# TODO TODO TODO also accept if_panel_missing here (and try to implement such that some
# logic is shared w/ sort_odors, or at least try to keep behavior consistent)
def panel_odor_orders(df: pd.DataFrame,
    panel2name_order: Optional[Dict[str, List[str]]] = None, **kwargs):
    # TODO doctest example
    # TODO test+clarify in doc whether odor names in df can be a subset of those from a
    # particular panel's name_order (and whether or not solvent/pfo is a special case
    # here)
    """Returns dict of panel names to ordered unique odor strs (with concentration).

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
            # TODO implement (+ warn in right condition of) if_panel_missing here
            name_order = panel2name_order.get(panel)

        panel_df = sort_odors(panel_df, name_order=name_order, **kwargs)

        # TODO maybe factor out this (and preceding finding + sorting odor_cols)?
        mix_strs = [
            format_mix_from_strs(ser) for _, ser in panel_df[odor_cols].iterrows()
        ]
        panel2order[panel] = mix_strs

    return panel2order


# TODO type hint return type (just List[int], or List[List[int]]?)
def yaml_data2pin_lists(yaml_data: dict):
    """
    Pins used as balances can be part of these lists despite not having a corresponding
    odor in 'pins2odors'.
    """
    return [x['pins'] for x in yaml_data['pin_sequence']['pin_groups']]


def yaml_data2odor_lists(yaml_data: dict, *, sort: bool = True):
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


# TODO doc
def load_stimulus_yaml(yaml_path: Pathlike):

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    odor_lists = yaml_data2odor_lists(yaml_data)

    return yaml_data, odor_lists


# TODO may want to move to util
# TODO make a union type that also accepts pd.Series and np.ndarray in addition to
# Sequence? just trying to require that it can be sliced w/ arbitrary stride.
# TODO how to indicate that the Hashable in the argument and return type should be of
# the same type? does Hashable take any type of arguments? make my own ~mixin of
# Hashable + Generic(?)
# TODO support non-hashable input types too?
# TODO TODO add kwarg to enforce consecutive (also raising some kind of error if not)?
def remove_consecutive_repeats(odor_lists: Sequence[Hashable]
    ) -> Tuple[List[Hashable], int]:
    # TODO doc what happens if there is not an equal number of repeats for each thing,
    # or if not consecutive. error? (two diff cases)
    """Returns a list without any consecutive repeats and int # of consecutive repeats.

    Raises ValueError if there is a variable number of consecutive repeats.

    Assumed that all elements of `odor_lists` are repeated the same number of times,
    for each consecutive group of repeats. As long as any repeats are to full
    `n_repeats` and consecutive, it is ok for a particular odor (e.g. solvent control)
    to be repeated `n_repeats` times in each of several different positions.

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','a','b','b','b'])
    >>> without_repeats
    ['a', 'b']
    >>> n
    3

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','b','b','a','a'])
    >>> without_repeats
    ['a', 'b', 'a']
    >>> n
    2

    >>> without_repeats, n = remove_consecutive_repeats(['a','a','a','b','b'])
    Traceback (most recent call last):
    ValueError: variable number of consecutive repeats

    Wanted to also take a list-of-lists-of-dicts, where each dict represents one odor
    and each internal list represents all of the odors on one trial, but the internal
    lists (nor the dicts they contain) would not be hashable, and thus cannot work with
    Counter as-is.
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


# TODO take float format specifier?
def format_odor(odor_dict, conc: bool = True, name_conc_delim: Optional[str] = None,
    conc_key: str = 'log10_conc', cast_int_concs: bool = False):
    """Takes a dict representation of an odor to a pretty str.

    Expected to have at least 'name' key, but will also use 'log10_conc' (or `conc_key`)
    if available, unless `conc=False`.

    Args:
        cast_int_concs: if True, will convert (log10) concentrations to integer if they
            are `np.isclose` to their nearest integer.

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
        # TODO warn/fail if name isn't in some set here (e.g. NO_ODOR/solvent_str)?
        # or just don't do this? wouldn't this fail for some of undiluted kiwi samples,
        # for one?
        if odor_dict[conc_key] is None:
            return solvent_str

        if conc:
            log10_conc = odor_dict[conc_key]

            if cast_int_concs:
                int_log10_conc = int(round(log10_conc))
                if np.isclose(log10_conc, int_log10_conc):
                    log10_conc = int_log10_conc

            ostr += f'{name_conc_delim}{log10_conc}'

    return ostr


# TODO TODO decorator or some other place to store minimum set of keys (and types?) for
# these formatting functions (at least those that take a Series as one option)?
# (so that hong2p.viz.callable_ticklabels can automatically convert / make good error
# messages if they are missing)
# TODO factor out this union type (+ probably add np.ndarray), and use in
# remove_consecutive_repeats as well (or maybe in this particular fn, i actually want
# Iterable[str] in the Union? not striding here..
def format_mix_from_strs(odor_strs: Union[Sequence[str], pd.Series, Dict[str, Any]], *,
    delim: str = component_delim, warn_unused_levels: bool = False):

    # TODO what's an example of Series input? why using same fn name for this?
    # doc at least... (can make some handling of iterating over index values easier,
    # when we get tuples that we can then zip with index names. can then be easier to
    # drop repeats / other info, without worrying about lengths and things like that)
    if hasattr(odor_strs, 'keys'):
        odor_keys = [x for x in odor_strs.keys() if is_odor_var(x)]

        if warn_unused_levels and len(odor_keys) < len(odor_strs):
            nonodor_keys = [x for x in odor_strs.keys() if x not in odor_keys]
            # TODO replace w/ logging warning?
            warnings.warn('format_mix_from_strs: ignoring levels not starting with '
                f"'odor' ({nonodor_keys})"
            )

        odor_strs = [odor_strs[k] for k in odor_keys]

    odor_strs = [x for x in odor_strs if x != solvent_str]
    if len(odor_strs) > 0:
        return delim.join(odor_strs)
    else:
        return solvent_str


def format_odor_list(odor_list: SingleTrialOdors, *, delim: str = component_delim,
    **kwargs) -> str:
    """Takes list of dicts representing odors for one trial to pretty str.
    """
    odor_strs = [format_odor(x, **kwargs) for x in odor_list]
    return format_mix_from_strs(odor_strs, delim=delim)


# TODO use to format odor[mixtures] in al_analysis
def strip_concs_from_odor_str(odor_str: str, **kwargs) -> str:
    """
    Works with input representing either single components or air mixtures of multiple.

    Args:
        **kwargs: passed thru to `format_odor`
    """
    # TODO thread thru component delim (delim= kwarg) here?
    odor_list = parse_odor_list(odor_str)
    return format_odor_list(odor_list, conc=False, **kwargs)


# TODO indicate subclass of pd.Index (Type[pd.Index]?) as return type
def odor_lists_to_multiindex(odor_lists: ExperimentOdors, *,
    # TODO try to update all over code to work w/ pad_to_n_odors=None default?  (would
    # prob be a step in direction of being agnostic to number of components, rather than
    # generally expecting the metadata for 2 [even if often only 1 was used])
    sort_components: bool = True, pad_to_n_odors: Optional[int] = None, #2,
    **format_odor_kwargs) -> pd.MultiIndex:
    # TODO doctest
    """
    Args:
        pad_to_n_odors: if `int`, returned `MultiIndex` will have at least this many
            levels dedicated to odor components (+ the 1 'repeat' level always
            included).
    """
    if min(len(x) for x in odor_lists) < 1:
        raise ValueError('odor_lists should not have any empty lists')

    max_components_per_trial = max(len(x) for x in odor_lists)

    if pad_to_n_odors is not None and pad_to_n_odors > max_components_per_trial:
        max_components_per_trial = pad_to_n_odors

    # of length equal to number of trials. each element will have a str for each
    # component (currently padding w/ 'solvent' up to max # of components seen, so all
    # of length equal to max # of components)
    odor_strs = []

    odor_mix_counts = defaultdict(int)
    odor_mix_repeats = []

    for odor_list in odor_lists:

        curr_trial_odor_strs = []
        for odor in odor_list:
            # TODO move into format_odor? why not? (behind abbrev=True kwarg?)
            if odor['name'] in odor2abbrev:
                odor = dict(odor)
                odor['name'] = odor2abbrev[odor['name']]
            #

            odor_str = format_odor(odor, **format_odor_kwargs)
            curr_trial_odor_strs.append(odor_str)

        if sort_components:
            curr_trial_odor_strs = sorted(curr_trial_odor_strs)

        curr_trial_odor_strs = (curr_trial_odor_strs +
            [solvent_str] * (max_components_per_trial - len(odor_list))
        )
        assert len(curr_trial_odor_strs) == max_components_per_trial

        odor_strs.append(curr_trial_odor_strs)

        # TODO test w/ input where odor1 odor2 are not sorted / always in a
        # consistent order (+ probably sort before counting to fix)
        # TODO maybe i want it to be the responsibility of the caller to sort
        # multiindices if they want (after) (they'd then have to recalc repeat, so
        # idk...) (maybe i should just err / warn if same mix seen in >1 diff orders?)?
        odor_mix = tuple(curr_trial_odor_strs)
        odor_mix_repeats.append(odor_mix_counts[odor_mix])
        odor_mix_counts[odor_mix] += 1

    # NOTE: relying on sorting odor_list(s) at load time (so not here? where?) seems to
    # produce consistent ordering (but doing the same here shouldn't cause any extra
    # issues, right?), though that alphabetical ordering (based on full odor names) is
    # different from what would be produced sorting on abbreviated odor names (at least
    # in some cases)

    index = pd.MultiIndex.from_arrays(list(np.array(odor_strs).T) + [odor_mix_repeats])
    names = [f'odor{i}' for i in range(1, max_components_per_trial + 1)] + ['repeat']
    index.names = names

    return index


# TODO decide if i want 'odor' -> True (and if so, add that behavior here before
# replacing is_odor_var with this)
# TODO and do i want to support just anything starting w/ 'odor'? e.g. 'odor_a'? was i
# actually using w/ any inputs like that? just leaving both this and is_odor_var for
# now...
def is_odor_component_level(level_name: Optional[str]) -> bool:
    """Returns True if column/level name or Series-key is named to store odor metadata

    Values for matching keys should store strings representing *one*, of potentially
    multiple, component odors presented (simultaneously) on a given trial. My convention
    for representing multiple components presented together one one trial is to make
    multiple variables (e.g. columns), named such as ['odor1', 'odor2', ...], with a
    different sufffix number for each component.
    """
    # for index [level] names that are not defined.
    if level_name is None:
        return False

    # .isdigit() also works for things with multiple characters (e.g. '10')
    return (
        level_name.startswith(component_level_prefix) and
        level_name[len(component_level_prefix):].isdigit()
    )


def n_odor_component_levels(df: pd.DataFrame) -> int:
    # TODO do i want this to drop levels w/ all solvent_str? flag? prob doesn't matter
    # for current usage.
    odor_level_names = [x for x in df.index.names if is_odor_component_level(x)]
    # TODO relax? may need to modify assertion below, if so
    assert len(odor_level_names) > 0

    # should all be consecutive starting at 1
    level_nums = set(int(x[len(component_level_prefix):]) for x in odor_level_names)
    assert level_nums == set(range(1, 1 + len(level_nums)))

    return len(odor_level_names)


def pad_odor_index_to_n_components(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pads dataframe odor index, so that it has `n` 'odor<n>' component levels.

    Args:
        n: target number of odor levels

    Odors presented together (e.g. in one trial, mixed in air), should each have their
    own level in the odor MultiIndex, with `olf.solvent_str` used to fill when a given
    trial had less components presented at once.
    """
    assert n >= 1

    n_odor_levels = n_odor_component_levels(df)
    if n_odor_levels == n:
        return df

    assert n > n_odor_levels

    old_names = list(df.index.names)
    last_idx = max(i for i, x in enumerate(old_names) if is_odor_component_level(x))

    new_names = old_names[:(last_idx + 1)]
    for i in range(n_odor_levels, n):
        # my convention for these level names has to been to start from 'odor1', hence
        # the +1
        new_name = f'{component_level_prefix}{i+1}'
        assert new_name not in df.index.names
        df = util.addlevel(df, new_name, solvent_str)
        new_names.append(new_name)

    new_names.extend(old_names[(last_idx + 1):])

    return df.reorder_levels(new_names)


def pad_odor_indices_to_max_components(dfs: Sequence[pd.DataFrame]
    ) -> Sequence[pd.DataFrame]:
    """Pads odor index each each dataframe to max number of input component levels.
    """
    max_n_odor_component_levels = max(n_odor_component_levels(x) for x in dfs)
    return [pad_odor_index_to_n_components(df, max_n_odor_component_levels)
        for df in dfs
    ]

