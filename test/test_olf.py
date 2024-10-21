
from typing import List

import pytest

import pandas as pd
import numpy as np

from hong2p.olf import solvent_str, sort_odors


# TODO move to conftest.py? share w/ test_util.py where i copied some of this to?
def _make_df(rng, odor1, odor2=None, odors_in_index=True, panel=None):
    odor_keys = ['odor1', 'odor2']
    for_df = {
        'odor1': odor1,
        # TODO flag to not make with odor2 at all (and test with that)
        'odor2': [solvent_str] * len(odor1) if odor2 is None else odor2,
        'delta_f': rng.random(len(odor1)),
    }
    if panel is not None:
        for_df['panel'] = panel
        odor_keys.insert(0, 'panel')

    df = pd.DataFrame(for_df)

    if odors_in_index:
        df = df.set_index(odor_keys)

    return df

def _c(names: List[str], c: int = 0) -> List[str]:
    """Adds concentrations suffixes '@ <c>' (e.g. '@ 0') to list of odor names
    """
    return [f'{n} @ {c}' for n in names]

def _odor_level(df, n):
    # TODO fall back to trying to get it as a column of df (for odors_in_index=False
    # case. even want to support that case tho?)
    return list(df.index.get_level_values(f'odor{n}'))

def _odor1(df):
    return _odor_level(df, 1)

def _odor2(df):
    return _odor_level(df, 2)

def _panel(df):
    return list(df.index.get_level_values('panel'))

# TODO easier way?
# Just need since float('nan') != float('nan') (and pandas seems to turn many None into
# NaN internally).
def naneq(seq1, seq2):
    if len(seq1) != len(seq2):
        return False

    for x, y in zip(seq1, seq2):
        if not (x == y or (pd.isnull(x) and pd.isnull(y)) ):
            return False

    return True


def test_sort_odors_with_panel(rng):
    input_odor1 = _c(['B', 'A', 'Y', 'X', 'Y', 'Z', 'D', 'C'])
    input_panel = ['1', '1', '2', '2', '3', '3', None, None]
    panel2name_order = {
        '1': ['A', 'B'],
        '2': ['X', 'Y'],
        # this also checks it doesn't blow up if two panels have an odor w/ same name
        '3': ['Y', 'Z'],
    }
    input_df = _make_df(rng, odor1=input_odor1, panel=input_panel)
    expected_panel = ['1', '1', '2', '2', '3', '3', None, None]
    expected_odor1 = _c(['A', 'B', 'X', 'Y', 'Y', 'Z', 'C', 'D'])

    df = sort_odors(input_df, panel2name_order=panel2name_order)
    odor1 = _odor1(df)
    panel = _panel(df)
    assert naneq(panel, expected_panel)
    assert naneq(odor1, expected_odor1)

    # TODO also try with if_panel_missing=None (should not warn)
    # TODO test that by default it does warn (if panel missing)
    # TODO test that if if_panel_missing='err', it errs

    # TODO TODO try a few different kwargs (which?)?

    # TODO TODO try one with panel_order=['2', '1'] (leave '3' out)

    # TODO TODO test w/ panel missing from panel_order (pass panel_order explicitly or
    # leave out panel from panel2name_order)

    # TODO change sort_odors to not warn (about panel not in panel2name_order) if panel
    # is null (None/NaN)
    input_odor1 = _c(['D', 'C', 'B', 'A', 'J', 'I', 'Y', 'X', 'Y', 'Z'])
    # Checking that null panel get moved to end
    input_panel = [None, None, '1', '1', '0', '0', '2', '2', '3', '3']
    # Note that panel '0' is missing from keys (and thus, from default panel_order)
    # Also checking this gets moved to end.
    # TODO what should order of null and '0' panel be tho? probably null at very end
    # always?
    panel2name_order = {
        # Panels '2' and '1' should be in reverse order now (should be in order in keys
        # here)
        '2': ['X', 'Y'],
        '1': ['A', 'B'],
        # this also checks it doesn't blow up if two panels have an odor w/ same name
        '3': ['Y', 'Z'],
    }
    input_df = _make_df(rng, odor1=input_odor1, panel=input_panel)
    # TODO where will it put the NaN tho? at end always? test it moves from beginning
    # then? panels even comparable (test w/ panel_order passed) then?
    expected_panel = ['2', '2', '1', '1', '3', '3', '0', '0', None, None]
    expected_odor1 = _c(['X', 'Y', 'A', 'B', 'Y', 'Z', 'I', 'J', 'C', 'D'])

    # just setting is_panel_missing=None to not have the warning about panel '0'
    # missing from panel2name_order (which was intentional)
    df = sort_odors(input_df, panel2name_order=panel2name_order, if_panel_missing=None)
    odor1 = _odor1(df)
    panel = _panel(df)
    assert naneq(panel, expected_panel)
    assert naneq(odor1, expected_odor1)

    # TODO add test w/ panel=<x> instead of index-level / column specifying <x>


def test_sort_odors(rng):
    # TODO should these all just be separate tests? or grouped into a few?
    # use pytest parameterize?
    #
    # TODO add test w/ name_order containing names not in data (shouldn't affect
    # anything)
    #
    # TODO test case w/ both rows and columns needing sorting (w/ and w/o panel
    # presorting)
    #
    # Tuples of (odor1 input, kwargs to sort_odors, expected output odor1)
    input_kwarg_output_tuples = [
        # These first two were just copied from my doctest examples here.
        # TODO use doctest tests in pytest and remove them?
        (
            ['B @ -2', 'A @ -2', 'A @ -3'],
            dict(),
            # Sort by name (alphabetically), then concentration. Ascending (as always).
            ['A @ -3', 'A @ -2', 'B @ -2']
        ),
        (
            ['B @ -2', 'A @ -2', 'A @ -3'],
            dict(name_order=['B', 'A']),
            # Sorted by name, then concentration.
            ['B @ -2', 'A @ -3', 'A @ -2']
        ),

        (
            ['B @ -2', 'A @ -2', solvent_str, 'A @ -3'],
            # Sort by concentration only (by default names are sorted alphabetically,
            # with priority over concs). This input would cause failure if an
            # alphabetical sort *were* applied first.
            dict(sort_names=False),
            # The sort is stable wrt things not sorted on, so 'B @ -2' is first in the
            # output here simply because it occurrs first in the input.
            [solvent_str, 'A @ -3', 'B @ -2', 'A @ -2']
        ),

        (
            ['B @ -2', 'A @ -2', 'A @ -3'],
            # Should be equiv to no kwargs, since by default names are sorted first,
            # alphabetically.
            dict(name_order=['A', 'B']),
            # Just reverse name_order wrt above.
            ['A @ -3', 'A @ -2', 'B @ -2']
        ),

        (
            ['B @ -2', 'A @ -2', solvent_str, 'A @ -3'],
            dict(),
            [solvent_str, 'A @ -3', 'A @ -2', 'B @ -2']
        ),
        (
            ['B @ -2', 'A @ -2', solvent_str, 'A @ -3'],
            dict(name_order=['B', 'A']),
            # Sorted by name, then concentration.
            [solvent_str, 'B @ -2', 'A @ -3', 'A @ -2']
        ),

        (
            # Has to change order of A and B while still sorting on concs first.
            ['B @ -2', 'A @ -2', solvent_str, 'B @ -3'],
            dict(names_first=False),
            [solvent_str, 'B @ -3', 'A @ -2', 'B @ -2']
        ),
    ]
    for odor1_in, kwargs, odor1_out in input_kwarg_output_tuples:
        df = _make_df(rng, odor1_in)
        sdf = sort_odors(df, **kwargs)
        odor1 = _odor1(sdf)
        assert odor1 == odor1_out

        df = df.T.copy()
        sdf = sort_odors(df, **kwargs)
        odor1 = list(sdf.columns.get_level_values('odor1'))
        assert odor1 == odor1_out

        # Testing all the same cases (overkill, but w/e...) also work w/ odors in
        # columns instead of MultiIndex levels.
        df = _make_df(rng, odor1_in).reset_index()
        sdf = sort_odors(df, **kwargs)
        odor1 = list(sdf['odor1'])
        assert odor1 == odor1_out

        # TODO try some where _make_df call has odors_in_index=False?
        # (eh... want to support?)

        # TODO TODO why was this test not failing w/ the MultiIndexed input, but was w/
        # just one 'odor' col? (for input where it's there is a 'solvent' element and we
        # are sorting names alphabetically, and cannot compare float -inf for solvent
        # and unprocessed str names)
        #
        # Testing all same cases work w/o 'odor2' level, first where it's a column
        df = df.rename(columns={'odor1': 'odor'}).drop(columns='odor2')
        sdf = sort_odors(df, **kwargs)
        odor1 = list(sdf['odor'])
        assert odor1 == odor1_out

        # And now where it's a regular index (NOT a MultiIndex, which all prior tests
        # sorting index have been).
        df = df.set_index('odor')
        sdf = sort_odors(df, **kwargs)
        odor1 = list(sdf.index)
        assert odor1 == odor1_out


    # Tuples of:
    # (odor1 in, odor2 in, sort_odors kwargs, expected odor1 out, expected odor2 out)
    input_kwarg_output_tuples = [
        (
            ['B @ -2', 'B @ -3', solvent_str, solvent_str, 'B @ -2', 'B @ -3', 'B @ -2'],
            ['A @ -2', 'A @ -2', solvent_str, 'A @ -2', 'A @ -3', 'A @ -3', solvent_str],
            dict(),
            [solvent_str, solvent_str, 'B @ -3', 'B @ -3', 'B @ -2', 'B @ -2', 'B @ -2'],
            [solvent_str, 'A @ -2', 'A @ -3', 'A @ -2', solvent_str, 'A @ -3', 'A @ -2'],
        ),
    ]
    for odor1_in, odor2_in, kwargs, odor1_out, odor2_out in input_kwarg_output_tuples:
        df = _make_df(rng, odor1_in, odor2_in)
        sdf = sort_odors(df, **kwargs)
        odor1 = _odor1(sdf)
        odor2 = _odor2(sdf)
        assert odor1 == odor1_out
        assert odor2 == odor2_out

        df = _make_df(rng, odor1_in, odor2_in)
        df = df.reorder_levels(['odor2', 'odor1'])
        sdf = sort_odors(df, **kwargs)
        odor1 = _odor1(sdf)
        odor2 = _odor2(sdf)
        assert odor1 == odor1_out
        assert odor2 == odor2_out

    # TODO TODO (if i implement / want to implement) sort_odors kwarg(s) to sort on
    # other keys before / after sorting on odors, test that here too
    # (could probably just do sequential sorts w/ a stable sort. actually, not if i want
    # to sort on something with higher lexsort priority than sorting on odors...)
