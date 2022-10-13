
import pytest

import pandas as pd
import numpy as np

from hong2p.olf import solvent_str, sort_odors


def _make_df(rng, odor1, odor2=None, odors_in_index=True):
    df = pd.DataFrame({
        'odor1': odor1,
        'odor2': [solvent_str] * len(odor1) if odor2 is None else odor2,
        'delta_f': rng.random(len(odor1)),
    })

    if odors_in_index:
        df = df.set_index(['odor1', 'odor2'])

    return df

def _odor_level(df, n):
    return list(df.index.get_level_values(f'odor{n}'))

def _odor1(df):
    return _odor_level(df, 1)

def _odor2(df):
    return _odor_level(df, 2)


# TODO test panel2name_order + panel_order kwargs (include odor overlap between panels)
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
