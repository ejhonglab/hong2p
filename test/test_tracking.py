#!/usr/bin/env python3

import pytest
import numpy as np
from scipy.spatial.distance import pdist

import hong2p.util as u


def _to_center_seq(centers, reverse=tuple(), randomize=tuple(),
        add_n=None, sub_n=None, verbose=False):

    assert len({cs.shape for cs in centers}) == 1
    orig_idx = np.arange(len(centers[0]))

    if add_n is not None:
        add_idx, add_n = add_n
        held_out = len(add_idx) * add_n
        # just holding out stuff at end now, b/c otherwise naive comparison
        # of center sequences would fail (cause orig has held out stuff in
        # orig position, but output of renumber will put it at end)
        #held_out_idx = \
        #    np.random.choice(len(centers[0]), size=held_out, replace=False)
        held_out_idx = np.arange(len(centers[0]) - held_out, len(centers[0]))
        assert len(held_out_idx) == held_out
        orig_idx = orig_idx[~ np.isin(orig_idx, held_out_idx)]

    if sub_n is not None:
        sub_idx, sub_n = sub_n

    if verbose and not (add_n or sub_n):
        # not meaningful here (cause more / less data than used for matching)
        print('Original centers:')
        _print_center_seq([cs[orig_idx] for cs in centers])

    indices = []
    center_sequence = []
    some_subbed = False
    for i, cs in enumerate(centers):
        if sub_n and i in sub_idx:
            to_sub = np.random.choice(orig_idx, size=sub_n, replace=False)
            orig_idx = orig_idx[~ np.isin(orig_idx, to_sub)]
            some_subbed = True

        if add_n and i in add_idx:
            assert len(held_out_idx[:add_n]) == add_n
            last_n_held_out = len(held_out_idx)
            orig_idx = np.sort(np.concatenate((orig_idx, held_out_idx[:add_n])))
            held_out_idx = held_out_idx[add_n:]
            assert len(held_out_idx) == last_n_held_out - add_n
            held_out -= add_n

        idx = orig_idx.copy()
        if i in reverse:
            assert i not in randomize
            idx = idx[::-1]
        elif i in randomize:
            idx = np.random.permutation(idx)

        indices.append(idx)

        # Tossing any diameter information for now.
        center_sequence.append(cs[idx, :2])

    if add_n:
        assert held_out == 0
    if sub_n:
        assert some_subbed

    return indices, center_sequence


def _print_center_seq(center_sequence):
    for t, centers_t in enumerate(center_sequence):
        print(f'centers at t={t}')
        print(centers_t)
    print('')


max_cost = 20
def help_test_correspond_and_renumber(nt, verbose=False, **kwargs):
    kwarg_str = f'nt={nt}, '
    kwarg_str += ', '.join(['='.join([k, str(v)])  for k,v in kwargs.items()])

    n_max_attempts = 100
    np.random.seed(7)
    initial_n = 5
    if 'add_n' in kwargs:
        add_indices, add_n = kwargs['add_n']
        initial_n += len(add_indices) * add_n
    else:
        add_n = None
    if 'sub_n' in kwargs:
        sub_n = True
    else:
        sub_n = False

    # TODO maybe also check we cant sub_n to having no ROIs before end

    for _ in range(n_max_attempts):
        centers = u.make_test_centers(initial_n=initial_n, nt=nt, p=None,
            add_diameters=False, verbose=True
        )
        if all([pdist(ct).min() > max_cost for ct in centers]):
            break
        centers = None
    assert centers is not None, ('could not generate points all far enough '
        f'apart in {n_max_attempts} attempts')

    indices, center_sequence = \
        _to_center_seq(centers, verbose=verbose, **kwargs)

    if verbose:
        print('AFTER any transformations:')
        _print_center_seq(center_sequence)

    try:
        new_centers = u.correspond_and_renumber_rois(
            center_sequence, max_cost=max_cost
        )
    except:
        print(kwarg_str)
        raise

    if 'reverse' in kwargs and 0 in kwargs['reverse']:
        # If we flipped the first set of centers, all centers should be flipped,
        # since renumber_rois gets initial indices from first set of centers.
        new_centers = np.flip(new_centers, axis=1)

    if 'randomize' in kwargs and 0 in kwargs['randomize']:
        new_centers_c0_unpermuted = []
        indices0 = np.argsort(indices[0])
        for cs in new_centers:
            new_centers_c0_unpermuted.append(cs[indices0])
        new_centers = np.stack(new_centers_c0_unpermuted)

    if add_n or sub_n:
        assert not ('reverse' in kwargs or 'randomize' in kwargs)

    if add_n or sub_n:
        new_centers_tmp = []
        for nc in new_centers:
            nc_isnan = np.isnan(nc)
            assert np.array_equal(
                nc_isnan.any(axis=1), nc_isnan.all(axis=1)
            )
            nc = nc[~ nc_isnan.any(axis=1)]
            new_centers_tmp.append(nc)
        new_centers = new_centers_tmp

    if add_n or sub_n:
        for c, nc in zip(center_sequence, new_centers):
            assert np.array_equal(c, nc)
    else:
        assert np.array_equal(centers, new_centers)

    """
    # TODO delete try / except
    try:
        if add_n or sub_n:
            for c, nc in zip(center_sequence, new_centers):
                assert np.array_equal(c, nc)
        else:
            assert np.array_equal(centers, new_centers)
        print(f'assertion PASS (nt={nt}, {kwarg_str})')
        return True
    except AssertionError:
        print(f'assertion failed (nt={nt}, {kwarg_str})')
        print(kwargs)
        #'''
        print('original centers:')
        if add_n or sub_n:
            for c in center_sequence:
                print(c)
        else:
            for c in centers:
                print(c)

        print('returned centers:')
        for c in new_centers:
            print(c)
        #'''
        #import sys; sys.exit()
        #import ipdb; ipdb.set_trace()
        return False
    """


# TODO move to something under test dir
def test_c_and_r_const_n_rois():
    # seems like 4,5 match behavior of 3 so far
    #for nt in (2, 3, 4, 5):
    for nt in (2, 3):
        #'''
        idx_to_change = list(set(
            ((1,), (0,), tuple(), tuple(range(1, nt)), (nt - 1,))
        ))
        kwargs_list = []
        # TODO TODO uncomment
        kwargs_list += [{'reverse': i} for i in idx_to_change]
        #

        # seem to track w/ results from reversing
        #kwargs_list += [{'randomize': i} for i in idx_to_change]

        # Not adding / removing to first set of centers,
        # since that will likely mean more work here to compare
        idx_to_change = list(set(
            tuple(i for i in idxs if i != 0) for idxs in idx_to_change
        ))
        idx_to_change = [x for x in idx_to_change if len(x) > 0]

        # maybe also do interactions between these and above
        for add in (0, 1, 2):
            #for sub in (0, ): #1, 2):
            for sub in (0, 1, 2):
                if add == 0 and sub == 0:
                    continue

                for i in idx_to_change:
                    a_s_kwargs = dict()
                    if add:
                        a_s_kwargs['add_n'] = (i, add)
                    if sub:
                        a_s_kwargs['sub_n'] = (i, sub)
                    kwargs_list.append(a_s_kwargs)
        '''

        nt = 3
        kwargs_list = [dict(add_n=((1, 2), 1), sub_n=((1, 2), 2))]
        '''

        for kwargs in kwargs_list:

            verbose = False
            #verbose = True
            #if nt == 3 and kwargs == {'reverse': (2,)}:
            #    verbose = True

            passed = help_test_correspond_and_renumber(nt, verbose=verbose,
                **kwargs
            )
            #import sys; sys.exit()
            '''
            if not passed:
                help_test_correspond_and_renumber(nt, verbose=True, **kwargs)
                print('exiting')
                import sys; sys.exit()
            '''

