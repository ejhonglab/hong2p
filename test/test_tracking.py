#!/usr/bin/env python3

import pytest
import numpy as np
from scipy.spatial.distance import pdist

from hong2p import util


pytestmark = pytest.mark.skip(reason="WIP + tested code unused")

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

    if verbose: #and not (add_n or sub_n):
        # not meaningful here (cause more / less data than used for matching)
        print('Original centers:')
        _print_center_seq([cs[orig_idx] for cs in centers])

    indices = []
    center_sequence = []
    some_subbed = False
    # TODO have starting shape reasonable (same as w/o add/sub), even
    # if i do want to leave NaN for comparison
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
        '''
        cs = cs.copy()
        # To leave NaN in output.
        cs[np.setdiff1d(np.arange(len(cs)), idx)] = np.nan
        '''
        center_sequence.append(cs[idx])

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


max_cost = 10
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
        centers = util.make_test_centers(initial_n=initial_n, nt=nt, p=None,
            add_diameters=False, verbose=True
        )
        util.check_no_roi_jumps(centers, max_cost)
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

    center_seq_no_nan = []
    for cs in center_sequence:
        center_seq_no_nan.append(cs[~ np.isnan(cs[:,0])])

    print(kwarg_str, end=' ... ')
    try:
        new_centers = util.correspond_and_renumber_rois(
            center_seq_no_nan, max_cost=max_cost
        )
    except:
        raise

    new_center_seq = []
    #import ipdb; ipdb.set_trace()
    for i, cs in enumerate(center_sequence):
        #ncs = np.empty_like(cs) * np.nan
        #print(ncs)
        #import ipdb; ipdb.set_trace()
        new_center_seq.append(cs[np.argsort(indices[i])])
    center_sequence = new_center_seq

    if verbose:
        print('\nnew_centers before any post processing:')
        for nc in new_centers:
            print(nc)
        print('')

    if 'reverse' in kwargs and 0 in kwargs['reverse']:
        # If we flipped the first set of centers, all centers should be flipped,
        # since renumber_rois gets initial indices from first set of centers.
        new_centers = np.flip(new_centers, axis=1)

    #'''
    if 'randomize' in kwargs and 0 in kwargs['randomize']:
        new_centers_c0_unpermuted = []
        indices0 = np.argsort(indices[0])
        for cs in new_centers:
            new_centers_c0_unpermuted.append(cs[indices0])
        new_centers = np.stack(new_centers_c0_unpermuted)

        indices[0] = indices[0][indices0]

        if verbose:
            print('after undoing randomization of first set of centers')
            for nc in new_centers:
                print(nc)
            print('')
    #'''

    # TODO TODO TODO some tests where NaN is NOT removed from *output*
    # (so that offsets across across frames can be checked)
    '''
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

        print('after removing NaN:')
        for nc in new_centers:
            print(nc)
        print('')
    '''

    #'''
    if add_n or sub_n:
        for idx, c, nc in zip(indices, center_sequence, new_centers):
            try:
                assert np.array_equal(c, nc[idx])
            except AssertionError:
                print('assertion fail')
                #import ipdb; ipdb.set_trace()
                raise
    else:
        assert np.array_equal(centers, new_centers)
    print('PASS')
    #'''

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


def test_correspond_and_renumber(exit_after_first=False):
    # seems like 4,5 match behavior of 3 so farc
    #for nt in (2, 3, 4, 5):
    '''
    kwargs_list = []
    for nt in (2, 3):
        verbose = False
        idx_to_change = list(set(
            ((1,), (0,), tuple(), tuple(range(1, nt)), (nt - 1,))
        ))
        kwargs_list += [{'reverse': i} for i in idx_to_change]

        # seem to track w/ results from reversing
        kwargs_list += [{'randomize': i} for i in idx_to_change]

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

        for kwargs_dict in kwargs_list:
            if 'nt' not in kwargs_dict:
                kwargs_dict['nt'] = nt
    '''

    verbose = True
    kwargs_list = [
        #dict(nt=2, randomize=(0,)),
        dict(nt=2, randomize=(0,), sub_n=((1,), 1))
    ]
    #kwargs_list = [dict(randomize=(0,))]
    #'''

    for ki, kwargs in enumerate(kwargs_list):
        nt = kwargs.pop('nt')
        #verbose = False
        #if nt == 3 and kwargs == {'reverse': (2,)}:
        #    verbose = True

        passed = help_test_correspond_and_renumber(nt, verbose=verbose,
            **kwargs
        )
        if exit_after_first:
            import sys; sys.exit()

        '''
        if not passed:
            help_test_correspond_and_renumber(nt, verbose=True, **kwargs)
            print('exiting')
            import sys; sys.exit()
        '''


if __name__ == '__main__':
    #test_correspond_and_renumber(exit_after_first=True)
    test_correspond_and_renumber()

