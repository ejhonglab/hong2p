#!/usr/bin/env python3

"""
Generates some descriptors of whole-frame-average responses to each odor, using
each recording in the database.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import util as u


def format_odor_conc(name, log10_conc):
    if log10_conc is None:
        return name
    else:
        # TODO tex formatting for exponent
        #return r'{} @ $10^{{'.format(name) + '{:.2f}}}$'.format(log10_conc)
        return '{} @ $10^{{{:.2f}}}$'.format(name, log10_conc)


def format_mixture(*args):
    if len(args) == 2:
        n1, n2 = args
        log10_c1 = None
        log10_c2 = None
    elif len(args) == 4:
        n1, n2, log10_c1, log10_c2 = args
    else:
        raise ValueError('incorrect number of args')

    if n1 == 'paraffin':
        title = format_odor_conc(n2, log10_c2)
    elif n2 == 'paraffin':
        title = format_odor_conc(n1, log10_c1)
    else:
        title = '{} + {}'.format(
            format_odor_conc(n1, log10_c1),
            format_odor_conc(n2, log10_c2)
        )

    return title


def main():
    response_stats = u.latest_response_stats()
    response_stats = u.merge_odors(response_stats)

    cols_to_hist = [
        #'exp_scale',
        'avg_dff_5s',
        'avg_zchange_5s'
    ]

    odor_cols = [
        'name1',
        'name2',
        'log10_conc_vv1',
        'log10_conc_vv2'
    ]
    odor_groups = response_stats.groupby(odor_cols)
    # TODO count # w/ len2 for sanity checking (should be equal to number of
    # comparisons and size of my odors panel)
    print(len(odor_groups))
    for odors, odor_group in odor_groups:
        title = format_mixture(*odors)
        print(title)
        # TODO TODO why is n_flies not always coming out even?
        # stuff from 18th w/ 2 repeats? or some other mistake?
        print('n_flies={}'.format(len(odor_group) / 3))
        print('n_repeats={}'.format(len(odor_group)))

        # TODO TODO investigate cases where this is not the case
        # not really 2 repeat cases right? seems like no...
        # which would indicate some other error
        assert len(odor_group) == 3 * len(odor_group.recording_from.unique())
        # TODO TODO maybe bring back that fn to check that each recording has
        # all the stuff it should (should print comparisons w/ missing repeats?)
        # TODO was this from stopping some inserts early? transactions could
        # have helped, if so...
        # TODO what was 02-27 16:45:33, if there was a recording started at
        # 16:54:22 ????

        odor_group[cols_to_hist].hist()
        # TODO actually set title. this isn't working.
        # use ret of hist()?
        plt.title(title)

        # TODO plots response stats over days (to notice experimental drift)


        # TODO plot histograms of (a subset of?) the stats


        # TODO plot histograms of (a subset of?) the stats conditional on
        # recording ultimately being accepted or not
        # (maybe green vs red overlayed or something?)

        
    # TODO TODO rank odors (maybe restricted to an odor panel) from
    # strongest to weakest

    # TODO anything across all odors?
    # maybe overlay within stats to compare odors?
    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

