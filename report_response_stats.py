#!/usr/bin/env python3

"""
Generates some descriptors of whole-frame-average responses to each odor, using
each recording in the database.
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns

import util as u


# TODO move these to util?
def format_odor_conc(name, log10_conc):
    if log10_conc is None:
        return name
    else:
        # TODO tex formatting for exponent
        #return r'{} @ $10^{{'.format(name) + '{:.2f}}}$'.format(log10_conc)
        return '{} @ $10^{{{:.2f}}}$'.format(name, log10_conc)


def format_mixture(*args):
    log10_c1 = None
    log10_c2 = None
    if len(args) == 2:
        n1, n2 = args
    elif len(args) == 4:
        n1, n2, log10_c1, log10_c2 = args
    elif len(args) == 1:
        row = args[0]
        n1 = row['name1']
        n2 = row['name2']
        if 'log10_conc_vv1' in row:
            log10_c1 = row['log10_conc_vv1']
            log10_c2 = row['log10_conc_vv2']
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
    # TODO TODO assert that only one analysis for other index variables?
    # just do it before returning here?
    response_stats = u.latest_response_stats()
    response_stats = u.merge_odors(response_stats)
    response_stats = u.merge_recordings(response_stats)

    cols_to_hist = [
        #'exp_scale',
        'avg_dff_5s',
        'avg_zchange_5s'
    ]

    rec_cols = ['prep_date','fly_num','thorimage_id']
    rec_ts2triple = response_stats[['recording_from'] +
        rec_cols].drop_duplicates().set_index('recording_from')

    #
    index_cols = [ 
        'prep_date', 
        'fly_num', 
        'recording_from', 
        'analysis', 
        'comparison', 
        'odor1', 
        'odor2', 
        'repeat_num' 
    ]
    presentation_cols_to_get = index_cols + u.response_stat_cols
    db_presentations = pd.read_sql('presentations', u.conn, 
    	columns=(presentation_cols_to_get + ['presentation_id'])) 

    db_analysis_runs = pd.read_sql('analysis_runs', u.conn)
    #db_analysis_runs.set_index(['recording_from', 'run_at'], inplace=True)

    ar = set(db_analysis_runs.recording_from.unique())
    pr = set(db_presentations.recording_from.unique())
    # TODO why are there any differences here???
    print(ar - pr)
    print(pr - ar)

    db_presentations = u.merge_odors(db_presentations)

    sorted_presentation_analyses = sorted(db_presentations.analysis.unique())
    aps = [db_presentations[db_presentations.analysis == a]
        for a in sorted_presentation_analyses]
    shapes = [ap.shape for ap in aps]

    # TODO note how many times temp_all_repeats is true but all_repeats is false
    all_repeats = [u.have_all_repeats(ap, n_repeats=3) for ap in aps]
    temp_all_repeats = [u.have_all_repeats(ap) for ap in aps]
    full_comparisons = [u.have_full_comparisons(ap) for ap in aps]
    skipped_comparisons = [u.no_skipped_comparisons(ap) for ap in aps]

    stats_all_repeats = u.have_all_repeats(response_stats, n_repeats=3)
    stats_full_comps = u.have_full_comparisons(response_stats)
    stats_skipped_comps = u.no_skipped_comparisons(response_stats)

    # TODO TODO TODO what accounts for difference in len(aps[-1]) and
    # len(response_stats) (486 and 297, respectively, in one instance)
    # TODO could stats not be computed for some of them??
    # (what could cause that... should they still be inserted?)
    # TODO is this an issue of the referenced recording not showing up in the
    # analysis_runs? (maybe insertion into analysis_runs is failing?)

    stat_pids = set(response_stats.presentation_id.unique())
    last_ap = u.merge_recordings(aps[-1])
    last_ap_pids = set(last_ap.presentation_id.unique())

    print('presentation_ids not in response stats:')
    print(sorted(last_ap_pids - stat_pids))

    print('presentation_ids not in last presentation analysis:')
    print(sorted(stat_pids - last_ap_pids))

    # TODO maybe compare unique combinations of recording + presentation cols?
    pcols = [ 
        'prep_date', 
        'fly_num', 
        'thorimage_id', 
        'comparison'
    ]
    rs_nblocks = response_stats[pcols].groupby(pcols[:-1]).comparison.nunique()
    print(rs_nblocks)
    print(rs_nblocks.shape)

    last_ap_nblocks = last_ap[pcols].groupby(pcols[:-1]).comparison.nunique()
    print(last_ap_nblocks)
    print(last_ap_nblocks.shape)

    fly_df = u.mb_team_gsheet(use_cache=True)
    # TODO TODO TODO how come this has a bunch more than what was put in the db
    # in the last run??
    # (it's all natural_odors. populate_db should have not been restricted to 
    # stuff marked attempt_analysis...)
    print(fly_df[['date','fly_num','thorimage_dir']].drop_duplicates())


    # TODO TODO somewhere, check for all contiguous comparison #s
    # (shouldn't skip #s)

    ##import ipdb; ipdb.set_trace()
    #

    #
    # TODO so is stuff w/ only 2 repeats not in db now? shouldn't it fail it any
    # of those things are in there?
    missing_repeats = u.missing_repeats(response_stats, n_repeats=3)

    if len(missing_repeats) > 0:
        print(missing_repeats)

        no_missing_repeats = (set(response_stats.recording_from.unique()) -
            set(missing_repeats.recording_from.unique()))

        print('Recordings with all repeats for each comparison:')
        print(rec_ts2triple.loc[no_missing_repeats].to_string(index=False))

        print('\nRecordings with comparisons missing some repeats:')
        print(rec_ts2triple.loc[missing_repeats.recording_from.unique()
            ].to_string(index=False))
        print('')

        # TODO are those missing some missing everything? some other pattern?

        # TODO TODO are they in db (and just lost in latest_response_stats) or
        # missing in db? insertion failing for some reason? not happening?
        # TODO TODO if not in db, maybe debug by checking right after insertion
        # in whichever script?

        # TODO TODO just subset to each analysis run, and see which are missing
        # stuff?

        ##import ipdb; ipdb.set_trace()
        sys.exit()
    #

    odor_cols = [
        'name1',
        'name2'#,
        #'log10_conc_vv1',
        #'log10_conc_vv2'
    ]
    id_vars = [
        'prep_date', 
        'fly_num', 
        'thorimage_id', 
        'comparison', 
    ] + odor_cols + ['repeat_num']

    date_plot_margin = pd.Timedelta('7 days')
    date_plot_min = response_stats.prep_date.min() - date_plot_margin
    date_plot_max = response_stats.prep_date.max() + date_plot_margin

    odor_groups = response_stats.groupby(odor_cols)
    # TODO count # w/ len2 for sanity checking (should be equal to number of
    # comparisons and size of my odors panel) (?)
    print(len(odor_groups))
    for odors, odor_group in odor_groups:
        title = format_mixture(*odors)
        print(title)
        print('n_repeats={}'.format(len(odor_group)))

        assert u.have_all_repeats(odor_group, n_repeats=3)
        assert len(odor_group) >= 3 * len(odor_group.recording_from.unique())

        axes = odor_group[cols_to_hist].hist()
        fig = axes[0,0].figure
        fig.suptitle(title)

        # TODO plots response stats over days (to notice experimental drift)
        # TODO (one line plot per variable in cols_to_hist)
        # TODO like below, but:
        # - one axes per variable (facetgrid? otherwise use seaborn?)
        # - color points by fly per day
        # - more margin around start / end date
        # - maybe horizontal line derived from good/bad histogram?
        #odor_group.plot(x='prep_date', y=cols_to_hist, title=title, style='.')

        # TODO maybe just make a facetgrid per stat, and do across all odors
        # after loop or something?
        sns_odor_df = pd.melt(odor_group, id_vars=id_vars,
            value_vars=cols_to_hist, var_name='stat')
        g = sns.FacetGrid(sns_odor_df, col='stat', hue='fly_num', sharey=False,
            xlim=(date_plot_min, date_plot_max))
        g.map(sns.scatterplot, 'prep_date', 'value')
        # TODO not sure why i was unable to get these column titles on the y
        # axis... tried using set_ylabels and making stat row instead
        g.set_titles('{col_name}')
        g.set_ylabels('')
        g.add_legend()
        g.fig.suptitle(title)
        # TODO maybe also have fixed dates on axis (or at least the dates
        # actually used)
        # TODO before doing this, hovering seemed to show a date, now a float.
        # maintain old behavior somehow.
        # could try axes.format_coord (a fn). see:
        # stackoverflow.com/questions/46853259
        '''
        for ax in g.axes.ravel():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        '''
        # same problem as above (otherwise works tho)
        g.set_xticklabels(rotation=90)

        #plt.show()
        #import ipdb; ipdb.set_trace()

        # TODO plot histograms of (a subset of?) the stats conditional on
        # recording ultimately being accepted or not
        # (maybe green vs red overlayed or something?)

        # TODO only print / group if both True and False?
        good_bad_means = odor_group.groupby('accepted')[cols_to_hist +
            ['exp_scale', 'exp_tau']].mean()

        # TODO maybe print # accepted vs not, for context.
        # might just be a lot not labelled one way or the other?
        # TODO list what was accepted and what wasn't too
        print(good_bad_means)

        # TODO maybe also take repeat_num into account?

        print('')

    # TODO by others vars or just avg dff?
    by_activation_str = response_stats[response_stats.name2 == 'paraffin'
        ].groupby(odor_cols).avg_dff_5s.mean().sort_values(ascending=False)
    by_activation_str.name = 'mean_avg_dff_5s'
    print('Odors ranked by how strongly activating they are alone:')
    print(by_activation_str.reset_index()[['name1', 'mean_avg_dff_5s']
        ].to_string(index=False))

    #
    #plt.close('all')
    #

    # TODO TODO TODO maybe timeseries across odors, normalized to max across
    # days or something?
    # TODO first try normalizing w/in each existing plot to see what those look
    # like / if comparable
    sns_df = pd.melt(response_stats, id_vars=id_vars, value_vars=cols_to_hist,
        var_name='stat')
    sns_df['normed_stat'] = sns_df.groupby(odor_cols + ['stat']
        ).value.transform(lambda x: x / x.max())

    # TODO maybe kwarg flag to suppress concs even if we do have those columns?
    sns_df['odor_title'] = sns_df.apply(format_mixture, axis=1)

    g = sns.FacetGrid(sns_df, col='stat', hue='odor_title', sharey=False,
        xlim=(date_plot_min, date_plot_max))

    g.map_dataframe(sns.scatterplot, 'prep_date', 'normed_stat',
        style='fly_num')
    # make things kind of hard to read
    #g.map(plt.plot, 'prep_date', 'normed_stat')

    g.set_titles('{col_name}')
    g.set_ylabels('')
    g.add_legend()
    g.fig.suptitle('Full-frame stats for all odors')

    # TODO maybe look at exp fit params for stuff w/ avg dff change / zchange 
    # over good threshold? or from good experiments?

    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

