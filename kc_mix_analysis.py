#!/usr/bin/env python3

import os
from os.path import exists, join
import glob
from pprint import pprint as pp
import warnings
import time
import pickle

import numpy as np
#from scipy.optimize import minimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import chemutils as cu

import hong2p.util as u


trial_matrices = False
odor_matrices = False
fit_matrices = False
#odor_and_fit_matrices = True
odor_and_fit_matrices = False
# TODO maybe modify things s.t. avg traces already go under
# odor_and_fit_matrices and odor_matrices (w/ gridspec probably)
# + eag too
avg_traces = True

test = False
#test = True

fig_dir = 'mix_figs'
if not exists(fig_dir):
    os.mkdir(fig_dir)

png_dir = join(fig_dir, 'png')
if not exists(png_dir):
    os.mkdir(png_dir)

svg_dir = join(fig_dir, 'svg')
if not exists(svg_dir):
    os.mkdir(svg_dir)

# TODO maybe just use u.matshow? or does that do enough extra stuff that it
# would be hard to get it to just do what i want in this linearity-checking
# case?
# TODO factor this functionality into u.matshow if i end up using that
def matshow(ax, data, as_row=False, **kwargs):
    if len(data.shape) == 1:
        if as_row:
            ax_idx = 0
        else:
            ax_idx = -1
        data = np.expand_dims(data, -1)
    return ax.matshow(data, **kwargs)


def component_sum_error(weights, components, mix):
    component_sum = (weights * components.T).sum(axis=1)
    return np.linalg.norm(component_sum - mix)**2


def odor_and_fit_plot(odor_cell_stats, weighted_sum, ordered_cells, fname,
    title, odor_labels, cbar_label):

    f3, f3_axs = plt.subplots(1, 2, figsize=(10, 20), gridspec_kw={
        'wspace': 0,
        # assuming only output of imshow filled ax, this would seem to
        # be correct, but the column in the right axes seemed to small...
        #'width_ratios': [1, 1 / (len(odor_cell_stats.index.unique()) + 1)]
        # this might not be **exactly** right either, but pretty close
        'width_ratios': [1, 1 / len(odor_cell_stats.index.unique())]
    })
    cells_odors_and_fit = odor_cell_stats.loc[:, ordered_cells].T.copy()
    fit_name = 'WEIGHTED SUM'
    cells_odors_and_fit[fit_name] = weighted_sum[ordered_cells]
    labels = [x for x in odor_labels] + [fit_name]
    ax = f3_axs[0]

    xtickrotation = 'horizontal'
    fontsize = 8

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='3%', pad=0.5)

    im = u.matshow(cells_odors_and_fit, xticklabels=labels,
        xtickrotation=xtickrotation, fontsize=fontsize,
        title=title, ax=f3_axs[0])
    # Default is 'equal'
    ax.set_aspect('auto')

    f3.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    cax.set_ylabel(cbar_label)

    ax = f3_axs[1]
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes('right', size='20%', pad=0.08)

    mix = odor_cell_stats.loc['mix']
    weighted_mix_diff = mix - weighted_sum
    im2 = matshow(ax, weighted_mix_diff[ordered_cells], cmap='coolwarm',
        aspect='auto')

    # move to right if i want to keep the y ticks
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(['mix - sum'], fontsize=fontsize,
        rotation=xtickrotation)
    ax.tick_params(bottom=False)

    f3.colorbar(im2, cax=cax2)
    diff_cbar_label = r'$\frac{\Delta F}{F}$ difference'
    cax2.set_ylabel(diff_cbar_label)

    f3.savefig(join(fig_dir, 'png', 'odorandfit_' + fname + '.png'))
    f3.savefig(join(fig_dir, 'svg', 'odorandfit_' + fname + '.svg'))


# TODO add appropriately formatted descrip of mix to input index cols so it can
# be used in plotting
def plot_pca(df, fname=None):
    pca_unstandardized = True
    if pca_unstandardized:
        pca_2 = PCA(n_components=2)
        pca_data = pca_2.fit_transform(df)

        pca_data = pd.DataFrame(index=df.index, data=pca_data)
        pca_data.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)

        f1a = plt.figure()
        sns.scatterplot(data=pca_data.reset_index(), x='PC1', y='PC2',
            hue='name1', legend='full')
        #    hue='sample_type', style='fermented', size='day', legend='full')
        plt.title('PCA on raw data')

        pca_obj = PCA()
        pca_obj.fit(df)

        f1b = plt.figure()
        # TODO is explained_variance_ratio_ what i want exactly?
        plt.plot(pca_obj.explained_variance_ratio_[:15], 'k.')
        plt.title('PCA on raw data')
        plt.xlabel('Principal component')
        plt.ylabel('Fraction of explained variance')

        '''
        pc_df = pd.DataFrame(columns=df.columns,
                             data=pca_obj.components_)

        ###pc_df.rename(columns=fmt_chem_id, inplace=True)
        ###pc_df.columns.name = 'Chemical'
        for pc in range(2):
            print('Unstandardized PC{}:'.format(pc))
            # TODO check again that this is doing what i want
            # (factor to fn too)
            print(pc_df.iloc[pc].abs().sort_values(ascending=False)[:10])
        '''

    standardizer = StandardScaler()
    df_standardized = standardizer.fit_transform(df)

    pca_obj = PCA()
    pca_obj.fit(df_standardized)

    pca_2 = PCA(n_components=2)
    pca_data = pca_2.fit_transform(df_standardized)

    for n in range(pca_2.n_components):
        assert np.allclose(pca_obj.components_[n], pca_2.components_[n])

    # From Wikipedia page on PCA:
    # "If there are n observations with p variables, then the number of
    # distinct principal components is min(n - 1, p)."
    assert len(df.columns) == pca_2.n_features_
    assert len(df.index) == pca_2.n_samples_

    pca_data = pd.DataFrame(index=df.index, data=pca_data)
    pca_data.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)

    f2a = plt.figure()
    # TODO plot trajectories instead of using marker size to indicate time
    # point
    # TODO what about day? size? (saturation / alpha would be ideal i think)
    sns.scatterplot(data=pca_data.reset_index(), x='PC1', y='PC2',
        hue='name1', legend='full')
        #hue='sample_type', style='fermented', size='day')#, legend='full')

    plt.title('PCA on standardized data')
    # TODO save figure

    f2b = plt.figure()
    plt.plot(pca_obj.explained_variance_ratio_[:15], 'k.')
    plt.title('PCA on standardized data')
    plt.xlabel('Principal component')
    plt.ylabel('Fraction of explained variance')

    '''
    pc_df = pd.DataFrame(columns=df.columns, data=pca_obj.components_)
    ####pc_df.rename(columns=fmt_chem_id, inplace=True)
    ####pc_df.columns.name = 'Chemical'
    for pc in range(2):
        print('Standardized PC{}:'.format(pc))
        print(pc_df.iloc[pc].abs().sort_values(ascending=False)[:10])
    '''

    if fname is not None:
        f1a.savefig(join(fig_dir, 'png', 'pca_unstandardized_' + fname + '.png'))
        f1a.savefig(join(fig_dir, 'svg', 'pca_unstandardized_' + fname + '.svg'))
        f1b.savefig(join(fig_dir, 'png', 'skree_unstandardized_' + fname + '.png'))
        f1b.savefig(join(fig_dir, 'svg', 'skree_unstandardized_' + fname + '.svg'))
        f2a.savefig(join(fig_dir, 'png', 'pca_' + fname + '.png'))
        f2a.savefig(join(fig_dir, 'svg', 'pca_' + fname + '.svg'))
        f2b.savefig(join(fig_dir, 'png', 'skree_' + fname + '.png'))
        f2b.savefig(join(fig_dir, 'svg', 'skree_' + fname + '.svg'))


# TODO maybe try on random subsets of cells (realistic # that MBONs read out?)
# TODO TODO count mix as same class or not? both?
# how do shen et al. disentagle this again? their generalization etc figures?
# TODO TODO use metric other than AU(roc)C since imbalanced classes
# area under ~"precision-recall" curve?
def roc_analysis(window_trial_stats, reliable_responders, fname=None):
    # TODO maybe don't subset to "reliable" responders as shen et al do?
    # TODO maybe just two distributions, one for natural one control?
    odors = window_trial_stats.index.get_level_values('name1')
    df = window_trial_stats[odors != 'pfo'].reset_index()

    auc_dfs = []
    for segmentation in (True, False):
        for odor in df.name1.unique():
            if segmentation and (odor == 'mix' or odor == 'kiwi'):
                continue

            odor_reliable_responders = reliable_responders.loc[(odor,)]
            odor_reliable_cells = odor_reliable_responders[
                odor_reliable_responders].index

            cells = []
            aucs = []
            for gn, gdf in df[df.cell.isin(odor_reliable_cells)
                ].groupby('cell'):

                if segmentation:
                    gdf = gdf[gdf.name1 != 'kiwi']

                if segmentation:
                    labels = (gdf.name1 == 'mix') | (gdf.name1 == odor)
                else:
                    labels = gdf.name1 == odor

                scores = gdf.df_over_f
                fpr, tpr, thresholds = roc_curve(labels, scores)
                curr_auc = auc(fpr, tpr)
                cells.append(gn)
                aucs.append(curr_auc)

            auc_dfs.append(pd.DataFrame({
                'task': 'segmentation' if segmentation else 'discrimination',
                'odor': odor,
                'cell': cells,
                'auc': aucs
            }))

            fig = plt.figure()
            plt.hist(aucs, range=(0,1))
            plt.axvline(x=0.5, color='r')
            # TODO TODO TODO add stuff to identify recording to titles
            if segmentation:
                plt.title('AUC distribution, segmentation for {}'.format(odor))
            else:
                plt.title('AUC distribution, {} vs all discrimination'.format(
                    odor))

            if fname is not None:
                odor_fname = odor.replace(' ', '_')
                task_fname = 'seg' if segmentation else 'discrim'

                fig.savefig(join(fig_dir, 'png', fname + '_' + odor_fname + '_'
                    + task_fname + '.png'))

                fig.savefig(join(fig_dir, 'svg', fname + '_' + odor_fname + '_'
                    + task_fname + '.svg'))

    # TODO maybe also plot average distributions for segmentation and
    # discrimination, within this fly, but across all odors?
    # (if so, should odors be weighted equally, or should each cell (of which
    # some odors will have less b/c less reliable responders)?)

    #plt.show()
    # TODO TODO savefigs

    # TODO TODO also return auc values (+ n cells used in each case?) for
    # analysis w/ data from other flies
    auc_df = pd.concat(auc_dfs, ignore_index=True)
    return auc_df


pickle_dir = '/mnt/nas/mb_team/analysis_output/trace_pickles'
glob_strs = ['20190815*.p', '20190909*.p']
pickles = []
for gs in glob_strs:
    pickles.extend(glob.glob(join(pickle_dir, gs)))
# TODO TODO TODO if multiple analysis exist for any (date, fly, run)
# pick most recent timestamp prefix

if test:
    warnings.warn('Only reading one pickle for testing! ' +
        'Set test = False to analyze all.')
    pickles = [p for p in pickles if '08-27_9_fn_0001' in p]

# TODO maybe use for this everything to speed things up a bit?
# (not just plotting)
# this seemed to yield *intermittently* clean looking plots,
# even within a single experiment
#plotting_td = pd.Timedelta(0.5, unit='s')
# both of these still making spiky plots... i feel like something might be up
#plotting_td = pd.Timedelta(1.0, unit='s')
#plotting_td = pd.Timedelta(2.0, unit='s')

# TODO TODO TODO why does it seem that, almost no matter the averaging
# window, there is almost identical up-and-down noise for a large fraction of
# traces????
max_plotting_td = 1.0

cell_cols = ['name1','name2','repeat_num','cell']
response_calling_s = 5.0
responder_frac_dfs = []
auc_dfs = []
for df_pickle in pickles:
    # TODO also support case here pickle has a dict
    # (w/ this df behind a trace_df key & maybe other data, like PID, behind
    # something else?)
    df = pd.read_pickle(df_pickle)
    # TODO resolve whatever caused recording_from to get split into _x and _y
    # cols (seems to be same values though)

    assert 'original_name1' in df.columns
    df.name1 = df.original_name1.map(cu.odor2abbrev)

    # TODO delete
    '''
    if 'original_name1' in df.columns:
        df.name1 = df.original_name1.map(odor2abbrev)

    # TODO fix that data output s.t. it has original name and delete this and
    # other stuff back translating from case specific abbreviations
    else:
        assert '/20190815_' in df_pickle
        corrected = False
        for n, corr in key_corrs.items():
            if df_pickle.endswith('_00{}.p'.format(n)):
                df.name1 = df.name1.map(corr)
                assert not pd.isnull(df.name1).any()
                corrected = True
                break
        assert corrected
    '''
    prefix = u.df_to_odorset_name(df)
    odor_order = [cu.odor2abbrev(o) for o in u.df_to_odor_order(df)]
    # TODO delete this hack
    if 'mix' not in odor_order:
        odor_order = odor_order + ['mix']
    #
    
    title = '/'.join([x for x in df_pickle[:-2].split('_')[-4:] if len(x) > 0])
    fname = prefix.replace(' ','') + '_' + title.replace('/','_')
    title = prefix.title() + ': ' + title

    print(fname)

    # TODO maybe convert other handling of from_onset to timedeltas?
    # (also for ease of resampling / using other time based ops)
    #in_response_window = ((df.from_onset > 0.0) &
    #                      (df.from_onset <= response_calling_s))
    df.from_onset = pd.to_timedelta(df.from_onset, unit='s')
    in_response_window = ((df.from_onset > pd.Timedelta(0, unit='s')) &
        (df.from_onset <= pd.Timedelta(response_calling_s, unit='s')))

    window_df = df.loc[in_response_window,
        cell_cols + ['order','from_onset','df_over_f']]
    window_df.set_index(cell_cols + ['order','from_onset'], inplace=True)

    window_by_trial = window_df.groupby(cell_cols + ['order'])['df_over_f']

    # TODO use a longer/shorter baseline window?
    in_baseline_window = ((df.from_onset >= pd.Timedelta(-2, unit='s')) &
        (df.from_onset <= pd.Timedelta(0, unit='s')))

    stat = 'max'
    if stat == 'max':
        window_trial_stats = window_by_trial.max()
    elif stat == 'mean':
        window_trial_stats = window_by_trial.mean()

    baseline_df = df.loc[in_baseline_window,
        cell_cols + ['order','from_onset','df_over_f']]
    baseline_by_trial = baseline_df.groupby(cell_cols + ['order'])['df_over_f']

    '''
    # TODO could just groupby once and then pass multiple functions to
    # agg?
    baseline_stddev = baseline_by_trial.std()
    baseline_mean = baseline_by_trial.mean()
    response_criteria = pd.Series(index=window_df.index)
    # TODO maybe z change should be computed from raw f rather than dff?
    # just calc ratio over stddev with this metric?
    for gn, gdf in window_df.groupby(cell_cols + ['order']):
        zchange = ((gdf.df_over_f - baseline_mean.loc[gn]) /
            baseline_stddev.loc[gn])
        response_criteria.loc[gn] = zchange

    # TODO not sure why it seems i need such a high threshold here, to get
    # reasonable sparseness... was the fly i'm testing it with just super
    # responsive?
    mean_zchange_response_thresh = 4.0
    trial_responders = (response_criteria.groupby(cell_cols).agg('mean') >
        mean_zchange_response_thresh)
    '''
    # picked from histogram on one fly's data
    max_responder_thresh = 0.7
    trial_responders = window_by_trial.max() > max_responder_thresh
    # TODO exclude pfo for this and below? (probably for reported means)
    print(('Fraction of cells trials counted as responses (max dff > {}): '
        '{:.3f}').format(max_responder_thresh, 
        trial_responders.sum() / len(trial_responders)))

    # As >= 50% response to odor criteria in Shen paper
    reliable_responders = trial_responders.groupby(['name1','cell']).sum() >= 2
    print(('Mean fraction of cells responding to at least 2/3 trials: {:.3f}'
        ).format(reliable_responders.sum() / len(reliable_responders)))

    n_cells = len(df.cell.unique())
    print('Fraction of cells responding to each trial of each odor:')
    # TODO TODO is the population becoming silent to kiwi more quickly? is that
    # consistent?
    frac_odor_trial_responders = (trial_responders.groupby(['name1','repeat_num'
        ]).sum() / n_cells)
    print(frac_odor_trial_responders.to_string(float_format='%.3f'))

    mean_frac_odor_responders = \
        frac_odor_trial_responders.groupby('name1').mean()

    print('Mean fraction of cells responding to each odor:')
    print(mean_frac_odor_responders.to_string(float_format='%.3f'))
    # TODO TODO also concat these responder fraction things across flies
    # w/ appropriate index info -> make plots after loop over data

    # TODO also breakdown reliable by odor?

    # TODO TODO concat across recordings, w/ appropriate index info added
    auc_df = roc_analysis(window_trial_stats, reliable_responders, fname=fname)

    def get_single_index_val(var):
        values  = df[var].unique()
        assert len(values) == 1
        return values[0]

    def add_metadata(out_df):
        for var in ('prep_date', 'fly_num', 'thorimage_id'):
            out_df[var] = get_single_index_val(var)
        return out_df

    responder_frac_dfs.append(add_metadata(trial_responders))
    auc_dfs.append(add_metadata(auc_df))

    do_pca = True
    if do_pca:
        pivoted_window_trial_stats = pd.pivot_table(
            window_trial_stats.to_frame(name=stat), columns='cell',
            index=['name1','name2','repeat_num'], values=stat
        )
        # TODO TODO add stuff to identify recording to titles
        plot_pca(pivoted_window_trial_stats, fname=fname)

    responsiveness = window_trial_stats.groupby('cell').mean()
    cellssorted = responsiveness.sort_values(ascending=False)

    top_n = 100
    order = cellssorted.index

    trial_by_cell_stats = window_trial_stats.to_frame().pivot_table(
        index=['name1','name2','repeat_num','order'],
        columns='cell', values='df_over_f')

    # TODO maybe also add support for single letter abbrev case
    # (as was handled w/ commented sort stuff below)
    trial_by_cell_stats = trial_by_cell_stats.reindex(odor_order, level='name1')
    #trial_by_cell_stats.sort_index(level='name1', sort_remaining=False,
    #    inplace=True)

    if trial_matrices:
        trial_by_cell_stats_top = trial_by_cell_stats.loc[:, order[:top_n]]

        cbar_label = stat.title() + r' response $\frac{\Delta F}{F}$'

        odor_labels = u.matlabels(trial_by_cell_stats_top, u.format_mixture)
        # TODO fix x/y in this fn... seems T required
        f1 = u.matshow(trial_by_cell_stats_top.T, xticklabels=odor_labels,
            group_ticklabels=True, colorbar_label=cbar_label, fontsize=6,
            title=title)
        ax = plt.gca()
        ax.set_aspect(0.1)
        f1.savefig(join(fig_dir, 'png', 'trials_' + fname + '.png'))
        f1.savefig(join(fig_dir, 'svg', 'trials_' + fname + '.svg'))

    odor_cell_stats = trial_by_cell_stats.groupby('name1').mean()
    # TODO TODO factor linearity checking in kc_analysis to use this,
    # since A+B there is pretty much a subset of this case
    # (-> hong2p.util, both use that?)

    # TODO TODO check kiwi and pfo are excluded from fit
    component_names = [x for x in odor_cell_stats.index
        if x not in ('mix', 'kiwi', 'pfo')] 
    # TODO also do on traces as in kc_analysis?
    # or at least per-trial rather than per-mean?

    # TODO delete try/except
    try:
        mix = odor_cell_stats.loc['mix']
    except KeyError:
        import ipdb; ipdb.set_trace()

    components = odor_cell_stats.loc[component_names]
    component_sum = components.sum()
    assert mix.shape == component_sum.shape

    mix_norm = np.linalg.norm(mix)
    component_sum_norm = np.linalg.norm(component_sum)

    scaled_sum = (mix_norm / component_sum_norm) * component_sum
    scaled_sum_norm = np.linalg.norm(scaled_sum)
    assert np.isclose(scaled_sum_norm, mix_norm), '{} != {}'.format(
        scaled_sum_norm, mix_norm)

    scaled_sum_mix_diff = mix - scaled_sum
    # A: of dimensions (M, N)
    # B: of dimensions (M,)
    a = components.T
    b = mix
    try:
        # TODO worth also contraining coeffs to sum to 1 or something?
        # / be non-neg? and how?
        coeffs, residuals, rank, svs = np.linalg.lstsq(a, b, rcond=None)
        # TODO any meaning to svs? worth checking anything about that or rank?
    except np.linalg.LinAlgError as e:
        raise

    # TODO maybe print the coefficients (or include on plot?)?
    weighted_sum = (coeffs * a).sum(axis=1)
    weighted_mix_diff = mix - weighted_sum
    assert np.isclose(residuals[0], np.linalg.norm(mix - weighted_sum)**2)
    assert np.isclose(residuals[0],
        component_sum_error(coeffs, components, mix))

    # Just since we'd expect the model w/ more parameters to do better,
    # given it's actually optimizing what we want.
    assert (np.linalg.norm(weighted_mix_diff) <
            np.linalg.norm(scaled_sum_mix_diff)), 'lstsq did no better'

    # This was to check that lstsq was doing what I wanted (and it seems to be),
    # but it could also be used to introduce constraints.
    '''
    x0 = coeffs.copy()
    res0 = minimize(component_sum_error, x0, args=(components, mix))
    x0 = np.ones(components.shape[0]) / components.shape[0]
    res1 = minimize(component_sum_error, x0, args=(components, mix))
    '''

    if fit_matrices:
        diff_fig, diff_axs = plt.subplots(2, 2, sharex=True, sharey=True)
        ax = diff_axs[0, 0]

        #aspect_one_col = 0.05 #0.1
        aspect_one_col = 'auto'
        title_rotation = 0 #90
        # TODO delete after figuring out spacing
        titles = False
        #

        matshow(ax, scaled_sum[order[:top_n]], aspect=aspect_one_col)
        #ax.matshow(scaled_sum, vmin=vmin, vmax=vmax,
        #           extent=[xmin,xmax,ymin,ymax], aspect='auto')

        ax.set_xticks([])
        ax.set_yticks([])
        #
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #
        if titles:
            ax.set_title('Scaled sum of monomolecular responses',
                rotation=title_rotation)

        ax = diff_axs[0, 1]
        # TODO TODO appropriate vmin / vmax here?
        # [-1, 1] or maybe [-1.5, 1.5] would seem to work ok..?
        # TODO share a colorbar between the two difference plots?
        # (if fixed range, would seem reasonable)
        mat = matshow(ax, scaled_sum_mix_diff[order[:top_n]],
            aspect=aspect_one_col, cmap='coolwarm')
        #mat = matshow(ax, scaled_sum_mix_diff, extent=[xmin,xmax,ymin,ymax],
        #    aspect='auto', cmap='coolwarm')
        # TODO why this not seem to be working?
        diff_fig.colorbar(mat, ax=ax)

        if titles:
            ax.set_title('Mixture response - scaled sum',
                rotation=title_rotation)
        # TODO probably change from responder_traces... fn? use u.matshow?
        #ax.set_xlabel(responder_traces.columns.name)
        #ax.set_ylabel(responder_traces.index.name)
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('bottom')
        #
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #

        ax = diff_axs[1, 0]
        #matshow(ax, weighted_sum, vmin=vmin, vmax=vmax,
        #    extent=[xmin,xmax,ymin,ymax], aspect='auto')
        matshow(ax, weighted_sum[order[:top_n]], aspect=aspect_one_col)
        ax.set_xticks([])
        ax.set_yticks([])
        #
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #
        if titles:
            ax.set_title('Weighted sum', rotation=title_rotation)

        ax = diff_axs[1, 1]
        #mat2 = matshow(ax, weighted_mix_diff, extent=[xmin,xmax,ymin,ymax],
        #    aspect='auto', cmap='coolwarm')
        mat2 = matshow(ax, weighted_mix_diff[order[:top_n]],
            aspect=aspect_one_col, cmap='coolwarm')
        diff_fig.colorbar(mat2, ax=ax)

        if titles:
            ax.set_title('Mixture response - weighted sum',
                rotation=title_rotation)
        #ax.set_xlabel(responder_traces.columns.name)
        #ax.set_ylabel(responder_traces.index.name)
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('bottom')
        #
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #

        diff_fig.subplots_adjust(wspace=0)
        #diff_fig.tight_layout(rect=[0, 0, 1, 0.9])
        '''
        diff_fig_path = 
        diff_fig.savefig(diff_fig_path)
        '''

    cbar_label = 'Mean ' + stat + r' response $\frac{\Delta F}{F}$'
    odor_cell_stats_top = odor_cell_stats.loc[:, order[:top_n]]
    odor_labels = u.matlabels(odor_cell_stats_top, u.format_mixture)

    if odor_matrices:
        # TODO TODO modify u.matshow to take a fn (x/y)labelfn? to generate
        # str labels from row/col indices
        f2 = u.matshow(odor_cell_stats_top.T, xticklabels=odor_labels,
            colorbar_label=cbar_label, fontsize=6, title=title)
        ax = plt.gca()
        ax.set_aspect(0.1)
        f2.savefig(join(fig_dir, 'png', 'avg_' + fname + '.png'))
        f2.savefig(join(fig_dir, 'svg', 'avg_' + fname + '.svg'))

    if odor_and_fit_matrices:
        odor_and_fit_plot(odor_cell_stats, weighted_sum, order[:top_n], fname,
            title, odor_labels, cbar_label)

    # TODO TODO TODO one plot with avg_traces across flies, w/ hue maybe being
    # the fly
    if avg_traces:
        frame_delta = df.from_onset.diff().median().total_seconds()
        # To deal w/ extra samples sometimes getting picked up, going to
        # decrease the downsampling factor by an amount that won't equal
        # a new timedelta after multiplied by max frames per trial.
        max_n_frames = df.groupby(cell_cols).size().max()
        plot_ds_factor = (np.floor(max_plotting_td / frame_delta) - 
            frame_delta / (max_n_frames + 1))
        plotting_td = pd.Timedelta(plot_ds_factor * frame_delta, unit='s')
        print('median frame_delta:', frame_delta)
        print('plot_ds_factor:', plot_ds_factor)
        print('plotting_td:', plotting_td)
        '''

        # TODO could also using rolling window if just wanting it to look more
        # smooth, rather than actually have less points
        start = time.time()
        resampler = df.groupby(cell_cols)[['from_onset','df_over_f']
            ].resample(plotting_td, on='from_onset')
        downsampled_df = resampler.median().reset_index()
        print('pure downsampling took {:.3f}s'.format(time.time() - start))
        '''

        #start = time.time()
        #smoothed_df = df.copy()
        #smoothed_df = downsampled_df.copy()
        #smoothing_td = 2 * plotting_td

        '''
        smoothing_td = pd.Timedelta(2.0, unit='s')
        # no matter as_index+group_keys=False, there is still another level
        # added to index... this just how it works?
        smoothed_df_over_f = smoothed_df.groupby(cell_cols, sort=False,
            as_index=False, group_keys=False)[['from_onset','df_over_f']
            ].rolling(smoothing_td, on='from_onset').mean(
            ).df_over_f.reset_index(level=0, drop=True)
        '''
        """
        smoothing_iterations = 10
        smoothing_window_size = 10
        smoothing_td = pd.Timedelta(smoothing_window_size * frame_delta,
            unit='s')
        for _ in range(smoothing_iterations):
            smoothed_df.df_over_f = smoothed_df.groupby(cell_cols, sort=False,
                as_index=False, group_keys=False).df_over_f.rolling(
                smoothing_window_size).mean().reset_index(level=0, drop=True)

            smoothed_df.from_onset = smoothed_df.from_onset - smoothing_td / 2

        smoothed_df.from_onset = smoothed_df.from_onset.apply(
            lambda x: x.total_seconds())
        """
        avg_df = df.groupby(cell_cols[:-1] + ['from_onset']).mean(
            ).reset_index()

        smoothed_df = avg_df.copy()
        smoothing_window_size = 7
        smoothing_td = pd.Timedelta(smoothing_window_size * frame_delta,
            unit='s')
        smoothed_df.df_over_f = smoothed_df.groupby(cell_cols, sort=False,
            as_index=False, group_keys=False).df_over_f.rolling(
            smoothing_window_size).mean().reset_index(level=0, drop=True)

        smoothed_df.from_onset = smoothed_df.from_onset - smoothing_td / 2

        '''
        resampler = smoothed_df.groupby(cell_cols)[['from_onset','df_over_f']
            ].resample(plotting_td, on='from_onset')
        smoothed_df = resampler.mean().reset_index()
        '''

        '''
        smoothed_df_over_f = smoothed_df.groupby(cell_cols, sort=False,
            as_index=False, group_keys=False).df_over_f.apply(lambda ts:
            pd.Series(index=ts.index, data=u.smooth(ts, window_len=30))
            ).reset_index(level=0, drop=True)
        smoothed_df.df_over_f = smoothed_df_over_f 
        '''

        '''
        smoothed_downsampled_df = smoothed_df.groupby(cell_cols)[
            ['from_onset','df_over_f']].resample(plotting_td,
            on='from_onset').mean().reset_index()

        print('smoothing downsampling took {:.3f}s'.format(
            time.time() - start))
        '''
        '''
        start = time.time()
        g = sns.relplot(data=df, x='from_onset', y='df_over_f',
            col='name1', col_order=odor_order, kind='line', ci=None,
            color='black')
        print('plotting raw took {:.3f}s'.format(time.time() - start))
        '''

        # TODO maybe re-enable ci after downsampling
        g = sns.relplot(data=smoothed_df,
            x='from_onset', y='df_over_f',
            col='name1', col_order=odor_order, kind='line', ci=None,
            color='black', alpha=0.7)
        #, linewidth=0.5)
        #, hue='repeat_num')
        #print('plotting downsampled took {:.3f}s'.format(time.time() - start))
        g.set_titles('{col_name}')
        g.fig.suptitle('Average trace across all cells, ' + title[0].lower() +
            title[1:])
        g.axes[0,0].set_xlabel('Seconds from onset')
        g.axes[0,0].set_ylabel(r'Mean $\frac{\Delta F}{F}$')
        for a in g.axes.flat[1:]:
            a.axis('off')

        g.fig.subplots_adjust(top=0.9, left=0.05)
        g.fig.savefig(join(fig_dir, 'png', 'avg_traces_' + fname + '.png'))
        g.fig.savefig(join(fig_dir, 'svg', 'avg_traces_' + fname + '.svg'))

    # TODO TODO TODO plot pid too

    # TODO another flag for this part?
    for odor in odor_cell_stats.index:
        # TODO maybe put all sort orders in one plot as subplots?
        order = odor_cell_stats.loc[odor, :].sort_values(ascending=False).index
        sort_odor_labels = [o + ' (sorted)' if o == odor else o
            for o in odor_labels]
        ss = '_{}_sorted'.format(odor)

        # TODO TODO TODO here and in plots that also have fits, show (in
        # general, nearest?  est? scalar magnitude of one of these?) eag + in
        # roi / full frame MB fluorescence under each column?
        if odor_matrices:
            odor_cell_stats_top = odor_cell_stats.loc[:, order[:top_n]]
            fs = u.matshow(odor_cell_stats_top.T, xticklabels=sort_odor_labels,
                colorbar_label=cbar_label, fontsize=6, title=title)
            ax = plt.gca()
            ax.set_aspect(0.1)
            fs.savefig(join(fig_dir, 'png', 'avg_' + fname + ss + '.png'))
            fs.savefig(join(fig_dir, 'svg', 'avg_' + fname + ss + '.svg'))

        if odor_and_fit_matrices:
            odor_and_fit_plot(odor_cell_stats, weighted_sum, order[:top_n],
                fname + ss, title, sort_odor_labels, cbar_label)

    print('\n')

responder_frac_df = pd.concat(responder_frac_dfs, ignore_index=True)
auc_df = pd.concat(auc_dfs, ignore_index=True)

with open('kc_mix_analysis_outputs.p', 'wb') as f:
    data = {
        'responder_frac_df': responder_frac_df,
        'auc_df': auc_df
    }
    pickle.dump(data, f)

plt.show()
import ipdb; ipdb.set_trace()

