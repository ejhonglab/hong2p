#!/usr/bin/env python3

import os
from os.path import exists, join
import glob

import numpy as np
#from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import hong2p.util as u


trial_matrices = False
odor_matrices = False
fit_matrices = False
odor_and_fit_matrices = True


# Desired
'''
key = {
    'etb': 'A',
    'eta': 'B',
    'iaa': 'C',
    'iaol': 'D',
    'etoh': 'E',
    #
    '1o3ol': 'A',
    'fur': 'B',
    'va': 'C',
    'ms': 'D',
    '2h': 'E'
}
'''
key = {
    'etb': 'etb',
    'eta': 'eta',
    'iaa': 'iaa',
    'iaol': 'iaol',
    'etoh': 'etoh',
    #
    '1o3ol': '1o3ol',
    'fur': 'fur',
    'va': 'va',
    'ms': 'ms',
    '2h': '2h'
}

key_corrs = {
    '4': {
        'A': key['eta'],
        'B': key['etb'],
        'C': key['iaol'],
        'D': key['etoh'],
        'E': key['iaa'],
        'MIX': 'MIX'
    },
    '5': {
        'A': key['va'],
        'C': key['ms'],
        'D': key['fur'],
        'E': key['2h'],
        'F': key['1o3ol'],
        'MIX': 'MIX'
    },
    '7': {
        'A': key['fur'],
        'B': key['1o3ol'],
        'C': key['ms'],
        'D': key['va'],
        'F': key['2h'],
        'MIX': 'MIX'
    },
    '8': {
        'A': key['iaa'],
        'C': key['eta'],
        'D': key['etoh'],
        'E': key['etb'],
        'F': key['iaol'],
        'MIX': 'MIX'
    }
}
# TODO consolidate this w/ above somehow? (change keycorrs to A -> odor and
# figure this out from values in each of those, then regen keycorrs as-is now
# by looking up values)
expt2mix = {
    '4': 'kiwi',
    '5': 'control 1',
    '7': 'control 1',
    '8': 'kiwi'
}
mix2odor_order = {
    'kiwi': ['etb', 'eta', 'iaa', 'iaol', 'etoh'],
    'control 1': ['1o3ol', 'fur', 'va', 'ms', '2h']
}

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

    mix = odor_cell_stats.loc['MIX']
    weighted_mix_diff = mix - weighted_sum
    im2 = matshow(ax, weighted_mix_diff[ordered_cells], cmap='coolwarm',
        aspect='auto')

    # move to right if i want to keep the y ticks
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(['MIX - SUM'], fontsize=fontsize,
        rotation=xtickrotation)
    ax.tick_params(bottom=False)

    f3.colorbar(im2, cax=cax2)
    diff_cbar_label = r'$\frac{\Delta F}{F}$ difference'
    cax2.set_ylabel(diff_cbar_label)

    f3.savefig(join(fig_dir, 'png', 'odorandfit_' + fname + '.png'))
    f3.savefig(join(fig_dir, 'svg', 'odorandfit_' + fname + '.svg'))


cell_cols = ['name1','name2','repeat_num','cell']
response_calling_s = 5.0
dfs = []
for df_pickle in glob.glob('/mnt/nas/mb_team/analysis_output/20190815*.p'):
    df = pd.read_pickle(df_pickle)

    corrected = False
    for n, corr in key_corrs.items():
        if df_pickle.endswith('_00{}.p'.format(n)):
            df.name1 = df.name1.map(corr)
            assert not pd.isnull(df.name1).any()
            corrected = True
            prefix = expt2mix[n]
            break
    assert corrected
    
    odor_order = mix2odor_order[prefix] + ['MIX']
    title = '/'.join([x for x in df_pickle[:-2].split('_')[-4:] if len(x) > 0])
    fname = prefix.replace(' ','') + '_' + title.replace('/','_')
    title = prefix.title() + ': ' + title

    in_response_window = ((df.from_onset > 0.0) &
                          (df.from_onset <= response_calling_s))

    window_df = df.loc[in_response_window,
        cell_cols + ['order','from_onset','df_over_f']]
    window_by_trial = window_df.groupby(cell_cols + ['order'])['df_over_f']

    stat = 'max'
    if stat == 'max':
        window_trial_stats = window_by_trial.max()
    elif stat == 'mean':
        window_trial_stats = window_by_trial.mean()

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

    component_names = [x for x in odor_cell_stats.index if x != 'MIX'] 
    # TODO also do on traces as in kc_analysis?
    # or at least per-trial rather than per-mean?
    mix = odor_cell_stats.loc['MIX']

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

    #plt.show()
    #import sys; sys.exit()

plt.show()

#df = pd.concat(dfs, ignore_index=True)
#import ipdb; ipdb.set_trace()
