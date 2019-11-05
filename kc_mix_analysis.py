#!/usr/bin/env python3

import os
from os.path import exists, join, getmtime
import glob
from pprint import pprint as pp
import warnings
import time
import pickle

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import chemutils as cu

import hong2p.util as u


analyze_cached_outputs = True
#analyze_cached_outputs = False

# TODO should look up odor_set of this odor, then pass that to ordering fn,
# s.t. that odor_set is first
ref_odor = 'eb'
# This + 1o3ol~11.1 taken from 0.7 thresh on 2019-8-27/9 fly, which may not have
# been ideal.
ref_response_percent = 18.6

# TODO TODO implement checking against these / maybe something like these
# (normed to rate of ref odor) across all odors and flies automatically, to
# sanity check response calling across flies
'''
other_odors_to_check = {
    '1o3ol': 11.1
}
'''

# If True, the threshold is set within each fly, such that the response percent
# to the reference odor is the reference response percent.
# If False, responders are called with the fixed mean Z-scored dF/F (in response
# window) (mean_zchange_response_thresh) across all flies.
fix_ref_odor_response_fracs = True
mean_zchange_response_thresh = 3.0

responder_threshold_plots = True

# TODO rename to be more clear about what each of these mean / add comment
# clarifying
trial_matrices = False
odor_matrices = False

correlations = True
plot_correlations = True
trial_order_correlations = True
odor_order_correlations = True
# TODO return to trying to aggregate some type of this analysis across flies
# after doing corr + linearity stuff across flies (+ ROC!)
#do_pca = True
do_pca = False
do_roc = True
# TODO was one of these types of plots obselete? did odor_and_fit_matrices
# replace fit_matrices? (seems like fit_matrices may be obselete)
fit_matrices = False
odor_and_fit_matrices = True
# TODO maybe modify things s.t. avg traces are plotted under the matrix
# portion of the odor_and_fit_matrices and odor_matrices plots
# (w/ gridspec probably) + eag too
avg_traces = False

# How many of the most broadly reponsive cells to plot.
# In some contexts, this may instead take this many cells sorted by other
# criteria, like responsiveness to a particular odor.
top_n = 100

# If True, prints which types of plots are generated when they are saved.
verbose_savefig = True
# TODO disable the many open plots warning if this is True
# Otherwise, they are closed right after being saved.
show_plots_interactively = False
save_figs = True

print_mean_responder_frac = False
print_responder_frac_by_trial = False
print_reliable_responder_frac = False

# If True, only two input pickles are loaded (both from one fly),
# to test parts of the code faster.
test = False
#test = True
# Data whose trace pickles contain this substring are loaded.
#test_substr = '08-27_9'
test_substr = '10-04_1'
if test:
    show_plots_interactively = True
    save_figs = False

start_time_s = time.time()

fig_dir = 'mix_figs'
if not exists(fig_dir):
    os.mkdir(fig_dir)

# Which formats to save plots in.
plot_formats = ['png', 'pdf']
for pf in plot_formats:
    pf_dir = join(fig_dir, pf)
    if not exists(pf_dir):
        os.mkdir(pf_dir)


def get_single_index_val(df, var):
    values  = df[var].unique()
    assert len(values) == 1
    return values[0]


def add_metadata(df, out_df):
    keys_to_add = ['prep_date', 'fly_num', 'thorimage_id']
    vals_to_add = [get_single_index_val(df, k) for k in keys_to_add]

    # TODO i thought this would be the one line way to do it, but i
    # guess not... is there one?
    #out_df = pd.concat([out_df], names=keys_to_add,
    #    keys=vals_to_add)
    for k, v in zip(keys_to_add[::-1], vals_to_add[::-1]):
        out_df = pd.concat([out_df], names=[k], keys=[v])
    return out_df


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


# TODO maybe move into loop so it gets suffix (fname) as a closure?
plots_made_this_run = set()
# TODO change so can be called w/ 2 args when prefix would be None?
# (suffix always passed, right? change order make most sense?)
def savefigs(fig, prefix, suffix):
    if save_figs:
        if prefix is not None:
            assert not prefix.endswith('_')
            prefix = prefix + '_'
        else:
            prefix = ''

        if verbose_savefig:
            print(f'writing plots for {prefix + suffix}')

        for pf in plot_formats:
            plot_fname = join(fig_dir, pf, prefix + suffix + '.' + pf)
            fig.savefig(plot_fname)
            plots_made_this_run.add(plot_fname)

    if not show_plots_interactively:
        plt.close(fig)


def trace_pickle_odor_set(trace_pickle):
    """Returns the name of the odor_set presented in trace_pickle recording.

    Will be one of either 'kiwi' or 'control'.
    """
    # TODO maybe something faster than loading each pickle twice? maybe return
    # dataframes from the load in here, and just use those?
    # TODO TODO one fn that loads all pickles in correct order?
    # at <~1s per pickle now, prob not worth it.
    df = pd.read_pickle(trace_pickle)
    unique_odor_abbrevs = df.name1.unique()
    if 'eb' in unique_odor_abbrevs:
        return 'kiwi'
    elif '1o3ol' in unique_odor_abbrevs:
        return 'control'
    else:
        raise ValueError(f'{trace_pickle} had neither reference odor '
            'indicating odor_set (eb / 1o3ol) in name1 column')


def order_one_odor_set_before_other(trace_pickles, first_odor_set='kiwi'):
    """
    Returns re-ordered list of pickle filenames, with data from one odor_set
    before all data from the other odor_set.

    This is so the odor_set with the reference odor can be analyzed first,
    and that threshold applied to the other recording with the same fly.
    No other properties of order are guaranteed (data from the same fly won't
    necessarily be adjacent or not).
    """
    odor_set2fnames = dict()
    for tp in trace_pickles:
        odor_set = trace_pickle_odor_set(tp)
        if odor_set not in odor_set2fnames:
            odor_set2fnames[odor_set] = [tp]
        else:
            odor_set2fnames[odor_set].append(tp)

    assert len(odor_set2fnames) == 2, 'expected 2 odor_set names'
    assert first_odor_set in odor_set2fnames, \
        f'expected odor_set {first_odor_set}'

    second_odor_set = [os for os in odor_set2fnames.keys()
        if os != first_odor_set][0]

    # TODO maybe also order recordings within fly next to each other.
    # might make any print outputs easier to follow within fly,
    # which might be useful

    return odor_set2fnames[first_odor_set] + odor_set2fnames[second_odor_set]


# TODO TODO factor into util / use other places (maybe even in natural_odors)?
# TODO maybe rename from melt (if that's not the closest-functionality
# pandas fn)?
def melt_symmetric(symmetric_df, drop_constant_levels=True,
    suffixes=('_a', '_b'), name=None, keep_duplicate_values=True):
    """Takes a symmetric DataFrame to a tidy version with unique values.

    Symmetric means the row and columns indices are equal, and values should
    be a symmetric matrix.
    """
    # TODO flag "checks" or something and check matrix actually is symmetric,
    # in *values* (as well as index already checked below)

    assert symmetric_df.columns.equals(symmetric_df.index)
    symmetric_df = symmetric_df.copy()
    symmetric_df.dropna(how='all', axis=0, inplace=True)
    symmetric_df.dropna(how='all', axis=1, inplace=True)
    assert symmetric_df.notnull().all(axis=None), 'not tested w/ non-all NaN'

    # To de-clutter what would otherwise become a highly-nested index.
    if drop_constant_levels:
        # TODO may need to call index.remove_unused_levels() first, if using
        # levels here... (see docs of that remove fn)
        constant_levels = [n for n, levels in zip(symmetric_df.index.names,
            symmetric_df.index.levels) if len(levels) == 1]

        symmetric_df = symmetric_df.droplevel(constant_levels, axis=0)
        symmetric_df = symmetric_df.droplevel(constant_levels, axis=1)

    # TODO adapt to work in non-multiindex case too! (rename there?)
    symmetric_df.index.rename([n + suffixes[0] for n in
        symmetric_df.index.names], inplace=True
    )
    symmetric_df.columns.rename([n + suffixes[1] for n in
        symmetric_df.columns.names], inplace=True
    )

    # TODO maybe an option to interleave the new index names
    # (so it's like name1_a, name1_b, ... rather than *_a, *_b)
    # or would that not ever really be useful?

    if keep_duplicate_values:
        tidy = symmetric_df.stack(level=symmetric_df.columns.names)
        assert tidy.shape == (np.prod(symmetric_df.shape),)
    else:
        # From: https://stackoverflow.com/questions/34417685
        keep = np.triu(np.ones(symmetric_df.shape)).astype('bool')
        masked = symmetric_df.where(keep)
        n_nonnull = masked.notnull().sum().sum()
        # We already know both elements of shape are the same from equality
        # check on indices above.
        n = symmetric_df.shape[0]
        # TODO test
        assert n_nonnull == (n * (n - 1) / 2)

        # TODO make sure this also still works in non-multiindex case!
        tidy = masked.stack(level=masked.columns.names)
        assert tidy.shape == (n_nonnull,)

    tidy.name = name
    return tidy


# TODO update to include case where there are multiple time points for each
# cell (weights shape should stay the same, components and mix should probably
# get a new axis for time)?
def component_sum_error(weights, components, mix):
    """
    Args:
    weights (array-like): One dimensional, of length equal to number of single
        component odors in mix.

    components (array-like): Of shape (# single component odors, # cells). Each
        row is scaled by the corresponding element in weights before summing.

    mix (array-like): One dimensional mixture response, of length equal to
        # of cells.
    """
    component_sum = (weights * components.T).sum(axis=1)
    return np.linalg.norm(component_sum - mix)**2


def one_scale_model_err(scale, component_sum, mix):
    return np.linalg.norm(scale * component_sum - mix)**2


# TODO maybe delete this
def one_scale_one_offset_model_err(scale, offset, component_sum, mix):
    return np.linalg.norm(scale * component_sum + offset - mix)**2


def minimize_multiple_init(fn, initial_param_list, args, squeeze=True,
    allow_failures=True, **kwargs):
    """
    Finds params to minimize fn, checking outputs are consistent across
    choices of inital parameters.

    kwargs are passed to `np.allclose`.
    """
    # TODO options to randomly initialize stuff over certain ranges?
    # scipy / other libs already have solution for that?

    if 'rtol' not in kwargs:
        # With default of 1e-5, was getting some failures.
        kwargs['rtol'] = 1e-4

    optimized_params = None
    opt_param_list = []
    failure = False
    nonequiv_params = False
    for initial_params in initial_param_list:
        ret = minimize(fn, initial_params, args=args)
        if not ret.success:
            opt_param_list.append(None)
            failure = True
            continue

        opt_param_list.append(ret.x)
        if optimized_params is None:
            optimized_params = ret.x
        else:
            # Passing kwargs so we get whatever numpy's defaults are for
            # atol and rtol.
            if not np.allclose(optimized_params, ret.x, **kwargs):
                nonequiv_params = True

    if allow_failures and any([x is not None for x in opt_param_list]):
        failure = False

    if failure:
        raise RuntimeError('minimize call failed with initial params: '
            f'{initial_params}')

    if nonequiv_params:
        raise ValueError('different optimized params across initial conditions')

    # TODO maybe only o if len(initial_param_list) is undefined
    # (so shape preserved if (1,) shape elements passed in)
    if squeeze and len(optimized_params) == 1:
        optimized_params = optimized_params[0]

    return optimized_params


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

    savefigs(f3, 'odorandfit', fname)


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
        # TODO TODO factor the 15 into a kwarg or something
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
    # TODO TODO TODO if pca_obj from fit (above) already has .components_, then
    # why even call fit_transform???
    pca_data = pca_2.fit_transform(df_standardized)
    import ipdb; ipdb.set_trace()

    for n in range(pca_2.n_components):
        assert np.allclose(pca_obj.components_[n], pca_2.components_[n])

    # From Wikipedia page on PCA:
    # "If there are n observations with p variables, then the number of
    # distinct principal components is min(n - 1, p)."
    assert len(df.columns) == pca_2.n_features_
    assert len(df.index) == pca_2.n_samples_

    pca_data = pd.DataFrame(index=df.index, data=pca_data)
    # TODO TODO TODO is it correct to call the columns PC<N> here?
    # or is it actually # PCs == # odors x trials, and each (of # odor x trials)
    # rows is how much cell gets of that i-th across-odortrials-PCs?
    pca_data.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)

    f2a = plt.figure()
    # TODO plot trajectories instead of using marker size to indicate time
    # point
    # TODO what about day? size? (saturation / alpha would be ideal i think)
    sns.scatterplot(data=pca_data.reset_index(), x='PC1', y='PC2',
        hue='name1', legend='full')
        #hue='sample_type', style='fermented', size='day')#, legend='full')

    plt.title('PCA on standardized data')

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
        savefigs(f1a, 'pca_unstandardized', fname)
        savefigs(f1b, 'skree_unstandardized', fname)
        savefigs(f2a, 'pca', fname)
        savefigs(f2b, 'skree', fname)

    # TODO TODO TODO return PCA data somehow (4 things, 2 each std/nonstd?)
    # or components, explained variance, and fits (which are what exactly?),
    # for each?
    import ipdb; ipdb.set_trace()


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

                savefigs(fig, None, fname + '_' + odor_fname + '_' + task_fname)

    # TODO maybe also plot average distributions for segmentation and
    # discrimination, within this fly, but across all odors?
    # (if so, should odors be weighted equally, or should each cell (of which
    # some odors will have less b/c less reliable responders)?)

    # TODO also return auc values (+ n cells used in each case?) for
    # analysis w/ data from other flies (n cells in current return output?)
    auc_df = pd.concat(auc_dfs, ignore_index=True)
    return auc_df


cell_cols = ['name1','name2','repeat_num','cell']
within_recording_stim_cols = ['name1','name2','repeat_num','order']

pickle_outputs_dir = 'output_pickles'
if not exists(pickle_outputs_dir):
    os.mkdir(pickle_outputs_dir)

pickle_outputs_fstr = join(pickle_outputs_dir,
    'kc_mix_analysis_outputs_meanzthr{:.2f}{}.p')

if not analyze_cached_outputs:
    # TODO use a longer/shorter baseline window?
    baseline_start = -2
    baseline_end = 0
    response_start = 0
    response_calling_s = 5.0
    trial_stat = 'max'

    # TODO TODO update loop to use this df, and get index values directly from
    # there, rather than re-calculating them
    latest_pickles = u.latest_trace_pickles()
    pickles = list(latest_pickles.trace_pickle_path)
    if test:
        warnings.warn('Only reading two pickles for testing! '
            'Set test = False to analyze all.')
        pickles = [p for p in pickles if test_substr in p]

    # takes a little under 1s per pickle
    #b = time.time()
    pickles = order_one_odor_set_before_other(pickles)
    #print('determining pickle order took {:.2f}s'.format(time.time() - b))

    # TODO maybe use for this everything to speed things up a bit?
    # (not just plotting)
    # this seemed to yield *intermittently* clean looking plots,
    # even within a single experiment
    #plotting_td = pd.Timedelta(0.5, unit='s')
    # both of these still making spiky plots... i feel like something might be
    # up
    #plotting_td = pd.Timedelta(1.0, unit='s')
    #plotting_td = pd.Timedelta(2.0, unit='s')

    # TODO TODO TODO why does it seem that, almost no matter the averaging
    # window, there is almost identical up-and-down noise for a large fraction
    # of traces????
    max_plotting_td = 1.0

    responder_sers = []
    response_magnitude_sers = []

    if correlations:
        correlation_dfs_from_means = []
        correlation_dfs_from_maxes = []

    if fit_matrices or odor_and_fit_matrices:
        linearity_analysis = True
    else:
        linearity_analysis = False

    if linearity_analysis:
        linearity_sers = []
        linearity_odor_dfs = []
        linearity_cell_dfs = []

    # TODO TODO also aggregate some things from pca analysis?

    if do_roc:
        auc_dfs = []

    if fix_ref_odor_response_fracs:
        # (date, fly_num) -> threshold
        fly2response_threshold = dict()
    else:
        ref_odor = None
        ref_response_percent = None

    for df_pickle in pickles:
        # TODO also support case here pickle has a dict
        # (w/ this df behind a trace_df key & maybe other data, like PID, behind
        # something else?)
        df = pd.read_pickle(df_pickle)

        assert 'original_name1' in df.columns
        df.name1 = df.original_name1.map(cu.odor2abbrev)

        prefix = u.df_to_odorset_name(df)
        odor_order = [cu.odor2abbrev(o) for o in u.df_to_odor_order(df)]
        
        parts = df_pickle[:-2].split('_')[-4:]
        title = '/'.join([parts[0], parts[1], '_'.join(parts[2:])])
        fname = prefix.replace(' ','') + '_' + title.replace('/','_')
        title = prefix.title() + ': ' + title

        print(fname)

        # TODO maybe convert other handling of from_onset to timedeltas?
        # (also for ease of resampling / using other time based ops)
        #in_response_window = ((df.from_onset > 0.0) &
        #                      (df.from_onset <= response_calling_s))
        df.from_onset = pd.to_timedelta(df.from_onset, unit='s')
        in_response_window = (
            (df.from_onset > pd.Timedelta(response_start, unit='s')) &
            (df.from_onset <= pd.Timedelta(response_start + response_calling_s,
                unit='s'))
        )

        window_df = df.loc[in_response_window,
            cell_cols + ['order','from_onset','df_over_f']]
        window_df.set_index(cell_cols + ['order','from_onset'], inplace=True)

        in_baseline_window = (
            (df.from_onset >= pd.Timedelta(baseline_start, unit='s')) &
            (df.from_onset <= pd.Timedelta(baseline_end, unit='s')))

        baseline_df = df.loc[in_baseline_window,
            cell_cols + ['order','from_onset','df_over_f']]
        baseline_by_trial = baseline_df.groupby(cell_cols + ['order']
            )['df_over_f']

        baseline_stddev = baseline_by_trial.std()
        baseline_mean = baseline_by_trial.mean()

        # I checked this against my old (slower) method of calculating Z-score,
        # using a loop over a groupby (equal after sort_index on both).
        response_criteria = \
            (window_df.df_over_f - baseline_mean) / baseline_stddev

        scalar_response_criteria = \
            response_criteria.groupby(cell_cols).agg('mean')

        if fix_ref_odor_response_fracs:
            fly_nums = df.fly_num.unique()
            assert len(fly_nums) == 1
            fly_num = fly_nums[0]
            dates = df.prep_date.unique()
            assert len(dates) == 1
            date = dates[0]
            fly_key = (date, fly_num)

            if fly_key not in fly2response_threshold:
                mean_zchange_response_thresh = np.percentile(
                    scalar_response_criteria.loc[(ref_odor,)],
                    100 - ref_response_percent
                )
                fly2response_threshold[fly_key] = mean_zchange_response_thresh
                # TODO maybe print for all flies in one place at the end?
                print(('Calculated mean Z-scored response threshold, from '
                    'reference response percentage of {:.1f} to {}: '
                    '{:.2f}').format(ref_response_percent, ref_odor,
                    mean_zchange_response_thresh))
            else:
                unique_name1s = scalar_response_criteria.index.get_level_values(
                    'name1').unique()
                assert ref_odor not in unique_name1s
                mean_zchange_response_thresh = fly2response_threshold[fly_key]

        if responder_threshold_plots:
            zthreshes = np.linspace(0, 25.0, 40)
            resp_frac_over_zthreshes = np.empty_like(zthreshes) * np.nan
            mean_rc = response_criteria.groupby(cell_cols).agg('mean')
            # TODO for all odors? specific reference odors?) [-> pick threshold
            # from there?]
            for i, z_thr in enumerate(zthreshes):
                z_thr_trial_responders = mean_rc >= z_thr
                frac = \
                    z_thr_trial_responders.sum() / len(z_thr_trial_responders)

                resp_frac_over_zthreshes[i] = frac

            thr_fig, thr_ax = plt.subplots()
            thr_ax.plot(zthreshes, resp_frac_over_zthreshes)
            thr_ax.set_title(title + ', response threshold sensitivity')
            thr_ax.set_xlabel('Mean Z-scored response threshold')
            thr_ax.set_ylabel('Fraction of cells responding (across all odors)')

            thr_ax.axvline(x=mean_zchange_response_thresh, color='gray',
                linestyle='--', label='Response threshold')
            
            # TODO maybe pick thresh from some kind of max of derivative (to
            # find an elbow)?
            # TODO could also use on one / a few flies for tuning, then disable?
            savefigs(thr_fig, 'threshold_sensitivity', fname)

        # TODO not sure why it seems i need such a high threshold here, to get
        # reasonable sparseness... was the fly i'm testing it with just super
        # responsive?
        trial_responders = \
            scalar_response_criteria >= mean_zchange_response_thresh

        print(('Fraction of cells trials counted as responses (mean z-scored '
            'dff > {:.2f}): {:.2f}').format(mean_zchange_response_thresh, 
            trial_responders.sum() / len(trial_responders)))

        responder_sers.append(add_metadata(df, trial_responders))

        # TODO deal w/ case where there are only 2 repeats (shouldn't the number
        # be increased by fraction expected to respond if given a 3rd trial?)
        # ...or just get better data and ignore probably
        # As >= 50% response to odor criteria in Shen paper
        reliable_responders = \
            trial_responders.groupby(['name1','cell']).sum() >= 2

        if print_reliable_responder_frac:
            print(('Mean fraction of cells responding to at least 2/3 trials:'
                ' {:.3f}').format(
                reliable_responders.sum() / len(reliable_responders)))

        n_cells = len(df.cell.unique())
        frac_odor_trial_responders = trial_responders.groupby(
            ['name1','repeat_num']).sum() / n_cells

        # TODO TODO TODO just print all of these things below anyway
        # (in the loop over the outputs)

        if print_responder_frac_by_trial:
            print('Fraction of cells responding to each trial of each odor:')
            # TODO TODO TODO is the population becoming silent to kiwi more
            # quickly? is that consistent?

            # TODO maybe just set float output format once up top
            # (like i could w/ numpy in thing i was working w/ Han on)?
            print(frac_odor_trial_responders.to_string(float_format='%.3f'))

        if print_mean_responder_frac:
            mean_frac_odor_responders = \
                frac_odor_trial_responders.groupby('name1').mean()

            print('Mean fraction of cells responding to each odor:')
            print(mean_frac_odor_responders.to_string(float_format='%.3f'))

        # The shuffling with 'cell' is just so it is the last level in the
        # index.
        window_by_trial = window_df.groupby([c for c in cell_cols if c != 'cell'
            ] + ['order','cell'])['df_over_f']

        window_trial_stats = window_by_trial.agg(trial_stat)
        response_magnitude_sers.append(add_metadata(df, window_trial_stats))

        # TODO TODO TODO save correlation plots using both max and mean,
        # and compare them to see they look comparable.
        # maybe after doing that, settle on one, with a note that they looked
        # similar, and maybe leaving code to make the comparison.
        if correlations:
            # TODO maybe include a flag to check [+ plot] both or just use
            # contents of `trial_stat` variable
            # TODO maybe refactor this section to be a loop over the two stats,
            # if i'm not gonna soon switch to just using one stat
            #for corr_trial_stat in ('mean', 'max'):

            window_trial_means = window_by_trial.mean()
            trial_by_cell_means = window_trial_means.to_frame().pivot_table(
                index='cell', columns=within_recording_stim_cols,
                values='df_over_f'
            )
            # TODO check plots generated w/ missing odors handled in this fn
            # are equiv to figure outputs from gui
            trial_by_cell_means = u.add_missing_odor_cols(df,
                trial_by_cell_means)
            odor_corrs_from_means = trial_by_cell_means.corr()

            window_trial_maxes = window_by_trial.max()
            trial_by_cell_maxes = window_trial_maxes.to_frame().pivot_table(
                index='cell', columns=within_recording_stim_cols,
                values='df_over_f'
            )
            trial_by_cell_maxes = u.add_missing_odor_cols(df,
                trial_by_cell_maxes)
            odor_corrs_from_maxes = trial_by_cell_maxes.corr()

            if plot_correlations:
                title_suffix = '\n' + title
                # TODO TODO and are there other plots / outputs that will be
                # affected by missing odors?
                if trial_order_correlations:
                    porder_corr_mean_fig = u.plot_odor_corrs(
                        odor_corrs_from_means, title_suffix=title_suffix
                    )
                    savefigs(porder_corr_mean_fig, 'porder_corr_mean', fname)

                    porder_corr_max_fig = u.plot_odor_corrs(
                        odor_corrs_from_maxes, trial_stat='max',
                        title_suffix=title_suffix
                    )
                    savefigs(porder_corr_max_fig, 'porder_corr_max', fname)

                if odor_order_correlations:
                    oorder_corr_mean_fig = u.plot_odor_corrs(
                        odor_corrs_from_means, odors_in_order=odor_order,
                        title_suffix=title_suffix
                    )
                    savefigs(oorder_corr_mean_fig, 'oorder_corr_mean', fname)

                    oorder_corr_max_fig = u.plot_odor_corrs(
                        odor_corrs_from_maxes, trial_stat='max',
                        odors_in_order=odor_order,
                        title_suffix=title_suffix
                    )
                    savefigs(oorder_corr_max_fig, 'oorder_corr_max', fname)

            tidy_corrs_from_means = melt_symmetric(odor_corrs_from_means,
                name='corr')
            # TODO maybe rename to indicate they are series not dataframes
            correlation_dfs_from_means.append(
                add_metadata(df, tidy_corrs_from_means)
            )

            tidy_corrs_from_maxes = melt_symmetric(odor_corrs_from_maxes,
                name='corr')
            correlation_dfs_from_maxes.append(
                add_metadata(df, tidy_corrs_from_maxes)
            )

        if do_roc:
            auc_df = roc_analysis(window_trial_stats, reliable_responders,
                fname=fname
            )
            auc_dfs.append(add_metadata(df, auc_df))

        # TODO TODO would it make more sense to do some kind of PCA across
        # flies? ideally in some way that weights flies w/ diff #s of cells
        # similarly?? or just cell anyway? something other than PCA at that
        # point?
        if do_pca:
            # TODO check that changing index to this, from
            # ['name1','name2','repeat_num'] (only diff is the 'order' col at
            # end) didn't screw up pca stuff
            pivoted_window_trial_stats = pd.pivot_table(
                window_trial_stats.to_frame(name=trial_stat), columns='cell',
                index=within_recording_stim_cols, values=trial_stat
            )
            # TODO TODO add stuff to identify recording to titles (still
            # relevant?)
            plot_pca(pivoted_window_trial_stats, fname=fname)

        responsiveness = window_trial_stats.groupby('cell').mean()
        cellssorted = responsiveness.sort_values(ascending=False)

        order = cellssorted.index

        trial_by_cell_stats = window_trial_stats.to_frame().pivot_table(
            index=within_recording_stim_cols,
            columns='cell', values='df_over_f'
        )

        # TODO maybe also add support for single letter abbrev case?
        trial_by_cell_stats = \
            trial_by_cell_stats.reindex(odor_order, level='name1')

        if trial_matrices:
            trial_by_cell_stats_top = trial_by_cell_stats.loc[:, order[:top_n]]

            cbar_label = trial_stat.title() + r' response $\frac{\Delta F}{F}$'

            odor_labels = u.matlabels(trial_by_cell_stats_top, u.format_mixture)
            # TODO fix x/y in this fn... seems T required
            f1 = u.matshow(trial_by_cell_stats_top.T, xticklabels=odor_labels,
                group_ticklabels=True, colorbar_label=cbar_label, fontsize=6,
                title=title)
            ax = plt.gca()
            ax.set_aspect(0.1)
            savefigs(f1, 'trials', fname)

        odor_cell_stats = trial_by_cell_stats.groupby('name1').mean()

        if linearity_analysis:
            # TODO TODO factor linearity checking in kc_analysis to use this,
            # since A+B there is pretty much a subset of this case
            # (-> hong2p.util, both use that?)

            component_names = [x for x in odor_cell_stats.index
                if x not in ('mix', 'kiwi', 'pfo')] 
            # TODO TODO also do on traces as in kc_analysis?
            # or at least per-trial(?) rather than per-mean?

            mix = odor_cell_stats.loc['mix']
            components = odor_cell_stats.loc[component_names]
            component_sum = components.sum()
            assert mix.shape == component_sum.shape

            mix_norm = np.linalg.norm(mix)
            component_sum_norm = np.linalg.norm(component_sum)

            # So mix_norm / component_sum_norm, both:
            # 1) IS the right scale to give the vectors the same norm, and
            # 2) Is NOT the scale that minimizes the difference between the
            #    scaled sum (of component responses) and the mix response.

            # TODO delete? any merit to this scale?
            '''
            scaled_sum = (mix_norm / component_sum_norm) * component_sum
            scaled_sum_norm = np.linalg.norm(scaled_sum)
            assert np.isclose(scaled_sum_norm, mix_norm), '{} != {}'.format(
                scaled_sum_norm, mix_norm)
            '''
            # TODO TODO some simple formula to find the best global scale?
            # it feels like there should be...
            opt_scale = minimize_multiple_init(one_scale_model_err,
                [mix_norm / component_sum_norm, 1, 0.3], (component_sum, mix)
            )
            scaled_sum = opt_scale * component_sum

            scaled_mix_diff = mix - scaled_sum
            scaled_sum_residual = np.linalg.norm(scaled_mix_diff)**2

            epsilon = 0.1
            r_plus = np.linalg.norm(mix - scaled_sum * (1 + epsilon))**2
            assert scaled_sum_residual < r_plus, \
                'scaled sum fit worse than larger scale'

            r_minus = np.linalg.norm(mix - scaled_sum * (1 - epsilon))**2
            assert scaled_sum_residual < r_minus, \
                'scaled sum fit worse than smaller scale'

            # TODO in addition to / in place of scaled_sum, should i also test a
            # model with a scale and an offset parameter? as:
            # https://math.stackexchange.com/questions/2050607
            # the subtracting An from A1 stuff makes me think the answer to
            # that question is wrong though...

            # A: of dimensions (M, N)
            # B: of dimensions (M,)
            a = components.T
            b = mix
            try:
                # TODO worth also contraining coeffs to sum to 1 or something?
                # / be non-neg? and how?
                coeffs, residuals, rank, svs = np.linalg.lstsq(a, b, rcond=None)
                # TODO any meaning to svs? worth checking anything about that or
                # rank?
            except np.linalg.LinAlgError as e:
                raise

            # TODO maybe print the coefficients (or include on plot?)?
            weighted_sum = (coeffs * a).sum(axis=1)
            weighted_mix_diff = mix - weighted_sum

            assert residuals.shape == (1,)
            residual = residuals[0]
            assert np.isclose(residual, np.linalg.norm(weighted_mix_diff)**2)
            assert np.isclose(residual,
                component_sum_error(coeffs, components, mix)
            )

            # Just since we'd expect the model w/ more parameters to do better,
            # assuming that it's actually optimizing what we want.
            assert residual < scaled_sum_residual, 'lstsq did no better'

            # This was to check that lstsq was doing what I wanted (and it seems
            # to be), but it could also be used to introduce constraints.
            '''
            x0s = [
                # TODO copy necessary? shouldn't be, right?
                coeffs.copy(),
                np.ones(components.shape[0]) / components.shape[0]
            ]
            opt_coeffs = minimize_multiple_init(component_sum_error, x0s,
                (components, mix)
            )
            assert np.allclose(opt_coeffs, coeffs, rtol=1e-4)
            '''

            # TODO TODO ~"scale factor" of fit model??? (or is it something
            # either not interesting or derivable from other things in this
            # df?)
            linearity_ser = pd.Series({
                'residual': residual,
                'rank': rank,
                'opt_single_scale': opt_scale,
                'residual_single_scale': scaled_sum_residual,
            })
            linearity_odor_df = pd.DataFrame(index=components.index, data={
                'component_weights': coeffs,
                'singular_values': svs
            })
            linearity_cell_df = pd.DataFrame({
                'mix_response': mix,
                'weighted_sum': weighted_sum,
                'weighted_mix_diff': weighted_mix_diff,
                'scaled_sum': scaled_sum,
                'scaled_mix_diff': scaled_mix_diff
            })
            linearity_sers.append(add_metadata(df, linearity_ser))
            linearity_odor_dfs.append(add_metadata(df, linearity_odor_df))
            linearity_cell_dfs.append(add_metadata(df, linearity_cell_df))

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
            mat = matshow(ax, scaled_mix_diff[order[:top_n]],
                aspect=aspect_one_col, cmap='coolwarm')
            #mat = matshow(ax, scaled_mix_diff,
            #    extent=[xmin,xmax,ymin,ymax], aspect='auto', cmap='coolwarm')
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

        cbar_label = 'Mean ' + trial_stat + r' response $\frac{\Delta F}{F}$'
        odor_cell_stats_top = odor_cell_stats.loc[:, order[:top_n]]
        odor_labels = u.matlabels(odor_cell_stats_top, u.format_mixture)

        if odor_matrices:
            # TODO TODO modify u.matshow to take a fn (x/y)labelfn? to generate
            # str labels from row/col indices
            f2 = u.matshow(odor_cell_stats_top.T, xticklabels=odor_labels,
                colorbar_label=cbar_label, fontsize=6, title=title
            )
            ax = plt.gca()
            ax.set_aspect(0.1)
            savefigs(f2, 'avg', fname)

        if odor_and_fit_matrices:
            odor_and_fit_plot(odor_cell_stats, weighted_sum, order[:top_n],
                fname, title, odor_labels, cbar_label
            )

        # TODO one plot with avg_traces across flies, w/ hue maybe being the
        # fly?
        # TODO (so should i aggregate mean traces across flies, then?)
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

            # TODO could also using rolling window if just wanting it to look
            # more smooth, rather than actually have less points
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
            '''
            smoothing_iterations = 10
            smoothing_window_size = 10
            smoothing_td = pd.Timedelta(smoothing_window_size * frame_delta,
                unit='s')
            for _ in range(smoothing_iterations):
                smoothed_df.df_over_f = smoothed_df.groupby(cell_cols,
                    sort=False, as_index=False, group_keys=False
                    ).df_over_f.rolling(smoothing_window_size).mean(
                    ).reset_index(level=0, drop=True)

                smoothed_df.from_onset = \
                    smoothed_df.from_onset - smoothing_td / 2

            smoothed_df.from_onset = smoothed_df.from_onset.apply(
                lambda x: x.total_seconds())
            '''
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
            resampler = smoothed_df.groupby(cell_cols)[['from_onset',
                'df_over_f']].resample(plotting_td, on='from_onset')

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
            #print('plotting downsampled took {:.3f}s'.format(
            #    time.time() - start))
            g.set_titles('{col_name}')
            g.fig.suptitle('Average trace across all cells, ' +
                title[0].lower() + title[1:])

            g.axes[0,0].set_xlabel('Seconds from onset')
            g.axes[0,0].set_ylabel(r'Mean $\frac{\Delta F}{F}$')
            for a in g.axes.flat[1:]:
                a.axis('off')

            g.fig.subplots_adjust(top=0.9, left=0.05)
            savefigs(g.fig, 'avg_traces', fname)

        # TODO TODO TODO plot pid too

        for odor in odor_cell_stats.index:
            # TODO maybe put all sort orders in one plot as subplots?
            order = \
                odor_cell_stats.loc[odor, :].sort_values(ascending=False).index

            sort_odor_labels = [o + ' (sorted)' if o == odor else o
                for o in odor_labels]
            ss = '_{}_sorted'.format(odor)

            # TODO TODO TODO here and in plots that also have fits, show (in
            # general, nearest? est? scalar magnitude of one of these?) eag +
            # in roi / full frame MB fluorescence under each column?
            if odor_matrices:
                odor_cell_stats_top = odor_cell_stats.loc[:, order[:top_n]]
                fs = u.matshow(odor_cell_stats_top.T,
                    xticklabels=sort_odor_labels, colorbar_label=cbar_label,
                    fontsize=6, title=title)
                ax = plt.gca()
                ax.set_aspect(0.1)
                savefigs(fs, 'avg', fname + ss)

            if odor_and_fit_matrices:
                odor_and_fit_plot(odor_cell_stats, weighted_sum, order[:top_n],
                    fname + ss, title, sort_odor_labels, cbar_label)

        print('')

    responders = pd.concat(responder_sers)
    response_magnitudes = pd.concat(response_magnitude_sers)
    response_magnitudes.name = f'trial{trial_stat}_{response_magnitudes.name}'

    # TODO TODO some way to cut down on boilerplate w/ lists before loop,
    # flags in loop, and conditional concatenating after loop?
    # (so adding new analysis types doesn't require that)
    # TODO i mean should i maybe just put everything in one big dataframe in the
    # loop?
    # TODO maybe a context manager where i concat everything, s.t. it
    # automatically saves any variables defined there? too hacky?
    # someone have a library implementing something like that?

    if correlations:
        corr_df_from_means = pd.concat(correlation_dfs_from_means)
        corr_df_from_maxes = pd.concat(correlation_dfs_from_maxes)
    else:
        corr_df_from_means = None
        corr_df_from_maxes = None

    if do_roc:
        auc_df = pd.concat(auc_dfs)
    else:
        auc_df = None

    if linearity_analysis:
        linearity_df = pd.concat(linearity_sers).unstack()
        linearity_odor_df = pd.concat(linearity_odor_dfs)
        linearity_cell_df = pd.concat(linearity_cell_dfs)
    else:
        linearity_df = None
        linearity_odor_df = None
        linearity_cell_df = None

    # TODO symlink kc_mix_analysis_output.p to most recent? or just load most
    # recent by default?
    # Including threshold in name, so that i can try responder based analysis on
    # output calculated w/ multiple thresholds, w/o having to wait for slow
    # re-calculation.
    if fix_ref_odor_response_fracs:
        pickle_outputs_name = pickle_outputs_fstr.format(
            ref_response_percent, ref_odor)
        # Because although it will have a value, it changes across flies in this
        # cases. This is to avoid confusion in things using these outputs.
        mean_zchange_response_thresh = None
    else:
        pickle_outputs_name = pickle_outputs_fstr.format(
            mean_zchange_response_thresh, '')

    print(f'Writing computed outputs to {pickle_outputs_name}')
    with open(pickle_outputs_name, 'wb') as f:
        data = {
            'fix_ref_odor_response_fracs': fix_ref_odor_response_fracs,

            'ref_odor': ref_odor,
            'ref_response_percent': ref_response_percent,

            'mean_zchange_response_thresh': mean_zchange_response_thresh,

            'baseline_start': baseline_start,
            'baseline_end': baseline_end,
            'response_start': response_start,
            'response_calling_s': response_calling_s,
            'trial_stat': trial_stat,

            'responders': responders,
            'response_magnitudes': response_magnitudes,

            'corr_df_from_means': corr_df_from_means,
            'corr_df_from_maxes': corr_df_from_maxes,

            'auc_df': auc_df,

            'linearity_df': linearity_df,
            'linearity_odor_df': linearity_odor_df,
            'linearity_cell_df': linearity_cell_df
        }
        pickle.dump(data, f)

    after_raw_calc_time_s = time.time()
    print('Calculations on raw data took {:.0f}s'.format(
        after_raw_calc_time_s - start_time_s))

else:
    load_most_recent = True
    if load_most_recent:
        out_pickles = glob.glob(pickle_outputs_fstr.replace('{:.2f}{}','*'))
        out_pickles.sort(key=getmtime)
        pickle_outputs_name = out_pickles[-1]
    else:
        # Ref odor frac if using desired_ref_odor.
        desired_thr = 2.5
        # '' if loading outputs where fix_ref_odor_response_fracs was False.
        desired_ref_odor = ''
        pickle_outputs_name = pickle_outputs_fstr.format(desired_thr,
            desired_ref_odor)

    print(f'Loading computed outputs from {pickle_outputs_name}')
    with open(pickle_outputs_name, 'rb') as f:
        data = pickle.load(f)

    # Modifying globals rather than locals, because at least globals
    # would still (sort-of) work if this got refactored into a function,
    # and the docs explicitly warn not to modify what locals() returns.
    # See: https://stackoverflow.com/questions/2597278 for discussion about
    # alternatives.
    # Doing this so I don't have to repeat which variables specified in pickle
    # saving above / so variables loaded can't diverge (assuming pickle was
    # generated with most recent version...).
    globals().update(data)

# TODO TODO TODO some kind of plot of the correlations themselves, to make a
# statement across flies?

fly_keys = ['prep_date', 'fly_num']
rec_key = 'thorimage_id'
n_flies = len(responders.index.to_frame()[fly_keys].drop_duplicates())
fly_colors = sns.color_palette('hls', n_flies)
# could also try ':' or '-.'
odorset2linestyle = {'kiwi': '-', 'control': '--'}

n_ro_hist_fig, n_ro_hist_ax = plt.subplots()
# Only the last bin includes both ends. All other bins only include the
# value at their left edge. So with n_odors + 2, the last bin will be
# [n_odors, n_odors + 1] (since arange doesn't include end), and will thus
# only count cells that respond (reliably) to all odors, since it is not
# possible for them to respond to > n_odors.
n_odors = 6
n_ro_bins = np.arange(n_odors + 2)

rec_keys2odor_set = dict()

frac_responder_dfs = []
for i, (fly_gn, fly_gser) in enumerate(responders.groupby(fly_keys)):
    fly_color = fly_colors[i]

    assert len(fly_gser.index.get_level_values(rec_key).unique()) == 2
    # TODO TODO TODO just add variable for odorset earlier, so that i can group
    # on that instead of (/in addition to) thorimage id, so i can loop over them
    # in a fixed order
    for rec_gn, rec_gser in fly_gser.groupby(rec_key):
        # TODO maybe refactor this (kinda duped w/ odorset finding from the more
        # raw dfs)?
        if 'eb' in rec_gser.index.get_level_values('name1'):
            odorset = 'kiwi'
        else:
            odorset = 'control'
        rec_keys2odor_set[fly_gn + (rec_gn,)] = odorset
        linestyle = odorset2linestyle[odorset]

        label = (odorset + ', ' +
            fly_gn[0].strftime(u.date_fmt_str) + '/' + str(fly_gn[1]))
        # TODO include thorimage id anyway?
        fname = label.replace(',','').replace(' ','_').replace('/','_')

        trial_responders = rec_gser
        # TODO probably delete this. i want to be able to plot pfo + kiwi
        # response fractions / reliable response fraction in one set of plots.
        # at least save as a redefinition for below.
        #trial_responders = rec_gser[~ rec_gser.index.get_level_values('name1'
        #    ).isin(('pfo', 'kiwi'))]

        n_rec_cells = \
            len(trial_responders.index.get_level_values('cell').unique())

        # TODO TODO dedupe this section with above (just do here?)
        ########################################################################
        # TODO deal w/ case where there are only 2 repeats (shouldn't the number
        # be increased by fraction expected to respond if given a 3rd trial?)
        # ...or just get better data and ignore probably
        # As >= 50% response to odor criteria in Shen paper
        reliable_responders = \
            trial_responders.groupby(['name1','cell']).sum() >= 2

        frac_reliable_responders = \
            reliable_responders.groupby('name1').sum() / n_rec_cells

        # TODO TODO try to consolidate this w/ odor_order (they should have
        # the same stuff, just one also has order info, right?)
        odors = [x for x in
            trial_responders.index.get_level_values('name1').unique()]

        full_odor_order = [cu.odor2abbrev(o) for o in u.odor_set2order[odorset]]
        seen_odors = set()
        odor_order = []
        for o in full_odor_order:
            if o not in seen_odors and o in odors:
                seen_odors.add(o)
                odor_order.append(o)
        del seen_odors

        # TODO TODO TODO TODO check that we are not getting wrong results by
        # dividing by 3 anywhere when the odor is actually only recorded twice
        # (count unique repeat_num?) (check above too!) (i think i might have
        # handled it correctly here...)
        n_odor_repeats = trial_responders.reset_index().groupby('name1'
            ).repeat_num.nunique()

        frac_responders = (trial_responders.groupby('name1').sum() /
            (n_odor_repeats * n_rec_cells))[odor_order]

        resp_frac_fig = plt.figure()
        resp_frac_ax = frac_responders.plot.bar(color='black')
        resp_frac_ax.set_title(label.title() +
            '\nAverage fraction responding by odor')
        # TODO hide name1 xlabel before saveing each of these bar plots / 
        # change it to "Odor" or something
        savefigs(resp_frac_fig, 'resp_frac', fname)

        '''
        reliable_frac_fig = plt.figure()
        reliable_frac_ax = frac_reliable_responders[odor_order].plot.bar(
            color='black')
        reliable_frac_ax.set_title(label.title() +
            '\nReliable responder fraction by odor')
        savefigs(reliable_frac_fig, 'reliable_frac', fname)
        '''

        reliable_of_resp_fig = plt.figure()
        reliable_of_resp_ax = (frac_reliable_responders / frac_responders
            )[odor_order].plot.bar(color='black')
        reliable_of_resp_ax.set_title(label.title() +
            '\nFraction of responders that are reliable, by odor')
        savefigs(reliable_of_resp_fig, 'reliable_of_resp', fname)

        frac_responders = add_metadata(rec_gser.reset_index(), frac_responders)
        frac_responders = pd.concat([frac_responders],
            names=['odor_set'], keys=[odorset]
        )
        frac_responder_dfs.append(frac_responders)

        odors = [o for o in odors if o not in ('pfo', 'kiwi')]
        odor_order = [o for o in odor_order if o not in ('pfo', 'kiwi')]

        odor_resp_subset_fracs_list = []
        for odor in odors:
            oresp = reliable_responders.loc[odor]
            cells = oresp[oresp].index.get_level_values('cell').unique()

            n_odor_cells = len(cells)

            #other_odors = [o for o in odors if o != odor]
            other_odors = list(odors)

            '''
            # TODO or maybe use reliable responders here too?
            # (if not, there might actually be more cells that respond (on
            # average) to another odor, than to the odor for which they were
            # determined to be reliable responders)
            # TODO breakdown by trial as well?
            resp_fracs_to_others = trial_responders.loc[:, :, :, other_odors,
                :, :, cells ].groupby('name1').sum() / (n_odor_cells * 3)

            # TODO revisit (something more clear?)
            resp_fracs_to_others.name = 'of_' + odor + '_resp'

            odor_resp_subset_fracs_list.append(resp_fracs_to_others)
            '''
            norm_frac_reliable_to_others = False
            fracs_reliable_to_others = reliable_responders.loc[other_odors,
                cells].groupby('name1').sum() / n_odor_cells

            if norm_frac_reliable_to_others:
                fracs_reliable_to_others /= frac_reliable_responders
                # TODO TODO TODO does the weird value of the (mix, of_mix_resp)
                # (and other identities) mean that this normalization is not
                # meaningful? should i be doing something differently?

                # TODO maybe in the normed ones, just don't show mix thing,
                # since that value seems weird?

            fracs_reliable_to_others.name = 'of_' + odor + '_resp'

            odor_resp_subset_fracs_list.append(fracs_reliable_to_others)
        del odors

        odor_resp_subset_fracs = pd.concat(odor_resp_subset_fracs_list, axis=1,
            sort=False)

        fig = plt.figure()
        of_mix_reliable_to_others = \
            odor_resp_subset_fracs['of_mix_resp'][odor_order]

        ax = of_mix_reliable_to_others.plot.bar(color='black')
        ax.set_title(label.title() + '\nFraction of mix reliable responders '
            'reliable to other odors')
        savefigs(fig, 'mix_rel_to_others', fname)

        of_mix_reliable_to_others_ratio = \
            of_mix_reliable_to_others / frac_reliable_responders

        # TODO see note above. excluding mix disingenuous?
        of_mix_reliable_to_others_ratio = \
            of_mix_reliable_to_others_ratio[[o for o in odor_order
            if o != 'mix']]
        assert 'mix' not in of_mix_reliable_to_others_ratio.index

        ratio_fig = plt.figure()
        ratio_ax = of_mix_reliable_to_others_ratio.plot.bar(color='black')
        ratio_ax.set_title(label.title() + '\nFraction of mix reliable'
            ' responders reliable to other odors (ratio)')
        savefigs(ratio_fig, 'ratio_mix_rel_to_others', fname)

        # TODO TODO TODO aggregate across flies somehow? points for each
        # fraction ratio? (swarm?) (mean frac at least?)

        # TODO TODO TODO still maybe show this as a matshow, since that seems to
        # be one version of the all-pairwise-venn-diagram thing betty was trying
        # to envision, right?
        '''
        # TODO print the above in some nice(r) way?
        # TODO TODO or save to figure? (like df_to_image?)
        print('Responder fractions to subset of cells responding reliably'
            ' to each odor:')
        print(odor_resp_subset_fracs)
        print('')
        '''

        # TODO maybe try w/ any responses counting it as a responder for the
        # odor, instead of the 2 required for reliable_responders?
        rec_n_reliable_responders = reliable_responders.groupby('cell').sum()

        # TODO maybe use kde instead?
        n_ro_hist_ax.hist(rec_n_reliable_responders, bins=n_ro_bins,
            density=True, histtype='step', color=fly_color, linestyle=linestyle,
            label=label)
        ########################################################################
        # end section to de-dupe w/ code in first loop

n_ro_hist_ax.legend()
n_ro_hist_ax.set_title('Cell tuning breadth')
n_ro_hist_ax.set_xlabel('Number of odors cells respond to (at least twice)')
n_ro_hist_ax.set_ylabel('Fraction of cells')

savefigs(n_ro_hist_fig, None, 'n_ro_hist')

# this is just the mean frac responding... maybe i should have gotten the trial
# info? or rename appropriately?
frac_responder_df = u.add_fly_id(pd.concat(frac_responder_dfs).reset_index(
    name='frac_responding'))

# TODO need to do this for other stuff? do above? in first loop even?
# To differentiate between kiwi mix and control mix in seaborn plots that
# include both in one subfigure.
# Not sure any plots actually *should* have both mixes on literally the
# same subfigure though...
#frac_responder_df.loc[((frac_responder_df.odor_set == 'control') &
#    (frac_responder_df.name1 == 'mix')), 'name1'] = 'cmix'

# This added a column for each element in the order, in each facet,
# including when there was no data. This was not what I wanted, and
# it's unclear if there is a built-in solution to this problem.
'''
odor_sets = set(frac_responder_df.odor_set.unique())
assert len(odor_sets) == 2 and 'control' in odor_sets
odor_set_order = [s for s in odor_sets if s != 'control'] + ['control']
odors_in_order = [cu.odor2abbrev(o) for s in odor_set_order
    for o in u.odor_set2order[s]]

seen_odors = set()
global_odor_order = []
for o in odors_in_order:
    if o != 'mix' and o not in seen_odors:
        global_odor_order.append(o)
        seen_odors.add(o)

# Need to force mix to be at the end, since otherwise it'll be in the middle,
# concatenating the two lists that both contain it together.
global_odor_order.append('mix')
'''

odor_set2order_with_dupes = {s: [cu.odor2abbrev(o) for o in os] for s, os
    in u.odor_set2order.items()}

odor_set2order = dict()
for s, os in odor_set2order_with_dupes.items():
    this_set_order = []
    for o in os:
        if o not in this_set_order:
            this_set_order.append(o)
    odor_set2order[s] = this_set_order

assert len(odor_set2order) == 2, 'below approach only works in that case'
odor_sets = [set(os) for os in odor_set2order.values()]
# Any odors in both can't tell us which set we have.
nondiagnostic_odors = odor_sets[0] & odor_sets[1]

def with_odor_order(plot_fn):
    def ordered_plot_fn(*args, **kwargs):
        odors = args[0].values

        odor_set = None
        for s, os in odor_set2order.items():
            for o in odors:
                if o in nondiagnostic_odors:
                    continue

                if o in os:
                    odor_set = s
                    break

            if odor_set is not None:
                break

        if odor_set is None:
            print('args[0]:')
            print(args[0])
            print('args[1]:')
            print(args[1])
            import ipdb; ipdb.set_trace()
            raise ValueError('could not determine odor_set. needed for order.')

        order = [o for o in odor_set2order[odor_set] if o in odors]

        return plot_fn(*args, order=order, **kwargs)

    return ordered_plot_fn

# This only works because the groupby in loop over colors above and the 
# groupby in add_fly_id both sort, and thus have the same order.
# Also assuming sort order of keys matches sort order of fly_ids, and using
# np.unique rather than <ser>.unique() to also sort.
palette = {i: c for i, c in zip(np.unique(frac_responder_df.fly_id),
    fly_colors)}

# TODO delete after fixing legend business
# (commented b/c colors do at least seem to be in correspondence between
# n_ro_hist and facetgrid below)
#print(frac_responder_df[fly_keys + ['fly_id']].drop_duplicates())
#

g = sns.FacetGrid(frac_responder_df, col='odor_set', hue='fly_id', 
    palette=palette, sharex=False)

# TODO TODO maybe somehow connect lines that share a hue (to make it more
# visually clear how much overal responsiveness is the main thing that varies)
g.map(with_odor_order(sns.swarmplot), 'name1', 'frac_responding')

# TODO TODO either somehow show (date, fly_num) in this legend, or show
# fly_id in n_ro_hist above, to match between them
# TODO TODO maybe modify add_fly_id to add a concatenation of string
# representations of each of the group keys?
g.add_legend()
g.set_axis_labels('Odor', 'Fraction responding')
# TODO way to just capitalize?
g.set_titles('{col_name}')
savefigs(g.fig, None, 'mean_frac_responding')

# TODO TODO actually, try plotting this 2d (maybe i want both triangulars
# after all, for convenience?)
# TODO maybe use map_upper + map_diag?

# TODO lookup which corrs (max / mean) to use from trial_stat, if not gonna
# delete saving multiple
corr_df = u.add_fly_id(corr_df_from_maxes.reset_index())
corr_df['odor_pair'] = [a + ', ' + b for a, b in zip(corr_df.name1_a,
    corr_df.name1_b)]

def add_odorset(df):
    # use rec_keys2odor_set
    df['odor_set'] = df.set_index(fly_keys + [rec_key]).index.map(
        rec_keys2odor_set)
    return df

add_odorset(corr_df)

# TODO TODO want to agg across trials here? mean? max?

# TODO some way to have diff sets of cols on each row?
# (similar to odor order problem i dealt with w/ wrapper fn above...)
# may just want two separate plots...

# TODO tune fig size (+ fonts) for all of these to look good in pdf
# (+ above facetgrid stuff). should be roughly final size, too, so not
# scaled too much.

col_order = odor_set2order['kiwi']
kiwi_corr_df = corr_df[corr_df.odor_set == 'kiwi']
g = sns.FacetGrid(kiwi_corr_df, col='name1_a', hue='fly_id', 
    palette=palette, sharex=False, col_wrap=4, col_order=col_order
)
g.map(with_odor_order(sns.swarmplot), 'name1_b', 'corr')
g.add_legend()
g.set_axis_labels('Odor B', 'Correlation')
g.set_titles('{col_name}')
savefigs(g.fig, None, 'kiwi_corrs')

col_order = odor_set2order['control']
ctrl_corr_df = corr_df[corr_df.odor_set == 'control']
g = sns.FacetGrid(ctrl_corr_df, col='name1_a', hue='fly_id', 
    palette=palette, sharex=False, col_wrap=4, col_order=col_order
)
g.map(with_odor_order(sns.swarmplot), 'name1_b', 'corr')
g.add_legend()
g.set_axis_labels('Odor B', 'Correlation')
g.set_titles('{col_name}')
savefigs(g.fig, None, 'ctrl_corrs')

"""
g = sns.FacetGrid(corr_df, row='odor_set', col='name1_a', hue='fly_id', 
    palette=palette, sharex=False, margin_titles=True)
g.map(with_odor_order(sns.swarmplot), 'name1_b', 'corr')
'''
g.add_legend()
g.set_axis_labels('Odor pair', 'Correlation')
g.set_titles('{col_name}')
'''
#savefigs(g.fig, None, 'corrs')
"""

# TODO TODO TODO TODO plot distribution of linearity of individual cells

# TODO TODO TODO make sense to plots residuals? should i scale by # of cells or
# something (like average residual per cell)?

# TODO TODO TODO some kind of plot showing adaptation to each odor, to see if
# that thing i saw once to real kiwi (faster adaptation, i think) holds
# (could use difference in response_magnitudes across first and last trials?
# fit line / exp and get slope / tau(?) ?)

# TODO TODO some way to make a statement about the PCA stuff across flies?
# worth it?
# TODO TODO TODO also aggregate some pca outputs + linearity analysis outputs,
# so that down here i can do stuff like compare # of cells superlinear / or at
# least compare fits

# TODO TODO TODO repeat this analysis only using kiwi data for fixed
# concentration!
linearity_cell_df = add_odorset(linearity_cell_df.reset_index())
# TODO TODO maybe also filter by some minimum of responsiveness? / reliability?
# / noise?
kiwi_cell_errs = linearity_cell_df.loc[linearity_cell_df.odor_set == 'kiwi',
    'weighted_mix_diff'
]
ctrl_cell_errs = linearity_cell_df.loc[linearity_cell_df.odor_set == 'control',
    'weighted_mix_diff'
]
linearity_dist_fig, ax = plt.subplots()
# TODO TODO pick consistent colors for odor_sets?
sns.distplot(kiwi_cell_errs, label='Kiwi')
sns.distplot(ctrl_cell_errs, label='Control')
ax.set_title('Distributions of cell residuals from linear mixture model')
ax.set_xlim([-1, 1])
ax.legend()
savefigs(linearity_dist_fig, None, 'cell_linearity_dists')

# TODO TODO maybe only render stuff generated on this run into report, so that
# it doesn't include things using different parameters
# (how to implement... keep track of filenames saved in here?)
# TODO TODO TODO generate report pdf here
# (any point to having that in its own module anyway?)
# TODO TODO render in parameters to report + stuff about flies
# (include them in a summary section(?) at the top or something)

# TODO and maybe symlink 'latest_report.pdf' to this or something

plt.show()
import ipdb; ipdb.set_trace()

