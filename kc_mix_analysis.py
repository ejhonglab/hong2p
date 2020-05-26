#!/usr/bin/env python3

import os
from os.path import exists, join, split, getmtime
import glob
from pprint import pprint
import warnings
import time
import pickle
import argparse
import shutil
from itertools import zip_longest, product, combinations, starmap
from collections import deque
import subprocess
import multiprocessing as mp
import socket

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import h5py

import chemutils as cu

import hong2p.util as u

# Having all matplotlib-related imports come after `hong2p.util` import,
# so that I can let `hong2p.util` set the backend, which it seems must be set
# before the first import of `matplotlib.pyplot`
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Note: some imports were pushed down, just before code that uses them, to
# reduce the number of hard dependencies.


parser = argparse.ArgumentParser(description='Analyzes calcium imaging traces '
    'stored as pickled pandas DataFrames that gui.py outputs.'
)
parser.add_argument('-c', '--only-analyze-cached', default=False,
    action='store_true', help='Only analyzes cached outputs from previous runs '
    'of this script. Will not load any trace pickles.'
)
parser.add_argument('-n', '--no-save-figs', default=False, action='store_true',
    help='Does not save any figures.'
)
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='Otherwise prints which figs are being saved.'
)
parser.add_argument('-p', '--print-full-fig-paths', default=False,
    action='store_true', help='Prints full paths to figs as they are saved.'
)
parser.add_argument('-t', '--test', default=False, action='store_true',
    help='Only loads two trace pickles, for faster testing.'
)
parser.add_argument('-d', '--delete-existing-figs', default=False,
    action='store_true', help='Deletes any existing figs at the start, '
    'to prevent confusion about which figs were generated this run.'
)
parser.add_argument('-a', '--across-flies-only', default=False,
    action='store_true', help='Only make plots for across-fly analyses.'
)
parser.add_argument('-i', '--interactive-plots', default=False,
    action='store_true', help='Show plots interactively (at end).'
)
parser.add_argument('-m', '--max-open-saved-figs', type=int, default=3,
    help='This number of figs will remain open despite being saved. '
    'Use for debugging with minimal other changes.'
)
parser.add_argument('-o', '--no-report-pdf', default=False,
    action='store_true', help='Does not generate a PDF report at the end.'
)
parser.add_argument('-e', '--earlier-figs', default=False,
    action='store_true', help='Includes any existing PDF figures in report, '
    'not just figures generated this run (or the run that produced the '
    'intermediates used for this run).'
)
parser.add_argument('-f', '--save-nonreport-figs', default=False,
    action='store_true', help='If passed, will save figs whether or not '
    'they will be placed in the PDF report.'
)
parser.add_argument('-b', '--debug-shell', default=False,
    action='store_true', help='Drops to an ipdb shell at end, for further '
    'interactive analysis or debugging.'
)
parser.add_argument('-u', '--skip-cell-subset-linearity', default=False,
    action='store_true', help='Skips linearity analysis on functional subsets'
    ' of cells.'
)
parser.add_argument('-j', '--no-parallel-process-traces', default=False,
    action='store_true', help='Disables parallel calls to process_traces. '
    'Useful for debugging internals of that function.'
)
parser.add_argument('-r', '--plot-formats', action='store', default='png,pdf',
    help='Extensions for plot formats to save figures in. Just include what '
    'comes after the period. Use commas to separate multiple format extensions.'
)
args = parser.parse_args()

# TODO maybe get this from some object that stores the params / a context
# manager of some kind that captures param variable names defined inside?
param_names = [
    'fix_ref_odor_response_fracs',
    'ref_odor',
    'ref_response_percent',
    'mean_zchange_response_thresh',
    'baseline_start',
    'baseline_end',
    'response_start',
    'response_calling_s',
    'trial_stat'
]

odor_set_order = ['kiwi', 'control', 'flyfood']
use_existing_abbrevs = False

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

# TODO TODO err if one of these flags is true but we are analyzing cached
# data, and cache doesn't have the data of interest
correlations = True
plot_correlations = True
trial_order_correlations = True
odor_order_correlations = True

hallem_correlations = True

# TODO return to trying to aggregate some type of this analysis across flies
# after doing corr + linearity stuff across flies (+ ROC!)
do_pca = True
do_roc = False
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
verbose_savefig = not args.silent
print_full_plot_paths = args.print_full_fig_paths
# Otherwise, they are closed right after being saved.
show_plots_interactively = args.interactive_plots
if show_plots_interactively:
    plt.rcParams.update({'figure.max_open_warning': 0})

save_figs = not args.no_save_figs

print_mean_responder_frac = False
print_responder_frac_by_trial = False
print_reliable_responder_frac = False

# If True, only two input pickles are loaded (both from one fly),
# to test parts of the code faster.
test = args.test

# Data whose trace pickles contain this substring are loaded.
#test_substr = '08-27_9'
test_substr = '10-04_1'

start_time_s = time.time()

fig_dir = 'mix_figs'
if args.delete_existing_figs:
    if exists(fig_dir):
        print(f'Deleting {fig_dir} and all contents!')
        shutil.rmtree(fig_dir)

if not exists(fig_dir):
    os.mkdir(fig_dir)

# TODO validate this at argparse level. and maybe check each is supported by
# matplotlib at that point too? they have some list of supported extensions?
plot_formats = args.plot_formats.split(',')
if 'pdf' not in plot_formats:
    # TODO err if report requested
    warnings.warn('will not be able to generate PDF report, as LaTeX report '
        'generation currently requires PDF inputs'
    )

for pf in plot_formats:
    pf_dir = join(fig_dir, pf)
    if not exists(pf_dir):
        os.mkdir(pf_dir)

# TODO derive these from key defs in util?
fly_keys = ['prep_date', 'fly_num']
rec_key = 'thorimage_id'
cell_cols = ['name1', 'name2', 'repeat_num', 'cell']
within_recording_stim_cols = ['name1', 'name2', 'repeat_num', 'order']


# (_with_dupes, because some things in u.odor_set2order evaluate to the same
# abbreviation)
odor_set2order_with_dupes = {s: [cu.odor2abbrev(o) for o in os] for s, os
    in u.odor_set2order.items()
}
odor_set2order = dict()
for s, oset in odor_set2order_with_dupes.items():
    this_set_order = []
    for o in oset:
        if o not in this_set_order:
            this_set_order.append(o)
    odor_set2order[s] = this_set_order

# TODO clean this up / only optionally print?
for os, unabbrev_odors in u.odor_set2order.items():
    abbrev_odors = odor_set2order_with_dupes[os]
    assert len(unabbrev_odors) == len(abbrev_odors)
    seen = set()
    print(os)
    for abbrev, full in zip(abbrev_odors, unabbrev_odors):
        if abbrev in seen:
            continue
        print(f' {abbrev} = {full}')
        seen.add(abbrev)
    print()
#
del odor_set2order_with_dupes

odor_set_odors = [set(os) for os in odor_set2order.values()]
odor_counts = dict()
for odors in odor_set_odors:
    for odor in odors:
        if odor not in odor_counts:
            odor_counts[odor] = 1
        else:
            odor_counts[odor] += 1
nondiagnostic_odors = {o for o, c in odor_counts.items() if c > 1}
del odor_counts


def in_worker_process():
    pname = mp.current_process().name
    # Despite Bakuriu's answer here,https://stackoverflow.com/questions/18216050
    # this seems to be reliable as I've tested it.
    if pname == 'MainProcess':
        return False
    else:
        assert 'PoolWorker' in pname
        return True


# TODO also move this if i move solvents + natural definitions?
def is_component(name):
    return name not in (('mix',) + u.natural + u.solvents)


def check_fraction_series(ser):
    assert (ser.isnull() | ((ser <= 1.0) & (0.0 <= ser))).all()


# TODO maybe refactor this + add_metadata into util (+ use them elsewhere)
# (aren't there other places i do stuff like this? gui?)
def get_single_index_val(df, var):
    values  = df[var].unique()
    assert len(values) == 1
    return values[0]


def add_metadata(df, out_df, keys=('prep_date', 'fly_num', 'thorimage_id')):
    vals = [get_single_index_val(df, k) for k in keys]
    # TODO i thought this would be the one line way to do it, but i
    # guess not... is there one?
    #out_df = pd.concat([out_df], names=keys, keys=vals)
    for k, v in zip(keys[::-1], vals[::-1]):
        out_df = pd.concat([out_df], names=[k], keys=[v])
    return out_df
#


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


# TODO TODO maybe factor this save figs logic out (if not basically the whole
# fn. may want to wrap a little for use here though?). keeping the last n
# figs open could be useful in a lot of my analysis

# So that the last few figs open can still be shown interactively while
# debugging, in contrast to show_plots_interactively case, where all are shown.
if in_worker_process():
    # TODO 0 works for this right?
    max_open_noninteractive = 0
else:
    max_open_noninteractive = args.max_open_saved_figs
fig_queue = deque()

# Currently putting in LaTeX report in the order saved here, though still
# with all paired stuff coming after non-paired stuff.

plot_prefix2latex_data = dict()
plots_made_this_run = set()
# TODO do i need to include any of these in what gets [un]pickled now?
manual_section2order = dict()
section2order = dict()
section2subsection_orders = dict()
#
def savefigs(fig, plot_type_prefix, *vargs, odor_set=None, section=None,
    subsection=None, note=None, exclude_from_latex=False, section_order=None,
    order=None):
    """
    section_order (None or int): If an int, it is used as a sort key to order
        this section relative to the others. Default order start at 0 and
        increases with each section added.
    """
    # TODO maybe rename subsection to figure name or something?
    # (though if section is only thing passed, it has the same meaning...)
    # change to have more consistent meanings (e.g. generally do NOT pass
    # section, and only do so when we want a section, rather than a fig title)
    # TODO TODO test case where we actually do want both a section title and
    # fig title (in adaptation case now, not actually making fig titles)

    # TODO maybe add flag to toggle this behavior. using it for debugging
    # across ssh -X to atlas and local blackbox (want to compare figures,
    # so hostname somewhere would be good)
    # TODO TODO maybe just monkey patch matplotlib to change the default
    # behavior? (factor into fn and call in util, where backend is selected?)
    # or call just after imports in util?
    man = plt.get_current_fig_manager()
    old_title = man.canvas.get_window_title()
    man.canvas.set_window_title(f'{old_title} ({socket.gethostname()})')
    del man, old_title
    #

    out_strs = []
    if save_figs and (args.save_nonreport_figs or not exclude_from_latex):
        if len(vargs) == 1:
            recording_info_suffix = vargs[0]
            assert not recording_info_suffix.startswith('_')
        elif len(vargs) == 0:
            recording_info_suffix = None
        else:
            raise ValueError('too many positional arguments')

        plot_type_parts = [plot_type_prefix]
        if odor_set is not None:
            plot_type_parts.append(odor_set)

        if recording_info_suffix is not None:
            plot_type_parts.append(recording_info_suffix)

        plot_type = '_'.join(plot_type_parts)

        if verbose_savefig:
            out_s = f'writing plots for {plot_type}'
            if not in_worker_process():
                print(out_s)
            out_strs.append(out_s)

        for pf in plot_formats:
            plot_fname = join(fig_dir, pf, plot_type + '.' + pf)
            if print_full_plot_paths:
                if not in_worker_process():
                    print(plot_fname)
                out_strs.append(plot_fname)

            # This probably means we are unintentionally saving two different
            # things to the same filename.
            if args.delete_existing_figs and exists(plot_fname):
                raise IOError('likely saving two different things to '
                    f'{plot_fname}! fix code.'
                )

            # This should be redundant with check above, but just to be sure.
            assert plot_fname not in plots_made_this_run
            fig.savefig(plot_fname)
            if not exclude_from_latex:
                plots_made_this_run.add(plot_fname)

        if print_full_plot_paths:
            if not in_worker_process():
                print('')
            out_strs.append('')

        if not exclude_from_latex:
            paired = recording_info_suffix is not None
            # TODO maybe assert that section is not None if subsection is not
            # None
            if section is None:
                section = ''
            if subsection is None:
                subsection = ''

            # TODO should this subsection_order stuff be in if below? or can i
            # remove if below / move some of order stuff up here?
            if section not in section2subsection_orders:
                section2subsection_orders[section] = dict()

            if subsection not in section2subsection_orders[section]:
                # This is to order subsections within a section.
                # I have not had a need to manually specify this yet.
                subsection_order = len(section2subsection_orders[section])
                section2subsection_orders[section][subsection] = \
                    subsection_order
            else:
                subsection_order = \
                    section2subsection_orders[section][subsection]
            #

            if plot_type_prefix not in plot_prefix2latex_data:
                if section_order is None:
                    # TODO handle case where manual section order specified
                    # after an automatic one. want to apply manual order
                    # back to any previous instances of the section
                    if section in manual_section2order:
                        section_order = manual_section2order[section]

                    elif section in section2order:
                        section_order = section2order[section]

                    else:
                        # TODO modify this to only count sections before, not
                        # all items before (shouldt affect function though as
                        # long as section_order is constant w/in a section)
                        n_before = len([
                            (s,d) for s, d in plot_prefix2latex_data.items()
                            if d['paired'] == paired
                        ])
                        section_order = n_before
                        section2order[section] = section_order
                else:
                    if section in manual_section2order:
                        assert section_order == manual_section2order[section]
                    else:
                        manual_section2order[section] = section_order

                # TODO maybe replace this with a scalar / arbitrary length
                # sequence of keys or sort by? or detect groups within a section
                # / subsection and only apply one of these ordering keys within
                # those groups? (so as to not mix linearity analysis looking at
                # cell subsets with analysis on all cells/responders)

                # For now, using this to order within both sections AND
                # subsections.
                if order is None:
                    n_before_in_subsection = len([
                        (s,d) for s, d in plot_prefix2latex_data.items()
                        if d['paired'] == paired and d['section'] == section and
                        d['subsection'] == subsection
                    ])
                    order = n_before_in_subsection

                latex_data = {
                    'section': section,
                    'subsection': subsection,
                    'note': note,
                    'paired': paired,
                    'section_order': section_order,
                    'subsection_order': subsection_order,
                    'order': order
                }
                plot_prefix2latex_data[plot_type_prefix] = latex_data
            else:
                # Can't use plot_type_prefix as a key if the data would need
                # to be different for two cases with the same would-be-key.
                prev_latex_data = plot_prefix2latex_data[plot_type_prefix]
                assert section == prev_latex_data['section']
                assert subsection == prev_latex_data['subsection']
                assert note == prev_latex_data['note']
                assert paired == prev_latex_data['paired']

    if not show_plots_interactively:
        fig_queue.append(fig)
        fignums = plt.get_fignums()
        n_open = len(fignums)
        while n_open > max_open_noninteractive:
            try:
                fig_to_close = fig_queue.popleft()
                plt.close(fig_to_close)
                n_open -= 1
            # Since we may have open figures that we have not yet tried to
            # save. They should not be closed, but will still influence
            # n_open.
            except IndexError:
                break

    # So that functions parallelizing stuff that calls savefigs can print all of
    # the usual output in the same order as it would be printed when not
    # parallelized.
    return '\n'.join(out_strs)


def load_pid_data(df):
    keys = u.recording_df2keys(df)
    mat = u.matfile(*keys)
    print(mat)
    ti = u.load_mat_timing_info(mat)
    # TODO do all the mat files i'm currently using have these variables?
    # they seem like what i want. are they?
    # TODO TODO need to do same kind of painting into blocks i remember doing
    # in gui at some point w/ frames -> trials?? use same code?
    stim_ic_idx = ti['stim_ic_idx']
    stim_fc_idx = ti['stim_fc_idx']

    ts_paths = df.thorsync_path.unique()
    assert len(ts_paths) == 1
    thorsync_dir = ts_paths[0]

    h5_files = glob.glob(join(thorsync_dir, '*.h5'))
    if len(h5_files) == 0:
        warnings.warn(f'No .h5 file found under ThorSync dir {thorsync_dir}! '
            'Could not load PID data!'
        )
        return None
    assert len(h5_files) == 1
    h5_file = h5_files[0]
    print(f'loading {h5_file}... ', end='', flush=True)
    # TODO switch back to context manager after debugging
    '''
    with h5py.File(h5_file, 'r') as f:
        print(f.keys())
        import ipdb; ipdb.set_trace()
    '''
    f = h5py.File(h5_file, 'r')
    # TODO isn't there timing information? no way here to get times for thorsync
    # samples?????
    # TODO otherwise, use xml to get sampling rate i guess?
    # TODO cases where it was named something slightly different?
    #print(f.keys())
    try:
        assert 'pid' in f['AI'].keys()
    except AssertionError:
        print('pid not in f["AI"].keys()!')
        print('f["AI"].keys():', f['AI'].keys())
        import ipdb; ipdb.set_trace()
    pid = f['AI']['pid']
    f.close()
    print('done')

    # TODO TODO TODO indexed w/ pandas metadata as current df is
    import ipdb; ipdb.set_trace()
    


trace_pickle2df = dict()
trace_pickle2other_data = dict()
# TODO maybe flag here + thread through to CLI arg for loading PID stuff if we
# want to make PID plots
# TODO maybe move this + caching to util?
def read_pickle(trace_pickle):
    had_other_data = False
    other_data = None
    if trace_pickle in trace_pickle2df:
        if trace_pickle in trace_pickle2other_data:
            other_data = trace_pickle2other_data[trace_pickle]
            had_other_data = True

        df = trace_pickle2df[trace_pickle]
    else:
        with open(trace_pickle, 'rb') as f:
            data = pickle.load(f)
            # not sure this will always work...
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                assert type(data) is dict
                trace_pickle_key = 'trace_df'
                assert trace_pickle_key in data.keys()
                df = data[trace_pickle_key]
                assert isinstance(data, pd.DataFrame)
                other_data = {
                    k: v for k, v in data.items() if k != trace_pickle_key
                }
                if len(other_data) == 0:
                    other_data = None
                else:
                    trace_pickle2other_data[trace_pickle] = other_data
                    had_other_data = True

        trace_pickle2df[trace_pickle] = df

    if not had_other_data:
        # TODO uncomment after lab meeting
        '''
        other_data = dict()
        other_data['pid'] = load_pid_data(df)
        if other_data['pid'] is not None:
            trace_pickle2other_data[trace_pickle] = other_data
        else:
            other_data = None
        '''
        other_data = None

    return df, other_data


def order_by_odor_sets(trace_pickles, drop_if_missing={'kiwi'},
    odorset_first=True):
    """
    Returns re-ordered list of pickle filenames, with data from within flys now
    appearing in a fixed order.

    This is so the odor_set with the reference odor can be analyzed first,
    and that threshold applied to the other recording(s) with the same fly.
    """
    # TODO rename to differentiate from global odor_set2order
    _odor_set2order = {o: i for i, o in enumerate(odor_set_order)}
    fname2keys = dict()
    fly_keys2odor_sets = dict()
    fly_keys2fnames = dict()
    for tp in trace_pickles:
        df, _ = read_pickle(tp)

        # Excluding last recording col, so it just indexes a fly, not a
        # recording.
        fly_keys = df[u.recording_cols[:-1]].drop_duplicates()
        assert len(fly_keys) == 1
        fly_keys = tuple(fly_keys.iloc[0])

        odor_set = u.odorset_name(df)
        # TODO delete all other add_odorset stuff?
        assert 'odor_set' not in df.columns
        # TODO what if anything depended on this being set?
        # (because it used to be incorrect, w/ rhs = 'odor_set', so anything
        # dependent may have also behaved incorrectly...
        # and if nothing is dependent on this, delete it
        df['odor_set'] = odor_set
        #

        if fly_keys not in fly_keys2odor_sets:
            fly_keys2odor_sets[fly_keys] = {odor_set}
            fly_keys2fnames[fly_keys] = [tp]
        else:
            fly_keys2odor_sets[fly_keys].add(odor_set)
            fly_keys2fnames[fly_keys].append(tp)

        if odorset_first:
            keys = (_odor_set2order[odor_set],) + fly_keys
        else:
            keys = fly_keys + (_odor_set2order[odor_set],) 
        fname2keys[tp] = keys

    if drop_if_missing is None or len(drop_if_missing) == 0:
        raise NotImplementedError('no support for missing ref odor')
    else:
        # TODO TODO TODO implement some kind of propagation of pinned thresholds
        # through all odorsets, so we don't need to throw out this data
        with_all_required_odorsets = []
        for fly_keys, odor_sets in fly_keys2odor_sets.items():
            if all([required in odor_sets for required in drop_if_missing]):
                with_all_required_odorsets.extend(fly_keys2fnames[fly_keys])
            else:
                warnings.warn(f'excluding data for fly {fly_keys} because it '
                    f'did not have all required odor sets.\nHad: {odor_sets}'
                    f'\nRequired: {drop_if_missing}'
                )
        trace_pickles = with_all_required_odorsets

    return sorted(trace_pickles, key=fname2keys.get)


def fill_to_cartesian(df, cols_to_combine):
    unique_per_col = [df[c].unique() for c in cols_to_combine]
    # TODO the list call necessary? or need further processing, like explicit
    # tuples for elements?
    idx = list(product(*unique_per_col))

    full_len = np.prod([len(u) for u in unique_per_col])
    assert len(idx) == full_len

    # Fills missing values with NaN.
    full_df = df.set_index(cols_to_combine).reindex(idx)
    assert len(full_df) == full_len
    return full_df


# TODO if i ever get registration of cell IDs across recordings working, maybe
# adapt this to not just operate within (fly_id AND odor_set) (though mix, etc,
# should still be considered as different things in different odor_set
# contexts...)
def group_cells_by_odors_responded(responded, per_fly=False, verbose=True,
    exclude_solvents=True, exclude_natural=True,
    title_and_fname_strs=True,
    title_str_prefix='cells reliable only to ',
    nonresponder_title_str='non-reliable',
    nonresponder_fname_str='nonreliable'):
    """
    Args:
    responded (pd.Series): A mask of whether the cell was considered to respond,
        indexed by fly_id, odor_set, cell, name1. Pass mask of whether a cell
        was reliable to each odor to get cells grouped by which odors they were
        reliable to.
    """
    assert set(responded.index.names) == {'fly_id', 'odor_set', 'cell', 'name1'}

    if exclude_solvents:
        responded = responded[
            (~ responded.index.get_level_values('name1').isin(u.solvents))
        ]
    if exclude_natural:
        responded = responded[
            (~ responded.index.get_level_values('name1').isin(u.natural))
        ]

    group_keys = ['odor_set']
    if per_fly:
        group_keys.append('fly_id')

    groups = []
    for gn, gser in responded.groupby(group_keys):
        ggroups = []
        if len(group_keys) == 1:
            # So the zip still works as expected in this case.
            gn = (gn,)
        group_key_dict = dict(zip(group_keys, gn))

        oset = group_key_dict['odor_set']
        odors = odor_set2order[oset]
        if exclude_solvents:
            odors = [o for o in odors if o not in u.solvents]
        if exclude_natural:
            odors = [o for o in odors if o not in u.natural]
        odor2idx = {o: i for i, o in enumerate(odors)}

        '''
        from math import factorial
        n = len(odors)
        print('Expected number of combinations:',
            int(sum([factorial(n) / (factorial(k) * factorial(n - k))
            for k in range(len(odors) + 1)]))
        )
        '''
        # This will also include non-responders when n_subset_odors=0.
        for n_subset_odors in range(len(odors) + 1):
            for odor_subset in combinations(odors, n_subset_odors):
                in_subset = gser.index.isin(odor_subset, level='name1')
                keys = ['fly_id','odor_set','cell']
                all_in_subset = gser[in_subset].groupby(keys).all()
                none_outside_subset = (~ gser[~ in_subset]).groupby(keys).all()

                if len(odor_subset) == 0:
                    cell_subset = none_outside_subset
                elif len(odor_subset) == len(odors):
                    cell_subset = all_in_subset
                else:
                    cell_subset = all_in_subset & none_outside_subset

                odor_subset = sorted(odor_subset, key=odor2idx.get)
                if len(odor_subset) == 0:
                    odors_str = nonresponder_title_str
                else:
                    odors_str = ','.join(odor_subset)

                group_dict = {
                    'cell_subset': cell_subset,
                    'odors': odor_subset,
                    'odors_str': odors_str,
                    'cell_fraction': cell_subset.sum() / len(cell_subset)
                }
                if title_and_fname_strs:
                    if len(odor_subset) == 0:
                        fname_str = nonresponder_fname_str
                    else:
                        fname_str = '_'.join(odor_subset)

                    title_str = f'{title_str_prefix}{odors_str} ({oset})'
                    fname_str = f'{oset}_{fname_str}'

                    group_dict['cell_subset_title_str'] = title_str
                    group_dict['cell_subset_fname_str'] = fname_str

                for k, v in group_key_dict.items():
                    assert k not in group_dict
                    group_dict[k] = v

                ggroups.append(group_dict)

        assert np.isclose(1.0, sum([x['cell_fraction'] for x in ggroups]))
        ggroups = sorted(ggroups, key=lambda x: x['cell_subset'].sum())[::-1]

        if verbose:
            # TODO TODO TODO some graphic/metric that could show how well these
            # rankings match up across flies, within an odorset?
            # maybe show bar plot in sorted order computed across all flies,
            # then show points of individual flies about that (in usual hues)
            # (or show deviations?)
            print(group_key_dict)
            for i, x in enumerate(ggroups):
                if x['cell_fraction'] > 0:
                    # denominator assumes len(cell_subset) is same across all
                    # iterations of inner loop (within oset), but that is true
                    print(f"{x['odors']}: {x['cell_fraction']:.4f} "
                        f"({x['cell_subset'].sum()}", end=''
                    )
                    if i == 0:
                        print(f"/{len(cell_subset)})")
                    else:
                        print(")")
            print('')
        
        groups.extend(ggroups)

    return groups


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
            symmetric_df.index.levels) if len(levels) == 1
        ]
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
        # The right expression is the number of elements expected for the
        # triangular of a square matrix w/ side length n, if the diagonal
        # is INCLUDED.
        assert n_nonnull == (n * (n + 1) / 2)

        # TODO make sure this also still works in non-multiindex case!
        tidy = masked.stack(level=masked.columns.names)
        assert tidy.shape == (n_nonnull,)

    tidy.name = name
    return tidy


# TODO rename if it could make it more accurate
def invert_melt_symmetric(ser, suffixes=('_a', '_b')):
    """
    """
    assert len(ser.shape) == 1, 'not a series'
    assert len(ser.index.names) == len(set(ser.index.names)), \
        'index names should be unique'

    assert len(suffixes) == 2 and len(set(suffixes)) == 2
    s0, s1 = suffixes

    levels_to_drop = set(ser.index.names)
    col_prefixes = []
    for c in ser.index.names:
        if type(c) is not str:
            continue

        if c.endswith(s0):
            prefix = c[:-len(s0)]
            if (prefix + s1) in ser.index.names:
                col_prefixes.append(prefix)
                levels_to_drop.remove(prefix + s0)
                levels_to_drop.remove(prefix + s1)

    levels_to_drop = list(levels_to_drop)
    # This does also work in the case where `levels_to_drop` is empty.
    ser = ser.droplevel(levels_to_drop)
    return ser.unstack([p + s0 for p in col_prefixes])


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


def minimize_multiple_init(fn, initial_param_list, args=tuple(), squeeze=True,
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
    weighted_mix_diff = weighted_sum - mix
    im2 = matshow(ax, weighted_mix_diff[ordered_cells], cmap='coolwarm',
        aspect='auto')

    # move to right if i want to keep the y ticks
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(['sum - mix'], fontsize=fontsize,
        rotation=xtickrotation)
    ax.tick_params(bottom=False)

    f3.colorbar(im2, cax=cax2)
    diff_cbar_label = f'{u.dff_latex} difference'
    cax2.set_ylabel(diff_cbar_label)

    savefigs(f3, 'odorandfit', fname, exclude_from_latex=True)


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

        unstd_pca_obj = PCA()
        unstd_pca_obj.fit(df)

        f1b = plt.figure()
        # TODO TODO factor the 15 into a kwarg or something
        # TODO is explained_variance_ratio_ what i want exactly?
        plt.plot(unstd_pca_obj.explained_variance_ratio_[:15], 'k.')
        plt.title('PCA on raw data')
        plt.xlabel('Principal component')
        plt.ylabel('Fraction of explained variance')

        '''
        pc_df = pd.DataFrame(columns=df.columns,
                             data=unstd_pca_obj.components_)

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

    std_pca_obj = PCA()
    std_pca_obj.fit(df_standardized)

    pca_2 = PCA(n_components=2)
    # TODO TODO TODO if pca_obj from fit (above) already has .components_, then
    # why even call fit_transform???
    pca_data = pca_2.fit_transform(df_standardized)
    #import ipdb; ipdb.set_trace()

    for n in range(pca_2.n_components):
        assert np.allclose(std_pca_obj.components_[n], pca_2.components_[n])

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
    plt.plot(std_pca_obj.explained_variance_ratio_[:15], 'k.')
    plt.title('PCA on standardized data')
    plt.xlabel('Principal component')
    plt.ylabel('Fraction of explained variance')

    '''
    pc_df = pd.DataFrame(columns=df.columns, data=std_pca_obj.components_)
    ####pc_df.rename(columns=fmt_chem_id, inplace=True)
    ####pc_df.columns.name = 'Chemical'
    for pc in range(2):
        print('Standardized PC{}:'.format(pc))
        print(pc_df.iloc[pc].abs().sort_values(ascending=False)[:10])
    '''

    if fname is not None:
        savefigs(f1a, 'pca_unstandardized', fname, exclude_from_latex=True)
        savefigs(f1b, 'scree_unstandardized', fname, exclude_from_latex=True)
        savefigs(f2a, 'pca', fname, exclude_from_latex=True)
        savefigs(f2b, 'scree', fname, exclude_from_latex=True)

    # TODO TODO TODO return PCA data somehow (4 things, 2 each std/nonstd?)
    # or components, explained variance, and fits (which are what exactly?),
    # for each?
    ret_df = pd.DataFrame({
        # TODO TODO is this right? is it right that cells are considered
        # features?
        'component_number':
            np.arange(len(std_pca_obj.explained_variance_ratio_)) + 1,

        'std_explained_var_ratio': std_pca_obj.explained_variance_ratio_,
        'unstd_explained_var_ratio': unstd_pca_obj.explained_variance_ratio_
    })
    return ret_df


# TODO maybe try on random subsets of cells (realistic # that MBONs read out?)
# TODO TODO count mix as same class or not? both?
# how do shen et al. disentagle this again? their generalization etc figures?
# TODO TODO use metric other than AU(roc)C since imbalanced classes
# area under ~"precision-recall" curve?
def roc_analysis(window_trial_stats, reliable_responders, fname=None):
    # TODO maybe don't subset to "reliable" responders as shen et al do?
    # TODO maybe just two distributions, one for natural one control?
    odors = window_trial_stats.index.get_level_values('name1')
    df = window_trial_stats[~ odors.isin(u.solvents)].reset_index()

    auc_dfs = []
    for segmentation in (True, False):
        for odor in df.name1.unique():
            if segmentation and odor in ('mix',) + u.natural:
                continue

            odor_reliable_responders = reliable_responders.loc[(odor,)]
            odor_reliable_cells = odor_reliable_responders[
                odor_reliable_responders].index

            cells = []
            aucs = []
            for gn, gdf in df[df.cell.isin(odor_reliable_cells)
                ].groupby('cell'):

                if segmentation:
                    gdf = gdf[~ gdf.name1.isin(u.natural)]

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
                # TODO maybe don't save these figs tho?
                # (would probably also screw up my paired-plot-detection
                # scheme in savefig)
                # TODO would need to at least check that pairing in report
                # generation also works w/ arbitrary extra parts (beyond
                # glob str, maybe), to also pair on the odor here
                savefigs(fig, task_fname, odor_fname + '_' + fname,
                    exclude_from_latex=True
                )

    # TODO maybe also plot average distributions for segmentation and
    # discrimination, within this fly, but across all odors?
    # (if so, should odors be weighted equally, or should each cell (of which
    # some odors will have less b/c less reliable responders)?)

    # TODO also return auc values (+ n cells used in each case?) for
    # analysis w/ data from other flies (n cells in current return output?)
    auc_df = pd.concat(auc_dfs, ignore_index=True)
    return auc_df


# TODO TODO more checks possible after the fact, to catch cases where a bug
# might have led to data being entered in a place not consistent w/ the
# label that section of the plot will ultimately have
# (e.g. odor data filled in, w/ x-labels ultimately sometimes not referring to
# the odor for the odor data in a particular column)
# TODO TODO same as above w/ odor_facetgrids
component_sum_odor_name = 'sum'
component_sum_diff_odor_name = component_sum_odor_name + ' - mix'
ax_id2order = dict()
# TODO TODO TODO refactor / make similar fn to allow plotting across a facetgrid
# / catplot w/ levels of x variable different in each facet. provide some
# general way of specifying order (can we also just rely on order in input?).
# want this for plotting y=cell_fraction, x=odors_str for cells grouped by which
# odors they respond to (catplot w/ sharex=False doesn't do what i wanted)
# see https://github.com/mwaskom/seaborn/issues/209 (probably others too)
# TODO actually facetgrid w/ sharex=False *can* work, but it raises a warning
# saying order argument should be passed to ensure correctness. ...under what
# conditions can lack of order make a wrong plot???
# actually facetgrid made as above shares labels still... (was just
# g.set_xticklabels that cased the problem in this case)
def with_odor_order(plot_fn, **fn_kwargs):
    def ordered_plot_fn(*args, **kwargs):
        for fk, fv in fn_kwargs.items():
            if fk not in kwargs:
                kwargs[fk] = fv

        odors = args[0].values
        yvals = args[1].values
        assert len(odors) == len(yvals)

        # TODO refactor to use util fn i might have made for this?
        odor_set = None
        for s, oset in odor_set2order.items():
            for o in odors:
                if o in nondiagnostic_odors:
                    continue

                if o in oset:
                    odor_set = s
                    break

            if odor_set is not None:
                break

        if odor_set is None:
            raise ValueError('could not determine odor_set. needed for order.')

        # order must be the same across all calls on the same axis,
        # or else the x-labels will be wrong for some of the points.
        order = list(odor_set2order[odor_set])
        if component_sum_odor_name in odors:
            # TODO refactor to use presence in `natural` rather than this
            # hardcode
            if odor_set == 'kiwi':
                assert order[-1] == 'kiwi'
                order = order[:-1]

            if odor_set == 'flyfood':
                assert order[-1] == 'fly food'
                order = order[:-1]

            order.append(component_sum_odor_name)
            if component_sum_diff_odor_name in odors:
                order.append(component_sum_diff_odor_name)

            if odor_set == 'kiwi':
                order.append('kiwi')

            if odor_set == 'flyfood':
                order.append('fly food')
        del odor_set

        ax_id = id(plt.gca())
        if ax_id in ax_id2order:
            prev_order = ax_id2order[ax_id]
            assert order == prev_order, \
                f'previous order: {prev_order}\ncurrent order: {order}'
        else:
            ax_id2order[ax_id] = order
        # was thinking maybe i could also check order == xticklabels here,
        # but seems they might not be set yet?

        # TODO comment explaining reason for this whole 2/3 args thing
        assert len(args) == 2 or len(args) == 3
        if len(args) == 3:
            # TODO what does this mean? how would the user do that?
            err_msg = ('fill in missing odors before calling, if using the '
                'third positional hue arg'
            )
            missing_odors = set(odors) - set(order)
            if len(missing_odors) > 0:
                raise ValueError(err_msg)

            hue_keys = args[2].values
            assert len(hue_keys) == len(odors)
            full_cartesian_len = len(set(odors)) * len(set(hue_keys))
            # TODO TODO TODO why do i have this check again? what does it
            # prevent, and in what circumstances might it be triggered?
            if len(odors) < full_cartesian_len:
                raise ValueError(err_msg)

        return plot_fn(*args, order=order, **kwargs)

    return ordered_plot_fn


# These control whether these odors are included in plots that average some
# quantity across all odors within values of odor_set (e.g. cell adaptation
# rates).
# TODO TODO TODO make sure these are always used where appropriate
# (cell adaptions, linearity, n_ro_hist, etc)
def drop_excluded_odors(df, solvent=True, real=False, mix=False,
    components=False):

    assert not (mix and components)

    odor_cols = ([c for c in df.columns if 'name' in c.lower()] + 
        [c for c in df.index.names if c is not None and 'name' in c.lower()]
    )
    odor_col = 'name1'
    if (odor_col not in df.columns or
        not (len(odor_cols) == 1 and set(odor_cols) == {odor_col})):
        raise NotImplementedError

    excluded = []
    if solvent:
        excluded.extend(list(u.solvents))

    if real:
        excluded.extend(list(u.natural))

    if mix:
        excluded.append('mix')
    elif components:
        for c in df[odor_col].unique():
            if c != 'mix':
                excluded.append(c)

    return df[~ df[odor_col].isin(excluded)]


def odor_facetgrids(df, plot_fn, xcol, ycol, xlabel, ylabel, title,
    by_fly=True, mix_facet=True, sharey=True, **plot_kwargs):
    gs = []
    for i, oset in enumerate(odor_set_order):
        os_df = df[df.odor_set == oset]

        odor_col = None
        for n in ('name1', 'name1_a'):
            if n in df.columns:
                odor_col = n
                break
        assert odor_col is not None

        col_order = odor_set2order[oset]
        if not mix_facet:
            col_order = [c for c in col_order if c != 'mix']

        if by_fly:
            g = sns.FacetGrid(os_df, col=odor_col, hue='fly_id',
                palette=fly_id_palette, sharex=False, sharey=sharey,
                col_wrap=cw, col_order=col_order
            )
        else:
            g = sns.FacetGrid(os_df, col=odor_col, sharex=False, sharey=sharey,
                col_wrap=cw, col_order=col_order
            )

        vargs = (xcol,)
        if ycol is not None:
            vargs += (ycol,)
        g.map(plot_fn, *vargs, **plot_kwargs)

        g.set_xlabels(xlabel)
        if ylabel is not None:
            g.set_ylabels(ylabel)

        g.set_titles('{col_name}')

        # TODO not sure why this wasn't necessary in
        # Adaptation of reliable responders/kiwi case (where again fly_ids 1
        # and 2 are missing data for pfo and kiwi, and presumably things are
        # mapped in a similar order to correlation case, but fly_ids 1 and 2
        # have their appropriate artists already in the legend...
        if plot_fn is sns.lineplot:
            g.add_legend(title=fly_id_legend_title)
        else:
            try:
                # TODO use this elsewhere if needed (it gets artists across all
                # axes, so if some fly_ids are missing from last axis plotted,
                # that doesn't cause the legend to have empty lines for those
                # fly ids)
                u.set_facetgrid_legend(g, title=fly_id_legend_title)

            except AttributeError as err:
                warnings.warn(f'Title: {title}\nOdor set: {oset}\n'
                    f'Error in set_facetgrid_legend:\n{err}\n'
                    'Defaulting to g.add_legend()'
                )
                g.add_legend(title=fly_id_legend_title)

        u.fix_facetgrid_axis_labels(g, shared_in_center=False)

        if xcol == 'repeat_num':
            xticks = np.arange(df.repeat_num.min(),
                df.repeat_num.max() + 1
            )
            for ax in g.axes.flat:
                ax.xaxis.set_ticks(xticks)

        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(title + f'\n\n{oset}')

        gs.append(g)

    # So that saving can be deferred until after any modifications to the plots.
    def save_fn(file_prefix, **kwargs):
        for g, oset in zip(gs, odor_set_order):
            savefigs(g.fig, file_prefix, odor_set=oset, **kwargs)

    return gs, save_fn


# TODO factor to util (would need to change implementation...)?
def add_odorset(df):
    df['odor_set'] = df.set_index(fly_keys + [rec_key]).index.map(
        rec_keys2odor_set
    )
    return df


def add_odor_id(df):
    # This adds an odor ID that is also unique across odor_set, since
    # 'mix' means different things for different values of odor_set.
    oid = 'odor_id'
    assert oid not in df.columns
    df[oid] = [os + ', ' + o for os, o in zip(df.odor_set, df.name1)]
    return df


# TODO TODO probably try to move as much of the plotting as possible to the loop
# below this (or move the few remaining single fly plots from there here?)
# want to be able to calulate responders w/ different thresholds easily though /
# compute linearity props, etc w/ diff params
def process_traces(df_pickle, fly2response_threshold):
    """Computes responders and does initial analyses on one recording's traces.

    Args:
    df_pickle (str): filename of pickled DataFrame with cell dF/F traces
    """
    verbose = True
    being_parallelized = not type(fly2response_threshold) is dict

    # TODO maybe factor if verbose: print(out_s); if being_parallelized: ...
    # to internal fn?
    if being_parallelized:
        verbose = False

    ret_dict = dict()
    out_strs = []

    # TODO also support case here pickle has a dict
    # (w/ this df behind a trace_df key & maybe other data, like PID, behind
    # something else?)
    df, _ = read_pickle(df_pickle)

    assert 'original_name1' in df.columns
    if not use_existing_abbrevs or 'name1' not in df.columns:
        on1_unique = df.original_name1.unique()
        o2n = {o: cu.odor2abbrev(o) for o in on1_unique}
        df.name1 = df.original_name1.map(o2n)

    odor_set = u.odorset_name(df)
    odor_order = [cu.odor2abbrev(o) for o in u.df_to_odor_order(df)]
    
    parts = df_pickle[:-2].split('_')[-4:]
    title = '/'.join([parts[0], parts[1], '_'.join(parts[2:])])
    del parts
    fname = odor_set.replace(' ','') + '_' + title.replace('/','_')
    if verbose:
        print(fname)

    if being_parallelized:
        out_strs.append(fname)

    fly_nums = df.fly_num.unique()
    assert len(fly_nums) == 1
    fly_num = fly_nums[0]
    dates = df.prep_date.unique()
    assert len(dates) == 1
    date = dates[0]
    fly_key = (date, fly_num)

    # old style title
    #title = odor_set.title() + ': ' + title
    try:
        fly_id = fly_keys2fly_id.loc[fly_key]
        title = f'Fly {fly_id}, {odor_set}'
    except KeyError:
        fly_id = None

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

    used_for_thresh = False
    if fix_ref_odor_response_fracs:
        #if fly_key not in fly2response_threshold:
        if ref_odor in scalar_response_criteria.index.unique(level='name1'):
            assert fly_key not in fly2response_threshold
            mean_zchange_response_thresh = np.percentile(
                scalar_response_criteria.loc[(ref_odor,)],
                100 - ref_response_percent
            )
            fly2response_threshold[fly_key] = mean_zchange_response_thresh
            # TODO maybe print for all flies in one place at the end?
            out_s = ('Calculated mean Z-scored response threshold, from '
                'reference response percentage of {:.1f} to {}: '
                '{:.2f}').format(
                ref_response_percent, ref_odor, mean_zchange_response_thresh
            )
            if verbose:
                print(out_s)

            if being_parallelized:
                out_strs.append(out_s)

            used_for_thresh = True
        else:
            # If data with ref_odor happens to finish processing after data
            # requiring that threshold reaches this point, we will need to
            # wait for the threshold to be computed from the data with the
            # ref_odor.
            if being_parallelized:
                total_seconds_slept = 0
                sleep_s = 0.5
                max_sleep_s = 20.0
                while fly_key not in fly2response_threshold:
                    time.sleep(sleep_s)
                    total_seconds_slept += sleep_s
                    if total_seconds_slept > max_sleep_s:
                        raise RuntimeError('worker waited too long '
                            f'(>{max_sleep_s:.0f}s waiting for fly_key in '
                            'fly2response_threshold'
                        )
            assert fly_key in fly2response_threshold

            unique_name1s = scalar_response_criteria.index.get_level_values(
                'name1').unique()
            assert ref_odor not in unique_name1s
            mean_zchange_response_thresh = fly2response_threshold[fly_key]

    thorimage_ids = df.thorimage_id.unique()
    assert len(thorimage_ids) == 1
    thorimage_id = thorimage_ids[0]
    rec_key = (pd.Timestamp(fly_key[0]), fly_key[1], thorimage_id)
    if rec_key in rejects:
        warnings.warn(f'skipping rest of analysis on {rec_key} because '
            'marked reject in tom_data_ledger'
        )
        # TODO only do this if some stuff for fly is not dropped
        if used_for_thresh:
            warnings.warn('AND this recording was used to set the threshold'
                ' for this fly!'
            )

        if being_parallelized:
            # So there is always a consistent number of returned arguments in
            # the `being_parallelized` case.
            return None, None, None, None
        else:
            return None

    del used_for_thresh

    # Since we also excluded rejects when assigning fly_ids, need to check
    # that as long as a fly wasn't a reject, it still has fly_id defined.
    assert fly_id is not None

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
        # TODO check this still looks good in pdf and everything,
        # now that title is two lines
        thr_ax.set_title(title + '\nResponse threshold sensitivity')
        thr_ax.set_xlabel('Mean Z-scored response threshold')
        thr_ax.set_ylabel('Fraction of cells responding (across all odors)')

        thr_ax.axvline(x=mean_zchange_response_thresh, color='gray',
            linestyle='--', label='Response threshold')
        
        # TODO maybe pick thresh from some kind of max of derivative (to
        # find an elbow)?
        # TODO could also use on one / a few flies for tuning, then disable?
        out_s = savefigs(thr_fig, 'threshold_sensitivity', fname,
            section='Threshold sensitivity'
        )
        out_strs.append(out_s)

    # TODO not sure why it seems i need such a high threshold here, to get
    # reasonable sparseness... was the fly i'm testing it with just super
    # responsive?
    trial_responders = \
        scalar_response_criteria >= mean_zchange_response_thresh

    out_s = ('Fraction of cells trials counted as responses (mean z-scored '
        'dff > {:.2f}): {:.2f}').format(mean_zchange_response_thresh,
        trial_responders.sum() / len(trial_responders)
    )
    if verbose:
        print(out_s)
    if being_parallelized:
        out_strs.append(out_s)

    # TODO delete after getting refactoring in to process_traces to work.
    # commented to show original position.
    #responder_sers.append(add_metadata(df, trial_responders))
    trial_responders = add_metadata(df, trial_responders)
    ret_dict['responder_ser'] = trial_responders

    # TODO deal w/ case where there are only 2 repeats (shouldn't the number
    # be increased by fraction expected to respond if given a 3rd trial?)
    # ...or just get better data and ignore probably
    # As >= 50% response to odor criteria in Shen paper
    reliable_responders = \
        trial_responders.groupby(['name1','cell']).sum() >= 2

    if print_reliable_responder_frac:
        out_s = ('Mean fraction of cells responding to at least 2/3 trials:'
            ' {:.3f}').format(
            reliable_responders.sum() / len(reliable_responders)
        )
        if verbose:
            print(out_s)
        if being_parallelized:
            out_strs.append(out_s)

    n_cells = len(df.cell.unique())
    frac_odor_trial_responders = trial_responders.groupby(
        ['name1','repeat_num']).sum() / n_cells

    # TODO TODO TODO just print all of these things below anyway
    # (in the loop over the outputs)

    if print_responder_frac_by_trial:
        # TODO maybe just set float output format once up top
        # (like i could w/ numpy in thing i was working w/ Han on)?
        out_s = ('Fraction of cells responding to each trial of each odor:\n' +
            frac_odor_trial_responders.to_string(float_format='%.3f')
        )
        # TODO TODO TODO is the population becoming silent to kiwi more
        # quickly? is that consistent?
        if verbose:
            print(out_s)
        if being_parallelized:
            out_strs.append(out_s)

    if print_mean_responder_frac:
        mean_frac_odor_responders = \
            frac_odor_trial_responders.groupby('name1').mean()

        out_s = ('Mean fraction of cells responding to each odor:' +
            mean_frac_odor_responders.to_string(float_format='%.3f')
        )
        if verbose:
            print(out_s)
        if being_parallelized:
            out_strs.append(out_s)

    # The shuffling with 'cell' is just so it is the last level in the
    # index.
    window_by_trial = window_df.groupby([c for c in cell_cols if c != 'cell'
        ] + ['order','cell'])['df_over_f']

    window_trial_stats = window_by_trial.agg(trial_stat)
    # TODO delete after getting refactoring in to process_traces to work.
    # commented to show original position.
    #response_magnitude_sers.append(add_metadata(df, window_trial_stats))
    window_trial_stats = add_metadata(df, window_trial_stats)
    ret_dict['response_magnitude_ser'] = window_trial_stats

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
            trial_by_cell_means
        )
        odor_corrs_from_means = trial_by_cell_means.corr()

        window_trial_maxes = window_by_trial.max()
        trial_by_cell_maxes = window_trial_maxes.to_frame().pivot_table(
            index='cell', columns=within_recording_stim_cols,
            values='df_over_f'
        )
        trial_by_cell_maxes = u.add_missing_odor_cols(df,
            trial_by_cell_maxes
        )
        odor_corrs_from_maxes = trial_by_cell_maxes.corr()

        # TODO probably move outside of process_traces?
        if plot_correlations:
            title_suffix = '\n' + title
            # TODO TODO and are there other plots / outputs that will be
            # affected by missing odors?
            # TODO exclude 3/4 of these from latex / don't compute at all
            if trial_order_correlations:
                porder_corr_mean_fig = u.plot_odor_corrs(
                    odor_corrs_from_means, title_suffix=title_suffix
                )
                out_s = savefigs(porder_corr_mean_fig, 'porder_corr_mean',
                    fname, exclude_from_latex=True
                )
                out_strs.append(out_s)

                porder_corr_max_fig = u.plot_odor_corrs(
                    odor_corrs_from_maxes, trial_stat='max',
                    title_suffix=title_suffix
                )
                out_s = savefigs(porder_corr_max_fig, 'porder_corr_max', fname,
                    exclude_from_latex=True
                )
                out_strs.append(out_s)

            if odor_order_correlations:
                oorder_corr_mean_fig = u.plot_odor_corrs(
                    odor_corrs_from_means, odors_in_order=odor_order,
                    title_suffix=title_suffix
                )
                out_s = savefigs(oorder_corr_mean_fig, 'oorder_corr_mean',
                    fname, exclude_from_latex=True
                )
                out_strs.append(out_s)

                oorder_corr_max_fig = u.plot_odor_corrs(
                    odor_corrs_from_maxes, trial_stat='max',
                    odors_in_order=odor_order,
                    title_suffix=title_suffix
                )
                # TODO TODO crop as much as possible from this (and other
                # above) (w/ tight_layout??), so that they can be fit
                # side-by-side in PDF w/ minimum amount of whitespace
                # between!
                out_s = savefigs(oorder_corr_max_fig, 'oorder_corr_max', fname,
                    section='Trial-max response correlations',
                    # A large number to put this at the end.
                    section_order=200
                )
                out_strs.append(out_s)

        tidy_corrs_from_means = melt_symmetric(odor_corrs_from_means,
            name='corr'
        )
        # TODO delete after getting refactoring in to process_traces to work.
        # commented to show original position.
        #correlation_dfs_from_means.append(
        #    add_metadata(df, tidy_corrs_from_means)
        #)
        ret_dict['correlation_ser_from_means'] = \
            add_metadata(df, tidy_corrs_from_means)

        tidy_corrs_from_maxes = melt_symmetric(odor_corrs_from_maxes,
            name='corr'
        )

        # TODO delete after getting refactoring in to process_traces to work.
        # commented to show original position.
        #correlation_dfs_from_maxes.append(
        #    add_metadata(df, tidy_corrs_from_maxes)
        #)
        ret_dict['correlation_ser_from_maxes'] = \
            add_metadata(df, tidy_corrs_from_maxes)

    if do_roc:
        # TODO TODO TODO maybe also return matrix of each pair of odors
        # that produced one auc value for (task, odor) pairs
        # (presumably those were averaged over to produce that auc value?)
        auc_df = roc_analysis(window_trial_stats, reliable_responders,
            fname=fname
        )
        auc_df.set_index(['task','odor','cell'], inplace=True)
        # TODO delete after getting refactoring in to process_traces to work.
        # commented to show original position.
        #auc_dfs.append(add_metadata(df, auc_df))
        ret_dict['auc_df'] = add_metadata(df, auc_df)

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
        pca_df = plot_pca(pivoted_window_trial_stats, fname=fname)
        pca_df.set_index('component_number', inplace=True)
        # TODO delete after getting refactoring in to process_traces to work.
        # commented to show original position.
        #pca_dfs.append(add_metadata(df, pca_df))
        ret_dict['pca_df'] = add_metadata(df, pca_df)

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

        cbar_label = trial_stat.title() + ' response ' + u.dff_latex

        odor_labels = u.matlabels(trial_by_cell_stats_top, u.format_mixture)
        # TODO fix x/y in this fn... seems T required
        f1 = u.matshow(trial_by_cell_stats_top.T, xticklabels=odor_labels,
            group_ticklabels=True, colorbar_label=cbar_label, fontsize=6,
            title=title
        )
        ax = plt.gca()
        ax.set_aspect(0.1)
        out_s = savefigs(f1, 'trials', fname)
        out_strs.append(out_s)

    odor_cell_stats = trial_by_cell_stats.groupby('name1').mean()

    if do_linearity_analysis:
        # TODO TODO factor linearity checking in kc_analysis to use this,
        # since A+B there is pretty much a subset of this case
        # (-> hong2p.util, both use that?)

        component_names = [x for x in odor_cell_stats.index
            if x not in (('mix',) + u.solvents + u.natural)
        ] 
        # TODO TODO also do on traces as in kc_analysis?
        # or at least per-trial(?) rather than per-mean?

        mix = odor_cell_stats.loc['mix']
        components = odor_cell_stats.loc[component_names]
        component_sum = components.sum()
        assert mix.shape == component_sum.shape

        simple_sum = component_sum
        simple_mix_diff = simple_sum - mix

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

        scaled_mix_diff = scaled_sum - mix
        scaled_sum_residual = np.linalg.norm(scaled_mix_diff)**2

        epsilon = 0.1
        r_plus = np.linalg.norm(scaled_sum * (1 + epsilon) - mix)**2
        assert scaled_sum_residual < r_plus, \
            'scaled sum fit worse than larger scale'

        r_minus = np.linalg.norm(scaled_sum * (1 - epsilon) - mix)**2
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
        weighted_mix_diff = weighted_sum - mix

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
            'scaled_mix_diff': scaled_mix_diff,
            'simple_sum': simple_sum,
            'simple_mix_diff': simple_mix_diff
        })
        # TODO delete after getting refactoring in to process_traces to work.
        # commented to show original position.
        #linearity_sers.append(add_metadata(df, linearity_ser))
        #linearity_odor_dfs.append(add_metadata(df, linearity_odor_df))
        #linearity_cell_dfs.append(add_metadata(df, linearity_cell_df))
        ret_dict['linearity_ser'] = add_metadata(df, linearity_ser)
        ret_dict['linearity_odor_df'] = add_metadata(df, linearity_odor_df)
        ret_dict['linearity_cell_df'] = add_metadata(df, linearity_cell_df)

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
                rotation=title_rotation
            )

        ax = diff_axs[0, 1]
        # TODO TODO appropriate vmin / vmax here?
        # [-1, 1] or maybe [-1.5, 1.5] would seem to work ok..?
        # TODO share a colorbar between the two difference plots?
        # (if fixed range, would seem reasonable)
        mat = matshow(ax, scaled_mix_diff[order[:top_n]],
            aspect=aspect_one_col, cmap='coolwarm'
        )
        #mat = matshow(ax, scaled_mix_diff,
        #    extent=[xmin,xmax,ymin,ymax], aspect='auto', cmap='coolwarm'
        #)
        # TODO why this not seem to be working?
        diff_fig.colorbar(mat, ax=ax)
        if titles:
            ax.set_title('Mixture response - scaled sum',
                rotation=title_rotation
            )

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
            aspect=aspect_one_col, cmap='coolwarm'
        )
        diff_fig.colorbar(mat2, ax=ax)
        if titles:
            ax.set_title('Mixture response - weighted sum',
                rotation=title_rotation
            )

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

    cbar_label = 'Mean ' + trial_stat + ' response ' + u.dff_latex
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
        out_s = savefigs(f2, 'avg', fname)
        out_strs.append(out_s)

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

        out_s = '\n'.join([
            f'median frame_delta: {frame_delta}',
            f'plot_ds_factor: {plot_ds_factor}',
            f'plotting_td: {plotting_td}'
        ])
        if verbose:
            print(out_s)

        if being_parallelized:
            out_strs.append(out_s)

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
            color='black', alpha=0.7
        )
        #, linewidth=0.5)
        #, hue='repeat_num')
        #print('plotting downsampled took {:.3f}s'.format(
        #    time.time() - start))
        g.set_titles('{col_name}')
        g.fig.suptitle('Average trace across all cells, ' +
            title[0].lower() + title[1:]
        )

        g.axes[0,0].set_xlabel('Seconds from onset')
        g.axes[0,0].set_ylabel('Mean ' + u.dff_latex)
        for a in g.axes.flat[1:]:
            a.axis('off')

        g.fig.subplots_adjust(top=0.9, left=0.05)
        out_s = savefigs(g.fig, 'avg_traces', fname)
        out_strs.append(out_s)

    # TODO TODO TODO plot pid too

    for odor in odor_cell_stats.index:
        # TODO maybe put all sort orders in one plot as subplots?
        order = \
            odor_cell_stats.loc[odor, :].sort_values(ascending=False).index

        sort_odor_labels = [o + ' (sorted)' if o == odor else o
            for o in odor_labels
        ]
        ss = '_{}_sorted'.format(odor)

        # TODO TODO TODO here and in plots that also have fits, show (in
        # general, nearest? est? scalar magnitude of one of these?) eag +
        # in roi / full frame MB fluorescence under each column?
        if odor_matrices:
            odor_cell_stats_top = odor_cell_stats.loc[:, order[:top_n]]
            fs = u.matshow(odor_cell_stats_top.T,
                xticklabels=sort_odor_labels, colorbar_label=cbar_label,
                fontsize=6, title=title
            )
            ax = plt.gca()
            ax.set_aspect(0.1)
            # TODO support? this will prob not work w/ paired inference in
            # savefig.
            # TODO would need to at least check that pairing in report
            # generation also works w/ arbitrary extra parts (beyond
            # glob str, maybe), to also pair on the odor here
            out_s = savefigs(fs, 'avg', fname + ss, exclude_from_latex=True)
            out_strs.append(out_s)

        if odor_and_fit_matrices:
            odor_and_fit_plot(odor_cell_stats, weighted_sum, order[:top_n],
                fname + ss, title, sort_odor_labels, cbar_label
            )

    if verbose:
        print('')

    if being_parallelized:
        out_str = '\n'.join([s for s in out_strs if len(s) > 0] + [''])
        return ret_dict, plot_prefix2latex_data, plots_made_this_run, out_str
    else:
        return ret_dict


pickle_outputs_dir = 'output_pickles'
if not exists(pickle_outputs_dir):
    os.mkdir(pickle_outputs_dir)

pickle_outputs_fstr = join(pickle_outputs_dir,
    'kc_mix_analysis_outputs_meanzthr{:.2f}{}.p'
)

if not args.only_analyze_cached:
    # TODO use a longer/shorter baseline window?
    baseline_start = -2
    baseline_end = 0
    response_start = 0
    response_calling_s = 5.0
    trial_stat = 'max'

    gsheet_link = u.gsheet_csv_export_link('tom_data_ledger_sheet_link.txt')
    # TODO maybe factor this into a "single_sheet" kwarg to above?
    # Adding gid of default sheet.
    gsheet_link += '0'
    gsheet_df = pd.read_csv(gsheet_link)
    gsheet_df.date = pd.to_datetime(gsheet_df.date)
    gsheet_df = gsheet_df.iloc[:gsheet_df.date.isna().idxmax()]
    gsheet_df.fly_num = gsheet_df.fly_num.astype(np.uint16)
    rejects = {tuple(x) for _, x in gsheet_df.loc[gsheet_df['reject?'],
        ['date','fly_num','thorimage_id']].iterrows()
    }

    # TODO TODO update loop to use this df, and get index values directly from
    # there, rather than re-calculating them
    latest_pickles = u.latest_trace_pickles()
    pickles = list(latest_pickles.trace_pickle_path)
    if test:
        warnings.warn('Only reading two pickles for testing! '
            'Set test = False to analyze all.'
        )
        pickles = [p for p in pickles if test_substr in p]

    b = time.time()
    print('reading all pickles to decide their order... ', end='', flush=True)
    pickles = order_by_odor_sets(pickles)
    print('done ({:.2f}s)'.format(time.time() - b), flush=True)

    # This filters out data that did not have enough odor sets.
    latest_pickles = \
        latest_pickles[latest_pickles.trace_pickle_path.isin(pickles)]
    latest_pickles.index.set_names(['prep_date', 'fly_num', 'thorimage_id'],
        inplace=True
    )

    # And this filters out data manually marked as a reject
    # (in tom_data_ledger gsheet)
    latest_pickles = latest_pickles[~ latest_pickles.index.isin(rejects)]

    latest_pickles = u.add_fly_id(latest_pickles)
    keys = ['prep_date', 'fly_num', 'fly_id']
    # TODO TODO save this and assert it's equal to stuff from generating
    # fly_id after load
    fly_keys2fly_id = latest_pickles.reset_index()[keys].drop_duplicates(
        ).set_index(keys[:2], verify_integrity=True).fly_id
    fly_keys2fly_id.sort_index(inplace=True)
    del keys

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

    if fit_matrices or odor_and_fit_matrices:
        do_linearity_analysis = True
    else:
        do_linearity_analysis = False

    if fix_ref_odor_response_fracs:
        # (date, fly_num) -> threshold
        fly2response_threshold = dict()
    else:
        ref_odor = None
        ref_response_percent = None

    # TODO maybe assert all outputs are same running w/ mp vs in loop here?
    # (pdfs / pngs & cached stuff) (in case it's possible diff scope causes
    # differences? is it though?) (and ideally compare to output before
    # factoring loop contents to fn above / argument for why output would
    # not be expected to change ...which it shouldn't, right?)
    '''
    ret_dicts = []
    for df_pickle in pickles:
        ret_dicts.append(process_traces(df_pickle))
    '''

    # TODO some existing solution to output stdout output of workers in order
    # after map finishes? explore trying to make one w/ redirecting stdout,
    # like: https://stackoverflow.com/questions/5884517 at some point
    # TODO TODO TODO probably refactor so rejection happens later?
    # so that correlations of rejected flies can still be saved / averaged over
    # later?
    parallel_process_traces = not args.no_parallel_process_traces
    if parallel_process_traces:
        with mp.Manager() as manager:
            # TODO also handle case where fix_ref_odor_response_fracs == False!
            # (or delete code supporting that case)
            fly2response_threshold = manager.dict()
            # The multiprocessing docs say this is the default value if the
            # `processes` kwarg is not specified in the Pool constructor.
            print(f'Processing traces with {os.cpu_count()} workers')
            # TODO maybe make calls to multiprocessing stuff that allows me
            # to explicitly specify ordering? in case waiting for some ref_odor
            # data inside calls is making some things take too long. could maybe
            # order to avoid / minimize any need for waiting.
            with manager.Pool() as pool:
                ret = pool.starmap(
                    process_traces, product(pickles, [fly2response_threshold])
                )
        assert len(ret) == len(pickles)

        # Note, as the second two elements here are just worker variables that
        # are returned, they can grow across across calls to process_traces
        # within a particular worker.
        # "Zip is its own inverse" https://stackoverflow.com/questions/12974474
        ret_dicts, plot_pfx2ltx_data_list, plots_made_list, out_strs = zip(*ret)

        plot_pfx2ltx_data_list = [x for x in plot_pfx2ltx_data_list if x]
        assert all([
            x == plot_pfx2ltx_data_list[0] for x in plot_pfx2ltx_data_list
        ])
        plot_prefix2latex_data = plot_pfx2ltx_data_list[0]
        del plot_pfx2ltx_data_list

        plots_made_list = [x for x in plots_made_list if x]

        plots_made_this_run = set().union(*plots_made_list)
        # A crude check that would catch some ways different plot_traces
        # calls might make different numbers of plots.
        assert len(plots_made_this_run) % len(plots_made_list) == 0
        del plots_made_list

        for out_str in out_strs:
            if out_str is not None:
                print(out_str)
        del out_strs
    else:
        ret_dicts = list(starmap(
            process_traces, product(pickles, [fly2response_threshold])
        ))

    # This filters out any recordings that were used to set some threshold, but
    # were otherwise intentionally skipped from the rest of the analysis.
    ret_dicts = [x for x in ret_dicts if x is not None]

    responder_sers = [x['responder_ser'] for x in ret_dicts]
    responders = pd.concat(responder_sers)

    response_magnitude_sers = [x['response_magnitude_ser'] for x in ret_dicts]
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
        correlation_sers_from_means = [
            x['correlation_ser_from_means'] for x in ret_dicts
        ]
        corr_ser_from_means = pd.concat(correlation_sers_from_means)

        correlation_sers_from_maxes = [
            x['correlation_ser_from_maxes'] for x in ret_dicts
        ]
        corr_ser_from_maxes = pd.concat(correlation_sers_from_maxes)
    else:
        corr_ser_from_means = None
        corr_ser_from_maxes = None

    if do_roc:
        auc_dfs = [x['auc_df'] for x in ret_dicts]
        auc_df = pd.concat(auc_dfs)
    else:
        auc_df = None

    if do_pca:
        pca_dfs = [x['pca_df'] for x in ret_dicts]
        pca_df = pd.concat(pca_dfs)
    else:
        pca_df = None

    if do_linearity_analysis:
        linearity_sers = [x['linearity_ser'] for x in ret_dicts]
        linearity_df = pd.concat(linearity_sers).unstack()

        linearity_odor_dfs = [x['linearity_odor_df'] for x in ret_dicts]
        linearity_odor_df = pd.concat(linearity_odor_dfs)

        linearity_cell_dfs = [x['linearity_cell_df'] for x in ret_dicts]
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
            ref_response_percent, ref_odor
        )
        # Because although it will have a value, it changes across flies in this
        # cases. This is to avoid confusion in things using these outputs.
        mean_zchange_response_thresh = None
    else:
        pickle_outputs_name = pickle_outputs_fstr.format(
            mean_zchange_response_thresh, ''
        )

    print(f'Writing computed outputs to {pickle_outputs_name}')
    # TODO TODO maybe save code version into df too? but diff name than one
    # below, to be clear about which code did which parts of the analysis?
    # TODO TODO or maybe equally important, i might want to save (input trace)
    # pickles here, so i know all input data that went into final outputs.
    # could then include both inputs (& maybe code versions) in latex report
    with open(pickle_outputs_name, 'wb') as f:
        _locals = locals()
        data = {n: _locals[n] for n in param_names}
        data.update({
            # TODO maybe also need to save hashes / full figs, to check that
            # what exists on disk under the same names are actually what these
            # filenames refer to?
            'plots_made_this_run': plots_made_this_run,
            'plot_prefix2latex_data': plot_prefix2latex_data,
            'fly_keys2fly_id': fly_keys2fly_id,

            'responders': responders,
            'response_magnitudes': response_magnitudes,

            'corr_ser_from_means': corr_ser_from_means,
            'corr_ser_from_maxes': corr_ser_from_maxes,

            'auc_df': auc_df,

            'pca_df': pca_df,

            'linearity_df': linearity_df,
            'linearity_odor_df': linearity_odor_df,
            'linearity_cell_df': linearity_cell_df
        })
        pickle.dump(data, f)

    after_raw_calc_time_s = time.time()
    print('Loading and processing traces took {:.0f}s'.format(
        after_raw_calc_time_s - start_time_s
    ))

else:
    after_raw_calc_time_s = time.time()

    load_most_recent = True
    if load_most_recent:
        out_pickles = glob.glob(pickle_outputs_fstr.replace('{:.2f}{}','*'))
        if len(out_pickles) == 0:
            raise IOError(f'no pickles under {pickle_outputs_dir}! re-run '
                'without -c option.'
            )

        out_pickles.sort(key=getmtime)
        pickle_outputs_name = out_pickles[-1]
    else:
        # Ref odor frac if using desired_ref_odor.
        desired_thr = 2.5
        # '' if loading outputs where fix_ref_odor_response_fracs was False.
        desired_ref_odor = ''
        pickle_outputs_name = pickle_outputs_fstr.format(desired_thr,
            desired_ref_odor
        )

    print(f'Loading computed outputs from {pickle_outputs_name}')
    with open(pickle_outputs_name, 'rb') as f:
        # TODO revert after debugging
        data = pickle.load(f)
        #data = u.unpickler_load(f)

    expected_params_found = []
    expected_params_missing = []
    for n in param_names:
        if n in data:
            expected_params_found.append(n)
        else:
            expected_params_missing.append(n)

    if len(expected_params_missing) > 0:
        print('\nFound the following expected cache parameters:')
        pprint(expected_params_found)

        print('\nMissing the following expected cache parameters:')
        pprint(expected_params_missing)

        import ipdb; ipdb.set_trace()
        raise ValueError(f'data loaded from {pickle_outputs_name} was missing'
            ' some expected parameters that could affect how the traces were '
            'processed to produce the intermediates in the cache. try '
            'recomputing cache contents and see above.'
        )

    # Modifying globals rather than locals, because at least globals
    # would still (sort-of) work if this got refactored into a function,
    # and the docs explicitly warn not to modify what locals() returns.
    # See: https://stackoverflow.com/questions/2597278 for discussion about
    # alternatives.
    # Doing this so I don't have to repeat which variables specified in pickle
    # saving above / so variables loaded can't diverge (assuming pickle was
    # generated with most recent version...).
    globals().update(data)

# TODO maybe define each downstream analysis in its own script file?
# (would allow ordering in report by which analysis was worked on most recently
# + maybe cleaner [dis/en]abling of different portions)
# (run everything following some naming convention / in some dir?)
# TODO TODO and best of all, this might facilitate only recalculating stuff
# for which the analysis has changed (storing hashes of files / mtimes (/ hashes
# of python ast/bytecode?)

n_flies = len(responders.index.to_frame()[fly_keys].drop_duplicates())
fly_colors = sns.color_palette('hls', n_flies)

odor_set_odors_nosolvent = [os - set(u.solvents) for os in odor_set_odors]

# TODO fix cause of warnings that first plt call generates, e.g.:
# Gtk-Message: ...: Failed to load module ["overlay-scrollbar"/...]

# TODO delete? not sure i ever want this True...
# Whether to include data from presentations of actual kiwi.
n_ro_include_real = False
if n_ro_include_real:
    n_ro_exclude = {}
else:
    n_ro_exclude = set(u.natural)

# Only the last bin includes both ends. All other bins only include the
# value at their left edge. So with n_odors + 2, the last bin will be
# [n_odors, n_odors + 1] (since arange doesn't include end), and will thus
# only count cells that respond (reliably) to all odors, since it is not
# possible for them to respond to > n_odors.
n_max_odors_per_panel = max(
    [len(os - n_ro_exclude) for os in odor_set_odors_nosolvent]
)
# TODO maybe this should be + 1 then, since n odors used to be 6, even though
# kiwi (may?) have been included?
n_ro_bins = np.arange(n_max_odors_per_panel + 2)
del n_max_odors_per_panel

# Needed to also exclude this data in loop.
for s in u.solvents:
    n_ro_exclude.add(s)

rec_keys2odor_set = dict()

np.random.seed(1337)
# TODO try increasing to 1000 if it works
n_shuffles = 100

# TODO TODO TODO maybe shuffle after excluding non-(reliable)responders first?

frac_responder_sers = []
# n_ros = [N]umber of (reliably) [R]esponded [O]dors (per cell)
n_ros_sers = []
shuffle_n_ros_sers = []
for i, (fly_gn, fly_gser) in enumerate(responders.groupby(fly_keys)):
    fly_color = fly_colors[i]

    assert len(fly_gser.index.get_level_values(rec_key).unique()) in {1,2,3}

    # TODO just add variable for odorset earlier, so that i can group on that
    # instead of (/in addition to) thorimage id, so i can loop over them in a
    # fixed order (? order doesn't matter here, does it?)
    for rec_gn, rec_gser in fly_gser.groupby(rec_key):
        # TODO maybe refactor this (kinda duped w/ odorset finding from the more
        # raw dfs)? (now there should be some util fns that accomplish this.
        # use those!!!)
        unique_odors = rec_gser.index.get_level_values('name1')
        if 'eb' in unique_odors:
            odorset = 'kiwi'
        elif '1o3ol' in unique_odors:
            odorset = 'control'
        elif 'aa' in unique_odors:
            odorset = 'flyfood'
        else:
            raise ValueError('odor set not recognized')
        del unique_odors

        # TODO still needed?
        rec_keys2odor_set[fly_gn + (rec_gn,)] = odorset

        # old style title
        old_title = (odorset + ', ' +
            fly_gn[0].strftime(u.date_fmt_str) + '/' + str(fly_gn[1])
        )
        fname = old_title.replace(',','').replace(' ','_').replace('/','_')
        del old_title

        fly_id = fly_keys2fly_id.loc[fly_gn]
        title = f'Fly {fly_id}, {odorset}'

        trial_responders = rec_gser
        # TODO probably delete this. i want to be able to plot pfo + kiwi
        # response fractions / reliable response fraction in one set of plots.
        # at least save as a redefinition for below.
        #trial_responders = rec_gser[~ rec_gser.index.get_level_values('name1'
        #    ).isin(('pfo', 'kiwi', 'water', 'fly food'))]

        n_rec_cells = \
            len(trial_responders.index.get_level_values('cell').unique())

        # TODO TODO dedupe this section with above (just do here?)
        ########################################################################
        # TODO deal w/ case where there are only 2 repeats (shouldn't the number
        # be increased by fraction expected to respond if given a 3rd trial?)
        # ...or just get better data and ignore probably
        # As >= 50% response to odor criteria in Shen paper
        n_trials_responded_to = trial_responders.groupby(['name1','cell']).sum()

        assert (trial_responders.index.get_level_values('repeat_num').max() + 1
            == n_trials_responded_to.max()
        )
        reliable_responders = n_trials_responded_to >= 2
        reliable_responders.name = 'reliable_responder'

        frac_reliable_responders = \
            reliable_responders.groupby('name1').sum() / n_rec_cells
        check_fraction_series(frac_reliable_responders)

        # TODO try to consolidate this w/ odor_order (they should have
        # the same stuff, just one also has order info, right?)
        odors = [x for x in
            trial_responders.index.get_level_values('name1').unique()
        ]
        full_odor_order = [cu.odor2abbrev(o) for o in u.odor_set2order[odorset]]
        seen_odors = set()
        odor_order = []
        for o in full_odor_order:
            if o not in seen_odors and o in odors:
                seen_odors.add(o)
                odor_order.append(o)
        del seen_odors, odors

        # TODO TODO TODO TODO check that we are not getting wrong results by
        # dividing by 3 anywhere when the odor is actually only recorded twice
        # (count unique repeat_num?) (check above too!) (i think i might have
        # handled it correctly here...)
        n_odor_repeats = trial_responders.reset_index().groupby('name1'
            ).repeat_num.nunique()

        frac_responders = (trial_responders.groupby('name1').sum() /
            (n_odor_repeats * n_rec_cells))[odor_order]
        check_fraction_series(frac_responders)

        reliable_of_responder_barplots = False
        if not args.across_flies_only and reliable_of_responder_barplots:
            responded_at_all = n_trials_responded_to >= 1
            # These are both per-odor.
            n_cells_responded_at_all = responded_at_all.groupby('name1').sum()
            del responded_at_all
            n_cells_reliable = reliable_responders.groupby('name1').sum()

            reliable_of_resp_fig = plt.figure()
            rel_of_resp_frac = n_cells_reliable / n_cells_responded_at_all
            del n_cells_reliable, n_cells_responded_at_all

            # TODO is there anywhere else that i make a similar mistake?
            # This was very wrong.
            #rel_of_resp_frac = frac_reliable_responders / frac_responders

            check_fraction_series(rel_of_resp_frac)

            reliable_of_resp_ax = rel_of_resp_frac[odor_order].plot.bar(
                color='black'
            )
            # TODO TODO maybe at least color text of odors missing any
            # presentations red or something, if not gonna try to fix those
            # values. / otherwise mark
            # TODO TODO TODO maybe use just SEM / similar of full response /
            # magnitudes as another measure. might have more info + less
            # discretization noise.
            reliable_of_resp_ax.set_title(title +
                '\nFraction of responders that are reliable, by odor'
            )
            reliable_of_resp_ax.set_xlabel('')
            savefigs(reliable_of_resp_fig, 'reliable_of_resp', fname,
                section='Response reliability'
            )

        def add_rec_metadata(ser_or_df):
            ser_or_df = add_metadata(rec_gser.reset_index(), ser_or_df)
            return pd.concat([ser_or_df], names=['odor_set'],
                keys=[odorset]
            )

        frac_responders = add_rec_metadata(frac_responders)
        frac_responder_sers.append(frac_responders)

 
        of_mix_reliable_to_others_barplot = False
        of_mix_rel_to_others_ratio_barplt = False

        if (of_mix_reliable_to_others_barplot or
            of_mix_rel_to_others_ratio_barplt):

            odors = [o for o in odors
                if o not in (u.solvents + u.natural)
            ]
            odor_order = [o for o in odor_order
                if o not in (u.solvents + u.natural)
            ]
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
                resp_fracs_to_others = trial_responders.loc[:, :, :,
                    other_odors, :, :, cells
                    ].groupby('name1').sum() / (n_odor_cells * 3)

                # TODO revisit (something more clear?)
                resp_fracs_to_others.name = 'of_' + odor + '_resp'

                odor_resp_subset_fracs_list.append(resp_fracs_to_others)
                '''
                norm_frac_reliable_to_others = False
                fracs_reliable_to_others = reliable_responders.loc[other_odors,
                    cells].groupby('name1').sum() / n_odor_cells

                if norm_frac_reliable_to_others:
                    fracs_reliable_to_others /= frac_reliable_responders
                    # TODO TODO TODO does the weird value of the (mix,
                    # of_mix_resp) (and other identities) mean that this
                    # normalization is not meaningful? should i be doing
                    # something differently?

                    # TODO maybe in the normed ones, just don't show mix thing,
                    # since that value seems weird?

                fracs_reliable_to_others.name = 'of_' + odor + '_resp'

                odor_resp_subset_fracs_list.append(fracs_reliable_to_others)
            del odors

            odor_resp_subset_fracs = pd.concat(odor_resp_subset_fracs_list,
                axis=1, sort=False
            )
            del odor_resp_subset_fracs_list

            of_mix_reliable_to_others = \
                odor_resp_subset_fracs['of_mix_resp'][odor_order]
            del odor_resp_subset_fracs

            check_fraction_series(of_mix_reliable_to_others)

            if not args.across_flies_only and of_mix_reliable_to_others_barplot:
                fig = plt.figure()
                ax = of_mix_reliable_to_others.plot.bar(color='black')
                ax.set_title(title + '\nFraction of mix reliable responders'
                    ' reliable to other odors'
                )
                ax.set_xlabel('')
                savefigs(fig, 'mix_rel_to_others', fname,
                    section='Mix responder tuning'
                )

            # TODO maybe figure out how to compute something like this
            # in the across fly stuff below (if correct and meaningful...).
            # then delete this section.
            '''
            if not args.across_flies_only and of_mix_rel_to_others_ratio_barplt:
                of_mix_reliable_to_others_ratio = \
                    of_mix_reliable_to_others / frac_reliable_responders

                # TODO see note above. excluding mix disingenuous?
                of_mix_reliable_to_others_ratio = \
                    of_mix_reliable_to_others_ratio[[o for o in odor_order
                    if o != 'mix']]
                assert 'mix' not in of_mix_reliable_to_others_ratio.index

                ratio_fig = plt.figure()
                ratio_ax = \
                    of_mix_reliable_to_others_ratio.plot.bar(color='black')

                # TODO update title on these to try to clarify how the value is
                # computed
                ratio_ax.set_title(title + '\nFraction of mix reliable'
                    ' responders reliable to other odors (ratio)'
                )
                ratio_ax.set_xlabel('')
                savefigs(ratio_fig, 'ratio_mix_rel_to_others', fname)
            '''
        ########################################################################
        # end section to de-dupe w/ code in first loop

        rr_idx_names = reliable_responders.index.names
        rr = reliable_responders.reset_index()
        reliable_responders = rr[~ rr.name1.isin(n_ro_exclude)
            ].set_index(rr_idx_names)
        assert reliable_responders.shape[1] == 1
        reliable_responders = reliable_responders.iloc[:, 0]

        # TODO maybe try w/ any responses counting it as a responder for the
        # odor, instead of the 2 required for reliable_responders?
        rec_n_reliable_responders = reliable_responders.groupby('cell').sum()
        rec_n_reliable_responders.name = 'n_odors_reliable_to'
        assert rec_n_reliable_responders.notnull().all(), \
            'cant cast to int if any NaN'
        rec_n_reliable_responders = rec_n_reliable_responders.astype(np.uint16)

        n_ro_hist_odors = \
            reliable_responders.index.get_level_values('name1').unique()
        assert 'pfo' not in n_ro_hist_odors
        assert 'water' not in n_ro_hist_odors
        if not n_ro_include_real:
            assert 'kiwi' not in n_ro_hist_odors
            assert 'fly food' not in n_ro_hist_odors

        # Since last bin includes right edge, but beyond that stuff would not
        # have a bin.
        assert rec_n_reliable_responders.max() <= n_ro_bins[-1]

        # TODO TODO TODO double check myself, and with B, that this is actually
        # the shuffle we want to do
        # TODO TODO TODO maybe first want to subset boolean (odor, cell)
        # responder flags to only include cells that reliable responded to
        # anything??
        # TODO TODO TODO maybe keep number of times each cell is counted a
        # responder constant, sampling from something like the distribution of
        # how often each odor was responded to??? (so explicitly just changing
        # real data so it is now as if each odor WAS independent? (this
        # correct?))

        shuffle_indices = np.stack([
            np.random.permutation(len(reliable_responders)) for _
            in range(n_shuffles)
        ])
        # Transposing so first dimension will be length of reliable_responders
        # again, so the index from reliable_responders can be applied to it.
        shuffled_rrs = reliable_responders.values[shuffle_indices].T

        # Checking that different shuffles actually differ.
        if reliable_responders.sum() > 0:
            first_shuffle = shuffled_rrs[:, 0]
            shuffles_to_check = np.random.choice(np.arange(1, n_shuffles),
                size=5, replace=False
            )
            for i in shuffles_to_check:
                assert i != 0, 'next assertion would fail b/c identity'
                assert not np.array_equal(shuffled_rrs[:, i], first_shuffle), \
                    'shuffles did not differ. fix shuffling procedure.'

        shuffled_rrs = pd.DataFrame(shuffled_rrs,
            index=reliable_responders.index
        )
        shuffled_rrs.columns.name = 'shuffle_num'

        # TODO delete this section?
        shuffled_rr_sums = shuffled_rrs.sum()
        assert shuffled_rr_sums.shape == (n_shuffles,)
        unq_shuffled_rr_sums = shuffled_rr_sums.unique()
        assert len(unq_shuffled_rr_sums) == 1
        assert unq_shuffled_rr_sums[0] == reliable_responders.sum()
        #

        rec_shuffled_n_rrs = shuffled_rrs.groupby('cell').sum()
        assert rec_shuffled_n_rrs.notnull().all(axis=None), \
            'cant cast to int if any NaN'
        rec_shuffled_n_rrs = rec_shuffled_n_rrs.astype(np.uint16)

        # TODO provide CIs for the histogram value in each bin?
        # TODO and then plot w/ fill between? or save all hist vals and use
        # seaborn to plot after this loop?

        # TODO TODO how to aggregate these across flies (and how to
        # get CI on [mean-per-bin(?)]-shuffle-hist across flies at end, given
        # diff #s of cells per fly?)

        # TODO TODO KDE w/ weights based approach like in shuffle_fruits??
        # TODO TODO maybe still check that against mean-of-mean-hists +
        # bootstrapped mean there? (or mean of CIs from each fly? those
        # seem like they are quite different, though not sure which i want)

        # TODO TODO do i want to plot the mean hist from shuffle
        # and then bootstrap shuffled hists to compute CI of this mean??
        # or just plot percentiles of the histogram from the shuffles?
        # maybe at that point, plot median rather than mean hist?

        rec_shuffled_n_rrs = rec_shuffled_n_rrs.unstack()
        rec_shuffled_n_rrs.name = rec_n_reliable_responders.name

        n_ros_sers.append(add_rec_metadata(rec_n_reliable_responders))
        shuffle_n_ros_sers.append(add_rec_metadata(rec_shuffled_n_rrs))
del odor_order

# this is just the mean frac responding... maybe i should have gotten the trial
# info? or rename appropriately?
frac_responder_df = u.add_fly_id(pd.concat(frac_responder_sers).reset_index(
    name='frac_responding'
))

keys = ['prep_date', 'fly_num', 'fly_id']
new_fly_keys2fly_id = frac_responder_df[keys].drop_duplicates().set_index(
    keys[:2], verify_integrity=True).fly_id

# If this is True, and assuming add_fly_id acts same in other calls down here,
# then we are good.
assert new_fly_keys2fly_id.equals(fly_keys2fly_id)
del new_fly_keys2fly_id, keys


# TODO TODO TODO use this as input for all other analyses that
# deal w/ binary responses (and linearity stuff too, somehow)?
# spike_recordings actually useful there????
# TODO start by shuffling as responses above, to do the analysis below?
# (or just compare model to real stuff, w/o also comparing model shuffle
# to model?)
model_mixture_responses = True
if model_mixture_responses:
    # So we don't depend on `olfsysm` otherwise.
    from model_mix_responses import fit_model

    # TODO save something about input in filename?
    model_output_cache = 'model_responses.p'
    if exists(model_output_cache):
        print(f'reading cached model responses from {model_output_cache}')
        model_df = pd.read_pickle(model_output_cache)
    else:
        # TODO maybe convert 'responsed' col to boolean before returning?
        model_df = fit_model(frac_responder_df)
        print(f'writing model responses to cache at {model_output_cache}')
        # TODO TODO TODO maybe also save data about what inputs (+parameters
        # used to process those inputs, probably) were used to fit the model,
        # so i can actually use these caches elsewhere without needed all the
        # preceding code to be working, while still being able to know
        # what was fit and how
        model_df.to_pickle(model_output_cache)

    # TODO TODO TODO why was the scaling process apparently not able to get
    # model_df.groupby(['odor_set','name1']).responded.sum()
    # to have a closer ordering (~activtion strength) to my data?
    # something i could do differently? are we at some floor / ceiling in the
    # scaling?
    # TODO TODO TODO without apparently changing anything, the problem seems to
    # have gone away. maybe it was the cached bit for some reason? the loading?
    # (seems OK this most recent run w/ no intermediate caching)
    # (loading it in the same run still seems ok...)

    # TODO delete after fixing above two todos
    print(model_df.groupby(['odor_set','name1']).responded.mean())
    #

    ############################################################################
    mc = model_df.copy()

    cbar_label = 'Correlation'
    for oset in odor_set_order:
        comp_order = [o for o in odor_set2order[oset] if is_component(o)]

        # TODO TODO TODO did she want me to add them and correlate that??
        # maybe passing through the PN model *would* matter at that point,
        # assuming it's not just linear...
        # (if so, need to deal w/ 'mix' too, which is_component excludes)

        # TODO does drosolf let me modify input to pns.pns() fn? i forget.
        # (yes, first arg / orn=)

        odor_order = comp_order + ['mix']
        #odf = mc.loc[odor_order]
        # TODO TODO TODO also need to index by odor set just to get the right
        # order?
        odf = mc[mc.odor_set == oset].pivot(index='name1', columns='cell',
            values='responded'
        )

        # TODO TODO TODO want to limit cells to those that EVER respond before
        # computing correlations?

        '''
        if additive_mix_responses:
            #odf.loc['mix'] = odf[comp_order].sum()
            import ipdb; ipdb.set_trace()
        '''

        ocdf = odf.T.corr()
        # TODO TODO TODO just refactor plot_odor_corrs so it can detect either
        # name1 / name / odor as prefix, as long as it's unique (start w/ name1
        # and search all, probably)
        ocdf.index.name = 'name1'
        ocdf.columns.name = 'name1'
        #
        fig = u.plot_odor_corrs(ocdf, odors_in_order=odor_order,
            colorbar_label=cbar_label,
            title=f'{oset} components\n\nModel KC correlations'
        )
        savefigs(fig, f'model_kc_corrs_{oset}', section='Model KC correlations')
    ############################################################################

    plt.show()
    import ipdb; ipdb.set_trace()

# TODO something similar for legends w/ odor set
odor_set_legend_title = 'Panel'
fly_id_legend_title = 'Fly'
trial_stat_desc = f'{trial_stat} {u.dff_latex}'

# This only works because the groupby in loop over colors above and the 
# groupby in add_fly_id both sort, and thus have the same order.
# Also assuming sort order of keys matches sort order of fly_ids, and using
# np.unique rather than <ser>.unique() to also sort.
fly_id_palette = {i: c for i, c in
    zip(np.unique(frac_responder_df.fly_id), fly_colors)
}

# https://xkcd.com/color/rgb/
odor_set2color = {
    'kiwi': sns.xkcd_rgb['light olive green'],
    'control': sns.xkcd_rgb['orchid'],
    'flyfood': sns.xkcd_rgb['light brown']
}

odorset_distplot_hist_kws = {
    'histtype': 'step',
    'alpha': 1.0,
    'linewidth': 2.5
}
# col_wrap for FacetGrids
cw = 4

# TODO delete after fixing legend business
# (commented b/c colors do at least seem to be in correspondence between
# n_ro_hist and facetgrid below)
#print(frac_responder_df[fly_keys + ['fly_id']].drop_duplicates())
#
real_n_ros_df = u.add_fly_id(pd.concat(n_ros_sers).reset_index())
shuffle_n_ros_df = u.add_fly_id(pd.concat(shuffle_n_ros_sers).reset_index())

real_n_ros_df['is_shuffle'] = False
shuffle_n_ros_df['is_shuffle'] = True
n_ros_df = pd.concat([real_n_ros_df, shuffle_n_ros_df], ignore_index=True,
    sort=False
)

def set_shuffle_facet_titles(g, say_within_each_fly=True):
    for ax in g.axes.flat:
        old_title = ax.get_title()
        if old_title == 'is_shuffle = False':
            title = 'Real data'
        elif old_title == 'is_shuffle = True':
            # Whether a cell responded reliably to a certain odor is shuffled
            # across all cells and also across all odors, within each fly.
            title = 'Cells and odors shuffled'
            if say_within_each_fly:
                title += ' within each fly'
        else:
            raise AssertionError('unexpected title')
        ax.set_title(title)

aspect = 1.5
yvar = 'n_odors_reliable_to'
xlabel = 'Number of odors responded to reliably'
# TODO change 'Density of cells' in other places to this? or vice versa?
ylabel = 'Fraction of cells'
# TODO better name?
title = 'Cell tuning breadth vs. shuffle'
hspace = 0.3
fname = 'tuning_breadth_vs_shuffle'
section = 'Cell tuning breadth'

def tuning_breadth_vs_shuffle(fly=None):
    df = n_ros_df
    curr_title = title
    curr_fname = fname
    top = 0.9
    if fly is not None:
        df = df[df.fly_id == fly]
        curr_title = f'Fly {fly}\n{curr_title}'
        curr_fname += f'_fly{fly}'
        top = 0.87

    g = sns.FacetGrid(df, row='is_shuffle', hue='odor_set',
        palette=odor_set2color, aspect=aspect
    )
    g.map(sns.distplot, yvar, bins=n_ro_bins, kde=False,
        norm_hist=True, hist_kws=dict(odorset_distplot_hist_kws)
    )
    g.add_legend(title=odor_set_legend_title)
    g.set_axis_labels(xlabel, ylabel)
    g.fig.suptitle(curr_title)
    g.fig.subplots_adjust(top=top, hspace=hspace)
    u.fix_facetgrid_axis_labels(g)
    set_shuffle_facet_titles(g)
    savefigs(g.fig, curr_fname, section=section, order=200 if fly else None)

tuning_breadth_vs_shuffle()
for fly in n_ros_df.fly_id.unique():
    tuning_breadth_vs_shuffle(fly=fly)

for oset in odor_set_order:
    oset_df = n_ros_df[n_ros_df.odor_set == oset]

    # May not easily be able to show non-normalized plots here (or at least
    # counts w/in shuffle might stand to be adjusted, because raw counts are
    # also counted across shuffles, so scale is very different in shuffle case)
    g = sns.FacetGrid(oset_df, row='is_shuffle',
        hue='fly_id', palette=fly_id_palette, aspect=aspect
    )
    g.map(sns.distplot, yvar, norm_hist=True, kde=False,
        bins=n_ro_bins, hist_kws=dict(odorset_distplot_hist_kws)
    )
    g.add_legend(title=fly_id_legend_title)
    g.set_axis_labels(xlabel, ylabel)

    g.fig.suptitle(title + f'\n{oset}')
    set_shuffle_facet_titles(g, say_within_each_fly=False)
    g.fig.subplots_adjust(top=0.86, hspace=hspace, left=0.13)
    u.fix_facetgrid_axis_labels(g)
    savefigs(g.fig, f'{fname}_{oset}', section=section)


# Used swarmplot before, but I like the jittering here, to avoid overlapping
# points.

# TODO TODO check this works in all cases i was previously using stripplot...
# docs say swarmplot points don't overlap, so maybe i was mistaken when i
# previously thought points were overlapping (no i think it was a matter
# of facetgrid making multiple calls, across which points can overlap)?
# (check that swarmplot still works. see note @ redef of categ_pt_plot_fn below)

# TODO TODO maybe somehow connect lines that share a hue (to make it more
# visually clear how much overall responsiveness is the main thing that varies)

categ_pt_plot_fn = with_odor_order(sns.stripplot)
#categ_pt_plot_fn = with_odor_order(sns.swarmplot)

g = sns.FacetGrid(frac_responder_df, col='odor_set', hue='fly_id', 
    palette=fly_id_palette, sharex=False
)
g.map(categ_pt_plot_fn, 'name1', 'frac_responding')

# TODO TODO either somehow show (date, fly_num) in this legend, or show
# fly_id in n_ro_hist above, to match between them
# TODO TODO maybe modify add_fly_id to add a concatenation of string
# representations of each of the group keys?
g.add_legend(title=fly_id_legend_title)
g.set_axis_labels('Odor', 'Fraction responding')
# TODO way to just capitalize?
g.set_titles('{col_name}')
savefigs(g.fig, 'mean_frac_responding', section='Mean fraction responding',
    section_order=-1
)


# TODO TODO TODO TODO drop this fly from everything in the future
bad_corr_fly_ids = [4]


fdf = frac_responder_df.copy()
fdf = fdf.loc[~ fdf.fly_id.isin(bad_corr_fly_ids)]

g = sns.FacetGrid(fdf, col='odor_set', height=6.5,
    col_order=odor_set_order, sharex=False, dropna=False
)
ci = 68
g.map(with_odor_order(sns.barplot, ci=ci), 'name1', 'frac_responding')

ylabel = 'Mean fraction responding'
assert ci == 68
ebar_str = '\nError bars are SEM'
g.set_axis_labels('Odor', ylabel + ebar_str)

g.fig.suptitle('Mean fraction responding')
g.set_titles('{col_name}')
g.fig.subplots_adjust(left=0.05, top=0.85)

savefigs(g.fig, f'mean_frac_barplot', section='Mean fraction responding')


old_len = len(responders.index.drop_duplicates())
responders.name = 'response'
responders = add_odorset(u.add_fly_id(responders.reset_index()))
# These cols wouldn't cause any harm apart from being distracting.
responders.drop(columns=['prep_date', 'fly_num', 'thorimage_id', 'name2'],
    inplace=True
)
responders.set_index(['fly_id', 'odor_set', 'cell', 'name1', 'repeat_num'],
    inplace=True
)
responders = responders.response
assert len(responders.index.drop_duplicates()) == old_len

keys = [k for k in responders.index.names if k != 'repeat_num']
# TODO maybe rename to indicate it's also fly + cell info
# (suitable for selecting trial data from the full df)
# TODO TODO TODO drop data missing trials before this?
# (may complicate interpretation of anything depending on that...)
per_odor_reliable = responders.groupby(keys).sum() >= 2
per_odor_reliable.name = 'per_odor_reliable'

keys = [k for k in keys if k != 'name1']
reliable_to_any = per_odor_reliable.groupby(keys).any()
del keys
reliable_to_any.name = 'reliable_to_any'


if hallem_correlations:
    from drosolf import orns, pns

    # Not doing this for now, b/c I'd probably have to at least scale the ORN
    # responses... (but I guess computing correlations for the components also
    # relies on the ORN responses being just a linear rescaling...)
    # TODO TODO TODO discuss the above w/ B
    additive_mix_responses = False
    olsen_model_pn_correlations = False

    hdf = orns.orns()

    # Just dropping everything that chemutils cant abbrev, since all stuff in
    # here can be abbreviated that way.
    hdf.index = hdf.index.map(cu.odor2abbrev)
    hdf = hdf.loc[hdf.index.dropna()].copy()

    # it was just acetoin that wasn't in hallem, right?
    # (include null / X's on plot for that one?)
    # TODO and maybe one version w/o null regions?

    cbar_label = 'Correlation'
    for oset in odor_set_order:
        comp_order = [o for o in odor_set2order[oset] if is_component(o)]

        # TODO TODO TODO did she want me to add them and correlate that??
        # maybe passing through the PN model *would* matter at that point,
        # assuming it's not just linear...
        # (if so, need to deal w/ 'mix' too, which is_component excludes)

        # TODO does drosolf let me modify input to pns.pns() fn? i forget.
        # (yes, first arg / orn=)

        # TODO TODO TODO have i actually used this yet? try it!
        if additive_mix_responses:
            odor_order = comp_order + ['mix']
        else:
            odor_order = comp_order

        odf = hdf.loc[odor_order]

        if additive_mix_responses:
            #odf.loc['mix'] = odf[comp_order].sum()
            import ipdb; ipdb.set_trace()

        ocdf = odf.T.corr()
        # TODO TODO TODO just refactor plot_odor_corrs so it can detect either
        # name1 / name / odor as prefix, as long as it's unique (start w/ name1
        # and search all, probably)
        ocdf.index.name = 'name1'
        ocdf.columns.name = 'name1'
        #
        fig = u.plot_odor_corrs(ocdf, odors_in_order=odor_order,
            colorbar_label=cbar_label,
            title=f'{oset} components\n\nHallem ORN correlations'
        )
        savefigs(fig, f'hallem_corrs_{oset}', section='Hallem ORN correlations')

        if any([o not in hdf.index for o in comp_order]):
            odf = odf.dropna()
            odor_order = list(odf.index)

            ocdf = odf.T.corr()
            ocdf.index.name = 'name1'
            ocdf.columns.name = 'name1'
            fig = u.plot_odor_corrs(ocdf, odors_in_order=odor_order,
                colorbar_label=cbar_label,
                title=f'{oset} components\n\nHallem ORN correlations'
            )
            savefigs(fig, f'hallem_corrs_nonull_{oset}',
                exclude_from_latex=True
            )


# TODO lookup which corrs (max / mean) to use from trial_stat, if not gonna
# delete saving multiple
corr_df = u.add_fly_id(corr_ser_from_maxes.reset_index())
add_odorset(corr_df)

cols = ['fly_id','odor_set','name1_a','repeat_num_a','name1_b','repeat_num_b',
    'corr'
]
for tstat in ('max',):
    cbar_label = f'Mean of fly {tstat} response {u.dff_latex} correlations'
    if tstat == 'max':
        cdf = corr_df
    else:
        cdf = u.add_fly_id(corr_ser_from_means.reset_index())
        add_odorset(cdf)

    # TODO just drop this fly_id earlier so it applies everywhere
    df = cdf.loc[~ cdf.fly_id.isin(bad_corr_fly_ids), cols].copy()
    # so we don't need to worry about nanmean / something like that
    assert not df['corr'].isnull().any()

    n_per_odorset = df.groupby('odor_set').fly_id.nunique()

    # this averages corr over fly_id
    ser = df.groupby(cols[1:-1])['corr'].mean()
    name_cols = [c for c in ser.index.names if c.startswith('name1')]

    # TODO TODO in the future, maybe turn this into a facetgrid thing somehow,
    # and just share one colorbar to the right? (one facet per odor_set)
    for go, gser in ser.groupby('odor_set'):
        odor_order = odor_set2order[go]
        n = n_per_odorset[go]

        gdf = invert_melt_symmetric(gser)

        # TODO TODO TODO TODO for these, should probably not average over the
        # correlations between trials and themselves, right (since they are
        # always 1...)?
        mdf = invert_melt_symmetric(gser.groupby(name_cols).mean())

        for no_real in (False, True):
            noreal_str = '_noreal' if no_real else ''

            if no_real:
                #if len([o for o in odor_order if o in u.natural]) == 0:
                if go == 'control':
                    # (no need to also make this figure for the control set(s),
                    # which do not have a real odor)
                    continue

                def _get_mask(multiindex):
                    mask =  ~ multiindex.get_level_values(0).isin(u.natural)
                    assert mask.any()
                    return mask

                def _filter_real(rdf):
                    imask = _get_mask(rdf.index)
                    cmask = _get_mask(rdf.columns)
                    return rdf.loc[imask, cmask]

                p_gdf = _filter_real(gdf)
                p_mdf = _filter_real(mdf)
                porder = [o for o in odor_order if o not in u.natural]
            else:
                p_gdf = gdf
                p_mdf = mdf
                porder = odor_order

            # TODO maybe have the title n=<how many flies>. though i risk being
            # misleading since some odors like pfo have less...  or real kiwi...
            # maybe just drop stuff that has less? idk... real stuff is nice...
            fig = u.plot_odor_corrs(p_gdf, odors_in_order=porder,
                colorbar_label=cbar_label, title=f'{go}\n\n(n={n})'
            )
            savefigs(fig, f'mean_trial_corrs_{tstat}_{go}{noreal_str}',
                section='Mean KC response correlations, trial-by-trial',
                exclude_from_latex=no_real
            )

            fig = u.plot_odor_corrs(p_mdf, odors_in_order=porder,
                colorbar_label=cbar_label, title=f'{go}\n\n(n={n})'
            )
            savefigs(fig, f'mean_corrs_{tstat}_{go}{noreal_str}',
                section='Mean KC response correlations',
                exclude_from_latex=no_real
            )
del cols, ser, df, gdf, odor_order, fig, cbar_label

keys = ['odor_set','fly_id','name1_a','name1_b']
for corr_agg_fn in ('mean', 'max'):
    agged_corrs = corr_df.groupby(keys)['corr'].agg(corr_agg_fn).reset_index()

    cap_agg_fn = corr_agg_fn.title()
    gs, save_fn = odor_facetgrids(agged_corrs, categ_pt_plot_fn, 'name1_b',
        'corr', 'Odor B', f'{cap_agg_fn} of trial correlations',
        f'{cap_agg_fn} correlation consistency', marker='o'
    )
    for g in gs:
        g.fig.subplots_adjust(hspace=0.4)

    # TODO subsection for agg fn? look like what i want in pdf?
    save_fn(f'corrs_{corr_agg_fn}', section='Correlation consistency',
        subsection=cap_agg_fn, section_order=200
    )
del keys


response_magnitudes = u.add_fly_id(add_odorset(
    response_magnitudes.reset_index())
)

# could calculate these right after loading stuff / finishing first loop
# , and they might be useful up there. maybe replace some existing stuff
# computed above.
n_odor_repeats_per_rec = response_magnitudes.groupby(
    ['fly_id','odor_set','name1']).repeat_num.nunique()
#n_cells_per_rec = response_magnitudes.groupby(['fly_id','odor_set']
#    ).cell.nunique()

per_odor_reliable_idx = per_odor_reliable[per_odor_reliable].index

n_odor_reliable_cells_per_rec = per_odor_reliable_idx.to_frame(index=False
    ).groupby(['fly_id','odor_set','name1']).cell.nunique()


n_odor_colors = max([len(os) for os in odor_set_odors_nosolvent])
# TODO also, maybe just vary alpha? maybe use something that varies color more
# than this cmap does?
#odor_colors = sns.diverging_palette(10, 220, n=n_odor_colors, center='dark',
#    sep=1
#)
odor_colors = sns.color_palette('cubehelix', n_odor_colors)

nonsolvent_odor_orders = [[o for o in os if o not in u.solvents] for os in
    odor_set2order.values()
]

# TODO maybe hue by actual rather than intended (from odor_order) activation
# order of odor_set?
odor_hues_within_odorset = dict()
for color_and_odors in zip_longest(odor_colors, *nonsolvent_odor_orders):
    c = color_and_odors[0]
    odors = color_and_odors[1:]
    odors_are_nondiag = [o in nondiagnostic_odors for o in odors]
    # Any odors with common names (even if they can mean different things, like
    # 'mix'), should get the same color.
    # TODO TODO fix how this breaks in etoh / 2h / etoh case
    # (just don't have assert fail in that case?)
    '''
    if any(odors_are_nondiag):
        try:
            assert all(odors_are_nondiag), f'{odors} ({odors_are_nondiag})'
        except AssertionError:
            import ipdb; ipdb.set_trace()
    '''

    for o in odors:
        if o is not None:
            odor_hues_within_odorset[o] = c


mix_v_component_response_plots = False
if mix_v_component_response_plots:
    # TODO TODO do the same, but w/ responses to maybe the real thing on the 
    # axis where the mix is? (or just leave it as part of the mix plot?)
    # TODO maybe another plot binning to just show response (as hists maybe?) to
    # those considered responders to mix

    rmags = response_magnitudes.copy()
    # odor_set doesn't need to be explicitly included because recording keys
    # will also only have one odorset. intersection will remove 'comparison'.
    # need to exclude 'name1', so mix data can be put in the same row as 
    keys = [k for k in u.trial_cols if k in rmags.columns]
    keys = [k for k in keys if k != 'name1'] + ['cell']
    mix_rmags = rmags[rmags.name1 == 'mix'].set_index(keys)

    len_before = len(rmags)
    rmags.set_index(keys, inplace=True)
    rmags['mix_trialmax_df_over_f'] = mix_rmags.trialmax_df_over_f
    del mix_rmags
    rmags.reset_index(inplace=True)
    assert len_before == len(rmags)

    # Doing this after adding the mixture responses, so we DON'T constrain the 
    # mixture responses to need to come from mixture-reliably-responsive cells.
    rmags = rmags.set_index(per_odor_reliable_idx.names).loc[
        per_odor_reliable_idx
    ]
    rmags.reset_index(inplace=True)

    rmags = rmags[rmags.name1 != 'mix']

    # TODO and ideally, draw a dotted box or something on each plot to
    # represent the responsivenesses that don't qualify as responders
    # (or *just* draw such indicators, and leave the data?)
    # TODO or i think i just want to exclude non-reliable-responders for the
    # component responses, so i guess it should be a horizontal line?
    # TODO where do i actually get the threshold at this point?
    # (even possible? doesn't it vary per fly? scale s.t. looks same across
    # flies? take max? show region?)

    title = 'Mixture vs. component responses'
    section_name = title
    title += '\n(only cells reliable to the non-mix odor)'
    ts_label = trial_stat_desc

    # TODO TODO maybe also hue w/ fly_id, to see whether some things only hold
    # for certain flies, etc (or otherwise maybe make a plot like this for each
    # fly?)
    # TODO add flag to this fn to title facets w/ full odor names rather than
    # abbrev? or implement such a flag at the whole script level?
    gs, save_fn = odor_facetgrids(rmags, sns.scatterplot,
        'mix_trialmax_df_over_f', 'trialmax_df_over_f', 'Mixture ' + ts_label,
        'Other odor ' + ts_label, title, by_fly=False, mix_facet=False,
        sharey=False, alpha=0.2, color='black'
    )
    for oset, g in zip(odor_set_order, gs):
        for ax in g.axes.flat:
            os_rmags = rmags[rmags.odor_set == oset]
            os_max = max(
                os_rmags.trialmax_df_over_f.max(),
                os_rmags.mix_trialmax_df_over_f.max()
            )
            os_min = min(
                os_rmags.trialmax_df_over_f.min(),
                os_rmags.mix_trialmax_df_over_f.min()
            )
            os_lim = (os_min, os_max)
            ax.set_xlim(*os_lim)
            ax.set_ylim(*os_lim)
            # TODO maybe also set x and y ticks to be the same, so it's just a
            # little more obvious the scales are the same
    save_fn('mix_v_comp_rmags', section=section_name)
    del keys, rmags


seen_index_ids = set()
max_subset_df_len = None
df_len_unlabelled_subset = None
seen_subset_title_strs = set()
seen_subset_fname_strs = set()
def linearity_analysis(linearity_cell_df, response_magnitudes, cell_subset=None,
    cell_subset_title_str=None, cell_subset_fname_str=None,
    n_params2bins=None, reliable_to_any=None, odorsets_within_fly=False,
    **kwargs):

    # **kwargs just so I can pass extra information returned by grouping
    # functions, without it causing the call to this function to fail because
    # of the unrecognized kwargs.

    # The references to the sets does not change, so we don't need to declare
    # them globals as well.
    global max_subset_df_len
    global df_len_unlabelled_subset

    assert id(cell_subset) not in seen_index_ids, \
        'multiple calls with cell_subset pointing to same object'
    seen_index_ids.add(id(cell_subset))
    # TODO TODO TODO also try varying responder threshold (would require this
    # step to be able to calculate responders as above though...)

    if cell_subset is None:
        assert df_len_unlabelled_subset is None, \
            'multiple calls not subsetting data'
        df_len_unlabelled_subset = len(linearity_cell_df)

        lin_df = linearity_cell_df
        rmags = response_magnitudes

        cell_subset_title_str = 'all cells'
        cell_subset_fname_str = 'allcells'
    else:
        # TODO TODO TODO TODO but would i want to do the fits with this
        # restrictions too?? not sure just subsetting after the fact makes
        # sense...
        # TODO TODO TODO maybe at least try both (referring to TODO above)?

        n_in_cell_subset = cell_subset.sum()
        cell_subset_idx = cell_subset[cell_subset].index
        lin_df = linearity_cell_df.set_index(cell_subset.index.names)
        original_len = len(lin_df)
        assert n_in_cell_subset < original_len
        lin_df = lin_df.loc[cell_subset_idx].reset_index()
        assert len(lin_df) == n_in_cell_subset
        rmags = response_magnitudes.set_index(cell_subset.index.names)
        rmags = rmags.loc[cell_subset_idx].reset_index()
        del n_in_cell_subset, cell_subset_idx

        assert cell_subset_title_str is not None
        assert cell_subset_fname_str is not None
        assert cell_subset_title_str not in seen_subset_title_strs
        assert cell_subset_fname_str not in seen_subset_fname_strs
        seen_subset_title_strs.add(cell_subset_title_str)
        seen_subset_fname_strs.add(cell_subset_fname_str)

    if max_subset_df_len is None:
        max_subset_df_len = len(lin_df)
    else:
        max_subset_df_len = max(max_subset_df_len, len(lin_df))

    lin_odor_df = u.add_fly_id(add_odorset(linearity_odor_df.reset_index()))
    unq_n_components = lin_odor_df.groupby(['fly_id','odor_set']).name1.nunique(
        ).unique()
    assert len(unq_n_components) == 1
    n_components = unq_n_components[0]

    n_params_and_diff_col_prefixes = (
        (0, 'simple'),
        (1, 'scaled'),
        (n_components, 'weighted')
    )
    n_params2latex_eq = {
        0: r'$\left(\sum_{c\in components} R^{cell}_{c}\right) - '
            r'R^{cell}_{mix}$',
        1: r'$W^*\left(\sum_{c\in components} R^{cell}_{c}\right) - '
            r'R^{cell}_{mix}$',
        n_components: r'$\left(\sum_{c\in components} W^*_c R^{cell}_{c}\right)'
            r' - R^{cell}_{mix}$'
    }

    # TODO also replace categ_pt_plot_fn above w/ this after making sure that
    # stuff all still works + refactoring so hue+palettes are in map call
    # rather than facetgrid, and legend has explicit title
    categ_pt_plot_fn = with_odor_order(sns.swarmplot)

    # TODO TODO maybe try this kind of analysis w/ mean as trial stat rather
    # than max? seems likely that sum of max might blow up in a way that sum
    # of means might not...

    keys = ['fly_id', 'odor_set', 'name1']
    mean_rmags = rmags.groupby(keys).trialmax_df_over_f.mean()
    mean_rmags.name = 'mean_' + mean_rmags.name
    # mtrs = mean trial response stat
    mtrs_name = mean_rmags.name

    keys = ['fly_id', 'odor_set']
    mean_rmag_df = mean_rmags.reset_index()
    csum = mean_rmag_df.loc[mean_rmag_df.name1.map(is_component)].groupby(
        keys)[mtrs_name].sum()
    del mean_rmag_df, keys
    csum = pd.concat([csum], names=['name1'], keys=[component_sum_odor_name]
        ).reorder_levels(mean_rmags.index.names)

    csum_minus_mix = csum - mean_rmags.loc[:, :, 'mix']
    csum_minus_mix = csum_minus_mix.reset_index()
    csum_minus_mix.name1 = component_sum_diff_odor_name
    csum_minus_mix = \
        csum_minus_mix.set_index(mean_rmags.index.names)[mtrs_name]

    mean_rmags = pd.concat([mean_rmags, csum, csum_minus_mix])
    mean_rmag_df = mean_rmags.reset_index()

    # TODO not sure how to get this to work w/ groupby, b/c i i'm not sure
    # what to do w/ original range index of group df
    '''
    mean_rmag_df = mean_rmag_df.groupby('odor_set').apply(
        lambda gdf: fill_to_cartesian(gdf, ['fly_id','odor_set','name1'])
    )
    '''
    full_mean_rmag_dfs = []
    for gn, gdf in mean_rmag_df.groupby('odor_set'):
        full_gdf = fill_to_cartesian(gdf, ['fly_id','odor_set','name1'])
        full_mean_rmag_dfs.append(full_gdf)
    mean_rmag_df = pd.concat(full_mean_rmag_dfs).reset_index()

    # Since we subset the input by cell_subset, this may include a subset
    # of all the odor sets.
    curr_odor_set_order = [o for o in odor_set_order
        if o in lin_df.odor_set.unique()
    ]
    # Expect 1 in case where cells were categorized in a way only meaningful
    # within an odorset, otherwise all odorsets.
    assert (len(curr_odor_set_order) == 1 or
        len(curr_odor_set_order) == len(odor_set_order)
    )
    # TODO TODO maybe keep aspect ratio of each facet constant, indep. of
    # whether there is one or three. lack of this property seems to be why
    # these next two facetgrid plots get split onto two pages for the
    # cell-category-specific stuff, while only taking up one page for the
    # across-all-cells/responders stuff
    # TODO TODO TODO what broke this?
    g = sns.FacetGrid(mean_rmag_df, col='odor_set', height=6.5,
        col_order=curr_odor_set_order, sharex=False, dropna=False
    )
    # TODO make all barplots this blue color? or make these ones black?
    # (for consistency's sake)
    # still color as odor_set despite not sharing a sub figure?]
    ci = 68
    g.map(with_odor_order(sns.barplot, ci=ci), 'name1', mtrs_name)

    ylabel = 'Mean of cell-trial ' + trial_stat_desc
    if ci == 68:
        ebar_str = '\nError bars are SEM'
    else:
        ebar_str = f'\nError bars are {ci}% CI'
    g.set_axis_labels('Odor', ylabel + ebar_str)

    g.fig.suptitle('Mean response magnitudes and their raw sum\n'
        f'{cell_subset_title_str.capitalize()}'
    )
    top = 0.85
    if len(curr_odor_set_order) > 1:
        g.set_titles('{col_name}')
        g.fig.subplots_adjust(left=0.05, top=top)
    else:
        # Assuming the odorset is also mentioned in the suptitle in this case.
        g.set_titles('')
        g.fig.subplots_adjust(left=0.13) #, top=top)

    savefigs(g.fig, f'simple_linearity_{cell_subset_fname_str}',
        section='Linearity',
        subsection=f'Simple linearity, {cell_subset_title_str}'
    )

    # was necesssary to get dodge kwarg to work w/ wrapped stripplot fn
    # see: https://stackoverflow.com/questions/46134282 for explanation
    # also necessary to actually have swarmplot points not overlap, for
    # same reason i believe.
    g = sns.FacetGrid(mean_rmag_df, col='odor_set', height=6.5,
        col_order=curr_odor_set_order, sharex=False, dropna=False
    )
    g.map(categ_pt_plot_fn, 'name1', mtrs_name, 'fly_id',
        palette=fly_id_palette
    )
    # TODO maybe rename ylabel to Mean response magnitude, and just define
    # that somewhere (maybe a definitions section in pdf or something?)
    # cell-trial / cell+trial / (cell,trial) ?
    g.set_axis_labels('Odor', ylabel)
    g.add_legend(title=fly_id_legend_title)
    g.fig.suptitle('Fly-mean response magnitudes and their raw sum\n' +
        f'{cell_subset_title_str.capitalize()}'
    )
    if len(curr_odor_set_order) > 1:
        g.set_titles('{col_name}')
        g.fig.subplots_adjust(top=top)
    else:
        g.set_titles('')
        #g.fig.subplots_adjust(top=top)

    savefigs(g.fig, f'flymean_simple_linearity_{cell_subset_fname_str}',
        section='Linearity',
        subsection=f'Simple linearity, {cell_subset_title_str}'
    )

    assert reliable_to_any is not None, 'must pass kwarg reliable_to_any'
    # Omitting the checks in similar section above. They should have
    # been triggered if applicable when this function was called with
    # reliable-to-any input (Which it should have been, as I'm doing
    # things. Otherwise we wouldn't have n_params2bins).
    ra_lin_df = linearity_cell_df.set_index(reliable_to_any.index.names)
    reliable_to_any_idx = reliable_to_any[reliable_to_any].index
    ra_lin_df = ra_lin_df.loc[reliable_to_any_idx].reset_index()

    if n_params2bins is None:
        n_params2bins = dict()
        had_n_params2bins = False
    else:
        had_n_params2bins = True

    for n_params, diff_col_prefix in n_params_and_diff_col_prefixes:
        diff_col = f'{diff_col_prefix}_mix_diff'
        if had_n_params2bins:
            bins = n_params2bins[n_params]
            # TODO or maybe *only* use the xlim from bins, but make our own bins
            # here??
            xlim = np.array([bins.min(), bins.max()])
        else:
            # TODO see notes in other place i use this strategy to pick range /
            # bins
            percent_excluded = 2 * 0.5
            xlim_percentiles = [
                percent_excluded / 2,
                100 - percent_excluded / 2
            ]
            xlim = np.percentile(lin_df[diff_col], xlim_percentiles)
            del xlim_percentiles

            # NOTE: Don't change this to False without recovering the computed
            # bins, and entering those in n_params2bins.
            fixed_bins = True
            if fixed_bins:
                bins = np.linspace(*xlim, num=(30 + 1))
                n_params2bins[n_params] = bins
            else:
                bins = None
                raise NotImplementedError('see comment above')

            print_percentiles = False
            if print_percentiles:
                print(diff_col)
                print('xlim:', xlim)
                percentiles = [100 * f for f in (0.01, 0.005, 0.001)]
                percentiles += [100 - p for p in percentiles]
                percentiles = sorted(percentiles)
                for p, s in zip(percentiles,
                    np.percentile(lin_df[diff_col], percentiles)):

                    print('{:0.2f} percentile diff: {:.2f}'.format(p, s))

        t1 = f'Residuals from ({n_params} parameter) linear model'
        title = f'{t1}, {cell_subset_title_str}'
        two_line_title = f'{t1}\n{cell_subset_title_str.capitalize()}'
        del t1
        xlabel = n_params2latex_eq[n_params]
        ylabel = 'Density of cells'

        def compare_odorset_residual_dists(fly=None):
            ctitle = str(title)
            if fly is not None:
                ctitle = f'Fly {fly}\n{ctitle}'
                fly_indices = lin_df.fly_id == fly

            fig, ax = plt.subplots()
            for oset in curr_odor_set_order:
                cell_err_indices = lin_df.odor_set == oset
                if fly is not None:
                    cell_err_indices = cell_err_indices & fly_indices

                # Since there are flies (most) that don't have all odorsets.
                if not cell_err_indices.any():
                    continue

                os_cell_errs = lin_df.loc[cell_err_indices, diff_col]

                if len(curr_odor_set_order) > 1:
                    label = oset.title()
                else:
                    label = 'Title subset'

                # TODO TODO add note to pdf somewhere mentioning that bins
                # are picked separately in this case
                # (there are fewer points and sometimes things looked like
                # combs otherwise, but using xlim from passed bins)

                sns.distplot(os_cell_errs, label=label,
                    bins=bins if fly is None else None,
                    color=odor_set2color[oset], kde=False, norm_hist=True,
                    hist_kws=dict(odorset_distplot_hist_kws)
                )

            if len(curr_odor_set_order) == 1:
                # This means we never need to worry about indexing the stuff
                # in this branch by fly.
                assert fly is None

                # oset still defined from loop above.
                ra_os_cell_errs = \
                    ra_lin_df.loc[ra_lin_df.odor_set == oset, diff_col]

                hist_kws = dict(odorset_distplot_hist_kws)
                hist_kws['linestyle'] = '--'
                sns.distplot(ra_os_cell_errs, label='Cells reliable to any',
                    bins=bins, color=odor_set2color[oset], kde=False,
                    norm_hist=True, hist_kws=hist_kws
                )

                ac_os_cell_errs = linearity_cell_df.loc[
                    linearity_cell_df.odor_set == oset, diff_col
                ]
                hist_kws = dict(odorset_distplot_hist_kws)
                hist_kws['linestyle'] = 'dotted'
                sns.distplot(ac_os_cell_errs, label='All cells', bins=bins,
                    color=odor_set2color[oset], kde=False, norm_hist=True,
                    hist_kws=hist_kws
                )

            if len(curr_odor_set_order) > 1:
                ax.set_title(ctitle)
            else:
                ax.set_title(two_line_title)

            ax.set_xlim(*xlim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            if len(curr_odor_set_order) > 1:
                # Because otherwise the LaTeX equations in the xlabels are cut
                # off.
                fig.subplots_adjust(bottom=0.15)
            else:
                fig.subplots_adjust(bottom=0.15, top=0.85)

            cfname = f'linearity_dists_{cell_subset_fname_str}_{n_params}param'
            if fly is not None:
                cfname = f'{cfname}_fly{fly}'

            # TODO option to start a new page after this subsection
            # (to avoid confusion as to which subsection plots belong to,
            # particularly as long as there are not subsection headings).
            # and maybe that should be the default behavior?
            savefigs(fig, cfname, section='Linearity',
                subsection=f'({n_params} parameter) linearity distributions, '
                f'{cell_subset_title_str}', order=200 if fly else None
            )
        # End compare_odorset_residual_dists def.

        compare_odorset_residual_dists()
        if odorsets_within_fly:
            # TODO TODO make a subsection w/ link and everything for these fly
            # specific dists. also maybe make figures a little smaller,
            # so we can fit multiple per page (or will latex stuff just
            # zoom them anyway? implement way to avoid that if so?)
            # TODO TODO for cases where only one odorset would be plotted,
            # do not make this plot (because it's redundant w/ some of the other
            # plots)
            for fly in lin_df.fly_id.unique():
                compare_odorset_residual_dists(fly=fly)

        # TODO delete
        #if cell_subset is not None and cell_subset.sum() < 100:
        #plt.close('all')
        #
        g = sns.FacetGrid(lin_df, row='odor_set', row_order=curr_odor_set_order,
            hue='fly_id', palette=fly_id_palette, aspect=2
        )
        # TODO TODO maybe don't exactly norm_hist here, but maybe scale
        # so flies that contribute less cells to a particular category have
        # a shorter summed bar height?
        # TODO TODO maybe this means i'll have to plot the all cells /
        # cells reliable to any plots w/o distplot, scaling to match the scale
        # of the summed count of cells w/in whatever subset being plotted?
        # TODO maybe rugplot?

        # TODO how to compare to next two distributions sometimes plotted
        # on same axis???
        norm_multifly_hists = False
        # Copying value passed to hist_kws, because it seems seaborn modified
        # this dict such that norm_hist=False had no effect after a
        # norm_hist=True call (because of an added density=True key).
        g.map(sns.distplot, diff_col, norm_hist=norm_multifly_hists, kde=False,
            bins=bins, hist_kws=dict(odorset_distplot_hist_kws)
        )

        label_order = list(map(str, g.hue_names))
        if norm_multifly_hists:
            hist_kws = dict(odorset_distplot_hist_kws)
            hist_kws['linestyle'] = 'dotted'
            label = 'All cells'
            for i, (ax, oset) in enumerate(
                zip(g.axes.flat, curr_odor_set_order)):

                ac_os_cell_errs = linearity_cell_df.loc[
                    linearity_cell_df.odor_set == oset, diff_col
                ]
                sns.distplot(ac_os_cell_errs, label=label, bins=bins,
                    color='gray', kde=False, norm_hist=True, hist_kws=hist_kws,
                    ax=ax
                )
                if i == 0:
                    g._update_legend_data(ax)
                    # Generally called next to _update_legend_data in seaborn
                    # source, but may not be necessary in my case.
                    #g._clean_axis(ax)
                    label_order =  label_order + [label]

            hist_kws = dict(odorset_distplot_hist_kws)
            hist_kws['linestyle'] = '--'
            label = 'Cells reliable to any'
            for i, (ax, oset) in enumerate(
                zip(g.axes.flat, curr_odor_set_order)):

                ra_os_cell_errs = \
                    ra_lin_df.loc[ra_lin_df.odor_set == oset, diff_col]

                sns.distplot(ra_os_cell_errs, label=label, bins=bins,
                    color='gray', kde=False, norm_hist=True, hist_kws=hist_kws,
                    ax=ax
                )
                if i == 0:
                    g._update_legend_data(ax)
                    #g._clean_axis(ax)
                    label_order = label_order + [label]

        g.add_legend(label_order=label_order, title=fly_id_legend_title)

        if norm_multifly_hists:
            mf_ylabel = ylabel
        else:
            mf_ylabel = 'Cell counts'
        g.set_axis_labels(xlabel, mf_ylabel)

        g.fig.suptitle(two_line_title, size='medium')
        if len(curr_odor_set_order) > 1:
            g.set_titles('{row_name}')
            g.fig.subplots_adjust(bottom=0.08, top=0.91)
        else:
            g.set_titles('')
            g.fig.subplots_adjust(bottom=0.2, left=0.1)

        u.fix_facetgrid_axis_labels(g)
        # TODO delete
        #if cell_subset is not None and cell_subset.sum() < 100:
        #plt.show()
        #import ipdb; ipdb.set_trace()
        #
        savefigs(g.fig,
            f'linearity_dists_per_fly_{cell_subset_fname_str}_'
            f'{n_params}param', section='Linearity',
            subsection=f'({n_params} parameter) linearity distributions, '
            f'{cell_subset_title_str}'
        )

        # TODO TODO maybe just always show some example cells w/ numerical
        # labels to indicate where they came from?

        # TODO TODO either return odor_set, fly_id, cell -> bins (or inverse) or
        # have kwarg flag to plot some example cells based on this info (pulling
        # from different parts of dist like B wanted) (still want to do this?)

    # So that we it's obvious the returned value should not be used if it was
    # passed in.
    if had_n_params2bins:
        n_params2bins = None

    return n_params2bins

# TODO TODO repeat this analysis only using kiwi data for fixed
# concentration (what was the change again?)!
linearity_cell_df = u.add_fly_id(add_odorset(linearity_cell_df.reset_index()))

# TODO move above / maybe return this by default (as another positional return
# val?) in fn that generates groups?
def groups2cell_fraction_series(groups):
    g = groups[0]
    keys = [k for k in g['cell_subset'].index.names
        if k != 'cell' and k in g.keys() 
    ] + ['odors_str', 'cell_fraction']
    del g

    series = pd.DataFrame(data=[{k: v for k, v in g.items() if k in keys}
        for g in groups]).set_index(keys[:-1]).squeeze()
    assert len(series.shape) == 1
    # TODO maybe append part of name of input ser to name (now just
    # 'cell_fraction') (e.g. to indicate 'reliable' or whatever other reduction
    # across trial response calls)
    return series

print_classification_rankings = False
largest_reliable_groups = None
for at_least_n_trials in (1, 2, 3):
    if at_least_n_trials == 2:
        # TODO maybe exclude stuff w/ missing data. or at least be wary of it
        # complicating these things...
        # Same as if computed w/ n=2 in else case below.
        per_odor_enough_responded_trials = per_odor_reliable
        name = 'reliable'
        nonresponder_title_str = 'non-reliable'
        title_and_fname_strs = True
        title_str = 'reliably'
    else:
        keys = [k for k in responders.index.names if k != 'repeat_num']
        # TODO maybe exclude stuff w/ missing data. or at least be wary of it
        # complicating these things...
        per_odor_enough_responded_trials = \
            responders.groupby(keys).sum() >= at_least_n_trials
        del keys

        if at_least_n_trials == 3:
            name = 'all trial'
        else:
            name = f'>={at_least_n_trials} trial'

        nonresponder_title_str = 'non-responders'
        title_and_fname_strs = False
        title_str = f'on {name}s'

    if at_least_n_trials == 1:
        subsection = 'At least 1 trial'
    else:
        subsection = f'At least {at_least_n_trials} trials'

    if print_classification_rankings:
        print(f'\nOdorset AND fly {name} responder classification:')

    per_fly_groups = group_cells_by_odors_responded(
        per_odor_enough_responded_trials,
        per_fly=True,
        verbose=print_classification_rankings,
        title_and_fname_strs=title_and_fname_strs,
        nonresponder_title_str=nonresponder_title_str
    )

    if print_classification_rankings:
        print(f'\nOdorset {name} responder classification:')

    groups = group_cells_by_odors_responded(
        per_odor_enough_responded_trials,
        verbose=print_classification_rankings,
        title_and_fname_strs=title_and_fname_strs,
        nonresponder_title_str=nonresponder_title_str
    )

    largest_groups = []
    # TODO different parameters for barplot and for passing to
    # linearity_analysis?
    n_largest_groups = 9
    for oset in odor_set_order:
        # Assumes that group_cells_by_odors_responded sorts by fraction of
        # cells in the group, which it does (within odorsets).
        oset_largest = [
            g for g in groups if g['odor_set'] == oset
        ][:n_largest_groups]
        largest_groups.extend(oset_largest)

    if at_least_n_trials == 2:
        largest_reliable_groups = largest_groups

    ser = groups2cell_fraction_series(largest_groups)
    per_fly_ser = groups2cell_fraction_series(per_fly_groups)

    # Subsetting to only data from groups w/ largest cell fractions (ser).
    per_fly_df = per_fly_ser.reset_index().set_index(
        ser.index.names).loc[ser.index].reset_index()
    del per_fly_ser
    df = ser.reset_index()
    del ser

    # Whether to use rows or cols for the odor_set in the FacetGrid.
    use_rows = False
    if use_rows:
        fg_kwargs = dict(row='odor_set', row_order=odor_set_order)
    else:
        fg_kwargs = dict(col='odor_set', col_order=odor_set_order)

    def set_odors_cell_fraction_plot_props(g):
        if use_rows:
            g.set_titles('{row_name}')
        else:
            g.set_titles('{col_name}')

        # TODO maybe also indicate non-natural only
        g.fig.suptitle(
            f'Top {n_largest_groups} odor combos responded to {title_str}'
        )
        g.set_xlabels('')
        g.set_ylabels('Fraction of cells')
        u.fix_facetgrid_axis_labels(g)

        # Since g.set_xticklabels(rotation=90) changes all xticklabels to only
        # those of first axis, even though sharex=False.
        for ax in g.axes.flat:
            labels = [x.get_text() for x in ax.get_xticklabels()]
            ax.set_xticklabels(labels, rotation=90)

        if use_rows:
            # need to fix bounds so suptitle not clipped in this case
            # (if i want to use this branch... which i don't now)
            g.fig.subplots_adjust(hspace=1.5, bottom=0.2, top=0.85)
        else:
            g.fig.subplots_adjust(bottom=0.6, top=0.85)

    # NOTE: Important that True comes second, since df and per_fly_df are
    # changed in that case.
    for exclude_nonresponders in (False, True):
        if exclude_nonresponders:
            df = df[df.odors_str != nonresponder_title_str]
            per_fly_df = \
                per_fly_df[per_fly_df.odors_str != nonresponder_title_str]

        # TODO maybe somehow combine bar plot + plots that shows points for each
        # fly into one plot? could use sns.boxplot to get a line for each
        # across-fly point

        fname = f'{at_least_n_trials}responders_by_odors'
        if not exclude_nonresponders:
            fname += '_with_nonresponders'

        g = sns.FacetGrid(data=df, sharex=False, **fg_kwargs)
        # Using plt.bar rather than sns.barplot cause I don't think I need any
        # of the functionality only in latter, and it warns about lack of order
        # kwarg.
        g.map(plt.bar, 'odors_str', 'cell_fraction')
        set_odors_cell_fraction_plot_props(g)
        # TODO maybe order this section so it's just before linearity section
        savefigs(g.fig, fname, section='Responder breakdown',
            subsection=subsection
        )

        g = sns.FacetGrid(data=per_fly_df, sharex=False, hue='fly_id',
            palette=fly_id_palette, **fg_kwargs
        )
        # TODO suppress / deal with this warning
        # (same reason using plt.bar above)
        g.map(sns.stripplot, 'odors_str', 'cell_fraction', alpha=0.5)
        set_odors_cell_fraction_plot_props(g)
        g.add_legend(title=fly_id_legend_title)
        savefigs(g.fig, 'per_fly_' + fname, section='Responder breakdown',
            subsection=subsection
        )

assert largest_reliable_groups is not None

# TODO add parameters like this + things that control binning to params
# rendered into pdf?
linearity_analysis_on_all_cells = True
lin_subset_kwargs_list = [{
    'cell_subset': reliable_to_any,
    'cell_subset_title_str': 'cells reliable to any odor',
    'cell_subset_fname_str': 'reliabletoany',
    # Makes 1 extra plot per (fly x model) w/ each odorset distribution
    # overlayed.
    'odorsets_within_fly': True
}]
if linearity_analysis_on_all_cells:
    lin_subset_kwargs_list.append(dict())

if not args.skip_cell_subset_linearity:
    # TODO TODO TODO also do stuff similar w/ clustering cells?
    # (how to verify number of clusters is reasonable though? just don't?)
    # TODO maybe also see how things categorize as above w/in each
    # cluster i'm using?
    largest_reliable_groups_excluding_nonresponders = [
        g for g in largest_reliable_groups if len(g['odors']) > 0
    ]
    lin_subset_kwargs_list.extend(
        largest_reliable_groups_excluding_nonresponders
    )

# TODO make sense to plots residuals? should i scale by # of cells or something
# (like average residual per cell)?
# TODO one / multiple plots to compare fits of diff models?

n_params2bins = None
for i, lin_subset_kwargs in enumerate(lin_subset_kwargs_list):
    if i > 0:
        assert n_params2bins is not None

    # TODO TODO TODO update latex equations on these plots to indicate that 
    # weights are recording specific (they are, right?)

    # TODO TODO TODO maybe each of these sets of plots should include the
    # fraction of cells in the subset in the title / somewhere on the plot?
    # (would probably want it in legend or something for per-fly plot...)

    # TODO TODO actually get subsection entry working so i can click to
    # these things from index. maybe also link from cognate bar chart entries
    # for rankings of cell classifications, if possible.
    # TODO and even more, would be great to be able to click from points/bars
    # in odor-combo-cell-frac-ranking plots to corresponding plots here
    # (but this may involve modifying pdf output?)
    ret = linearity_analysis(linearity_cell_df, response_magnitudes,
        n_params2bins=n_params2bins, reliable_to_any=reliable_to_any,
        **lin_subset_kwargs
    )
    # Assuming the bins we want to use are from the first element of the
    # kwarg list (as cells reliable to any odor should be).
    if i == 0:
        n_params2bins = ret

assert (df_len_unlabelled_subset is None or
    max_subset_df_len == df_len_unlabelled_subset), \
    'unlabelled subset should have longest df, as only call not subsetting'
del seen_index_ids, max_subset_df_len, df_len_unlabelled_subset
del seen_subset_title_strs, seen_subset_fname_strs


if do_roc:
    auc_df = u.add_fly_id(add_odorset(auc_df.reset_index()))
    auc_df.rename(columns={'odor': 'name1'}, inplace=True)
    # TODO why is this:
    # len(auc_df[['fly_id','odor_set','cell']].drop_duplicates()) (1113)
    # slightly different from this: reliable_to_any.sum() (1118)
    # ???

    auc_vline_alpha = 0.5
    fixed_auc_bins = True
    if fixed_auc_bins:
        n_auc_bins = 30
        auc_bins = np.linspace(0, 1, num=(n_auc_bins + 1))
    else:
        auc_bins = None

    g = sns.FacetGrid(auc_df, col='task', hue='odor_set',
        palette=odor_set2color
    )
    g.map(sns.distplot, 'auc', norm_hist=True, kde=False, bins=auc_bins,
        hist_kws=dict(odorset_distplot_hist_kws)
    )
    xlabel = 'AUC'
    ylabel = 'Density of cells'
    g.set_axis_labels(xlabel, ylabel)
    g.set_titles('{col_name}')
    g.add_legend(title=odor_set_legend_title)
    for ax in g.axes.flat:
        ax.axvline(x=0.5, color='r', alpha=auc_vline_alpha)

    g.fig.subplots_adjust(left=0.08)
    savefigs(g.fig, 'roc_task_and_odorset', section='ROC',
        subsection='Mean AUC distributions'
    )

    # TODO TODO TODO remove plots that are not that meangingful here and w/
    # other auc stuff. make sure data going into each makes sense.
    for task in ('segmentation', 'discrimination'):
        gs, save_fn = odor_facetgrids(auc_df, sns.distplot, 'auc', None,
            xlabel, ylabel, f'Mean {task} AUC distributions', by_fly=False,
            bins=auc_bins, kde=False, norm_hist=True
        )
        for g in gs:
            for ax in g.axes.flat:
                ax.axvline(x=0.5, color='r', alpha=auc_vline_alpha)
            g.fig.subplots_adjust(left=0.05)

        save_fn(f'roc_{task}_by_odor', section='ROC', subsection=task.title())

    # TODO anything by fly (averages w/in odor_set?) or just leave it at
    # individual (task, odor) distributions for the per-fly analyses?


if do_pca:
    # TODO TODO some way to make a statement about the PCA stuff across flies?
    # worth it?
    # TODO maybe some kind of matshow of first n PCs in each fly? 

    pca_df = u.add_fly_id(add_odorset(pca_df.reset_index()))
    for col_prefix, title_part in (('std', ''), ('unstd', 'un')):
        x = 'component_number'
        pca_xlim = (1, 8)
        xlabel = 'Component number'
        ylabel = 'Explained variance fraction'
        ev_col = col_prefix + '_explained_var_ratio'
        shared_title = f'cree plot\n(from PCA on {title_part}standardized data)'
        section_prefix = \
            'Standardized' if col_prefix == 'std' else 'Unstandardized'

        os_scree_fig, ax = plt.subplots()
        pca_ci = 90
        sns.lineplot(data=pca_df, x=x, y=ev_col, hue='odor_set',
            palette=odor_set2color, ax=ax, ci=pca_ci
        )
        ax.set_xlim(pca_xlim)
        ax.set_title('S' + shared_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean ' + ylabel.lower() + f'\n(with {pca_ci:.0f}% CI)')
        savefigs(os_scree_fig, f'scree_{col_prefix}',
            section=f'{section_prefix} PCA',
            subsection='Mean scree'
        )

        # TODO change back to row if i can fix layout in pdf
        g = sns.FacetGrid(pca_df, col='odor_set', col_order=odor_set_order,
            hue='fly_id', palette=fly_id_palette, xlim=pca_xlim
        )
        g.map(sns.lineplot, x, ev_col)
        g.add_legend(title=fly_id_legend_title)
        g.set_titles('{col_name}')
        g.set_axis_labels(xlabel, ylabel)

        g.fig.suptitle('Across-fly s' + shared_title)
        g.fig.subplots_adjust(top=0.75)
        savefigs(g.fig, f'acrossfly_scree_{col_prefix}',
            section=f'{section_prefix} PCA', subsection='Across-fly scree'
        )


# TODO factor? didn't i use something like this in natural_odors?
def pd_linregress(gdf, xvar, yvar, verbose=True):
    # TODO maybe still return a Series w/ NaN for everything *but*
    # 'n' (which would be 1) here?
    if len(gdf) == 1:
        if verbose:
            print('Only 1 point for:')
            print(gdf[[c for c in gdf.columns if c not in (xvar, yvar)]])
            print('...so could not fit a line describing the adaptation.\n')

        return None

    slope, intercept, r_value, p_value, std_err = linregress(gdf[[xvar, yvar]])
    return pd.Series({
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'n': len(gdf),
        'first_data_y': gdf[yvar].iloc[0]
    })

do_adaptation_analysis = False
if do_adaptation_analysis:
    # TODO some kind of differential analysis on adaptation speeds
    # (each mix relative to its components)??
    # might be more interesting than just claiming basically either:
    # "the particular components in one of our panels happened to cause more
    # adaptation" or
    # "the mixture response of one adapted faster"
    # maybe mixture response was fast-adapting but component responses were less
    # so (and there was a difference in this difference across odor_sets) that
    # would be it? anymore analysis at cell resolution?

    adapt_sec_order = 100

    responders_only_values = (True,)
    adaptation_analysis_on_all_cells = False
    if adaptation_analysis_on_all_cells:
        responders_only_values += (False,)

    for responders_only in responders_only_values:
        if responders_only:
            rmags = response_magnitudes.set_index(per_odor_reliable.index.names)
            rmags = rmags.loc[per_odor_reliable_idx]

            # This is just to prove that the cause of the length discrepency
            # between rmags and my original expectation:
            # len(per_odor_reliable_idx) * 3
            # ...was indeed the odors with some missing trials.
            # TODO maybe check those odor are missing from (or NaN in) full,
            # unfiltered dataframes too?
            assert (len(rmags) + (3 - rmags.groupby(rmags.index.names
                ).repeat_num.nunique()).sum() == len(per_odor_reliable_idx) * 3)
            rmags = rmags.reset_index()

            # This product produces some NaN, I think b/c there were no reliable
            # cells for those odors in certain recordings (i.e. (odor_set,
            # fly_id)).
            expected_len = int(
                (n_odor_repeats_per_rec * n_odor_reliable_cells_per_rec).sum()
            )
            assert expected_len == len(rmags)
        else:
            rmags = response_magnitudes

        # So that repeat_num would never start the loop changed.
        rmags = rmags.copy()
        # Just so these don't start at 0. Doing this after indexing responders
        # so changing this doesn't throw off the correspondence.
        rmags.repeat_num = rmags.repeat_num + 1

        # This should roughly be the average # of cells, and it seems to be.
        # response_magnitudes.shape[0] / mean_rmags.shape[0]
        # (in case where all cells are included, at least)
        mean_rmags = rmags.groupby(['odor_set','fly_id','name1','repeat_num']
            ).trialmax_df_over_f.mean().reset_index()

        # TODO TODO TODO if fitting across flies, should i normalize to the
        # response of the first trial first or something???

        # TODO and depending on how i reduce adaptation speed to a scalar,
        # probably need to throw out stuff where line was fit from two points
        # (maybe just don't fit lines on only two points) because maybe there is
        # more adaptation between 1/2 than 2/3 or something, so i can't
        # interpolate?

        # TODO TODO maybe plot distribution of response magnitudes (for all
        # cells) but for each (odor_set, fly, odor, trial), to see if there are
        # shifts in this distribution or changes in proportions between two
        # populations.
        # TODO maybe just looking at the data from a strong odor, some
        # bimodality in response magnitudes is more apparent?

        # TODO TODO TODO some kind of scatter plot / 2d hist of initial response
        # vs slope (color by odor here? or avoid, just to limit the possible
        # meanings colors can have?)
        # TODO style w/in odorset for odor? too much?

        adaptation_fits = mean_rmags.groupby(['odor_set','name1']).apply(
            lambda x: pd_linregress(x, 'repeat_num', 'trialmax_df_over_f')
        )

        if responders_only:
            fname_suffix = 'reliable_responder'
            title = 'Adaptation of reliable responders'
        else:
            fname_suffix = 'allcell'
            title = 'Adaptation of all cells'

        # TODO want to change this to "mean of trial max...", or too verbose?
        # some way to do it? and should i include whether it's restricted to
        # responders here (i'm leaning towards no, for space again)?
        # TODO maybe just get from capitalizing first char in trial_stat_desc?
        ylabel = f'{trial_stat.title()} trial {u.dff_latex}'
        gs, save_fn = odor_facetgrids(mean_rmags, sns.lineplot, 'repeat_num',
            'trialmax_df_over_f', 'Repeat', ylabel, title, marker='o'
        )
        xs = np.unique(mean_rmags.repeat_num)
        xs = np.array([xs[0], xs[-1]])
        for oset, g in zip(odor_set_order, gs):
            assert len(g.row_names) == 0

            os_adaptation_fits = adaptation_fits.loc[(oset,)]
            assert set(g.col_names) == set(os_adaptation_fits.index)

            for name1, ax in zip(g.col_names, g.axes.flat):
                os_odor_fit = os_adaptation_fits.loc[name1]
                ys = os_odor_fit.intercept + os_odor_fit.slope * xs
                assert len(xs) == len(ys)
                # don't want this to change anything else about the plot
                # (x/ylim, etc)
                ax.plot(xs, ys, color='gray', linestyle='--', linewidth=2.0)

            # TODO TODO do try to patch a label for the fit line into the 
            # existing facetgrid legend somehow

        save_fn(f'adaptation_{fname_suffix}', section=title,
            section_order=adapt_sec_order
        )

        # TODO TODO maybe explicitly compare t1 - t0 and t2 - t2 adaptation?
        # if it's not linear, maybe measuring it with slope of line is
        # underestimating it's magnitude?

        # TODO since this takes kinda long, probably share fits in two loop
        # iterations (only fit on responders, then just subset fits)
        print('fitting adaptations for each cell...', end='', flush=True)
        before = time.time()
        cell_adaptation_fits = rmags.groupby(['odor_set','name1','fly_id','cell'
            ]).apply(lambda x: pd_linregress(x, 'repeat_num',
            'trialmax_df_over_f', verbose=False
        ))
        print(' done {:.2f}s'.format(time.time() - before))

        kde = False
        # By setting this to False, a lobe of positive slopes around ~2 (and
        # maybe past?) show up.
        at_least_3 = True
        if at_least_3:
            # Should only be missing trials from some of the kiwi recordings.
            assert ((cell_adaptation_fits.n < 3).groupby('odor_set').sum(
                ).control == 0)
            # TODO maybe print how many tossed here?
            cell_adaptation_fits = \
                cell_adaptation_fits[cell_adaptation_fits.n == 3]
        else:
            # Weird slopes in the n=2 fit cases (i.e. subset of kiwi data) cause
            # kde fit to fail in this case (could maybe fix by passing
            # equivalent of my fixed bins parameter? cause fixed bins are what
            # made the histograms look reasonable again.)
            kde = False

        #print('fraction of slopes that are NaN:',
        #    cell_adaptation_fits.isnull().sum().slope
        #    / len(cell_adaptation_fits))

        # TODO TODO probably try to show this on the plot somewhere. otherwise,
        # at least put it in the pdf w/ other params
        # TODO maybe try half this?
        percent_excluded = 2 * 0.5
        xlim_percentiles = [percent_excluded / 2, 100 - percent_excluded / 2]
        xlim = np.percentile(cell_adaptation_fits.slope, xlim_percentiles)
        # TODO maybe drop data outside some percentile and then let automatic
        # rule do it's thing?
        # TODO maybe play around w/ fixed bins some more, if i'm going to use
        # them, because in the reliable repsonder case, it does seem like the
        # automatic bins might make the effect larger than my fixed bins.
        fixed_bins = True
        if fixed_bins:
            bins = np.linspace(*xlim, num=(30 + 1))
        else:
            bins = None

        print_percentiles = False
        if print_percentiles:
            percentiles = [100 * f for f in (0.01, 0.005, 0.001)]
            percentiles += [100 - p for p in percentiles]
            percentiles = sorted(percentiles)

        # TODO TODO try multiple vals. maybe only do just before each plot so i
        # can still use kiwi data
        # TODO clean this index shuffling up. (at least to not hardcode second
        # set of key) (ideally don't also reset_index below)
        cell_adaptation_fits.reset_index(inplace=True)
        # TODO probably drop pfo before fitting on cells anyway...
        cell_adaptation_fits = drop_excluded_odors(cell_adaptation_fits)

        cell_adaptation_fits = add_odor_id(cell_adaptation_fits)

        cell_adaptation_fits.set_index(['odor_set','name1','fly_id','cell'],
            inplace=True)
        #

        # may have to be pretty careful about interpretation. motion could cause
        # some artifacts... maybe(?) in some kind of systematic way?
        # TODO could probably replace this loop with a facetgrid call
        # (i'm not actually using percentiles calculated w/in odor_sets
        # anyway...)
        cell_adapt_fig, ax = plt.subplots()
        for oset in odor_set_order:
            os_cell_slopes = cell_adaptation_fits.loc[(oset,), 'slope']

            # TODO TODO maybe do this (and stuff below that plots one thing per
            # odor set) both with and without kiwi
            old_idx_names = os_cell_slopes.index.names
            os_cell_slopes = os_cell_slopes.reset_index()
            os_cell_slopes = \
                os_cell_slopes[~ os_cell_slopes.name1.isin(u.natural)]

            os_cell_slopes = os_cell_slopes.set_index(old_idx_names).slope

            # TODO OK to weight all *cell* equally, yea?
            sns.distplot(os_cell_slopes, bins=bins, kde=kde, norm_hist=True,
                label=oset.title(), color=odor_set2color[oset],
                hist_kws=dict(odorset_distplot_hist_kws)
                #hist_kws=dict(alpha=0.4)
            )
            if print_percentiles:
                print(oset)
                for p, s in zip(percentiles,
                    np.percentile(os_cell_slopes, percentiles)):
                    print('{:0.2f} percentile slope: {:.2f}'.format(p, s))

        if print_percentiles:
            print('overall:')
            for p, s in zip(percentiles,
                np.percentile(cell_adaptation_fits.slope, percentiles)):

                print('{:0.2f} percentile slope: {:.2f}'.format(p, s))

        # TODO try to get a title that expresses more what this is / at least
        # what kind of response magnitude it's computed from
        if responders_only:
            ax.set_title('Adaptation of reliable responders')
        else:
            ax.set_title('Adaptation of all cells')

        slope_label = f'Slope between {trial_stat_desc} across repeats'

        ax.set_xlabel(slope_label)
        density_label = 'Normalized cell count'
        ax.set_ylabel(density_label)
        ax.set_xlim(xlim)
        ax.legend()
        savefigs(cell_adapt_fig, f'adapt_dist_{fname_suffix}', section=title,
            section_order=adapt_sec_order
        )

        cell_adaptation_fits.reset_index(inplace=True)
        cell_adaptation_fits_noreal = cell_adaptation_fits[
            ~ cell_adaptation_fits.name1.isin(u.natural)
        ]
        # TODO delete. was for figuring out appropriate kde bandwidth.
        '''
        # TODO TODO TODO so probably plot each odor in it's own facet and then
        # check that kde bandwith is set such that kdes always follow hist
        # pretty well 'scott' is the default. 1,2,3 all seemed WAY too smoothed.
        # scott and silverman seem identical. 0.25 still too smoothed.  must be
        # using "statsmodels backend", because "scipy treats (bw) as a scaling
        # factor for the stddev of the data" (so 1 should be ~default, right? or
        # what is default bw relative to stddev?)
        print('slope stddev: {:.2f}'.format(cell_adaptation_fits.slope.std()))
        # 0.1 seems maybe slightly more smoothed than scott. 0.05 much more so.
        # (and both scalars give kdes in n=probably 1 or 2 cases as well, which
        # i may not want)
        bws = ['scott', 0.1, 0.05]
        for bw in bws:
            g = sns.FacetGrid(cell_adaptation_fits, col='odor_id', col_wrap=4,
                xlim=xlim
            )
            g.map(sns.distplot, 'slope', bins=bins, kde_kws=dict(bw=bw))
            g.fig.suptitle(f'KDE bandwidth = {bw}')
        #
        plt.show()
        import ipdb; ipdb.set_trace()
        '''

        # TODO maye dotted line for kiwi kde or something? (if including)
        g = sns.FacetGrid(cell_adaptation_fits, col='odor_set',
            col_order=odor_set_order, hue='name1',
            palette=odor_hues_within_odorset
        )
        g.map(sns.distplot, 'slope', hist=False)
        # not using hist now just b/c it seems harder to follow multiple than
        # smoother kde for some reason...
        #bins=bins, hist_kws=dict(histtype='step'))
        g.set_axis_labels(slope_label, density_label)
        g.set_titles('{col_name}')
        g.add_legend()
        g._legend.set_title('Odor')
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.85)
        savefigs(g.fig, f'adapt_dist_per_odor_{fname_suffix}', section=title,
            section_order=adapt_sec_order
        )

        # TODO maybe don't use this fixed xlim
        # TODO maybe increase range a bit in odor specific case, as noiser +
        # differences may be in tails
        g = sns.FacetGrid(cell_adaptation_fits_noreal, col='odor_set',
            col_order=odor_set_order, height=4, xlim=(0, 7), ylim=xlim
        )
        # TODO maybe just separate plots of initial trial magnitude dists,
        # rather than this?
        g.map(sns.scatterplot, 'first_data_y', 'slope', color='black',
            alpha=0.3
        )
        g.set_axis_labels(f'First trial {trial_stat_desc}', slope_label)
        g.set_titles('{col_name}')
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.85)
        savefigs(g.fig, f'rmag0_v_adapt_{fname_suffix}', section=title,
            section_order=adapt_sec_order
        )

        # The green kinda won out over the pink of the control, so this plot
        # wasn't so good.
        '''
        sns.scatterplot(data=cell_adaptation_fits, x='first_data_y',
            y='slope', hue='odor_set', palette=odor_set2color, alpha=0.3, ax=ax
        )
        '''
        # TODO maybe still use these?
        '''
        for oset in odor_set_order:
            fig, ax = plt.subplots()
            sns.scatterplot(data=cell_adaptation_fits_noreal[
                cell_adaptation_fits_noreal.odor_set == oset],
                x='first_data_y', y='slope', hue='odor_set',
                palette=odor_set2color, ax=ax
            )
            ax.set_xlim((0, 7))
            ax.set_ylim((-3.5, 1.2))
        '''


# TODO TODO maybe plot (fly, trial) max / mean responses image grids, and
# include them in pdf (B said something like that might useful for determining
# if any pfo responsiveness is some analysis artifact or not / etc)


print('Calculations on computed intermediates took {:.0f}s'.format(
    time.time() - after_raw_calc_time_s))

if show_plots_interactively:
    plt.show()
else:
    plt.close('all')

# TODO TODO also provide plain text descriptions of these params
# TODO + optional units for params?
# TODO and support all of these in latex template

if not args.no_report_pdf:
    # So that we don't depend on the `latex` module this requires unless we
    # actually want to make a PDF report.
    import generate_pdf_report
    # TODO what does this mean if these are not equivalent?
    #from . import generate_pdf_report

    if args.no_save_figs:
        warnings.warn('--report-pdf was passed despite --no-save-figs! '
            'The report generated will ONLY contain any figures already in the '
            'PDF output directory!!!'
        )
        # (so this technically implies NOT -e)
        plots_made_this_run = None

        # Because we can't really say what parameters were used to generate the
        # existing plot files.
        params_for_report = None
        github_link = None
        uncommitted_changes = None
        outputs_pickle = None
        input_trace_pickles = None
    else:
        if args.earlier_figs:
            # Because None let's all matching files into the report, whereas
            # if filenames are passed, only the matching files that intersect
            # them are kept.
            plots_made_this_run = None
        else:
            plots_made_this_run = {p for p in plots_made_this_run if
                p.endswith('.pdf')
            }

        if fix_ref_odor_response_fracs:
            exclude_params = {'mean_zchange_response_thresh'}
        else:
            exclude_params = {'ref_odor', 'ref_response_percent'}

        report_param_names = [p for p in param_names if p not in exclude_params]

        # Since locals() is different inside the comprehension.
        _locals = locals()
        params_for_report = {n: _locals[n] for n in report_param_names}
        # So strings have single quotes around them. Then all the params section
        # should be usable a direct copy-paste fashion into the Python code.
        params_for_report = {n: (f"'{p}'" if type(p) is str else p) for n, p
            in params_for_report.items()
        }
        
        git_info = u.version_info()
        github_link = git_info['git_remote'].replace(':','/').replace(
            'git@','https://'
        )
        if github_link.endswith('.git'):
            github_link = github_link[:-len('.git')]
        github_link += '/blob/'

        uncommitted_changes = len(git_info['git_uncommitted_changes']) > 0
        # Only using a substring prefix, but with a reasonable number of
        # characters, this still has a very low chance of collisions.
        github_link += git_info['git_hash'][:20] + '/'

        this_script_fname = split(__file__)[1]
        github_link += this_script_fname

        # TODO way to check if local commits are also on remote / github
        # specifically?

        # TODO update this logic, corresponding logic above (need to save input
        # pickles into pickle), and templating s.t. input includes input
        # piclkes used to generate intermediate, as well as the intermediate
        # actually used as input
        if args.only_analyze_cached:
            params_for_report['args.earlier_figs'] = args.earlier_figs
            outputs_pickle = pickle_outputs_name
            input_trace_pickles = None
        else:
            outputs_pickle = None
            input_trace_pickles = pickles

        if input_trace_pickles is not None:
            input_trace_pickles = sorted(input_trace_pickles)

    # TODO didn't i have something in latex report generation to take an
    # iterable of globstrs as second element here? that could support my
    # sections + subsections, but i'm not seeing it in generate_pdf_report.py
    # now...

    plot_prefix2latex_data_list = sorted(plot_prefix2latex_data.items(),
        key=lambda x: (x[1]['section_order'], x[1]['subsection_order'],
        x[1]['order'])
    )

    # Downstream stuff may be more well behaved with less thought
    # if this assertion is True.
    assert (len(plot_prefix2latex_data_list) ==
        len({k for k, v in plot_prefix2latex_data_list})
    )
    section_and_subsec2last_plot_prefix = dict()
    for k, v in plot_prefix2latex_data_list:
        s = v['section']
        ss = v['subsection']
        section_and_subsec2last_plot_prefix[(s, ss)] = k
    pagebreak_after = set(section_and_subsec2last_plot_prefix.values())

    section_names_and_globstrs = [(d['section'], p + '*') for p, d in
        plot_prefix2latex_data_list if not d['paired']
    ]
    paired_section_names_and_globstrs = [(d['section'], p + '*') for p, d in
        plot_prefix2latex_data_list if d['paired']
    ]

    exclude_globstrs = {
        'pca_unstandardized*',
        'scree_unstandardized*',
        'pca*',
        'scree*'
    }
    section_names_and_globstrs = [(n, g) for n, g in
        section_names_and_globstrs if g not in exclude_globstrs
    ]
    paired_section_names_and_globstrs = [(n, g) for n, g in
        paired_section_names_and_globstrs if g not in exclude_globstrs
    ]
    # TODO change latex report generation to actually include per-section
    # notes, and then don't just throw them away here
    '''
    print('\nsection_names_and_globstrs:')
    pprint(section_names_and_globstrs)
    print('paired_section_names_and_globstrs:')
    pprint(paired_section_names_and_globstrs)
    import ipdb; ipdb.set_trace()
    '''

    # TODO try to also use u.format_keys in other places fly/rec keys are used
    # to make a title / fname
    fly_id2key_str = {i: u.format_keys(*k) for k, i in
        fly_keys2fly_id.iteritems()
    }
    print('')
    # TODO don't show "Within-fly analysis" section if empty?
    # (same for across fly)
    pdf_fname = generate_pdf_report.main(params=params_for_report,
        codelink=github_link, uncommitted_changes=uncommitted_changes,
        fly_id2key_str=fly_id2key_str,
        input_trace_pickles=input_trace_pickles,
        outputs_pickle=outputs_pickle,
        filenames=plots_made_this_run,
        section_names_and_globstrs=section_names_and_globstrs,
        paired_section_names_and_globstrs=paired_section_names_and_globstrs,
        pagebreak_after=pagebreak_after
    )

    # Linux specific.
    symlink_name = 'latest_report.pdf'
    print(f'Making a shortcut to this report at {symlink_name}')
    if exists(symlink_name):
        # In case it's pointing to something else.
        os.remove(symlink_name)
    os.symlink(pdf_fname, symlink_name)
    subprocess.Popen(['xdg-open', pdf_fname])

if args.debug_shell:
    import ipdb; ipdb.set_trace()

