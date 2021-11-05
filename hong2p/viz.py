"""
Functions for visualizing movies / cells / extracted traces, some intended to
provide context specfically useful for certain types of olfaction experiments.
"""

from os.path import join, exists
import time

import numpy as np
import pandas as pd

from hong2p import util, thor

# Having all matplotlib-related imports come after `hong2p.util` import,
# so that I can let `hong2p.util` set the backend, which it seems must be set
# before the first import of `matplotlib.pyplot`
# TODO maybe put these imports (and those in util) back to standard order if
# this backend thing is no longer an issue? was it just because of gui.py?
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# TODO use this other places that redefine, now that it's module-level
dff_latex = r'$\frac{\Delta F}{F}$'


def matlabels(df, rowlabel_fn):
    """
    Takes DataFrame and function that takes one row of index to a label.

    `rowlabel_fn` should take a DataFrame row (w/ columns from index) to a str.
    """
    return df.index.to_frame().apply(rowlabel_fn, axis=1)


# TODO maybe change __init__.py to make this directly accessible under hong2p?
def matshow(df, title=None, ticklabels=None, xticklabels=None,
    yticklabels=None, xtickrotation=None, colorbar_label=None,
    group_ticklabels=False, ax=None, fontsize=None, fontweight=None, figsize=None,
    transpose_sort_key=None, colorbar=True, shrink=None, colorbar_kwargs=None,
    **kwargs):
    """
    Args:
        transpose_sort_key (None | function): takes df.index/df.columns and compares
            output to decide whether matrix should be transposed before plotting

        **kwargs: passed thru to `matplotlib.pyplot.matshow`
    """
    # TODO TODO w/ all ticklabels kwargs, also support them being functions,
    # which operate on (what type exactly?) index rows
    # TODO shouldn't this get ticklabels from matrix if nothing else?
    # maybe at least in the case when both columns and row indices are all just
    # one level of strings?

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if figsize is not None:
            raise ValueError('figsize only allowed if ax not passed in')

        fig = ax.get_figure()

    # NOTE: if i'd like to also sort on [x/y]ticklabels, would need to move this block
    # after possible ticklabel enumeration, and then assign correctly to index/cols and
    # use that as input to sort_key_val in appropriate instead
    if transpose_sort_key is not None:
        if any([x is not None for x in [ticklabels, xticklabels, yticklabels]]):
            raise NotImplementedError('transpose_sort_key not supported if any '
                'ticklabels are explicitly passed'
            )

        row_sort_key = transpose_sort_key(df.index)
        col_sort_key = transpose_sort_key(df.columns)

        if row_sort_key > col_sort_key:
            df = df.T

    def one_level_str_index(index):
        return (len(index.shape) == 1 and
            all(index.map(lambda x: type(x) is str)))

    if (xticklabels is None) and (yticklabels is None):
        if ticklabels is None:
            # TODO maybe default to joining str representations of values at each level,
            # joined on some kwarg delim like '/'?
            if one_level_str_index(df.columns):
                xticklabels = df.columns
            else:
                xticklabels = None

            if one_level_str_index(df.index):
                yticklabels = df.index
            else:
                yticklabels = None
        else:
            # TODO maybe also assert indices are actually equal?
            assert df.shape[0] == df.shape[1]

            if callable(ticklabels):
                ticklabels = matlabels(df, ticklabels)

            # TODO should i check output is same on rows/cols in this case?
            # currently, matlabels seems to just operate on the row labels...
            xticklabels = ticklabels
            yticklabels = ticklabels
    else:
        # TODO fix. probably need to specify axes of df or something.
        # (maybe first modifying matlabels to accept that...)
        '''
        if callable(xticklabels):
            xticklabels = matlabels(df, xticklabels)

        if callable(yticklabels):
            yticklabels = matlabels(df, yticklabels)
        '''
        pass

    # TODO update this formula to work w/ gui corrs (too big now)
    if fontsize is None:
        fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))

    im = ax.matshow(df, **kwargs)

    if colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = dict()

        # rotation=270?
        cbar = add_colorbar(fig, im, label=colorbar_label, shrink=shrink,
            **colorbar_kwargs
        )

    def grouped_labels_info(labels):
        if not group_ticklabels or labels is None:
            return labels, 1, 0

        labels = pd.Series(labels)
        n_repeats = int(len(labels) / len(labels.unique()))

        # TODO TODO assert same # things from each unique element.
        # that's what this whole tickstep thing seems to assume.

        # Assumes order is preserved if labels are grouped at input.
        # May need to calculate some other way if not always true.
        labels = labels.unique()
        tick_step = n_repeats
        offset = n_repeats / 2 - 0.5
        return labels, tick_step, offset

    # TODO automatically only group labels in case where all repeats are
    # adjacent?
    # TODO make fontsize / weight more in group_ticklabels case?
    xticklabels, xstep, xoffset = grouped_labels_info(xticklabels)
    yticklabels, ystep, yoffset = grouped_labels_info(yticklabels)

    if xticklabels is not None:
        # TODO nan / None value aren't supported in ticklabels are they?
        # (couldn't assume len is defined if so)
        if xtickrotation is None:
            if all([len(x) == 1 for x in xticklabels]):
                xtickrotation = 'horizontal'
            else:
                xtickrotation = 'vertical'

        ax.set_xticks(np.arange(0, len(df.columns), xstep) + xoffset)
        ax.set_xticklabels(
            xticklabels, fontsize=fontsize, fontweight=fontweight,
            rotation=xtickrotation
        #    rotation='horizontal' if group_ticklabels else 'vertical'
        )

    if yticklabels is not None:
        ax.set_yticks(np.arange(0, len(df), ystep) + yoffset)
        ax.set_yticklabels(
            yticklabels, fontsize=fontsize, fontweight=fontweight, rotation='horizontal'
        #    rotation='vertical' if group_ticklabels else 'horizontal'
        )

    # TODO test this doesn't change rotation if we just set rotation above

    # this doesn't seem like it will work, since it seems to clear the default
    # ticklabels that there actually were...
    #ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize,
    #    fontweight=fontweight
    #)

    # didn't seem to do what i was expecting
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False)

    if title is not None:
        ax.set_xlabel(title, fontsize=(fontsize + 1.5), labelpad=12)

    return fig, im


def imshow(img, title=None, cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)

    if title is not None:
        ax.set_title(title)

    ax.axis('off')
    return fig


def add_colorbar(fig, im, label=None, shrink=None, **kwargs):

    # I think this relies on use of Matplotlib's constrained layout
    cbar = fig.colorbar(im, ax=fig.axes, shrink=shrink, **kwargs)

    if label is not None:
        cbar.ax.set_ylabel(label)

    return cbar


def image_grid(image_list, **imshow_kwargs):
    # TODO TODO see: https://stackoverflow.com/questions/42850225 or related to find a
    # good solution for reliably eliminating all unwanted whitespace between subplots
    n = int(np.ceil(np.sqrt(len(image_list))))
    fig, axs = plt.subplots(n,n)
    for ax, img in zip(axs.flat, image_list):
        ax.imshow(img, cmap='gray', **imshow_kwargs)

    for ax in axs.flat:
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.05)
    return fig


# TODO should i actually compute correlations in here too? check input, and
# compute if input wasn't correlations (/ symmetric?)?
# if so, probably return them as well.
def plot_odor_corrs(corr_df, odor_order=False, odors_in_order=None,
    trial_stat='mean', title_suffix='', **kwargs):
    """Takes a symmetric DataFrame with odor x odor correlations and plots it.
    """
    # TODO TODO TODO test this fn w/ possible missing data case.
    # bring guis support for that in here?
    if odors_in_order is not None:
        odor_order = True

    if odor_order:
        # 'name2' is just 'no_second_odor' for a lot of my data
        # (the non-pair stuff)
        name_prefix = 'name1'

        # TODO probably refactor the two duped things below
        odor_name_rows = [c for c in corr_df.index.names
            if c.startswith(name_prefix)
        ]
        if len(odor_name_rows) != 1:
            raise ValueError('expected the name of exactly one index level to '
                f'start with {name_prefix}'
            )
        odor_name_row = odor_name_rows[0]

        odor_name_cols = [c for c in corr_df.columns.names
            if c.startswith(name_prefix)
        ]
        if len(odor_name_cols) != 1:
            raise ValueError('expected the name of exactly one column level to '
                f'start with {name_prefix}'
            )
        odor_name_col = odor_name_cols[0]
        #

        if len(corr_df.index.names) == 1:
            assert len(corr_df.columns.names) == 1
            # Necessary to avoid this error:
            # KeyError: 'Requested level...does not match index name (None)'
            odor_name_row = None
            odor_name_col = None

        corr_df = corr_df.reindex(odors_in_order, level=odor_name_row,
            axis='index').reindex(odors_in_order, level=odor_name_col,
            axis='columns'
        )
        if odors_in_order is None:
            # TODO 
            raise NotImplementedError

        if 'group_ticklabels' not in kwargs:
            kwargs['group_ticklabels'] = True
    else:
        corr_df = corr_df.sort_index(
            axis=0, level='order', sort_remaining=False).sort_index(
            axis=1, level='order', sort_remaining=False
        )

    if 'title' not in kwargs:
        kwargs['title'] = ('Odor' if odor_order else 'Presentation') + ' order'
        kwargs['title'] += title_suffix

    if 'ticklabels' not in kwargs:
        kwargs['ticklabels'] = util.format_mixture

    if 'colorbar_label' not in kwargs:
        # TODO factor out latex for delta f / f stuff (+ maybe use in analysis that uses
        # this pkg: kc_natural_mixes, al_pair_grids)
        kwargs['colorbar_label'] = \
            trial_stat.title() + r' response $\frac{\Delta F}{F}$ correlation'

    return matshow(corr_df, **kwargs)


# TODO get x / y from whether they were declared share<x/y> in facetgrid
# creation?
def fix_facetgrid_axis_labels(facet_grid, shared_in_center=False,
    x=True, y=True) -> None:
    """Modifies a FacetGrid to not duplicate X and Y axis text labels.
    """
    # regarding the choice of shared_in_center: WWMDD?
    if shared_in_center:
        # TODO maybe add a axes over / under the FacetGrid axes, with the same
        # shape, and label that one (i think i did this in my gui or one of the
        # plotting fns. maybe plot_traces?)
        raise NotImplementedError
    else:
        for ax in facet_grid.axes.flat:
            if not (ax.is_first_col() and ax.is_last_row()):
                if x:
                    ax.set_xlabel('')
                if y:
                    ax.set_ylabel('')


def set_facetgrid_legend(facet_grid, **kwargs) -> None:
    """
    In cases where different axes have different subsets of the hue levels,
    the legend may not contain the artists for the union of hue levels across
    all axes. This sets a legend from the hue artists across all axes.
    """
    #from matplotlib.collections import PathCollection
    legend_data = dict()
    for ax in facet_grid.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for label, h in zip(labels, handles):
            #if type(h) is PathCollection:
            # From inspecting facet_grid._legend_data in cases where some labels
            # pointed to empty lines (the phenotype in the case where things
            # weren't behaving as I wanted), the empty lines had this empty
            # facecolor.
            facecolor = h.get_facecolor()
            if len(facecolor) == 0:
                continue
            #else:
            #    print(type(h))
            #    import ipdb; ipdb.set_trace()

            if label in legend_data:
                # TODO maybe assert a wide variety of properties of the
                # matplotlib.collections.PathCollection objects are the same
                # (line width, dash, etc)
                past_facecolor = legend_data[label].get_facecolor()
                # TODO TODO TODO fix! this is failing again 2020-08-25
                # (after re-installing requirements.txt, when running
                # kc_mix_analysis.py w/ no just -j arg)
                assert np.array_equal(facecolor, past_facecolor), \
                    f'{facecolor} != {past_facecolor}'
            else:
                legend_data[label] = h

    facet_grid.add_legend(legend_data, **kwargs)


# TODO generalize / move to project specific repo
def plot_traces(*args, footprints=None, order_by='odors', scale_within='cell',
    n=20, random=False, title=None, response_calls=None, raw=False,
    smoothed=True, show_footprints=True, show_footprints_alone=False,
    show_cell_ids=True, show_footprint_with_mask=False, gridspec=None,
    linewidth=0.5, verbose=True):
    # TODO TODO be clear on requirements of df and cell_ids in docstring
    """
    n (int): (default=20) Number of cells to plot traces for if cell_ids not
        passed as second positional argument.
    random (bool): (default=False) Whether the `n` cell ids should be selected
        randomly. If False, the first `n` cells are used.
    order_by (str): 'odors' or 'presentation_order'
    scale_within (str): 'none', 'cell', or 'trial'
    gridspec (None or matplotlib.gridspec.*): region of a parent figure
        to draw this plot on.
    linewidth (float): 0.25 seemed ok on CNMF data, but too small w/ clean
    traces.
    """
    import tifffile
    import cv2
    # TODO maybe use cv2 and get rid of this dep?
    from skimage import color

    # TODO make text size and the spacing of everything more invariant to figure
    # size. i think the default size of this figure ended up being bigger when i
    # was using it in kc_analysis than it is now in the gui, so it isn't display
    # well in the gui, but fixing it here might break it in the kc_analysis case
    if verbose:
        print('Entering plot_traces...')

    if len(args) == 1:
        df = args[0]
        # TODO flag to also subset to responders first?
        all_cells = cell_ids(df)
        n = min(n, len(all_cells))
        if random:
            # TODO maybe flag to disable seed?
            cells = all_cells.sample(n=n, random_state=1)
        else:
            cells = all_cells[:n]

    elif len(args) == 2:
        df, cells = args

    else:
        raise ValueError('must call with either df or df and cells')

    if show_footprints:
        # or maybe just download (just the required!) footprints from sql?
        if footprints is None:
            raise ValueError('must pass footprints kwarg if show_footprints')
        # decide whether this should be in the preconditions or just done here
        # (any harm to just do here anyway?)
        #else:
        #    footprints = footprints.set_index(recording_cols + ['cell'])

    # TODO TODO TODO fix odor labels as in matrix (this already done?)
    # (either rotate or use abbreviations so they don't overlap!)

    # TODO check order_by and scale_within are correct
    assert raw or smoothed

    # TODO maybe automatically show_cells if show_footprints is true,
    # otherwise don't?
    # TODO TODO maybe indicate somehow the multiple response criteria
    # when it is a list (add border & color each half accordingly?)

    extra_cols = 0
    # TODO which of these cases do i want to support here?
    if show_footprints:
        if show_footprints_alone:
            extra_cols = 2
        else:
            extra_cols = 1
    elif show_footprints_alone:
        raise NotImplementedError

    # TODO possibility of other column for avg + roi overlays
    # possible to make it larger, or should i use a layout other than
    # gridspec? just give it more grid elements?
    # TODO for combinatorial combinations of flags enabling cols on
    # right, maybe set index for each of those flags up here

    # TODO could also just could # trials w/ drop_duplicates, for more
    # generality
    n_repeats = n_expected_repeats(df)
    n_trials = n_repeats * len(df[['name1','name2']].drop_duplicates())

    if gridspec is None:
        # This seems to hang... not sure if it's usable w/ some changes.
        #fig = plt.figure(constrained_layout=True)
        fig = plt.figure()
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
        made_fig = True
    else:
        fig = gridspec.get_topmost_subplotspec().get_gridspec().figure
        gs = gridspec.subgridspec(4, 6, hspace=0.4, wspace=0.3)
        made_fig = False

    if show_footprints:
        trace_gs_slice = gs[:3,:4]
    else:
        trace_gs_slice = gs[:,:]

    # For common X/Y labels
    bax = fig.add_subplot(trace_gs_slice, frameon=False)
    # hide tick and tick label of the big axes
    bax.tick_params(top=False, bottom=False, left=False, right=False,
        labelcolor='none')
    bax.grid(False)

    trace_gs = trace_gs_slice.subgridspec(len(cells),
        n_trials + extra_cols, hspace=0.15, wspace=0.06)

    axs = []
    for ti in range(trace_gs._nrows):
        axs.append([])
        for tj in range(trace_gs._ncols):
            ax = fig.add_subplot(trace_gs[ti,tj])
            axs[-1].append(ax)
    axs = np.array(axs)

    # TODO want all of these behind show_footprints?
    if show_footprints:
        # TODO use 2/3 for widgets?
        # TODO or just text saying which keys to press? (if only
        # selection mechanism is going to be hover, mouse clicks
        # wouldn't make sense...)

        avg_ax = fig.add_subplot(gs[:, -2:])
        # TODO TODO maybe show trial movie beneath this?
        # (also on hover/click like (trial,cell) data)

    if title is not None:
        #pad = 40
        pad = 15
        # was also using default fontsize here in kc_analysis use case
        # increment pad by 5 for each newline in title?
        bax.set_title(title, pad=pad, fontsize=9)

    bax.set_ylabel('Cell')

    # This pad is to make it not overlap w/ time label on example plot.
    # Was left to default value for kc_analysis.
    # TODO negative labelpad work? might get drawn over by axes?
    labelpad = -10
    if order_by == 'odors':
        bax.set_xlabel('Trials ordered by odor', labelpad=labelpad)
    elif order_by == 'presentation_order':
        bax.set_xlabel('Trials in presentation order', labelpad=labelpad)

    ordering = pair_ordering(df)

    '''
    display_start_time = -3.0
    display_stop_time = 10
    display_window = df[
        (comparison_df.from_onset >= display_start_time) &
        (comparison_df.from_onset <= display_stop_time)]
    '''
    display_window = df

    smoothing_window_secs = 1.0
    fps = thor.fps_from_thor(df)
    window_size = int(np.round(smoothing_window_secs * fps))

    group_cols = trial_cols + ['order']

    xmargin = 1
    xmin = display_window.from_onset.min() - xmargin
    xmax = display_window.from_onset.max() + xmargin

    response_rgb = (0.0, 1.0, 0.2)
    nonresponse_rgb = (1.0, 0.0, 0.0)
    response_call_alpha = 0.2

    if scale_within == 'none':
        ymin = None
        ymax = None

    cell2contour = dict()
    cell2rect = dict()
    cell2text_and_rect = dict()

    seen_ij = set()
    avg = None
    for i, cell_id in enumerate(cells):
        if verbose:
            print('Plotting cell {}/{}...'.format(i + 1, len(cells)))

        cell_data = display_window[display_window.cell == cell_id]
        cell_trials = cell_data.groupby(group_cols, sort=False)[
            ['from_onset','df_over_f']
        ]

        prep_date = pd.Timestamp(cell_data.prep_date.unique()[0])
        date_dir = prep_date.strftime(date_fmt_str)
        fly_num = cell_data.fly_num.unique()[0]
        thorimage_id = cell_data.thorimage_id.unique()[0]

        #assert len(cell_trials) == axs.shape[1]

        if show_footprints:
            if avg is None:
                # only uncomment to support dff images and other stuff like that
                '''
                try:
                    # TODO either put in docstring that datetime.datetime is
                    # required, or cast input date as appropriate
                    # (does pandas date type support strftime?)
                    # or just pass date_dir?
                    # TODO TODO should not use nr if going to end up using the
                    # rig avg... but maybe lean towards computing the avg in
                    # that case rather than deferring to rigid?
                    tif = motion_corrected_tiff_filename(
                        prep_date, fly_num, thorimage_id
                    )
                except IOError as e:
                    if verbose:
                        print(e)
                    continue

                # TODO maybe show progress bar / notify on this step
                if verbose:
                    print('Loading full movie from {} ...'.format(tif),
                        end='', flush=True
                    )
                movie = tifffile.imread(tif)
                if verbose:
                    print(' done.')
                '''

                # TODO modify motion_corrected_tiff_filename to work in this
                # case too?
                tif_dir = join(analysis_output_root(), date_dir, str(fly_num),
                    'tif_stacks'
                )
                avg_nr_tif = join(tif_dir, 'AVG', 'nonrigid',
                    'AVG{}_nr.tif'.format(thorimage_id)
                )
                avg_rig_tif = join(tif_dir, 'AVG', 'rigid',
                    'AVG{}_rig.tif'.format(thorimage_id)
                )

                avg_tif = None
                if exists(avg_nr_tif):
                    avg_tif = avg_nr_tif
                elif exists(avg_rig_tif):
                    avg_tif = avg_rig_tif

                if avg_tif is None:
                    raise IOError(('No average motion corrected TIFs ' +
                        'found in {}').format(tif_dir)
                    )

                avg = tifffile.imread(avg_tif)
                '''
                avg = cv2.normalize(avg, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
                )
                '''
                # TODO find a way to histogram equalize w/o converting
                # to 8 bit?
                avg = cv2.normalize(avg, None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
                )
                better_constrast = cv2.equalizeHist(avg)

                rgb_avg = color.gray2rgb(better_constrast)

            cell_row = (prep_date, fly_num, thorimage_id, cell_id)
            footprint_row = footprints.loc[cell_row]

            # TODO TODO TODO probably need to transpose how footprint is handled
            # downstream (would prefer not to transpose footprint though)
            # (as i had to switch x_coords and y_coords in db as they were
            # initially entered swapped)
            footprint = db_row2footprint(footprint_row, shape=avg.shape)

            # TODO maybe some percentile / fixed size about maximum
            # density?
            cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
                crop_to_nonzero(footprint, margin=6)

            cell2rect[cell_id] = (x_min, x_max, y_min, y_max)

            cropped_avg = \
                better_constrast[x_min:x_max + 1, y_min:y_max + 1]

            if show_footprint_with_mask:
                # TODO figure out how to suppress clipping warning in the case
                # when it's just because of float imprecision (e.g. 1.0000001
                # being clipped to 1) maybe just normalize to [0 + epsilon, 1 -
                # epsilon]?
                # TODO TODO or just set one channel to be this
                # footprint?  scale first?
                cropped_footprint_rgb = \
                    color.gray2rgb(cropped_footprint)

                for c in (1,2):
                    cropped_footprint_rgb[:,:,c] = 0
                # TODO plot w/ value == 1 to test?

                cropped_footprint_hsv = \
                    color.rgb2hsv(cropped_footprint_rgb)

                cropped_avg_hsv = \
                    color.rgb2hsv(color.gray2rgb(cropped_avg))

                # TODO hue already seems to be constant at 0.0 (red?)
                # so maybe just directly set to red to avoid confusion?
                cropped_avg_hsv[..., 0] = cropped_footprint_hsv[..., 0]

                alpha = 0.3
                cropped_avg_hsv[..., 1] = cropped_footprint_hsv[..., 1] * alpha

                composite = color.hsv2rgb(cropped_avg_hsv)

                # TODO TODO not sure this is preserving hue/sat range to
                # indicate how strong part of filter is
                # TODO figure out / find some way that would
                # TODO TODO maybe don't normalize within each ROI? might
                # screw up stuff relative to histogram equalized
                # version...
                # TODO TODO TODO still normalize w/in crop in contour
                # case?
                composite = cv2.normalize(composite, None, alpha=0.0,
                    beta=1.0, norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F
                )

            else:
                # TODO could also use something more than this
                # TODO TODO fix bug here. see 20190402_bug1.txt
                # TODO TODO where are all zero footprints coming from?
                cropped_footprint_nonzero = cropped_footprint > 0
                if not np.any(cropped_footprint_nonzero):
                    continue

                level = \
                    cropped_footprint[cropped_footprint_nonzero].min()

            if show_footprints_alone:
                ax = axs[i,-2]
                f_ax = axs[i,-1]
                f_ax.imshow(cropped_footprint, cmap='gray')
                f_ax.axis('off')
            else:
                ax = axs[i,-1]

            if show_footprint_with_mask:
                ax.imshow(composite)
            else:
                ax.imshow(cropped_avg, cmap='gray')
                # TODO TODO also show any other contours in this rectangular ROI
                # in a diff color! (copy how gui does this)
                cell2contour[cell_id] = \
                    util.closed_mpl_contours(cropped_footprint, ax, colors='red')

            ax.axis('off')

            text = str(cell_id + 1)
            h = y_max - y_min
            w = x_max - x_min
            rect = patches.Rectangle((y_min, x_min), h, w,
                linewidth=1.5, edgecolor='b', facecolor='none')
            cell2text_and_rect[cell_id] = (text, rect)

        if scale_within == 'cell':
            ymin = None
            ymax = None

        for n, cell_trial in cell_trials:
            #(prep_date, fly_num, thorimage_id,
            (_, _, _, comp, o1, o2, repeat_num, order) = n

            # TODO TODO also support a 'fixed' order that B wanted
            # (which should also include missing stuff[, again in gray,]
            # ideally)
            if order_by == 'odors':
                j = n_repeats * ordering[(o1, o2)] + repeat_num

            elif order_by == 'presentation_order':
                j = order

            else:
                raise ValueError("supported orderings are 'odors' and "+
                    "'presentation_order'"
                )

            if scale_within == 'trial':
                ymin = None
                ymax = None

            assert (i,j) not in seen_ij
            seen_ij.add((i,j))
            ax = axs[i,j]

            # So that events that get the axes can translate to cell /
            # trial information.
            ax.cell_id = cell_id
            ax.trial_info = n

            # X and Y axis major tick label fontsizes.
            # Was left to default for kc_analysis.
            ax.tick_params(labelsize=6)

            trial_times = cell_trial['from_onset']

            # TODO TODO why is *first* ea trial the one not shown, and
            # apparently the middle pfo trial
            # (was it not actually ordered by 'order'/frame_num outside of
            # odor order???)
            # TODO TODO TODO why did this not seem to work? (or only for
            # 1/3.  the middle one. iaa.)
            # (and actually title is still hidden for ea and pfo trials
            # mentioned above, but numbers / ticks / box still there)
            # (above notes only apply to odor order case. presentation order
            # worked)
            # TODO and why is gray title over correct axes in odor order case,
            # but axes not displaying data are in wrong place?
            # TODO is cell_trial messed up?

            # Supports at least the case when there are missing odor
            # presentations at the end of the ~block.
            missing_this_presentation = \
                trial_times.shape == (1,) and pd.isnull(trial_times.iat[0])

            if i == 0:
                # TODO group in odors case as w/ matshow?
                if order_by == 'odors':
                    trial_title = util.format_mixture({
                        'name1': o1,
                        'name2': o2,
                    })
                elif order_by == 'presentation_order':
                    trial_title = util.format_mixture({
                        'name1': o1,
                        'name2': o2
                    })

                if missing_this_presentation:
                    tc = 'gray'
                else:
                    tc = 'black'

                ax.set_title(trial_title, fontsize=6, color=tc)
                # TODO may also need to do tight_layout here...
                # it apply to these kinds of titles?

            if missing_this_presentation:
                ax.axis('off')
                continue

            trial_dff = cell_trial['df_over_f']

            if raw:
                if ymax is None:
                    ymax = trial_dff.max()
                    ymin = trial_dff.min()
                else:
                    ymax = max(ymax, trial_dff.max())
                    ymin = min(ymin, trial_dff.min())

                ax.plot(trial_times, trial_dff, linewidth=linewidth)

            if smoothed:
                # TODO kwarg(s) to control smoothing?
                sdff = smooth(trial_dff, window_len=window_size)

                if ymax is None:
                    ymax = sdff.max()
                    ymin = sdff.min()
                else:
                    ymax = max(ymax, sdff.max())
                    ymin = min(ymin, sdff.min())

                # TODO TODO have plot_traces take kwargs to be passed to
                # plotting fn + delete separate linewidth
                ax.plot(trial_times, sdff, color='black', linewidth=linewidth)

            # TODO also / separately subsample?

            if response_calls is not None:
                was_a_response = \
                    response_calls.loc[(o1, o2, repeat_num, cell_id)]

                if was_a_response:
                    ax.set_facecolor(response_rgb +
                        (response_call_alpha,))
                else:
                    ax.set_facecolor(nonresponse_rgb +
                        (response_call_alpha,))

            if i == axs.shape[0] - 1 and j == 0:
                # want these centered on example plot or across all?

                # I had not specified fontsize for kc_analysis case, so whatever
                # the default value was probably worked OK there.
                ax.set_xlabel('Seconds from odor onset', fontsize=6)

                if scale_within == 'none':
                    scaletext = ''
                elif scale_within == 'cell':
                    scaletext = '\nScaled within each cell'
                elif scale_within == 'trial':
                    scaletext = '\nScaled within each trial'

                # TODO just change to "% maximum w/in <x>" or something?
                # Was 70 for kc_analysis case. That's much too high here.
                #labelpad = 70
                labelpad = 10
                # TODO factor out latex for delta f / f stuff
                ax.set_ylabel(r'$\frac{\Delta F}{F}$' + scaletext,
                    rotation='horizontal', labelpad=labelpad
                )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            else:
                if show_cell_ids and j == len(cell_trials) - 1:
                    # Indexes as they would be from one. For comparison
                    # w/ Remy's MATLAB analysis.
                    # This and default fontsize worked for kc_analysis case,
                    # not for GUI.
                    #labelpad = 18
                    labelpad = 25
                    ax.set_ylabel(str(cell_id + 1),
                        rotation='horizontal', labelpad=labelpad, fontsize=5)
                    ax.yaxis.set_label_position('right')
                    # TODO put a label somewhere on the plot indicating
                    # these are cell IDs

                for d in ('top', 'right', 'bottom', 'left'):
                    ax.spines[d].set_visible(False)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        # TODO change units in this case on ylabel?
        # (to reflect how it was scaled)
        if scale_within == 'cell':
            for r in range(len(cell_trials)):
                ax = axs[i,r]
                ax.set_ylim(ymin, ymax)

    if scale_within == 'none':
        for i in range(len(cells)):
            for j in range(len(cell_trials)):
                ax = axs[i,j]
                ax.set_ylim(ymin, ymax)
    
    if show_footprints:
        fly_title = '{}, fly {}, {}'.format(
            date_dir, fly_num, thorimage_id)

        # title like 'Recording average fluorescence'?
        #avg_ax.set_title(fly_title)
        avg_ax.imshow(rgb_avg)
        avg_ax.axis('off')

        cell2rect_artists = dict()
        for cell_id in cells:
            # TODO TODO fix bug that required this (zero nonzero pixel
            # in cropped footprint thing...)
            if cell_id not in cell2text_and_rect:
                continue

            (text, rect) = cell2text_and_rect[cell_id]

            box = rect.get_bbox()
            # TODO appropriate font properties? placement still good?
            # This seemed to work be for (larger?) figures in kc_analysis,
            # too large + too close to boxes in gui (w/ ~8"x5" gridspec,dpi 100)
            # TODO set in relation to actual fig size (+ dpi?)
            #boxlabel_fontsize = 9
            boxlabel_fontsize = 6
            text_artist = avg_ax.text(box.xmin, box.ymin - 2, text,
                color='b', size=boxlabel_fontsize, fontweight='bold')
            # TODO jitter somehow (w/ arrow pointing to box?) to ensure no
            # overlap? (this would be ideal, but probably hard to implement)
            avg_ax.add_patch(rect)

            cell2rect_artists[cell_id] = (text_artist, rect)

    for i in range(len(cells)):
        for j in range(len(cell_trials)):
            ax = axs[i,j]
            ax.set_xlim(xmin, xmax)

    if made_fig:
        fig.tight_layout()
        return fig


def showsync(thorsync_dir, verbose=False, **kwargs):
    """Shows ThorSync .h5 data interactively, using `pyqtgraph`.

    Args:
        thorsync_dir: path to a directory created by ThorSync
        kwargs: passed through to `thor.load_thorsync_hdf5`
    """
    import pyqtgraph as pg

    # TODO TODO make it possible to click to toggle display of lines like in thorsync
    # itself

    app = pg.mkQApp()

    before = time.time()
    print('loading HDF5...', flush=True, end='\n' if verbose else '')

    df = thor.load_thorsync_hdf5(thorsync_dir, verbose=verbose, **kwargs)

    took_s = time.time() - before
    print(f'done ({took_s:.1f}s)')

    # TODO set title to last 3 parts of path if parent two directories can be parsed as
    # (date, fly_num) (or if just under raw_data_root maybe?)
    title = thorsync_dir

    # plot_widget if of type pyqtgraph.widgets.PlotWidget.PlotWidget
    # not really sure how 'all' and 'pairs' differ... ('pairs' skip some?)
    # TODO wanted to set clipToView=True and some other stuff here, but downsampling=100
    # didn't seem to work at all (unlike call below), so i'm not sure anythere here
    # will.
    plot_widget = pg.plot(title=title)

    plot_widget.setLabels(bottom='Seconds')

    # Passing `autoDownsample=True` to either the `plot` call above or those in the loop
    # below caused the startup time to increase beyond what is acceptable. Was also slow
    # after loading. Not sure why it only seems to work here.
    #plot_widget.setDownsampling(100)
    # TODO TODO could downsampling cause problems? how? maybe add cli flag to disable
    # it?
    # TODO TODO in `auto` case, is automatically selected factor changed when zooming?
    # (or would i (could i even?) need to link settting it to change in scale or
    # something?)
    plot_widget.setDownsampling(auto=True)

    # Much snappier with this. Doesn't try to draw data out of bounds of view box.
    plot_widget.setClipToView(True)

    # TODO set initial view box and / or build in margin such that legend is out of way
    # of traces immediately

    plot_widget.addLegend()

    time_col = thor.time_col

    x_min = 0
    x_max = df[time_col].iat[-1]
    y_min = -1
    y_max = 6
    # TODO maybe add a margin around this, but only if it can be made clear where the
    # region the data actually exists in is (not just black everywhere...)
    plot_widget.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)

    plot_cols = [c for c in df.columns if c != time_col]
    for i, c in enumerate(plot_cols):
        if c == 'frame_counter':
            if verbose:
                print('not plotting frame_counter')
            continue

        col = df[c]
        # Otherwise this produces an in error in pyqtgraph internals
        if col.dtype == np.dtype('bool'):
            col = col.astype(np.dtype('float64'))

        # TODO maybe store hardcoded map for colors for these?
        # not sure how consistent column order is as loaded from hdf5...
        plot_widget.plot(df[time_col], col, name=c, pen=(i, len(plot_cols)))

    app.exec_()


# TODO maybe one fn that puts in matrix format and another in table
# (since matrix may be sparse...)
# TODO delete / move to project specific repo / generalize
# (currently only used in no-longer-used natural_odors/kc_analysis.py)
def plot_pair_n(df, *args):
    """Plots a matrix of odor1 X odor2 w/ counts of flies as entries.

    Args:
    df (pd.DataFrame): DataFrame with columns:
        - prep_date
        - fly_num
        - thorimage_id
        - name1
        - name2
        Data already collected w/ odor pairs.

    odor_panel (pd.DataFrame): (optional) DataFrame with columns:
        - odor_1
        - odor_2
        - reason (maybe make this optional?)
        The odor pairs experiments are supposed to collect data for.
    """
    import imgkit
    import seaborn as sns

    odor_panel = None
    if len(args) == 1:
        odor_panel = args[0]
    elif len(args) != 0:
        raise ValueError('incorrect number of arguments')
    # TODO maybe make df optional and read from database if it's not passed?
    # TODO a flag to show all stuff marked attempt analysis in gsheet?

    # TODO borrow more of this / call this in part of kc_analysis that made that
    # table w/ these counts for repeats?

    # TODO also handle no_second_odor
    df = df.drop(
        index=df[(df.name1 == 'paraffin') | (df.name2 == 'paraffin')].index)

    # TODO possible to do at least a partial check w/ n_accepted_blocks sum?
    # (would have to do outside of this fn. presentations here doesn't have it.
    # whatever latest_analysis returns might.)

    replicates = df[
        ['prep_date','fly_num','recording_from','name1','name2']
    ].drop_duplicates()

    # TODO do i actually want margins? (would currently count single odors twice
    # if in multiple comparison... may at least not want that?)
    # hide margins for now.
    pair_n = pd.crosstab(replicates.name1, replicates.name2) #, margins=True)

    # Making the rectangular matrix pair_n square
    # (same indexes on row and column axes)

    if odor_panel is None:
        # This is basically equivalent to the logic in the branch below,
        # although the index is not defined separately here.
        full_pair_n = pair_n.combine_first(pair_n.T).fillna(0.0)
    else:
        # TODO [change odor<n> to / handle] name<n>, to be consistent w/ above
        # TODO TODO TODO also make this triangular / symmetric
        odor_panel = odor_panel.pivot_table(index='odor_1', columns='odor_2',
            aggfunc=lambda x: True, values='reason')

        full_panel_index = odor_panel.index.union(odor_panel.columns)
        full_data_index = pair_n.index.union(pair_n.columns)
        assert full_data_index.isin(full_panel_index).all()
        # TODO also check no pairs occur in data that are not in panel
        # TODO isin-like check for tuples (or other combinations of rows)?
        # just iterate over both after drop_duplicates?

        full_pair_n = pair_n.reindex(index=full_panel_index
            ).reindex(columns=full_panel_index)
        # TODO maybe making symmetric is a matter of setting 0 to nan here?
        # and maybe setting back to nan at the end if still 0?
        full_pair_n.update(full_pair_n.T)
        # TODO full_pair_n.fillna(0, inplace=True)?

    # TODO TODO delete this hack once i find a nicer way to make the
    # output of crosstab symmetric
    for i in range(full_pair_n.shape[0]):
        for j in range(full_pair_n.shape[1]):
            a = full_pair_n.iat[i,j]
            b = full_pair_n.iat[j,i]
            if a > 0 and (pd.isnull(b) or b == 0):
                full_pair_n.iat[j,i] = a
            elif b > 0 and (pd.isnull(a) or a == 0):
                full_pair_n.iat[i,j] = b
    # TODO also delete this hack. this assumes that anything set to 0
    # is not actually a pair in the panel (which should be true right now
    # but will not always be)
    full_pair_n.replace(0, np.nan, inplace=True)
    #

    # TODO TODO TODO make crosstab output actually symmetric, not just square
    # (or is it always one diagonal that's filled in? if so, really just need
    # that)
    assert full_pair_n.equals(full_pair_n.T)

    # TODO TODO TODO how to indicate which of the pairs we are actually
    # interested in? grey out the others? white the others? (just set to nan?)
    # (maybe only use to grey / white out if passed in?)
    # (+ margins for now)

    # TODO TODO TODO color code text labels by pair selection reason + key
    # TODO what to do when one thing falls under two reasons though...?
    # just like a key (or things alongside ticklabels) that has each color
    # separately? just symbols in text, if that's easier?

    # TODO TODO display actual counts in squares in matshow
    # maybe make colorbar have discrete steps?

    full_pair_n.fillna('', inplace=True)
    cm = sns.light_palette('seagreen', as_cmap=True)
    # TODO TODO if i'm going to continue using styler + imgkit
    # at least figure out how to get the cmap to actually work
    # need some css or something?
    html = full_pair_n.style.background_gradient(cmap=cm).render()
    imgkit.from_string(html, 'natural_odors_pair_n.png')


