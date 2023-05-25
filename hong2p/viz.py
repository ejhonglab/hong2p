"""
Functions for visualizing movies / cells / extracted traces, some intended to
provide context specfically useful for certain types of olfaction experiments.
"""

from os.path import join, exists
import time
import functools
import sys
from typing import Dict, List, Optional
from pprint import pformat
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import linkage
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from hong2p import util, thor
from hong2p import roi as hong_roi
from hong2p.olf import remove_consecutive_repeats
from hong2p.types import DataFrameOrDataArray


# TODO consider making a style sheet as in:
# https://matplotlib.org/stable/tutorials/introductory/customizing.html?highlight=style%20sheets

DEFAULT_ANATOMICAL_CMAP = 'gray'

# TODO use this other places that redefine, now that it's module-level
dff_latex = r'$\Delta F/F$'

# TODO machinery to register combinations of level names -> how they should be formatted
# into str labels for matshow (e.g. ('odor1','odor2') -> olf.format_mix_from_strs)?
# TODO and like to set default cmap(s)?


# TODO change docstring to indicate rowlabel_fn should take a pd.Series, if that is
# always correct (for multiindex / not input, etc)
def matlabels(df, rowlabel_fn, axis='index'):
    # TODO should i modify so it takes an Index/MultiIndex rather than a DataFrame row?
    # what would make these functions more natural to write?
    """
    Takes DataFrame and function that takes one row of index to a label.

    `rowlabel_fn` should take a DataFrame row (w/ columns from index) to a str.
    """
    # TODO move to something like util._validate_axis?
    if axis == 'index':
        index = df.index
    elif axis == 'columns':
        index = df.columns
    else:
        raise ValueError("axis must be either 'index' or 'columns'")

    return list(index.to_frame().apply(rowlabel_fn, axis=1))


# TODO should i thread delim kwarg to default fn thru here (e.g. just threading all
# kwargs through to label_fn)?
def row_labels(df, label_fn):
    return matlabels(df, label_fn, axis='index')


def col_labels(df, label_fn):
    return matlabels(df, label_fn, axis='columns')


def format_index_row(row: pd.Series, delim: str = ' / '):
    """Takes Series to a str with the concatenated values, separated by `delim`.
    """
    return delim.join([str(x) for x in row.values])


# TODO rename to indicate xarray coersion (/ move that to a separate decorator called
# first?)?
# TODO specify plot_fn must take a df (maybe also numpy?) input (via type hinting)?
# TODO TODO add kwarg to wrapped fns that allows specifying axes that are supposed to
# include odor mixture metadata (or maybe more generally some minimum set of index
# levels, defined by matching some filtering fn?), to enforce (and maybe automatically
# convert for some types of input) that we have DataFrame / (especially) DataArray
# indices set correctly before calling the wrapped fn
# TODO also provide a means of specifying the dimension of input the fns expect (at
# least squeezed dimension, maybe w/ a squeeze=True kwarg added by wrapper?)
def callable_ticklabels(plot_fn):
    # TODO still true that fn must accept [x|y]ticklabels? change doc if not.
    """Allows [x/y]ticklabel functions evaluated on indices of `df` (`plot_fn(df, ...)`)

    First parameter to `plot_fn` must be a `pandas.DataFrame` (or `xarray.DataArray`
    convertable to one) to compute the `str` ticklabels from. `plot_fn` must also accept
    the kwargs '[x|y]ticklabels'.

    If the input to the decorated function is a `xarray.DataArray`, the decorated
    function will receive a `DataFrame` (created via `arr.to_pandas()`) as input
    instead. Note that only 'dimension coordinates' (see xarray terminology page) are
    preserved when converting from `DataArray` to `DataFrame`, so use
    `<arr>.set_index(<dimension name>=...)` appropriately before passing the array to a
    wrapped function.
    """
    def _check_bools(ticklabels):
        """Makes True equivalent to passing str, and False equivalent to None.
        """
        if ticklabels == True:
            # TODO TODO probably also default to this in case ticklabels=None
            # (might just need to modify some of the calling code)
            return format_index_row

        elif ticklabels == False:
            return None

        # TODO delete after checking code that could be affected to see if it trips.
        #
        # I.e. if it is specifically the callable `str`, which I had sometimes used for
        # single level indices in the past, but have encountered issues with and made
        # `format_index_row` to replace.
        elif ticklabels is str:
            warnings.warn('replace usage of str with hong2p.viz.format_index_row')
            return ticklabels

        else:
            return ticklabels

    # TODO should this also include numpy.ndarray subclasses in input types?
    @functools.wraps(plot_fn)
    def wrapped_plot_fn(df_or_arr: DataFrameOrDataArray, *args, **kwargs):

        if isinstance(df_or_arr, xr.DataArray):
            # Requires df_or_arr to be <=2d
            df = df_or_arr.to_pandas()
        else:
            df = df_or_arr

        if 'xticklabels' in kwargs:
            xticklabels = _check_bools(kwargs['xticklabels'])
            if callable(xticklabels):
                kwargs['xticklabels'] = col_labels(df, xticklabels)

        if 'yticklabels' in kwargs:
            yticklabels = _check_bools(kwargs['yticklabels'])
            if callable(yticklabels):
                kwargs['yticklabels'] = row_labels(df, yticklabels)

        return plot_fn(df, *args, **kwargs)

    return wrapped_plot_fn


# Was originally wanting to make enable a kwarg, but it seems the code to do that would
# be excessively complicated. See norok2's answer here:
# https://stackoverflow.com/questions/5929107
def _mpl_constrained_layout(enable):
    """
    To make decorators for fns that create Figure(s) and need contrained layout on/off.

    Use `@constrained_layout` or `@no_constrained_layout` rather than this directly.

    NOTE: savefig call will often also have to happen inside the same kind of context
    manager, at least for calls like `seaborn.ClusterGrid.savefig`.
    """
    def wrap_plot_fn(plot_fn):
        """
        Args:
            plot_fn: must make figure from start to finish inside, so changing
                constrained layout setting works
        """
        @functools.wraps(plot_fn)
        def wrapped_plot_fn(*args, **kwargs):
            with mpl.rc_context({'figure.constrained_layout.use': enable}):
                # TODO maybe monkeypatch any .savefig methods on returned object to warn
                # about need to add context manager around that call too? or just
                # monkeypatch to have constrained layout forced to be consistent in any
                # savefig methods?
                return plot_fn(*args, **kwargs)

        return wrapped_plot_fn

    return wrap_plot_fn


# TODO how to give these docstrings?
constrained_layout = _mpl_constrained_layout(True)
no_constrained_layout = _mpl_constrained_layout(False)


# TODO maybe combine w/ decorator to add title/xlabel/ylabel/etc kwargs + their
# reformatting (+ setting? some differences in how i do this, and would need Axes to
# call on...)
def _ylabel_kwargs(*, ylabel_rotation=None, ylabel_kws=None):
    # Will no longer have same behavior as passing rotation=None explicitly to the
    # corresponding set_ylabel call, but would need another sentinel for unset
    # values than None to fix.
    if ylabel_kws is None:
        ylabel_kws = dict()

    if ylabel_rotation is not None:
        if ylabel_kws is not None and 'ylabel_rotation' in ylabel_kws:
            raise ValueError('only specify either ylabel_rotation or '
                "ylabel_kws['rotation']"
            )

        ylabel_kws['rotation'] = ylabel_rotation

    return ylabel_kws


_panel2order = None
def set_panel2order(panel2order: Dict[str, List[str]]) -> None:
    """Sets module-level dict mapping panels to order for plotting odors.
    """
    global _panel2order
    _panel2order = panel2order


_ax_id2order = dict()
# TODO kwarg for specifying name of odor column?
def with_panel_orders(plot_fn, panel2order=None, **fn_kwargs):

    @functools.wraps(plot_fn)
    def ordered_plot_fn(*args, **kwargs):

        # If this fails, see code I adapted this from in
        # kc_natural_mixes/kc_mix_analysis.py
        assert len(args) == 0, 'expected .map/.map_dataframe to pass all via **kwargs'

        nonlocal panel2order
        if panel2order is None:
            panel2order = _panel2order

        if panel2order is None:
            raise ValueError('must either pass panel2order keyword argument, or call '
                'hong2p.viz.set_panel2order'
            )

        if 'data' not in kwargs:
            raise NotImplementedError('use map_dataframe rather than map')
        df = kwargs['data']

        for fk, fv in fn_kwargs.items():
            if fk not in kwargs:
                kwargs[fk] = fv

        # TODO add unit test for this
        if 'order' in kwargs:
            raise RuntimeError(f'when calling {plot_fn.__name__} wrapped with '
                'with_panel_orders, pass panel2order to wrapper or use '
                'hong2p.viz.set_panel2order. do NOT pass order keyword argument '
                'to the wrapped function.'
            )

        panels = df['panel'].unique()
        if len(panels) != 1:
            raise ValueError('each call to wrapped plot function must present data '
                'from a single panel'
            )
        panel = panels[0]
        if panel not in panel2order:
            raise ValueError(f'{panel=} was not in {panel2order=} keys')

        odor_col = kwargs['x']
        data_col = kwargs['y']

        odors = df[odor_col]

        order = panel2order[panel]

        # It would be nice if I could just pass a panel2order dict in that worked if the
        # values were just the names of odors (sorting naturally on concentration), but
        # I don't see an easy way to do it, as `df` may contain different subsets of the
        # data across calls to wrapped plot functions.
        assert all([o in order for o in odors])

        # TODO TODO TODO do i need to ensure that all the odors in the order are
        # present? data association getting broken yet?

        # TODO delete after some testing
        # TODO TODO TODO TODO confirm it is not a problem that this was failing
        # (i think there might be a containing axes object that is getting picked up?
        # not sure there is a mechanism to get most specific underlying one though...)
        '''
        ax_id = id(plt.gca())
        if ax_id in _ax_id2order:
            prev_order = _ax_id2order[ax_id]
            assert order == prev_order, \
                f'previous order: {prev_order}\ncurrent order: {order}'
        else:
            _ax_id2order[ax_id] = order
        '''
        #

        # TODO TODO TODO do i need to deal with hues? (to keep consistent across
        # panels...) or does FacetGrid have me covered on that?

        return plot_fn(*args, order=order, **kwargs)

    return ordered_plot_fn


@no_constrained_layout
@callable_ticklabels
# TODO TODO do [x|y]ticklabels now need to be extracted from kwargs? if seaborn doesn't
# handle that, then the @callable_ticklabels decorator is doing nothing here.
# TODO modify to call matshow internally, rather than relying on seaborn much/at all?
# (to get the convenience features i added to matshow...) (or at least make another
# decorator like callable_ticklabels to deal w/ [h|v]line_level_fn matshow kwargs)
def clustermap(df, *, optimal_ordering=True, title=None, xlabel=None, ylabel=None,
    ylabel_rotation=None, ylabel_kws=None, cbar_label=None, cbar_kws=None,
    row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None,
    method='average', metric='euclidean', z_score=None, standard_scale=None,
    **kwargs):
    """Same as seaborn.clustermap but allows callable [x/y]ticklabels + adds opts.

    Adds `optimal_ordering` kwarg to `scipy.cluster.hierarchy.linkage` that is not
    exposed by seaborn version.

    Also turns off constrained layout for the duration of the seaborn function, to
    prevent warnings + disabling that would otherwise happen.
    """

    if row_linkage is not None:
        # seaborn will just show a subset of the data if passed a linkage of a smaller
        # shape. Not sure what happens in reverse case. Either way, I think failing
        # is safer.
        expected_row_linkage_shape = (df.shape[0] - 1, 4)
        if row_linkage.shape != expected_row_linkage_shape:
            raise ValueError(f'row_linkage.shape must be {expected_row_linkage_shape}')

    if col_linkage is not None:
        expected_col_linkage_shape = (df.shape[1] - 1, 4)
        if col_linkage.shape != expected_col_linkage_shape:
            raise ValueError(f'col_linkage.shape must be {expected_col_linkage_shape}')

    valid_preproc_kw_values = {None, 0, 1}

    if z_score not in valid_preproc_kw_values:
        raise ValueError(f'z_score must be in {valid_preproc_kw_values}')

    if standard_scale not in valid_preproc_kw_values:
        raise ValueError(f'standard_scale must be in {valid_preproc_kw_values}')

    if cbar_label is not None:
        if cbar_kws is None:
            cbar_kws = dict()

        cbar_kws['label'] = cbar_label

    if z_score is not None or standard_scale is not None:
        warnings.warn('disabling optimal_ordering since z_score or standard_scale')
        optimal_ordering = False

        kwargs['z_score'] = z_score
        kwargs['standard_scale'] = standard_scale

    # TODO if z-scoring / standard-scaling requested, calculate before in this case
    # (so it actually affects linkage, as it would w/ seaborn version)
    # (currently just disabling optimal ordering in these cases)
    if optimal_ordering:
        def _linkage(df):
            # TODO way to get this to work w/ some NaNs? worth it?
            return linkage(df.values, optimal_ordering=True, method=method,
                metric=metric
            )

        # This behavior of when to transpose for which linkage is consistent w/ seaborn
        # (I read clustermap implementation)

        if row_cluster:
            if row_linkage is not None:
                raise ValueError('can not pass row_linkage if using '
                    'optimal_ordering=True'
                )

            row_linkage = _linkage(df)

        if col_cluster:
            if col_linkage is not None:
                raise ValueError('can not pass col_linkage if using '
                    'optimal_ordering=True'
                )

            col_linkage = _linkage(df.T)

    clustergrid = sns.clustermap(df, row_cluster=row_cluster, col_cluster=col_cluster,
        row_linkage=row_linkage, col_linkage=col_linkage, method=method, metric=metric,
        cbar_kws=cbar_kws, **kwargs
    )

    if title is not None:
        clustergrid.ax_heatmap.set_title(title)

    if xlabel is not None:
        # This will overwrite whatever labels the seaborn call slaps on
        clustergrid.ax_heatmap.set_xlabel(xlabel)

    if ylabel is not None:
        ylabel_kws = _ylabel_kwargs(
            ylabel_rotation=ylabel_rotation, ylabel_kws=ylabel_kws
        )
        clustergrid.ax_heatmap.set_ylabel(ylabel, **ylabel_kws)

    return clustergrid


# TODO consider calling sns.heatmap internally? or replacing some uses of this w/ that?
# TODO may want to add xlabel / ylabel kwargs to be consistent w/ my clustermap wrapper,
# but then again i'm currently using xlabel for a "title" here...
# TODO do any uses of this actually use the returned `im` (output of plt.matshow)?
# if not, maybe delete?
# TODO modify group_ticklabels / add option to support contiguous, but varying-length,
# groups (bar along edge of plot indicating extent of group label + group label placed
# by center of that bar)
# TODO try to delete levels_from_labels (transitioning to only the == False case)
# TODO TODO try options to specify figsize in terms of inches per row/col in dataframe
# (maybe plus some offset for legends, etc) (might wanna move to a constant fontsize in
# that case)
# TODO also allow specifying just index levels for [h|v]line_levels, rather than needing
# functions?
@constrained_layout
@callable_ticklabels
def matshow(df, title=None, ticklabels=None, xticklabels=None, yticklabels=None,
    xtickrotation=None, ylabel=None, ylabel_rotation=None, ylabel_kws=None,
    cbar_label=None, group_ticklabels=False, vline_level_fn=None,
    hline_level_fn=None, vline_group_text=False, hline_group_text=False,
    vgroup_label_offset=15, hgroup_label_offset=8, group_fontsize=None,
    group_fontweight=None, linewidth=0.5, linecolor='w', ax=None, fontsize=None,
    bigtext_fontsize_scaler=1.5, fontweight=None, figsize=None, dpi=None,
    inches_per_cell=None, extra_figsize=None, transpose_sort_key=None, colorbar=True,
    cbar_shrink=1.0, cbar_kws=None, levels_from_labels=True,
    allow_duplicate_labels=False, **kwargs):
    """
    Args:
        transpose_sort_key (None | function): takes df.index/df.columns and compares
            output to decide whether matrix should be transposed before plotting

        vline_level_fn: callable whose output varies along axis labels/index (see
            `levels_from_labels` for details). vertical lines will be drawn between
            changes in the output of this function.

        hline_level_fn: as `vline_level_fn`, but for horizontal lines.

        levels_from_labels: if True, `[h|v]line_level_fn` functions use formatted
            `[x|y]ticklabels` as input. Otherwise, a dict mapping index level names to
            values are used. Currently only support drawing labels for each group if
            this is False.

        **kwargs: passed thru to `matplotlib.pyplot.matshow`
    """
    # NOTE: if i'd like to also sort on [x/y]ticklabels, would need to move this block
    # after possible ticklabel enumeration, and then assign correctly to index/cols and
    # use that as input to sort_key_val in appropriate instead
    if transpose_sort_key is not None:
        if any([x is not None for x in [ticklabels, xticklabels, yticklabels]]):
            # TODO maybe update just to only allowing `ticklabels` (since then x,y
            # should be the same, which i think is the only case where we do want
            # transpose_sort_key anyway)
            raise NotImplementedError('transpose_sort_key not supported if any '
                'ticklabels are explicitly passed'
            )

        row_sort_key = transpose_sort_key(df.index)
        col_sort_key = transpose_sort_key(df.columns)

        if row_sort_key > col_sort_key:
            df = df.T

    # TODO shouldn't this get ticklabels from matrix if nothing else?
    # maybe at least in the case when both columns and row indices are all just
    # one level of strings?
    if inches_per_cell is not None:
        if figsize is not None:
            raise ValueError('only specify one of inches_per_cell or figsize')

        if extra_figsize is None:
            extra_figsize = (2.0, 1.0)

        extra_width, extra_height = extra_figsize
        figsize = (
            inches_per_cell * df.shape[1] + extra_width,
            inches_per_cell * df.shape[0] + extra_height
        )

    elif extra_figsize is not None:
        raise ValueError('extra_figsize must be specified with inches_per_cell')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        if figsize is not None:
            raise ValueError('figsize only allowed if ax not passed in')

        fig = ax.get_figure()

    # TODO can probably delete this and replace w/ usage of format_index_row, maybe with
    # some slight modifications
    def is_one_level_str_index(index):
        return len(index.shape) == 1 and all(index.map(lambda x: type(x) is str))

    if (xticklabels is None) and (yticklabels is None):
        if ticklabels is None:
            # TODO maybe default to joining str representations of values at each level,
            # joined on some kwarg delim like '/'?
            xticklabels = df.columns if is_one_level_str_index(df.columns) else None
            yticklabels = df.index if is_one_level_str_index(df.index) else None
        else:
            # TODO TODO assert indices are actually equal
            assert df.shape[0] == df.shape[1]

            if callable(ticklabels):
                ticklabels = matlabels(df, ticklabels)

            # TODO should i check output is same on rows/cols in this case?
            # currently, matlabels seems to just operate on the row labels...
            xticklabels = ticklabels
            yticklabels = ticklabels

    # TODO update this formula to work w/ gui corrs (too big now)
    # TODO see whether default font actually is inappropriate in any cases where i'm
    # currently calling this (particularly using constrained layout from
    # al_analysis.py)
    if fontsize is None:
        fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))

    bigtext_fontsize = bigtext_fontsize_scaler * fontsize

    im = ax.matshow(df, **kwargs)

    if colorbar:
        if cbar_kws is None:
            cbar_kws = dict()

        # rotation=270?
        #cbar = add_colorbar(fig, im, label=cbar_label, shrink=cbar_shrink, **cbar_kws)
        # TODO thread fontsize thru this?
        cbar = add_colorbar(fig, im, label=cbar_label, shrink=cbar_shrink,
            fontsize=bigtext_fontsize, **cbar_kws
        )

    if group_fontsize is None:
        group_fontsize = bigtext_fontsize

    def _index_entry_dict(index, index_entry):
        # If the index is a MultiIndex, iterating over it should yield tuples of the
        # level values, otherwise we'll convert to a tuple so zipping is the same.
        if type(index_entry) is not tuple:
            index_entry = (index_entry,)

        names = index.names
        assert len(names) == len(index_entry)
        return dict(zip(names, index_entry))

    # TODO TODO lines showing extent that each text label applies too?
    # (e.g. parallel to labelled axis, with breaks between levels? might crowd fly
    # labels less than separator lines perpendicular to axis)
    # TODO light refactoring to share x/y (v/x) code?
    # TODO TODO for both of these, make linewidth a constant fration of cell
    # width/height (whichever is appropriate) (at least by default)
    # ~(figsize[i] / df.shape[i])?
    # TODO what is default linewidth here anyway? unclear. 1?
    # TODO default to only formatting together index levels not used by
    # [h|v]line_level_fn (possible?), when ?
    hline_levels = None
    if hline_level_fn is not None:
        if levels_from_labels:
            ranges = util.const_ranges([hline_level_fn(x) for x in yticklabels])
        else:
            # TODO need to handle case where we might transpose (e.g. via
            # transpose_sort_key?)
            hline_levels = [
                hline_level_fn(_index_entry_dict(df.index, x)) for x in df.index
            ]
            # TODO modify const_ranges to have include_val=True behavior be default?
            # (+ delete switching flag, if so)
            ranges = util.const_ranges(hline_levels, include_val=True)

            if hline_group_text:
                for label, y0, y1 in ranges:
                    # TODO TODO compute group_label_offset? way to place the text using
                    # constrained layout?
                    ax.text(-hgroup_label_offset, np.mean((y0, y1)) + 0.5, label,
                        fontsize=group_fontsize, fontweight=group_fontweight,
                        # Right might make consistent spacing wrt line indicating extent
                        # of group easier to see.
                        ha='right',
                        # Seemed to be a bit lower than center? Some other offset?
                        #va='center'
                    )

        # If all the ranges have the same start and stop, all groups are length 1, and
        # the lines would just add visual noise, rather than helping clarify boundaries
        # between groups.
        if any([x[-1] > x[-2] for x in ranges]):
            line_positions = [x[-1] + 0.5 for x in ranges[:-1]]
            # TODO if we have a lot of matrix elements, may want to decrease size of
            # line a bit to not approach size of matrix elements...
            for v in line_positions:
                # 'w'=white. https://matplotlib.org/stable/tutorials/colors/colors.html
                ax.axhline(v, linewidth=linewidth, color=linecolor)

    vline_levels = None
    if vline_level_fn is not None:
        if levels_from_labels:
            ranges = util.const_ranges([vline_level_fn(x) for x in xticklabels])
        else:
            vline_levels =[
                vline_level_fn(_index_entry_dict(df.columns, x)) for x in df.columns
            ]
            ranges = util.const_ranges(vline_levels, include_val=True)

            if vline_group_text:
                for label, x0, x1 in ranges:
                    ax.text(np.mean((x0, x1)) + 0.5, -vgroup_label_offset, label,
                        fontsize=group_fontsize, fontweight=group_fontweight,
                        ha='center'
                    )

        if any([x[-1] > x[-2] for x in ranges]):
            line_positions = [x[-1] + 0.5 for x in ranges[:-1]]
            for v in line_positions:
                ax.axvline(v, linewidth=linewidth, color=linecolor)

    def grouped_labels_info(labels):
        if not group_ticklabels or labels is None:
            return labels, 1, 0

        without_consecutive_repeats, n_repeats = remove_consecutive_repeats(labels)
        tick_step = n_repeats
        offset = n_repeats / 2 - 0.5
        return without_consecutive_repeats, tick_step, offset

    # TODO make fontsize / weight more in group_ticklabels case?
    xticklabels, xstep, xoffset = grouped_labels_info(xticklabels)
    yticklabels, ystep, yoffset = grouped_labels_info(yticklabels)

    def set_ticklabels(ax, x_or_y, labels, *args, **kwargs):
        if not allow_duplicate_labels:
            # TODO refactor?
            if x_or_y == 'x' and (vline_group_text and vline_levels is not None):
                to_check = list(zip(vline_levels, labels))
                err_msg = 'duplicate (vline_level, xticklabel) combinations'

            elif x_or_y == 'y' and (hline_group_text and hline_levels is not None):
                to_check = list(zip(hline_levels, labels))
                err_msg = 'duplicate (hline_level, yticklabel) combinations'
            else:
                to_check = labels
                err_msg = f'duplicate {x_or_y}ticklabels'

                if x_or_y == 'x' and vline_level_fn is not None:
                    err_msg += ('. specifying vline_group_text=True may resolve '
                        'duplicates.'
                    )

                elif x_or_y == 'y' and hline_level_fn is not None:
                    err_msg += ('. specifying hline_group_text=True may resolve '
                        'duplicates.'
                    )

            if len(to_check) != len(set(to_check)):
                raise ValueError(err_msg)

        assert x_or_y in ('x', 'y')
        if x_or_y == 'x':
            set_fn = ax.set_xticklabels
        elif x_or_y == 'y':
            set_fn = ax.set_yticklabels

        try:
            set_fn(labels, *args, **kwargs)
        # Intended to catch stuff like:
        # "ValueError: The number of FixedLocator locations (19), usually from a call to
        # set_ticks, does not match the number of ticklabels (16)."
        # ...so we can provide more additional debug info.
        except ValueError as err:
            print('\nDebug info for following matplotlib error:\n'
                f'{x_or_y}ticklabels={pformat(labels)}\nlen={len(labels)}\n',
                file=sys.stderr
            )
            raise

    if xticklabels is not None:
        # TODO nan / None value aren't supported in ticklabels are they?
        # (couldn't assume len is defined if so)
        if xtickrotation is None:
            if all([len(x) == 1 for x in xticklabels]):
                xtickrotation = 'horizontal'
            else:
                xtickrotation = 'vertical'

        ax.set_xticks(np.arange(0, len(df.columns), xstep) + xoffset)
        set_ticklabels(ax, 'x', xticklabels,
            fontsize=fontsize, fontweight=fontweight, rotation=xtickrotation
        )

    if yticklabels is not None:
        ax.set_yticks(np.arange(0, len(df), ystep) + yoffset)
        set_ticklabels(ax, 'y', yticklabels,
            fontsize=fontsize, fontweight=fontweight, rotation='horizontal'
        )

    # didn't seem to do what i was expecting
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False)

    if title is not None:
        ax.set_xlabel(title, fontsize=bigtext_fontsize, labelpad=12)

    if ylabel is not None:
        ylabel_kws = _ylabel_kwargs(
            ylabel_rotation=ylabel_rotation, ylabel_kws=ylabel_kws
        )
        ax.set_ylabel(ylabel, fontsize=bigtext_fontsize, **ylabel_kws)

    return fig, im


def imshow(img, title=None, cmap=DEFAULT_ANATOMICAL_CMAP):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)

    if title is not None:
        ax.set_title(title)

    ax.axis('off')
    return fig


def add_colorbar(fig, im, label=None, fontsize=None, shrink=1.0, **kwargs):
    """
        shrink: same default as matplotlib
    """

    # TODO check whether kwargs `label=label` to fig.colorbar can replace
    # cbar.ax.set_ylabel. i think so, but poorly documented.

    # I think this relies on use of Matplotlib's constrained layout
    cbar = fig.colorbar(im, ax=fig.axes, shrink=shrink, **kwargs)

    if label is not None:
        # TODO test fontsize=None doesn't change default behavior
        cbar.ax.set_ylabel(label, fontsize=fontsize)

    return cbar


def contour_center_of_mass(contour):
    # TODO doc shape / type of contour
    # Taken from ImportanceOfBeingErnests's answer here:
    # https://stackoverflow.com/questions/48168880
    # calculate center of mass of a closed polygon
    x = contour[:, 0]
    y = contour[:, 1]
    g = (x[:-1] * y[1:] - x[1:] * y[:-1])
    A = 0.5 * g.sum()
    cx = ((x[:-1] + x[1:]) * g).sum()
    cy = ((y[:-1] + y[1:]) * g).sum()
    return 1. / (6 * A) * np.array([cx, cy])


def plot_closed_contours(footprint, if_multiple: str = 'err', _pad=True, ax=None,
    label: Optional[str] = None, text_kws: Optional[dict] = None, colors='red',
    linewidths=1.2, linestyles='dotted', **kwargs):
    # TODO doc / delete
    """Plots line around the contiguous positive region(s) in footprint.

    Args:
        ax: Axes to plot onto. will use current Axes otherwise.

        if_multiple: 'take_largest'|'join'|'err'|'ignore'. what to do if there are
            multiple closed contours within footprint. contour will be plotted
            regardless, but error will happen before a contour is returned for use in
            other analysis.

        **kwargs: passed through to matplotlib `ax.contour` call
    """
    # NOTE: linewidths=0.8 seems good for linestyles='solid', but I prefer thicker for
    # dotted.

    if ax is None:
        ax = plt.gca()

    # TODO TODO fix what seems to be a (1, 1) pixel offset of contour wrt footprint
    # passed in (when plotted on same axes).
    # TODO TODO does extent= kwarg to contour call just change the plotted range, or
    # would it prevent my padding code from fixing footprint-touching-border cases, if
    # i were to find values for extent= that would invert plotting offset?
    if _pad:
        # I needed to pad so that footprints touching the edge of the image would still
        # get contours correctly (I believe). Not sure how to fix the 1 pixel offset in
        # plotting without changing contour that is returned.
        dims = footprint.shape
        padded_footprint = np.zeros(tuple(d + 2 for d in dims))
        padded_footprint[tuple(slice(1,-1) for _ in dims)] = footprint

        mpl_contour = ax.contour(padded_footprint > 0, [0.5], colors=colors,
            linewidths=linewidths, linestyles=linestyles,
            #extent=(0, dims[0] - 1, 0, dims[1] - 1),
            **kwargs
        )
        #mpl_contour_nopad = ax.contour(footprint > 0, [0.5], colors=colors,
        #    linewidths=linewidths, linestyles=linestyles, **kwargs
        #)
    else:
        # Checking for what I believe was the reason we needed padding, though it might
        # not affect plotting if that's all we want in _pad=False case anyway
        positive = footprint > 0
        assert not any([
            np.array(positive[0, :]).any(),
            np.array(positive[-1, :]).any(),
            np.array(positive[:, 0]).any(),
            np.array(positive[:, -1]).any(),
        ])
        mpl_contour = ax.contour(positive, [0.5], colors=colors,
            linewidths=linewidths, linestyles=linestyles, **kwargs
        )

    if label is not None:
        assert len(mpl_contour.allsegs) == 1
        # TODO TODO warn? might need to use a particular segment / combination in other
        # cases
        #assert len(mpl_contour.allsegs[-1]) == 1
        assert len(mpl_contour.allsegs[-1]) >= 1

        # Also partially taken from https://stackoverflow.com/questions/48168880
        cx, cy = contour_center_of_mass(mpl_contour.allsegs[-1][0])

        if text_kws is None:
            text_kws = dict()

        default_text_kws = {
            'color': colors,
            'horizontalalignment': 'center',
            'fontweight': 'bold',
            # Default should be 10.
            'fontsize': 8,
        }
        for k, v in default_text_kws.items():
            if k not in text_kws:
                text_kws[k] = v

        ax.text(cx, cy, label, **text_kws)

    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1

    paths = mpl_contour.collections[0].get_paths()
    assert len(paths) > 0

    if len(paths) != 1:
        # NOTE: this will be after drawing contour, but before drawing any label...
        if if_multiple == 'err':
            raise RuntimeError('multiple disconnected paths in one footprint')

        elif if_multiple == 'ignore':
            return None

        elif if_multiple == 'take_largest':
            raise NotImplementedError

            largest_sum = 0
            largest_idx = 0
            total_sum = 0
            for p in range(len(paths)):
                path = paths[p]

                # TODO maybe replace mpl stuff w/ cv2 drawContours? (or related...) (fn
                # now in here as roi.contour2mask)
                # TODO shouldn't these (if i want to keep this branch anyway...)
                # be using padded_footprint instead of footprint?
                # TODO TODO factor to something like a "mplpath2mask" fn
                mask = np.ones_like(footprint, dtype=bool)
                for x, y in np.ndindex(footprint.shape):
                    # TODO TODO not sure why this seems to be transposed, but it
                    # does (make sure i'm not doing something wrong?)
                    if path.contains_point((x, y)):
                        mask[x, y] = False
                # Places where the mask is False are included in the sum.
                path_sum = MaskedArray(footprint, mask=mask).sum()
                # TODO maybe check that sum of all path_sums == footprint.sum()?
                # seemed there were some paths w/ 0 sum... cnmf err?
                '''
                print('mask_sum:', (~ mask).sum())
                print('path_sum:', path_sum)
                print('regularly masked sum:', footprint[(~ mask)].sum())
                plt.figure()
                plt.imshow(mask)
                plt.figure()
                plt.imshow(footprint)
                plt.show()
                import ipdb; ipdb.set_trace()
                '''
                if path_sum > largest_sum:
                    largest_sum = path_sum
                    largest_idx = p

                total_sum += path_sum

            footprint_sum = footprint.sum()
            # TODO float formatting / some explanation as to what this is
            print('footprint_sum:', footprint_sum)
            print('total_sum:', total_sum)
            print('largest_sum:', largest_sum)
            # TODO is this only failing when stuff is overlapping?
            # just merge in that case? (wouldn't even need to dilate or
            # anything...) (though i guess then the inequality would go the
            # other way... is it border pixels? just ~dilate by one?)
            # TODO fix + uncomment
            #assert np.isclose(total_sum, footprint_sum)
            path = paths[largest_idx]

        elif if_multiple == 'join':
            raise NotImplementedError
    else:
        path = paths[0]

    # TODO TODO need to test that anything that used return value here is still correct,
    # now that i deleted old padding code, after it seemed to be just causing an offset
    # in plot_rois

    contour = path.vertices
    # Correct index change caused by padding.
    return contour - 1


def image_grid(image_list, *, nrows=None, ncols=None, figsize=None, dpi=None,
    cmap=DEFAULT_ANATOMICAL_CMAP, **imshow_kwargs):

    def ceil(x):
        return int(np.ceil(x))

    def n_other_axis(n_first_axis):
        return ceil(len(image_list) / n_first_axis)

    if nrows is None and ncols is None:
        n = ceil(np.sqrt(len(image_list)))
        nrows = n
        ncols = n

    elif nrows is None and ncols is not None:
        nrows = n_other_axis(ncols)

    elif ncols is None and nrows is not None:
        ncols = n_other_axis(nrows)

    if figsize is None:
        # Assuming all images in image_list are the same shape
        image_shape = image_list[0].shape
        assert len(image_shape) == 2
        # TODO actually test w/ images where w != h. i might have them flipped.
        w = image_shape[1]
        h = image_shape[0]

        aspect = (ncols * w) / (nrows * h)

        height_inches = 10
        figsize = (aspect * height_inches, height_inches)

    # TODO (if not passed) set figsize according to aspect ratio you'd get multiplying
    # nrows/ncols by single image dimensions (to try to fill space as much as possible)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

    # TODO see: https://stackoverflow.com/questions/42850225 or related to find a
    # good solution for reliably eliminating all unwanted whitespace between subplots
    # (honestly i think it will ultimately come down to figsize, potentially more so
    # when using constrained layout, as i'd now generally like to)

    for ax, img in zip(axs.flat, image_list):
        ax.imshow(img, cmap=cmap, **imshow_kwargs)
        ax.axis('off')

    for ax in axs.flat[len(image_list):]:
        fig.delaxes(ax)

    # TODO detect if contstrained layout is enabled, and if not, call this?
    # or maybe just put the @constrained_layout decorator on this fn to guarantee it?
    #fig.subplots_adjust(wspace=0, hspace=0.05)

    return fig, axs


# TODO unit test (that at least it produces a figure without failing for
# correctly-formatted DataArray input)
# TODO support similarly-indexed DataArray for background (+ maybe remove ndarray code)
def plot_rois(rois: xr.DataArray, background: np.ndarray, show_names: bool = True,
    ncols: int = 2, _pad: bool = False, cmap=DEFAULT_ANATOMICAL_CMAP, **kwargs
    ) -> Figure:
    # TODO doc
    """
    Args:
        rois: with dims ('roi', 'z', 'y', 'x'). coords must have at least
            ('roi_z', 'roi_name') on the 'roi' dimension.

        background: must have shape equal to the (<z>, <y>, <x>) lengths of the
            corresponding entries in `rois.sizes`

        show_names: whether to plot ROI names in the center of each ROI

        ncols: how many columns for grid showing the background of each plane
            (one panel per plane)

        **kwargs: passed thru to `plot_closed_contours`
    """
    # TODO TODO option to [locally?] histogram equalize the image (or something else to
    # increase contrast + prevent hot pixels from screwing up range in a plane)
    # TODO option to color ROIs randomly (perhaps also specifically so no neighboring
    # ROIs share a color, if possible) (sharing colors across planes if the e.g. ROI
    # name doesn't change, at least in the show_names=True case)
    # TODO support background being None
    # TODO option to "equalize" background image (see old code in plot_traces
    # show_footprints path)?

    # TODO check rois and background have compatible shape

    z_size = rois.sizes['z']

    fig, axs = image_grid(background, ncols=ncols, cmap=cmap)

    # Moving 'roi' from end to start.
    rois = rois.transpose('roi', 'z', 'y', 'x')

    err_msg = None
    for z, ax in enumerate(axs.flat):
        try:
            rois_in_curr_z = rois.sel(roi_z=z)
        except KeyError:
            continue

        for roi in rois_in_curr_z:
            roi = roi[z]

            label = None
            if show_names:
                # TODO what if not all coordinates associated w/ this dimension are on
                # the index? if i make a solution to also handle that, put in
                # hong2p.xarray
                index_names = rois.get_index('roi').names

                # Since the .sel operation above removes this coordinate.
                index_names = [x for x in index_names if x != 'roi_z']

                index_vals = roi.roi.item()
                assert len(index_vals) == len(index_names)
                index = dict(zip(index_names, index_vals))
                name = index['roi_name']

            try:
                plot_closed_contours(roi, label=name, ax=ax, _pad=_pad, **kwargs)

            except RuntimeError as err:
                # +1 to index as in ImageJ
                curr_err_msg = f'{name} (z={z + 1}): {err}'
                if err_msg is None:
                    err_msg = curr_err_msg
                else:
                    err_msg += f'\n{curr_err_msg}'

        if z == (z_size - 1):
            break

    if err_msg is not None:
        raise RuntimeError(err_msg)

    return fig


# TODO should i actually compute correlations in here too? check input, and
# compute if input wasn't correlations (/ symmetric?)?
# if so, probably return them as well.
def plot_odor_corrs(corr_df, odor_order=False, odors_in_order=None,
    trial_stat='mean', title_suffix='', **kwargs):
    """Takes a symmetric DataFrame with odor x odor correlations and plots it.
    """
    # TODO replace ordering stuff w/ new fns for that in olf.py (or maybe just delete
    # all of plot_odor_corrs if not using...)

    # TODO test this fn w/ possible missing data case.
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

    if 'cbar_label' not in kwargs:
        # TODO factor out latex for delta f / f stuff (+ maybe use in analysis that uses
        # this pkg: kc_natural_mixes, al_analysis)
        kwargs['cbar_label'] = \
            trial_stat.title() + f' response {dff_latex} correlation'

    return matshow(corr_df, **kwargs)


# TODO get x / y from whether they were declared share<x/y> in facetgrid
# creation?
# TODO TODO rename to something like "hide_all_but_first_axes_label" -> accept fig input
# TODO also support hiding all but first xticklabels/similar? or make similar fns for
# that?
def fix_facetgrid_axis_labels(facet_grid, shared_in_center: bool = False,
    x: bool = True, y: bool = True) -> None:
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
            # why did i get a deprecation warning for this ax.is_first_col() in 3.4.3
            # (in my local recreation of remy_suite2p) but not in 3.5.1 (suite2p) which
            # i was testing with earlier?
            spec = ax.get_subplotspec()

            if not (spec.is_first_col() and spec.is_last_row()):
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
            footprint = hong_roi.db_row2footprint(footprint_row, shape=avg.shape)

            # TODO maybe some percentile / fixed size about maximum
            # density?
            cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
                hong_roi.crop_to_nonzero(footprint, margin=6)

            cell2rect[cell_id] = (x_min, x_max, y_min, y_max)

            cropped_avg = better_constrast[x_min:x_max + 1, y_min:y_max + 1]

            if show_footprint_with_mask:
                # TODO figure out how to suppress clipping warning in the case
                # when it's just because of float imprecision (e.g. 1.0000001
                # being clipped to 1) maybe just normalize to [0 + epsilon, 1 -
                # epsilon]?
                # TODO TODO or just set one channel to be this
                # footprint?  scale first?
                cropped_footprint_rgb = color.gray2rgb(cropped_footprint)

                for c in (1,2):
                    cropped_footprint_rgb[:,:,c] = 0
                # TODO plot w/ value == 1 to test?

                cropped_footprint_hsv = color.rgb2hsv(cropped_footprint_rgb)

                cropped_avg_hsv = color.rgb2hsv(color.gray2rgb(cropped_avg))

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
                composite = cv2.normalize(composite, None, alpha=0.0, beta=1.0,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
                )

            else:
                # TODO could also use something more than this
                # TODO TODO fix bug here. see 20190402_bug1.txt
                # TODO TODO where are all zero footprints coming from?
                cropped_footprint_nonzero = cropped_footprint > 0
                if not np.any(cropped_footprint_nonzero):
                    continue

                level = cropped_footprint[cropped_footprint_nonzero].min()

            if show_footprints_alone:
                ax = axs[i,-2]
                f_ax = axs[i,-1]
                f_ax.imshow(cropped_footprint, cmap=DEFAULT_ANATOMICAL_CMAP)
                f_ax.axis('off')
            else:
                ax = axs[i,-1]

            if show_footprint_with_mask:
                ax.imshow(composite)
            else:
                ax.imshow(cropped_avg, cmap=DEFAULT_ANATOMICAL_CMAP)
                # TODO TODO also show any other contours in this rectangular ROI
                # in a diff color! (copy how gui does this)
                cell2contour[cell_id] = plot_closed_contours(cropped_footprint, ax,
                    colors='red'
                )

            ax.axis('off')

            text = str(cell_id + 1)
            h = y_max - y_min
            w = x_max - x_min
            rect = patches.Rectangle((y_min, x_min), h, w, linewidth=1.5, edgecolor='b',
                facecolor='none'
            )
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
    title = str(thorsync_dir)

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


