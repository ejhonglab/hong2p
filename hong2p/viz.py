"""
Functions for visualizing movies / cells / extracted traces, some intended to
provide context specfically useful for certain types of olfaction experiments.
"""

from os.path import join, exists
import time
import functools
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from pprint import pformat, pprint
# TODO replace w/ logging.warning?
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from colorsys import rgb_to_hsv
from random import Random

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import linkage
import colorcet as cc
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
# Only for type hinting
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib_scalebar.scalebar import ScaleBar

from hong2p import util, thor
from hong2p import roi as hong_roi
# TODO replace w/ select_certain_rois (after adapting it to work w/ DataArray input)
# TODO possible to fix circular import error this seemed to cause? maybe via changing
# hong2p.roi (only/too)?
#from hong2p.roi import is_ijroi_certain
#
from hong2p.olf import remove_consecutive_repeats
from hong2p.types import DataFrameOrDataArray, KwargDict


# TODO consider making a style sheet as in:
# https://matplotlib.org/stable/tutorials/introductory/customizing.html?highlight=style%20sheets

DEFAULT_ANATOMICAL_CMAP = 'gray'

# TODO use this other places that redefine, now that it's module-level
dff_latex = r'$\Delta F/F$'

_debug = False

# TODO machinery to register combinations of level names -> how they should be formatted
# into str labels for matshow (e.g. ('odor1','odor2') -> olf.format_mix_from_strs)?
# TODO and like to set default cmap(s)?


def is_categorical(values: pd.Series) -> bool:
    """Returns whether input seems categorical (vs continuous), based mainly on dtype.

    Values with `int` dtype are considered continuous if they have any negative values,
    otherwise categorical.
    """
    # TODO only count int as categorical if min is 0? or if there are no negative
    # values?
    # TODO maybe also require a sufficiently small number of unique ints?
    # (< ~20-30?)
    #
    # assume we just have continuous/categorical now, and that if not continuous
    # it's categorical
    dtype = values.dtype
    if dtype == float or (dtype == int and values.min() < 0):
        categorical = False
    else:
        categorical = True

    return categorical


# TODO type specifically for 0/1 and 'index'[+'rows'?]/'columns' instead of current
# `axis` type annotation
#
# TODO allow passing in premade sns.color_palette output (could also take
# ListedColormap, and have caller define that as
# `ListedColormap(sns.color_palette(...))`?)? in addition to palette names, for my fn to
# generate row_colors from row_values? (in case i did want to share kc_type_palette /
# type2color as a module-level variable, and for similar cases)
#
# TODO does a 1-column DataFrame work just as well (as a Series) for input to
# sns.clustermap [row|col]_colors (i.e. can my fn always return a DataFrame?)?
#
# TODO also return palettes (w/ min/max? unique values? callables derived from palettes
# + [min/max]/unique_values?), for use w/ future calls? how?
# TODO optional argument for min/max / (ordered) unique_values per variable?
def map_each_series_to_rgb(df: pd.DataFrame, *, axis: Union[int, str] = 'index',
    name2palette: Optional[Dict[str, Union[str, mpl.colors.LinearSegmentedColormap,
    Dict]]] = None, share_palettes_with_same_name: bool = True, _debug: bool = False
    ) -> Tuple[pd.DataFrame, List, List]:
    """Maps frame values to RGB tuples, from per-column/row colormaps.

    Args:
        df: dataframe with values to map to colors. If `axis=0`/`axis='index'`, each
            column will have its own palette, and each row will if `axis=1` /
            `axis='columns'`.

        name2palette: if a variable (column/row name from `df`, depending on `axis=`.
            see above.) name is a key in this, the corresponding palette value will be
            used, rather than chosen within this function.

        share_palettes_with_same_name: if True, variables who share a `str`
            `name2palette` value will all share a common value -> color mapping, and
            thus also share an entry in either `for_legends` or `for_cbars`.
            If `add_legends_and_colorbars` is used to draw the legends and colorbars,
            this will also mean they will share a legend or colorbar.

    Returns:
        color_df: frame of same shape as `df`, with all values RGB 3-tuple colors.

        for_legends: list of dicts, each with keyword arguments for a call to
            `fig.legend`. Can be passed to `add_legends_and_colorbars` argument of the
            same name.

        for_cbars: list of dicts, each with keyword arguments for a call to
            `fig.colorbar`. Can be passed to `add_legends_and_colorbars` argument of the
            same name.
    """
    if axis in (0, 'index', 'rows'):
        names = df.columns
    else:
        assert axis in (1, 'columns')
        names = df.index
        # TODO implement + test this case
        raise NotImplementedError('may just need to update df indexing below')

    assert not names.duplicated().any()
    assert all(type(x) is str for x in names)

    if name2palette is None:
        name2palette = dict()
    else:
        # so we can add k/v pairs below, w/o changing input
        name2palette = dict(name2palette)

    # NOTE: will also be considered taken if a name2palette key does not exist in df
    # index names
    # TODO care to do anything to avoid dict inputs overlapping w/ named palettes
    # (either strs in input name2palette values, or str palettes chosen in here)?
    # (check values? or depend too much on spacing? prob not for some of the cyclic
    # colormaps?)
    taken_palettes: Set[str] = {x if isinstance(x, str) else x.name
        for x in name2palette.values() if not isinstance(x, dict)
    }

    # TODO replace these w/ some maplotlib/seaborn/other-3rd-party-library (colorcet?)
    # based enumeration of palettes of different kinds? already have anything like this
    # in hong2p?
    # TODO allow user to pass their own lists of palettes (in preferred order) in?
    # TODO some way to programmatically get name of current default seaborn colormap?
    # (and use for first element of categorical_cmaps)
    # sns.color_palette (despite what docs say) does not *always* return a
    # mpl.colors.ListedColormap object, e.g. if called with no `name` (1st positional
    # arg), like `sns.color_palette(as_cmap=True)`
    categorical_cmaps: List[str] = ['tab10', 'Accent', 'hls']

    # TODO automatically pick a diverging one if values are both negative and positive?
    # (separate list for those?) ('vlag' and 'icefire' are two perceptually linear
    # options)
    #
    # other perceptually uniform cmaps listed on current:
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html look too similar
    # to one of the other ones
    #
    # NOTE: access colorcet cmaps as matplotlib cmaps by referencing them like:
    # `cc.m_<name>` (of type mpl.colors.LinearSegmentedColormap)
    continuous_cmaps: List[Union[str, mpl.colors.LinearSegmentedColormap]] = [
        'magma', 'cividis', cc.m_fire, cc.m_gray, cc.m_bmw
    ]

    # TODO move this to module level? get_palette_id too?
    def get_palette_name(p: Union[str, mpl.colors.LinearSegmentedColormap]) -> str:
        if isinstance(p, str):
            name = p
        else:
            assert isinstance(p, mpl.colors.LinearSegmentedColormap)
            # not super worried about possibility of two cmaps (accidentally) having
            # same name but diff colors, or same colors but diff name
            name = p.name
            assert isinstance(name, str)

        return name

    def get_palette_id(p: Union[str, mpl.colors.LinearSegmentedColormap, Dict]
        ) -> Union[int, str]:
        if isinstance(p, dict):
            return id(p)
        return get_palette_name(p)

    def _next_palette(order: List[Union[str, mpl.colors.LinearSegmentedColormap]],
        prefix: str) -> str:

        palette = None
        for p in order:
            name = get_palette_name(p)
            if name in taken_palettes:
                continue

            palette = p
            break

        if palette is None:
            raise RuntimeError(f'ran out of palettes! set {prefix}_cmaps longer '
                'than {order=} ({len(order)=})'
            )

        if _debug:
            print(f'chose {name=}')

        taken_palettes.add(name)
        return palette

    palette2vars: Dict[Union[str, int], List[str]] = defaultdict(list)
    # since mpl.colors.LinearSegmentedColormaps not hashable, will use their .name in
    # palette_name2vars, then look up here when needed
    mpl_palette_lookup: Dict[str, mpl.colors.LinearSegmentedColormap] = dict()
    dict_palette_lookup: Dict[int, Dict] = dict()
    for n in names:
        # TODO update indexing to support axis=1|'columns'
        values = df[n]
        dtype = values.dtype
        if _debug:
            print()
            print(f'{n=}')
            print(f'{dtype=}')

        # TODO allow user to pass in which vars should be treated as
        # categorical[/continuous[/diverging]]?
        categorical = is_categorical(values)
        if _debug:
            print(f'{categorical=}')

        if n in name2palette:
            palette = name2palette[n]
        else:
            if categorical:
                order = categorical_cmaps
                desc = 'categorical'
            else:
                order = continuous_cmaps
                desc = 'continuous'

            palette = _next_palette(order, desc)
            name2palette[n] = palette

        # will be palette name for all but dict palettes, and int ID for those
        palette_id = get_palette_id(palette)

        if isinstance(palette, mpl.colors.LinearSegmentedColormap):
            # TODO assert palette_id is a str here?
            mpl_palette_lookup[palette_id] = palette

        elif isinstance(palette, dict):
            # TODO assert palette_id is an int here?
            dict_palette_lookup[palette_id] = palette

        palette2vars[palette_id].append(n)

        # TODO assert no two manually-passed-in dict values in name2palette are the
        # same? don't want the extra hassle of supporting vars sharing palettes that
        # way.

    # TODO delete
    if _debug:
        print()
        print('palette2vars:')
        pprint(palette2vars)
        print('mpl_palette_lookup:')
        pprint(mpl_palette_lookup)
        print('dict_palette_lookup:')
        pprint(dict_palette_lookup)
    #

    color_list = []
    for_legends = []
    for_cbars = []
    def _map_vars_to_colors(values: pd.Series, palette: Union[str,
        mpl.colors.LinearSegmentedColormap, dict], dtype, categorical: bool) -> None:
        """Appends one entry to `colors`, and one to either `for_legends` or
        `for_cbars`.
        """
        # TODO change to support axis=1/'columns'?
        var_names = values.columns

        # mostly assuming we won't be wanting to have mpl.colors.LinearSegmentedColormap
        # in the categorical case, for now
        if categorical:
            # strs could either have been manually passed in, or chosen in this fn
            if isinstance(palette, str):
                if dtype == int:
                    vmin = 0
                    vmax = values.max().max()
                    # assuming we need separate color for 0, as well as every int
                    # between 0 and max (inclusive)
                    unique_values = range(vmin, vmax + 1)
                else:
                    assert dtype == np.dtype('O'), f'unrecognized {dtype=}'
                    unique_values = set()
                    for c in values.columns:
                        # TODO assert values are str? (maybe w/ some None/NaN allowed?)
                        unique_values |= set(values[c].unique())

                    # NOTE: can no longer rely on order-of-first-appearance output
                    # ordering of <Series>.unique(), though can pass manual dict palette
                    # for a particular association with colors
                    unique_values = sorted(unique_values)

                n_unique = len(unique_values)
                palette = sns.color_palette(palette, n_colors=n_unique)
                assert len(palette) == n_unique

                # assuming we will pass in palette, if we care for a particular
                # value->color map (i.e. if we would have wanted a particular order
                # on values here, e.g. by sorting)
                value2color = dict(zip(unique_values, palette))

            # dicts must have been manually passed in via name2palette
            else:
                assert isinstance(palette, dict), ('only str or dict palette allowed '
                    'for categorical variables'
                )
                value2color = palette
                assert values.isin(value2color.keys()).all().all()

            # colors (whether str or RGB tuples) should all be hashable, so making a
            # set from them should work
            if len(set(value2color.values())) < len(value2color):
                # sns.color_palette docs say that, for certain other parameters,
                # potentially all where as_cmap=False, colors will cycle. potetially
                # often cycling at 6 colors?)

                # TODO add flag to allow duplicate colors (or maybe warn instead?)
                # TODO also show # of distinct colors needed (+ # distinct current
                # palette has), and recommend palettes like hsl/husl?
                raise ValueError(f'{var_names=}: duplicate colors in {palette=}')

            colors = values.applymap(lambda x: value2color[x])
            # TODO condense into one call, rather than looping?
            for c in colors.columns:
                assert all(len(x) == 3 for x in colors[c])
        else:
            vmin = values.min().min()
            vmax = values.max().max()
            if vmax == vmin:
                # TODO add flag that (if True) drops these variables instead?
                # TODO if not, maybe err instead here (requiring they are dropped in
                # advance)?
                warnings.warn(f'{var_names}: min and max are the same (={vmax:.2f})!'
                    ' mapping all to center of cmap!'
                )
                # 0.5 should be argument that produces center of cmap below
                normed = pd.DataFrame(index=values.index, columns=values.columns,
                    data=0.5
                )
            else:
                normed = (values - vmin) / (vmax - vmin)
                assert np.allclose([normed.min().min(), normed.max().max()], [0, 1])

            # TODO or still assume user knows what they were doing if they pass this
            # w/ float values. maybe there really is just a small number of distinct
            # ones? have a dict there imply categorical?
            if isinstance(palette, dict):
                raise ValueError('continuous data incompatible with dict palette')

            # TODO err if one of these continuous variables has palette that is a
            # categorical colormap? (prob more trouble than worth to define which
            # those are...)

            if isinstance(palette, str):
                # sns.color_palette does NOT allow a
                # mpl.colors.LinearSegmentedColormap as argument
                # TODO though mabye i could get list of colors, pass that, and get
                # equivalent output? would have to test
                palette = sns.color_palette(palette, as_cmap=True)
            else:
                assert isinstance(palette, mpl.colors.LinearSegmentedColormap)

            # TODO assert behavior of cmap? can i set indicator values into
            # set_[over|under] and check we get those if i  call w/ 1+eps or 0-eps,
            # but colors at 1 / 0?

            colors = normed.applymap(palette)

            for c in colors.columns:
                # NOTE: these as_cmap=True outputs seem to return RGBA 4-tuples instead
                # of RGB 3-tuples
                # TODO can we just throw away the alpha for all of them? or maybe add
                # alpha of 1 for all the 3-tuples otherwise (renaming fn rgb->rgba
                # then)? (seems all alpha is 1.0 so far, so we can toss)
                assert all(len(x) == 4 for x in colors[c])
                assert all(np.isclose(x[-1], 1) for x in colors[c])
                colors[c] = [x[:-1] for x in colors[c]]

        assert not colors.isna().any().any()
        # TODO .extend instead (w/ `colors` being a list now), or expect DataFrame
        # inputs now?
        color_list.append(colors)

        make_legend = False
        make_cbar = False

        # intended to cover strings, perhaps w/ None/NaN as well
        if dtype == np.dtype('O'):
            make_legend = True

        elif dtype == int or dtype == float:
            # TODO still treat (some?) int types as diff from float, perhaps drawing
            # colorbar with discrete steps (at least for cases where above would treat
            # as categorical, which may in the future only be for certain subsets of
            # dtype int data)?
            make_cbar = True
        else:
            # TODO delete (/move earlier)? dupe w/ above?
            raise NotImplementedError(f'{dtype=} unexpected')

        # TODO take input for these titles? don't specify?
        # (dict of var name -> title, and assert that vars sharing palette also share
        # title [at least unless separate flag indicates we shouldn't share any
        # palettes])
        title = '/'.join(var_names)

        if make_legend:
            # TODO allow some separate specification of formatter fn for each
            # column (other than (str(v))?
            handles = [
                Patch(facecolor=c, label=str(v)) for v, c in value2color.items()
            ]
            for_legends.append(dict(handles=handles, title=title))

        if make_cbar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            if isinstance(palette, sns.palettes._ColorPalette):
                # TODO test by passing current output to imshow, and inspecting
                # ScalarMappable from output of that? (to check we are getting same
                # value->color mapping as would be displayed w/o manually making
                # ScalarMappable from this)
                cmap = mpl.colors.ListedColormap(palette)
            else:
                assert not isinstance(palette, dict)
                cmap = palette

            # TODO check labels are centered on value, for discrete cmaps
            # (discrete now seems to be default behavior for int dtype, but not sure
            # labels are centered how i might want?)

            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            for_cbars.append(dict(mappable=sm, label=title))


    for palette_id, var_names in palette2vars.items():
        if isinstance(palette_id, int):
            palette = dict_palette_lookup[palette_id]
        else:
            assert isinstance(palette_id, str)
            if palette_id in mpl_palette_lookup:
                palette = mpl_palette_lookup[palette_id]
            else:
                palette = palette_id

        first_categorical = None
        first_dtype = None
        for n in var_names:
            # TODO update indexing to support axis=1|'columns'
            values = df[n]
            dtype = values.dtype
            categorical = is_categorical(values)
            if first_categorical is None:
                first_categorical = categorical
                first_dtype = dtype
            else:
                assert categorical == first_categorical
                assert dtype == first_dtype
        del values, n

        # TODO delete
        if _debug:
            print()
            print(f'{palette_id=}')
            print(f'{var_names=}')
            print(f'{palette=}')
            print(f'{categorical=}')
            print(f'{dtype=}')
        #

        if not share_palettes_with_same_name and len(var_names) > 1:
            for n in var_names:
                # TODO update indexing to support axis=1|'columns'
                values = df[n].to_frame()
                _map_vars_to_colors(values, palette, dtype, categorical)
        else:
            # TODO update indexing to support axis=1|'columns'
            values = df[var_names]
            # this appends one element to colors and (for_legends or for_cbars), per
            # call.
            _map_vars_to_colors(values, palette, dtype, categorical)

    color_df = pd.concat(color_list, axis='columns')
    assert color_df.index.equals(df.index)
    assert color_df.columns.equals(df.columns)

    # TODO or if i'm going to allow input in NaN, also allow (only those same NaNs!)
    # in output (or want a parameter for a color NaN should be assigned to? specific to
    # cmaps? if i take mpl cmaps as input, which have set_bad() done, use that?)
    assert not color_df.isna().any().any()

    return color_df, for_legends, for_cbars


# TODO test all of these are valid and put things where i expect
# TODO delete if i can programmatically layout based on bbox of previous legend
# (or otherwise automatically layout the legends, maybe w/ constrained layout, or
# some kind of Axes grid just for these [w/ one legend fully filling each axes in
# grid]?)
# TODO share for colorbars too (these named locations even work for those calls?)?
# rename? (/ remove 'center right' as option? currently that's where i'm putting
# colorbars, though referencing that location via fig coordinates, not as a named
# location)
# TODO update type hint for legend_locations to allow for non-str locations? what are
# other options mpl call takes (4-tuples of floats? anything else?)?
#
# picking title/xlabel/ylabel positions last, as I often include lots of information
# there.
all_legend_locations = ['upper right', 'lower right', 'lower left', 'upper left',
    'center right', 'lower center', 'center left', 'upper center'
]
def add_legends_and_colorbars(fig: Figure, for_legends: List[Dict[str, Any]],
    for_cbars: List[Dict[str, Any]], *,
    legend_locations: Sequence[str] = tuple(all_legend_locations), left: float = 0.85,
    width: float = 0.02, hmargin: float = 0.06, bottom: float = 0.3,
    height: float = 0.25, cbar_fontsize: Optional[float] = None,
    cbar_label_fontsize: Optional[float] = None,
    cbar_ticklabel_fontsize: Optional[float] = None) -> None:
    """
    Args:
        fig: figure to add legends and colorbars too

        for_legends: as returned by `map_each_series_to_rgb`. List of dicts, with each
            dict containing keyword arguments for a `fig.legend` call. This function
            adds a unique `loc=` argument to each call, chosen in order from
            `legend_locations`.

        for_cbars: as returned by `map_each_series_to_rgb`. List of dicts, with each
            dict containing keyword arguments for a `fig.colorbar` call. This function
            adds `cax=<Axes>` argument to each call.

        legend_locations: unique sequence of locations for `loc=` argument to each
            `fig.legend` call. Will use in order, up to length of `for_legends`. Must
            currently be >= length of `for_legends`.

        left|width|hmargin|bottom|height: used to place colorbars. All floats [0,1] in
            fractions of figure size, with [0, 0] at bottom left of figure. `hmargin` is
            space between colorbars, which is needed for labels and ticklabels.
            Colorbars start at `left`, and then each subsequent colorbar is placed to
            the right (with `width` and `hmargin` controlling spacing).

        cbar_fontsize: if passed, will be used to define both `cbar_label_fontsize` and
            `cbar_ticklabel_fontsize`

        cbar_label_fontsize: font size of colorbar label
        cbar_ticklabel_fontsize: font size of colorbar ticklabels
    """
    if len(for_legends) > len(legend_locations):
        # TODO choose loc for one based on loc of current? or layout not
        # done yet? (prob more complicated than it's worth anyway,
        # unless i need it)
        raise NotImplementedError('need to come up with new strategy '
            f'to pick legend locations, for >{len(legend_locations)} '
            'legends'
        )

    for legend_kws, loc in zip(for_legends, legend_locations):
        assert 'loc' not in legend_kws
        fig.legend(**legend_kws, loc=loc)

    if cbar_fontsize is not None:
        assert cbar_label_fontsize is None
        assert cbar_ticklabel_fontsize is None
        cbar_label_fontsize = cbar_fontsize
        cbar_ticklabel_fontsize = cbar_fontsize

    # TODO also need param for cbar fontsize (for label and ticklabels)?
    # width=0.05 was a bit too much, at least in context of first test
    # hmargin needed for labels on each axes, which aren't fit within width
    for i, cbar_kws in enumerate(for_cbars):
        curr_left = left + i * (width + hmargin)
        # left, bottom, width, height (all in fractions of fig size)
        ax = fig.add_axes([curr_left, bottom, width, height])
        # TODO check we don't exceed 1.0 with curr_left (or make some adjust call if we
        # do?)? matter (even if saved w/ bbox_inches='tight' to savefig call?)
        # TODO check curr_left doesn't overlap w/ anything already in figure? matter?
        # TODO check these axes are in layout (or separately add?)? matter?
        cbar = fig.colorbar(cax=ax, **cbar_kws)

        for t in cbar.ax.get_yticklabels():
            # TODO delete
            # default currently 10.0
            #print(f'cbar tick fontsize: {t.get_fontsize()}')
            if cbar_ticklabel_fontsize is not None:
                t.set_fontsize(cbar_ticklabel_fontsize)

        # TODO delete
        # default currently 10.0
        #print('{cbar.ax.yaxis.label.get_size()=}')
        if cbar_label_fontsize is not None:
            cbar.ax.yaxis.label.set_fontsize(cbar_label_fontsize)


# TODO also work w/ 4-tuple RGBA inputs? (allow mixed 3 and 4 tuples too?)
def rgb_tuple_df_to_3d_array(color_df: pd.DataFrame) -> np.ndarray:
    """Converts (M, N) shape dataframe w/ RGB tuple values to (M, N, 3) array.

    `map_each_series_to_rgb` can produce RGB tuple dataframe suitable for input to this
    function.
    """
    # TODO how does seaborn clustermap row_colors code handle the same problem, for
    # input w/ RGB tuples?
    n_channels = 3
    assert (color_df.apply(lambda x: x.str.len()) == n_channels).all().all()
    channels = []
    for i in range(n_channels):
        channels.append(color_df.applymap(lambda x: x[i]))

    arr = np.stack(channels, axis=-1)
    assert arr.shape == color_df.shape + (n_channels,)

    assert arr.dtype == float
    assert arr.min() >= 0
    assert arr.max() <= 1

    return arr


def remove_axes_ticks(ax: Axes) -> None:
    # (was) trying to recreate ax.axis('off'), but in a way where i can still see
    # the ylabels (so micrometer_depth_title's as_ylabel kwarg can work)
    #
    # i feel like at some point i felt ax.axis('off') did more, but seems OK for now
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


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


# TODO take dict of level -> formatting str/fn (name `formatters` as
# DataFrame.to_string())?
def format_index_row(row: pd.Series, *, delim: str = ' / ',
    float_format: Optional[str] = None, _debug=False):
    """Takes Series to a str with the concatenated values, separated by `delim`.
    """
    def _format_value(x) -> str:
        # TODO handle NaN separately (defaulting to not showing?) like na_rep in
        # DataFrame.to_string()?
        if isinstance(x, float) and float_format is not None:
            # assuming float_format is a valid float formatting string (e.g. '.2f')
            return f'{x:{float_format}}'

        return str(x)

    return delim.join([_format_value(x) for x in row.values])


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
    def _check_bools(ticklabels, **kwargs):
        # TODO rewrite doc. don't know what i was even doing passing (a single?) str
        # before...
        """Makes True equivalent to passing str, and False equivalent to None.
        """
        if ticklabels == True:
            # TODO TODO probably also default to this in case ticklabels=None
            # (might just need to modify some of the calling code)
            format_fn = lambda x: format_index_row(x, **kwargs)
            return format_fn

        # NOTE: this broke ticklabel=False support sns.clustermap has (from
        # sns.heatmap). currently moved this processing into matshow, which had come to
        # expect this behavior (but maybe revert the matshow changes alongside when this
        # code was added?)
        #elif ticklabels == False:
        #    return None

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

        # TODO use inspect to get any kwargs to format_index_row, then pass that subset
        # of kwargs thru to _check_bools (so i don't have to keep manually including in
        # this list?)?
        default_format_fn_kwarg_names = ['delim', 'float_format', '_debug']
        _pass_thru_to_plot_fn = ['_debug']
        default_format_fn_kws = {
            k: kwargs[k] for k in default_format_fn_kwarg_names if k in kwargs
        }
        for k in default_format_fn_kws.keys():
            if k not in _pass_thru_to_plot_fn:
                kwargs.pop(k)
        #

        if isinstance(df_or_arr, xr.DataArray):
            # Requires df_or_arr to be <=2d
            df = df_or_arr.to_pandas()
        else:
            df = df_or_arr

        if 'xticklabels' in kwargs:
            xticklabels = _check_bools(kwargs['xticklabels'], **default_format_fn_kws)
            if callable(xticklabels):
                xticklabels = col_labels(df, xticklabels)

            kwargs['xticklabels'] = xticklabels

        if 'yticklabels' in kwargs:
            yticklabels = _check_bools(kwargs['yticklabels'], **default_format_fn_kws)
            if callable(yticklabels):
                yticklabels = row_labels(df, yticklabels)

            kwargs['yticklabels'] = yticklabels

        return plot_fn(df, *args, **kwargs)

    return wrapped_plot_fn


# TODO delete (or add to test_viz.test_is_cmap_diverging)
'''
def _palettable_diverging_cmaps():
    # TODO move up if i end up using
    from importlib import import_module

    from palettable.palette import Palette
    #

    # TODO TODO where is 'vlag'? in here? doesn't seem to be in palettable. fuck.
    # (not in 'matplotlib', it seems)

    # copied from doc
    modules_with_palettes = [
        'cartocolors.diverging',
        #'cartocolors.qualitative',
        #'cartocolors.sequential',
        'cmocean.diverging',
        #'cmocean.sequential',
        'colorbrewer.diverging',
        #'colorbrewer.qualitative',
        #'colorbrewer.sequential',
        'lightbartlein.diverging',
        #'lightbartlein.sequential',

        # TODO do these also have cmaps labelled as divering?
        'matplotlib',
        'mycarta',

        'scientific.diverging',
        #'scientific.sequential',

        # TODO do these also have cmaps labelled as divering?
        'tableau',
        'wesanderson',
    ]

    all_diverging_palettes = []
    for mod_name in modules_with_palettes:
        mod = import_module(f'.{mod_name}', 'palettable')
        print(mod_name)

        # TODO delete
        #mod.print_maps()

        # (at least for cartocolors.diverging)
        # this seems to include stuff with '_r' at end, where mod.print_maps() does not.
        # each there has a '_r' variant. otherwise, palette lists look the same as in
        # mod.print_maps.
        palettes = [
            x for x in dir(mod) if isinstance(getattr(mod, x), Palette)
        ]
        # TODO check against internal variable listing palettes? can we rely on such a
        # variable? what name?

        diverging_palettes = [
            f'{mod_name}.{x}' for x in palettes if getattr(mod, x).type == 'diverging'
        ]
        # TODO check if plt.get_cmap(x) works for each?
        # TODO TODO maybe only return ones where that is true?

        all_diverging_palettes.extend(diverging_palettes)

        # TODO delete
        #pprint(diverging_palettes)
        #print()

    # TODO TODO check whether just last component of names are unique across modules
    # (hopefully all could be provided to plt.get_cmap?, so we can check cmap.name
    # against a set from these names)
    pprint(all_diverging_palettes)
    import ipdb; ipdb.set_trace()
'''


def is_cmap_diverging(cmap) -> bool:
    """Returns guess as to whether input colormap is a diverging colormap.

    Args:
        cmap: anything that could be passed to `plt.get_cmap`, such as str colormap name
    """
    # NOTE: does not support 'RdGy', or other diverging cmaps where one side is not
    # saturated

    cmap = plt.get_cmap(cmap)

    # TODO delete
    '''
    # TODO just iterate over list of entries in cmap (many already enumerated, right?)
    # do they interpolate between (lut? is that what lut is?) entries?
    # TODO or at least change num=50 to default # of entries in lut (/cmap)?
    xs = np.linspace(0, 1, 50)
    cs = np.array([cmap(x) for x in xs])
    # shape: (n_colors, 3)
    cs_hsv = np.array([rgb_to_hsv(*c[:3]) for c in cs])
    saturations = cs_hsv[:, 1]
    plt.close('all')
    print(f'{cmap.name=}')
    plt.plot(xs, saturations)
    plt.title(cmap.name)
    plt.show()
    import ipdb; ipdb.set_trace()
    '''
    #

    def _get_hsv(x):
        assert 0. <= x <= 1.
        return rgb_to_hsv(*cmap(x)[:3])

    def _get_sat(x):
        return _get_hsv(x)[1]

    # TODO also fn to check value (as in V from HSV) at middle, to determine whether to
    # draw using white / black over it?

    # TODO check minimum saturation is ~middle. anything else?

    mid = _get_sat(0.5)

    low = _get_sat(0.0)
    high = _get_sat(1.0)

    # mid should be pretty much either black or white. adding this check allows us to
    # support stuff like 'icefire' w/o false positive of 'cividis' (which I also use)
    mid_val = _get_hsv(0.5)[2]

    # TODO delete
    #print(f'{cmap.name}: {mid=}, {low=}, {high=}')
    #print(f'mid val: {mid_val}')
    #
    if 0.15 < mid_val < 0.85:
        return False

    return mid < 0.28 and (low > mid * 2 and high > mid * 2)


def add_norm_options(plot_fn):
    # TODO edit doc. copied from back before this was a decorator
    # TODO prob rewrite doc + fn to use str 'centered' / 'two-slope' instead of (more
    # complicated for user) classes
    """Processes kwargs to allow passing non-instantiated `Normalize` subclasses.

    Args:
        norm: matplotlib expects str (e.g. 'linear', 'log', 'symlog', 'logit') or
            an instantiated `Normalize` subclass (or `None`), but passing a normalize
            subclass like this conflicts with non-`None` `vmin`|`vmax`. this function
            adds the option to pass a non-instantiated `Normalize` class, which will
            receive `vmin`|`vmax`, removing those from kwargs. this will let plotting
            functions determine the range of data to plot, while still allowing use of
            norms not able to be passed a str.

    Returns **kwargs, with vmin,vmax,norm modified when necessary.
        kwargs['norm'] (if not str or None) will always be an instantiated `Normalize`
        subclass (unlike input, where it's allowed to also be non-instantiated).

    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    for descriptions of `vmin`/`vmax`/`norm` kwargs.
    """
    # TODO possible to inspect whether wrapped fn already has defaults for vmin/vmax,
    # and use those if so (for dff_imshow in al_analysis)?
    @functools.wraps(plot_fn)
    def wrapped_plot_fn(data, *args, norm=None, vmin=None, vmax=None, vcenter=None,
        **kwargs):

        if norm == 'two-slope':
            norm = TwoSlopeNorm

        elif norm == 'centered':
            norm = CenteredNorm

        # NOTE: TwoSlopeNorm/CenteredNorm (as defined in conditionals above) are classes
        # and not instances of them, and so the isinstance call will be False
        if norm is None or type(norm) is str or isinstance(norm, Normalize):
            assert vcenter is None, 'explicit vcenter not supported in this case'
            return plot_fn(data, *args, norm=norm, vmin=vmin, vmax=vmax, **kwargs)

        assert issubclass(norm, Normalize)

        # using None should give us consistent default behavior w/ plt.get_cmap
        cmap = kwargs.get('cmap', None)
        if not is_cmap_diverging(cmap):
            warnings.warn(f'using seemingly non-diverging cmap ({cmap.name}) with '
                f'{norm=}, where center has special meaning. probably a mistake!'
            )

        # TODO also support discrete colormap here (instantiating from just str cmap
        # name)? would be to replace stuff like in al_analysis.plot_n_per_odor_and_glom

        dmin = data.min().min()
        dmax = data.max().max()

        if vmin is None:
            vmin = dmin
            if vcenter is not None:
                assert vmin > vcenter

        if vmax is None:
            vmax = dmax
            if vcenter is not None:
                assert vcenter < vmax

        # TODO could also try to instead check if norm takes vcenter kwarg?
        # are there actually any (builtin to matplotlib, at least) norms that take
        # vcenter besides these two?
        centered_norm = False
        if issubclass(norm, TwoSlopeNorm) or issubclass(norm, CenteredNorm):
            centered_norm = True

        if centered_norm and vcenter is None:
            vcenter = 0
            if vmin > vcenter:
                # TODO maybe warn in this case?
                vmin = 0

        # modifying vmin=0 for convenience when calling (when vcenter would also be 0),
        # rather than hardcoding vmin=-epsilon a bunch of places
        if vmin == vcenter:
            # NOTE: this would break CenteredNorm stuff below, but might want to
            # delete that option anyway...
            #
            # needs to actually be slightly less than 0, as vmin=vcenter triggers
            # same ValueError about vmin, vcenter, vmax not being in ascending order
            old_vmin = vmin
            epsilon = 1e-6
            vmin = vcenter - epsilon
            warnings.warn(
                f'setting {vmin=} (was {old_vmin}) to make vmin < {vcenter=} True'
            )
            del old_vmin

        assert vmin is not None and vmax is not None

        # could pop('vcenter', 0) from kwargs, as both Centered/TwoSlope take it, but i
        # think i prob always want it at 0 (the default for these)
        # TODO actually, probably do want to pop it (why?)

        if issubclass(norm, TwoSlopeNorm):
            # NOTE: if (vmin < vcenter < vmax) is not strictly True
            # (e.g. if vmin == vcenter), the following line would raise:
            # ValueError: vmin, vcenter, and vmax must be in ascending order
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            # TODO test + doc clip behavior of TwoSlopeNorm (same as clip=False? True?)

        # TODO delete? even care about dynamically instantiated CenteredNorm in plotting
        # fns (maybe if i warn, but allow+process vmin/vmax not symmetric about 0?)?
        elif issubclass(norm, CenteredNorm):
            if vcenter != 0 or vmin < 0:
                # TODO TODO may need to take difference from vcenter first -> add back?
                print()
                print('finish supporting vcenter in CenteredNorm case')
                print(f'{vmin=}')
                print(f'{vcenter=}')
                import ipdb; ipdb.set_trace()

            # TODO require/accept a halfrange kwarg? (mutex w/ vmin/vmax)
            assert vmin < 0, 'halfrange calculation below assumes vmin > 0 (center=0)'

            halfrange = max(-vmin, vmax)
            # NOTE: this has clip=False by default (UNLIKE two slope, which doesn't
            # support clip kwarg)
            #
            # regarding halfrange "Defaults to the largest absolute difference to
            # vcenter for the values in the dataset."
            # TODO how does it do that? via what imshow does w/ norm input?
            norm = CenteredNorm(vcenter=vcenter, halfrange=halfrange)

        else:
            raise NotImplementedError

        # TODO threshold fraction before warning?
        # TODO TODO only warn here if we can't show clipping in plot
        # (like w/ default TwoSlopeNorm [right?]) (and/or not configured to)

        # .<op>().<op>() should work w/ at least ndarrays and DataFrames
        # (e.g. for min/max/sum ops)
        vmin_n_clipped = (data < vmin).sum().sum()
        vmin_clipped_frac = vmin_n_clipped / data.size
        clip_msgs = []
        if vmin_clipped_frac > 0:
            clip_msgs.append(
                f'{vmin_clipped_frac:.4f} of data ({vmin_n_clipped} points) < {vmin=}'
            )

        vmax_n_clipped = (data > vmax).sum().sum()
        vmax_clipped_frac = vmax_n_clipped / data.size
        if vmax_clipped_frac > 0:
            clip_msgs.append(
                f'{vmax_clipped_frac:.4f} of data ({vmax_n_clipped} points) > {vmax=}'
            )

        if len(clip_msgs) > 0:
            # TODO have this be an error by default, allowing decorator (/decorated
            # fn) to be run w/ warning instead, so that it can err if any of matshow
            # data is over/under (but more OK/expected to have imshow data over/under)
            # (just watch for plot_fn.__name__ == 'matshow' in output...)
            warnings.warn(f"{plot_fn.__name__} (add_norm_options wrapper): "
                f"{', '.join(clip_msgs)}"
            )

        # TODO delete
        # see notes in add_colorbar TwoSlopeNorm section for some hints about plots that
        # might be affected by this (maybe more plots affected by this fn than that
        # one?)
        if _debug:
            print(f'{plot_fn.__name__}: end of add_norm_options: {vmin=}, {vmax=}')
        #

        # not passing vmin/vmax, as matplotlib would raise ValueError since norm also
        # passed
        return plot_fn(data, *args, norm=norm, **kwargs)

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
            # TODO TODO doc how this interacts w/ ax passed in already having particular
            # layout. probably want to be able to override (e.g. to deal with matshow's
            # issue where once i start adding text (for group labels), constrained
            # layout can cut them off)
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


@add_norm_options
@no_constrained_layout
@callable_ticklabels
# TODO TODO do [x|y]ticklabels now need to be extracted from kwargs? if seaborn doesn't
# handle that, then the @callable_ticklabels decorator is doing nothing here.
# TODO modify to call matshow internally, rather than relying on seaborn much/at all?
# (to get the convenience features i added to matshow...) (or at least make another
# decorator like callable_ticklabels to deal w/ [h|v]line_level_fn matshow kwargs)
def clustermap(df, *, optimal_ordering: bool = True, title=None, xlabel=None,
    ylabel=None, ylabel_rotation=None, ylabel_kws=None, cbar_label=None, cbar_kws=None,
    row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None,
    method='average', metric='euclidean', z_score=None, standard_scale=None,
    return_linkages: bool = False, **kwargs):
    """Same as seaborn.clustermap but allows callable [x/y]ticklabels + adds opts.

    Adds `optimal_ordering` kwarg to `scipy.cluster.hierarchy.linkage` that is not
    exposed by seaborn version.

    Also turns off constrained layout for the duration of the seaborn function, to
    prevent warnings + disabling that would otherwise happen.

    Args:
        method: default 'average' is same as `sns.clustermap`
        metric: default 'euclidean' is same as `sns.clustermap`
    """
    if row_linkage is not None:
        # TODO or does seaborn already have passed row_linkage imply row_cluster=True?
        row_cluster = True
        # seaborn will just show a subset of the data if passed a linkage of a smaller
        # shape. Not sure what happens in reverse case. Either way, I think failing
        # is safer.
        expected_row_linkage_shape = (df.shape[0] - 1, 4)
        if row_linkage.shape != expected_row_linkage_shape:
            raise ValueError(f'row_linkage.shape must be {expected_row_linkage_shape}')

    if col_linkage is not None:
        col_cluster = True
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

    let_seaborn_compute_linkages = False
    if z_score is not None or standard_scale is not None:
        assert row_linkage is None and col_linkage is None

        if return_linkages:
            # TODO unless it's available in the seaborn object, or i want to
            # re-implement this preprocessing before my own linkage computations?
            raise NotImplementedError('can not return linkages while z_score or '
                'standard_scale is True'
            )

        if optimal_ordering:
            warnings.warn('disabling optimal_ordering since z_score or standard_scale')
            optimal_ordering = False

        let_seaborn_compute_linkages = True

        kwargs['z_score'] = z_score
        kwargs['standard_scale'] = standard_scale

    # TODO if z-scoring / standard-scaling requested, calculate before in this case
    # (so it actually affects linkage, as it would w/ seaborn version)
    # (currently just disabling optimal ordering in these cases)
    def _linkage(df):
        # TODO way to get this to work w/ some NaNs? worth it?
        return linkage(df.values, optimal_ordering=optimal_ordering, method=method,
            metric=metric
        )

    if not let_seaborn_compute_linkages:
        if row_cluster and row_linkage is None:
            row_linkage = _linkage(df)

        # This behavior of when to transpose for which linkage is consistent w/ seaborn
        # (I read clustermap implementation)
        if col_cluster and col_linkage is None:
            col_linkage = _linkage(df.T)

    # TODO assert len(df) > 0 (both dims?) early on / raise ValueError
    # clustermap call will fail in this case, w/ somewhat confusing error message:
    # ValueError: zero-size array to reduction operation fmin which has no identity

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

    if not return_linkages:
        return clustergrid
    else:
        # can use scipy.cluster.hierarchy.leaves_list to get order of input data from
        # these linkages
        return clustergrid, row_linkage, col_linkage


# TODO appropriate type hints for x and y? Sequence[str]?
# TODO type hint x & y
def add_group_labels_and_lines(ax: Axes, x=None, *, y=None, lines: bool = True,
    labels: bool = True, formatter: Optional[Callable] = None,
    label_offset: Optional[float] = None, label_name: Optional[str] = None,
    label_name_offset: float = 3, linewidth=0.5, linecolor='k', _debug=False, **kwargs
    ) -> None:
    """Adds labels to (and lines between) groups of common x/y labels.

    Args:
        **kwargs: passed thru to `ax.text` calls for group text labels
    """
    # TODO validation on length of x/y? what to check against? doc, if i come up
    # with something (check against [x|y]ticklabels, from Axes [assuming already set]?)

    # TODO maybe allow specifying both in one call?
    if x is not None and y is not None:
        raise ValueError('must only specify x OR y')

    if x is not None:
        levels = x
        line_fn = ax.axvline
        # TODO modify so it's above axes, not below (2024-06-05: still relevant?)
        group_text_coords = (ax.transData, ax.transAxes)

        if label_offset is None:
            label_offset = 0.08

    elif y is not None:
        levels = y
        line_fn = ax.axhline
        group_text_coords = (ax.transAxes, ax.transData)

        if label_offset is None:
            label_offset = 0.12
    else:
        raise ValueError('must specify either x/y')

    if not (lines or labels):
        raise ValueError('at least one of lines/labels must be True')

    # TODO lines showing extent that each text label applies to?
    # (e.g. parallel to labelled axis, with breaks between levels? might crowd fly
    # labels less than separator lines perpendicular to axis)
    # TODO make linewidth a constant fration of cell width/height (whichever is
    # appropriate) (at least by default) ~(figsize[i] / df.shape[i])?
    # TODO what is default linewidth here anyway? unclear. 1?
    # TODO default to only formatting together index levels not used by
    # [h|v]line_level_fn (possible?), when ?

    # TODO modify const_ranges to have include_val=True behavior be default?
    # (+ delete switching flag, if so)
    ranges = util.const_ranges(levels, include_val=True)

    # If all the ranges have the same start and stop, all groups are length 1, and
    # the lines would just add visual noise, rather than helping clarify boundaries
    # between groups.
    if lines and any([x[-1] > x[-2] for x in ranges]):
        line_positions = [x[-1] + 0.5 for x in ranges[:-1]]
        # TODO if we have a lot of matrix elements, may want to decrease size of
        # line a bit to not approach size of matrix elements...
        for v in line_positions:
            # TODO delete try/except. not sure i can repro this error...
            try:
                # 'w'=white. https://matplotlib.org/stable/tutorials/colors/colors.html
                line_fn(v, linewidth=linewidth, color=linecolor)

            except np.linalg.LinAlgError as err:
                import ipdb; ipdb.set_trace()

    # TODO allow separating group text from levels? accept yet another fn mapping
    # from [v|h]line_levels (dict level name -> value form?) to formatted strs?
    if labels:
        trans = transforms.blended_transform_factory(*group_text_coords)

        if x is not None:
            def text_fn(label, start, stop):
                ax.text(np.mean((start, stop)), 1 + label_offset, label,
                    ha='center', va='bottom', transform=trans, **kwargs
                )

        elif y is not None:
            def text_fn(label, start, stop):
                # TODO compute group_label_offset? way to place the text using
                # constrained layout?
                # TODO possible to get this work w/ constrained layout /
                # bbox_inches='tight' (work w/ either as-is?)?
                # TODO what happens is label is not str? does nothing? why not erring?
                # or is it not displaying for another reason? (converting via str didn't
                # seem to change anything...)
                ax.text(-label_offset, np.mean((start, stop)), label,
                    # Right might make consistent spacing wrt line indicating extent of
                    # group easier to see.
                    ha='right',
                    # for 9x9 input at least, va='center' was better than without
                    # (though still seemed *slightly* above corresponding ticklabel)
                    va='center',
                    transform=trans, **kwargs
                )
                # TODO TODO how is above text seemingly not clipped and in layout, but
                # not showing up after constrained layout, and getting that warning:
                # ./test_matshow.py:41: UserWarning: constrained_layout not applied
                # because axes sizes collapsed to zero.  Try making figure larger or
                # axes decorations smaller.
                # ipdb> text.get_in_layout()
                # True
                # ipdb> text.get_clip_on()
                # False
                # TODO TODO can i register (some restricted version of?)
                # warnings.filterwarnings('error',
                #     message='constrained_layout not applied.*'
                # )
                # here? (to ensure text is either fully shown or there is an error)
        else:
            assert False

        if _debug:
            # NOTE: may need to tweak this offset if current value is causing
            # constrained layout warning for a given call
            print(f'{label_offset=}')
            print('group labels:')

        for label, start, stop in ranges:
            # assuming const_ranges index output wouldn't change by formatting
            if formatter is not None:
                label = formatter(label)

            text_fn(label, start, stop)

            if _debug:
                print(f'{label=} ({start=}, {stop=})')

        if label_name is not None:
            # TODO ideally allow configuring such that it could also be centered on
            # relevant axis (rather than off to one side), but then would have to place
            # in a way not relying on text_fn... (or would at least need to modify
            # text_fns)
            text_fn(label_name, -label_name_offset, 1 - label_name_offset)


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
# TODO TODO make default behavior show all ticklabels (of any multiindices), and only if
# [x|y]ticklabels=False don't. might be a bit of work to have defaults look reasonable
# in a wide variety of cases tho... (mostly b/c fontsize issues?)?
@add_norm_options
# TODO TODO should i move off of constrained layout, now that it's giving me issues
# placing group labels via ax.text (may just need to manually tweak an offset for each
# plot, without changing implementation substantially...)
@constrained_layout
@callable_ticklabels
# TODO TODO should be raising a warning by default if [h|v]line_level_fn not producing
# any divisions
def matshow(df, title: Optional[str] = None, ticklabels=None, xticklabels=None,
    yticklabels=None, xtickrotation=None, xlabel: Optional[str] = None,
    ylabel: Optional[str] = None, ylabel_rotation=None,
    ylabel_kws: Optional[KwargDict] = None,
    cbar_label: Optional[str] = None, group_ticklabels=False, vline_level_fn=None,
    hline_level_fn=None, vline_group_text: bool = False, hline_group_text: bool = False,
    # TODO combine [h|v]line_group_text into these, using True for no formatting?
    # (or these into former...)
    vgroup_formatter=None, hgroup_formatter=None,
    # TODO TODO can maybe specify in axes coords w/ new blended transform approach i'm
    # trying
    # TODO move fontsize default out too. too big for small matrices (e.g. 9x9)
    # TODO what value to use for axes-coords offets? still need these kwargs to vary?
    # TODO rename label->text to be consistent (or vice versa)
    # TODO rename h/v -> x/y for easier understanding?
    # these values seemed ok for 9x9 sensitivity analysis input
    # (but causing issues w/ plot_n_per_odor_and_glom plot, and probably also similar
    # old top-level al_analysis ijroi plots)
    vgroup_label_offset: float = 0.08, hgroup_label_offset: float = 0.12,
    vgroup_label_rotation: Union[float, str] = 'horizontal',
    hgroup_label_rotation: Union[float, str] = 'horizontal',
    group_fontsize: Optional[float] = None, group_fontweight=None, linewidth=0.5,
    linecolor='w', ax: Optional[Axes] = None, fontsize: Optional[float] = None,
    bigtext_fontsize_scaler: float = 1.5, fontweight=None, figsize=None, dpi=None,
    inches_per_cell=None, extra_figsize=None, transpose_sort_key=None,
    colorbar: bool = True, cbar_shrink: float = 1.0,
    cbar_kws: Optional[KwargDict] = None, levels_from_labels: bool = True,
    allow_duplicate_labels: bool = False, xticks_also_on_bottom: bool = False,
    overlay_values: bool = False, overlay_fmt: str = '.2f',
    overlay_kws: Optional[KwargDict] = None, _debug=_debug, **kwargs
    ) -> Tuple[Figure, AxesImage]:
    # TODO doc [v|h]line_group_text
    # TODO check that levels_from_labels means *_level_fn get a single dict as input,
    # not an iterable of dicts (or update doc)
    # TODO in levels_from_labels doc, specify what would be doing the formatting of
    # ticklabels (presumably it's just the callable passed in for [x|y]ticklabels, which
    # is expected to map to str (from what again?)?, or nothing if input is single level
    # str index? what happens if [x|y]ticklabels input is not callable and input data
    # does not have a single level str index?)
    # TODO switch levels_from_labels default to False? would prob need to change a fair
    # bit of al_analysis code...
    # TODO accept str level names for specifying [v|h]line_levels (maybe rename
    # [v|h]line_level_fn to include this type of input?)
    # TODO + support labelling level names off to side in that case?
    # TODO am i supporting False for [x|y]ticklabels? how else to hide default
    # regularly-spaced int ones?
    """
    Args:
        transpose_sort_key (None | function): takes df.index/df.columns and compares
            output to decide whether matrix should be transposed before plotting

        vline_level_fn: callable whose output varies along axis labels/index (see
            `levels_from_labels` for details). vertical lines will be drawn between
            changes in the output of this function.

        hline_level_fn: as `vline_level_fn`, but for horizontal lines.

        [h|v]group_formatter: optional function to map group values to str

        [h|v]group_label_offset: in axes (not data) coordinates. should probably be in
            [0, 1] and pretty close to 0. increase if constrained layout warning emitted
            when showing / saving plot.

        [h|v]group_label_rotation: passed to `rotation` of corresponding `Axes.text`
            call

        levels_from_labels: if True, `[h|v]line_level_fn` functions use formatted
            `[x|y]ticklabels` as input. Otherwise, a dict mapping index level names to
            values are used. Currently only support drawing labels for each group if
            this is False (still true?).

        xticks_also_on_bottom: if True, show same xticks on bottom as will be shown on
            top. for very tall plots where we can sometimes more easily reference the
            bottom than the top, when scrolling through.

        overlay_values: if True, overlays text with numerical value for each cell

        overlay_fmt: float format str, used to format `overlay_values` text

        overlay_kws: optional dict of kwargs to pass to overlay `ax.text` calls

        fontsize: directly used for ticklabels, and scaled by `bigtext_fontsize_scaler`
            for xlabel/ylabel fontsize. also used for group label font, unless
            `group_fontsize` is passed a float.

        **kwargs: passed thru to `matplotlib.pyplot.matshow`
    """
    # added after reverting callable_ticklabels code that would do the same,
    # which was needed to continue using w/ sns.clustermap
    if xticklabels == False:
        xticklabels = None
    if yticklabels == False:
        yticklabels = None

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

    # TODO warn if ax not set? or at least in the case where plt.gca() returns
    # something? (to make sure we aren't making figures that don't get populated/closed)
    # (factor this check into wrapper shared w/ imshow?)
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

            # TODO is this even possible at this point? if so, why not handled by
            # callable_ticklabels wrapper?
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
        # TODO delete this... more trouble than it's worth these days
        fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))

    bigtext_fontsize = bigtext_fontsize_scaler * fontsize

    im = ax.matshow(df, **kwargs)

    if colorbar:
        if cbar_kws is None:
            cbar_kws = dict()

        # TODO TODO automatically set default extend='both'|'min'|'max'|'neither' based
        # on which limits actually exceeded by data?

        # TODO TODO use same extend= kwarg logic as i settle on for image_grid? refactor
        # to share? (want to be able to show clipped data in both cases)
        #
        # rotation=270?
        # TODO thread fontsize thru this?
        # TODO TODO add option to fix size to that of axes we are stealing from
        # (or maybe to set size relative to those?)?
        cbar = add_colorbar(fig, im, label=cbar_label, shrink=cbar_shrink,
            fontsize=bigtext_fontsize, **cbar_kws
        )

    if group_fontsize is None:
        group_fontsize = bigtext_fontsize

    # TODO delete?
    def grouped_labels_info(labels):
        if not group_ticklabels or labels is None:
            return labels, 1, 0

        # TODO TODO delete remove_consecutive_repeats stuff (replacing w/
        # util.const_ranges based code, which i'm already using below, and which [unlike
        # this] supports variable numbers of repeats)?
        without_consecutive_repeats, n_repeats = remove_consecutive_repeats(labels)
        tick_step = n_repeats
        offset = n_repeats / 2 - 0.5
        return without_consecutive_repeats, tick_step, offset

    # TODO skip if either is None?
    # TODO make fontsize / weight more in group_ticklabels case?
    xticklabels, xstep, xoffset = grouped_labels_info(xticklabels)
    yticklabels, ystep, yoffset = grouped_labels_info(yticklabels)
    #

    if _debug and (hline_level_fn is not None or vline_level_fn is not None):
        print(f'{levels_from_labels=}')

    hline_levels = None
    if hline_level_fn is not None:
        if _debug:
            print(f'{hline_level_fn=}')

        if not levels_from_labels:
            # TODO need to handle case where we might transpose (e.g. via
            # transpose_sort_key?) (or maybe delete support for transpose_sort_key?)
            hline_levels = [hline_level_fn(x) for x in util.index2dict_list(df.index)]
        else:
            hline_levels = [hline_level_fn(x) for x in yticklabels]

        if _debug:
            print('hline_levels:')
            pprint(hline_levels)

    vline_levels = None
    if vline_level_fn is not None:
        if _debug:
            print(f'{vline_level_fn=}')

        if not levels_from_labels:
            vline_levels = [vline_level_fn(x) for x in util.index2dict_list(df.columns)]
        else:
            vline_levels = [vline_level_fn(x) for x in xticklabels]

        if _debug:
            print('vline_levels:')
            pprint(vline_levels)

    # TODO TODO factor out? what all do i actually want tho...
    # TODO TODO need to combine into one fn that also adds group labels (so we have
    # access to group labels for allow_duplicate_labels=False checks)
    #
    # TODO allow specifying `x=labels` (in kwargs) instead of `x_or_y='x', labels`
    # (and same for y, if so)?
    # TODO type hint for labels? Sequence[str]?
    def set_ticklabels(ax: Axes, x_or_y: str, labels, *args,
        # TODO delete allow_duplicate_labels here if i don't end up factoring this inner
        # fn out of matshow
        allow_duplicate_labels: bool = False, **kwargs) -> None:

        assert x_or_y in ('x', 'y')

        # TODO allow_duplicate_labels='warn' option, and make that default?
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

                # TODO TODO only print something like this if doing so would actually
                # remove duplicates?
                # TODO just make [v|h]line_group_text=True the default in these cases,
                # rather than erring, no? there aren't even duplicates here are there?
                # TODO at least dont include these parts of the err msg if
                # [v|h]line_group_text are already True...
                if (x_or_y == 'x' and vline_level_fn is not None and
                    not vline_group_text):

                    err_msg += ('. specifying vline_group_text=True may resolve '
                        'duplicates.'
                    )

                elif (x_or_y == 'y' and hline_level_fn is not None and
                    not hline_group_text):

                    err_msg += ('. specifying hline_group_text=True may resolve '
                        'duplicates.'
                    )

            if len(to_check) != len(set(to_check)):

                err_msg += ' duplicated entries, with counts:\n'
                for x, count in Counter(to_check).items():
                    if count > 1:
                        err_msg += f'{repr(x)} ({count})\n'

                err_msg += 'you may also set allow_duplicate_labels=True'
                raise ValueError(err_msg)


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
            # TODO delete this guesswork and just back a default probably...
            # (yea, it's causing TypeErrors w/o clear explanation)
            if all([len(x) == 1 for x in xticklabels]):
                xtickrotation = 'horizontal'
            else:
                xtickrotation = 'vertical'

        # TODO what was the purpose of this? to ensure each is shown?
        ax.set_xticks(np.arange(0, len(df.columns), xstep) + xoffset)

        set_ticklabels(ax, 'x', xticklabels,
            allow_duplicate_labels=allow_duplicate_labels,
            fontsize=fontsize, fontweight=fontweight, rotation=xtickrotation
        )

    if yticklabels is not None:
        ax.set_yticks(np.arange(0, len(df), ystep) + yoffset)

        set_ticklabels(ax, 'y', yticklabels,
            allow_duplicate_labels=allow_duplicate_labels,
            fontsize=fontsize, fontweight=fontweight, rotation='horizontal'
        )

    if hline_levels is not None:
        add_group_labels_and_lines(ax, y=hline_levels, labels=hline_group_text,
            formatter=hgroup_formatter, label_offset=hgroup_label_offset,
            rotation=hgroup_label_rotation, linewidth=linewidth, linecolor=linecolor,
            fontsize=group_fontsize, fontweight=group_fontweight, _debug=_debug
        )

    if vline_levels is not None:
        add_group_labels_and_lines(ax, x=vline_levels, labels=vline_group_text,
            formatter=vgroup_formatter, label_offset=vgroup_label_offset,
            rotation=vgroup_label_rotation, linewidth=linewidth, linecolor=linecolor,
            fontsize=group_fontsize, fontweight=group_fontweight, _debug=_debug
        )

    # TODO precompute constrained layout here, and check that no group / xticklabel text
    # overlaps (or that group text doesn't go out of bounds?). and/or provide fn for
    # this (layout might change if stuff added later...). all this is just for lack of a
    # good way to layout text and ensure it doesn't overlap / is seen, as i understand
    # it...

    # TODO change xlabel/ylabel defaults from None->True?
    # (at least if only one level for each?)
    def _names_to_label(curr_label: Union[bool, str], index: pd.Index) -> str:
        if curr_label == True:
            return '/'.join(index.names)

        assert type(curr_label) is str
        return curr_label


    if xlabel is not None:
        assert title is None, 'currently title also uses xlabel in this fn'
        title = xlabel

    if title is not None:
        title = _names_to_label(title, df.columns)
        ax.set_xlabel(title, fontsize=bigtext_fontsize, labelpad=12)

    if ylabel is not None:
        ylabel = _names_to_label(ylabel, df.index)
        ylabel_kws = _ylabel_kwargs(
            ylabel_rotation=ylabel_rotation, ylabel_kws=ylabel_kws
        )
        # TODO allow ylabel_kws to override bigtext_fontsize (w/ 'fontsize' key)?
        ax.set_ylabel(ylabel, fontsize=bigtext_fontsize, **ylabel_kws)


    if not xticks_also_on_bottom:
        # didn't seem to do what i was expecting
        #ax.spines['bottom'].set_visible(False)
        # TODO this implicitly make them shown at top?
        ax.tick_params(bottom=False)
    else:
        # TODO difference between top and labeltop? former always required for latter?
        # https://stackoverflow.com/questions/55289921
        ax.tick_params(axis='x', bottom=True, top=True, labelbottom=True, labeltop=True)

    # TODO test w/ input that is not symmetric
    # (thought this same code caused issues in one of the al_analysis N-plotting fns.
    # seemed to work in al_analysis.plot_corr tho)
    if overlay_values:
        if overlay_kws is None:
            overlay_kws = dict()

        # https://stackoverflow.com/questions/20998083
        for (i, j), c in np.ndenumerate(df):
            # TODO color visible enough? way to put white behind?
            # or just use some color distinguishable from whole colormap?
            ax.text(j, i, f'{c:{overlay_fmt}}', ha='center', va='center', **overlay_kws)

    # TODO also return ax?
    return fig, im


# TODO TODO keep making/returning figure (none of current uses seem to. delete!)?
# (actually they all do, as none pass in ax, but also none use returned fig)
#
# TODO TODO want to keep title/cmap handling? replace cmap w/ **kwargs?
# (only hong2p/scripts/extract_template.py and hong2p/roi.py currently seem to use
# this. al_analysis dff_imshow could maybe be replaced by this (or image_grid?))
# TODO type hint img (arraylike?)
@add_norm_options
def imshow(img, ax=None, title=None, cmap=DEFAULT_ANATOMICAL_CMAP, **kwargs):
    # TODO warn if ax not set? or at least in the case where plt.gca() (/gcf?) returns
    # something? (to make sure we aren't making figures that don't get populated/closed)
    # (factor this check into wrapper shared w/ matshow?)
    # (could test by removing the ax=ax from the imshow call inside image_grid, where i
    # originally noticed issues as a result of this)
    if ax is None:
        # none of the cases i was calling this actually used returned figure
        fig, ax = plt.subplots()

    # TODO better name for output (indicating type, ideally)
    im = ax.imshow(img, cmap=cmap, **kwargs)

    if title is not None:
        ax.set_title(title)

    # TODO option to disable this?
    # NOTE: this used to be ax.axis('off'), may have broken some code expecting its
    # behavior (also hiding labels, other things?)
    remove_axes_ticks(ax)

    return im


# TODO type hint matplotlib.cm.ScalarMappable for im (as in mpl docs)?
# TODO possible to get it to work reasonably w/o explicit im passed in?
# (maybe use a generic ScalarMappable as
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html talks about
# in this case?) or behave as `plt.colorbar` does in that case (also mentioned in link)
#
# TODO TODO TODO even when im is passed in, should i check all axes have the same
# vmin/vmax as im? (or if im not passed, check they are all the same? there should be
# multiple colorbars otherwise, right? what happens currently?)
# TODO TODO TODO warn (unless flag set False) if any values would be clipped? especially
# if extend='both' not passed.
# TODO TODO or automatically set_[over|under] if needed (to show which things are
# clipped)? test it actually shows correctly in all cases (rather than pre-clipping or
# something)
# TODO TODO maybe search any AxesImages in fig and do same for all of them (if sharing a
# colorbar... can i tell that in here tho?)
def add_colorbar(fig: Figure, im, match_axes_size: bool = False, axes_index: int = -1,
    # TODO delete if not useful. trying to add support for ImageGrid cax's
    location: str = 'right', size=None, pad=None, cax: Optional[Axes] = None,
    label: Optional[str] = None, fontsize=None, **kwargs) -> Colorbar:
    # TODO TODO does size= steal that size from specified ax tho? answer change
    # depending on whether we are using constrained layout or not?
    #
    # TODO TODO TODO support matching single axes when input has one col + many rows
    # TODO TODO support putting colorbar at bottom (with text horizontal then)
    # (and also matching axes width there)
    """
    Args:
        fig: figure to add colorbar to

        im: typically output of a `ax.imshow`/similar call
            see `Figure.colorbar` docs for more details:
            https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar

        match_axes_size: if True, colorbar will be made same size (for now, just height)
            as Axes. assumed that all Axes are same height and in one row (if multiple).

        axes_index: if `match_axes_size=True`, use `fig.axes[axes_index]` to match

        location: {'left', 'right', 'bottom', 'top'}. where new colorbar axes is placed
            relative to axes it's created from. if `match_axes_size=True`, passed as 1st
            arg to `AxesDivider.append_axes`, otherwise passed as `location` kwarg to
            `fig.colorbar`.

        size: passed to `AxesDivider.append_axes` (default: '7%'), if
            `match_axes_size=True`. unused otherwise. '<x>%' means the size of the
            created colorbar axes will be x% of matched `Axes` size (width, if appending
            on right).

            see matplotlib docs and examples listed there:
            https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_divider.AxesDivider.html
            https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html

        pad: passed to `AxesDivider.append_axes` (default: '2%'), if
            `match_axes_size=True`. passed to `fig.colorbar` otherwise. spacing between
            created colorbar (vs matched) `Axes`.

        cax: passed to `fig.colorbar`. use for more control of size/shape of colorbar,
            e.g. to make colorbar same height as `Axes`, as in
            https://stackoverflow.com/a/18195921

        label: label for colorbar

        fontsize: passed to `Axes.set_ylabel`

        **kwargs: passed to `Figure.colorbar`
    """
    # TODO check whether kwargs `label=label` to fig.colorbar can replace
    # cbar.ax.set_ylabel. i think so, but poorly documented.
    # (i.e. does it handle None the same way [/ close enough])
    # (would also need to check fontsize works as fig.colorbar kwarg now tho, and not
    # sure it does...)

    # TODO does this decrease size of current axes (YES!)? if so, another solution that
    # makes new axes (to accomplish similar thing)
    # https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_inset_locator.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html
    # https://matplotlib.org/stable/users/explain/axes/colorbar_placement.html
    # last resort, the manual method: https://stackoverflow.com/questions/18195758
    #
    # bit tricky w/ constrained layout tho... all approaches might make main axes differ
    # in size between versions of plots with and without colorbars... might just need to
    # add colorbars for all, and manually remove after. maybe one or both of these
    # solutions [described in this comment or implemented below] can work as long as i
    # have excess space in axis colorbars will be added along?
    #
    # TODO delete? currently unused... some code might benefit from it tho
    if match_axes_size:
        assert cax is None
        assert 'shrink' not in kwargs or kwargs['shrink'] == 1.0

        # TODO need to handle / err in case Axes are not all taking up same vertical
        # extent? what if one is larger? what if there's more than one row?

        ax = fig.axes[axes_index]

        # TODO update doc w/ default change (or just use normal kwarg... why not?)
        size = '5%' if size is None else size

        # TODO TODO find args for placing on bottom too
        cax = inset_axes(
            ax,
            # TODO expose width/height (+ update doc for 'size', if deleting
            # axes_divider method)
            # width: % of parent_bbox width
            width=size,

             # height % of parent axes
            height='100%',

            # TODO what is this even doing? why is it on the right of the image then?
            # (it [or something] is required, as commenting it out seems to remove the
            # colorbar)
            loc='lower left',
            # also didn't work (couldn't see cbar). why?
            #loc="lower right",
            # puts flush w/ top of axes (also on RIGHT)
            #loc="upper left",

            # (0, 0, 1, 1) used if None
            # this anchor will place cax to right of ax (spaced by 0.05)
            bbox_to_anchor=(1.05, 0., 1, 1),

            bbox_transform=ax.transAxes,

            # why use borderpad vs whatever epsilon is added to bbox_to_anchor[0]?
            #
            # was not flush w/ bottom unless this was set. bbox_to_anchor[0] set l/r
            # distances, not up/down.
            borderpad=0,
        )
        # TODO fix so cbar label / ticks not cut off (was w/ one row at least, when
        # called from plot_roi_util.py, where i actually didn't want to end up using it)

    if cax is None:
        kwargs['ax'] = fig.axes

        # TODO TODO this causing problems in natmix_data/analysis.py usage? (when cax
        # defined from inset_axes) (moving into `if cax is None` conditional to see)
        if not match_axes_size:
            # TODO TODO may not want in case cax is passed in (e.g. from ImageGrid
            # output)
            kwargs['location'] = location

            if pad is not None:
                # in match_axes_size case, we already used pad above in this case, and
                # shouldn't try using it again in call below. pad=None does not work in
                # fig.colorbar
                kwargs['pad'] = pad

    cbar = fig.colorbar(im, cax=cax, **kwargs)

    assert im.norm is cbar.norm
    assert np.isclose(cbar.norm.vmin, cbar.vmin)
    assert np.isclose(cbar.norm.vmax, cbar.vmax)
    cbar_min = cbar.vmin
    cbar_max = cbar.vmax

    # TODO comment explaining why we are doing this. seems there is often not a tick
    # shown on lower end by default, but a bit confused about that b/c i was getting:
    # ipdb> cbar.get_ticks()
    # array([-0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
    # ...and if the -0.25 set manually below works, then why doesn't the -0.2 display?
    #
    # TODO TODO should i do something else to clarify scales are diff on two sides?
    # break in middle (could add in illustrator / inkscape tho)?
    # maybe https://stackoverflow.com/questions/53642861 ?
    # see also:
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    if isinstance(cbar.norm, TwoSlopeNorm):
        # this seems to change the cbar ticks, so we need to re-set them after
        #
        # i think it would just make size of each side of TwoSlopeNorm cbar seem
        # proportionate, w/o changing colors in image plotted.
        # https://github.com/matplotlib/matplotlib/issues/22197
        cbar.ax.set_yscale('linear')

        # TODO delete
        if 'ticks' not in kwargs:
            # NOTE: some of the plots mentioned below probably currently have their cbar
            # ticks hardcoded (via cbar_kws=dict(ticks=<x>), e.g. [-0.3, 2.5] set for
            # some in al_analysis.response_matrix_plots), in which case this branch
            # should not be hit (if below, outside conditional on norm type, should be)
            #
            # some plots i care about that this probably applies to:
            # - ijroi/certain[_mean|_stddev].pdf
            #   (stddev using same norm, but starts from midpoint=0)
            #   - and ijroi/by_panel/<panel>/*certain[_mean|_stddev].pdf
            #
            # - ijroi/by_panel/glomeruli_diagnostics/
            #   - each_fly/
            #     - *_certain.pdf
            #     - diag-highlight_certain.pdf
            #
            # - 2023-05-10_1_diagnostics1/ijroi/with_rois/*
            #
            # other plots with this norm that i'm not using/generating as actively:
            # - 2023-05-10_1_diagnostics1/ijroi/timeseries/*
            # - 2023-05-10_1_diagnostics1/ijroi/*_rois_on_max_triamean_dff.pdf
            #
            # 2023-05-10/1 is the fly i was using for diagnostic examples, so often only
            # generated some <fly_dir>/ijroi/* plots for that one
            #
            # main thing we are missing, is no ticks on lower end of scale
            # (after set_yscale('linear') call above)
            # TODO at least also print existing ticks if i still wanna find a better
            # solution for this?
            # TODO delete? OK w/ ticks not set, as-is?
            if _debug:
                print('add_colorbar: not setting cbar ticks for twoslopenorm')
            #

            # TODO cbar doesn't have a vcenter does it? (to use instead of 0)

            # wouldn't work w/ cmap i set up for correlation-distance viz.matshow plots
            # (from al_analysis.plot_corrs)
            # TODO keep? (no, replace w/ only adding 0 tick if this is true)
            #assert cbar.vmin < 0

            # TODO TODO only show stuff at nice even numbers
            # (or at least force formatter to not show 5 sigfigs?)
            # TODO TODO find |smallest| nearby multiple of 0.1, 0.25, 1.0, etc that is
            # under vmin/vmax? (how does mpl do it by default? can i use some of their
            # code?)
            #cbar.set_ticks([cbar.vmin, 0., cbar.vmax / 2, cbar.vmax])

            # TODO this work OK?
            #cbar.set_ticks([cbar.vmin, 0., cbar.vmax])

            # TODO TODO just prepend cbar.vmin to ticks?

            # TODO delete
            '''
            if cbar.vmin < -0.2:
                print(f'{cbar.vmin=}')
                old_ticks = cbar.get_ticks()
                print(f'{cbar.get_ticks()=}')
                import ipdb; ipdb.set_trace()
            '''
            #
        #

        # TODO TODO draw axes break or something (hline at least?) to indicate diff
        # scales on two sides of cbar?

    if 'ticks' in kwargs:
        # TODO TODO maybe also default to setting ticks to include these limits by
        # default?

        # TODO TODO can i change how ticks are formatted to have a diff number of
        # sigfigs per tick (only as many as needed for each)?

        # TODO delete. redundant w/ warnings in add_norm_options.
        # (could be used to set default ticks tho...?)
        '''
        # NOTE: this is a MaskedArray (at least, as I'm looking at it now), but not sure
        # if it will ever actually be masked (maybe some cmap's use that for things like
        # set_bad/etc?)
        # TODO some way to ensure i'm getting min/max of full data, not masked portion?
        #
        # get_array() seemed to be only way to access data from `im`. limits matched w/
        # what I expected in my first test.
        data = im.get_array()
        dmin = data.min()
        dmax = data.max()
        # these are both scalars after single min|max operation (unlike pandas where i'd
        # need 2 generally, for 2d input)
        assert dmin.shape == tuple() and dmax.shape == tuple()
        '''
        #

        ticks = kwargs['ticks']
        # TODO delete. ticks can be off cbar and that's fine. (still in shared kwargs
        # for some plots that set vmin to -1e-6, and seems to not cause bad output)
        #assert cbar_min <= min(ticks), f'{cbar_min} > {min(ticks)=}'
        #assert cbar_max >= max(ticks), f'{cbar_max} > {max(ticks)=}'

        # NOTE: important that this happens after possible set_yscale('linear') call
        # above, as that seems to reset the cbar ticks
        cbar.set_ticks(ticks)

    if label is not None:
        # TODO test fontsize=None doesn't change default behavior
        cbar.ax.set_ylabel(label, fontsize=fontsize)

    return cbar


def _check_contour_array(contour):
    # just takes normal array input (not the QuadContourSet directly returned by
    # matplotlib's contour fns)
    shape = contour.shape
    assert len(shape) == 2
    assert shape[0] > 1
    # x,y
    assert shape[-1] == 2


def contour_bbox(contour):
    _check_contour_array(contour)

    x = contour[:, 0]
    y = contour[:, 1]
    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)
    # TODO maybe return as NamedTuple or something instead?
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def contour_center_of_mass(contour):
    _check_contour_array(contour)

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


# TODO change default _pad=False? plot_rois and al_analysis both effectively change this
# default to False, and i'm not sure i'm using this code from anywhere else now...
# TODO delete label_to_bbox_dy? seems ok w/ just verticalalignment='bottom' w/ dy=0
def plot_closed_contours(footprint, if_multiple: str = 'err', _pad=True,
    ax: Optional[Axes] = None, label: Optional[str] = None,
    label_over_contour: bool = False, label_to_bbox_dy: int = 0,
    text_kws: Optional[dict] = None, colors=None, color=None, linewidths=1.2,
    linestyles='dotted', label_outline: bool = False,

    label_outline_linewidth=0.75, label_outline_color='black',
    #label_outline_linewidth=0.9, label_outline_color='white',

    smooth=False, **kwargs):
    # TODO doc / delete
    # TODO doc footprint type, to start
    # TODO specify if label_to_bbox_dy is in data/axes coords (former, i think?)
    """Plots line around the contiguous positive region(s) in footprint.

    Args:
        ax: Axes to plot onto. will use current Axes otherwise.

        if_multiple: 'take_largest'|'join'|'err'|'ignore'. what to do if there are
            multiple closed contours within footprint. contour will be plotted
            regardless, but error will happen before a contour is returned for use in
            other analysis.

        label: name of ROI. drawn in center of ROI contour by default.

        label_over_contour: if True, will draw `label` above ROI contour (rather than on
            center of mass). see also `label_to_bbox_dy`.

        label_to_bbox_dy: how many pixels above ROI bounding box to draw ROI label.
            ignored if `label_over_contour=False`.

        label_outline: if True, adds a `PathEffects` to `text_kws` to add an outline
            around drawn label.

        label_outline_color: color for outline from `label_outline=True`

        label_outline_linewidth: linewidth for outline from `label_outline=True`

        **kwargs: passed through to matplotlib `ax.contour` call
    """
    if ax is None:
        ax = plt.gca()

    # So we don't need to define default in multiple places.
    if colors is None:
        if color is not None:
            colors = color
        else:
            colors = 'red'

    if smooth:
        raise NotImplementedError

        # TODO delete
        # TODO TODO TODO is this not triggering?
        '''
        if label == 'DM3':
            plt.close('all')
            plt.figure()
            plt.imshow(footprint)
        '''
        #

        # does not work with my data (interior of each ROI gets filled with jagged
        # things), and image is in top left corner (still at original size, w/ ax and
        # contours scaled by factor here)
        #
        # TODO TODO this even work w/ binary data i'm using? alternative?
        #
        # https://stackoverflow.com/questions/12274529
        # "Resample your data grid by a factor of 3 using cubic spline interpolation."
        # TODO TODO probably need to rescale coordinates / other things by same
        # factor (e.g. for text placement)
        #footprint = scipy.ndimage.zoom(footprint, 3)

        # TODO delete
        '''
        if label == 'DM3':
            plt.figure()
            plt.imshow(footprint)
            plt.show()
            # TODO TODO TODO does zoom keep size same?
            # need to fix if not...
            import ipdb; ipdb.set_trace()
        '''
        #

    # TODO TODO fix what seems to be a (1, 1) pixel offset of contour wrt footprint
    # passed in (when plotted on same axes). (still?)
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

            # TODO also try line font props similar to solo stuff

            #extent=(0, dims[0] - 1, 0, dims[1] - 1),
            **kwargs
        )
        #mpl_contour_nopad = ax.contour(footprint > 0, [0.5], colors=colors,
        #    linewidths=linewidths, linestyles=linestyles, **kwargs
        #)
    else:
        # Checking for what I believe was the reason we needed padding, though it might
        positive = footprint > 0
        # TODO TODO better error message (and print label if available)
        # (this will be triggered by stuff touching the edge, even at a single point)
        # (maybe i should just always be padding tho? i.e. above branch?)
        # TODO TODO and catch in plot_rois and print plane / other info if there
        # (so people can actually fix the ROI that has an issue)
        assert not any([
            np.array(positive[0, :]).any(),
            np.array(positive[-1, :]).any(),
            np.array(positive[:, 0]).any(),
            np.array(positive[:, -1]).any(),
        ])
        mpl_contour = ax.contour(positive, [0.5], colors=colors,
            linewidths=linewidths, linestyles=linestyles,
            **kwargs
        )

    # TODO really want conditional on linestyles?
    # TODO just flag / path_effects kwarg to disable for some calls?
    # (the 'dotted' one was the whole AL plane outline)
    if linestyles != 'dotted':
        # TODO delete
        # TODO define default color dynamically based on cmap (like elsewhere)
        #'''
        # TODO TODO was ~one call not using black + a diff lw? maybe thru defaults?
        if _debug and label_outline_color != 'w':
            print(f'{label_outline_linewidth=} {label_outline_color=}')

        #import ipdb; ipdb.set_trace()
        #'''

        # TODO possible to make it go all the way around the dash (not just the 2 long
        # edges of the line)
        path_effects = [
            PathEffects.withStroke(linewidth=label_outline_linewidth,
                foreground=label_outline_color
            )
        ]

        # TODO delete
        #print(f'{type(mpl_contour)=}')

        # TODO TODO why do docs make it seem like .set should be availaible directly on what
        # ax.contour returns? version thing?
        #mpl_contour.set(path_effects=path_effects)
        # https://matplotlib.org/stable/gallery/misc/tickedstroke_demo.html
        for c in mpl_contour.collections:
            c.set(path_effects=path_effects)

    # TODO delete
    # (assuming i can set dash params via linestyles=[offset, (dash on, dash off)]`
    '''
    # https://stackoverflow.com/questions/12434426
    for c in mpl_contour.collections:
        c.set_dashes([(0, (2.0, 2.0))])
    import ipdb; ipdb.set_trace()
    '''

    if label is not None:
        if text_kws is None:
            text_kws = dict()
        else:
            text_kws = dict(text_kws)

        color = colors
        if not isinstance(colors, str):
            if len(colors) == 1:
                color = colors[0]

        assert len(mpl_contour.allsegs) == 1
        # TODO warn? might need to use a particular segment / combination in other cases
        #assert len(mpl_contour.allsegs[-1]) == 1
        # TODO what does it mean if this is >1? haven't checked this in a while, but in
        # first check in a while, it was 1
        assert len(mpl_contour.allsegs[-1]) >= 1

        # (n, 2) array w/ x,y data for contour
        contour_array = mpl_contour.allsegs[-1][0]

        default_text_kws = {
            'color': color,

            'horizontalalignment': 'center',

            'fontweight': 'bold',
            # Default should be 10.
            'fontsize': 8,
            # TODO move plot_rois PathEffects in here (to simplify text_kws passthru
            # from plot_rois, maybe as a new roi_text_kws arg to plot_rois)?
        }

        if not label_over_contour:
            # Also partially taken from https://stackoverflow.com/questions/48168880
            text_x, text_y = contour_center_of_mass(contour_array)

        else:
            # TODO if default va='top', then how come center of mass approach worked ok
            # so far in !label_over_contour case (without being offset along Y axis)?
            #
            # hopefully should make parameter for space between label and top of contour
            # bbox simpler (default va='top') (seems to)
            default_text_kws['verticalalignment'] = 'bottom'

            # TODO some way to get this bbox easily from matplotlib stuff itself?
            # couldn't find it easily myself...
            bbox = contour_bbox(contour_array)

            xmin, xmax, ymin, ymax = (
                bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
            )
            text_x = xmin + (xmax - xmin) / 2

            # TODO subtract an amount in relation to fontsize? fixed kwarg alongside one
            # controlling whether to draw roi above contour?
            text_y = ymin - label_to_bbox_dy

        for k, v in default_text_kws.items():
            if k not in text_kws:
                text_kws[k] = v

        if label_outline:
            assert 'path_effects' not in text_kws
            text_kws['path_effects'] = [
                # (for huge ~20" x ~3" plot_roi figs initially made)
                # TODO update values for reasonable (e.g. fittable in page size) figs
                # -> delete most of comment below
                #
                # 0.2 was too small linewidth, at least with other settings as
                # they are. 1.0 was possibly too big. might need to change other
                # settings if I can't find a good middle ground otherwise.
                # 1.0 probably still looks better than 0.5
                #
                # https://osxastrotricks.wordpress.com/2014/12/02/add-border-around-text-with-matplotlib/
                # see also:
                # https://matplotlib.org/stable/users/explain/artists/patheffects_guide.html
                PathEffects.withStroke(linewidth=label_outline_linewidth,
                    foreground=label_outline_color
                )
            ]

        ax.text(text_x, text_y, label, **text_kws)

        # TODO delete
        '''
        abox = ax.bbox
        print(f'{[abox.xmin, abox.xmax, abox.ymin, abox.ymax]=}')

        pc = mpl_contour.collections[0]

        cbox = pc.get_clip_box()
        #[9.275000000000425, 208.35238095238137, 285.3263095238095, 484.4036904761905]
        print(f'{[cbox.xmin, cbox.xmax, cbox.ymin, cbox.ymax]=}')

        #(9.275000000000425, 285.3263095238095, 199.07738095238093, 199.07738095238096)
        # TODO how does this differ from above? why not just re-ordered?
        print(f'{cbox.bounds=}')

        tbox = pc.get_tightbbox()
        #Bbox([[9.275000000000425, 285.3263095238095], [208.35238095238137, 484.4036904761905]])
        print(f'{tbox=}')

        plt.show()
        import ipdb; ipdb.set_trace()
        '''
        #

    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1

    paths = mpl_contour.collections[0].get_paths()
    assert len(paths) > 0

    if len(paths) != 1:
        # NOTE: this will be after drawing contour, but before drawing any label...
        if if_multiple == 'err':
            raise RuntimeError('multiple disconnected paths in one footprint')

        # TODO TODO still try plotting each in a separate color / printing vertices in
        # each, to try to get a sense of where these are coming from / how to fix.
        # main issue currently is just that name seems to get drawn off center for these
        # ROIs, despite overall ROI shape seeming to match up w/ what I have in ImageJ
        # in all examples checked so far.
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


# TODO option to burn in D/V M/L A/P axis labels (or another fn to handle that?)?
# TODO add kwarg flag to include colorbar (always added now?) (delete comment?)
# TODO rename scale_per_plane to scale_per_image?
# TODO type hint image_list as list/ndarray (of ndarrays, in either case)?
# TODO type hint return
def image_grid(image_list, *, nrows: Optional[int] = None, ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None, width: Optional[float] = None,
    height: Optional[float] = None, extra_figsize: Optional[Tuple[float, float]] = None,
    inches_per_pixel=0.014, imagegrid_rect=None, dpi=None,
    scale_per_plane: bool = False, minmax_clip_frac: float = 0.0, vmin=None, vmax=None,
    cbar: bool = True, cmap=DEFAULT_ANATOMICAL_CMAP, cbar_label: Optional[str] = None,
    cbar_kws: Optional[KwargDict] = None, **kwargs):
    # TODO also allow specifying either height_inches/width_inches instead of
    # inches_per_pixel (would only save specifying one component of figsize...)?
    """
    Args:
        image_list: list/array of images, whose length is equal to the number of images
            (i.e. number of planes, for typical input of single-fly volumetric data,
            where input has been reduced across timepoints)

        ncols: how many columns for grid showing the background of each plane
            (one panel per plane)

        figsize: passed to `plt.subplots`. if passed, none of `width`, `height`,
            `extra_figsize` should be, and `inches_per_pixel` will be ignored.
            (width, height)

        width: figure width in inches. only pass either this or `height`. if you want to
            specify both, use `figsize` instead. this is to specify one and have the
            other calculated (from `nrows`, `ncols`, and `extra_figsize`).
            will ignore and recalulate `inches_per_pixel` to fit figure in this width.

        height: figure height in inches. see `width` documentation.

        extra_figsize: if calculating `figsize`, this can be used to add space for other
            elements besides the images (colorbars, titles, etc). (width, height)

        inches_per_pixel: used to calculate `figsize`, if `figsize` not passed

        imagegrid_rect: passed to `ImageGrid`, to determine size / position it will
            occupy in figure. defaults to something automatically determined.
            colorbar(s) and any labels/titles should (i think) extent past this.

            (left, bottom, width, height) tuple, where (0, 0) is bottom left, and
            everything should be in [0, 1]. width and height are fractions of figure
            sizes.

        scale_per_plane: if False, min/max of colorscale will be picked from min/max of
            all data (after excluding `minmax_clip_frac`, if it's greater than 0).

        minmax_clip_frac: clip this fraction of data from BOTH the lower/upper end, when
            deciding the limits of the colormap for the background images.
            `v[min|max]` must not be set if this is, and vice versa.

        cbar: set to `False` to disable colorbar(s)

        cbar_label: label for colorbar

        cbar_kws: passed to `add_colorbar`

        **kwargs: passed to `imshow`
    """
    if figsize is not None:
        assert all(x is None for x in (width, height, extra_figsize))

    elif width is not None:
        assert height is None

    # TODO even want to support scale_per_plane=True? delete all related branches?
    # TODO want to check vmin/vmax not passed if scale_per_plane=True?
    # TODO make a decorator to convert assertion errors like these to ValueErrors?
    # (and maybe also have it modify the docstring to add something like "raises
    # ValueError"?)
    # TODO TODO and replace `0 <=` w/ `0 <`?
    assert 0 <= minmax_clip_frac < 0.5
    if vmin is not None or vmax is not None:
        assert vmin is not None and vmax is not None

        # TODO TODO also need to not conflict with any norms i might want to use
        #
        # (assuming if it's 0, it wasn't explicitly passed in. good enough for now.
        # this shouldn't be passed if vmin/vmax are, and vice versa)
        # TODO TODO replace w/ this being None (and replace `0 <=` w/ `0 <` above)?
        assert minmax_clip_frac == 0
        assert not scale_per_plane

    if cbar_kws is None:
        cbar_kws = dict()

    # TODO just disallow this, telling them to use cbar_label instead
    elif 'label' in cbar_kws:
        # (since i change cbar_kws by popping below)
        cbar_kws = dict(cbar_kws)

        assert cbar_label is None
        cbar_label = cbar_kws.pop('label')

    image_list = np.array(image_list)

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
        extra_width = 0
        extra_height = 0
        if extra_figsize is not None:
            extra_width, extra_height = extra_figsize

        # Assuming all images in image_list are the same shape
        image_shape = image_list[0].shape
        assert len(image_shape) == 2
        # TODO actually test w/ images where w != h. i might have them flipped.
        w = image_shape[1]
        h = image_shape[0]

        if width is not None:
            # NOTE: ignoring any input inches_per_pixel
            inches_per_pixel = (width - extra_width) / (w * ncols)

        elif height is not None:
            # NOTE: ignoring any input inches_per_pixel
            inches_per_pixel = (height - extra_height) / (h * nrows)

        # TODO try to account for (minor) small spaces between images?
        # 'compressed' does minimize well enough (assuming just images in layout)
        figsize = (
            inches_per_pixel * (w * ncols) + extra_width,
            inches_per_pixel * (h * nrows) + extra_height
        )
        # TODO delete? /uncomment after resolving other things i'm using _debug for
        # (+ removing `viz._debug = True` in my al_analysis.py)
        #if _debug:
        #    print(f'{inches_per_pixel=}')
        #    print(f'{(inches_per_pixel * (w * ncols))=}')
        #    print(f'{(inches_per_pixel * (h * nrows))=}')
        #    print(f'{(inches_per_pixel * (w * ncols) + extra_width)=}')
        #    print(f'{(inches_per_pixel * (h * nrows) + extra_height)=}')
        #    print(f'{figsize=}')

        if width is not None:
            assert np.isclose(figsize[0], width)

        elif height is not None:
            assert np.isclose(figsize[1], height)

    # NOTE: 'compressed' layout ended up being key to getting decent (e.g. minimal)
    # spacing between the images if anything (suptitle, colorbar) was added to fig after
    # this call. from matplotlib docs:
    # "'compressed': uses the same algorithm as 'constrained', but removes extra space
    # between fixed-aspect-ratio Axes. Best for simple grids of axes."
    #fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi,
    #    layout='compressed'
    #)

    # TODO TODO fix: (handle some other way for 3.4.3 [or anything <3.7.3]? or just
    # require >=3.7.3 and fix the seaborn negative-input-to-errorbar issue which is why
    # i downgraded to 3.4.3? or is there a version between that works w/ both this and
    # the errorbar code?
    # (currently decided to revert to 3.7.3 because of this)
    # ...
    #   File "/home/tom/src/hong2p/hong2p/viz.py", line 2256, in image_grid
    #     fig = plt.figure(figsize=figsize, dpi=dpi, layout='compressed')
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/pyplot.py", line 797, in figure
    #     manager = new_figure_manager(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/pyplot.py", line 316, in new_figure_manager
    #     return _backend_mod.new_figure_manager(*args, **kwargs)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 3544, in new_figure_manager
    #     fig = fig_cls(*args, **kwargs)
    # TypeError: __init__() got an unexpected keyword argument 'layout'
    #
    # TODO still use 'compressed'? maybe regular constrained?
    fig = plt.figure(figsize=figsize, dpi=dpi, layout='compressed')

    # TODO TODO still have cbar=False disable these cbar (maybe still make and remove so
    # size changes the same??? so i can edit together plots w/ and w/o cbars more
    # easily)
    #
    # (gonna just set cbar_mode=None for now and leave my old cbar code to handle
    # it... sike)
    #cbar_mode=None
    #
    # TODO still need to call colorbar myself, it just makes colorbar axes, and by
    # default associates them w/ corresponding axes:
    # https://stackoverflow.com/questions/37917219
    #
    # makes one giant one (same height as all axes together)
    #cbar_mode='single'

    # at least w/ other settings are they are now (making <fly>/ijroi/with_rois plots to
    # show ROIs and diag responses, where height=10.4 and nrows=1), size of image axes
    # are equal whether showing cbar or not (so no need to add -> remove)
    if cbar:
        cbar_mode = 'each'
    else:
        cbar_mode = None

    # TODO experiment w/ cbar options here. may not need (as much of) my own cbar
    # handling (also have cbar_location, cbar_pad, cbar_size)

    # TODO TODO why do i seem to be getting:
    # Warning: There are no gridspecs with layoutgrids. Possibly did not call parent
    # GridSpec with the "figure" keyword
    # ...w/ new ImageGrid code (warning seems to come after this call, maybe in
    # plot_rois? or in savefig?)

    # TODO try using inches_per_pixel again, to calc rect (2nd arg here)
    # TODO or how else do i want to specify size here (or margin wrt figsize?)
    # treat all extra_figsize as the margin?
    #
    # TODO check + include in comment whether this area is ONLY for the images, or
    # whether also for cbars (must just be images, no?)
    if imagegrid_rect is None:
        # NOTE: can output these values and tweak from there
        # divider = grid.get_divider()
        # divider.get_position()
        rect = 111
    else:
        # TODO ideally come up w/ a general way of calculating something roughly like
        # what i'm passing in from al_analysis diag_example_plot_roi_kws
        # was never TOO bad with just rect=111 tho...
        rect = imagegrid_rect

    # TODO if i keep, use option to not make axes beyond what we actually have images
    # for (if actually in a grid, not relevant for single row/col stuff)
    grid = ImageGrid(fig, rect, nrows_ncols=(nrows, ncols), axes_pad=0.02,
        cbar_mode=cbar_mode
    )

    # TODO this work same as old subplots axs.flat? test in cases w/ more than a single
    # row/col used.
    # TODO change other code to not expect array? just converting to not have to modify
    # the axs.flat stuff for now
    axs = np.array(grid.axes_all)

    # TODO see: https://stackoverflow.com/questions/42850225 or related to find a
    # good solution for reliably eliminating all unwanted whitespace between subplots
    # (honestly i think it will ultimately come down to figsize, potentially more so
    # when using constrained layout, as i'd now generally like to)

    # TODO TODO TODO either make it so minmax_clip_frac=0 doesn't set vmin/vmax, or
    # allow None kwarg there (or handle vmin/vmax w/ norms i'd wanna use)

    single_colorbar = True
    if vmin is None and vmax is None:
        # If we aren't scaling per plane, then we are scaling ACROSS ALL planes
        # (by default, just using the min/max across all)
        if not scale_per_plane:
            # (across all axes by default, as if flattening array first)
            vmin = np.quantile(image_list, minmax_clip_frac)
            vmax = np.quantile(image_list, 1 - minmax_clip_frac)
            # TODO want to do any checking we are throwing away (only) a reasonable
            # amount of data? like not all of it at least?

            # TODO TODO also need to not conflict with any norms i might want to
            # use (here and anywhere else we compute vmin/vmax, which can't be passed
            # with anything other than str norm, which seems to preclude
            # centered/two-slope norms)
            # TODO delete
            #print(f'from minmax_clip_frac: {vmin=} {vmax=}')
            #
        else:
            single_colorbar = False

    for ax, img in zip(axs.flat, image_list):
        if scale_per_plane:
            # TODO TODO also test this branch works w/ new add_norm_options stuff
            # (including if norm=CenteredNorm/similar)
            # TODO delete
            #print(f'(curr plane) {vmin=} {vmax=}')
            #
            vmin = np.quantile(img, minmax_clip_frac)
            vmax = np.quantile(img, 1 - minmax_clip_frac)
            # TODO TODO also don't fuck up non-str norms (test?)

        im = imshow(img, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if scale_per_plane:
        del vmin, vmax

    # TODO add support for putting colorbars below images (try just one, either first /
    # last axes, and also try one per image [for use in plots where all odors at the
    # same depth have a fixed scale])
    # TODO and should it work in scale_per_plane=True case then, just drawing one under
    # each?
    if cbar and single_colorbar:
        # TODO delete
        # from before i started letting ImageGrid create and position cbar axes
        #add_colorbar(fig, im, label=cbar_label, **cbar_kws)
        #add_colorbar(fig, im, label=cbar_label, match_axes_size=True, **cbar_kws)

        single_cax = axs[-1].cax

        # TODO TODO try to keep, but only set if values actually go out of range?
        # TODO delete
        #print('viz.image_grid: delete extend=both (on add_colorbar call)?')
        #
        add_colorbar(fig, im, cax=single_cax, label=cbar_label,
            # TODO may need cmap.set_[under|over]

            # TODO comment explaining purpose of this!
            # TODO move to al_analysis, and only in diverging cmap kwargs?
            # (or something more specific?) don't really care for anatomical version
            #
            # TODO delete. testing cmap/vmin/vmax/norm/clipping handling.
            extend='both',
            #

            **cbar_kws
        )

        # TODO check all image axes data wrt limits of cbar (+clip status?) -> warn
        # about clipping? may make more sense here than ~add_colorbar, which may not see
        # the rest of the data

        n_matching_cax = 0
        for cax in grid.cbar_axes:
            if cax is single_cax:
                n_matching_cax += 1
                # seems to be useful for telling if cbar has been drawn on.
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.has_data.html
                assert cax.has_data()
            else:
                assert not cax.has_data()
                fig.delaxes(cax)

        assert n_matching_cax == 1

    # TODO if i keep using ImageGrid, could use it's kwargs to just not make these last
    # (unused) ones
    for ax in axs.flat[len(image_list):]:
        fig.delaxes(ax)

    return fig, axs


def micrometer_depth_title(ax: Axes, zstep_um: float, z_index: int, *,
    as_ylabel: bool = False, as_overlay: bool = False, fontsize=8, **kwargs) -> None:
    # TODO cleanup formatting of doc re: as_ylabel/as_overlay changes to which fn gets
    # args passed thru to it
    """
    Args:
        as_ylabel: if True, use `ax.set_ylabel` instead of `ax.title`, to put this
            information on the left (may want to use if have a series of single column
            plots, rather than a series of rows). adds `rotation=0` and `labelpad=16`
            by default. mutually exclusive with `as_overlay`.

        as_overlay: draws on `ax` instead of labelling above / to the side. in top left
            corner by default. mutually exclusive with `as_ylabel`.

        fontsize: passed to `ax.set_title` (or `ax.set_ylabel`, if `as_ylabel=True`;
            `ax.text` if `as_overlay=True`)

        **kwargs: passed to `ax.set_title` (or `ax.set_ylabel`, if `as_ylabel=True`)
    """
    curr_z = -zstep_um * z_index

    if not (as_ylabel or as_overlay):
        set_title = ax.set_title
        extra_default_kws = dict()

    elif as_ylabel:
        set_title = ax.set_ylabel
        # w/o labelpad text overlaps w/ images. labelpad=20 worked w/ fontsize=8.
        # labelpad=16 works w/ fontsize=7 (and max title length from "-70 um")
        extra_default_kws = dict(rotation=0, labelpad=16)

    elif as_overlay:
        # color='w' works w/ black background (e.g. w/ cmap='gray'), but would prefer
        # color='k' (black) w/ white background (e.g. w/ cmap='vlag', or other diverging
        # where center is white). plot_rois will change default according to input image
        # + cmap.
        extra_default_kws = dict(x=0.02, y=0.92, color='w')

        # TODO TODO TODO make scalebar font consistent with this
        # (one way or the other. preferably by changing scalebar stuff to match this)
        def set_title(text, x=None, y=None, **kwargs):
            # these should be specified either by input kwargs or by extra_default_kws
            assert x is not None and y is not None
            ax.text(x, y, text, transform=ax.transAxes, **kwargs)

    # TODO replace w/ `kwargs = {**extra_default_kws, **kwargs}`?
    for k, v in extra_default_kws.items():
        if k not in kwargs:
            kwargs[k] = v

    # TODO check formatting of 0 looks good (or disable that here? currently some
    # code excludes this call on that plane, so not sure of formatting)
    # (it currently looks like "-0uM")
    set_title(f'{curr_z:.0f} $\\mu$m', fontsize=fontsize, **kwargs)


# TODO use is_cmap_diverging instead? is what i want here actually different?
# TODO also use in micrometer_depth_title
def get_default_overlay_color(ax: Axes) -> str:
    """Returns 'w' or 'k' (white/black), depending on current image colors.

    Aims to provide the color that would be higher contrast with respect to the
    background.
    """
    # NOTE: requires that imshow (/equiv) has been called already
    images = ax.get_images()
    if len(images) == 0:
        # TODO catch in add_scalebar -> provide better error message including that we
        # can just pass in color explicitly
        raise RuntimeError('has imshow() [/ similar] been called on this Axes yet?')

    assert len(images) == 1
    im = images[0]
    # TODO just get directly from background?
    arr = im.get_array()

    # TODO delete
    # for 192x192 frames, this seems to be about the section where depth title +
    # scalebar are currently drawn.
    #lowish_img_value = arr[:40, :].mean()

    # get warning "'partition' will ignore the 'mask' of the MaskedArray." if we don't
    # convert from MaskedArray w/ np.array(arr) first
    lowish_img_value = np.percentile(np.array(arr).flatten(), 15)

    # should be subclass of Normalize if not str
    assert type(im.norm) is not str
    # should map image value to [0.0, 1.0] (which should be input range of cmap)
    cmap_input = im.norm(lowish_img_value)
    lowish_value_color = im.cmap(cmap_input)

    # excluding alpha channel (which should be 1.0 anyway)
    luminance = np.mean(lowish_value_color[:3])

    if luminance > 0.5:
        # black
        default_color_over_image = 'k'
    else:
        default_color_over_image = 'w'

    return default_color_over_image


def add_scalebar(ax: Axes, pixelsize_um: float, *, color=None, **kwargs) -> None:
    # TODO move default_kwargs to actual default kwarg values in fn def? why not?
    #
    # I did check (in 2023-05-10/1, where FOV width/height are 100.[6?] uM,
    # that w/ fixed_value=99 [instead of length_fraction], the bar is just
    # slightly shorter than width of image)
    default_kwargs = dict(
        frameon=False,
        units='um',
        # TODO restore? or just keep at hardcoded 10um (via fixed_value=10)?
        #length_fraction=0.25
        fixed_value=10,
        # OK for contexts of plot_rois? (got this value from 10uM ScaleBar's made in
        # al_analysis stuff not using plot_rois...)
        width_fraction=0.025,
        # NOTE: tried font_properties=dict(family='DejaVu Sans'), but didn't seem to
        # get same italic font as in titles (not sure if font changed at all?)
        # (can also set font size via size= in same font_properties dict)
    )
    kwargs = {**default_kwargs, **kwargs}

    # color is for "the scale bar, scale and label"
    if color is None:
        color = get_default_overlay_color(ax)

    # TODO make font (see lowercase mu character) consistent with ax.text called
    # from micrometer_depth_titles
    # (one way or the other. preferably by changing this)
    #
    # for other ScaleBar options, see:
    # https://github.com/ppinard/matplotlib-scalebar
    sb = ScaleBar(pixelsize_um, color=color, **kwargs)
    ax.add_artist(sb)


# TODO TODO option to only show name for focus_roi (like betty wanted)
# (and maybe offset so it's above actual response. option to configure this?)
#
# TODO also try just coloring focus_roi and not showing name for anything
# (tried it. OK, but now kinda wanting thicker lines on focus one?)
#
# TODO unit test (that at least it produces a figure without failing for
# correctly-formatted DataArray input)
# TODO support similarly-indexed DataArray for background (+ maybe remove ndarray code)
# TODO should background be optional (probably...)?
# TODO refactor scale_per_plane/minmax_clip_frac handling to just use defaults from
# image_grid?
# TODO see fig 4 in https://www.sciencedirect.com/science/article/pii/S096098222301285X
# for some ideas (about colorscale / use of color / other aesthetic choices)
# TODO TODO option for plotting dotted outline around whole AL (per plane, perhaps
# specified as sam's plane<n> ROIs) (like in paper above, tho seen before)
# TODO TODO also include A/P L/M axis arrows in plot (or option to)
# TODO maybe include (D<->V in/near titles that have depths?)
# TODO TODO TODO move depth into top left of each image
# TODO type i can use to type hint colors (what does seaborn do?). use for
# color/focus_roi_color/palette/etc.
# TODO delete seed? only used before shuffling, and not sure i really want to shuffle at
# all...
def plot_rois(rois: xr.DataArray, background: np.ndarray, *,
    certain_only: bool = False, best_planes_only: bool = False, show_names=True,
    color=None, palette=None, seed=0, focus_roi=None, focus_roi_color='red',
    focus_roi_linewidth: Optional[float] = None, palette_desat: Optional[float] = 0.4,
    scalebar: bool = False, cbar=True, cbar_label=None, cbar_kws=None,
    pixelsize_um: Optional[float] = None, scalebar_kws: Optional[dict] = None,
    image_kws: Optional[KwargDict] = None, zstep_um: Optional[float] = None,
    depth_text_kws: Optional[dict] = None, title: Optional[str] = None,
    title_kws: Optional[dict] = None, linewidth=1.2, _pad: bool = False,
    zero_outside_plane_outlines: bool = False, plane_outline_line_kws=None, **kwargs
    ) -> Figure:
    # TODO doc whether palette can be a dict (str -> color) (yea, right?) anything else?
    # TODO use color= kwarg in making 'ijroi/with_focusroi' variant (/similar) in
    # al_analysis
    #
    # TODO add kwargs to position scalebar? kws param for scalebar?
    # just kws + something to pick between first/last axes to draw it on?
    # TODO rename 'depth_text_kws' (+ maybe also 'micrometer_depth_title') to
    # 'zlabel_kws' or something?
    #
    # TODO rename label_outline* kws in plot_closed_contours to name_outline* instead,
    # and then just handle here via kwargs (ideally also rename `label` there to `name`,
    # but may need to refactor some other stuff if so)
    # (handling via kwargs now anyway, but i don't love the 'label' vs 'name' in param
    # names)
    # TODO refactor default plane outline kwargs, so defaults can also be mentioned in
    # plane_outline_line_kws doc here
    """
    Args:
        rois: with dims ('roi', 'z', 'y', 'x'). coords must have at least
            ('roi_z', 'roi_name') on the 'roi' dimension.

        background: must have shape equal to the (<z>, <y>, <x>) lengths of the
            corresponding entries in `rois.sizes`

        certain_only: whether to only show ROIs whose names indicate certainty about
            their ID (see `roi.is_ijroi_certain`)

        best_planes_only: whether to only show best plane for each volumetric ROI.
            If True, requires input also has 'is_best_plane' on 'roi' dimension.

        show_names: whether to plot ROI names in the center of each ROI.
            `True`|`False`|`focus`.

        color: for coloring ROI labels/outlines. use either this or `palette`.
            this is never desaturated, like `palette` can be via `palette_desat`.

        palette: for coloring ROI labels/outlines. use either this or `color`.

        focus_roi_color: color to use when the ROI name matching `focus_roi` (if
            passed). set to `None` to color these ROIs no differently from others,
            where they will then also be desaturated by `palette_desat` (if set).

        palette_desat: desaturate each color in passed / generated `palette` color by
            this amount, except for any ROIs who name is that of `focus_roi` (if passed.
            see also `focus_roi_color` docs). if `focus_roi` not passed, still
            desaturate everything by this amount, for consistency. will not desaturate
            if color is specified via `color` kwarg instead of `palette`.

        scalebar: if True, will draw scalebar on first plane
            (must also pass `pixelsize_um`)

        pixelsize_um: for use drawing scalebar

        scalebar_kws: passed to `matplotlib_scalebar.scalebar.ScaleBar`

        zstep_um: if passed, axes titles will be added to indicate depth of each plane.
            no depth titles will be drawn if this is `None`.

        depth_text_kws: passed to `micrometer_depth_title` (if `zstep_um` is provided).

        title: will set as `suptitle` if passed

        title_kws: passed to `suptitle`. ignored if `title` not passed.

        image_kws: passed to `image_grid`, for plotting background(s)

        linewidth: linewidth for ROI outline

        zero_outside_plane_outlines: if True, and will zero the background outside ROIs
            matching `roi.is_ijroi_plane_outline` (assumed there is at most one per
            plane).

        plane_outline_line_kws: passed to `plot_closed_contours` for each ROI matching
            `roi.is_ijroi_plane_outline`. defaults specified internally (different from
            for regular ROIs).

        **kwargs: passed thru to `plot_closed_contours`
    """
    is_plane_outline = np.array(
        [hong_roi.is_ijroi_plane_outline(x) for x in rois.roi_name]
    )
    plane_outlines = rois.sel(roi=is_plane_outline)
    rois = rois.sel(roi= ~ is_plane_outline)

    if certain_only:
        # TODO replace w/ select_certain_rois (after adapting to work w/ DataArray
        # input)
        certain_rois = [hong_roi.is_ijroi_certain(x) for x in rois.roi_name.values]
        rois = rois.sel(roi=certain_rois)
        #

    # TODO also add options to just plot these diff (maybe brighter color / thicker
    # lines / only one with name shown / solid [/diff style] line?)
    if best_planes_only:
        # TODO catch error if is_best_plane not there -> better error message
        # (about it being required if best_planes_only=True)
        #
        # didn't seem to matter whether I added .values to the end of sel arg
        rois = rois.sel(roi=rois.is_best_plane)

    # TODO check rois and background have compatible shape?
    # leave to plot_closed_contours? it does some of that, right?

    # TODO warn/err if rois are empty here (after subsetting above)?
    # TODO warn/err (probably just warn) if focus_roi is not None and not in
    # rois.roi_name.values
    # TODO warn (def can't err) if focus_roi is passed and not in input ROIs?

    assert show_names in (True, False, 'focus')

    # this is what plot_closed_contours takes, and may have used it in some inputs, but
    # those cases should now use color= instead
    assert 'colors' not in kwargs

    if palette is None and color is None:
        # I think I generally like the way this looks better if it's a bit desaturated
        #
        # NOTE: intentionally making a hls palette w/ 10 colors before passing this to
        # another color_palette call below. hls doesn't wrap, but I liked it OK wrapped
        # at 10. some other cmaps might automatically wrap at 6-10 colors.
        palette = sns.color_palette('hls', n_colors=10)

    # (already have colors=[<color>], but color=<color> would be simpler...)
    roi_name2color = None
    if palette is not None:
        roi_names = rois.roi_name.values

        if isinstance(palette, Mapping):
            # TODO if we subset rois above, need to change this? not actually using this
            # branch now i don't think...
            assert set(roi_names) == set(palette.keys())
            # Assuming all keys are proper colors.
            roi_name2color = palette

        # this branch covers things like lists of colors (including sns.color_palette
        # output, which is actually a custom class, but seems to present like a list of
        # colors)
        else:
            roi_names = list(set(roi_names))

            # TODO warn if # roi names is less than n_colors in input?

            # TODO warn if # rois > 10 (i.e. we are cycling colors)? unless some flag is
            # set to suppress?
            #
            # If palette is a sequence (of colors), this call should leave it unchanged
            # (unless n_colors changes, where it should cycle).
            #
            # If input is another output of color_palette (or a shorter sequence of
            # colors), this will wrap it such that each ROI gets a color
            # (though not unique if # ROIs > # colors).
            palette = sns.color_palette(palette, n_colors=len(roi_names))

            # just to show this output would not trigger the branch above
            # (that's for dict input)
            assert not isinstance(palette, Mapping)

            assert len(palette) == len(roi_names)

            # wanted a seeded random order w/o changing global seeds.
            rng = Random(seed)
            rng.shuffle(roi_names)
            roi_name2color = dict(zip(roi_names, palette))

    del palette

    if roi_name2color is not None:
        assert color is None
        # want to desaturate whether focus_rois is passed in or not, to keep colors
        # consistent (and i just liked the desaturated hls more than default).
        #
        # intentionally not doing this desat if input is color= (and not palette=, or
        # default palette generated if neither is passed)
        if palette_desat is not None:
            assert 0 < palette_desat <= 1
            roi_name2color = {
                # TODO just use desat arg to color_palette now?
                # (assuming focus roi color will always be passed in or else it will be
                # desat along with everything else)
                n: c if n == focus_roi else sns.desaturate(c, palette_desat)
                for n, c in roi_name2color.items()
            }

    # TODO option to [locally?] histogram equalize the image (or something else to
    # increase contrast + prevent hot pixels from screwing up range in a plane)
    # TODO option to "equalize" background image (see old code in plot_traces
    # show_footprints path)?

    z_size = rois.sizes['z']

    # TODO actually use (+ remove explicit kwargs here when possible)
    image_grid_kwarg_names = (
        'nrows', 'ncols', 'scale_per_plane', 'minmax_clip_frac', 'vmin', 'vmax', 'norm',
        'cmap', 'cbar', 'cbar_label', 'cbar_kws'
    )

    if image_kws is None:
        image_kws = dict()
    else:
        # (so we don't modify input when adding keys from kwargs below)
        image_kws = dict(image_kws)

    for k in image_grid_kwarg_names:
        # (to let image_grid defaults come thru)
        if k in kwargs:
            image_kws[k] = kwargs.pop(k)


    z_with_outlines = set(plane_outlines.roi_z.values)
    assert len(z_with_outlines) == plane_outlines.sizes['roi']

    if zero_outside_plane_outlines:
        if len(z_with_outlines) > 0:
            background = background.copy()

        for z in sorted(z_with_outlines):
            # should be fully False in the planes other than that where z==roi_z
            outline_vol = plane_outlines.sel(roi_z=z)

            # the .squeeze() is to collapse a length-1 'roi' dimension. may not need.
            plane_outline = outline_vol.isel(z=z).squeeze()
            assert plane_outline.sum().item(0) == outline_vol.sum().item(0)

            # TODO TODO this type of assignment into background work?
            #
            # may not need the .values
            background[z] = (background[z] * plane_outline).values

    # TODO also support background being xarray, transposing z,y,x into place, if so
    # (or at least [/ in ndarray case], assert existing shape matches rois (z,y,x)
    # shape?)
    fig, axs = image_grid(background, **image_kws)

    default_color_over_image = get_default_overlay_color(axs.flat[0])

    if scalebar:
        assert pixelsize_um is not None

        if scalebar_kws is None:
            scalebar_kws = dict()

        # TODO add kwarg to select which?
        # TODO try ax=axs[-1] by default? (honestly probably less likely to hit
        # stuff in top plane (i.e. axs[0]). VL2p might be there-ish in bottom...)
        ax = axs[0]
        add_scalebar(ax, pixelsize_um, color=default_color_over_image, **scalebar_kws)

    if focus_roi is not None:
        if focus_roi_linewidth is None:
            focus_roi_linewidth = linewidth

    default_plane_outline_line_kws = dict(color='gray', linestyles='dotted',
        # TODO translate here OR don't in general kwarg linestyle->*s case,
        # for consistency (or move all translation into plot_closed*...)
        # TODO TODO make thicker on plots like certain_rois_on_avg.pdf
        # (via plane_outline_line_kws) or resize the entirety of those plots?
        linewidths=0.6
    )
    if plane_outline_line_kws is None:
        plane_outline_line_kws = dict()
    else:
        plane_outline_line_kws = dict(plane_outline_line_kws)

    for k, v in default_plane_outline_line_kws.items():
        if k not in plane_outline_line_kws:
            plane_outline_line_kws[k] = v
    # TODO take any defaults from general **kwargs (the ones that other ROIs use)?

    # TODO replace above w/ rhs if assertion works
    # (and try to make similar replacement elsewhere!)
    assert plane_outline_line_kws == {
        **default_plane_outline_line_kws, **plane_outline_line_kws
    }

    # Moving 'roi' from end to start.
    rois = rois.transpose('roi', 'z', 'y', 'x')

    err_msg = None
    # TODO refactor to "micrometer_depth_titleS", that loops over axes and does for each
    for z, ax in enumerate(axs.flat):

        if zstep_um is not None:
            if depth_text_kws is None:
                depth_text_kws = dict()
            else:
                depth_text_kws = dict(depth_text_kws)

            if (depth_text_kws.get('as_overlay', False) and
                'color' not in depth_text_kws):

                depth_text_kws['color'] = default_color_over_image

            # TODO rename this fn to have title->text (since may want to burn in)
            # (or do outside this fn?)
            micrometer_depth_title(ax, zstep_um, z, **depth_text_kws)

        # TODO fix! (what is issue? any times where we get KeyError when we
        # actually had a plane outline ROI for that plane? delete comment?)
        try:
            plane_outline = plane_outlines.sel(roi_z=z)[z].squeeze()

            # TODO TODO refactor to not have this inside try block. it's itself like to
            # raise errors...
            #
            # just naively assuming these ROIs won't trigger same exceptions as
            # plot_closed_contours call below. that's probably not true in general, and
            # will just need to copy try/except structure from there.
            plot_closed_contours(plane_outline, ax=ax, _pad=_pad,
                **plane_outline_line_kws
            )

        except KeyError:
            #import ipdb; ipdb.set_trace()
            pass

        try:
            rois_in_curr_z = rois.sel(roi_z=z)
        except KeyError:
            continue

        # TODO what if not all coordinates associated w/ this dimension are on
        # the index? if i make a solution to also handle that, put in hong2p.xarray
        index_names = rois.get_index('roi').names

        # Since the .sel operation above removes this coordinate.
        index_names = [x for x in index_names if x != 'roi_z']

        for roi in rois_in_curr_z:
            roi = roi[z]

            index_vals = roi.roi.item()
            assert len(index_vals) == len(index_names)
            index = dict(zip(index_names, index_vals))
            name = index['roi_name']

            del index, index_vals

            lw = linewidth

            # TODO move assertion above (/delete)
            assert roi_name2color is not None or color is not None

            # NOTE: must be a single-element list if trying to specify one color with
            # any type other than a string (from matplotlib Axes.contour docs)'
            if roi_name2color is not None:
                _color = roi_name2color[name]
            else:
                _color = color

            if show_names == True:
                show_name = True
            elif show_names in (False, 'focus'):
                show_name = False

            if focus_roi is not None and name == focus_roi:
                lw = focus_roi_linewidth

                if focus_roi_color is not None:
                    _color = focus_roi_color

                if show_names == 'focus':
                    show_name = True

            # since we make palette by default currently, we should always have color
            # defined
            assert _color is not None

            try:
                # TODO TODO option to offset name (to draw just above ROI)
                # (can i specify 'above' rather than some specific offset? use ROI bbox
                # in plot_closed_contours?)

                # TODO probably err, not warn, if this gets unknown kwargs
                # (it's currently seaborn doing the warning, right?)
                plot_closed_contours(roi, label=name if show_name else None, ax=ax,
                    _pad=_pad, colors=[_color], linewidths=lw, **kwargs
                )

            except RuntimeError as err:
                # +1 to index as in ImageJ
                curr_err_msg = f'{name} (z={z + 1}): {err}'
                if err_msg is None:
                    err_msg = curr_err_msg
                else:
                    err_msg += f'\n{curr_err_msg}'

        if z == (z_size - 1):
            break

    # TODO factor into image_grid, to handle the layout there?
    # TODO same for colorbar?
    # (image_grid is currently picking figsize based on image dims, but this doesn't
    # account for colorbar / suptitle / anything else)
    if title is not None:
        # TODO some other way to anchor suptitle (or text in general, maybe), to axes
        # positions? current way kinda hacky... (and requires title to be added ~last)

        # NOTE: we need this before loop dealing with axes positions below (at least if
        # layout is some type of constrained layout), because this will determine axes
        # positions (which i don't think should change left/right after adding suptitle)
        fig.canvas.draw()

        # don't seem to need fig.set_layout_engine('none') before placing title
        # (to disable constrained layout)

        assert type(fig.axes) is list
        xmin = None
        xmax = None
        for ax in fig.axes:
            # get_images: "return a list of AxesImages contained by the Axes."
            # using to separate Axes with images from the colorbar axes
            # (which may also have been added in image_grid)
            n_images = len(ax.get_images())

            assert n_images in (0, 1)
            if n_images == 0:
                continue

            pos = ax.get_position()
            if xmin is None:
                xmin = pos.xmin
                xmax = pos.xmax
            else:
                xmin = min(xmin, pos.xmin)
                xmax = max(xmax, pos.xmax)

        assert xmin is not None and xmax is not None
        x = xmin + (xmax - xmin) / 2

        if title_kws is None:
            title_kws = dict()

        fig.suptitle(title, x=x, **title_kws)

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
        # 'name2' is just olf.NO_ODOR for a lot of my data
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
# TODO replace w/ ax.label_outer() based approach, like
# https://stackoverflow.com/questions/4209467? (added in 3.8, which i can't seem to
# install w/ my current pip + python at least... may be stuck on 3.7 [specifically
# on 3.7.3)
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
                color='b', size=boxlabel_fontsize, fontweight='bold'
            )
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

    # TODO also handle olf.NO_ODOR
    df = df.drop(
        index=df[(df.name1 == 'paraffin') | (df.name2 == 'paraffin')].index
    )

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


