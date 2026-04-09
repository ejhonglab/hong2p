
import numpy as np
import pandas as pd
import seaborn as sns

from hong2p import viz
from hong2p.olf import format_mix_from_strs


def test_is_cmap_diverging():
    # copied from matplotlib example. surely not exhaustive.
    type_and_cmaps = [
        ('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
        ('Sequential', [
           'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
           'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
           'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
        ('Sequential (2)', [
           'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
           'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
           'hot', 'afmhot', 'gist_heat', 'copper']),
        ('Diverging', [
           'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
           'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
        ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
        ('Qualitative', [
           'Pastel1', 'Pastel2', 'Paired', 'Accent',
           'Dark2', 'Set1', 'Set2', 'Set3',
           'tab10', 'tab20', 'tab20b', 'tab20c']),
        ('Miscellaneous', [
           'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
           'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
           'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
           'gist_ncar'])
    ]
    type2cmaps = dict(type_and_cmaps)

    # i use 'vlag', so want to make sure it works on that.
    # 'icefire' (at least in seaborn docs) seems to be a version like 'vlag' w/ black
    # instead of white in middle.
    # TODO may not want to support icefire/similar, as seems desat by edges?
    diverging_cmaps = ['vlag', 'icefire'] + list(type2cmaps['Diverging'])
    for cmap in diverging_cmaps:
        # TODO support? right not saturated, just black (left red, white mid)
        if cmap == 'RdGy':
            continue

        assert viz.is_cmap_diverging(cmap)

    non_diverging_cmaps = [
        x for k, xs in type2cmaps.items() if k != 'Diverging' for x in xs
    ]
    for cmap in non_diverging_cmaps:
        # this one might as well be diverging, though maybe not as step around mid?
        # TODO actually test steepness of sat around mid?
        if cmap == 'twilight_shifted':
            continue

        assert not viz.is_cmap_diverging(cmap)

    # TODO could also get diverging/not palettes from palettable to test?
    # (palettable doesn't have vlag exactly tho, at least not by name)
    # (have commented code to do that in viz)


# establishing some behavior of sns.color_palette, so I can clean up viz.plot_rois code
# that may depend on some of that behavior
def test_seaborn_color_palette():
    # first tested in v0.11.1 (installed in hong2p test venv)
    # TODO also test in in v0.12.2 (what i have in al_analysis venv)

    # TODO TODO test for some color maps that should / should-not wrap (there def should
    # be some of former, but not 100% sure there are some for latter)
    # if there are both, 'hls' is the kind that seems arbitrarily divisible.
    pstr = 'hls'

    # TODO also try below w/ this <10? 9?
    p1 = sns.color_palette(pstr, n_colors=10)

    p2 = sns.color_palette(p1, n_colors=25)

    # (it wraps)
    assert p1 == p2[:10]
    assert p1 == p2[10:20]
    assert p1[:5] == p2[20:]

    p3 = sns.color_palette(pstr, n_colors=25)
    # p3 does not wrap
    # (not that it's clear from this assertion, but it is w/ sns.palplot)
    # p2 wraps every 10 colors. p3 divides 'hls' into 25 colors.
    assert p3 != p2

    # TODO how can we know if a cmap is divisible in >10 colors? or how many colors it's
    # divisible into?
    """

    # TODO delete
    import matplotlib.pyplot as plt

    '''
    sns.palplot(p1)
    sns.palplot(p2)
    sns.palplot(p3)
    '''

    # TODO TODO check whether p1 and p2 behave same? any point in wrapping?

    # TODO TODO have plot_rois err if it gets a list of colors for palette, and that's
    # (much?) > number of ROIs (cause then we're gonna use a smaller part of space)

    import ipdb; ipdb.set_trace()
    """
    #


def test_callable_ticklabels():
    # TODO test hong2p.olf.format_mix_from_strs separately
    # TODO share data w/ first test in here

    # This is decorated the same way as, for instance, viz.matshow, but just returns the
    # [x|y]ticklabels the decorator will compute and inject into the kwargs.
    @viz.callable_ticklabels
    def dummy_plot_fn(df, **kwargs):
        return kwargs.get('xticklabels'), kwargs.get('yticklabels')

    def _check_ticklabels(xindex, yindex, expected_xticklabels=None,
        expected_yticklabels=None, **kwargs):

        # Should just be filled w/ NaN w/o data explicitly passed, which is fine.
        df = pd.DataFrame(columns=xindex, index=yindex)

        xticklabels, yticklabels = dummy_plot_fn(df, **kwargs)

        if expected_xticklabels is not None:
            # TODO maybe also assert all of type str w/ just a single level?
            assert np.array_equal(xticklabels, expected_xticklabels)

        if expected_yticklabels is not None:
            assert np.array_equal(yticklabels, expected_yticklabels)

    glomeruli = ['DL5', 'DM1']
    rois_indices = (
        glomeruli,
        pd.Index(glomeruli, name='roi'),
    )

    odor_index = pd.MultiIndex.from_product([['solvent', 'ea'], ['solvent', 'eb']],
        names=['odor1', 'odor2']
    )
    expected_odor_strs = ['solvent', 'eb', 'ea', 'ea + eb']

    _check_ticklabels(glomeruli, odor_index, glomeruli, expected_odor_strs,
        yticklabels=format_mix_from_strs, xticklabels=True
    )

    # TODO test a default w/ this one (to test it can handle heterogenous types with the
    # ints
    odor_index = pd.MultiIndex.from_product([['ea'], ['eb'], [1,2,3]],
        names=['odor1', 'odor2', 'repeat']
    )
    expected_odor_strs = [f'ea / eb / {x}' for x in [1,2,3]]
    _check_ticklabels(glomeruli, odor_index, None, expected_odor_strs,
        yticklabels=True, xticklabels=False
    )

    odor_strs = ['ea', 'eb', 'ea + eb']
    odor_indices = (
        odor_strs,
        pd.Index(odor_strs, name='odor'),
    )
    for odor_index in odor_indices:
        for roi_index in rois_indices:
            # Mainly checking that if rows / cols indices are just single level indices
            # containing only strings, nothing fails and the way I call the
            # functions/wrote them doesn't introduce some pandas garbage I wouldn't want
            # (like the dtype).
            _check_ticklabels(roi_index, odor_index, roi_index, odor_index,
                xticklabels=True, yticklabels=True
            )
