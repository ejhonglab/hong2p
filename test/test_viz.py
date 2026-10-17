
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm, CenteredNorm, SymLogNorm

from hong2p import viz
from hong2p.viz import get_mpl_norm, min_tick_for_centered_cmap
from hong2p.olf import format_mix_from_strs


min_tick_cases = [
    (-0.3556, -0.4),
    (-.99999, -1.0),
    # would have needed np.isclose(min_tick, expected_min_tick) in comparisons for this
    # test, if not for adding code returning np.round(min_tick, 3), if it is np.isclose
    # to what output otherwise would be
    (-0.35374016010182685, -0.35),
]
@pytest.mark.parametrize('vmin, expected_min_tick', min_tick_cases)
def test_min_tick_for_centered_cmap(vmin: float, expected_min_tick: float):
    # this is the default increment
    increment = 0.05
    # also default kwarg in fn def
    round_tick_towards0_under_rtol: float = 0.1

    min_tick = min_tick_for_centered_cmap(vmin, increment=increment,
        round_tick_towards0_under_rtol=round_tick_towards0_under_rtol
    )
    assert min_tick == expected_min_tick


def test_is_centered_norm():
    kws = dict(vmin=-1, vmax=1)
    not_centered = [
        Normalize,
        Normalize(**kws),
    ]
    centered = [
        CenteredNorm,
        # TODO seems i can initialize this even without specifying any paramters. docs:
        # "Defaults to the largest absolute difference to vcenter for the values in the
        # dataset." (what exactly determines the "dataset"? must be deferred, but is it
        # always what i would want? all other norm classes similar in not requiring e.g.
        # vmin/vmax [i didn't think so, but maybe?]?)
        CenteredNorm(halfrange=0.5),
        TwoSlopeNorm,
        TwoSlopeNorm(vcenter=0, **kws),
        SymLogNorm,
        # TODO can i specify linthresh as keyword, despite being positional? yes, right?
        # (otherwise couldn't set via norm_kws in get_mpl_norm)
        SymLogNorm(linthresh=0.2, **kws),
    ]
    for x in not_centered:
        assert not viz.is_centered_norm(x)

    for x in centered:
        assert viz.is_centered_norm(x)


def test_get_mpl_norm():
    df = pd.DataFrame(data=[[1,2],[-1,0]])

    n1 = get_mpl_norm(**dict(norm=Normalize, vmin=-1, vmax=1))
    expect_success = [
        dict(norm=Normalize, vmin=-1, vmax=1),
        dict(norm=Normalize, vmin=-1, vmax=1),
        dict(norm=Normalize, data=df),
        dict(norm=Normalize, data=df, vmin=-1),
        dict(norm=Normalize, data=df, vmax=2),
        dict(norm=Normalize, data=df, vmin=-1),
        dict(norm=TwoSlopeNorm, data=df),
        dict(norm='two-slope', data=df),
        dict(norm=TwoSlopeNorm, vmin=-1, vmax=1),
        dict(norm=SymLogNorm, vmin=-1, vmax=1, linthresh=0.05),
    ]
    for kws in expect_success:
        print(f'{kws=}')
        get_mpl_norm(**kws)

    expect_assertion_err = [
        # TODO also test a fn wrapped with add_norm_options, to check it fails with n1
        # if vmin/vmax also passed
        #
        # passing n1 (and instantiated Normalize subclass) is also against get_mpl_norm
        # type hint
        dict(norm=n1),
        dict(norm=Normalize),
        dict(norm=Normalize, vmin=-1),
        dict(norm=Normalize, vmax=1),
        dict(norm=Normalize, vmin=1, vmax=-1),
        # this one will actually err becase not `vcenter < vmax`
        dict(norm=CenteredNorm, vmin=-1, vmax=0),
    ]
    for kws in expect_assertion_err:
        print(f'{kws=}')
        with pytest.raises(AssertionError):
            get_mpl_norm(**kws)

    with pytest.warns(UserWarning) as record:
        # TODO assert two warnings are emitted for this one? (and vmin of output
        # slightly < 0)
        # TODO TODO why no warnings emitted?
        t1 = get_mpl_norm(**dict(norm=TwoSlopeNorm, vmin=0.1, vmax=1))
        assert t1.vmin < 0
        assert t1.vcenter == 0

    assert len(record) == 2
    # 'setting vmin=0 since {vmin=} > {vcenter=}'
    w1 = record[0].message.args[0]
    assert w1.startswith('setting vmin=0 since')
    # 'setting vmin=({vcenter - epsion=}) (from {old_vmin}) to make vmin < {vcenter=}
    # True'
    w2 = record[1].message.args[0]
    assert w2.startswith('setting vmin=(')

    with pytest.warns(UserWarning) as record:
        t2 = get_mpl_norm(**dict(norm=TwoSlopeNorm, vmin=0, vmax=1))
        assert t2.vmin < 0
        assert t2.vcenter == 0

    w1 = record[0].message.args[0]
    assert w1.startswith('setting vmin=(')

    halfrange1_list = [
        dict(norm=CenteredNorm, vmin=0, vmax=1),
        dict(norm=CenteredNorm, vmin=-1, vmax=0.1),
    ]
    print('loop over halfrange1_list:')
    for kws in halfrange1_list:
        print(f'{kws=}')
        cn = get_mpl_norm(**kws)
        assert cn.vmin == -1 and cn.vmax == 1
        assert cn.halfrange == 1

    cn1 = get_mpl_norm(**dict(norm='centered', data=df))
    assert cn1.halfrange == 2
    cn2 = get_mpl_norm(**dict(norm='centered', data=-df))
    assert cn2.halfrange == 2

    with pytest.warns(UserWarning) as record:
        get_mpl_norm(**dict(norm=Normalize, vmin=1, vmax=1))

    w1 = record[0].message.args[0]
    assert w1 == 'vmin=1 == vmax=1. mistake?'

    # TODO assert about clipped fraction warning for this one? (and no warning w/
    # clip_warn=False?) (oh, that's actually only an option in fns wrapped with
    # add_norm_options, which makes sense [would want to warn whether norm initialized
    # or not]
    #dict(norm=Normalize, data=df, vmin=0, vmax=1),


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


def test_plot_closed_contours():
    import matplotlib.pyplot as plt

    footprint = np.zeros((20, 20))
    footprint[3:8, 3:8] = 1

    fig, ax = plt.subplots()
    # non-default linestyles and label, to also test setting path effects / drawing label
    contour = viz.plot_closed_contours(footprint, ax=ax, label='a', linestyles='solid')

    # contour is at 0.5 level, halfway between positive pixels and their zero neighbors
    assert contour.shape[1] == 2
    assert np.array_equal(contour.min(axis=0), [2.5, 2.5])
    assert np.array_equal(contour.max(axis=0), [7.5, 7.5])

    footprint[12:16, 12:16] = 1
    with pytest.raises(RuntimeError):
        viz.plot_closed_contours(footprint, ax=ax)

    plt.close(fig)
