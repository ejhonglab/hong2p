
import numpy as np
import pandas as pd

from hong2p import viz
from hong2p.olf import format_mix_from_strs


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
