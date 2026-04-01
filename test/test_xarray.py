
from pathlib import Path

# TODO why do i get this error whether or not i am using hong2p vs al_analysis
# venv, and don't get this when running al_analysis stuff?
# does not depend on pytest being imported first
# (and why not when importing in other places? like in hong2p.xarray itself? ig only
# one import is actually run, and that's probably this one?)
import xarray as xr
import pytest

from hong2p.xarray import (move_all_coords_to_index, coords_equal, all_coord_names,
    load_dataarray, save_dataarray, get_example_orn_dynamics
)


# TODO provide this in some other module installed w/ hong2p, and not in test/ dir?
# or the core of it? also want to import and use in al_analysis/test/test_al_util.py
@pytest.fixture(scope='session')
def orns() -> xr.DataArray:
    """Returns an example DataArray (small subset of model ORN dynamics)
    """
    return get_example_orn_dynamics()


def test_roundtrip(orns, tmp_path):
    path = tmp_path / 'test.nc'
    # TODO also test w/ a raw numpy array (of same type as used for some dynamics), and
    # check that we can get same values back, after passing through all this
    save_dataarray(orns, path)
    orns2 = load_dataarray(path)
    assert orns.identical(orns2)


def test_all_coord_names(orns):
    names1 = all_coord_names(orns)
    names2 = all_coord_names(orns.reset_index('stim'))
    # .reset_index('stim') itself will reorder levels, so not the fault of this fn that
    # we can't check original order too
    assert set(names1) == set(names2), f'{names1=} != {names2=}'


def test_coords_equal(orns):
    assert coords_equal(
        orns.reset_index('stim').assign_coords({'x': 1}),
        orns.reset_index('stim').assign_coords({'x': 1})
    )
    assert coords_equal(orns.reset_index('stim'), orns.reset_index('stim'))
    assert not coords_equal(orns.reset_index('stim'), orns)
    assert not coords_equal(
        orns.reset_index('stim').assign_coords({'x': 1}),
        orns.reset_index('stim').assign_coords({'x': 1, 'y': 0})
    )
    assert not coords_equal(orns, orns.reset_index('stim').assign_coords({'x': 1}))

