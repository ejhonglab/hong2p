#!/usr/bin/env python3

from os.path import join

import pytest
import numpy as np
import tifffile

import hong2p.util as u
import hong2p.thor as thor


_data = None
def read_movie():
    global _data
    if _data is None:
        data_dir = u.thorimage_dir('2020-03-09', 1, 'fn_007')
        data = thor.read_movie(data_dir)
        _data = data
    return _data


def test_read_movie_volume():
    data = read_movie()

    # Created in ImageJ by specifying appropriate offset and gap between
    # images, as specified in the ThorImage manual (but using # Z slices,
    # including the flyback frames, whereever the manual's formulas talk
    # about channels).
    test_data_dir = '/mnt/nas/Tom/unit_test_data/python_2p_analysis'
    z0_tif = join(test_data_dir, 'fn_007_z0.tif')
    z0_data = tifffile.imread(z0_tif)
    assert np.array_equal(z0_data, data[:, 0, :, :])

    z1_tif = join(test_data_dir, 'fn_007_z1.tif')
    z1_data = tifffile.imread(z1_tif)
    assert np.array_equal(z1_data, data[:, 1, :, :])


def test_save_volume():
    data = read_movie()
    fname = 'test_vol.tif'
    # TODO TODO what does warning triggered here mean?? it seems tiff still has
    # expected number of frames...
    # "tifffile.py:1587 UserWarning: truncating ImageJ file"
    # (source code seems to indicate file might actually contain a subset of the
    # data, but round trip test works...)
    # TODO test to find biggest size actually achievable w/o real truncation?
    # (or just always round trip test whenever saving?)
    print(f'writing to {fname}...', flush=True, end='')
    u.write_tiff(fname, data)
    print(' done')

    # TODO how to do round trip test here? (ideally in a way that checks
    # that ImageJ considers the dimensions to be the appropriate quantity)

    data2 = tifffile.imread(fname)
    assert np.array_equal(data, data2)


if __name__ == '__main__':
    #test_read_movie_volume()
    test_save_volume()

