#!/usr/bin/env python3

from os.path import join

import pytest
import ijroi
import matplotlib.pyplot as plt

from hong2p import util
from hong2p.roi import ijrois2masks, extract_traces_bool_masks
from hong2p.thor import read_movie


@pytest.mark.skip(reason='not fully implemented')
@pytest.mark.slow
def test_extract_volume_traces():
    # TODO include some reduced test data in a submodule / dvc thing / etc?
    data_dir = util.thorimage_dir('2020-03-09', 1, 'fn_007')
    movie = read_movie(data_dir)

    ijroiset_filename = join(data_dir, 'rois.zip')
    ijrois = ijroi.read_roi_zip(ijroiset_filename)

    # TODO TODO TODO wasn't z supposed to be first though? is some of my
    # dimension handling in this case inconsistent?
    frame_shape = movie.shape[1:]

    #test_nested_calls(ijrois[0][1], frame_shape)

    footprints = ijrois2masks(ijrois, frame_shape)

    # TODO TODO what is appropriate test this is actually working?
    raw_f = extract_traces_bool_masks(movie, footprints)
    import ipdb; ipdb.set_trace()


# TODO also include test w/ some of old ijrois for single plane KC stuff


if __name__ == '__main__':
    test_extract_volume_traces()

