#!/usr/bin/env python3

"""
To use this script, you just need to install this one dependency as so:
pip install git+https://github.com/ejhonglab/hong2p

Tested with python3.8 though anything 3.6+ is probably fine.
"""

from os.path import join
from pprint import pprint

import tifffile
import ijroi
import numpy as np
import matplotlib.pyplot as plt
from hong2p import thor, util


def main():
    # NOTE: change these paths to where you currently have your data
    data_dir = '/mnt/d1/2p_data/YZ_test'
    thorimage_dir = join(data_dir, '2FLfun1')
    thorsync_dir = join(data_dir, '2FLfun1_SyncData')

    # was currently broken on your computer (and just generally on this data actually i
    # think)
    '''
    bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_dir,
        thorimage_dir
    )
    pprint(bounding_frames)
    '''

    #movie = thor.read_movie(thorimage_dir)

    tif_path = join(data_dir, '20210513_2FLfun1_moco.tif')
    movie = tifffile.imread(tif_path)

    ijroi_path = join(data_dir, '20210513_2FLfun1_moco_dff_roi.roi')
    with open(ijroi_path, 'rb') as f:
        roi = ijroi.read_roi(f)

    mask = util.ijroi2mask(roi, movie.shape[-2:])

    mean = movie.mean(axis=0)
    plt.figure()
    plt.imshow(mean)

    plt.figure()
    plt.imshow(mask)

    # since my extract function expects the number of footprints as the last dimension
    masks = np.expand_dims(mask, -1)

    traces = util.extract_traces_bool_masks(movie, masks)
    trace = traces[:, 0]

    plt.figure()
    plt.plot(trace)

    plt.show()


if __name__ == '__main__':
    main()

