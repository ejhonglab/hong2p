#!/usr/bin/env python3

"""
"""

from os.path import join

#####import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
#####import pyqtgraph as pg

import hong2p.util as u


def test(a1, a2):
    print(a1)
    print(a2)


def wrap_update_image(update_image_fn):
    def wrapped_update_image(*args, **kwargs):
        update_image_fn(*args, **kwargs)
        # Since <ImageView object>.updateImage is a "bound method".
        self = update_image_fn.__self__

        # TODO is there some reason this only seems to go up to index
        # 298 when there are 300 frames (last index should be 299)
        # i found a similar diff w/ 1 vs. 2 arg fn connected to 
        # sigTimeChange signal (only 2 arg fns seemed to let first arg
        # [/ only arg in 1 arg case] go up to 299 [vs. 298 again])
        print(self.currentIndex)

    return wrapped_update_image


def monkey_patch_image_window(imw, data):
    # TODO maybe hide histogram thing / roi / roiPlot

    # TODO why do all the examples take care to setZValue of roi
    # to some positive # (like 10) to be "above image"?

    imw.scatter_plot = pg.ScatterPlotItem()
    imw.view.addItem(imw.scatter_plot)

    imw.updateImage = wrap_update_image(imw.updateImage, data)


'''
def show_centers_on_movie(movie, 
    imw = pg.image(movie)
    monkey_patch_image_window(imw)

    pg.QtGui.QApplication.exec_()
'''


def make_test_centers(n=50, nt=100, frame_shape=(256, 256), sigma=3,
    exlusion_radius=None, loss_probability=0.01, gain_probability=0.01,
    round_=True):

    # TODO TODO how to balance gain probability / loss probability s.t.
    # # ROIs stays about constant? any way to do this w/o having gain
    # probability depend on the current # of ROIs?
    # (since loss probability must be dependent on current # of ROIs at least
    # somewhat, since if there are 0, we can't lose more...)
    # maybe the probability should still not be considered for each ROI,
    # and then just do nothing when there are no ROIs left to take...
    # (and allow multiple per time step? which distribution to use?)

    if exlusion_radius is not None:
        raise NotImplementedError

    assert len(frame_shape) == 2
    assert frame_shape[0] == frame_shape[1]
    d = frame_shape[0]

    # TODO have shape (size) be consistent w/ other places that deal
    # w/ centers. (2 first or n first?)
    initial_centers = np.random.randint(d, size=(n, 2))

    # TODO TODO more idiomatic numpy way to generate cumulative noise?
    # (if so, just repeat initial_centers to generate centers, and add the 
    # two) (maybe not, with my constraints...)
    xy_steps = np.random.randn(nt - 1, n, 2) * sigma
    
    centers = np.empty((nt, n, 2)) * np.nan
    centers[0] = initial_centers
    max_coord = d - 1
    # TODO should i be generating the noise differently, so that the x and y
    # components are not independent (so that if deviation is high in one,
    # it's more likely to be lower in other coordinate, to more directly
    # constrain the distance? maybe it's just a scaling thing though...)
    for t in range(1, nt):
        centers[t] = centers[t - 1] + xy_steps[t - 1]
        centers[t][centers[t] > max_coord] = max_coord
        centers[t][centers[t] < 0] = 0
        # TODO TODO TODO also support losing / gaining centers, and have output
        # in same format as the fns in util to associate ROIs (keeping labels
        # const) across matchings

    if round_:
        centers = np.round(centers).astype(np.uint16)

    return centers



# TODO TODO TODO fns for generating test data. centers + gaussian updates
# (w/ edge + non-overlap constraints) first. const radii?
# use that to test gui + fitting + tracking
# TODO other test data where centers (mostly? constraints...) satisfy
# requirements for kalman filter to be optimal? (const accel?)
# TODO combine w/ fns i had to generate test images in that one test
# script mostly aimed at roi fitting / some ijroi stuff
# (to generate movie)


def main():
    centers = make_test_centers()
    import ipdb; ipdb.set_trace()
    '''
    test_movie = np.random.randn(300, 256, 256)
    imw = pg.image(movie)
    #imw.sigTimeChanged.connect(test)
    pg.QtGui.QApplication.exec_()
    '''

    tif = join(
        u.analysis_output_root(),
        '2019-08-27/9/tif_stacks/fn_0001_nr.tif'
    )
    keys = u.tiff_filename2keys(tif)
    fps = u.get_thorimage_fps(u.thorimage_dir(*keys))
    movie = tifffile.imread(tif)
    shape_before = movie.shape
    blocks = u.movie_blocks(tif, movie=movie)
    assert movie.shape == shape_before

    tiff_title = u.tiff_title(tif)


    block = blocks[0]
    target_fps = 1.0
    downsampled, new_fps = u.downsample_movie(block, target_fps, fps)
    print(f'new_fps: {new_fps:.2f}')

    # TODO delete. just to speed up testing.
    downsampled = downsampled[:10]
    #

    # This within block tracking may be less useful than across blocks (for me)
    n_ds_frames = len(downsampled)
    print(f'Fitting ROIs over {n_ds_frames} frames of downsampled movie:')
    # TODO try to parallelize this?
    withinblock_center_sequence = []
    for i in tqdm(range(n_ds_frames)):
        frame = downsampled[i]
        centers, radius, _, _ = u.fit_circle_rois(tif, avg=frame)
        withinblock_center_sequence.append(centers)

    lr_matches, unmatched_left, unmatched_right, cost_totals, fig = \
        u.correspond_rois(withinblock_center_sequence, max_cost=radius,
        show=False, progress=True
    )

    print('Finding ROIs stable across all timepoints...', end='', flush=True)
    withinblock_stable_cells, new_lost = u.stable_rois(lr_matches, verbose=True)
    print(' done')

    renumbered, new_centers = \
        u.renumber_rois(lr_matches, withinblock_center_sequence)


    # TODO TODO i think i want a pandas dataframe of
    # t, cell_id, x, y[, radius (if varying)]
    # (at least starting tidy for simplicity, though maybe another
    # representation would be better for getting data at a given time index
    # quickly)
    # where cell_id is constant for the stable cells, and assigned increasing
    # for each other cell (s.t. only one cell "tracklet" ever gets one cell_id)
    '''
    for i, centers in enumerate(withinblock_center_sequence):
        stable = 
        np.setdiff1d(
        import ipdb; ipdb.set_trace()
    '''

    '''
    pw = pg.image(downsampled)
    #pw.sigTimeChanged.connect(test)
    pg.QtGui.QApplication.exec_()
    '''

    import ipdb; ipdb.set_trace()
    #

    # TODO maybe fns to transform representations of roi labels over time above
    # to something where IDs go high enough to represent all unique across all
    # timepoints, and maybe NaN when ROI not there (non-NaN values should all be
    # in one run somewhere in each row)?
    # (might simplify [passing data to] plotting)

    # TODO TODO TODO fns to take a sequence of ROIs (+ frames / time ranges they
    # apply to!!) and extract traces as with my static-ROI imagej-roi-based
    # analysis
    # TODO TODO maybe just start w/ per-block trace extraction?

    # TODO maybe also a fn to display each (median?) ROI from sequence

    # TODO TODO or maybe start with a fn to correspond static ROIs to fit
    # (block-by-block) rois, and then use stability of block-by-block ROIs
    # as a proxy for stability of static ROIs
    # TODO or skip the static ROI part altogether?

    # TODO TODO fn to visualize centers being tracked over whole movie
    # save to movie? pyqtgraph? probably want shortish fixed-time window
    # beyond which displayed part of trajectories are culled.
    # may also want some somewhat persistent and noticeable marker for
    # trajectories getting lost.
    # TODO graph coloring to pick colors? prob easier to just rely on
    # randomness...



    center_sequence = []
    for i in range(len(blocks)):
        frame = blocks[i].mean(axis=0)
        centers, radius, _, _ = u.fit_circle_rois(tif, avg=frame)
        center_sequence.append(centers)

    roi_numbers = False

    avg = movie.mean(axis=0)
    lr_matches, unmatched_left, unmatched_right, cost_totals, fig = \
        u.correspond_rois(center_sequence, max_cost=radius + 1,
        draw_on=avg, title=tiff_title,
        pairwise_plots=True, roi_numbers=roi_numbers
    )
    stable_cells, new_lost = u.stable_rois(lr_matches)

    print(f'Number of cells detected in each block:',
        [len(cs) for cs in center_sequence])
    print(f'Number of cells stable across all blocks: {len(stable_cells)}')

    # Getting some random sequences of IDs for a given real cell,
    # to visually check ID correpondence.
    n_samples = 5
    rows = np.random.choice(len(stable_cells), size=n_samples, replace=False)
    id_sequences = stable_cells[rows]
    assert id_sequences.shape == (n_samples, len(center_sequence))

    print('IDs in each of these rows should match up', end='')
    if roi_numbers:
        print(' (white markers on plot):')
    else:
        print(':')

    for row in id_sequences:
        print(list(row))

    ax = fig.axes[0]
    # TODO TODO maybe fn to plot output as from (/ on top of) correspond_rois
    # output, but with a single large 'o' marker around any not-fully-stable
    # sets of ROIs (maybe not in correspond_rois, since we need output of
    # stable_rois, unless i want to combine the two)
    markerstyles = ['o', 'v', 's']
    for i, centers in enumerate(center_sequence):
        if i == len(lr_matches):
            unmatched = unmatched_right[i - 1]
        else:
            unmatched = unmatched_left[i]

        stable = stable_cells[:,i]

        # This should also include stuff in unmatched things above (?)
        to_drop = np.setdiff1d(np.arange(len(centers)), stable)

        # The unmatched stuff should be a subset of stuff we are dropping
        # because of a lack of a match across blocks.
        assert np.array_equal(np.union1d(to_drop, unmatched), to_drop)
        # TODO and their difference should be a subset of the input indices in
        # lr_match column. meaningful check?

        # TODO modify legend to say "Dropped" or something if possible
        # https://stackoverflow.com/questions/23689728 ?
        ax.scatter(*centers[to_drop].T, facecolors='none', edgecolors='r',
            s=50, marker=markerstyles[i])

        if roi_numbers:
            test_ids = id_sequences[:, i]
            ax.scatter(*centers[test_ids].T, facecolors='none', edgecolors='w',
                s=70, marker=markerstyles[i])

    #plt.show()
    #import ipdb; ipdb.set_trace()
    #

    # TODO test case where set of indices in each column of stable_cells
    # is not the same
    
    # TODO move to test
    # either fix this test to be meaningful again or delete
    '''
    # (it just was in one test case, and it's useful for the test)
    assert len(lr_matches[0]) > len(stable_cells)

    n_test = len(lr_matches[0])
    m1 = lr_matches[0][:n_test,:]
    n_non_stable = 0
    for i in range(n_test):
        m2_left = lr_matches[1][:,0]
        aw_out = np.argwhere(m2_left == m1[i,1])
        if len(aw_out) == 0:
            #import ipdb; ipdb.set_trace()
            n_non_stable += 1
            # something else to check here?
            continue

        m2_idx = aw_out[0,0]
        m2_row = lr_matches[1][m2_idx, :]
        assert m2_row[0] == m1[i,1]
        assert m2_row[1] == stable_cells[i, 2]

    assert n_non_stable == (len(lr_matches[0]) - len(stable_cells))
    '''
    #

    # TODO move to unit test
    # Checking ROI label correspondence.
    def mean_dist_from_mean(points):
        # TODO there some other name for this / some more appropriate value?
        # i want to get a sense of how clustered these points are in 2d space
        mean = np.mean(points, axis=0)
        return np.mean([u.euclidean_dist(pt, mean) for pt in points])

    np.random.seed(50)
    n_samples = 5
    row_indices = np.random.choice(len(stable_cells), size=n_samples,
        replace=False)

    idx_spreads = []
    for row in stable_cells[row_indices, :]:
        points = np.array([pts[i, :] for i, pts in zip(row, center_sequence)])
        assert points.shape == (len(center_sequence), 2)
        mean_dist = mean_dist_from_mean(points)

        # TODO delete at some point (or delete either mean_dist_fr... above
        # or np.linalg.norm approach below)
        # this argues that they are consistent.
        mean = np.mean(points, axis=0)
        diffs = points - mean
        dists = np.linalg.norm(diffs, axis=1)
        assert np.mean(dists) == mean_dist
        #
        idx_spreads.append(mean_dist)

    # maybe also compare to a same-index-in-centerlist approach
    random_pts = np.empty((len(center_sequence), n_samples, 2)) * np.nan
    for i, pts in enumerate(center_sequence):
        random_idxs = np.random.choice(len(pts), size=n_samples, replace=False)
        random_pts[i] = pts[random_idxs, :]

    means = np.mean(random_pts, axis=0)
    diffs = random_pts - means
    dists = np.linalg.norm(diffs, axis=2)
    random_spread = np.mean(dists)

    assert random_spread > np.mean(idx_spreads)
    #

    # TODO delete / move to test
    # (was only true w/ max_cost=radius)
    '''
    m1_only_good = [254, 256, 262, 271]
    m1_only_bad = [19, 38, 228, 241, 245, 248]
    m1_should_not_have = np.array(m1_only_good + m1_only_bad)
    assert len(np.intersect1d(m1_should_not_have, stable_cells[:,0])) == 0
    '''
    #

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()


