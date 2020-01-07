#!/usr/bin/env python3

"""
"""

from os.path import join, split, exists
import glob
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import pyqtgraph as pg
from scipy.spatial.distance import pdist

import hong2p.util as u


def split_to_xyd(roi_data_xyd):
    assert len(roi_data_xyd.shape) == 2
    assert roi_data_xyd.shape[-1] == 3
    return roi_data_xyd.T


def wrap_update_image(update_image_fn, roi_data_xyd, pen=None, brush=None,
    pens_last=None, pens_next=None, text_items=None, debug=False):
    def wrapped_update_image(*args, **kwargs):
        update_image_fn(*args, **kwargs)
        # Since <ImageView object>.updateImage is a "bound method".
        self = update_image_fn.__self__
        # TODO TODO some way to not have to set brush and pens at each 
        # call? maybe i should just plot all at beginning and then
        # make stuff visible or not based on frame range (rather
        # than calling setData here)?
        xs, ys, diams = split_to_xyd(roi_data_xyd[self.currentIndex])
        self.scatter_plot.setData(x=xs, y=ys, size=diams, pen=pen, brush=brush,
            pxMode=True
        )

        if debug:
            # Assuming it is an iterable of length >= # of ROIs
            for text, x, y in zip(text_items, xs, ys):
                if pd.isnull(x):
                    # TODO set not visible
                    text.setVisible(False)
                    continue
                text.setPos(x, y)
                text.setVisible(True)

            print('current frame:', self.currentIndex)

        if pens_last is not None:
            if self.currentIndex >= 1:
                xs, ys, diams = \
                    split_to_xyd(roi_data_xyd[self.currentIndex - 1])

                self.scatter_plot_last.setData(x=xs, y=ys, size=diams,
                    pen=pens_last, brush=None, pxMode=True
                )
            else:
                self.scatter_plot_last.clear()

        if pens_next is not None:
            try:
                xs, ys, diams = \
                    split_to_xyd(roi_data_xyd[self.currentIndex + 1])

                self.scatter_plot_next.setData(x=xs, y=ys, size=diams,
                    pen=pens_next, brush=None, pxMode=True
                )
            except IndexError:
                self.scatter_plot_next.clear()

    return wrapped_update_image


def monkey_patch_image_window(imw, roi_data_xyd, debug=False,
        show_surrounding_frame_rois=True):
    """
    roi_data_xyd of shape (# timepoints, (max) # ROIs, 3 (x, y, diameter))
        2 [or 3 if specifying diameters])
    """
    # TODO maybe hide histogram thing / roi / roiPlot

    # TODO why do all the examples take care to setZValue of roi
    # to some positive # (like 10) to be "above image"?

    imw.scatter_plot = pg.ScatterPlotItem()
    # TODO maybe pick colors by middle / initial positions of ROIs,
    # so neighboring things dont share colors?
    # (+ button to recolor from current frame [only matters if ROIs really
    # swap positions, otherwise, any frame should probably work..., right?])
    colors = [pg.hsvColor(np.random.rand(), sat=1.0, val=1.0, alpha=1.0)
        for _ in range(roi_data_xyd.shape[1])
    ]
    pens = [pg.mkPen(color=c, width=1.5) for c in colors]
    brushes = None
    xs, ys, diams = split_to_xyd(roi_data_xyd[0])
    imw.scatter_plot.setData(x=xs, y=ys, size=diams, pen=pens, brush=brushes,
        pxMode=True
    )
    imw.view.addItem(imw.scatter_plot)

    if debug:
        text_items = []
        for i, (p, x, y) in enumerate(zip(pens, xs, ys)):
            text = pg.TextItem(text=str(i), color=p.color(), anchor=(0.5, 0.5))
            imw.view.addItem(text)
            # TODO setTextWidth
            if pd.isnull(x):
                text.setVisible(False)
            else:
                text.setPos(x, y)
            text_items.append(text)
    else:
        text_items = None

    if show_surrounding_frame_rois:
        # TODO if this is the approach i go with, add some kind of legend to
        # indicate what the different linestyles mean (which time direction)
        # (otherwise mark some way that doesn't need explanation...)
        pens_last = [pg.mkPen(color=c, width=1.0, style=pg.QtCore.Qt.DotLine)
            for c in colors
        ]
        pens_next = [pg.mkPen(color=c, width=1.0, style=pg.QtCore.Qt.DashLine)
            for c in colors
        ]

        imw.scatter_plot_last = pg.ScatterPlotItem()
        imw.view.addItem(imw.scatter_plot_last)

        imw.scatter_plot_next = pg.ScatterPlotItem()
        assert len(roi_data_xyd) > 1, 'expected more than one frame'
        xs, ys, diams = split_to_xyd(roi_data_xyd[1])
        imw.scatter_plot_next.setData(x=xs, y=ys, size=diams, pen=pens_next,
            brush=brushes, pxMode=True
        )
        imw.view.addItem(imw.scatter_plot_next)
    else:
        pens_last = None
        pens_next = None

    # TODO try to just call this fn once (manually specifying currentIndex=0
    # if necessary) (rather than duplicating some plotting stuff above)
    imw.updateImage = wrap_update_image(imw.updateImage, roi_data_xyd,
        pen=pens, brush=brushes, pens_last=pens_last, pens_next=pens_next,
        text_items=text_items, debug=debug
    )
    #
    #locs = [imw.view.itemBoundingRect(i) for i in imw.view.allChildren()]
    #import ipdb; ipdb.set_trace()
    #


# TODO take some features from floris' gui if i get time
# (plot of # of ROIs, "interesting" points (=ROI creation/deletion, right?),
# clickable stuff, etc)
# TODO option to only follow one ROI (and maybe another option, or 3 states
# total, as to whether or not to show even the current frame versions of the
# other ROIs)
# TODO how best to illustrate time ordering of displayed ROIs?
def show_movie(movie, rois=None, show_surrounding_frame_rois=True,
        debug_rois=False):

    imw = pg.image(movie, title='Movie' if rois is None else 'ROIs over time')

    # With these at -1 and 1, you should be able to see a one pixel border
    # around edges, to prove that when they are both 0, no pixels are left
    # unshown (and this is the case).
    common_min = -1 #0
    add_for_max = 1 #0
    add_for_scale = 0 + add_for_max - common_min
    frame_shape = movie.shape[1:]
    xmax = frame_shape[0] + add_for_max
    ymax = frame_shape[1] + add_for_max
    # TODO play around w/ this to find something reasonable
    min_px = 20

    # there seems to be something asymmetric about x axis view box limits
    # (no matter setting of xmin/max, cant drag small test image cleanly
    # to either left or right edge. i checked the off by 1/2 case)

    # imw.view is of type pg.ViewBox
    imw.view.setLimits(
        xMin=common_min, xMax=xmax,
        yMin=common_min, yMax=ymax,
        # off by one here or above? test
        maxXRange=frame_shape[0] + add_for_scale,
        maxYRange=frame_shape[1] + add_for_scale,
        # TODO some way to fix this? bug?
        #minXRange=min_px, minYRange=min_px
    )
    imw.view.setRange(xRange=(common_min, xmax), yRange=(common_min, ymax))

    # TODO does autolevels work on whole movie by default? way to make it?
    # if not, maybe setLevels from a given movies percentiles or something?

    # TODO delete. just so ROIs are visible while debugging
    if debug_rois:
        imw.setLevels(min=1.1, max=1.2)
    #

    # TODO opt to figure out max from dtype?
    # TODO maybe use pyqtgraphs subsampling range finding if the max is slow
    hmax = movie.max()

    # TODO TODO change scale on histogram to log
    # (may need to edit pyqtgraph to accomplish this)
    hist = imw.getHistogramWidget()
    # This does disable auto scaling.
    # (maybe remove this and go back to autoscaling? I originally just wanted
    # what the setLimits call is doing)
    hist.item.setHistogramRange(0, hmax)
    # actually, it might be worknig now that i also added xMin/Max
    # TODO but try to add some padding so harder to lose sliders at edges
    hist.item.vb.setLimits(yMin=0, yMax=hmax, xMin=0, xMax=hmax)

    imw.ui.roiBtn.hide()
    imw.ui.menuBtn.hide()

    # TODO later, modify pyqtgraph so scroll wheel work to move movie time
    # slider

    # TODO TODO params (frames before, frames after) [+ / OR] region sliders to
    # control which trajectories are visible about the current movie index

    if rois is not None:
        monkey_patch_image_window(imw, rois, debug=debug_rois,
            show_surrounding_frame_rois=show_surrounding_frame_rois
        )

    # TODO autoLevels() default on start (seems so)? at end timestep?
    # add it? (imageview[/item? or is it viewbox?])
    pg.QtGui.QApplication.exec_()


# TODO other test data where centers (mostly? constraints...) satisfy
# requirements for kalman filter to be optimal? (const accel?)
# TODO combine w/ fns i had to generate test images in that one test
# script mostly aimed at roi fitting / some ijroi stuff
# (to generate movie)


def main():
    np.random.seed(7)

    #"""
    nt = 4
    #nt = 2
    #test_movie = np.zeros((nt, 256, 256))
    test_movie = np.random.uniform(size=(nt, 256, 256))
    centers = u.make_test_centers(initial_n=3, nt=nt, p=None, verbose=True)
    #'''
    show_movie(test_movie, centers, show_surrounding_frame_rois=False,
        debug_rois=True
    )
    import sys; sys.exit()
    #'''
    #"""
    #import ipdb; ipdb.set_trace()

    tif = join(
        u.analysis_output_root(),
        #'2019-08-27/9/tif_stacks/fn_0001_nr.tif'
        '2019-11-18/3/tif_stacks/fn_0000_nr.tif'
    )
    tiff_title = u.tiff_title(tif)
    keys = u.tiff_filename2keys(tif)
    fps = u.get_thorimage_fps(u.thorimage_dir(*keys))
    movie = tifffile.imread(tif)
    shape_before = movie.shape

    blocks = u.movie_blocks(tif, movie=movie)
    assert movie.shape == shape_before


    block = blocks[0]
    target_fps = 1.0
    downsampled, new_fps = u.downsample_movie(block, target_fps, fps)
    print(f'new_fps: {new_fps:.2f}')

    # TODO delete. just to speed up testing.
    downsampled = downsampled[:4]
    #downsampled = downsampled[:10]
    #

    #
    #show_movie(downsampled)
    #import sys; sys.exit()
    #

    # This within block tracking may be less useful than across blocks (for me)
    n_ds_frames = len(downsampled)
    print(f'Fitting ROIs over {n_ds_frames} frames of downsampled movie:')
    # TODO try to parallelize this?
    withinblock_center_sequence = []
    for i in tqdm(range(n_ds_frames)):
        frame = downsampled[i]
        #centers, radius, _, _ = u.fit_circle_rois(tif, avg=frame)
        # TODO fix how location where script is run from influences where
        # auto_rois is?
        centers, radius, _, _ = u.fit_circle_rois(tif, avg=frame, threshold=0.3,
            multiscale=True, roi_diams_from_kmeans_k=2,
            exclude_dark_regions=True,
            debug=True, _packing_debug=True, show_fit=False
        )
        withinblock_center_sequence.append(centers)
    del n_ds_frames

    renumbered, new_centers = u.correspond_and_renumber_rois(
        withinblock_center_sequence, max_cost=max_cost, progress=False
    )

    # TODO add diam (or just preserve throughout last two fns?)
    import ipdb; ipdb.set_trace()
    show_movie(downsampled, rois_xyd)
    import sys; sys.exit()


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
        pairwise_plots=True, roi_numbers=roi_numbers, show=True
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


