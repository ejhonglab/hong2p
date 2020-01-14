#!/usr/bin/env python3

"""
"""

from os.path import join, split, exists
import glob
from pprint import pprint
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import pyqtgraph as pg
from scipy.spatial.distance import pdist
import multiprocessing as mp

import hong2p.util as u


def split_to_xydm(roi_data_xyd, unmasked=False):
    assert len(roi_data_xyd.shape) == 2
    assert roi_data_xyd.shape[-1] == 3
    xs, ys, ds = roi_data_xyd.T
    mask = ~ np.isnan(xs)
    if unmasked:
        return xs, ys, ds, mask
    else:
        return xs[mask], ys[mask], ds[mask], mask


pxMode = False
def wrap_update_image(update_image_fn, roi_data_xyd, pens=None,
    pens_last=None, pens_next=None, text_items=None):

    def wrapped_update_image(*args, **kwargs):
        update_image_fn(*args, **kwargs)
        # Since <ImageView object>.updateImage is a "bound method".
        self = update_image_fn.__self__
        xs, ys, diams, mask = split_to_xydm(roi_data_xyd[self.currentIndex],
            unmasked=True
        )
        mpens = pens[mask]
        self.scatter_plot.setData(x=xs[mask], y=ys[mask], size=diams[mask],
            pen=mpens, brush=None, pxMode=pxMode
        )

        if text_items is not None:
            # Assuming it is an iterable of length >= # of ROIs
            for i, (text, x, y) in enumerate(zip(text_items, xs, ys)):
                if not mask[i]:
                    # TODO set not visible
                    text.setVisible(False)
                    continue
                text.setPos(x, y)
                text.setVisible(True)

        if pens_last is not None:
            if self.currentIndex >= 1:
                xs, ys, diams, mask = \
                    split_to_xydm(roi_data_xyd[self.currentIndex - 1])
                mpens_last = pens_last[mask]
                self.scatter_plot_last.setData(x=xs, y=ys, size=diams,
                    pen=mpens_last, brush=None, pxMode=pxMode
                )
            else:
                self.scatter_plot_last.clear()

        if pens_next is not None:
            try:
                xs, ys, diams, mask = \
                    split_to_xydm(roi_data_xyd[self.currentIndex + 1])
                mpens_next = pens_next[mask]
                self.scatter_plot_next.setData(x=xs, y=ys, size=diams,
                    pen=mpens_next, pxMode=pxMode
                )
            except IndexError:
                self.scatter_plot_next.clear()

    return wrapped_update_image


def monkey_patch_image_window(imw, roi_data_xyd,
    show_surrounding_frame_rois=True):
    """
    roi_data_xyd of shape (# timepoints, (max) # ROIs, 3 (x, y, diameter))
        2 [or 3 if specifying diameters])
    """
    # TODO maybe hide histogram thing / roi / roiPlot

    # TODO why do all the examples take care to setZValue of roi
    # to some positive # (like 10) to be "above image"?

    imw.scatter_plot = pg.ScatterPlotItem()

    # So colors are always the same.
    np.random.seed(7)
    # TODO maybe pick colors by middle / initial positions of ROIs,
    # so neighboring things dont share colors?
    # (+ button to recolor from current frame [only matters if ROIs really
    # swap positions, otherwise, any frame should probably work..., right?])
    colors = [pg.hsvColor(np.random.rand(), sat=1.0, val=1.0, alpha=1.0)
        for _ in range(roi_data_xyd.shape[1])
    ]
    pens = np.array([pg.mkPen(color=c, width=1.5) for c in colors])
    xs, ys, diams, mask = split_to_xydm(roi_data_xyd[0], unmasked=True)
    imw.scatter_plot.setData(x=xs[mask], y=ys[mask], size=diams[mask],
        pen=pens[mask], brush=None, pxMode=pxMode
    )
    imw.view.addItem(imw.scatter_plot)

    text_items = []
    for i, (p, x, y) in enumerate(zip(pens, xs, ys)):
        text = pg.TextItem(text=str(i), color=p.color(), anchor=(0.5, 0.5))
        imw.view.addItem(text)
        # TODO setTextWidth
        if not mask[i]:
            text.setVisible(False)
        else:
            text.setPos(x, y)
        text_items.append(text)
    text_items = np.array(text_items)

    if show_surrounding_frame_rois:
        # TODO if this is the approach i go with, add some kind of legend to
        # indicate what the different linestyles mean (which time direction)
        # (otherwise mark some way that doesn't need explanation...)
        pens_last = np.array([
            pg.mkPen(color=c, width=1.0, style=pg.QtCore.Qt.DotLine)
            for c in colors
        ])
        pens_next = np.array([
            pg.mkPen(color=c, width=1.0, style=pg.QtCore.Qt.DashLine)
            for c in colors
        ])

        imw.scatter_plot_last = pg.ScatterPlotItem()
        imw.view.addItem(imw.scatter_plot_last)

        imw.scatter_plot_next = pg.ScatterPlotItem()
        assert len(roi_data_xyd) > 1, 'expected more than one frame'
        xs, ys, diams, mask = split_to_xydm(roi_data_xyd[1])
        mpens_next = pens_next[mask]
        imw.scatter_plot_next.setData(x=xs, y=ys, size=diams, pen=mpens_next,
            brush=None, pxMode=pxMode
        )
        imw.view.addItem(imw.scatter_plot_next)
    else:
        pens_last = None
        pens_next = None

    # TODO try to just call this fn once (manually specifying currentIndex=0
    # if necessary) (rather than duplicating some plotting stuff above)
    imw.updateImage = wrap_update_image(imw.updateImage, roi_data_xyd,
        pens=pens, pens_last=pens_last, pens_next=pens_next,
        text_items=text_items
    )


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

    def mouse_click_fn(event):
        view_point = imw.view.mapSceneToView(event.scenePos())
        print('clicked:', (int(round(view_point.x())),
            int(round(view_point.y())))
        )
    imw.view.scene().sigMouseClicked.connect(mouse_click_fn)

    if rois is not None:
        monkey_patch_image_window(imw, rois,
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


# could maybe use this directly w/ starmap
#def fit_frame(frame_num, frame, tif):
def fit_frame(args):
    frame_num, frame, tif = args

    centers, radii, _, _ = u.fit_circle_rois(tif, avg=frame)
    rois_xyd = np.concatenate((centers,
        np.expand_dims(radii * 2, -1)), axis=-1
    )
    return frame_num, rois_xyd


def main():
    np.random.seed(7)
    '''
    nt = 4
    #nt = 2
    #test_movie = np.zeros((nt, 256, 256))
    test_movie = np.random.uniform(size=(nt, 256, 256))
    centers = u.make_test_centers(initial_n=3, nt=nt, p=None, verbose=True)
    print(centers[0])
    show_movie(test_movie, centers, show_surrounding_frame_rois=False,
        debug_rois=True
    )
    import sys; sys.exit()
    '''

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
    downsampled = downsampled[:5] #:10]
    #

    # TODO check if any time saved by rewriting to get um_per_pixel_xy
    # from first call (all tif should be used for), then passing that 
    # instead of tif (before splitting across processes)

    n_ds_frames = len(downsampled)
    print(f'Fitting ROIs over {n_ds_frames} frames of downsampled movie:')
    before = time.time()
    pool = mp.Pool()
    # TODO chunksize affect runtime (test on larger data)?
    # tqdm (grandularity at least)?
    ret_vals = list(tqdm(pool.imap_unordered(fit_frame,
        [x + (tif,) for x in enumerate(downsampled)]), total=n_ds_frames)
    )
    # Sort by frame number (first variable in return value).
    withinblock_center_sequence = [
        x[1] for x in sorted(ret_vals, key=lambda x: x[0])
    ]
    print('fitting frames took {:.1f}s'.format(time.time() - before))
    del n_ds_frames


    # TODO TODO TODO fix possible correspond_roi problems (ROIs seem close in
    # adjacent frames in GUI, but IDs change...): (from downsampled[:10])
    # 1/266 -> 2/268 (and same color... just chance?)
    # (w/ max_cost=10 or 20) 1/266 -> 2/265
    # 1/269 -> 2/267
    # 3/277 -> 4/275
    '''
    debug_points = {
        # TODO and musn't 268 have also been on first screen then, since it's a
        # lower number??? where? that jump must be incorrect, if it happened,
        # right? (likewise 266 must have been at roi_xyd[1])
        # tuple(roi_xyd[1][266, :2]) = (76, 122)
        # tuple(roi_xyd[2][268, :2]) = (nan, nan)
        1: [
            {'name': '268', 'xy0': (161, 148)},
            {'name': '266', 'xy0:': (76, 122)}
        ],
        2: [
            {'name': '266', 'xy0': (162, 148)}
        ]
    }
    '''
    # generated w/ u.roi_jumps
    debug_points = {
        1: [{'name': '265', 'xy0': (153, 116), 'xy1': (76, 122)},
             {'name': '266', 'xy0': (76, 122), 'xy1': (162, 148)},
             {'name': '267', 'xy0': (184, 154), 'xy1': (129, 110)}],
    }
    '''
        3: [{'name': '275', 'xy0': (162, 71), 'xy1': (168, 95)},
            {'name': '276', 'xy0': (153, 116), 'xy1': (204, 105)},
            {'name': '277', 'xy0': (167, 95), 'xy1': (174, 185)}],
        4: [{'name': '281', 'xy0': (236, 170), 'xy1': (185, 154)},
            {'name': '282', 'xy0': (69, 40), 'xy1': (156, 209)},
            {'name': '283', 'xy0': (120, 26), 'xy1': (186, 122)}],
        6: [{'name': '292', 'xy0': (69, 41), 'xy1': (94, 84)}],
        8: [{'name': '302', 'xy0': (68, 64), 'xy1': (247, 82)},
            {'name': '303', 'xy0': (247, 82), 'xy1': (60, 186)},
            {'name': '304', 'xy0': (236, 132), 'xy1': (164, 218)},
            {'name': '305', 'xy0': (58, 187), 'xy1': (244, 93)},
            {'name': '306', 'xy0': (136, 54), 'xy1': (98, 58)},
            {'name': '307', 'xy0': (164, 218), 'xy1': (174, 219)},
            {'name': '308', 'xy0': (164, 114), 'xy1': (42, 138)},
            {'name': '309', 'xy0': (244, 93), 'xy1': (166, 198)},
            {'name': '310', 'xy0': (139, 78), 'xy1': (36, 136)},
            {'name': '311', 'xy0': (148, 58), 'xy1': (136, 161)},
            {'name': '312', 'xy0': (134, 88), 'xy1': (160, 190)},
            {'name': '313', 'xy0': (227, 124), 'xy1': (84, 105)},
            {'name': '314', 'xy0': (170, 56), 'xy1': (184, 153)},
            {'name': '315', 'xy0': (138, 43), 'xy1': (77, 93)}]
    }
    '''
    #
    #debug_points = None

    # TODO take min over all of withinblock_center_sequence radii?
    # TODO test whether this value of max_cost leads to OK output
    # (used max_cost=10 before) could also try max_cost=radii.max()

    # TODO time this to see whether it's worth parallelizing
    before = time.time()
    roi_xyd = u.correspond_and_renumber_rois(
        withinblock_center_sequence, debug=True, debug_points=debug_points,
        progress=False, # if len(downsampled) < 10 else True
        #checks=False
        checks=True
    )
    print('matching ROIs across frames took {:.1f}s'.format(
        time.time() - before
    ))

    max_cost = 5
    jump_debug_points = u.roi_jumps(roi_xyd, max_cost)
    '''
    if len(jump_debug_points) > 0:
        print('debug_points = ', end='')
        pprint(jump_debug_points)
        for k, v in jump_debug_points.items():
            print(f'ci={k}, {len(v)} jumped')
        import ipdb; ipdb.set_trace()
    '''
    #import ipdb; ipdb.set_trace()

    nan_frac_per_frame = \
        np.sum(np.isnan(roi_xyd[:, :, 0]), axis=1) / roi_xyd.shape[1]
    print('Fraction of NaN ROIs per frame:')
    print(nan_frac_per_frame)

    show_movie(downsampled, roi_xyd)
    import ipdb; ipdb.set_trace()
    #import sys; sys.exit()


    # TODO TODO i think i want a pandas dataframe of
    # t, cell_id, x, y[, radius (if varying)]
    # (at least starting tidy for simplicity, though maybe another
    # representation would be better for getting data at a given time index
    # quickly)
    # where cell_id is constant for the stable cells, and assigned increasing
    # for each other cell (s.t. only one cell "tracklet" ever gets one cell_id)

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


