#!/usr/bin/env python3

import pytest
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tifffile
import ijroi

from hong2p import util
from hong2p.roi import (ijrois2masks, extract_traces_bool_masks, get_circle_ijroi_input,
    fit_circle_rois, load_template_data, _get_template_roi_radius_px
)


# Module level setup below is failing at the moment, so skipping via arguments added in
# ../pytest.ini now instead.
#pytestmark = pytest.mark.skip('test data loading broken + tested code unused')

# Only for interactively working on tests. Should be false if running
# automated tests.
interactive = False

# TODO test fixture this or something else to make it easier to get this (or at least
# some representative) test data on fresh installs
template_data = load_template_data()
assert template_data is not None, 'could not load template data'
template = template_data['template']
margin = template_data['margin']
mean_cell_diam_um = template_data['mean_cell_diam_um']
#frame_shape = template_data['frame_shape']
#frame_shapes = [(256, 256), (512, 512)]
frame_shapes = [(256, 256), (512, 512)][::-1]

# This was to try to figure out some boundary conditions in
# make_img_with_rois.
#template = np.random.randn(*frame_shape)

# TODO taken from 2019-11-21/3/fn_0004. maybe use template data here?
#um_per_pixel_xy = 0.467

# this right? (it is pretty close to the above. 0.43611...)
template_cell_diam_px = template.shape[0] - 2 * margin
um_per_pixel_xy = mean_cell_diam_um / template_cell_diam_px

# note that this may currently be a float...
template_cell_radius_px = template_cell_diam_px / 2

# First argument is normally a TIFF filename, but if: not writing ijrois
# , not loading the TIFF (movie or avg is passed), and not loading XML
# (if _um_per_pixel_xy is passed), then it is not used.
tiff_fname = None
u16_max = 2**16 - 1


def plot_template():
    assert template.shape[0] % 2 == 0, \
        'will probably look better if template has even dimension sizes'
    center = (template.shape[0] // 2,) * 2

    if show_template:
        fig, ax = plt.subplots()

        radius_px = int(round(template_cell_radius_px))
        roi_circle = plt.Circle((center[0] - 0.5, center[1] - 0.5), radius_px,
            fill=False, color='r'
        )
        ax.add_artist(roi_circle)

        ax.imshow(template)
        ax.set_title(f'Template {template.shape}\n'
            f'(integer pixel radius = {radius_px})'
        )

    roi_d = template.shape[0]
    radius_px = template_cell_radius_px
    print(f'Radius pixels at UNSCALED template d={roi_d}: {radius_px:.3f}')


def rint(float_num):
    return int(round(float_num))


def get_asymmetric_center(frame_shape):
    # Not in center, to be clear about coordinate systems
    return (rint(frame_shape[0] / 2) - 30, rint(frame_shape[1] / 2) + 5)


def get_wellseparated_centers(frame_shape):
    centers = [
        # Center (approx? exact? off-by-one?).
        (rint(frame_shape[0] / 2), rint(frame_shape[1] / 2)),

        get_asymmetric_center(frame_shape),

        (rint(frame_shape[0] / 2), rint(frame_shape[1] / 2) - 20),

        # Top left corner (check!), according to ImageJ.
        #(0, 0)
    ]
    return centers


# TODO TODO parameters for adding (realistic?) noise to the image

# TODO maybe also generate some noise from real ground truthed data

# TODO TODO maybe some way to add diff amounts of noise for each template
# (or scale intensity of that template? *should* template matching
# be insensitive to that (maybe not, even w/ constant bg / bg noise?)?)
# (-> check that ranking of components in output matches (in well-separated
# case, at least), the ordering of this input "SNR" value?)

#def make_img_with_rois(centers, radii_pixels=None, intensity_scales=None,
def make_img_with_rois(frame_shape, centers, template_dimension_changes=None,
    intensity_scales=None, bg_from='template_min'):

    img = np.zeros(frame_shape).astype(template.dtype)

    if bg_from == 'template_edge_avg':
        edge_avg = np.mean([template[0,:].mean(), template[-1,:].mean(),
            template[:,0].mean(), template[:,-1].mean()
        ])
        bg_constant = edge_avg

    elif bg_from == 'template_min':
        bg_constant = template.min()

    elif bg_from == 0:
        bg_constant = 0

    else:
        raise ValueError('invalid value of bg_from (see code)')

    ground_truth_radii_px = []
    for i, (cx, cy) in enumerate(centers):
        assert type(cx) is int and type(cy) is int, \
            'can only embed template at integer offsets'

        roi_d = template.shape[0]
        if template_dimension_changes is not None:
            roi_d += template_dimension_changes[i]

        radius_px = util._get_template_roi_radius_px(template_data,
            if_template_d=roi_d, _round=False
        )
        ground_truth_radii_px.append(radius_px)
        print(f'Radius pixels at template d={roi_d}: {radius_px:.3f}')

        # TODO TODO is main thing in greedy_roi_packing stopping me from using
        # fullying float radii the fact that testing relies on drawing
        # fn that only accepts integer radii (is that even true?)?
        # if so, maybe just write my own testing fn / find some other library
        # call? (cause might be useful to increment/decrement overall template
        # dimension (2 * [radii + margin]), and back-calculate float radius from
        # that)

        # TODO maybe test w/ roi_d an even number / odd?
        bbox_minx = cx - rint(roi_d / 2)
        bbox_miny = cy - rint(roi_d / 2)

        # Excludes the last point, just like numpy slicing.
        # (each element of shape does not index a single element in
        # that dimension, because it's one out of bounds, but it's
        # valid as the end of a slice)
        bbox_maxx = bbox_minx + roi_d
        bbox_maxy = bbox_miny + roi_d

        roi_minx, roi_miny = (0, 0)
        if bbox_minx < 0:
            roi_minx = abs(bbox_minx)
            bbox_minx = 0

        if bbox_miny < 0:
            roi_miny = abs(bbox_miny)
            bbox_miny = 0

        roi_maxx, roi_maxy = frame_shape
        if bbox_maxx > frame_shape[0]:
            roi_maxx = roi_d - (bbox_maxx - frame_shape[0])
            bbox_maxx = frame_shape[0]

        if bbox_maxy > frame_shape[1]:
            roi_maxy = roi_d - (bbox_maxy - frame_shape[1])
            bbox_maxy = frame_shape[1]

        roi = cv2.resize(template, (roi_d, roi_d))
        roi_slice = roi[
            roi_minx:roi_maxx,
            roi_miny:roi_maxy
        ]
        img_slice = img[bbox_minx:bbox_maxx, bbox_miny:bbox_maxy]
        assert img_slice.shape == roi_slice.shape

        roi_to_add = roi_slice.copy() - bg_constant
        if intensity_scales is not None:
            roi_to_add = roi_to_add * intensity_scales[i]

        img[bbox_minx:bbox_maxx, bbox_miny:bbox_maxy] = img_slice + roi_to_add

    if interactive and show_generated_scenes:
        fig, ax = plt.subplots()
        ax.imshow(img)
        title = f'centers = {centers}'
        if intensity_scales is not None:
            title += f'\nintensity_scales = {intensity_scales}'
        if template_dimension_changes is not None:
            title += (f'\ntemplate_dimension_changes = '
                f'{template_dimension_changes}'
            )
        ax.set_title(title)

    return img, ground_truth_radii_px 


'''
def np2ij_coords(np_coord):
    assert len(np_coord) == 2
    # TODO test it actually need to be flipped, similar to test for offsets
    # from outputs in extract_template w/ --save-ij-debug-info opt
    return (np_coord[1], np_coord[0])
#
'''


def test_single_template_knownoffset():
    if interactive:
        print('\nsingle_template_knownoffset')

    for frame_shape in frame_shapes:
        if interactive:
            print('TESTING WITH SOURCE FRAME SHAPE:', frame_shape)

        wellseparated_centers = get_wellseparated_centers(frame_shape)
        for center_point in wellseparated_centers[:2]:
            img_with_roi, _ = make_img_with_rois(frame_shape, [center_point])

            ij_centers, radii_px, _, _ = fit_circle_rois(tiff_fname, template_data,
                _um_per_pixel_xy=um_per_pixel_xy,
                avg=img_with_roi, min_neighbors=0,
                min_n_rois=1, max_n_rois=1,
                max_threshold_tries=1,
                threshold=0.999,
                debug=interactive, _packing_debug=interactive,
                multiscale=False
            )
            assert len(set(radii_px)) == 1
            if interactive:
                print(f'pixel radius of ROIs: {radii_px[0]:.2f}')

            try:
                assert len(ij_centers) == 1, f'{len(ij_centers)} > 1'
            except AssertionError:
                import ipdb; ipdb.set_trace()
            fit_center = tuple(ij_centers[0])

            #true_center = np2ij_coords(center_point)
            true_center = center_point

            if interactive:
                print('target:', true_center)
                print('fit:', fit_center)

            assert fit_center == true_center, f'{fit_center} != {true_center}'


def test_ordering_wellseparated_centers():
    if interactive:
        print('\nordering_wellseparated_centers')

    orig_intensity_scales = [0.5, 1]

    for frame_shape in frame_shapes:
        wellseparated_centers = get_wellseparated_centers(frame_shape)
        centers = wellseparated_centers[:2]

        for intensity_scales in (orig_intensity_scales,
            orig_intensity_scales[::-1]):

            img_with_rois, _ = make_img_with_rois(frame_shape, centers,
                intensity_scales=intensity_scales
            )
            ij_centers, radii_px, _, _ = fit_circle_rois(tiff_fname, template_data,
                _um_per_pixel_xy=um_per_pixel_xy,
                avg=img_with_rois, min_neighbors=0,
                min_n_rois=2, max_n_rois=2,
                max_threshold_tries=1,
                threshold=0.9,
                _packing_debug=interactive,
                multiscale=False
            )
            assert len(ij_centers) == len(centers)

            assert len(set(radii_px)) == 1
            if interactive:
                print(f'pixel radius of ROIs: {radii_px[0]:.2f}')

            # Should be highest scale first, and fit_circle_rois should
            # also have matches made first (better matches, b/c greedy)
            # at earlier indexes.
            centers_by_snr = np.array(centers)[
                np.argsort(intensity_scales)[::-1]
            ]
            for center_point, ij_center in zip(centers_by_snr, ij_centers):
                ij_center = tuple(ij_center)

                #ij_center_point = np2ij_coords(center_point)
                ij_center_point = center_point

                if interactive:
                    print('target:', ij_center_point)
                    print('fit:', ij_center)

                assert ij_center == tuple(ij_center_point)


def test_multiscale_packing():
    if interactive:
        print('\nmultiscale_packing')

    # should only really matter if multiscale_strategy='random'
    np.random.seed(1)
    for frame_shape in frame_shapes:
        centers = get_wellseparated_centers(frame_shape)
        templ_d_deltas= [-3, 0, 6]

        img_with_rois, float_roi_radii_px = make_img_with_rois(frame_shape,
            centers, template_dimension_changes=templ_d_deltas
        )

        ij_centers, radii_px, thresholds, ns_found = fit_circle_rois(tiff_fname,
            template_data,
            _um_per_pixel_xy=um_per_pixel_xy,
            avg=img_with_rois,
            min_neighbors=0,
            #min_n_rois=3,
            per_scale_min_n_rois=[1, 1, 1],
            per_scale_max_n_rois=[1, 1, 1],

            roi_diams_px=np.array(float_roi_radii_px) * 2,
            multiscale_strategy='one_order',

            # if this is some amount less than 0.5, issues in note below apply
            # not sure whether real data will have some val above which
            # i can also ignore that behavior...
            threshold=0.7,

            # TODO TODO TODO why are these values not causing the same behavior
            # as the values in the next group??? wrong assumptions?
            #threshold=0.12,
            # Taken from one run of fit_circle_rois w/ threshold=0.12,
            # which returned thresholds=[0.12, 0.35, 0.5].
            # This is the initial threshold divided by each fit thresholds.
            # TODO TODO how is it possible that this seems to change the
            # fraction of the each match images pixels above thresh to equal
            # values used w/ working thresholds below, yet the matches returned
            # are different???
            #match_value_weights=[1.0, 0.34285714285714286, 0.24],

            #
            #thresholds=np.array([0.12, 0.35,  0.5]),
            #

            #multiscale_strategy='one_order',
            #match_value_weights=[1.0, 0.87, 1.25],

            #multiscale_strategy='random', radii_px_ps=[0.2, 0.65, 0.15],
            #multiscale_strategy='random', radii_px_ps=[1/3, 1/3, 1/3],
            #multiscale_strategy='fixed_scale_order', scale_order=[2, 1, 0],
            max_threshold_tries=1,
            #max_threshold_tries=10,
            debug=interactive, _packing_debug=interactive
        )

        if interactive:
            print('centers:')
            print(ij_centers)
            print('radii of ROIs:', radii_px)

        centers = sorted(centers, key=dict(zip(centers, templ_d_deltas)).get)
        # cause np arrays are not hashable
        # (can't be used in dict key for next step)
        ij_centers = [tuple(x) for x in ij_centers]
        ij_centers = sorted(ij_centers, key=dict(zip(ij_centers, radii_px)).get)

        for true_center, fit_center in zip(centers, ij_centers):
            ij_true_center = true_center #np2ij_coords(true_center)
            assert ij_true_center == fit_center



# TODO TODO tests with ROIs that dont have the increase in intensity
# (in each corner) that the template has

# TODO TODO tests w/ ROIs close together, to get at circle packing
# (in cases where it should work)

# TODO TODO tests w/ ROIs that have different radii
# TODO + maybe also w/ different intensity scales on top of that

# TODO TODO TODO maybe tests to check for metrics that actually do work w/
# constant (or at least predictably-scaled) thresholds in the multi-scale
# matching context (or just read docs... but then still add test to verify)

# TODO TODO TODO manual check on ijroi writing by writing something that should
# be symmetric about the middle (write template to tiff / ij loadable format
# + roi over it. easier to see symmetry w/ small image)

# TODO TODO TODO check that best template match for my chosen metric is at
# template scale == ground truth scale (look for this behavior across all
# metrics if not)


# TODO move this, or a test like it, to a test module just for roi/mask stuff
# (do same in real code too...)
def test_ijrois2masks():
    w = 256
    h = 256
    frame_shape = (w, h)
    center = get_asymmetric_center(frame_shape)
    img, float_radii_px = make_img_with_rois(frame_shape, [center])

    # To manually make ImageJ ROI to test w/ ijrois2masks.
    '''
    u16_img = np.round((img - img.min()) / img.max()).astype(np.uint16)
    tiff_fname = f'{w}x{h}_asymmetric_center.tif'
    tifffile.imsave(tiff_fname, u16_img)
    import ipdb; ipdb.set_trace()
    '''

    roi_fname = f'{w}x{h}_asymmetric_center_{{}}.zip'
    # These both have two copies of the same ROI (just so I test ROI zip loading
    # fn, and stuff downstream of its output).
    oval_roiset_fname = roi_fname.format('ovals')
    poly_roiset_fname = roi_fname.format('polys')

    # TODO maybe also load singular ROIs and test some stuff (what?) downstream
    # of that

    # TODO maybe also test that all points in in polygon version are in what
    # loading oval rois returns + maybe that convex hulls are equivalent
    # (might not be true though, just b/c maybe poly i drew was not exactly
    # equal... could check they are centered though?)

    n_frames = 3
    movie = np.stack([img] * n_frames, axis=0)
    movie_t = np.stack([img.T] * n_frames, axis=0)
    n_rois = 2
    for roiset_fname in (oval_roiset_fname, poly_roiset_fname):
        ijrois = ijroi.read_roi_zip(roiset_fname)
        assert len(ijrois) == n_rois
        assert np.array_equal(ijrois[0][1], ijrois[1][1])
        masks = ijrois2masks(ijrois, frame_shape)
        # So last dim indexes mask #. I might want first index to do that...
        assert masks.shape[:2] == frame_shape
        assert masks.shape[-1] == n_rois

        '''
        fig = plt.figure()
        plt.imshow(img)
        fig = plt.figure()
        mask = masks[:,:,0]
        plt.imshow(mask)
        plt.show()
        '''

        traces = extract_traces_bool_masks(movie, masks, verbose=False)
        assert traces.shape[0] == n_frames
        assert traces.shape[1] == n_rois
        assert np.all(traces > 0)

        empty_traces = extract_traces_bool_masks(movie_t, masks, verbose=False)
        assert empty_traces.shape[0] == n_frames
        assert empty_traces.shape[1] == n_rois
        assert np.all(empty_traces == 0)

        masks_t = []
        for i in range(masks.shape[-1]):
            masks_t.append(masks[:,:,i].T)
        masks_t = np.stack(masks_t, axis=-1)
        empty_traces2 = extract_traces_bool_masks(movie, masks_t, verbose=False)
        assert empty_traces2.shape[0] == n_frames
        assert empty_traces2.shape[1] == n_rois
        assert np.all(empty_traces2 == 0)


def check_ijroi_circle_center_coords():
    center = (24, 8)
    # assuming template has shape of (16, 16) for now
    img_with_rois, float_radii_px = make_img_with_rois((32, 32), [center])
    # So integer center is appropriate for indexing arrays made w/
    # make_img_with_rois, but the coordinates seem transposed w.r.t. MPL
    # drawing coords (well, the first coordinate goes top-to-bottom, and the
    # second goes left-to-right)
    imin = img_with_rois.min()
    assert img_with_rois[center] > imin and img_with_rois[center[::-1]] == imin

    roi_file = 'check_corner.roi'
    radius = 8
    roi_repr = get_circle_ijroi_input(center, radius)
    print(f'Writing debuging ROI file to {roi_file}')
    with open(roi_file, 'wb') as f:
        ijroi.write_roi(roi_repr, f, name='0', roi_type=ijroi.RoiType.OVAL)

    cx = center[0]
    cy = center[1]
    rr = 1
    polygon = np.array([[cx - rr, cy - rr], [cx - rr, cy + rr],
        [cx + rr, cy - rr], [cx + rr, cy + rr]
    ])
    poly_roi_file = 'check_corner_polygon.roi'
    with open(poly_roi_file, 'wb') as f:
        ijroi.write_roi(polygon, f, name='0')

    with open(roi_file, 'rb') as f:
        points = ijroi.read_roi(f)

    with open(poly_roi_file, 'rb') as f:
        poly_points = ijroi.read_roi(f)

    # Since we now know these are the same, no point in plotting both.
    assert np.array_equal(poly_points, polygon)

    ij_center, ij_diam = ijroi.oval_points_center_diam(points)
    assert ij_diam == int(round(2 * radius))
    assert tuple(int(round(e)) for e in ij_center) == center

    fig, axs = plt.subplots(ncols=3)
    ax = axs[0]
    # I checked the .tiff and all the .roi files generated here, and they
    # all overlay in ImageJ (all in bottom left corner).
    # This is in contrast to the MPL stuff, which is all in top right corner.
    # (b/c X is conventionally left-to-right in MPL, but MPL also shows images
    # in usually math / numpy index notation where row comes first)
    # (row != x, convention-wise)
    ax.imshow(img_with_rois, cmap='gray')

    # Flipping these now to get them to display right.
    # TODO factor into a display coords fn?
    ax.plot(*np.flip(polygon, axis=1).T, color='green')
    ax.scatter(*np.flip(points, axis=1).T)
    ax.scatter(*center[::-1], marker='x', color='red')

    radius_px = ij_diam / 2
    roi_circle = plt.Circle((ij_center[0] - 0.5, ij_center[1] - 0.5)[::-1],
        radius_px, fill=False, color='r'
    )
    ax.add_artist(roi_circle)

    ax.set_title('All features should line up')

    img_tiff_path = 'check_corner.tif'
    print(f'Writing debugging TIFF of img to {img_tiff_path}')
    # Current type of float64 led to a TIFF unreadable by ImageJ.
    baselined = img_with_rois - img_with_rois.min()
    u16_img = (u16_max * baselined / baselined.max()).astype(np.uint16)
    tifffile.imsave(img_tiff_path, u16_img)

    ax = axs[1]
    draw_on = np.zeros_like(img_with_rois)
    # OK so row is the second index for cv2 drawing fns (circle center at least)
    # (since I need to flip them around)
    cv2.circle(draw_on, center[::-1], radius, 1, -1)
    ax.imshow(draw_on)
    ax.set_title('matches left')

    # TODO TODO revisit whether this has any implications for integer
    # offsets to drawing fns (can't give float inputs)
    # Checking cv2 circle drawing is also symmetric about middle as i expect
    ax = axs[2]
    # Not symmetric. Missing a pixel on the right.
    #draw_on = np.zeros((16, 16))
    #cv2.circle(draw_on, (8, 8), 8, 1, -1)
    draw_on = np.zeros((17, 17))
    cv2.circle(draw_on, (8, 8), 8, 1, -1)
    ax.imshow(draw_on)
    ax.set_title('symmetric')

    plt.show()



# TODO delete this after getting tests working (does pytest care if this is
# here?)
if __name__ == '__main__':
    interactive = True
    show_template = False
    show_generated_scenes = False

    test_ijrois2masks()

    #plot_template()

    '''
    if interactive:
        check_ijroi_circle_center_coords()

    test_single_template_knownoffset()
    test_ordering_wellseparated_centers()
    test_multiscale_packing()
    '''

    if interactive:
        plt.show()
#

