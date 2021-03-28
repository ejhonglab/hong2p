#!/usr/bin/env python3

"""
Loads ImageJ ROIs, gets subsets of movie they contain, and averages to form
a template for template matching.
"""

from os.path import join, split, getmtime, exists
import glob
from datetime import datetime
import warnings
import pickle
import argparse

import numpy as np
import pandas as pd
import cv2
import tifffile
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans

import ijroi
import hong2p.util as u
# TODO test whether replacing w/ 'from hong2p import db' works (w/ or w/o
# changing __init__.py)
import hong2p.db as db
import hong2p.thor as thor


def print_bad_keys(mask):
    print(mask.index[mask].to_frame().to_string(index=False))


def make_template(image_list, avg_all_flips_and_rotations=True):
    extents = [img.shape for img in image_list]
    x_extents = [x for x, _ in extents]
    y_extents = [y for _, y in extents]

    template_extent = int(round(np.median(x_extents + y_extents)))
    template_shape = (template_extent, template_extent)

    resized = [cv2.resize(im, template_shape) for im in image_list]

    if avg_all_flips_and_rotations:
        # TODO would need to modify if i were to support 3d (x,y,z)
        # frames here, b/c np.rot90 only rotates first two dims,
        # and we'd want rotations of all three
        to_avg = []
        for n_rotations in range(4):
            to_avg.extend([np.rot90(im, k=n_rotations) for im in resized])
        
        # No need to also add up-down filps, because fliplr on all the rotations
        # is redundant with that.
        flips = []
        for im in to_avg:
            flips.append(np.fliplr(im))
        to_avg.extend(flips)

    else:
        to_avg = resized

    return np.mean(to_avg, axis=0)


def try_all_template_matching_methods(scene, template):
    # TM_CCOEFF[_NORMED] is the only one that really looks promising
    # (at least how i'm constructing the template now / normalizing)

    # All the 6 methods for comparison in a list
    methods = [
        'cv2.TM_CCOEFF',
        'cv2.TM_CCOEFF_NORMED',
        'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED',
        'cv2.TM_SQDIFF',
        'cv2.TM_SQDIFF_NORMED'
    ]
    normed_template = u.baselined_normed_u8(template)
    for method in methods:
        res = cv2.matchTemplate(scene, normed_template, method)
        u.imshow(res, method)


def unconstrained_roi_finding(match_image, radius, d, draw_on, threshold=0.8):
    """Plots all template positions over match threshold.

    For checking other 
    """
    above_thresh = np.where(match_image >= threshold)

    scene = u.baselined_normed_u8(draw_on)
    # upsampling just so cv2 drawing functions look better
    ups = 4
    scene = cv2.resize(scene, tuple([ups * x for x in scene.shape]))
    w = ups * d
    h = ups * d
    radius = int(round(ups * radius))

    scene_bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

    for pt in zip(*above_thresh[::-1]):
        pt = (pt[0] * ups, pt[1] * ups)
        cv2.rectangle(scene_bgr, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        center = (pt[0] + int(round(w / 2)), pt[1] + int(round(h / 2)))
        cv2.circle(scene_bgr, center, radius, (255,0,0), 2)

    u.imshow(scene_bgr, 'unconstrained template matching maxima')


def enforce_nonneg_int(int_arg):
    # https://stackoverflow.com/questions/14117415
    try:
        as_int = int(int_arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{int_arg} could not be parsed as an '
            'integer'
        )

    if as_int < 0:
        raise argparse.ArgumentTypeError(f'{int_arg} was not non-negative')

    return as_int


def main():
    parser = argparse.ArgumentParser(description='Make template from TIFFs '
        'and manual ImageJ ROIs on disk, to use for automatic ROI finding.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--remake', action='store_true',
        help='If template already exists (from prior run of this script), '
        'remake it anyway. No effect if template does not already exist.'
    )

    parser.add_argument('--rescale-to', choices=['largest', 'most-common'],
        default='largest', help='Rescales all average frames to one of these '
        'frame sizes, computed over all inputs to template making. '
        'Also scales input ImageJ ROIs into these coordinates.'
    )
    # TODO maybe specify this as a percentage of cell diameter, which will
    # then be rounded?
    parser.add_argument('--margin', type=enforce_nonneg_int, default=2,
        help='How many pixels beyond input ROIs to include in template. '
        'Non-negative integer.'
    )
    parser.add_argument('--no-dilate-roi-mask', action='store_false',
        help='Disables dilation of input ROI mask. Dilation includes features '
        'that are neighboring the edge of the ROI, but not exactly within it. '
        "This has no affect if --outside-roi-mask='leave'"
    )
    parser.add_argument('--outside-roi-mask',
        choices=['zero', 'min', 'leave'], default='min',
        help='What to do with region outside mask ROI before averaging.'
    )
    parser.add_argument('--debug-plots', action='store_true',
        help='Makes additional plots, that may be useful for debugging.'
    )
    parser.add_argument('--debug-each-movie', action='store_true',
        help='Shows each input cell and templates created within each movie, '
        'in case this reveals movies where template creation is misbehaved.'
    )
    parser.add_argument('--show-all-input-cells', action='store_true',
        help='Shows a grid of all input cells.'
    )
    parser.add_argument('--dry-run', action='store_true',
        help='Does not save template to pickle. Use only for testing, without '
        'having to worry about overwriting a good template.'
    )

    parser.add_argument('--no-test', action='store_true',
        help='Do not test automatic ROI finding against the ground truth data, '
        'such as that used to generate the template.'
    )
    # TODO TODO should this maybe also influence input to template making?
    # (or is this already the behavior?) separate flag at least?
    parser.add_argument('--exclude-edited-auto-rois', action='store_true',
        help='Do not test ROI finding on any ROIs that even started out as '
        'automatically generated ROIs. To deal with bias of initial ROIs, '
        'for comparison to a potentially-better ground truth.'
    )
    parser.add_argument('--only-test-on-first', action='store_true',
        help='Only tests ROI matching on (arbitrary) first ground truth TIFF + '
        'ROIs, for faster-yet-lower-quality testing.'
    )

    parser.add_argument('--save-ij-debug-info', action='store_true',
        help='In current directory, saves template to TIFF (template.tif) '
        'and ImageJ ROI to template.roi, for inspection / troubleshooting.'
    )
    args = parser.parse_args()

    # Includes this many extra pixels on each side, for each dimension,
    # beyond ROI bounds.
    margin = args.margin
    debug_plots = args.debug_plots
    debug_each_movie = args.debug_each_movie
    show_cell_images = args.show_all_input_cells

    np.set_printoptions(precision=3)

    template_cache = u.template_data_file()
    if args.remake:
        template_data = None
    else:
        template_data = u.load_template_data()

    if template_data is not None:
        print(f'Loaded template data from {template_cache}')
        template = template_data['template']
        margin = template_data['margin']
        mean_cell_diam_um = template_data['mean_cell_diam_um']
        frame_shape = template_data['frame_shape']

        kmeans_k2cluster_cell_diams = \
            template_data['kmeans_k2cluster_cell_diams']

        df = template_data['df']
    else:
        # TODO TODO probably factor this whole bit about local analysis checking
        # against db contents into util + use in kc_mix_analysis
        print('Making template...')

        conn = db.get_db_conn()
        df = pd.read_sql_query('SELECT * FROM analysis_runs '
            'WHERE ijroi_file_path IS NOT NULL', conn)
        assert df.input_filename.notnull().all()

        # TODO actually test when there are multiple inputs
        # If the same inputs have multiple ijroi_file_paths,
        # only take the most recent (run_at should contain roi file mtime).
        df = df.loc[df.groupby('input_filename').run_at.idxmax()]

        # No real way to go from filenames under trace_pickles to imagej roi set
        # that was used, since mtime may have changed, but...
        latest_on_disk = u.latest_trace_pickles()

        # Getting keys from input_filename to not have to merge w/ other tables.
        # (uh... was there some other reason?)
        df = pd.concat([df, df.input_filename.apply(u.tiff_filename2keys)],
            axis=1
        )
        df.set_index(latest_on_disk.index.names, inplace=True)

        # Indicator puts 'both'/'left_only'/'right_only' in '_merge' column.
        merge_output = df.merge(latest_on_disk, how='outer', left_index=True,
            right_index=True, indicator=True, suffixes=('_db', '_on_disk')
        )
        # (if it's in database, we expect trace pickle outputs on disk)
        assert not (merge_output._merge == 'left_only').any()
        not_in_db = merge_output._merge == 'right_only'

        # TODO test this case

        if not_in_db.any():
            print_bad_keys(not_in_db)
            raise ValueError('keys above are on disk but not in db. re-run '
                'their analysis and upload.')

        # TODO test this case

        # Since filenames only include timestamps truncated to minute precision.
        merge_output.run_at_db = merge_output.run_at_db.apply(
            lambda x: x.floor('min')
        )
        newer_on_disk = merge_output.run_at_db < merge_output.run_at_on_disk
        if newer_on_disk.any():
            print_bad_keys(newer_on_disk)
            raise ValueError('keys above have newer outputs on disk than in db.'
                ' re-run their analysis and upload.')
        # end part that i might want to factor into util + use in
        # kc_mix_analysis

        # TODO maybe normalize movies or something first to better avg across?
        # local equalization to keep template more constant across cells w/ diff
        # amounts of contrast?
        all_avg_cells = []
        all_cell_diams_um = []

        # until i know for sure input avg frame shape does not matter (much)
        # here... see note below
        movie_xy_shape_counts = dict()

        row_index2frame_shape = dict()
        row_index2thorimage_xmlroots = dict()
        for row in df.itertuples():
            thorimage_dir = u.thorimage_dir(*row.Index)
            xmlroot = thor.get_thorimage_xmlroot(thorimage_dir)
            row_index2thorimage_xmlroots[row.Index] = xmlroot

            xy_shape, _, _ = thor.get_thorimage_dims_xml(xmlroot)
            row_index2frame_shape[row.Index] = xy_shape

            if xy_shape in movie_xy_shape_counts:
                movie_xy_shape_counts[xy_shape] += 1
            else:
                movie_xy_shape_counts[xy_shape] = 1

            # Necessary to at least assume all aspect ratios are the same, if
            # we want to nicely rescale everything to one scale (while 
            # preserving aspect ratio) (right?).
            # For now, for simplicity, just checking everything is square.
            xs, ys = xy_shape
            if xs != ys:
                raise NotImplementedError('assuming all images are square')

        if args.rescale_to == 'largest':
            max_n_pixels = 0 
            maximizing_shape = None
            for shape in movie_xy_shape_counts.keys():
                n_pixels = np.prod(shape)
                if n_pixels > max_n_pixels:
                    maximizing_shape = shape
                    max_n_pixels = n_pixels
            assert maximizing_shape is not None and max_n_pixels > 0
            print(f'Largest frame shape: {maximizing_shape}')
            frame_shape = maximizing_shape

        elif args.rescale_to == 'most-common':
            most_common_frame_shape = None
            max_shape_count = 0
            multiple_with_max_count = False
            for shape, count in movie_xy_shape_counts.items():
                if count > max_shape_count:
                    most_common_frame_shape = shape
                    max_shape_count = count
                    multiple_with_max_count = False

                elif count == max_shape_count:
                    multiple_with_max_count = True

            assert max_shape_count > 0 and most_common_frame_shape is not None
            assert not multiple_with_max_count, \
                'expected one frame shape to be most common'

            print(f'Most common frame shape: {most_common_frame_shape}')
            frame_shape = most_common_frame_shape

        else:
            raise NotImplementedError('invalid value of rescale_to')
        del movie_xy_shape_counts

        for row in df.itertuples():
            xy_shape = row_index2frame_shape[row.Index]

            xmlroot = row_index2thorimage_xmlroots[row.Index]
            um_per_pixel_xy = thor.get_thorimage_pixelsize_xml(xmlroot)

            ijroi_file = row.ijroi_file_path
            curr_mtime = datetime.fromtimestamp(getmtime(ijroi_file))
            # TODO test this case
            if row.run_at < curr_mtime:
                warnings.warn(f'ImageJ ROIs in {ijroi_file} seem to have been '
                    'changed since last analysis. Consider re-analyzing with '
                    'current ROIs.'
                )

            ijrois = ijroi.read_roi_zip(ijroi_file)

            tif = row.input_filename
            print(f'{tif}', flush=True)
            movie = tifffile.imread(tif)
            assert len(movie.shape) == 3, 'only (t,x,y) stacks supported'
            avg = movie.mean(axis=0)
            assert avg.shape == xy_shape, \
                f'ThorImage metadata on frame size was wrong for {tif}'

            scale_factor = None
            if xy_shape != frame_shape:
                # Already checked frames are square in previous pass.
                orig_frame_d = xy_shape[0]

                avg = cv2.resize(avg, frame_shape)
                assert avg.shape == frame_shape

                new_frame_d = frame_shape[0]

                scale_factor = new_frame_d / orig_frame_d
                um_per_pixel_xy /= scale_factor

            avg_cells = []
            for _, roi in ijrois:
                if scale_factor is not None:
                    # TODO TODO TODO if ijroi coordinates don't start at 0, need
                    # to use some other formula (edge should still be the edge
                    # after scaling). check w/ docs or some test case!
                    roi = np.round(roi * scale_factor).astype(roi.dtype)

                cropped, (x_bounds, y_bounds) = u.crop_to_coord_bbox(avg, roi,
                    margin=margin
                )
                if debug_plots:
                    plt.imshow(cropped)

                if args.outside_roi_mask != 'leave':
                    # Shifting contour to start from 0 in both dims, to not need
                    # a mask as large as the whole frame.
                    roi[:,0] = roi[:,0] - x_bounds[0]
                    roi[:,1] = roi[:,1] - y_bounds[0]
                    mask = u.contour2mask(roi, cropped.shape)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(mask)

                    if not args.no_dilate_roi_mask:
                        dilation_kernel = np.ones((3,3), np.uint8)
                        mask = cv2.dilate(mask.astype(np.uint8),
                            dilation_kernel, iterations=1
                        )
                    else:
                        mask = mask.astype(np.uint8)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(mask)

                    masked = cropped * mask
                    if args.outside_roi_mask == 'min':
                        masked = masked + (~ mask * np.min(cropped))
                    else:
                        assert args.outside_roi_mask == 'zero'

                    avg_cells.append(masked)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(masked)
                        plt.show()
                else:
                    avg_cells.append(cropped)

            # TODO if i'm going to store a distribution of cell extents in um
            # (so that i can cluster / subdivide that, for multi-scale
            # matching), should i also try to make a template specific for each
            # size? (bigger cells tend to be brighter, etc, to the extent that
            # pixel intensity scale matters to a given matching metric...)

            # "cell diameter" here is defined as the average of X and Y extents.
            # (since input ROIs are not necessarily something like radially
            # symmetric)
            cell_diams_pixels = (np.array([np.array(x.shape)
                for x in avg_cells]) - 2 * margin).mean(axis=1)

            cell_diams_um = cell_diams_pixels * um_per_pixel_xy
            all_cell_diams_um.extend(list(cell_diams_um))

            all_avg_cells.extend(avg_cells)

            if debug_each_movie:
                title = \
                    f'{str(row.Index[0])[:10]}/{row.Index[1]}/{row.Index[2]}'

                u.imshow(movie.mean(axis=0), f'{title} avg')

                fig = u.image_grid(avg_cells)
                fig.suptitle(title)

                template = make_template(avg_cells)
                u.imshow(template,
                    f'{title} template (averaged across rotations and flips)'
                )
                plt.show()

        mean_cell_diam_um = np.mean(all_cell_diams_um)

        if show_cell_images:
            fig = u.image_grid(all_avg_cells)
            title = 'All cells, across input recordings'
            fig.suptitle(title)

        template = make_template(all_avg_cells)
        u.imshow(template, 'Template')

        # This section is to assist in designing multi-scale template matching,
        # or see when it might/might-not be worth it.
        fig, ax = plt.subplots()
        ax.hist(all_cell_diams_um, bins=35)
        ax.set_title('Distribution of cell sizes')
        ax.set_xlabel('Cell extent (mean of bbox height and width) (um)')
        ax.set_ylabel('Number of cells')

        print('\nRunning k-means on cell extents (um), to get sizes for '
            'multi-scale matching:'
        )
        # Making each row an observation (so reshaping to a column vector).
        obs = np.expand_dims(all_cell_diams_um, -1)
        kmeans_k2cluster_cell_diams = dict()
        for k in range(2,5):
            print(f'k={k}')
            # Not whitening first because that would seem to only be beneficial
            # if there were multiple features (here, there is just the one)
            # (docs recommend whitening).
            codebook, distortion = kmeans(obs, k)
            centroids = codebook.squeeze()
            kmeans_k2cluster_cell_diams[k] = centroids
            print(f'centroids: {centroids}')
            print(f'distortion: {distortion:.3f}')
        print('')
        
        # End section related to multi-scale matching.

        if not args.dry_run:
            print(f'writing template data to {template_cache}')
            with open(template_cache, 'wb') as f:
                data = {
                    'template': template,
                    'margin': margin,
                    'mean_cell_diam_um': mean_cell_diam_um,
                    'df': df,
                    'frame_shape': frame_shape,
                    'frame_shape_from': args.rescale_to,
                    'all_cell_diams_um': all_cell_diams_um,
                    'kmeans_k2cluster_cell_diams': kmeans_k2cluster_cell_diams,
                    # TODO how costly would it be, space-wise, to always save
                    # all_avg_cells? worth it? (could then make diff templates
                    # for each size, or otherwise cluster, post-hoc)
                }
                pickle.dump(data, f)

    if args.save_ij_debug_info:
        # Circle is symmetric about template, so my ijroi writing is
        # working correctly.
        template_tiff_path = 'template.tif'
        print(f'Writing debugging TIFF of template to {template_tiff_path}')
        # Current type of float64 led to a TIFF unreadable by ImageJ.
        u16_max = 2**16 - 1
        baselined = template - template.min()
        u16_template = (u16_max * baselined / baselined.max()
            ).astype(np.uint16)
        tifffile.imsave(template_tiff_path, u16_template)

        margin = 0
        center = (template.shape[0] // 2,) * 2
        template_cell_diam_px = template.shape[0] - 2 * margin
        template_cell_radius_px = template_cell_diam_px / 2
        radius = int(round(template_cell_radius_px))
        ijroi_repr = u.get_circle_ijroi_input(center, radius)

        template_roi_path = 'template.roi'
        with open(template_roi_path, 'wb') as f:
            print(f'Writing debugging ImageJ ROI to {template_roi_path}')
            ijroi.write_roi(ijroi_repr, f, name='0',
                roi_type=ijroi.RoiType.OVAL
            )

        print('These markers should be centered about the template')
        with open(template_roi_path, 'rb') as f:
            roi_points = ijroi.read_roi(f)
        fig, ax = plt.subplots()
        ax.imshow(template)
        # The - 0.5s here are a correctly purely for matplotlib display
        # purposes.
        roi_circle = plt.Circle((center[0] - 0.5, center[1] - 0.5), radius,
            fill=False, color='r'
        )
        ax.add_artist(roi_circle)
        ax.scatter(*roi_points.T)
        plt.show()

    # TODO TODO some kind of "train" / test split?
    # (or maybe only if actually doing something complicated)

    # TODO store performance of fitting methods + params + template data
    # so it can be compared to code after any changes, without having to re-run
    # with old settings?
    # TODO related: some automatic backup of previous template data, rather than
    # overwriting it (above)?

    if not args.no_test:
        # If this is False, ROIs that were initially autogenerated, but have a
        # newer mtime, will still be used.
        exclude_even_initially_auto_rois = args.exclude_edited_auto_rois
        test_on_all = not args.only_test_on_first

        test_tifs = df.input_filename.unique()
        if not test_on_all:
            test_tifs = [test_tifs[0]]

        print('\nTesting...')
        total_center_matching_costs = []
        for tif in test_tifs:
            print(tif)
            tif_rows = df.input_filename == tif
            assert tif_rows.sum() == 1
            row = df[tif_rows].iloc[0]

            ijroi_file = row.ijroi_file_path
            auto_md_fname = u.autoroi_metadata_filename(ijroi_file)
            if exists(auto_md_fname):
                if exclude_even_initially_auto_rois:
                    print('skipping', tif, 'because ROIs (at least initially) '
                        'auto generated'
                    )
                    continue

                with open(auto_md_fname, 'rb') as f:
                    data = pickle.load(f)
                    saved_mtime = data['mtime']

                curr_mtime = getmtime(ijroi_file)
                if curr_mtime == saved_mtime:
                    print('skipping', tif, 'because ROIs have not been changed '
                        'since being autogenerated')
                    continue
                else:
                    warnings.warn('using initially autogenerated ROIs, b/c '
                        'mtime indicates they were modified'
                    )

            movie = tifffile.imread(tif)
            avg = movie.mean(axis=0)

            centers, radii_px, _, _ = u.fit_circle_rois(tif, template_data,
                avg=avg
            )
            # TODO delete
            #plt.show()
            #import ipdb; ipdb.set_trace()
            #

            draw_on = avg
            # TODO some other statistic of radii_px to use?
            # let this vary for each roi? way to handle that?
            # TODO TODO allow passing same-length costs for either centers
            # inputs to correspond_rois (=radii)
            max_cost = np.mean(radii_px)
            title = u.tiff_title(tif)

            center_diams = [
                ijroi.oval_points_center_diam(roi, assert_circular=False)
                for _, roi in ijroi.read_roi_zip(ijroi_file)
            ]
            true_centers = np.stack([c for c, _ in center_diams])

            lr_matches, unmatched_left, unmatched_right, total_cost, _ = \
                u.correspond_rois(centers, true_centers, max_cost=max_cost,
                left_name='Automatic', right_name='Manual', draw_on=draw_on,
                title=title, show=True
            )

            print('center matching cost: {:.2f}'.format(total_cost))
            total_center_matching_costs.append(total_cost)

        print('\nAverage center matching cost: {:.2f}'.format(
            np.mean(total_center_matching_costs)))

    plt.show()
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

