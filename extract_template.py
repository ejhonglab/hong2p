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

import numpy as np
import pandas as pd
import cv2
import tifffile
import matplotlib.pyplot as plt

import ijroi
import hong2p.util as u


# TODO factor into util + use in kc_mix_analysis
def latest_trace_pickles():
    """Returns (date, fly, id) indexed DataFrame w/ filename and timestamp cols.

    Only returns rows for filenames that had the latest timestamp for the
    combination of index values.
    """
    def vars_from_filename(tp_path):
        final_part = split(tp_path)[1][:-2]

        # Note that we have lost any more precise time resolution, so an 
        # exact search for this timestamp in database would fail.
        n_time_chars = len('YYYYMMDD_HHMM')
        run_at = pd.Timestamp(datetime.strptime(final_part[:n_time_chars],
            '%Y%m%d_%H%M'))

        parts = final_part.split('_')[2:]
        date = pd.Timestamp(datetime.strptime(parts[0], u.date_fmt_str))
        fly_num = int(parts[1])
        thorimage_id = '_'.join(parts[2:])
        return date, fly_num, thorimage_id, run_at, tp_path

    keys = ['date', 'fly_num', 'thorimage_id']
    tp_root = join(u.analysis_output_root(), 'trace_pickles')
    tp_data = [vars_from_filename(f) for f in glob.glob(join(tp_root, '*.p'))]

    df = pd.DataFrame(columns=keys + ['run_at', 'trace_pickle_path'],
        data=tp_data
    )

    # TODO TODO check this when there are actually duplicate outputs
    latest = df.groupby(keys).run_at.idxmax()
    df.drop(index=df.index.difference(latest), inplace=True)
    #

    df.set_index(keys, inplace=True)
    return df


def print_bad_keys(mask):
    print(mask.index[mask].to_frame().to_string(index=False))


# maybe set avg_all_rotations back to False
# TODO though compare quality of matches from template computed that way to that
# without
def make_template(image_list, avg_all_rotations=True):
    extents = [img.shape for img in image_list]
    x_extents = [x for x, _ in extents]
    y_extents = [y for _, y in extents]

    # TODO should i just pick one # from both to make sure it's always
    # symmetric? (i guess then i could also include each 90-degree rotation
    # of each image into the average, to make the template slightly rotation
    # invariant?) (actually do that now that it's symmetric?)
    template_extent = int(round(np.median(x_extents + y_extents)))
    template_shape = (template_extent, template_extent)

    resized = [cv2.resize(im, template_shape) for im in image_list]

    if avg_all_rotations:
        # TODO would need to modify if i were to support 3d (x,y,z)
        # frames here, b/c np.rot90 only rotates first two dims,
        # and we'd want rotations of all three
        to_avg = []
        for n_rotations in range(4):
            to_avg.extend([np.rot90(im, k=n_rotations) for im in resized])
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
    for method in methods:
        res = cv2.matchTemplate(scene, normed_template, method)
        u.imshow(res, method)


def unconstrained_roi_finding(match_image, radius, d, draw_on, threshold=0.8):
    """Plots all template positions over match threshold.

    For checking other 
    """
    above_thresh = np.where(match_image >= threshold)

    scene = u.normed_u8(draw_on)
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


def main():
    ignore_existing = False
    template_cache = u.template_data_file()
    if ignore_existing:
        template_data = None
    else:
        template_data = u.template_data()

    if template_data is not None:
        print(f'Loaded template data from {template_cache}')
        template = template_data['template']
        margin = template_data['margin']
        mean_cell_extent_um = template_data['mean_cell_extent_um']
        df = template_data['df']
    else:
        # TODO TODO probably factor this whole bit about local analysis checking
        # against db contents into util + use in kc_mix_analysis
        print('Making template...')

        conn = u.get_db_conn()
        df = pd.read_sql_query('SELECT * FROM analysis_runs '
            'WHERE ijroi_file_path IS NOT NULL', conn)
        assert df.input_filename.notnull().all()

        # TODO actually test when there are multiple inputs
        # If the same inputs have multiple ijroi_file_paths,
        # only take the most recent (run_at should contain roi file mtime).
        df = df.loc[df.groupby('input_filename').run_at.idxmax()]

        # No real way to go from filenames under trace_pickles to imagej roi set
        # that was used, since mtime may have changed, but...
        latest_on_disk = latest_trace_pickles()

        # Getting keys from input_filename to not have to merge w/ other tables.
        # (uh... was there some other reason?)
        df = pd.concat([df, df.input_filename.apply(u.tiff_filename2keys)],
            axis=1)
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

        # Includes this many extra pixels on each side, for each dimension,
        # beyond ROI bounds.
        margin = 2
        zero_outside_roi = False
        dilation_kernel = np.ones((3,3), np.uint8)
        debug_plots = False
        debug_each_movie = False
        show_cell_images = False

        # TODO maybe normalize movies or something first to better avg across?
        # local equalization to keep template more constant across cells w/ diff
        # amounts of contrast?
        all_avg_cells = []
        cell_extents_um = []

        for row in df.itertuples():
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

            avg_cells = []
            for _, roi in ijrois:
                cropped, (x_bounds, y_bounds) = u.crop_to_coord_bbox(avg, roi,
                    margin=margin)

                if debug_plots:
                    plt.imshow(cropped)

                if zero_outside_roi:
                    # Shifting contour to start from 0 in both dims, to not need
                    # a mask as large as the whole frame.
                    roi[:,0] = roi[:,0] - x_bounds[0]
                    roi[:,1] = roi[:,1] - y_bounds[0]
                    mask = u.contour2mask(roi, cropped.shape)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(mask)

                    mask = cv2.dilate(mask.astype(np.uint8), dilation_kernel,
                        iterations=1)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(mask)

                    masked = cropped * mask
                    avg_cells.append(masked)

                    if debug_plots:
                        plt.figure()
                        plt.imshow(masked)
                        plt.show()
                else:
                    avg_cells.append(cropped)

            thorimage_dir = u.thorimage_dir(*row.Index)
            xmlroot = u.get_thorimage_xmlroot(thorimage_dir)
            um_per_pixel_xy = u.get_thorimage_pixelsize_xml(xmlroot)

            cell_extent = (np.median([np.array(x.shape) for x in avg_cells]) -
                2 * margin)
            cell_extent_um = cell_extent * um_per_pixel_xy
            cell_extents_um.append(cell_extent_um)

            all_avg_cells.extend(avg_cells)

            if debug_each_movie:
                title = \
                    f'{str(row.Index[0])[:10]}/{row.Index[1]}/{row.Index[2]}'

                u.imshow(movie.mean(axis=0), f'{title} avg')

                fig = u.image_grid(avg_cells)
                fig.suptitle(title)

                #template = make_template(avg_cells, avg_all_rotations=False)
                #u.imshow(template, f'{title} template')

                template = make_template(avg_cells, avg_all_rotations=True)
                u.imshow(template,
                    f'{title} template (averaged across rotations)')

                plt.show()

        mean_cell_extent_um = np.mean(cell_extents_um)

        if show_cell_images:
            fig = u.image_grid(all_avg_cells)
            title = 'All cells, across input recordings'
            fig.suptitle(title)

        template = make_template(all_avg_cells)
        u.imshow(template, 'Template')

        print(f'writing template data to {template_cache}')
        with open(template_cache, 'wb') as f:
            data = {
                'template': template,
                'margin': margin,
                'mean_cell_extent_um': mean_cell_extent_um,
                'df': df
            }
            pickle.dump(data, f)

    # TODO TODO some kind of "train" / test split?
    # (or maybe only if actually doing something complicated)

    # If this is False, ROIs that were initially autogenerated, but have a newer
    # mtime, will still be used.
    exclude_even_initially_auto_rois = False
    test_on_all = True

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
                print('skipping', tif, 'because ROIs (at least initially) auto'
                    'generated')
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
                warnings.warn('using initially autogenerated ROIs, b/c mtime '
                    'indicates they were modified')

        # TODO kind of confused about why a transpose seems necessary here
        # , though it doesn't seem i used one in template-making portion above.
        # b/c similar diff wrt opencv coords? other reason?

        ijrois = [np.flip(roi, axis=1)
            for _, roi in ijroi.read_roi_zip(ijroi_file)]

        movie = tifffile.imread(tif)
        avg = movie.mean(axis=0)
        centers, radius = u.fit_circle_rois(tif, template, margin,
            mean_cell_extent_um, avg=avg
        )

        draw_on = avg
        max_cost = radius
        title = u.tiff_title(tif)

        lr_matches, unmatched_left, unmatched_right, total_cost, _ = \
            u.correspond_rois(centers, ijrois, max_cost=max_cost,
            left_name='Automatic', right_name='Manual', draw_on=draw_on,
            title=title)

        print('center matching cost: {:.2f}'.format(total_cost))
        total_center_matching_costs.append(total_cost)

    print('\nAverage center matching cost: {:.2f}'.format(
        np.mean(total_center_matching_costs)))

    plt.show()
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
