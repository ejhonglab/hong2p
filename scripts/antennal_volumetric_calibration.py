#!/usr/bin/env python3

# TODO at some point in the future, factor this functionality into
# populate_db.py, so it's run automatically

from os.path import join, split, exists
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import hong2p.util as u
from hong2p.thor import read_movie


# TODO maybe rename now that it also has antennal data...
gdf = u.mb_team_gsheet()
gdf = gdf[gdf.stimulus_data_file.apply(lambda x: '_vv-' in x) &
    gdf.attempt_analysis
]
# Assuming there are no non-calibration pickles on the same dates as 
# calibration experiments (that have their own trace pickles).
date_strs = [u.format_date(d) for d in gdf.date.unique()]

def write_trace_pickles():
    keys = ['date', 'fly_num', 'thorimage_dir']
    for ks, group_df in gdf.groupby(keys):
        data_dir = u.thorimage_dir(*ks)
        # TODO assert the dir exists?

        avg_tiff_fname = join(data_dir, 'avg.tif')
        if not exists(avg_tiff_fname):
            movie = read_movie(data_dir)
            avg = np.mean(movie, axis=0)

            # TODO factor into write_tiff / provide some other fn for this?
            # TODO use full range of dtype or try to keep scale comparable
            # across data? (check that # of unique values doesn't decrease at
            # least?)
            # TODO rounding before astype casting necessary?
            avg = np.round(avg).astype(np.uint16)

            print(f'writing average to {avg_tiff_fname}')
            u.write_tiff(avg_tiff_fname, avg)

            import ipdb; ipdb.set_trace()

            # TODO TODO also write rois from threshold (+ dilation)?

            print('You must save ImageJ ROIs to rois.zip in same folder before'
                'further analysis on this recording will be done.'
            )
            continue

        # TODO TODO TODO check trace pickle doesn't already exist
        # (unless maybe a flag to regenerate them...)
        import ipdb; ipdb.set_trace()

        fly_dir, thorimage_id = split(data_dir)
        tiff = join(fly_dir, 'tif_stacks', f'{thorimage_id}.tif')
        # TODO TODO how were these tiffs generated? manually in imagej? or using
        # read_movie / write_tiff?
        assert exists(tiff)
        # Calling this for its side effect of writing the "trace" pickle for the
        # current recording. It does interact with the db though and that might
        # be a problem...
        data = db.load_recording(tiff)


def process_trace_pickles():
    trace_pickle_dir = join(u.analysis_output_root(), 'trace_pickles')
    # TODO TODO TODO if enumerating thorimage dirs above from a list of dates,
    # use same dates here to glob the pickles
    pickles = [
        glob.glob(join(trace_pickle_dir, f'*_{s}*.p')) for s in date_strs
    ]
    print(pickles)
    import ipdb; ipdb.set_trace()
    #pickles = glob.glob(join(trace_pickle_dir, '*_2020-03-09*.p'))
    dfs = []
    for p in pickles:
        pdf = pd.read_pickle(p)
        dfs.append(pdf)
    df = pd.concat(dfs)

    # This is to exclude landmark odors (those intended to just activate
    # a single glomerulus).
    df = df[~ df.original_name1.isin(('2,3-butanedione', 'ammonia',
        'ethyl acetate', 'trans-2-hexenal', 'pfo', '2-butanone'))
    ]

    keys = ['fly_num', 'original_name1', 'log10_conc_vv1']
    dff_max = df.groupby(keys).df_over_f.max().reset_index()
    import ipdb; ipdb.set_trace()

    #'''
    g = sns.FacetGrid(dff_max, row='original_name1', hue='fly_num', sharey=False)
    g.map(plt.plot, 'log10_conc_vv1', 'df_over_f')
    # TODO savefig
    plt.show()

    return df


def main():
    # TODO probably just provide dates and have something enumerate thorimage
    # dirs from that?
    #write_trace_pickles()
    df = process_trace_pickles()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

