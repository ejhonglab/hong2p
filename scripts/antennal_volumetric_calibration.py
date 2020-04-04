#!/usr/bin/env python3

# TODO at some point in the future, factor this functionality into
# populate_db.py, so it's run automatically

from os.path import join, split, exists
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hong2p.util as u


def main():
    '''
    keys_list = [
        ('2020-03-09', 1, 'fn_007'),
        ('2020-03-09', 1, 'fn_008'),
        ('2020-03-09', 1, 'fn_009'),
        ('2020-03-09', 2, 'fn'),
        ('2020-03-09', 2, 'fn_001'),
        ('2020-03-09', 2, 'fn_002'),
    ]
    for ks in keys_list:
        data_dir = u.thorimage_dir(*ks)

        avg_tiff_fname = join(data_dir, 'avg.tif')
        if not exists(avg_tiff_fname):
            movie = u.read_movie(data_dir)
            avg = np.mean(movie, axis=0)

            # TODO factor into write_tiff / provide some other fn for this?
            # TODO use full range of dtype or try to keep scale comparable
            # across data? (check that # of unique values doesn't decrease at
            # least?)
            # TODO rounding before astype casting necessary?
            avg = np.round(avg).astype(np.uint16)

            print(f'writing average to {avg_tiff_fname}')
            u.write_tiff(avg_tiff_fname, avg)

            print('You must save ImageJ ROIs to rois.zip in same folder before'
                'further analysis on this recording will be done.'
            )
            continue

        fly_dir, thorimage_id = split(data_dir)
        tiff = join(fly_dir, 'tif_stacks', f'{thorimage_id}.tif')
        assert exists(tiff)
        data = u.load_recording(tiff)
    '''
    trace_pickle_dir = '/mnt/nas/mb_team/analysis_output/trace_pickles'
    pickles = glob.glob(join(trace_pickle_dir, '*_2020-03-09*.p'))
    dfs = []
    for p in pickles:
        pdf = pd.read_pickle(p)
        dfs.append(pdf)

    df = pd.concat(dfs)

    df = df[~ df.original_name1.isin(('2,3-butanedione', 'ammonia',
        'ethyl acetate', 'trans-2-hexenal', 'pfo', '2-butanone'))
    ]

    keys = ['fly_num', 'original_name1', 'log10_conc_vv1']
    dff_max = df.groupby(keys).df_over_f.max().reset_index()

    '''
    g = sns.FacetGrid(dff_max, row='original_name1', hue='fly_num', sharey=False)
    g.map(plt.plot, 'log10_conc_vv1', 'df_over_f')
    plt.show()
    '''
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

