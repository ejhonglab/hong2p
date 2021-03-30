#!/usr/bin/env python3

from pprint import pprint

import numpy as np
# TODO delete
import matplotlib.pyplot as plt
#

from hong2p import thor, util


def main():
    '''
    date = '2021-03-07'
    fly_num = 1

    fly_dir = util.raw_fly_dir(date, fly_num)
    ret = thor.pair_thor_subdirs(fly_dir, check_against_naming_conv=False)
    print(f'pair_thor_subdirs( {fly_dir} ) output:')
    pprint(ret)
    '''

    # TODO TODO TODO include some test data where recording pin actually goes
    # low (and ideally both volumetric and single-plane experiments where this
    # is the case)

    # Similar format to output of thor.pair_thor_subdirs, but just the tail part
    # of each path.
    exp2thor_dir_pairs = {
        # single plane KC recording w/ blocks
        ('2019-05-03', 3): [
            ('fn_0001', 'SyncData002'),
        ],
        ('2021-03-07', 1): [
            ('t2h_single_plane', 'SyncData002'),
            ('glomeruli_diagnostics_192', 'SyncData001'),
        ],
        # TODO uncomment after moving this code to atlas / moving the data to
        # blackbox
        ## TODO did this have blocks? remove if not, because glomeruli_diag...
        ## above is also a volumetric AL recording (and was it on downstairs 2p
        ## too?)
        #('2020-04-01', 2): [
        #    ('fn_002', 'SyncData002'),
        #],
    }

    for (date, fly_num), thor_dir_pairs in exp2thor_dir_pairs.items():

        for rel_ti, rel_ts in thor_dir_pairs:

            thorimage_dir = util.thorimage_dir(date, fly_num, rel_ti)
            thorsync_dir = util.thorsync_dir(date, fly_num, rel_ts)

            print('ThorImage:', thorimage_dir)
            print('ThorSync:', thorsync_dir)

            exclude_datasets = [
                # TODO useful for assigning frames on boundaries in any
                # volumetric cases?
                'Piezo Monitor',
                'flipperMirror',
                'pid',
                # TODO include Frame In if it doesn't end up seeming useful
            ]
            df = thor.load_thorsync_hdf5(thorsync_dir,
                exclude_datasets=exclude_datasets
            )

            movie = thor.read_movie(thorimage_dir)

            single_plane_fps, xy, z, c, n_flyback, _ = \
                thor.load_thorimage_metadata(thorimage_dir)

            print('xy:', xy)
            print('z:', z)
            if z > 1:
                print('n_flyback:', n_flyback)

            pre_odor_s = 5.0
            odor_s = 1.0
            post_odor_s = 14.0

            trial_s = pre_odor_s + odor_s + post_odor_s
            # this fps if just the xy fps i believe, so need to also include
            # number of z slices (including flyback)
            volumes_per_second = single_plane_fps / (z + n_flyback)
            # frames means volumes here
            frames_per_trial = int(round(trial_s * volumes_per_second))

            print('single_plane_fps:', single_plane_fps)
            print('volumes_per_second:', volumes_per_second)
            print('1 / volumes_per_second:', 1 / volumes_per_second)
            # frames means volumes here
            print('frames_per_trial:', frames_per_trial)

            print('n_frames:', len(movie))

            print('frame_counter.max():', df.frame_counter.max())
            # TODO TODO TODO but if it is zero before the first frame, is it
            # still zero DURING the first frame?
            print('frame_counter.min():', df.frame_counter.min())

            print('frame_in.max():', df.frame_in.max())
            print('frame_in.min():', df.frame_in.min())

            import ipdb; ipdb.set_trace()

            plt.plot(df.time_s, df.scopePin, label='trigger')
            plt.plot(df.time_s, df.frame_out, label='frame_out')
            plt.plot(df.time_s, df.frame_counter.diff(),
                label='frame_counter.diff()'
            )
            plt.xlabel('Time (s)')
            plt.plot(df.time_s, df.frame_counter, label='frame_counter')
            plt.legend()
            plt.show()
            import ipdb; ipdb.set_trace()

            print()


if __name__ == '__main__':
    main()

