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
        ('2021-03-07', 1): [
            ('glomeruli_diagnostics_192', 'SyncData001'),
            ('t2h_single_plane', 'SyncData002'),
        ],
        # single plane KC recording w/ blocks
        ('2019-01-23', 6): [
            ('_001', 'SyncData001'),
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

            single_plane_fps, xy, z, c, n_flyback, _, xml = \
                thor.load_thorimage_metadata(thorimage_dir, return_xml=True)

            # slow b/c has to read whole movie... read_movie has an assert that
            # n_frames derived this way + using get_thorimage_n_frames_xml are
            # same anyway, and didn't find a case where that assertion failed
            # yet (not true actually, the comparison in read_movie is against #
            # of xy frames but len(movie) would be # of volumes in volumetric
            # case)
            #movie = thor.read_movie(thorimage_dir)
            #print('n_frames:', len(movie))

            # TODO perhaps this should take arg to drop flyback?
            n_frames = thor.get_thorimage_n_frames_xml(xml)
            print('n_frames:', n_frames)

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
            print('1 / single_plane_fps:', 1 / single_plane_fps)
            print('volumes_per_second:', volumes_per_second)
            print('1 / volumes_per_second:', 1 / volumes_per_second)
            # frames means volumes here
            print('frames_per_trial:', frames_per_trial)

            # literally all of my data has this as all zeros...
            #print('frame_in.max():', df.frame_in.max())
            #print('frame_in.min():', df.frame_in.min())

            #frame_times, t0 = thor.get_frame_times(df, xml)
            frame_times, t0 = thor.get_frame_times(df, thorimage_dir)

            # (now that frame_times calculation drops flyback frame times,
            # this won't work in volumetric data)
            #assert len(frame_times) == n_frames, \
            #    f'{len(frame_times)} != {n_frames}'

            bounding_frames = thor.assign_frames_to_odor_presentations(df,
                thorimage_dir
            )
            lens = [end - start + 1 for start, end in bounding_frames]

            # TODO update assertion for flyback case
            #assert sum(lens) == n_frames, f'{sum(lens)} != {n_frames}'

            #'''
            # TODO automatically show just a bit of start and end of each block
            # (in subplot grid?)

            plt.plot(df.time_s, df.scopePin, label='trigger')
            plt.plot(df.time_s, df.frame_out, label='frame_out')
            plt.plot(df.time_s, df.frame_counter.diff(),
                label='frame_counter.diff()'
            )
            plt.xlabel('Time (s)')
            plt.plot(df.time_s, df.frame_counter, label='frame_counter')

            plt.plot(frame_times, [0.5] * len(frame_times), linestyle='None',
                marker='x', label='frame_times'
            )

            '''
            colors = ['r', 'g', 'b']
            assert len(colors) == len(bounding_frames), \
                f'!= {len(bounding_frames)}'

            for i, ((start_frame, end_frame), c) in enumerate(
                zip(bounding_frames, colors)):

                odor_frames = frame_times[start_frame:(end_frame + 1)]
                plt.plot(odor_frames, [2.5] * len(odor_frames),
                    linestyle='None', marker='x', color=c, label=f'{i}'
                )
            '''

            plt.legend()
            plt.show()
            #'''
            import ipdb; ipdb.set_trace()

            print()


if __name__ == '__main__':
    main()

