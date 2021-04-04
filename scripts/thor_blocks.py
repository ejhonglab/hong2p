#!/usr/bin/env python3

from pprint import pprint
import time

import numpy as np

# NOTE: these lines are only here so i could get plots to work interactively
# over 'ssh -X' one time, where they didn't seem to work w/ PyQt5 (though maybe
# PyQt5 is just broken even used locally as I have it installed. not sure.)
import matplotlib as mpl
# Needed to `sudo apt install python3.8-tk` (from same deadsnakes ppa I got 3.8
# from) on the 16.04 machine where this was an issue.
mpl.use('TkAgg')

import matplotlib.pyplot as plt

from hong2p import thor, util


def main():
    hdf5_exclude_datasets = [
        'piezo_monitor',
        'pockels1_monitor',
        'frame_in',
        'light_path_shutter',
        'flipper_mirror',
        'pid',
        # Can comment to include this in some of the plots, or uncomment for
        # *slightly* faster load times.
        'frame_counter',
    ]

    '''
    date = '2021-03-07'
    fly_num = 1

    fly_dir = util.raw_fly_dir(date, fly_num)
    ret = thor.pair_thor_subdirs(fly_dir)
    print(f'pair_thor_subdirs( {fly_dir} ) output:')
    pprint(ret)
    '''

    # TODO could maybe generate test data by modulating shutter / mirror / laser
    # power during recording, so it's effectively off for certain blocks, and
    # use that to test functions that assign frames to blocks. similar to having
    # an LED in field of view to test association of video data w/ other timed
    # events

    # TODO TODO try to include test data from both scopes + a variety of thor
    # software versions

    # Similar format to output of thor.pair_thor_subdirs, but just the tail part
    # of each path.
    exp2thor_dir_pairs = {
        # single plane KC recording w/ blocks
        ('2019-01-23', 6): [
            ('_001', 'SyncData001'),
        ],
        # volumetric AL recording w/ blocks (and was it on downstairs 2p?)
        ('2020-04-01', 2): [
            ('fn_002', 'SyncData002'),
        ],
        ('2021-03-07', 1): [
            # volumetric AL recording. no blocks.
            ('glomeruli_diagnostics_192', 'SyncData001'),
            # single plane AL recording. no blocks.
            ('t2h_single_plane', 'SyncData002'),
        ],
    }

    for (date, fly_num), thor_dir_pairs in exp2thor_dir_pairs.items():

        for rel_ti, rel_ts in thor_dir_pairs:

            thorimage_dir = util.thorimage_dir(date, fly_num, rel_ti)
            thorsync_dir = util.thorsync_dir(date, fly_num, rel_ts)

            print('ThorImage:', thorimage_dir)
            print('ThorSync:', thorsync_dir)

            # TODO TODO possible to parse / print thor software versions + which
            # computer it was acquired on?

            before = time.time()
            print('loading HDF5...', flush=True, end='')

            df = thor.load_thorsync_hdf5(thorsync_dir,
                exclude_datasets=hdf5_exclude_datasets #, verbose=True
            )

            took_s = time.time() - before
            print(f'done ({took_s:.1f}s)')

            print('df.columns:')
            pprint(list(df.columns))

            # literally all of my data has this as all zeros...
            #print('frame_in.max():', df.frame_in.max())
            #print('frame_in.min():', df.frame_in.min())

            single_plane_fps, xy, z, c, n_flyback, _, xml = \
                thor.load_thorimage_metadata(thorimage_dir, return_xml=True)

            # slow b/c has to read whole movie... read_movie has an assert that
            # n_frames derived this way + using get_thorimage_n_frames_xml are
            # same anyway, and didn't find a case where that assertion failed
            # yet (not true actually, the comparison in read_movie is against #
            # of xy frames but len(movie) would be # of volumes in volumetric
            # case)
            #movie = thor.read_movie(thorimage_dir)

            # TODO perhaps this should take arg to drop flyback?
            n_frames = thor.get_thorimage_n_frames_xml(xml)

            n_averaged_frames = thor.get_thorimage_n_averaged_frames_xml(xml)

            print('xy:', xy)
            print('z:', z)
            if z > 1:
                print('n_flyback:', n_flyback)

            print('n_averaged_frames:', n_averaged_frames)

            def plot(label, x, y=None, line=True, ax=None, **kwargs):
                if ax is None:
                    ax = plt.gca()

                if y is None:
                    y = 1.0

                try:
                    len(y)
                except TypeError:
                    y = len(x) * [y]
                    line = False

                if not line:
                    kwargs['linestyle'] = 'None'
                    if 'marker' not in kwargs:
                        kwargs['marker'] = 'x'

                ax.plot(x, y, label=label, **kwargs)

            acq_onsets_s, acq_offsets_s = thor.get_col_onset_offset_times(df,
                'scope_pin'
            )

            # TODO or maybe also / instead set offsets windows (and thus onset
            # windows too, if same) to always include last LOW of frame_out
            # after each acq_offset (before it starts pulsing again)
            # (likewise maybe always include buffer for first frame_out rise
            # after each acq_onset?)
            before_avg_fps = thor.get_thorimage_fps(thorimage_dir,
                before_averaging=True
            )

            # Show this much time before and after each scope_pin onset/offset.
            # OK enough for the data I've tested so far.
            show_around_acq_edges_s = 8 * (1 / before_avg_fps)

            debugging_get_frame_times = False
            #debugging_get_frame_times = True
            only_plot_near_acq_edges_s = True

            all_acq_edges = np.sort(np.concatenate([
                acq_onsets_s, acq_offsets_s
            ]))

            # So that if we remove data for plotting efficiency, it doesn't
            # affect `thor` calls below, like `get_frame_times`.
            pdf = df.copy()

            if 'frame_counter' in pdf.columns:
                # Need to diff up here in case we cause incontinuities in
                # `pdf.time_s` by removing chunks of data in the conditional
                # below.
                pdf['frame_counter_diff'] = pdf.frame_counter.diff()

            if only_plot_near_acq_edges_s:
                mask_to_plot = np.zeros(len(pdf), dtype=np.bool_)

                for edge_s in all_acq_edges:
                    t0 = edge_s - show_around_acq_edges_s
                    t1 = edge_s + show_around_acq_edges_s

                    # TODO check that these logical_* fns aren't acting weird w/
                    # series inputs...
                    for_curr_edge = np.logical_and(
                        t0 <= pdf.time_s, pdf.time_s <= t1
                    )
                    mask_to_plot = np.logical_or(mask_to_plot, for_curr_edge)

                pdf = pdf[mask_to_plot]

            # With the shape in this transpose, `axs.flat` iterates over in the
            # correct order for it to visit the 2nd element of the length-2
            # dimention on the second iteration, which is consistent w/ how the
            # edges are ordered in `all_acq_edges` (offset following onset).
            fig, axs = plt.subplots(nrows=len(acq_onsets_s), ncols=2)

            for i, (edge_s, ax) in enumerate(zip(all_acq_edges, axs.flat)):

                t0 = edge_s - show_around_acq_edges_s
                t1 = edge_s + show_around_acq_edges_s
                curr_time_range = (t0 <= pdf.time_s) & (pdf.time_s <= t1)

                curr_df = pdf[curr_time_range]

                plot('acquisition trigger', curr_df.time_s, curr_df.scope_pin,
                    ax=ax
                )
                plot('frame_out', curr_df.time_s, curr_df.frame_out, ax=ax)

                # TODO TODO maybe convert to just a set of points where this is
                # the appropriate value, and plot w/ line=False
                # (to make more readily comparable to my calculated frame times,
                # in plots that include them)
                if 'frame_counter' in curr_df.columns:
                    frame_counter_increases = curr_df.time_s[
                        (curr_df.frame_counter_diff > 0)
                    ]
                    plot('frame_counter increases', frame_counter_increases,
                        ax=ax
                    )

                # Just plotting this for the first one, so the scale doesn't
                # blow up.
                if 'frame_counter' in curr_df.columns and i == 0:
                    # TODO how to make it so this is also included in
                    # get_legend_handles_labels output (or how to modify
                    # output)?
                    plot('frame_counter', curr_df.time_s, curr_df.frame_counter,
                        ax=ax
                    )

                ax.set_xlabel('Time (s)')

            if debugging_get_frame_times:
                handles, labels = axs.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right')
                plt.show()

            frame_times = thor.get_frame_times(df, thorimage_dir)

            # Not plotting this stuff in this case just because we already
            # called `plt.show()` for the plot these should be added to above.
            # And `thor.get_frame_times` probably failed / produced known-bad
            # output anyway...
            if not debugging_get_frame_times:

                bounding_frames = thor.assign_frames_to_odor_presentations(df,
                    thorimage_dir
                )

                lens = [end - start + 1 for start, end in bounding_frames]
                if n_flyback > 0:
                    z_total = z + n_flyback
                    n_frames, remainder = divmod(n_frames, z_total)
                    assert remainder == 0

                assert sum(lens) == n_frames

                for i, (edge_s, ax) in enumerate(zip(all_acq_edges, axs.flat)):
                    prev_xlim = ax.get_xlim()

                    colors = ['r', 'g', 'b']
                    if len(bounding_frames) == len(colors):

                        for i, ((start_frame, end_frame), c) in enumerate(
                            zip(bounding_frames, colors)):

                            odor_frames = frame_times[
                                start_frame:(end_frame + 1)
                            ]
                            plot(f'{i}', odor_frames, 0.5, ax=ax, color=c)

                    else:
                        plot('frame_times', frame_times, 0.5, ax=ax)

                    ax.set_xlim(prev_xlim)

                # TODO issue w/ axes? maybe just plot all frame_times on all?
                handles, labels = axs.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right')
                plt.show()

            print()


if __name__ == '__main__':
    main()

