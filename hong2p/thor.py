"""
Functions for working with ThorImage / ThorSync outputs, including for dealing
with naming conventions we use for outputs of these programs.
"""

from os import listdir
from os.path import join, split, sep, isdir, normpath, getmtime, abspath
import xml.etree.ElementTree as etree
from datetime import datetime
import warnings
from pprint import pprint
import glob


def xmlroot(xml_path):
    return etree.parse(xml_path).getroot()


# TODO maybe rename to exclude get_ prefix, to be consistent w/
# thorimage_dir(...) and others above?
def get_thorimage_xml_path(thorimage_dir):
    """Takes ThorImage output dir to path to its XML output.
    """
    return join(thorimage_dir, 'Experiment.xml')


def get_thorimage_xmlroot(thorimage_dir):
    """Takes ThorImage output dir to object w/ XML data.
    """
    xml_path = get_thorimage_xml_path(thorimage_dir)
    return xmlroot(xml_path)


def get_thorimage_time_xml(xml):
    """Takes etree XML root object to recording start time.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    date_ele = xml.find('Date')
    from_date = datetime.strptime(date_ele.attrib['date'], '%m/%d/%Y %H:%M:%S')
    from_utime = datetime.fromtimestamp(float(date_ele.attrib['uTime']))
    assert (from_date - from_utime).total_seconds() < 1
    return from_utime


def get_thorimage_time(thorimage_dir, use_mtime=False):
    """Takes ThorImage directory to recording start time (from XML).
    """
    xml_path = get_thorimage_xml_path(thorimage_dir)

    # TODO delete. for debugging matching.
    '''
    xml = xmlroot(xml_path)
    print(thorimage_dir)
    print(get_thorimage_time_xml(xml))
    print(datetime.fromtimestamp(getmtime(xml_path)))
    print('')
    '''
    #
    if not use_mtime:
        xml = xmlroot(xml_path)
        return get_thorimage_time_xml(xml)
    else:
        return datetime.fromtimestamp(getmtime(xml_path))


def get_thorsync_time(thorsync_dir):
    """Returns modification time of ThorSync XML.

    Not perfect, but it doesn't seem any ThorSync outputs have timestamps.
    """
    syncxml = join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return datetime.fromtimestamp(getmtime(syncxml))


def get_thorimage_dims_xml(xml):
    """Takes etree XML root object to (xy, z, c) dimensions of movie.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm_attribs = xml.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x, y)

    # what is Streaming -> flybackLines? (we already have flybackFrames...)

    # TODO maybe either subtract flyback frames here or return that separately?
    # (for now, just leaving it out of this fn)
    z = int(xml.find('ZStage').attrib['steps'])
    # may break on my existing single plane data if value of z not meaningful
    # there

    if z != 1:
        streaming = xml.find('Streaming')
        assert streaming.attrib['enable'] == '1'
        # Not true, see: 2020-03-09/1/fn (though another run of populate_db.py
        # seemed to indicate this dir was passed over for tiff creation
        # anyway??? i'm confused...)
        #assert streaming.attrib['zFastEnable'] == '1'
        if streaming.attrib['zFastEnable'] != '1':
            z = 1

    # still not sure how this is encoded in the XML...
    c = None

    # may want to add ZStage -> stepSizeUM to TIFF metadata?

    return xy, z, c


def get_thorimage_pixelsize_xml(xml):
    """Takes etree XML root object to XY pixel size in um.

    Pixel size in X is the same as pixel size in Y.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    # TODO does thorimage (and their xml) even support unequal x and y?
    # TODO support z here?
    return float(xml.find('LSM').attrib['pixelSizeUM'])


def get_thorimage_fps_xml(xml):
    """Takes etree XML root object to (after-any-averaging) fps of recording.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm_attribs = xml.find('LSM').attrib
    raw_fps = float(lsm_attribs['frameRate'])
    # TODO is this correct handling of averageMode?
    average_mode = int(lsm_attribs['averageMode'])
    if average_mode == 0:
        n_averaged_frames = 1
    else:
        # TODO TODO TODO does this really not matter in the volumetric streaming
        # case, as Remy said? (it's still displayed in the software, and still
        # seems enabled in that averageMode=1...)
        # (yes it does seem to matter TAKE INTO ACCOUNT!)
        n_averaged_frames = int(lsm_attribs['averageNum'])
    saved_fps = raw_fps / n_averaged_frames
    return saved_fps


def get_thorimage_fps(thorimage_directory):
    """Takes ThorImage dir to (after-any-averaging) fps of recording.
    """
    xml = get_thorimage_xmlroot(thorimage_directory)
    return get_thorimage_fps_xml(xml)


def get_thorimage_n_flyback_xml(xml):
    streaming = xml.find('Streaming')
    assert streaming.attrib['enable'] == '1'
    if streaming.attrib['zFastEnable'] == '1':
        n_flyback_frames = int(streaming.attrib['flybackFrames'])
    else:
        n_flyback_frames = 0

    return n_flyback_frames


def load_thorimage_metadata(thorimage_directory, return_xml=False):
    """Returns (fps, xy, z, c, n_flyback, raw_output_path) for ThorImage dir.

    Returns xml as an additional final return value if `return_xml` is True.
    """
    xml = get_thorimage_xmlroot(thorimage_directory)

    # TODO TODO in volumetric streaming case (at least w/ input from thorimage
    # 3.0 from downstairs scope), this is the xy fps (< time per volume). also
    # doesn't include flyback frame. probably want to convert it to
    # volumes-per-second in that case here, and return that for fps. just need
    # to check it doesn't break other stuff.
    fps = get_thorimage_fps_xml(xml)
    xy, z, c = get_thorimage_dims_xml(xml)

    n_flyback_frames = get_thorimage_n_flyback_xml(xml)

    # So far, I have seen this be one of:
    # - Image_0001_0001.raw
    # - Image_001_001.raw
    # ...but not sure if there any meaning behind the differences.
    imaging_files = glob.glob(join(thorimage_directory, 'Image_*.raw'))
    assert len(imaging_files) == 1, 'multiple possible imaging files'
    imaging_file = imaging_files[0]

    if not return_xml:
        return fps, xy, z, c, n_flyback_frames, imaging_file
    else:
        return fps, xy, z, c, n_flyback_frames, imaging_file, xml


# TODO TODO TODO TODO add functions to assign times to frame / split movie into
# blocks. might be helpful to look at the following functions from some of my
# other projects that attempted something like this:
#
# al_imaging:
# - util.load_thor_hdf5
# - util.crop_trailing_frames (not sure i actually need this)
# - util.threshold_crossings
# - util.calc_odor_onsets (though prob don't wanna copy this one too much)
# (don't think i had a separate dedicated pin to just mirror valve timing though
# here, so don't need to do everything it does to get valve timings probably,
# and maybe we can do better)
#
# also look at the thor/remy matlab code that used some of the other data in the
# hdf5 ('DI' rather than 'AI', i think)
#
# assign_frames_to_trials in here might have some of the logic i want
#
# not useful:
# - ejhonglab/imaging_exp_mon
# - ejhonglab/automate2p
# - atlas:~/src/imaging_util (not a git repo)


# TODO rename to indicate a thor (+raw?) format
def read_movie(thorimage_dir, discard_flyback=True):
    """Returns (t,[z,]x,y) indexed timeseries as a numpy array.
    """
    fps, xy, z, c, n_flyback, imaging_file, xml = \
        load_thorimage_metadata(thorimage_dir, return_xml=True)
    # TODO TODO make printing of this and stuff like it part of a -v arg to
    # thor2tiff (maybe an opt to output all required input config for suite2p?
    # or save it automatically to whatever format it uses?)
    print('fps:', fps)

    x, y = xy

    # From ThorImage manual: "unsigned, 16-bit, with little-endian byte-order"
    dtype = np.dtype('<u2')

    with open(imaging_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)

    # TODO maybe just don't read the data known to be flyback frames?

    n_frame_pixels = x * y
    n_frames = len(data) // n_frame_pixels
    assert len(data) % n_frame_pixels == 0, 'apparent incomplete frames'

    # This does not fail in the volumetric case, because 'frames' here
    # refers to XY frames there too.
    assert n_frames == int(xml.find('Streaming').attrib['frames'])

    # TODO how to reshape if there are also multiple channels?

    if z > 0:
        # TODO TODO some way to set pockel power to zero during flyback frames?
        # not sure why that data is even wasting space in the file...
        # TODO possible to skip reading the flyback frames? maybe it wouldn't
        # save time though...
        # just so i can hardcode some fixes based on part of path (assuming
        # certain structure under mb_team, at least for the inputs i need to
        # fix)
        thorimage_dir = abspath(thorimage_dir)
        date_part = thorimage_dir.split(sep)[-3]
        try_to_fix_flyback = False

        # TODO TODO TODO do more testing to determine 1) if this really was a
        # flyback issue and not some bug in thor / some change in thor behavior
        # on update, and 2) what is appropriate flyback [and which change from
        # 2020-04-* stuff is what made this flyback innapropriate? less
        # averaging?]
        if date_part in {'2020-11-29', '2020-11-30'}:
            warnings.warn('trying to fix flyback frames since date_part match')
            # this branch is purely a hacky fix to what seems like an
            # insufficient number of flyback frames with data from a few
            # particular days.
            try_to_fix_flyback = True
            n_flyback = n_flyback + 1

        z_total = z + n_flyback

        orig_n_frames = n_frames
        n_frames, remainder = divmod(n_frames, z_total)
        if not try_to_fix_flyback:
            assert remainder == 0

        if try_to_fix_flyback and remainder != 0:
            # TODO maybe don't warn [the same way?] if dropped frames are just
            # flyback-equivalent?

            # remainder is int but checking equality against float still works
            assert (len(data) - len(data[:-(n_frame_pixels * remainder)])
                ) / n_frame_pixels == remainder

            warnings.warn(f'dropping last {remainder}/{orig_n_frames} frames '
                'because of flyback issue'
            )
            # otherwise the reshape won't work, because it requires even
            # division
            data = data[:-(n_frame_pixels * remainder)]

            # TODO maybe some other check that flyback time was appropriate (to
            # not have to hardcode certain paths + to identify accidental
            # flyback issues) [if flyback is even the issue at all... do more
            # tests] (or just pass through a flag... including to thor2tiff)

        # TODO check this against method by reshaping as before and slicing
        # w/ appropriate strides [+ concatenating?] (what was "before"?)
        data = np.reshape(data, (n_frames, z_total, x, y))

        if discard_flyback:
            data = data[:, :z, :, :]
    else:
        data = np.reshape(data, (n_frames, x, y))

    return data


def assign_frames_to_trials(movie, presentations_per_block, block_first_frames,
    odor_onset_frames):
    """Returns arrays trial_start_frames, trial_stop_frames
    """
    n_frames = movie.shape[0]
    # TODO maybe just add metadata['drop_first_n_frames'] to this?
    # (otherwise, that variable screws things up, right?)
    #onset_frame_offset = \
    #    odor_onset_frames[0] - block_first_frames[0]

    # TODO delete this hack, after implementing more robust frame-to-trial
    # assignment described below
    b2o_offsets = sorted([o - b for b, o in zip(block_first_frames,
        odor_onset_frames[::presentations_per_block])
    ])
    assert len(b2o_offsets) >= 3
    # TODO TODO TODO re-enable after fixing frame_times based issues w/
    # volumetric data
    # TODO might need to allow for some error here...? frame or two?
    # (in resonant scanner case, w/ frame averaging maybe)
    #assert b2o_offsets[-1] == b2o_offsets[-2]
    onset_frame_offset = b2o_offsets[-1]
    #
    
    # TODO TODO TODO instead of this frame # strategy for assigning frames
    # to trials, maybe do this:
    # 1) find ~max # frames from block start to onset, as above
    # TODO but maybe still warn if some offset deviates from max by more
    # than a frame or two...
    # 2) paint all frames before odor onsets up to this max # frames / time
    #    (if frames have a time discontinuity between them indicating
    #     acquisition did not proceed continuously between them, do not
    #     paint across that boundary)
    # 3) paint still-unassigned frames following odor onset in the same
    #    fashion (again stopping at boundaries of > certain dt)
    # [4)] if not using max in #1 (but something like rounded mean)
    #      may still have unassigned frames at block starts. assign those to
    #      trials.
    # TODO could just assert everything within block regions i'm painting
    # does not have time discontinuities, and then i could just deal w/
    # frames

    trial_start_frames = np.append(0,
        odor_onset_frames[1:] - onset_frame_offset
    )
    trial_stop_frames = np.append(
        odor_onset_frames[1:] - onset_frame_offset - 1, n_frames - 1
    )

    # TODO same checks are made for blocks, so factor out?
    total_trial_frames = 0
    for i, (t_start, t_end) in enumerate(
        zip(trial_start_frames, trial_stop_frames)):

        if i != 0:
            last_t_end = trial_stop_frames[i - 1]
            assert last_t_end == (t_start - 1)

        total_trial_frames += t_end - t_start + 1

    assert total_trial_frames == n_frames, \
        '{} != {}'.format(total_trial_frames, n_frames)
    #

    # TODO warn if all block/trial lens are not the same? (by more than some
    # threshold probably)

    return trial_start_frames, trial_stop_frames


# TODO rename to indicate it's parsing from directory name?
def old_fmt_thorimage_num(x):
    if pd.isnull(x) or not (x[0] == '_' and len(x) == 4):
        return np.nan
    try:
        n = int(x[1:])
        return n
    except ValueError:
        return np.nan


# TODO rename to indicate it's parsing from directory name?
def new_fmt_thorimage_num(x):
    parts = x.split('_')
    if len(parts) == 1:
        return 0
    else:
        return int(x[-1])


def thorsync_num(x):
    prefix = 'SyncData'
    return int(x[len(prefix):])


def is_thorsync_dir(d, verbose=False):
    """True if dir has expected ThorSync outputs, False otherwise.
    """
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_settings = False
    have_h5 = False
    for f in files:
        # checking for substring
        if 'ThorRealTimeDataSettings.xml' in f:
            have_settings = True
        if '.h5':
            have_h5 = True

    if verbose:
        print('have_settings:', have_settings)
        print('have_h5:', have_h5)

    return have_h5 and have_settings


def is_thorimage_dir(d, verbose=False):
    """True if dir has expected ThorImage outputs, False otherwise.

    Looks for .raw not any TIFFs now.
    """
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_xml = False
    have_raw = False
    # TODO support tif output case(s) as well
    #have_processed_tiff = False
    for f in files:
        if f == 'Experiment.xml':
            have_xml = True
        # Needs to match at least 'Image_0001_0001.raw' and 'Image_001_001.raw'
        elif f.startswith('Image_00') and f.endswith('001.raw'):
            have_raw = True
        #elif f == split(d)[-1] + '_ChanA.tif':
        #    have_processed_tiff = True

    if verbose:
        print('have_xml:', have_xml)
        print('have_raw:', have_raw)
        if have_xml and not have_raw:
            print('all dir contents:')
            pprint(files)

    if have_xml and have_raw:
        return True
    else:
        return False


def _filtered_subdirs(parent_dir, filter_funcs, exclusive=True, verbose=False):
    """Takes dir and indicator func(s) to subdirs satisfying them.

    Output is a flat list of directories if filter_funcs is a function.

    If it is a list of funcs, output has the same length, with each element
    a list of satisfying directories.
    """
    parent_dir = normpath(parent_dir)

    try:
        _ = iter(filter_funcs)
    except TypeError:
        filter_funcs = [filter_funcs]

    # [[]] * len(filter_funcs) was the inital way I tried this, but the inner
    # lists all end up referring to the same object.
    all_filtered_subdirs = []
    for _ in range(len(filter_funcs)):
        all_filtered_subdirs.append([])

    for d in glob.glob(f'{parent_dir}{sep}*{sep}'):
        if verbose:
            print(d)

        for fn, filtered_subdirs in zip(filter_funcs, all_filtered_subdirs):
            if verbose:
                print(fn.__name__)

            if verbose:
                try:
                    val = fn(d, verbose=True)
                except TypeError:
                    val = fn(d)
            else:
                val = fn(d)

            if verbose:
                print(val)

            if val:
                filtered_subdirs.append(d[:-1])
                if exclusive:
                    break

        if verbose:
            print('')

    if len(filter_funcs) == 1:
        all_filtered_subdirs = all_filtered_subdirs[0]

    return all_filtered_subdirs


def thorimage_subdirs(parent_dir):
    """
    Returns a list of any immediate child directories of `parent_dir` that have
    all expected ThorImage outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorimage_dir)


def thorsync_subdirs(parent_dir):
    """Returns a list of any immediate child directories of `parent_dir`
    that have all expected ThorSync outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorsync_dir)


def thor_subdirs(parent_dir, absolute_paths=True):
    """
    Returns a length-2 tuple, where the first element is all ThorImage children
    and the second element is all ThorSync children (of `parent_dir`).
    """
    thorimage_dirs, thorsync_dirs = _filtered_subdirs(parent_dir,
        (is_thorimage_dir, is_thorsync_dir)
    )
    if not absolute_paths:
        thorimage_dirs = [split(d)[-1] for d in thorimage_dirs]
        thorsync_dirs = [split(d)[-1] for d in thorsync_dirs]

    return (thorimage_dirs, thorsync_dirs)


# TODO TODO generalize / wrap in a way that also allows associating with
# stimulus files / arbitrary other files.
def pair_thor_dirs(thorimage_dirs, thorsync_dirs, use_mtime=False,
    use_ranking=True, check_against_naming_conv=True,
    check_unique_thorimage_nums=None, verbose=False):
    """
    Takes lists (not necessarily same len) of dirs, and returns a list of
    lists of matching (ThorImage, ThorSync) dirs (sorted by experiment time).

    Args:
    check_against_naming_conv (bool): (default=True) If True, check ordering
        from pairing is consistent with ordering derived from our naming
        conventions for Thor software output.

    check_unique_thorimage_nums (bool): (default=True) If True, check numbers
        parsed from ThorImage directory names, as-per convention, are unique.
        Requires check_against_naming_conv to be True.

    Raises ValueError if two dirs of one type match to the same one of the
    other, but just returns shorter list of pairs if some matches can not be
    made. These errors currently just cause skipping of pairing for the
    particular (date, fly) pair above (though maybe this should change?).

    Raises AssertionError when assumptions are violated in a way that should
    trigger re-evaluating the code.
    """
    if use_ranking:
        if len(thorimage_dirs) != len(thorsync_dirs):
            raise ValueError('can only pair with ranking when equal # dirs')

    if check_unique_thorimage_nums and not check_against_naming_conv:
        raise ValueError('check_unique_thorimage_nums=True requires '
            'check_against_naming_conv=True'
        )

    # So that we don't need to explicitly disable both of these flags if we want
    # to disable these checks. Just need to set check_against_naming_conv=False
    if check_unique_thorimage_nums is None and check_against_naming_conv:
        check_unique_thorimage_nums = True

    thorimage_times = {d: get_thorimage_time(d, use_mtime=use_mtime)
        for d in thorimage_dirs
    }
    # TODO should get_thorsync_time not implement/take the same use_mtime kwarg?
    thorsync_times = {d: get_thorsync_time(d) for d in thorsync_dirs}

    thorimage_dirs = np.array(
        sorted(thorimage_dirs, key=lambda x: thorimage_times[x])
    )
    thorsync_dirs = np.array(
        sorted(thorsync_dirs, key=lambda x: thorsync_times[x])
    )

    if use_ranking:
        pairs = list(zip(thorimage_dirs, thorsync_dirs))
    else:
        from scipy.optimize import linear_sum_assignment

        # TODO maybe call scipy func on pandas obj w/ dirs as labels?
        costs = np.empty((len(thorimage_dirs), len(thorsync_dirs))) * np.nan
        for i, tid in enumerate(thorimage_dirs):
            ti_time = thorimage_times[tid]
            if verbose:
                print('tid:', tid)
                print('ti_time:', ti_time)

            for j, tsd in enumerate(thorsync_dirs):
                ts_time = thorsync_times[tsd]

                cost = (ts_time - ti_time).total_seconds()

                if verbose:
                    print(' tsd:', tsd)
                    print('  ts_time:', ts_time)
                    print('  cost (ts - ti):', cost)

                # Since ts time should be larger, but only if comparing XML TI
                # time w/ TS mtime (which gets changed as XML seems to be
                # written as experiment is finishing / in progress).
                if use_mtime:
                    cost = abs(cost)

                elif cost < 0:
                    # TODO will probably just need to make this a large const
                    # inf seems to make the scipy imp fail. some imp it works
                    # with?
                    #cost = np.inf
                    cost = 1e7

                costs[i,j] = cost

            if verbose:
                print('')

        ti_idx, ts_idx = linear_sum_assignment(costs)
        print(costs)
        print(ti_idx)
        print(ts_idx)
        pairs = list(zip(thorimage_dirs[ti_idx], thorsync_dirs[ts_idx]))

    if check_against_naming_conv:
        ti_last_parts = [split(tid)[-1] for tid, _ in pairs]

        thorimage_nums = []
        not_all_old_fmt = False
        for tp in ti_last_parts:
            num = old_fmt_thorimage_num(tp)
            if pd.isnull(num):
                not_all_old_fmt = True
                break
            thorimage_nums.append(num)

        disable_msg = ('\n\nYou may disable this check by setting '
            'check_against_naming_conv=False'
        )
        if not_all_old_fmt:
            try:
                thorimage_nums = [new_fmt_thorimage_num(d)
                    for d in ti_last_parts
                ]
            # If ALL ThorImage directories are not in old naming convention,
            # then we assume they will ALL be named according to the new
            # convention.
            except ValueError as e:
                # (changing error type so it isn't caught, w/ other ValueErrors)
                raise AssertionError('check against naming convention failed, '
                    'because a new_fmt_thorimage_num parse call failed with: ' +
                    str(e) + disable_msg
                )

        # TODO TODO need to stable (arg)sort if not going to check this, but
        # still checking ordering below??? (or somehow ordering by naming
        # convention, so that fn comes before fn_0000, etc?)

        # Call from mb_team_gsheet disables this, so that fn / fn_0000 don't
        # cause a failure even though both have ThorImage num of 0, because fn
        # should always be dropped after the pairing in this case (should be
        # checked in mb_team_gsheet after, since it will then not be checked
        # here).
        if check_unique_thorimage_nums:
            if len(thorimage_nums) > len(set(thorimage_nums)):
                print('Directories where pairing failed:')
                print('ThorImage:')
                pprint(list(thorimage_dirs))
                print('Extracted thorimage_nums:')
                pprint(thorimage_nums)
                print('ThorSync:')
                pprint(list(thorsync_dirs))
                print('')
                raise AssertionError('thorimage nums were not unique')

        thorsync_nums = [thorsync_num(split(tsd)[-1]) for _, tsd in pairs]

        # Ranking rather than straight comparison in case there is an offset.
        ti_rankings = np.argsort(thorimage_nums)
        ts_rankings = np.argsort(thorsync_nums)
        if not np.array_equal(ti_rankings, ts_rankings):
            raise AssertionError('time based rankings inconsistent w/ '
                'file name convention rankings' + disable_msg
            )
        # TODO maybe also re-order pairs by these rankings? or by their own,
        # to also include case where not check_against... ?

    return pairs

    """
    thorimage_times = {d: get_thorimage_time(d) for d in thorimage_dirs}
    thorsync_times = {d: get_thorsync_time(d) for d in thorsync_dirs}

    image_and_sync_pairs = []
    matched_dirs = set()
    # TODO make sure this order is going the way i want
    for tid in sorted(thorimage_dirs, key=lambda x: thorimage_times[x]):
        ti_time = thorimage_times[tid]
        if verbose:
            print('tid:', tid)
            print('ti_time:', ti_time)

        # Seems ThorImage time (from TI XML) is always before ThorSync time
        # (from mtime of TS XML), so going to look for closest mtime.
        # TODO could also warn / fail if closest ti mtime to ts mtime
        # is inconsistent? or just use that?
        # TODO or just use numbers in names? or default to that / warn/fail if
        # not consistent?

        # TODO TODO would need to modify this alg to handle many cases
        # where there are mismatched #'s of recordings
        # (first tid will get the tsd, even if another tid is closer)
        # scipy.optimize.linear_sum_assignment looks interesting, but
        # not sure it can handle 

        min_positive_td = None
        closest_tsd = None
        for tsd in thorsync_dirs:
            ts_time = thorsync_times[tsd]
            td = (ts_time - ti_time).total_seconds()

            if verbose:
                print(' tsd:', tsd)
                print('  ts_time:', ts_time)
                print('  td (ts - ti):', td)

            # Since ts_time should be larger.
            if td < 0:
                continue

            if min_positive_td is None or td < min_positive_td:
                min_positive_td = td
                closest_tsd = tsd

            '''
            # didn't seem to work at all for newer output ~10/2019
            if abs(td) < time_mismatch_cutoff_s:
                if tid in matched_dirs or tsd in matched_dirs:
                    raise ValueError(f'either {tid} or {tsd} was already '
                        f'matched. existing pairs:\n{matched_dirs}')

                image_and_sync_pairs.append((tid, tsd))
                matched_dirs.add(tid)
                matched_dirs.add(tsd)
            '''

            matched_dirs.add(tid)
            matched_dirs.add(tsd)

        if verbose:
            print('')

    return image_and_sync_pairs
    """


# TODO maybe allow calling a fn 'pair_thor_dirs' with either this interface of
# that of current 'pair_thor_dirs', detecting type from args
def pair_thor_subdirs(parent_dir, verbose=False, **kwargs):
    """
    Raises ValueError/AssertionError when pair_thor_dirs does.

    Above, the former causes skipping of automatic pairing, whereas the latter
    is not handled and will intentionally cause failure, to prevent incorrect
    assumptions from leading to incorrect results.
    """
    # TODO TODO need to handle case where maybe one thorimage/sync dir doesn't
    # have all output, and then that would maybe offset the pairing? test!
    # (change filter fns to include a minimal set of data, s.t. all such cases
    # still are counted?)
    thorimage_dirs, thorsync_dirs = thor_subdirs(parent_dir)
    if verbose:
        print('thorimage_dirs:')
        pprint(thorimage_dirs)
        print('thorsync_dirs:')
        pprint(thorsync_dirs)

    return pair_thor_dirs(thorimage_dirs, thorsync_dirs, verbose=True, **kwargs)


# TODO maybe delete / refactor to use fns above
# TODO move this to gui.py if that's the only place i'd use it
# (or project/analysis specific repo/submodule)
def tif2xml_root(filename):
    """Returns etree root of ThorImage XML settings from TIFF filename,
    assuming TIFF was named and placed according to a certain convention.

    Path can be to analysis output directory, as long as raw data directory
    exists.
    """
    if filename.startswith(analysis_output_root()):
        filename = filename.replace(analysis_output_root(), raw_data_root())

    parts = filename.split(sep)
    thorimage_id = '_'.join(parts[-1].split('_')[:-1])

    xml_fname = sep.join(parts[:-2] + [thorimage_id, 'Experiment.xml'])
    return xmlroot(xml_fname)


# TODO TODO rename this one to make it clear why it's diff from above
# + how to use it (or just delete one...)
# TODO + also likely refactor this outside here as mentioned for tif2... above
def fps_from_thor(df):
    """Takes a DataFrame and returns fps from ThorImage XML.
    
    df must have a 'thorimage_dir' column (that can be either a relative or
    absolute path, as long as it's under raw_data_root), which is expected to
    only contain one unique value.

    Only the path in the first row is used.
    """
    # TODO assert unique first?
    thorimage_dir = df['thorimage_path'].iat[0]
    # TODO maybe factor into something that ensures path has a certain prefix
    # that maybe also validates right # parts?
    thorimage_dir = join(raw_data_root(), *thorimage_dir.split('/')[-3:])
    fps = get_thorimage_fps(thorimage_dir)
    return fps


# TODO likely refactor to an analysis/cnmf-interface specific submodule
# TODO at least rename to indicate input is tiff filename
def cnmf_metadata_from_thor(filename):
    """Takes TIF filename to key settings from XML needed for CNMF.
    """
    xml_root = tif2xml_root(filename)
    fps = get_thorimage_fps_xml(xml_root)
    # "spatial resolution of FOV in pixels per um" "(float, float)"
    # TODO do they really mean pixel/um, not um/pixel?
    pixels_per_um = 1 / get_thorimage_pixelsize_xml(xml_root)
    dxy = (pixels_per_um, pixels_per_um)
    # TODO maybe load dims anyway?
    return {'fr': fps, 'dxy': dxy}

