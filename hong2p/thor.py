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
from itertools import zip_longest

import numpy as np
import pandas as pd


_acquisition_trigger_names = ('scope_pin',)
_odor_timing_names = ('olf_disp_pin',)


def xmlroot(xml_path):
    """Loads contents of xml_path into xml.etree.ElementTree and returns root.

    Use calls to <node>.find(<child name>) to traverse down tree and at leaves,
    use <leaf>.attrib[<attribute name>] to get values. There are other functions
    too, but see `xml` documentation for more information.
    """
    return etree.parse(xml_path).getroot()


# TODO maybe rename everything to get rid of 'get_' prefix? mainly here so i
# can name what these functions return naturally without shadowing...

thorimage_xml_basename = 'Experiment.xml'
def get_thorimage_xml_path(thorimage_dir):
    """Takes ThorImage output dir to (expected) path to its XML output.
    """
    return join(thorimage_dir, thorimage_xml_basename)


# TODO maybe allow this to return identify in case it's passed xml root?
# so that *some* *_xml(...) fns can be collapsed into the corresponding fns
# without the suffix.
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

    if not use_mtime:
        xml = xmlroot(xml_path)
        return get_thorimage_time_xml(xml)
    else:
        return datetime.fromtimestamp(getmtime(xml_path))


def get_thorimage_n_frames_xml(xml):
    """Returns the number of XY planes (# of timepoints) in the recording.

    This is the number of frames *after* any averaging configured in ThorImage.

    Any flyback frames are included.
    """
    return int(xml.find('Streaming').attrib['frames'])


def get_thorimage_z_xml(xml):
    """Returns number of different Z depths measured in ThorImage recording.

    Does NOT include any flyback frames there may be.
    """
    return int(xml.find('ZStage').attrib['steps'])


def is_fast_z_enabled_xml(xml):
    streaming = xml.find('Streaming')
    # TODO may want to only do this check under certain other conditions / not
    # at all (in case someone were to try to call this / a function that calls
    # this on non-streaming data)
    assert streaming.attrib['enable'] == '1'
    enabled = streaming.attrib['zFastEnable'] == '1'

    if enabled:
        assert get_thorimage_z_xml(xml) > 1, 'z <= 1 but fast z seems enabled'

    return enabled


# TODO TODO maybe add a function to get expected movie.size from thorimage .raw
# file, and then include a check that these sizes match those expected by
# multiplying all the relevant dimensions (including flyback) in the metadata
# (would also check assumption that n_frames in ThorImage XML always acurately
# reflects the number of frames in the ThorImage .raw file, and would need to
# modify / not do such a check if this assumption turns out to be false)
# TODO also would need to check that i'm using the correct means of getting the
# size of the file, as we want how much data it actually contains not like how
# much space the particular storage media / filesystem happens to need to store
# it (which i think can slightly exceed the real amount of data the file should
# have, from what i remember using `du`)


# TODO probably also include number of "frames" (*planes* over time) here too
# (and in functions that call this)
# (though would need to take into account flyback as well as potentially
# averaging in order to have this dimension reflect shape of movie (as if the
# output of this function were `movie.shape` for the corresponding movie))
def get_thorimage_dims_xml(xml):
    """Takes etree XML root object to (xy, z, c) dimensions of movie.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm_attribs = xml.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x, y)

    # TODO what is Streaming -> flybackLines? (we already have flybackFrames...)

    z = get_thorimage_z_xml(xml)

    # TODO maybe move this check to inside get_thorimage_z_xml / remove it
    # (because it would probably not work correctly in the case of non-streaming
    # volumetric acquisitions). would then want to move logic of z > 1 check
    # from `is_fast_z_enabled_xml` to `get_thorimage_z_xml` most likely.
    if z != 1 and not is_fast_z_enabled_xml(xml):
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


def get_thorimage_n_averaged_frames_xml(xml):
    """Returns how many frames ThorImage averaged for a single output frame.
    """
    lsm_attribs = xml.find('LSM').attrib

    # TODO is this correct handling of averageMode?
    average_mode = int(lsm_attribs['averageMode'])

    if average_mode == 0:
        n_averaged_frames = 1
    else:
        if is_fast_z_enabled_xml(xml):
            n_averaged_frames = 1
        else:
            n_averaged_frames = int(lsm_attribs['averageNum'])

    return n_averaged_frames


def get_thorimage_fps_xml(xml, before_averaging=False):
    # TODO TODO clarify in doc whether this is volumes-per-second or
    # xy-planes-per-second in the volumetric case (latter, i believe, though
    # maybe add kwarg to get former?)
    """Takes XML root object to fps of recording.

    xml: etree XML root object as returned by `get_thorimage_xmlroot`.

    before_averaging (bool): (default=False) pass True to return the fps before
        any averaging.

    """
    lsm_attribs = xml.find('LSM').attrib
    raw_fps = float(lsm_attribs['frameRate'])

    if before_averaging:
        return raw_fps

    n_averaged_frames = get_thorimage_n_averaged_frames_xml(xml)

    saved_fps = raw_fps / n_averaged_frames
    return saved_fps


def get_thorimage_fps(thorimage_directory, **kwargs):
    # TODO TODO clarify in doc whether this is volumes-per-second or
    # xy-planes-per-second in the volumetric case (latter, i believe, though
    # maybe add kwarg to get former?)
    """Takes ThorImage dir to fps of recording.

    before_averaging (bool): (default=False) pass True to return the fps before
        any averaging.

    All `kwargs` are passed through to `get_thorimage_fps_xml`.
    """
    xml = get_thorimage_xmlroot(thorimage_directory)
    return get_thorimage_fps_xml(xml, **kwargs)


def get_thorimage_n_flyback_xml(xml):
    if is_fast_z_enabled_xml(xml):
        streaming = xml.find('Streaming')
        n_flyback_frames = int(streaming.attrib['flybackFrames'])
    else:
        n_flyback_frames = 0

    return n_flyback_frames


def get_thorimage_notes_xml(xml):
    return xml.find('ExperimentNotes').attrib['text']


def load_thorimage_metadata(thorimage_dir, return_xml=False):
    """Returns (fps, xy, z, c, n_flyback, raw_output_path) for ThorImage dir.

    Returns xml as an additional final return value if `return_xml` is True.
    """
    xml = get_thorimage_xmlroot(thorimage_dir)

    # TODO TODO in volumetric streaming case (at least w/ input from thorimage
    # 3.0 from downstairs scope), this is the xy fps (< time per volume). also
    # doesn't include flyback frame. probably want to convert it to
    # volumes-per-second in that case here, and return that for fps. just need
    # to check it doesn't break other stuff.
    fps = get_thorimage_fps_xml(xml)
    xy, z, c = get_thorimage_dims_xml(xml)

    n_flyback_frames = get_thorimage_n_flyback_xml(xml)
    if z == 1:
        assert n_flyback_frames == 0, 'n_flyback_frames > 0 but z == 1'

    # So far, I have seen this be one of:
    # - Image_0001_0001.raw
    # - Image_001_001.raw
    # ...but not sure if there any meaning behind the differences.
    imaging_files = glob.glob(join(thorimage_dir, 'Image_*.raw'))

    if len(imaging_files) == 0:
        raise IOError(f'no .raw files in ThorImage directory {thorimage_dir}')

    elif len(imaging_files) > 1:
        raise RuntimeError('multiple .raw files in ThorImage directory '
            f'{thorimage_dir}. ambiguous!'
        )

    imaging_file = imaging_files[0]

    if not return_xml:
        return fps, xy, z, c, n_flyback_frames, imaging_file
    else:
        return fps, xy, z, c, n_flyback_frames, imaging_file, xml


thorsync_xml_basename = 'ThorRealTimeDataSettings.xml'
def get_thorsync_xml_path(thorsync_dir):
    """Takes ThorSync output dir to (expected) path to its XML output.
    """
    return join(thorsync_dir, thorsync_xml_basename)


# TODO is this also updated past start of recording, as I think the ThorImage
# one is?
def get_thorsync_time(thorsync_dir):
    """Returns modification time of ThorSync XML.

    Not perfect, but it doesn't seem any ThorSync outputs have timestamps.
    """
    syncxml = get_thorsync_xml_path(thorsync_dir)
    return datetime.fromtimestamp(getmtime(syncxml))


thorsync_h5_basename = 'Episode001.h5'
def is_thorsync_h5(f):
    """True if filename indicates file is ThorSync HDF5 output.
    """
    _, f_basename = split(f)
    # So far I've only seen these files named *exactly* 'Episode001.h5', but
    # this function could be adapted if this naming convention has some
    # variations in the future.
    if f_basename == thorsync_h5_basename:
        return True

    return False


def get_thorsync_samplerate_hz(thorsync_dir):
    """Returns int sample rate (Hz) of ThorSync HDF5 data in `thorsync_dir`.
    """
    xml_path = get_thorsync_xml_path(thorsync_dir)
    xml = xmlroot(xml_path)
    devices = xml.find('DaqDevices')

    # TODO some of the keys seem to hint that this xml also describes which
    # channel was used to trigger the recording, though they don't seem set as i
    # would think... maybe they are for something else
    # (if this data is there, could automatically pull out the channel that is
    # used to trigger the thorimage recording, or something like that)
    # (maybe that data is in thorimage config actually?)

    active_device = None
    for device in devices.getchildren():
        attrib = device.attrib

        if attrib['type'] == 'Simulator' or attrib['devID'] == 'NONE':
            continue

        if int(attrib['active']):
            if active_device is not None:
                raise ValueError('multiple AcquireBoard elements active in '
                    f'{xml_path}'
                )
            active_device = device

    if active_device is None:
        raise ValueError(f'no AcquireBoard elements active in {xml_path}')

    samplerate_hz = None
    for samprate_ele in active_device.findall('SampleRate'):
        attrib = samprate_ele.attrib
        if int(attrib['enable']):
            if samplerate_hz is not None:
                raise ValueError('multiple SampleRate elements active in '
                    f'{xml_path}'
                )
            samplerate_hz = int(samprate_ele.attrib['rate'])

    if samplerate_hz is None:
        raise ValueError(f'no SampleRate elements active in {xml_path}')

    return samplerate_hz


def get_thorsync_h5(thorsync_dir):
    """Returns path to ThorSync .h5 output given a directory created by ThorSync
    """
    # NOTE: if in the future this filename varies, could instead iterate over
    # files, calling `is_thorsync_h5` and returning list / [asserting one +
    # returning it]
    return join(thorsync_dir, thorsync_h5_basename)


# TODO rename to indicate a thor (+raw?) format
# TODO rename to 'load_movie' to be consistent w/ other similar fns in here?
# TODO refactor this to something like 'load_thorimage_raw' + have
# '[load/read]_movie' call either this or appropriate tifffile calls to load any
# TIFF outputs thorimage might have saved (check that dimension orders are the
# same!)?
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
    # TODO test this assertion on all data, though perhaps via a new function
    # get get expected n_frames from size of .raw file + other metadata
    # (mentioned in comments above get_thorimage_dims_xml)
    assert n_frames == get_thorimage_n_frames_xml(xml)

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


def _get_column(df, possible_col_names):
    """Returns `pd.Series` corresponding to first matching column in `df`.

    Raises ValueError if no matches are found.
    """
    if isinstance(possible_col_names, str):
        return df[possible_col_names]

    col = None
    for n in possible_col_names:
        if n in df.columns:
            col = df[n]
            break

    if col is None:
        raise ValueError(
            f'no column with name in {possible_col_names} in df'
        )

    return col


time_col = 'time_s'
# Any datasets with names in the keys of this dict will have the corresponding
# value used for the column name. This will happen before any lowercasing /
# space->underscore conversion in `load_thorsync_hdf5`.
hdf5_dataset_rename_dict = {
    # Adding the space when column names lack it, so underscore separated
    # version will become the standard after normalization.
    'FrameIn': 'Frame In',
    'FrameOut': 'Frame Out',
    'FrameCounter': 'Frame Counter',

    # Also lower casing this one since it's under the AI group, and wouldn't be
    # lower cased automatically.
    'PiezoMonitor': 'piezo monitor',
    'Piezo Monitor': 'piezo monitor',

    # Stuff under AI that wouldn't be lowercased automatically, but that I still
    # want snakecase for, to be consistent. I believe the pockels monitor
    # channel might be a Thor built-in output, even though it's under AI?
    'Pockels1Monitor': 'pockels1_monitor',
    'Pockels1 Monitor': 'pockels1_monitor',
    'flipperMirror': 'flipper_mirror',
    'lightPathShutter': 'light_path_shutter',
    'lightpathshutter': 'light_path_shutter',
    'olfDispPin': 'olf_disp_pin',
    'scopePin': 'scope_pin',
}
# TODO maybe refactor this a bit and add a function to list datasets, just so
# people can figure out their own data post hoc w/o needing other tools
def load_thorsync_hdf5(thorsync_dir, datasets=None, exclude_datasets=None,
    drop_gctr=True, return_dataset_names_only=False, skip_dict_rename=False,
    skip_normalization=False, rename_dict=None, use_tqdm=False, verbose=False):
    """Loads ThorSync .h5 output within `thorsync_dir` into a `pd.DataFrame`

    A column 'time_s' will be added, which is derived from 'GCtr', and
    represents the time (in seconds) from the start of the ThorSync recording.

    Args:
    datasets (iterable of str | None): Load only datasets with these names.
        Do not include the group names preceding the dataset name. Pass only
        one of either this or `exclude_datasets`. Names are checked after any
        renaming via `rename_dict` or normalization.

    exclude_datasets (iterable of str | None): Load only datasets *except* those
        with these names. Do not include 'gctr' here.

    drop_gctr (bool): (default=True) Drop '/Global/GCtr' data (would be returned
        as column 'gctr') after using it to calculate 'time_s' column.

    rename_dict (None or dict): (default=None) a dict of original->new name. If
        not passed, `hdf5_dataset_rename_dict` is used. Applied before any
        further operations on the column (dataset) names.

    These HDF5 files have the following hierarchical structure, where leaves of
    this tree are "Datasets" and their parents are "Groups" (via inspection of a
    ThorSync 3.0 output):
    - Global:
      - GCtr
        (from ThorSync 3.0 manual) "ThorSync records data into a table with
        clock cycles beginning with 0.  The time of acquisition can be
        determined by dividing the clock cycle by the frequency of the data
        collection set at 20 MHz. Thus, each sequential clock cycle represents
        an increment of 0.05 μs."

        Note that this 20 MHz is not the same as the sampling rate specified in
        the ThorSync XML output. See commented example at end of this function.

    - DI:
      - Frame In
        - completely zero in the file I was exploring

      - Frame Out
        - may have one high pulse (==2 for some reason; low==0) per frame
        - seems to only be low briefly before returning high again. perhaps just
          for one / a few samples?
        - it may be possible there are cases where there are more high pulses
          here than there are frames in the movie, perhaps in cases with
          averaging or multiple separate acquisition periods.

    - CI:
      - Frame Counter

    - AI:
      - <one entry for each user-configured analog input>

    Three changes will be made in translating HDF5 dataset names to DataFrame
    column names:
    1. If any dataset name is in the keys of `rename_dict`, it will be replaced
       with the corresponding value, unless `skip_dict_rename` is passed.

    2. Names *except* those under the group 'AI' (mostly user configurable
       inputs) will be lowercased, unless `skip_normalization` is passed.

    3. All names will have any spaces converted to underscores, unless
       `skip_normalization` is passed.

    """
    # TODO TODO DI/Frame [In/Out] useful? how?

    # I tried to use `pd.read_hdf` in place of this, but no matter how I used it
    # (tried various arguments to key=), just got various errors.
    import h5py

    # TODO maybe just silently ignore exclude_datasets if datasets is passed, so
    # i can have some defaults in exclude_datasets that can be overridden if
    # need be...
    if not (datasets is None or exclude_datasets is None):
        raise ValueError('only pass at most one of datasets or exclude_datasets')

    # Structure of hdf5 can be explored via:
    # h5dump -H <h5 path>
    # (need to `sudo apt install hdf5-tools` first)
    hdf5_fname = get_thorsync_h5(thorsync_dir)

    if rename_dict is None:
        rename_dict = hdf5_dataset_rename_dict

    if return_dataset_names_only:
        full_dataset_names = []

    data_dict = dict()
    def load_datasets(name, obj):
        # Could also check if `obj` has a 'shape' attribute if this approach has
        # issues.
        if isinstance(obj, h5py.Dataset):
            if return_dataset_names_only:
                full_dataset_names.append(obj.name)
                return

            parent_name = obj.parent.name

            # In data from 2019-05-03/3/SyncData002, this has keys 'Hz' and
            # 'FitHz' under it, each of shape (3000, 1) (<< length of other
            # arrays, so would cause DataFrame creation to fail). min/max of
            # both datasets were 0, so they don't seem to be used, at least as I
            # had the acquisition configured.
            if parent_name == '/Freq':
                return

            # Excluding the names of the Group(s) containing this Dataset.
            dataset_name = obj.name[(len(parent_name) + 1):]
            if verbose:
                print('parent name:', parent_name)
                print('original name:', dataset_name)

            if not skip_dict_rename and dataset_name in rename_dict:
                dataset_name = rename_dict[dataset_name]

            if not skip_normalization:
                # Seemingly consistent with what the Thorlabs MATLAB scripts are
                # doing, and something I'd want to do anyway.
                dataset_name = dataset_name.replace(' ', '_')

                if parent_name != '/AI':
                    # This could in theory eliminate some uniqueness of the
                    # names, but in practice it really shouldn't.  Not doing
                    # this for all keys so that things like 'olfDispPin' don't
                    # become hard to read.
                    dataset_name = dataset_name.lower()

            if verbose:
                print('normalized name:', dataset_name)
                print()

            if datasets and dataset_name not in datasets:
                return

            if exclude_datasets and dataset_name in exclude_datasets:
                return

            shape = obj.shape
            assert len(shape) == 2 and shape[1] == 1, 'unexpected shape'
            # NOTE: would be an issue if someone named one of the user-nameable
            # analog inputs to be the same as one of the builtin dataset names
            assert dataset_name not in data_dict, 'dataset names not unique'

            values = obj[:, 0]

            if parent_name == '/DI':
                # Anything non-zero gets converted to True
                values = values.astype(np.bool_)

            data_dict[dataset_name] = values

    # TODO warn/err (configurable via kwarg?) if any datasets requested were not
    # found (to help identify naming convention changes in the HDF5 files, etc)

    # NOTE: for some reason, opening a debugger (e.g. via `ipdb.set_trace()`)
    # inside this context manager has `self` in `dir()`, seemingly pointing to
    # `f`, but `f` can not be referenced directly.
    with h5py.File(hdf5_fname, 'r') as f:
        # Populates data_dict
        f.visititems(load_datasets)

    if return_dataset_names_only:
        return full_dataset_names

    # TODO maybe compare performance w/ w/o conversion to Dataframe?
    df = pd.DataFrame(data_dict)


    gctr_col = 'gctr'
    if gctr_col in df.columns:
        # TODO check whether this is (nearly) equivalent to multiplying arange
        # len samples by 1 / thorsync sampling rate
        # Dividing what I think is the clock cycle counter by the 20MHz
        # mentioned in the 3.0 ThorSync manual (section 5.2 "Reviewing Data").
        df[time_col] = df[gctr_col] / int(2e7)

        if drop_gctr:
            df.drop(columns=gctr_col, inplace=True)

    else:
        # Valid as long as gctr_col has no spaces and (exclude_datasets and
        # datasets) are mutually exclusive.
        print(datasets is None)
        print({x.lower() for x in datasets})
        print(gctr_col.lower() in {x.lower() for x in datasets})
        assert (datasets is not None and
            gctr_col.lower() not in {x.lower() for x in datasets}
        )

    # Just to illustrate what the sampling rate in the XML is. This check should
    # work, but no need for it to be routine.
    #
    # samprate_hz = get_thorsync_samplerate_hz(thorsync_dir)
    # mean_sample_interval = df[time_col].diff().mean()
    # expected_sample_interval = 1 / samprate_hz
    # assert np.isclose(mean_sample_interval, expected_sample_interval), \
    #     'ThorSync XML sample rate or acquisition clock frequency wrong'

    return df


# TODO is this slow / are there faster alternatives?
# (copied from my deprecated al_imaging/al_imaging/util.py)
def threshold_crossings(signal, threshold=2.5, onsets=True, offsets=True):
    # TODO clarify "ignored" in equality case in doc
    """
    Returns indices where signal goes from < threshold to > threshold as onsets,
    and where signal goes from > threshold to < threshold as offsets.
    
    Cases where it at one index equals the threshold are ignored. Shouldn't
    happen and may indicate electrical problems for our application.
    """
    # TODO could redefine in terms of np.diff
    # might be off by one?
    # TODO TODO TODO detect whether input is pandas series and only use .values
    # in that case
    # NOTE: we must call .values or else some of the comparison operations
    # across series will behave in a manner we don't want (np.logical_and, I
    # think).
    shifted = signal[1:].values
    truncated = signal[:-1].values

    onset_indices = None
    offset_indices = None

    # TODO maybe special case boolean (np.bool_ dtype; digital) inputs to not
    # use comparison against a float, if something else is faster

    if onsets:
        onset_indices = np.where(np.logical_and(
            shifted > threshold,
            truncated < threshold
        ))[0]

    if offsets:
        offset_indices = np.where(np.logical_and(
            shifted < threshold,
            truncated > threshold
        ))[0]

    # TODO TODO check whether these indices lead to off-by-one if used to index
    # times (+ fix here if so)
    return onset_indices, offset_indices


# TODO TODO generalize + refactor other stuff to use / maybe delete
# (and maybe just take xml / thorimage_dir as input, as may want to handle `c`
# / averaging / etc later)
def get_flyback_indices(n_frames, z, n_flyback, series=None):
    """Returns indices of XY frames during piezo flyback, or empty array if none
    """
    if series is not None:
        assert n_frames == len(series), f'{n_frames} != {len(series)}'

    if n_flyback == 0:
        return np.array([])

    # TODO return appropriate values to cause no-op in subsequent operations if
    # input does not have flyback frames (e.g. not volumetric)

    z_total = z + n_flyback

    orig_n_frames = n_frames
    n_volumes, remainder = divmod(n_frames, z_total)
    assert remainder == 0

    #data = np.reshape(data, (n_frames, z_total, x, y))
    # TODO TODO slice a movie with the opposite of these indices and verify it's
    # same as what we'd get by reslicing as above (by checking equality)

    # TODO TODO what extra info we need if this fn is also supposed to drop
    # stuff beyond end of recording? maybe just leave that to other stuff and
    # make clear in doc?

    flyback_indices = np.concatenate([
        np.arange((z_total * i) + z, (z_total * i) + z_total)
        for i in range(n_volumes)
    ])

    return flyback_indices


def get_col_onset_offset_indices(df, possible_col_names, **kwargs):
    """Returns arrays onsets, offsets with appropriate indices in `df`.

    possible_col_names (str or tuple): can be either exact column name in
        `df` or an iterable of column names, where the first matching a column
        in `df` will be used.

    """
    col = _get_column(df, possible_col_names)

    # TODO have this (inside) probably warn if there are no threshold crossings
    # (and maybe compare thresh to max/min/dtype values in generating warning to
    # indicate if that might be the cause of the error, which i guess it
    # must...)
    onsets, offsets = threshold_crossings(col, **kwargs)

    # TODO maybe just have threshold_crossings make this check by default, w/ a
    # kwarg to override (which gets threaded through here)
    assert np.all(onsets < offsets), \
        'offset before onset OR mismatched number of the two'

    return onsets, offsets


def get_col_onset_offset_times(df, possible_col_names, **kwargs):
    """Returns arrays onsets, offsets with appropriate values from `df.time_s`.

    `df` must have a column `'time_s'`, as generated by `load_thorsync_hdf5`.

    possible_col_names (str or tuple): can be either exact column name in
        `df` or an iterable of column names, where the first matching a column
        in `df` will be used.

    """
    onsets, offsets = get_col_onset_offset_indices(df, possible_col_names,
        **kwargs
    )
    onset_times = df.time_s[onsets]
    offset_times = df.time_s[offsets]
    return onset_times, offset_times


def find_last_true(x):
    # TODO specify behavior + test in case there are NO True values
    # (probably just raise ValueError)
    """Returns the index of the last `True` in 1-dimensional `x`.
    """
    if len(x.shape) != 1:
        raise ValueError('input must be 1-dimensional')

    # may need to generalize this type checking...
    if x.dtype != np.dtype('bool'):
        raise ValueError('input must be of dtype bool')

    return len(x) - np.argmax(x[::-1]) - 1


# TODO add (fn specific?) cacheing util (decorator?) so that df can be generated
# automatically w/ thorsync dir input here / in assign_frames*, but so the df
# loaded in the background can be shared across these calls?
# TODO delete frame out name handling after forcing it to be a constant name in
# load_thorsync_hdf5
def get_frame_times(df, thorimage_dir, time_ref='mid',
    acquisition_trigger_names=None, _debug=False):
    # TODO update doc (+ maybe fn name) to reflect fact that initial acquisition
    # onset time also returned (and is this what i want to return? for zeroing
    # against odor stuff later)
    """Returns seconds from start of ThorSync recording for each frame.

    Arguments:
    df (`DataFrame`): as returned by `load_thorsync_hdf5`

    thorimage_dir (str): path to ThorImage directory to load metadata from

    time_ref ('mid' | 'end')

    Returns a `np.array` that should be of length equal to the number of frames
    actually saved by ThorImage.
    """
    # NOTE: initially I was planning on basing this off of one of the ThorLabs
    # supplied MATLAB scripts (see GenerateFrameTime.m referenced in ThorSync
    # manual), but it seems to not be behaving correctly (or my data does not
    # have the values for Frame_In that this script expects, as all of mine are
    # purely 0). even excluding the AND w/ Frame_In, however, the shape of
    # the `indexes` variable in this MATLAB script would not seem to be what I'd
    # expect (i.e. length is not equal to number of frames) for at least some of
    # my data (tried 2021-03-07/1/SyncData002).
    # If I had to guess, Frame_In is supposed to function as our copy of the
    # recording trigger does, but digital.
    # TODO try to make sure we are maintaining the same behavior as the official
    # thor provided matlab scripts (eh... nvm. see other comments explaining how
    # i don't think they are working correctly, with regards to Frame_In and
    # perhaps some other things):
    # From red "Note:" box on p37 of ThorSync3.0 user guide:
    # "Importing data into Matlab will automatically maintain the correct frame
    # reference by removing any unintended image frame(s) acquired during the
    # Trigger Out phase."

    if time_ref not in ('mid', 'end'):
        raise ValueError("time_ref must be either 'mid' or 'end'")

    if time_col not in df.columns:
        raise ValueError(f'{time_col} not in df.columns')

    frame_out_onsets, frame_out_offsets = get_col_onset_offset_indices(df,
        'frame_out', threshold=0.5
    )
    if _debug:
        # TODO delete if not useful
        frame_out_lens = frame_out_offsets - frame_out_onsets

        frame_out_onsets_s = df.time_s[frame_out_onsets].values
        frame_out_offsets_s = df.time_s[frame_out_offsets].values
        frame_out_lens_s = frame_out_offsets_s - frame_out_onsets_s

        # TODO probably replace w/ median / something (if used...)
        mean_frame_out_len = np.mean(frame_out_lens)
        print('mean_frame_out_len:', mean_frame_out_len)
        mean_frame_out_len_s = np.mean(frame_out_lens_s)

        min_frame_out_len = np.min(frame_out_lens)
        print('min_frame_out_len:', min_frame_out_len)

        max_frame_out_len = np.max(frame_out_lens)
        print('max_frame_out_len:', max_frame_out_len)
        max_frame_out_len_s = np.max(frame_out_lens_s)

        sorted_fo_lens = np.sort(frame_out_lens)
        sfo_n = 10
        print(f'min {sfo_n} frame_out_lens:', sorted_fo_lens[:sfo_n])
        print(f'max {sfo_n} frame_out_lens:', sorted_fo_lens[-sfo_n:])

    # NOTE: in all the single block data i've tested so far (though both of
    # these also happen to have been acquired downstairs...) the max is actually
    # ~= the mean, whereas that's not the case for the other data tested so far.
    # TODO check whether this explains why this data didn't work w/ matlab
    # provided GetFrameTimes.m / related
    #
    # min < mean ~= max:
    # - 2021-03-07/1
    #   - image: glomeruli_diagnostics_192
    #     sync: SyncData001
    #   - image: t2h_single_plane
    #     sync: SyncData002
    #
    # min ~= mean < max:
    # - 2019-01-23/6
    #   - image: _001
    #     sync: SyncData001
    # - 2020-04-01/2
    #   - image: fn_002
    #     sync: SyncData002

    # TODO TODO if frame_in can be recovered / configured to be saved in the
    # future, and it does indeed serve the same function as our
    # "scope_pin"/whatever, replace this with that (at least if it's available
    # in current data)
    if acquisition_trigger_names is None:
        acquisition_trigger_names = _acquisition_trigger_names

    acq_onsets, acq_offsets = get_col_onset_offset_indices(df,
        acquisition_trigger_names, threshold=2.5
    )

    _, _, z, c, n_flyback, _, xml = load_thorimage_metadata(thorimage_dir,
        return_xml=True
    )

    # Number of XY frames, even in the volumetric case. This is however, the
    # number of frames AFTER any frame averaging.
    n_frames = get_thorimage_n_frames_xml(xml)

    z_total = z + n_flyback

    # TODO delete after figuring out what should really be used
    #compare_to_acq_off = 'offset'
    compare_to_acq_off = 'onset'
    assert compare_to_acq_off in ('onset', 'offset')

    n_orig_frame_out_pulses = len(frame_out_onsets)
    #

    n_averaged_frames = get_thorimage_n_averaged_frames_xml(xml)

    if n_averaged_frames > 1:
        assert z_total == 1, ('ThorImage does not support averaging while '
            'recording fast Z'
        )

    # "Frame save groups" are either frames to be averaged to a single frame or
    # frames that together make up a volume (including any flyback frames!)
    n_frames_per_save_group = n_averaged_frames * z_total

    if _debug:
        print('n_blocks:', len(acq_onsets))
        print('z_total:', z_total)
        print('n_averaged_frames:', n_averaged_frames)
        print('N_FRAMES_PER_SAVE_GROUP:', n_frames_per_save_group)
        if n_frames_per_save_group == 1:
            print(f'comparing frame_out_{compare_to_acq_off.upper()}'
                ' to acq_offsets'
            )

    not_saved_bool = None
    on = acq_onsets[0]

    for off, next_on in zip_longest(acq_offsets, acq_onsets[1:]):
        if n_averaged_frames == 1 and z_total == 1:
            if compare_to_acq_off == 'onset':
                curr_not_saved_bool = off < frame_out_onsets

            elif compare_to_acq_off == 'offset':
                curr_not_saved_bool = off < frame_out_offsets

            else:
                assert False

            if next_on is not None:
                # Shouldn't matter which of the onsets/offsets we use to compare
                # to next_on.
                curr_not_saved_bool = np.logical_and(
                    curr_not_saved_bool, frame_out_offsets < next_on
                )

        else:
            # TODO TODO check whether we also need to do some advance filtering
            # in either of these cases (likely based on length of a given frame
            # out pulse, but also perhaps could always filter last one [/two] or
            # perhaps actually still need to do something things in relation to
            # acquisition trigger [sometimes])

            curr_not_saved_bool = np.zeros_like(frame_out_onsets,
                dtype=np.bool_
            )

            curr_frame_out_onsets_mask = on < frame_out_onsets

            if next_on is not None:
                curr_frame_out_onsets_mask = np.logical_and(
                    curr_frame_out_onsets_mask, frame_out_onsets < next_on
                )

            curr_n_frame_out_pulses = curr_frame_out_onsets_mask.sum()

            n_completed_frame_save_groups, n_trailing_unsaved_frames = divmod(
                curr_n_frame_out_pulses, n_frames_per_save_group
            )

            if _debug:
                print('curr_n_frame_out_pulses:', curr_n_frame_out_pulses)
                print('n_completed_frame_save_groups:',
                    n_completed_frame_save_groups
                )
                print('n_trailing_unsaved_frames:', n_trailing_unsaved_frames)

            # TODO TODO TODO try to find test cases where there are exactly
            # an even number of frame out pulses (both including and excluding
            # the very last one), in both averaging and fast-Z cases
            # (as another means of trying to find cases where an additional
            # filtering step is required)

            if n_trailing_unsaved_frames > 0:
                last_curr_idx = find_last_true(curr_frame_out_onsets_mask)
                i0 = last_curr_idx - n_trailing_unsaved_frames + 1
                curr_not_saved_bool[i0:(last_curr_idx + 1)] = True

            curr_n_dropped = curr_not_saved_bool.sum()

            assert curr_n_dropped == n_trailing_unsaved_frames, \
                f'{curr_n_dropped} != {n_trailing_unsaved_frames}'

        if _debug:
            print('curr_not_saved_bool.sum():', curr_not_saved_bool.sum())
            print()

        on = next_on

        if not_saved_bool is None:
            not_saved_bool = curr_not_saved_bool
        else:
            not_saved_bool = np.logical_or(not_saved_bool, curr_not_saved_bool)

    not_saved_indices = np.flatnonzero(not_saved_bool)
    frame_out_onsets = np.delete(frame_out_onsets, not_saved_indices)
    frame_out_offsets = np.delete(frame_out_offsets, not_saved_indices)

    # NOTE: without the .values call here, the 'mid' case below does not work
    # because pandas tries to align the series.
    onset_times = df[time_col].values[frame_out_onsets]
    offset_times = df[time_col].values[frame_out_offsets]

    if time_ref == 'end':
        frame_times = offset_times

    elif time_ref == 'mid':
        frame_times = (offset_times - onset_times) / 2 + onset_times

    if _debug:
        print('n_orig_frame_out_pulses:', n_orig_frame_out_pulses)

        n_frames_before_averaging = n_frames * n_averaged_frames
        print('n_frames_before_averaging:', n_frames_before_averaging)
        print('n_frames_after_dropping:', len(frame_times))
        e1 = n_frames_before_averaging - len(frame_times)

        n_frame_outs_to_drop = \
            n_orig_frame_out_pulses - n_frames_before_averaging

        print('n_frame_outs_to_drop:', n_frame_outs_to_drop)
        print('n_actually_dropped:', len(not_saved_indices))
        e2 = len(not_saved_indices) - n_frame_outs_to_drop
        assert e1 == e2
        print('EXCESS FRAMES DROPPED:', e1)
        print('\n')

    if n_averaged_frames > 1:
        frame_times = frame_times.reshape(-1, n_averaged_frames).mean(axis=-1)

    assert len(frame_times) == n_frames, f'{len(frame_times)} != {n_frames}'

    if z > 1:
        flyback_indices = get_flyback_indices(n_frames, z, n_flyback,
            frame_times
        )

        # https://stackoverflow.com/questions/47540800
        # This will raise `IndexError` if any exceed size of frame_times
        # (though it shouldn't unless I made a mistake)
        frame_times = np.delete(frame_times, flyback_indices)

        # (we basically already know, but just for the sake of it...)
        n_volumes, remainder = divmod(len(frame_times), z)
        assert remainder == 0

        frame_times = frame_times.reshape(-1, z).mean(axis=-1)
        assert len(frame_times) == n_volumes

    initial_acquisition_onset_time = df[time_col][acq_onsets[0]]

    return frame_times, initial_acquisition_onset_time


# TODO how should this deal w/ blocks[/similar block handling functions i might
# implement]
# TODO maybe move this function to util or something?
# TODO provide kwargs to crop (end of?) ranges so that all have same number of
# frames? might also be some cases where something similar is needed at start,
# especially when we have multiple blocks
def assign_frames_to_odor_presentations(df, thorimage_dir,
    odor_timing_names=None, **kwargs):
    """Returns list of (start, end) frame indices, one per odor presentation.

    Frames are indexed as they are along the first dimension of the movie,
    including for volumetric data (where a scalar index of this dimension will
    produce a volume) or data collectd via frame averaging.

    End frames are included in range, and thus getting a presentation must be
    done like `movie[start_i:(end_i + 1)]` rather  than `movie[start_i:end_i]`.

    Not all frames necessarily included. No overlap.
    """
    if odor_timing_names is None:
        odor_timing_names = _odor_timing_names

    # TODO finish refactoring (do i want to thread onset/offset kwargs through
    # this fn?)
    '''
    odor_onsets, odor_offsets = get_col_onset_offset_indices(df,
        odor_timing_names, threshold=2.5
    )
    '''

    odor_timing = _get_column(df, odor_timing_names)

    # (when the valve(s) are given the signal to open)
    odor_onsets, _ = threshold_crossings(odor_timing, threshold=2.5,
        offsets=False
    )
    odor_onset_times = df[time_col].values[odor_onsets]

    frame_times, _ = get_frame_times(df, thorimage_dir, **kwargs)

    # (with respect to the odor onset in both of these cases)
    first_frame_to_odor_s = odor_onset_times[0] - frame_times[0]
    #print('first_frame_to_odor_s:', first_frame_to_odor_s)

    start_times = odor_onset_times - first_frame_to_odor_s
    #print([x in frame_times for x in start_times])

    # If tied, seems to provide index that would insert the new element at the
    # earlier position.
    start_frames = np.searchsorted(frame_times, start_times)

    # Inclusive (so can NOT be used as the end of slices directly. need to add
    # one because slice ends are not inclusive.)
    end_frames = [x - 1 for x in start_frames[1:]] + [len(frame_times) - 1]

    # TODO TODO probably want to at least check for discontinuities within the
    # frames assigned to a given presentation, particularly in case where there
    # are multiple blocks

    return list(zip(start_frames, end_frames))


# TODO rename to indicate it's parsing from directory name?
def old_fmt_thorimage_num(x):
    # TODO provide example(s) of format in docstring

    if pd.isnull(x) or not (x[0] == '_' and len(x) == 4):
        return np.nan
    try:
        n = int(x[1:])
        return n
    except ValueError:
        return np.nan


# TODO rename to indicate it's parsing from directory name?
def new_fmt_thorimage_num(x):
    # TODO provide example(s) of format in docstring

    parts = x.split('_')
    if len(parts) == 1:
        return 0
    else:
        return int(x[-1])


thorsync_dir_prefix = 'SyncData'
def thorsync_num(thorsync_dir):
    """Returns number in suffix of ThorSync output directory name as an int.
    """
    return int(thorsync_dir[len(thorsync_dir_prefix):])


def is_thorsync_dir(d, verbose=False):
    """True if dir has expected ThorSync outputs, False otherwise.
    """
    if not isdir(d):
        return False
    
    # No matter how many directory levels `d` contains, `listdir` only returns
    # the basename of each file, not any preceding part of the path.
    files = {f for f in listdir(d)}

    have_settings = False
    have_h5 = False
    for f in files:
        if f == thorsync_xml_basename:
            have_settings = True

        if is_thorsync_h5(f):
            have_h5 = True

    if verbose:
        print('have_settings:', have_settings)
        print('have_h5:', have_h5)

    return have_h5 and have_settings


def is_thorimage_raw(f):
    """True if filename indicates file is ThorImage raw output.
    """
    _, f_basename = split(f)

    # Needs to match at least 'Image_0001_0001.raw' and 'Image_001_001.raw'
    if f_basename.startswith('Image_00') and f_basename.endswith('001.raw'):
        return True

    return False


def is_thorimage_dir(d, verbose=False):
    """True if dir has expected ThorImage outputs, False otherwise.

    Looks for .raw not any TIFFs now.
    """
    if not isdir(d):
        return False
   
    # No matter how many directory levels `d` contains, `listdir` only returns
    # the basename of each file, not any preceding part of the path.
    files = {f for f in listdir(d)}

    have_xml = False
    have_raw = False
    # TODO support tif output case(s) as well
    have_tiff = False
    for f in files:
        if f == thorimage_xml_basename:
            have_xml = True

        # TODO TODO would probably fail if experiment was configured to save
        # TIFF output? or does it also save .raw in that case? fix if not.
        elif is_thorimage_raw(f):
            have_raw = True

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
    use_ranking=True, check_against_naming_conv=False,
    check_unique_thorimage_nums=None, verbose=False):
    """
    Takes lists (not necessarily same len) of dirs, and returns a list of
    lists of matching (ThorImage, ThorSync) dirs (sorted by experiment time).

    Args:
    check_against_naming_conv (bool): (default=False) If True, check ordering
        from pairing is consistent with ordering derived from our naming
        conventions for Thor software output.

    check_unique_thorimage_nums (bool): If True, check numbers parsed from
        ThorImage directory names, as-per convention, are unique.
        Requires check_against_naming_conv to be True. Defaults to True if
        check_against_naming_conv is True, else defaults to False. 

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

    xml_fname = sep.join(parts[:-2] + [thorimage_id, thorimage_xml_basename])
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

