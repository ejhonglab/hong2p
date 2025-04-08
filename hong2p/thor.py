"""
Functions for working with ThorImage / ThorSync outputs, including for dealing
with naming conventions we use for outputs of these programs.
"""

from os import listdir
from os.path import join, split, sep, exists, isdir, normpath, getmtime, abspath
# TODO replace w/ stock ElementTree name? no good justification for this renaming
import xml.etree.ElementTree as etree
from xml.etree.ElementTree import Element
from datetime import datetime, timedelta
import warnings
from pprint import pprint, pformat
import glob
from itertools import zip_longest
import functools
from pathlib import Path
import re
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd

from hong2p.types import Pathlike, PathPair


# Note: these other names may get converted to these via hdf5_dataset_rename_dict/etc
_acquisition_trigger_names = ('scope_pin',)
_odor_timing_names = ('olf_disp_pin',)

DIGITAL_THRESHOLD = 0.5

# Since some electrical bug (?) on downstairs scope has led to ~4v pulses before and
# after recording. Used to use 2.5 before that.
#
# This change didn't fix handling of the 2022-10-07/1/megamat0_part2 data I was
# expecting it to fix though...
ANALOG_0_TO_5V_THRESHOLD = 4.5


class OnsetOffsetNumMismatch(Exception):
    pass

class NotAllFramesAssigned(Exception):
    pass


def xmlroot(xml_path: str) -> Element:
    """Loads contents of xml_path into xml.etree.ElementTree and returns root.

    Use calls to <node>.find(<child name>) to traverse down tree and at leaves,
    use <leaf>.attrib[<attribute name>] to get values. There are other functions
    too, but see `xml` documentation for more information.
    """
    return etree.parse(xml_path).getroot()


# TODO maybe rename everything to get rid of 'get_' prefix? mainly here so i
# can name what these functions return naturally without shadowing...

thorimage_xml_basename = 'Experiment.xml'
def get_thorimage_xml_path(thorimage_dir: str) -> str:
    """Takes ThorImage output dir to (expected) path to its XML output.

    Raises IOError if either thorimage_dir or Experiment.xml contained within it do not
    exist.
    """
    if not isdir(thorimage_dir):
        raise IOError(f'thorimage_dir {thorimage_dir} does not exist!')

    xml_path = join(thorimage_dir, thorimage_xml_basename)
    if not exists(xml_path):
        raise IOError(f'{thorimage_xml_basename} did not exist in {thorimage_dir}')

    return xml_path


# TODO does this work?
PathOrXML = Union[str, Element]


# TODO TODO now that this behaves as identity if given xml object, actually use that to
# collapse some of the functions w/ and w/o _xml suffix and allow those of only one type
# to work with both types of input
# TODO should i relax typehing to something like PathlikeOrXML or nah?
def get_thorimage_xmlroot(thorimage_dir_or_xmlroot: PathOrXML) -> Element:
    """Takes ThorImage output dir to object w/ XML data.

    Returns the input without doing anything if it is already the same type of XML
    object that would be returned, to allow writing functions that can either be given
    paths to ThorImage directories or re-use an already loaded representation of its
    XML.
    """
    if isinstance(thorimage_dir_or_xmlroot, Element):
        return thorimage_dir_or_xmlroot

    thorimage_dir = thorimage_dir_or_xmlroot

    xml_path = get_thorimage_xml_path(thorimage_dir)
    return xmlroot(xml_path)


# TODO doc
def thorimage_xml(fn_taking_xml):
    """Converts an attribute lookup fn taking XML to allow ThorImage directory input.
    """

    @functools.wraps(fn_taking_xml)
    def fn_taking_path_or_xml(thorimage_dir_or_xmlroot: PathOrXML, **kwargs):

        xml = get_thorimage_xmlroot(thorimage_dir_or_xmlroot)
        return fn_taking_xml(xml, **kwargs)

    return fn_taking_path_or_xml


@thorimage_xml
def get_thorimage_time(xml) -> datetime:
    """Takes etree XML root object to recording start time.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    date_ele = xml.find('Date')
    from_date = datetime.strptime(date_ele.attrib['date'], '%m/%d/%Y %H:%M:%S')
    from_utime = datetime.fromtimestamp(float(date_ele.attrib['uTime']))
    assert (from_date - from_utime).total_seconds() < 1
    return from_utime


@thorimage_xml
def get_thorimage_n_frames(xml, without_flyback=False, num_volumes=False):
    """Returns the number of XY planes (# of timepoints) in the recording.

    This is the number of frames *after* any averaging configured in ThorImage.

    Any flyback frames are included.

    If additional color channels are enabled but other parameters remain the same, this
    number will not change.

    Args:
        without_flyback: if True, subtract the number of flyback frames (if any)

        num_volumes: if True, return number of volumes instead of number of XY frames.
            since there are a fixed number of flyback frames per volume, this option
            will return the same number regardless of without_flyback.
    """
    if num_volumes:
        without_flyback = True

    n_raw_xy_frames = int(xml.find('Streaming').attrib['frames'])
    if not without_flyback:
        return n_raw_xy_frames

    n_flyback = get_thorimage_n_flyback_xml(xml)

    z = get_thorimage_z_xml(xml)
    z_total = z + n_flyback

    n_volumes, remainder = divmod(n_raw_xy_frames, z_total)
    assert remainder == 0

    if num_volumes:
        return n_volumes
    else:
        return n_raw_xy_frames - (n_flyback * n_volumes)


@thorimage_xml
def is_fast_z_enabled_xml(xml) -> bool:
    streaming = xml.find('Streaming')
    if streaming.attrib['enable'] != '1':
        # zFastEnable can still be 1 when we aren't doing fast Z (e.g. in some
        # anatomical stack data)
        return False

    return streaming.attrib['zFastEnable'] == '1'


def _get_zstage(xml):
    # TODO assert there is only one ZStage object? is it always only primary (assumed
    # piezo) saved into xml, even if there is a stepper secondary z-axis too?
    zstage = xml.find('ZStage')

    # TODO replace .find w/ findall(...)[0]?
    #
    # yang's test XML she sent has multiple ZStage elements, but only one with name
    # defined (='ThorZPiezo')
    assert len([x for x in xml.findall('ZStage') if 'name' in x.attrib]) == 1

    # TODO and if fastZ is enabled, is that sufficient evidence it is piezo?
    # NO! (at least it's possible to misconfigure downstairs system so that non-piezo
    # stage is selected for both primary/secondary, and it will still let you collect a
    # recording with fast Z apparently enabled)

    return zstage


@thorimage_xml
def get_thorimage_z_xml(xml) -> int:
    """Returns number of different Z depths measured in ThorImage recording.

    Does NOT include any flyback frames there may be.
    """
    zstage = _get_zstage(xml)
    z = int(zstage.attrib['steps'])
    assert z > 0
    return z


@thorimage_xml
def get_thorimage_z_stream_frames(xml) -> int:
    """Returns number of different Z depths measured in ThorImage recording.

    Does NOT include any flyback frames there may be.
    """
    zstage = _get_zstage(xml)
    n = int(zstage.attrib['zStreamFrames'])
    # TODO warn if streaming acqusition mode (or something else to rule out anatomical
    # recordings) AND is_fast_z_enabled_xml(xml) is False?
    assert n > 0
    return n


@thorimage_xml
def get_thorimage_zstep_um(xml) -> float:
    zstage = _get_zstage(xml)
    return float(zstage.attrib['stepSizeUM'])


# TODO maybe add a function to get expected movie.size from thorimage .raw
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


@thorimage_xml
def get_thorimage_n_channels_xml(xml) -> int:
    pmt = xml.find('PMT').attrib

    # It does seem that for channel B (perhaps also A but not C/D), you can have it
    # enabled with zero gain and it will still have corresponding data in the file.
    def is_enabled(channel):
        return int(pmt['enable' + channel])

    # TODO also check gain is nonzero? or maybe we don't want that since it seems some
    # data can have one channel w/ zero gain yet still "data" occupies that space in the
    # (at least raw) file format

    # Technically ThorImage metadata seems to go out to C and D as well, but even if
    # those are accidentally enabled, it seems that no additional garbage data going in
    # the .raw files at least (maybe it does into one of the TIFF output formats?), so
    # I'm just not checking C and D.
    n_channels = sum([is_enabled(c) for c in ['A', 'B']])
    return n_channels


# TODO probably also include number of "frames" (*planes* over time) here too
# (and in functions that call this)
# (though would need to take into account flyback as well as potentially
# averaging in order to have this dimension reflect shape of movie (as if the
# output of this function were `movie.shape` for the corresponding movie))
@thorimage_xml
def get_thorimage_dims(xml):
    """Takes etree XML root object to (xy, z, c) dimensions of movie.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    # TODO exclude ~empty LSM tags, when multiple (finding first *should* still work),
    # like in test data yang sent. attrib['name'] should only be defined for one tag?
    # TODO factor out LSM tag getting to do this (-> share w/ other fns getting it)
    lsm_attribs = xml.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x, y)

    # TODO what is Streaming -> flybackLines? (we already have flybackFrames...)

    z = get_thorimage_z_xml(xml)

    c = get_thorimage_n_channels_xml(xml)

    # may want to add ZStage -> stepSizeUM to TIFF metadata?

    return xy, z, c


@thorimage_xml
def get_thorimage_pixelsize_um(xml):
    """Takes etree XML root object to XY pixel size in um.

    Pixel size in X is the same as pixel size in Y.

    XML object should be as returned by `get_thorimage_xmlroot`.
    """
    lsm = xml.find('LSM')

    # TODO put behind a checks= bool kwarg?
    # TODO does thorimage (and their xml) ever support unequal x and y resolution?
    pixelsize_x = float(lsm.attrib['widthUM']) / float(lsm.attrib['pixelX'])
    pixelsize_y = float(lsm.attrib['heightUM']) / float(lsm.attrib['pixelY'])
    assert np.isclose(pixelsize_x, pixelsize_y), f'{pixelsize_x=} != {pixelsize_y}'

    pixelsize_xy = float(lsm.attrib['pixelSizeUM'])

    # pixelSizeUM is entered with 3 digits after decimal place in XML
    # [width|height]UM both have 5 sig figs in XML, with pixel[X|Y] generally having 3.
    #
    # atol=6e-4 should ignore anything different beyond pixelSizeUM's 3 digits after
    # decimal (and assuming rounding going into that)
    assert np.isclose(pixelsize_xy, pixelsize_x, atol=6e-4), \
        f'{pixelsize_xy=} != {pixelsize_x=}'

    return pixelsize_xy


# TODO replace w/ wrapped version + remove _xml suffix (and replace all calls similarly)
def get_thorimage_n_averaged_frames_xml(xml):
    """Returns how many frames ThorImage averaged for a single output frame.
    """
    lsm_attribs = xml.find('LSM').attrib

    # TODO is this correct handling of averageMode?
    average_mode = int(lsm_attribs['averageMode'])

    if average_mode == 0:
        n_averaged_frames = 1
    else:
        # TODO this sufficient check that frame averaging broken, or also/only check
        # z>1?
        # TODO is there some cutoff version beyond which volumetric frame averaging is
        # supported for real?
        # NOTE: i think all thorimage versions we are using don't actually support frame
        # averaging if recording volumetrically. raw output will have all frames despite
        # value of this.
        if is_fast_z_enabled_xml(xml):
            # TODO flag to disable warning?
            name = get_thorimage_name(xml)
            warnings.warn(f'{name}: XML indicates frame averaging was configured, '
                'but our ThorImage versions do not actually support this when fastZ '
                'is enabled. setting n_averaged_frames=1.'
            )

            n_averaged_frames = 1
        else:
            n_averaged_frames = int(lsm_attribs['averageNum'])

    return n_averaged_frames


# TODO replace w/ wrapped version + remove _xml suffix (and replace all calls similarly)
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


@thorimage_xml
def get_thorimage_n_flyback_xml(xml):
    if is_fast_z_enabled_xml(xml):
        streaming = xml.find('Streaming')
        n_flyback_frames = int(streaming.attrib['flybackFrames'])
    else:
        n_flyback_frames = 0

    return n_flyback_frames


@thorimage_xml
def get_thorimage_notes(xml):
    return xml.find('ExperimentNotes').attrib['text']


@thorimage_xml
def get_thorimage_name(xml):
    return xml.find('Name').attrib['name']


@thorimage_xml
def get_thorimage_version(xml):
    # e.g. '3.0.2016.10131'. no other attribs under Software tag (on this older
    # ThorImage version, at least)
    return xml.find('Software').attrib['version']


@thorimage_xml
def get_thorimage_scannertype(xml):
    lp = xml.find('LightPath').attrib
    # TODO can there ever be multiple LightPath tags (yes, but first should probably
    # still work. exclude elements with nothing or only cam enabled. share logic w/
    # other places i want to get a unique tag)? (test on output from systems w/
    # camera enabled too?)

    lsm_name = xml.find('LSM').attrib['name']

    if lp['GalvoGalvo'] == '1':
        scanner = 'GalvoGalvo'
        assert lsm_name == 'GalvoGalvo'
    else:
        assert lp['GalvoResonance'] == '1'
        scanner = 'GalvoResonance'
        # NOTE: different str from above. this is the case in at least output from
        # 4.3.2023.6261 Yang sent me
        assert lsm_name == 'ResonanceGalvo'

    return scanner


# TODO TODO delete this fn / fix. ThorStage does (hopefully) not seem at all to imply we
# are not using a piezo. probably need to check fast z.
# TODO TODO what happens if you try to do fast z downstairs w/ just non-piezo zstage? or
# w/ it as primary? does it work? data issues? slower? ideally, we could tell from xml
# output, or it wouldn't allow recording...
@thorimage_xml
def get_thorimage_zstage_type(xml):
    # TODO TODO fastZ enabled imply that a ThorStage is actually piezo?
    # NO! (at least it's possible to misconfigure downstairs system so that non-piezo
    # stage is selected for both primary/secondary, and it will still let you collect a
    # recording with fast Z apparently enabled)

    # NOTE: enable=1 set in yang's newer test data, but not referenced in one of mine.
    zstage = _get_zstage(xml).attrib
    regtype = zstage['name']
    assert regtype in ('ThorStage', 'ThorZPiezo')
    return regtype


@thorimage_xml
# TODO modify to return a third item, a dict w/ any remaining settings relevant to the
# particular power regulator (e.g. w/ 'offset' for 'non-pockel', or 'minV'/'maxV'/etc
# for pockel)?
def get_thorimage_power_regtype_and_level(xml) -> Tuple[str, float]:
    """Returns `regtype`, `power_level` where `regtype` is either 'pockel'|'non_pockel'
    """
    # looks like if there is any <Pockels> tag w/ start="<nonzero>" (presumably ==
    # stop), then it's a pockel?
    #
    # there can still be <PowerRegulator[2]> elements w/ enable="1", just start/stop
    # should be 0

    # first element should be fine. seems to be only one that has non-zero start/stop
    # ever, at least in 3 test outputs from diff systems.
    pockels = xml.find('Pockels').attrib
    pockel_start = float(pockels['start'])
    pockel_stop = float(pockels['stop'])
    # would need to support (and don't think we ever use)
    assert pockel_start == pockel_stop

    reg = xml.find('PowerRegulator').attrib
    reg_start = float(reg['start'])
    reg_stop = float(reg['stop'])
    # would need to support (and don't think we ever use)
    assert reg_start == reg_stop

    if pockel_start > 0:
        assert reg_start == 0
        # TODO blank percentage? min/maxV? always return these other settings in 3rd
        # return arg as dict (and use for offset above?)?
        return 'pockel', pockel_start
    else:
        assert reg_start > 0
        # TODO better str here ('waveplate'? accurate?)?
        # TODO also return offset?
        return 'non-pockel', reg_start


@thorimage_xml
def print_xml(xml: Element) -> None:
    encoding = 'utf-8'
    print(etree.tostring(xml, encoding).decode(encoding))


# TODO unit test
def _parse_driver_and_indicator(fly_str: str, debug: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:

    # TODO move a \b from indicator to uas regex? have at start of both? make
    # optional in one/both?
    # works, at least for all but negative lookahead within driver
    #uas_regex = 'u?(?:as)?'
    uas_regex = '(?:u(?:as)?)'
    # TODO explicitly exclude u[as] match in driver_regex?
    # (e.g. so that 'uas-6f' doesn't parse as driver='uas' indicator='6f')
    driver_regex = r'\b(?P<driver>\w+)\s?-\s?g?(?:al)?4?'
    # TODO fix? was trying to get "negative lookahead assertion" to work to exclude
    # uas from matching in driver portion
    # https://stackoverflow.com/questions/5030041
    # (but can't get this one to match anything)
    #driver_regex = f'\\b(?P<driver>^(?!{uas_regex})\\w+)\\s?-\\s?g?(?:al)?4?'

    # TODO fix how we aren't matching e.g. '6f' alone
    # (or add a hack to try matching w/ just indicator_regex?)
    #
    # worked w/ commented uas_regex above
    #indicator_regex = f'\\b{uas_regex}-?g?(?:camp)?(?P<indicator>[6-9][fms])\\b'
    indicator_regex = f'{uas_regex}?-?g?(?:camp)?(?P<indicator>[6-9][fms])\\b'

    # making driver_regex optional this way seems to break it... not sure why
    # fly_str='5 day old pb-Gal4/+;+;UAS-G6f/+ from prat'
    # ipdb> re.findall(f'{driver_regex}.*{indicator_regex}', fly_str, flags=re.IGNORECASE)
    # [('pb', '6f')]
    # ipdb> re.findall(f'(?:{driver_regex}).*{indicator_regex}', fly_str, flags=re.IGNORECASE)
    # [('pb', '6f')]
    # ipdb> re.findall(f'(?:{driver_regex})?.*{indicator_regex}', fly_str, flags=re.IGNORECASE)
    # [('', '6f')]

    # e.g. '5 day old pb-Gal4/+;+;UAS-G6f/+ from prat' -> ('pb', '6f')
    # ';' and '/' seem to count as a word boundaries (\b)
    # TODO can i make driver_regex optional? can i do that be wrapping it w/ another
    # non-matching group here, or can't nest?
    matches = re.findall(f'{driver_regex}.*{indicator_regex}', fly_str,
        flags=re.IGNORECASE
    )
    driver = None
    indicator = None
    if len(matches) > 0:
        assert len(matches) == 1
        driver, indicator = matches[0]

        driver = driver.lower()
        # hack since i couldn't figure out how to exclude uas_regex match from
        # <driver> group in driver_regex above
        if re.match(f'^{uas_regex}$', driver):
            driver = None
    else:
        # hack since i couldn't figure out how to get the one large regex to match
        # indicator strs by themselves
        matches = re.findall(f'\\b{indicator_regex}', fly_str, flags=re.IGNORECASE)
        # TODO test this branch
        if len(matches) > 0:
            assert len(matches) == 1
            indicator = matches[0]
            assert type(indicator) is str

    if indicator is not None:
        indicator = indicator.lower()

    return driver, indicator


@thorimage_xml
def parse_thorimage_notes(xml, *, debug: bool = False) -> dict:
    """Returns dict of metadata, with `<key>: <val>` lines and rest parsed separately.

    Args:
        thorimage_dir_or_xml: path to ThorImage output directory or XML Element
            containing parsed contents of the corresponding Experiment.xml file.

    Lines not matching the `<key>: <val>` format will be appended together under the
    'prose' key in the returned dict.

    It is assumed there will be a single line with the YAML path from `olf`, and this
    line is not included in output (should be handled separately, via
    `util.stimulus_yaml_from_thorimage`, and would only add noise in dealing with what
    remains here).
    """
    notes = get_thorimage_notes(xml)
    recording_start_time = get_thorimage_time(xml)
    # TODO delete
    debug = True
    #
    if debug:
        print()
        print('notes:')
        print(notes)

    # TODO unit test?
    # TODO rename (_parse_power?)
    def _match_power(power_str) -> Optional[dict]:
        m = re.match(r'~?(?P<power_mw>\d+(\.\d*)?)\s?m[wW]\s*(?P<power_note>.*)\s*',
            power_str
        )
        if m is None:
            return m

        power_dict = m.groupdict()
        assert ('power_mw' in power_dict and 'power_note' in power_dict
            and len(power_dict) == 2
        )
        assert power_dict['power_mw'] is not None
        power_dict['power_mw'] = float(power_dict['power_mw'])
        return power_dict

    data = dict()
    non_dict_lines = []
    n_yaml_parts = 0
    n_power_lines = 0
    # stimulus_yaml_from_thorimage uses .split() instead of .splitlines(), so just want
    # to check that I never actually had any experiments where a line has a YAML path
    # AND any other non-whitespace chars. would be simpler to just use .splitlines() in
    # both places, if checks below never fail.
    for line in notes.splitlines():
        line = line.strip()

        if line.endswith('.yaml'):
            n_yaml_parts += 1
            # should be the single line containing yaml_path (the path parsed by
            # hong2p.util.stimulus_yaml_from_thorimage)
            continue

        parts = line.split()
        # see comment above if this ever fails
        assert not any(p.strip().endswith('.yaml') for p in parts)

        p0 = parts[0]
        # seems this should be behavior of <str>.strip() no matter the amount of
        # whitespace
        assert p0 == p0.strip()

        if not p0.endswith(':'):
            # TODO may need to make parseing more complex (or just use a regex that
            # allows whitespace between key and ':' or something?), if this fails
            assert ':' not in line

            power_dict = _match_power(line)
            if power_dict:
                # TODO refactor to share?
                assert not any(k in data for k in power_dict.keys())
                data.update(power_dict)
                n_power_lines += 1
                #
                continue

            non_dict_lines.append(line)
            continue

        assert len(p0) > 1
        key = p0[:-1]
        value = ' '.join(parts[1:])

        assert key != 'prose'

        if key == 'power':
            power_dict = _match_power(value)
            # TODO refactor to share?
            try:
                assert power_dict is not None, f'{value=}'
            except AssertionError:
                warnings.warn(f"could not parse power from '{value}'")
                # TODO or put this whole line into power_note instead?
                non_dict_lines.append(line)
                continue

            assert not any(k in data for k in power_dict.keys())
            data.update(power_dict)
            n_power_lines += 1
            #
            continue

        assert key not in data
        data[key] = value

    data['prose'] = '\n'.join(non_dict_lines)

    # TODO or maybe <= 1 (if some recordings don't have this path, which should be true
    # for anatomical recording)
    assert n_yaml_parts == 1

    # not guaranteed. should ffill from previous recordings on same fly.
    assert n_power_lines <= 1

    if 'fly' in data:
        # TODO also support strings like this:
        # 'pb>6f (same lines/genetics as all recent experiments), ecclosed 11/14. from
        # sam.'
        # TODO TODO why current date parsing not working w/ ecclosed 11/14?

        # example strings to match:
        # '5 day old pb-Gal4/+;+;UAS-G6f/+ from prat'
        fly_str = data['fly']

        # TODO is this guaranteed to match largest number of chars for n_days_old that
        # would satisfy \d+? using search (vs match) in case i didn't always say this at
        # start of fly line value
        m = re.search(r'\b(?P<n_days_old>\d+)\s*days?(?:\sold)?\b', fly_str)
        if m:
            assert m['n_days_old'] is not None
            n_days_old = int(m['n_days_old'])
            # see comment below on why i'm assuming it starts from 1
            assert n_days_old >= 1
            # TODO delete
            #assert n_days_old >= 0
            assert 'n_days_old' not in data
            data['n_days_old'] = n_days_old
            if debug:
                print(f'{n_days_old=}')
        else:
            if debug:
                print('no match for n_days_old!')

            # TODO similar post-processing of 'odors' value (look for date)?
            # example odor strings to match:
            # (actually do i want to match this? can i always rely on just getting first
            # date?)
            # '3/27 except few diagnostics changed on 4/11'

            # TODO should there need to be an 'ec[c]losed ' prefix? is there always tho?
            # 'pb-Gal/x;;U-G6f/+ eclosed 4/23'
            matches = re.findall(r'\b(?P<month>\d\d?)/(?P<day>\d\d?)\b', fly_str)
            if matches:
                assert all(len(m) == 2 for m in matches)

                # only one date should be in line
                assert len(matches) == 1
                month, day = matches[0]

                # TODO warn if recording_start_time is in first few weeks of year?
                eclosion_date = datetime(year=recording_start_time.year,
                    month=int(month), day=int(day)
                )

                # TODO TODO are flies considered 1 or 0 days old on eclosion day?
                # (let's operate under assumption it's 1 for now)
                # what have i been doing before? what does prat think? what's min of
                # n_days_old values parsed directly in branch above?
                curr = recording_start_time - timedelta(hours=6)

                # this seems to be essentially taking floor, since seconds/etc fields
                # can be >=0, but are ignored by using .days (e.g. so any time on 4/26
                # is still 3 days after eclosion_date=4/23)
                n_days_old = (curr - eclosion_date).days
                data['n_days_old'] = n_days_old
                if debug:
                    print(f'{eclosion_date=}')
                    print(f'{curr=}')
                    print(f'{n_days_old=}')

            else:
                if debug:
                    print('no match for eclosion_date!')

        driver, indicator = _parse_driver_and_indicator(fly_str, debug=debug)
        assert 'driver' not in data
        data['driver'] = driver
        assert 'indicator' not in data
        data['indicator'] = indicator

    # there should also not be any whitespace-only string values at this point
    data = {k: v if v != '' else None for k, v in data.items()}

    if debug:
        print('data:')
        pprint(data)
        print()

    return data


def load_thorimage_metadata(thorimage_dir: Pathlike, return_xml=False):
    """Returns (fps, xy, z, c, n_flyback, raw_output_path) for ThorImage dir.

    Returns xml as an additional final return value if `return_xml` is True.
    """
    thorimage_dir = Path(thorimage_dir)
    xml = get_thorimage_xmlroot(thorimage_dir)

    # TODO TODO in volumetric streaming case (at least w/ input from thorimage
    # 3.0 from downstairs scope), this is the xy fps (< time per volume). also
    # doesn't include flyback frame. probably want to convert it to
    # volumes-per-second in that case here, and return that for fps. just need
    # to check it doesn't break other stuff.
    fps = get_thorimage_fps_xml(xml)
    xy, z, c = get_thorimage_dims(xml)

    n_flyback_frames = get_thorimage_n_flyback_xml(xml)
    if z == 1:
        assert n_flyback_frames == 0, 'n_flyback_frames > 0 but z == 1'

    # So far, I have seen this be one of:
    # - Image_0001_0001.raw
    # - Image_001_001.raw
    # ...but not sure if there any meaning behind the differences.
    imaging_files = list(thorimage_dir.glob('Image_*.raw'))

    if len(imaging_files) == 0:
        raise IOError(f'no .raw files in ThorImage directory {thorimage_dir}')

    elif len(imaging_files) > 1:
        raise RuntimeError('multiple .raw files in ThorImage directory '
            f'{thorimage_dir}. ambiguous!'
        )

    imaging_file = imaging_files[0]

    # TODO probably return as some kind of dict / dataclass, and always return xml while
    # we are at it
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
    # TODO is there not a timestamp embedded?
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


def get_thorsync_h5(thorsync_dir: Pathlike):
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
def read_movie(thorimage_dir: Pathlike, discard_flyback: bool = True,
    discard_channel_b: bool = False, checks: bool = True, _debug: bool = False):
    """Returns (t,[z,]y,x) indexed timeseries as a numpy array.
    """
    thorimage_dir = Path(thorimage_dir)
    fps, xy, z, c, n_flyback, imaging_file, xml = load_thorimage_metadata(thorimage_dir,
        return_xml=True
    )

    x, y = xy

    # From ThorImage manual: "unsigned, 16-bit, with little-endian byte-order"
    dtype = np.dtype('<u2')

    pmt = xml.find('PMT').attrib
    if checks:
        assert int(pmt['enableA']) and int(pmt['gainA']) > 0, 'channel A not used'

    if int(pmt['enableB']):
        if int(pmt['gainB']) == 0:
            warnings.warn('channel B was enabled but gain was zero. discarding.')
            discard_channel_b = True

        if not discard_channel_b:
            raise NotImplementedError('you may set discard_channel_b=True for now')
    else:
        # Not warning / erroring in this case so that discard_channel_b can be specified
        # for data that may or may not have channel B, without extra nuisance.
        discard_channel_b = False

    with open(imaging_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)

    # TODO maybe just don't read the data known to be flyback frames?

    n_frame_pixels = x * y
    n_frames = len(data) // n_frame_pixels
    assert len(data) % n_frame_pixels == 0, 'apparent incomplete frames'

    n_frames, remainder = divmod(n_frames, c)
    assert remainder == 0

    # This does not fail in the volumetric case, because 'frames' here
    # refers to XY frames there too.
    # TODO test this assertion on all data, though perhaps via a new function
    # get get expected n_frames from size of .raw file + other metadata
    # (mentioned in comments above get_thorimage_dims)
    assert n_frames == get_thorimage_n_frames(xml), \
        f'{n_frames} != {get_thorimage_n_frames(xml)}'

    # TODO delete?
    if _debug:
        print(f'read_movie: initial {data.shape=}')
        print(f'read_movie: {x=} {y=} {z=} {n_frames=}')
    #

    # TODO how to reshape if there are also multiple channels?

    # TODO TODO TODO just delete the data that needed special casing here unless it
    # actually seems like it might be useful -> delete the special casing in the code
    if z > 0:
        # TODO TODO some way to set pockel power to zero during flyback frames?
        # not sure why that data is even wasting space in the file...
        # TODO possible to skip reading the flyback frames? maybe it wouldn't
        # save time though...
        # just so i can hardcode some fixes based on part of path (assuming
        # certain structure under mb_team, at least for the inputs i need to
        # fix)
        thorimage_dir = thorimage_dir.resolve()
        date_part = str(thorimage_dir).split(sep)[-3]
        try_to_fix_flyback = False

        # TODO do more testing to determine 1) if this really was a flyback issue and
        # not some bug in thor / some change in thor behavior on update, and 2) what is
        # appropriate flyback [and which change from 2020-04-* stuff is what made this
        # flyback innapropriate? less averaging?]
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
        data = np.reshape(data, (n_frames, z_total, c, y, x))

        if discard_flyback:
            data = data[:, :z]
    else:
        # TODO test multi-channel handling in this case
        data = np.reshape(data, (n_frames, c, y, x))

    # TODO delete
    if _debug:
        print(f'read_movie: after reshaping {data.shape=}')
    #

    if discard_channel_b:
        # TODO delete?
        '''
        if _debug:
            from hong2p.viz import image_grid
            import matplotlib.pyplot as plt

            mean = data.mean(axis=0)

            if len(mean.shape) == 4:
                ch1_images = mean[:, 0]
                ch2_images = mean[:, 1]

            elif len(mean.shape) == 3:
                ch1_images = [mean[0]]
                ch2_images = [mean[1]]

            else:
                assert False

            # To visually verify that the data in this channel (when gain is zero) is
            # not meaningful and can be discarded
            image_grid(ch1_images)
            image_grid(ch2_images)
            plt.show()
        '''

        slices = [slice(None)] * len(data.shape)
        # TODO delete
        if _debug:
            print(f'read_movie: {slices=}')
            print(f'read_movie: assuming {slices[-3]=} corresponds to color channel dim')
        #

        # (the channel dimension)
        slices[-3] = 0

        data = data[tuple(slices)]
        # TODO delete
        if _debug:
            print(f'read_movie: after slicing to exclude color channel {data.shape=}')
        #
        assert len(data.shape) == len(slices) - 1, ('channel dimension should no longer'
            'be in shape'
        )

    else:
        # Just for now, since write_tiff currently doesn't support the extra dimension
        # the 'c' channel would add.
        assert (
            all(x > 1 for x in data.shape[:-3]) and
            all(x > 1 for x in data.shape[(-3 + 1):])
        ), f'data.shape: {data.shape}'

        assert data.shape[-3] == 1, f'c: {c}, data.shape: {data.shape}'
        data = np.squeeze(data)

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
hdf5_default_exclude_datasets = (
    'piezo_monitor',
    'pockels1_monitor',
    'frame_in',
    'light_path_shutter',
    'flipper_mirror',
    'pid',
    'frame_counter',
)
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
# TODO account for return_dataset_names_only=True path in return typehint
def load_thorsync_hdf5(thorsync_dir, datasets=None, exclude_datasets=None,
    drop_gctr=True, return_dataset_names_only=False, skip_dict_rename=False,
    skip_normalization=False, rename_dict=None, use_tqdm=False, verbose=False,
    _debug=False) -> pd.DataFrame:
    """Loads ThorSync .h5 output within `thorsync_dir` into a `pd.DataFrame`

    A column 'time_s' will be added, which is derived from 'GCtr', and
    represents the time (in seconds) from the start of the ThorSync recording.

    Args:
        datasets (iterable of str | None): Load only datasets with these names.
            Do not include the group names preceding the dataset name. Pass only
            one of either this or `exclude_datasets`. Names are checked after any
            renaming via `rename_dict` or normalization.

        exclude_datasets (iterable of str | False | None): Load only datasets *except*
            those with these names. Do not include 'gctr' here. Defaults to
            `hdf5_default_exclude_datasets` if neither this nor `datasets` is passed.
            If `False`, all datasets are loaded.

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

    using_default_excludes = False

    if datasets is None and exclude_datasets is None:
        using_default_excludes = True
        exclude_datasets = hdf5_default_exclude_datasets

    # Structure of hdf5 can be explored via:
    # h5dump -H <h5 path>
    # (need to `sudo apt install hdf5-tools` first)
    hdf5_fname = get_thorsync_h5(thorsync_dir)

    if _debug:
        from os.path import getsize
        print(f'HDF5 ({hdf5_fname}) size: {getsize(hdf5_fname):,} bytes')

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
                if verbose:
                    print(f'skipping {obj.name} because it is under /Freq, and will '
                        'have a different length than the other datasets'
                    )

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

            if datasets and dataset_name not in datasets:
                if verbose:
                    print('skipping because not among names passed in datasets\n')

                return

            if exclude_datasets and dataset_name in exclude_datasets:
                if verbose:
                    if using_default_excludes:
                        print('skipping because in default exclude_datasets')
                    else:
                        print('skipping because in exclude_datasets')
                    print()

                return

            shape = obj.shape
            assert len(shape) == 2 and shape[1] == 1, 'unexpected shape'
            # NOTE: would be an issue if someone named one of the user-nameable
            # analog inputs to be the same as one of the builtin dataset names
            assert dataset_name not in data_dict, 'dataset names not unique'

            # Seems to be about twice as fast as `values = obj[:, 0]`
            # Doesn't seem like there is a faster way to do this. IO limited hopefully.
            # There is h5py <dataset>.read_direct(<empty np array>, ...), but I think it
            # would behave the same.
            values = np.array(obj).squeeze()

            if parent_name == '/DI':
                # Anything non-zero gets converted to True
                values = values.astype(np.bool_)

            data_dict[dataset_name] = values

            if verbose:
                print()


    # TODO warn/err (configurable via kwarg?) if any datasets requested were not
    # found (to help identify naming convention changes in the HDF5 files, etc)

    # NOTE: for some reason, opening a debugger (e.g. via `ipdb.set_trace()`)
    # inside this context manager has `self` in `dir()`, seemingly pointing to
    # `f`, but `f` can not be referenced directly.
    with h5py.File(hdf5_fname, 'r') as f:
        # Populates data_dict
        f.visititems(load_datasets)

    if return_dataset_names_only:
        # TODO account for this in return typehint
        return full_dataset_names

    # TODO maybe compare performance w/ w/o conversion to Dataframe?
    df = pd.DataFrame(data_dict)


    # TODO probably refactor this up top, along with time_col and something similar for
    # 'frame_out'
    gctr_col = 'gctr'
    if gctr_col in df.columns:
        # TODO check whether this is (nearly) equivalent to multiplying arange
        # len samples by 1 / thorsync sampling rate
        # Dividing what I think is the clock cycle counter by the 20MHz
        # mentioned in the 3.0 ThorSync manual (section 5.2 "Reviewing Data").
        df[time_col] = df[gctr_col] / int(2e7)

        if drop_gctr:
            # This is surprisingly slow. ~25% of runtime of this function loading a few
            # columns of a ~1.7Gb file (kernprof). Faster way?
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

    if _debug:
        print(f'ThorSync dataframe memory usage: {df.memory_usage(deep=True).sum():,} '
            'bytes'
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
def threshold_crossings(signal, threshold=None, onsets=True, offsets=True):
    # TODO clarify "ignored" in equality case in doc
    """
    Returns indices where signal goes from < threshold to > threshold as onsets,
    and where signal goes from > threshold to < threshold as offsets.

    Cases where it at one index equals the threshold are ignored. Shouldn't
    happen and may indicate electrical problems for our application.
    """
    if threshold is None:
        threshold = ANALOG_0_TO_5V_THRESHOLD

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


def get_col_onset_indices(df, possible_col_names, threshold=None):
    """Returns arrays onsets, offsets with appropriate indices in `df`.

    Args:
        possible_col_names (str or tuple): can be either exact column name in
            `df` or an iterable of column names, where the first matching a column
            in `df` will be used.

        **kwargs: passed through to `threshold_crossings`.

    """
    col = _get_column(df, possible_col_names)
    onsets, _ = threshold_crossings(col, threshold=threshold,
        offsets=False
    )
    return onsets


# TODO how to type hint arbitrary length tuple (/iterable) of str (in a union)?
def get_col_onset_offset_indices(df: pd.DataFrame, possible_col_names,
    checks: bool = True, threshold: Optional[float] = None):
    """Returns arrays onsets, offsets with appropriate indices in `df`.

    Args:
        possible_col_names (str or tuple): can be either exact column name in
            `df` or an iterable of column names, where the first matching a column
            in `df` will be used.

        threshold (float): passed to `threshold_crossings` under the same name.

    Raises OnsetOffsetNumMismatch if `checks=True` and the number of onsets and offsets
    differ.
    """
    col = _get_column(df, possible_col_names)

    # TODO TODO refactor to delete this hack (fixes change to scope_pin)
    if col.dtype == np.dtype('bool'):
        threshold = DIGITAL_THRESHOLD
    #

    # TODO have this (inside) probably warn if there are no threshold crossings
    # (and maybe compare thresh to max/min/dtype values in generating warning to
    # indicate if that might be the cause of the error, which i guess it
    # must...)
    onsets, offsets = threshold_crossings(col, threshold=threshold)

    if checks:
        # TODO maybe just have threshold_crossings make these checks by default, w/ a
        # kwarg to override (which gets threaded through here)?
        if len(onsets) != len(offsets):
            raise OnsetOffsetNumMismatch(f'{len(onsets)} != {len(offsets)}')

        assert np.all(onsets < offsets), 'at least one offset before onset'

    return onsets, offsets


def get_col_onset_offset_times(df: pd.DataFrame, possible_col_names, **kwargs):
    """Returns arrays onsets, offsets with appropriate values from `df.time_s`.

    Args:
        df (DataFrame): must have a column `'time_s'`, as generated by
            `load_thorsync_hdf5`.

        possible_col_names (str or tuple): can be either exact column name in
            `df` or an iterable of column names, where the first matching a column
            in `df` will be used.

        **kwargs: passed to `get_col_onset_offset_indices`
    """
    onsets, offsets = get_col_onset_offset_indices(df, possible_col_names,
        **kwargs
    )
    onset_times = df.time_s[onsets].values
    offset_times = df.time_s[offsets].values
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
def get_frame_times(df: pd.DataFrame, thorimage_dir, time_ref='mid',
    min_block_duration_s=3.0, acquisition_trigger_names=None, warn=True, _debug=False,
    _wont_use_df_after=False):
    """Returns seconds from start of ThorSync recording for each frame.

    Arguments:
        df: as returned by `load_thorsync_hdf5`

        thorimage_dir: path to ThorImage directory to load metadata from

        time_ref ('mid' | 'end')

        min_block_duration_s (float): (default=1.0) minimum time (in seconds) between
            onset and offset of acquisition trigger. Shorter blocks that precede all
            acceptable-length blocks will simply be disregarding, with a warning.
            Shorter blocks following any acceptable-length blocks will currently trigger
            an error.

    Returns a `np.array` that should be of length equal to the number of frames
    actually saved by ThorImage (i.e. `<output>.shape` should be equal to
    `(movie.shape[0],)`).
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

    # TODO TODO if frame_in can be recovered / configured to be saved in the
    # future, and it does indeed serve the same function as our
    # "scope_pin"/whatever, replace this with that (at least if it's available
    # in current data)
    if acquisition_trigger_names is None:
        acquisition_trigger_names = _acquisition_trigger_names

    # ~28% of time (kernprof on one test input)
    acq_onsets, acq_offsets = get_col_onset_offset_indices(df,
        acquisition_trigger_names, threshold=ANALOG_0_TO_5V_THRESHOLD
    )

    if len(acq_onsets) == 0:
        raise ValueError('no recording periods found in ThorSync data')

    # weirdly, in my one test case so far, this 1st line took ~9% of time and 2nd ~0%
    acq_onset_times = df.time_s[acq_onsets].values
    acq_offset_times = df.time_s[acq_offsets].values

    first_real_block_onset_s = None
    for block_idx, (on, off) in enumerate(zip(acq_onset_times, acq_offset_times)):

        if off - on >= min_block_duration_s:
            if first_real_block_onset_s is None:
                first_real_block_onset_s = on
                first_real_block_idx = block_idx
        else:
            if first_real_block_onset_s is not None:
                raise ValueError('block shorter than min_block_duration_s '
                    f'({min_block_duration_s}:.1f) after first acceptable-length block'
                )

    if first_real_block_onset_s is None:
        raise ValueError('no blocks longer than min_block_duration_s '
            f'({min_block_duration_s}:.1f)'
        )

    if warn and first_real_block_idx != 0:
        warnings.warn(f'dropping data up to block with index {first_real_block_idx}, '
            'because earlier "blocks" were shorter than min_block_duration_s '
            f'({min_block_duration_s:.1f})'
        )

    # Considerably faster than:
    # `df = df[df.time_s >= first_real_block_onset_s]`
    df = df.iloc[df.time_s.searchsorted(first_real_block_onset_s):]

    # I did some tests where I defined old_frame_out_[on/off]sets as below, just
    # computed before subsetting the DataFrame on the line above, and these held true:
    # np.array_equal(old_frame_out_onsets[3:] - new_first_thorsync_idx,
    #     frame_out_onsets
    # )
    # np.array_equal(old_frame_out_offsets[3:] - new_first_thorsync_idx,
    #     frame_out_offsets
    # )
    # (the 3 is just because that's how many spurious frame out pulses were filtered out
    # for the data I was testing this on)
    new_first_thorsync_idx = df.index[0]
    acq_onsets = acq_onsets[first_real_block_idx:] - new_first_thorsync_idx
    acq_offsets = acq_offsets[first_real_block_idx:] - new_first_thorsync_idx

    # ~37% of time (kernprof on one test input)
    # If we could guarantee we wouldn't use the dataframe after, we could do this
    # inplace and save all of that time, but it's not worth the possible bugs.
    if not _wont_use_df_after:
        df = df.reset_index(drop=True)
    else:
        df.reset_index(drop=True, inplace=True)

    if _debug:
        print('subsetting dataframe to rows where time_s >= '
            f'{first_real_block_onset_s:.3f}'
        )

    # TODO refactor 'frame_out' and other hardcoded col name strings to top w/ variable
    # (user renameable) ones
    # ~24% of time (kernprof on one test input)
    frame_out_onsets, frame_out_offsets = get_col_onset_offset_indices(df,
        'frame_out', threshold=DIGITAL_THRESHOLD
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

    _, _, z, c, n_flyback, _, xml = load_thorimage_metadata(thorimage_dir,
        return_xml=True
    )

    # Number of XY frames, even in the volumetric case. This is however, the
    # number of frames AFTER any frame averaging.
    n_frames = get_thorimage_n_frames(xml)

    z_total = z + n_flyback

    # TODO delete after figuring out what should really be used
    #compare_to_acq_off = 'offset'
    compare_to_acq_off = 'onset'
    assert compare_to_acq_off in ('onset', 'offset')

    n_orig_frame_out_pulses = len(frame_out_onsets)
    #

    n_averaged_frames = get_thorimage_n_averaged_frames_xml(xml)

    # TODO fix wrt yangs data
    '''
    if n_averaged_frames > 1:
        assert z_total == 1, ('ThorImage does not support averaging while '
            'recording fast Z'
        )
    '''

    # "Frame save groups" are either frames to be averaged to a single frame or
    # frames that together make up a volume (including any flyback frames!)
    n_frames_per_save_group = n_averaged_frames * z_total * c

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

            # NOTE: hack to get number of frames to work out correctly
            # Seemed to solve all issues on data from 2021-03-07 and 2021-03-08
            # TODO TODO TODO test on more data! any cases this fails? and if so, how to
            # handle differently?
            if n_trailing_unsaved_frames in (0, 1):
                n_trailing_unsaved_frames += n_frames_per_save_group

                if _debug:
                    print('increasing n_trailing_unsaved_frames by '
                        'n_frames_per_save_group!!!'
                    )

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
            # TODO also print which index is not saved
            print('curr_not_saved_bool.sum():', curr_not_saved_bool.sum())
            print()

        # TODO assert that curr_not_saved_bool has no overlap with the stuff marked to
        # not be saved in any other iterations (if not already doing something like
        # this...)

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
        print('n_frames_before_averaging (from ThorImage XML):',
            n_frames_before_averaging
        )
        print('n_frames_after_dropping:', len(frame_times))
        e1 = n_frames_before_averaging - len(frame_times)

        n_frame_outs_to_drop = \
            n_orig_frame_out_pulses - n_frames_before_averaging

        print('n_frame_outs_to_drop (how many *should* be dropped):',
            n_frame_outs_to_drop
        )
        print('n_actually_dropped (how many we are actually planning to drop):',
            len(not_saved_indices)
        )
        e2 = len(not_saved_indices) - n_frame_outs_to_drop
        assert e1 == e2
        print('EXCESS FRAMES DROPPED:', e1)
        print('\n')

    if n_averaged_frames > 1:
        frame_times = frame_times.reshape(-1, n_averaged_frames).mean(axis=-1)

    assert len(frame_times) == n_frames, (f'{len(frame_times)} (how many the code '
        f'thinks there were, from the ThorSync HDF5 data) != {n_frames} (from '
        'ThorImage XML)'
    )

    if z > 1:
        flyback_indices = get_flyback_indices(n_frames, z, n_flyback,
            frame_times
        )

        # https://stackoverflow.com/questions/47540800
        # This will raise `IndexError` if any exceed size of frame_times
        # (though it shouldn't unless I made a mistake)
        frame_times = np.delete(frame_times, flyback_indices)

        # TODO change these two AssertionErrors into a custom error

        # (we basically already know, but just for the sake of it...)
        n_volumes, remainder = divmod(len(frame_times), z)
        assert remainder == 0

        frame_times = frame_times.reshape(-1, z).mean(axis=-1)
        assert len(frame_times) == n_volumes

    return frame_times


def assign_frame_times_to_blocks(frame_times, rtol=1.5):
    """Takes array of frame times to (start, stop) indices for each block.

    Args:
        frame_times (np.array): as output by `get_frame_times`.
            should have a `shape` of `(movie.shape[0],)`.

        rtol (float): (optional, default=1.5) time differences between frames must
            be at least this multiplied by the median time difference in order for
            a block to be called there.

    Notes:
    This function defines blocks (periods of continuous acquisition) by regions
    of `frame_times` where the time difference between frames remains
    essentially constant. Large jumps in the time between two frames defines the
    start of a new block. Indices returned would be suitable to index the first
    dimension of the `movie`, the output of `get_frame_times`, etc. `stop`
    indices are included as part of the block, so you should add one when using
    them as the end of a slice.
    """
    dts = np.diff(frame_times, prepend=frame_times[0])
    median_dt = np.median(dts)

    # This should have <# of block> - 1 elements
    discontinuities = np.where(dts > (rtol * median_dt))[0]

    # This should contain the indices of the frames immediately AFTER each
    # discontinuity (as well as a 0 that I add to the front).
    start_frames = np.insert(discontinuities, 0, 0)

    end_frames = np.append(discontinuities - 1, len(frame_times) - 1)

    # TODO TODO (optional extra args+) tests involving checks of # of blocks
    # determined here against # of blocks measured via
    # get_col_onset_offset_indices or something like that. at least this, but
    # could maybe check additional things too.

    return list(zip(start_frames, end_frames))


def assign_frames_to_blocks(df, thorimage_dir, **kwargs):
    """Takes ThorSync+Image data to (start, stop) indices for each block.

    Args:
    df: as output by `load_thorsync_hdf5`

    thorimage_dir: path to a directory created by ThorImage

    **kwargs: passed through to `get_frame_times`

    See documentation of `assign_frame_times_to_blocks` for more details on the
    definition of blocks and the properties of the output.
    """
    frame_times = get_frame_times(df, thorimage_dir, **kwargs)
    return assign_frame_times_to_blocks(frame_times)


# TODO maybe move this function to util or something?
# TODO provide kwargs to crop (end of?) ranges so that all have same number of
# frames? might also be some cases where something similar is needed at start,
# especially when we have multiple blocks
# TODO maybe just take frame_times + odor_col instead? or rename in a way that
# makes what input should be more clear?
def assign_frames_to_odor_presentations(thorsync_input, thorimage_dir,
    odor_onset_to_frame_rel=None, odor_onset_to_frame_const=None,
    odor_timing_names=None, check_all_frames_assigned=True,
    check_no_discontinuity=True, **kwargs):
    """Returns list of (start, first_odor_frame, end) frame indices

    One 3-tuple per odor presentation.

    Frames are indexed as they are along the first dimension of the movie,
    including for volumetric data (where a scalar index of this dimension will
    produce a volume) and/or data collected via frame averaging.

    End frames are included in range, and thus getting a presentation must be
    done like `movie[start_i:(end_i + 1)]` rather  than `movie[start_i:end_i]`.

    Not all frames necessarily included. No overlap.

    Args:
        thorsync_input (str | pd.DataFrame): path to directory created by ThorSync
            or a dataframe as would be created by passing such a directory to
            `load_thorsync_hdf5`.

        thorimage_dir (str): path to directory created by ThorImage that corresponds
            to the same experiment as the ThorSync data in `thorsync_input`.

        odor_onset_to_frame_rel (float): (NOT IMPLEMENTED) factor of averaged
            volumes/frames per second used to determine how long after odor onset to
            call first odor frame.

            No first odor frames will be before: (odor onset time +
            odor_onset_to_frame_rel * averaged volumes/frames per second +
            odor_onset_to_frame_const)

        odor_onset_to_frame_const (float): (NOT IMPLEMENTED) seconds after odor onset to
            call first odor frame. mainly to compensate for known lag between valve
            opening and odor arriving at the animal.

        **kwargs: passed through to `get_frame_times`.

    """
    if odor_onset_to_frame_rel is not None:
        raise NotImplementedError

    if odor_onset_to_frame_const is not None:
        raise NotImplementedError

    if odor_timing_names is None:
        odor_timing_names = _odor_timing_names

    # This also actually works w/ stuff of type np.str_, which old comparison did not.
    if not isinstance(thorsync_input, (str, Path)):
        # Just assuming it's an appropriate DataFrame input in this case.
        df = thorsync_input
        df_was_passed_in = True
    else:
        # TODO maybe unify definition of these cols w/ default dataset names loaded in
        # load_thorsync_hdf5 (fn that takes kwargs to override certain manually-named
        # names?)
        dataset_names = ['gctr', 'frame_out'] + list(odor_timing_names)

        # Since this might be passed as a kwarg to this function, in which case it
        # would just get passed through to `get_frame_times`.
        acq_trigger_names = kwargs.get('acquisition_trigger_names',
            _acquisition_trigger_names
        )
        dataset_names += acq_trigger_names

        # ~78% of time (if called)
        df = load_thorsync_hdf5(thorsync_input, datasets=dataset_names)

        df_was_passed_in = False

    del thorsync_input


    # (when the valve(s) are given the signal to open)
    odor_onsets = get_col_onset_indices(df, odor_timing_names,
        threshold=ANALOG_0_TO_5V_THRESHOLD
    )
    odor_onset_times = df[time_col].values[odor_onsets]

    # unsafe if caller has a reference to `df`, but can save a small amount of time if
    # they don't (i.e. if we loaded it in this function)
    _wont_use_df_after = not df_was_passed_in

    # assuming load_thorsync_hdf5 is called in this fn, this is ~19% of the time,
    # and ~2/3rd that if _wont_use_df_after=True
    frame_times = get_frame_times(df, thorimage_dir,
        _wont_use_df_after=_wont_use_df_after, **kwargs
    )
    del df

    block_ranges = assign_frame_times_to_blocks(frame_times)

    # TODO assert all odor_onset_times are in one of the block ranges somewhere

    start_frames = []
    first_odor_frames = []
    end_frames = []
    curr_index_offset = 0

    for start, end in block_ranges:
        start_s = frame_times[start]
        end_s = frame_times[end]

        curr_frame_times = frame_times[start:(end + 1)]

        curr_odor_onset_times = odor_onset_times[np.logical_and(
            start_s <= odor_onset_times, odor_onset_times <= end_s
        )]

        # Assuming we can use the amount of time continuously recording before the first
        # odor onset as the duration before each odor presentation that should be
        # assigned to each following odor presentation, within this block.
        first_frame_to_odor_s = curr_odor_onset_times[0] - start_s
        curr_start_times = curr_odor_onset_times - first_frame_to_odor_s

        # Using this function because `curr_start_times` are generally not going to
        # actually exactly equal to any times in `curr_frame_times`. If tied, seems to
        # provide index that would insert the new element at the earlier position.
        curr_start_frames = np.searchsorted(curr_frame_times, curr_start_times)

        # TODO TODO TODO maybe (option to?) calculate/lookup (averaged) frames/volumes
        # per second and require the first odor frame be either a full / half duration
        # of that past odor onset???
        # TODO TODO TODO need to correct output of np.searchsorted to ensure it is AFTER
        # the odor onset, or not?
        curr_first_odor_frames = np.searchsorted(curr_frame_times,
            curr_odor_onset_times
        )
        assert len(curr_start_frames) == len(curr_first_odor_frames)

        # TODO maybe refactor to share w/ one other place that does this now?
        # Inclusive (so can NOT be used as the end of slices directly. need to
        # add one because slice ends are not inclusive.)
        curr_end_frames = np.append(
            curr_start_frames[1:] - 1, len(curr_frame_times) - 1
        )

        for s, o, e in zip(curr_start_frames, curr_first_odor_frames, curr_end_frames):
            assert s < o < e, f'({s}, {o}, {e})'

        # NOTE: i'm not sure there is a guarantee this will currently always be true,
        # but it has been for my data so far. may also be ok for it not to be true, but
        # may need to re-evaluate.
        # If this is True, and the frame times is assigned to the middle of all
        # corresponding "Frame Out" pulses, then at least >half of the time should have
        # been after the odor onset (or at least when the valve was triggered).
        assert np.all(curr_frame_times[curr_first_odor_frames] > curr_odor_onset_times)

        '''
        # TODO delete
        print('odor_onset_times:', odor_onset_times)
        print('first odor frame times:', )
        #
        # TODO maybe put this behind some _debug / verbose flag or something
        print('onset to first odor frame (s):',
            curr_frame_times[curr_first_odor_frames] - odor_onset_times
        )
        import ipdb; ipdb.set_trace()
        '''

        start_frames.append(curr_start_frames + curr_index_offset)
        first_odor_frames.append(curr_first_odor_frames + curr_index_offset)
        end_frames.append(curr_end_frames + curr_index_offset)

        curr_index_offset += len(curr_frame_times)

    start_frames = np.concatenate(start_frames)
    first_odor_frames = np.concatenate(first_odor_frames)
    end_frames = np.concatenate(end_frames)

    if check_all_frames_assigned:
        # TODO better name
        indices = []
        for s, e in zip(start_frames, end_frames):
            indices.append(np.arange(s, e + 1))

        indices = np.concatenate(indices)

        # TODO TODO TODO which of the 2022-10-07 data was triggering this again?
        # 1/megamat0_part2? what is correct handling there?
        # TODO TODO TODO replace assertion w/ raising NotAllFramesAssigned
        # (+ adapt any calling code that currently catches AssertionError)
        assert len(frame_times[indices]) == len(frame_times), (f'{thorimage_dir}: not '
            f'all frames were assigned ({len(frame_times[indices])=} != '
            f'{len(frame_times)=})'
        )

        # TODO delete try/except
        '''
        try:
            assert len(frame_times[indices]) == len(frame_times), \
                f'{thorimage_dir}: not all frames were assigned'

        except AssertionError:
            print(f'{thorimage_dir=}')
            print(f'{len(frame_times[indices])=}')
            print(f'{len(frame_times)=}')
            import ipdb; ipdb.set_trace()
        '''
        #

        assert len(np.unique(indices)) == len(indices), \
            'nonunique (probably overlapping ranges of) indices'

    if check_no_discontinuity:
        # Mainly intending to check for large discontinuitie that might arise
        # if, for example, the last odor presentation in one acquisition period
        # was also assigned the first frame in the next acquisition period.

        for s, e in zip(start_frames, end_frames):
            curr_odor_times = frame_times[s:(e+1)]
            dts = np.diff(curr_odor_times)
            median_dt = np.median(dts)
            max_dt = np.max(dts)

            assert max_dt < 1.5 * median_dt, \
                'discontinuity larger than usual frame delta t'

    return list(zip(start_frames, first_odor_frames, end_frames))


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
def thorsync_num(thorsync_dir: Pathlike) -> int:
    """Returns number in suffix of ThorSync output directory name as an int.
    """
    return int(str(thorsync_dir)[len(thorsync_dir_prefix):])


def is_thorsync_dir(d: Pathlike, verbose=False) -> bool:
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


def is_thorimage_raw(f: Pathlike) -> bool:
    """True if filename indicates file is ThorImage raw output.
    """
    _, f_basename = split(f)

    # Needs to match at least 'Image_0001_0001.raw' and 'Image_001_001.raw'
    if f_basename.startswith('Image_00') and f_basename.endswith('001.raw'):
        return True

    return False


def is_thorimage_dir(d: Pathlike, verbose=False) -> bool:
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

        # TODO replace w/ regex that also matches number parts in middle
        # (though exact number of digits may vary between 3 and 4, and not sure number
        # of parts is always 4)
        elif f.startswith('ChanA_') and f.endswith('.tif'):
            have_tiff = True

        if have_xml and (have_raw or have_tiff):
            break

    if verbose:
        print('have_xml:', have_xml)
        print('have_raw:', have_raw)
        print('have_tiff:', have_tiff)

    if have_xml and (have_raw or have_tiff):
        return True
    else:
        return False


# TODO some way to type hint the fact that if filter_funcs is a fn (not an iterable of
# them), the output will also be ~"squeezed"? otherwise it might be useful to change
# type to always be consistent idk
def _filtered_subdirs(parent_dir: Pathlike, filter_funcs, exclusive=True,
    verbose=False):
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
                filtered_subdirs.append(Path(d))
                if exclusive:
                    break

        if verbose:
            print('')

    if len(filter_funcs) == 1:
        all_filtered_subdirs = all_filtered_subdirs[0]

    return all_filtered_subdirs


def thorimage_subdirs(parent_dir: Pathlike) -> List[Path]:
    """
    Returns a list of any immediate child directories of `parent_dir` that have
    all expected ThorImage outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorimage_dir)


def thorsync_subdirs(parent_dir: Pathlike) -> List[Path]:
    """Returns a list of any immediate child directories of `parent_dir`
    that have all expected ThorSync outputs.
    """
    return _filtered_subdirs(parent_dir, is_thorsync_dir)


def thor_subdirs(parent_dir: Pathlike, absolute_paths=True
    ) -> Tuple[List[Path], List[Path]]:
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

    return thorimage_dirs, thorsync_dirs


# TODO TODO generalize / wrap in a way that also allows associating with
# stimulus files / arbitrary other files.
def pair_thor_dirs(thorimage_dirs, thorsync_dirs, use_mtime=False,
    use_ranking=True, check_against_naming_conv=False,
    check_unique_thorimage_nums=None, verbose=False, ignore_prepairing=None,
    ignore=None) -> List[PathPair]:
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

        ignore_prepairing (None | iterable of str): An optional iterable of substrings.
            If any are present in the name of a Thor directory, that directory will be
            excluded from consideration in pairing. This is mainly to keep the (fragile)
            implementation that requires equal numbers of ThorImage and ThorSync
            directories for pairing working if some particular experiments named a
            certain way only have data from one. Will also be try appending these to
            `ignore` if uneven numbers of directories and `use_ranking=True`.

        ignore (None | iterable of str): As `ignore_prepairing`, but ignore will happen
            after pairing. Both the ThorImage and ThorSync directories of a pair will be
            checked for these substrings and if any match the pair is not returned. This
            is mainly intended to ignore known-bad data.

    Raises ValueError if two dirs of one type match to the same one of the
    other, but just returns shorter list of pairs if some matches can not be
    made. These errors currently just cause skipping of pairing for the
    particular (date, fly) pair above (though maybe this should change?).

    Raises AssertionError when assumptions are violated in a way that should
    trigger re-evaluating the code.
    """
    if ignore_prepairing is not None:
        orig_thorimage_dirs = list(thorimage_dirs)
        orig_thorsync_dirs = list(thorsync_dirs)

        thorimage_dirs = [x for x in thorimage_dirs
            if not any([sub_str in str(x) for sub_str in ignore_prepairing])
        ]
        thorsync_dirs = [x for x in thorsync_dirs
            if not any([sub_str in str(x) for sub_str in ignore_prepairing])
        ]

    if use_ranking:
        if len(thorimage_dirs) != len(thorsync_dirs):

            if ignore_prepairing is not None:
                if ignore is None:
                    ignore = []
                else:
                    ignore = list(ignore)

                ignore.extend(ignore_prepairing)

                return pair_thor_dirs(orig_thorimage_dirs, orig_thorsync_dirs,
                    use_mtime=use_mtime, use_ranking=use_ranking,
                    check_against_naming_conv=check_against_naming_conv,
                    check_unique_thorimage_nums=check_unique_thorimage_nums,
                    verbose=verbose, ignore_prepairing=None, ignore=ignore
                )

            raise ValueError('can only pair with ranking when equal # dirs.\n\n'
                f'thorimage_dirs ({len(thorimage_dirs)}):\n{pformat(thorimage_dirs)}\n'
                f'\nthorsync_dirs ({len(thorsync_dirs)}):\n{pformat(thorsync_dirs)}'
            )

    if check_unique_thorimage_nums and not check_against_naming_conv:
        raise ValueError('check_unique_thorimage_nums=True requires '
            'check_against_naming_conv=True'
        )

    # So that we don't need to explicitly disable both of these flags if we want
    # to disable these checks. Just need to set check_against_naming_conv=False
    if check_unique_thorimage_nums is None and check_against_naming_conv:
        check_unique_thorimage_nums = True

    thorimage_times = {d: get_thorimage_time(d) for d in thorimage_dirs}

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

    if ignore is not None:
        pairs = [(ti, ts) for ti, ts in pairs if not
            any([(sub_str in str(ti) or sub_str in str(ts)) for sub_str in ignore])
        ]

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
def pair_thor_subdirs(parent_dir, verbose=False, **kwargs) -> List[PathPair]:
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
    pixels_per_um = 1 / get_thorimage_pixelsize_um(xml_root)
    dxy = (pixels_per_um, pixels_per_um)
    # TODO maybe load dims anyway?
    return {'fr': fps, 'dxy': dxy}

