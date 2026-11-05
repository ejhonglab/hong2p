
from datetime import datetime, timedelta
from xml.etree.ElementTree import Element, SubElement

import pytest

from hong2p import thor


def _date_xml(date: datetime, utime: float) -> Element:
    xml = Element('ThorImageExperiment')
    SubElement(xml, 'Date', date=date.strftime('%m/%d/%Y %H:%M:%S'), uTime=str(utime))
    return xml


def test_get_thorimage_time():
    utime = 1583776800.5
    local_time = datetime.fromtimestamp(utime)

    # 'date' is local time of acquisition computer, which may be in a different
    # timezone than the computer running the tests
    for tz_offset in (timedelta(0), timedelta(hours=3), timedelta(hours=-5, minutes=-30)):
        xml = _date_xml(local_time + tz_offset, utime)
        assert thor.get_thorimage_time(xml) == local_time

    # not a timezone offset (including in negative direction, which used to pass)
    for bad_offset in (timedelta(minutes=10), timedelta(seconds=-5), timedelta(hours=30)):
        with pytest.raises(AssertionError):
            thor.get_thorimage_time(_date_xml(local_time + bad_offset, utime))


def _layout_xml(*, fast_z: bool = True, average_num: int = 1, enable_b: bool = False,
    unnamed_zstage_first: bool = False) -> Element:

    xml = Element('ThorImageExperiment')
    SubElement(xml, 'Name', name='test')
    SubElement(xml, 'LSM', pixelX='8', pixelY='6', frameRate='30',
        averageMode='0' if average_num == 1 else '1', averageNum=str(average_num)
    )
    if unnamed_zstage_first:
        SubElement(xml, 'ZStage', steps='1', stepSizeUM='0')

    SubElement(xml, 'ZStage', name='ThorZPiezo', steps='3', stepSizeUM='10')
    SubElement(xml, 'Streaming', enable='1', zFastEnable='1' if fast_z else '0',
        flybackFrames='1'
    )
    # B enabled w/ zero gain still has data in the .raw file, so it should be counted
    SubElement(xml, 'PMT', enableA='1', gainA='50', enableB='1' if enable_b else '0',
        gainB='0'
    )
    return xml


def test_get_thorimage_dims():
    assert thor.get_thorimage_dims(_layout_xml()) == ((8, 6), 3, 1)
    assert thor.get_thorimage_dims(_layout_xml(enable_b=True)) == ((8, 6), 3, 2)

    # the named ZStage, wherever it is
    assert thor.get_thorimage_z_xml(_layout_xml(unnamed_zstage_first=True)) == 3
    assert thor.get_thorimage_zstep_um(_layout_xml(unnamed_zstage_first=True)) == 10


def test_get_thorimage_n_flyback_xml():
    assert thor.get_thorimage_n_flyback_xml(_layout_xml()) == 1
    assert thor.get_thorimage_n_flyback_xml(_layout_xml(fast_z=False)) == 0


def test_get_thorimage_fps_xml():
    assert thor.get_thorimage_fps_xml(_layout_xml()) == 30
    assert thor.get_thorimage_fps_xml(_layout_xml(fast_z=False, average_num=2)) == 15

    # ThorImage does not actually average frames with fast Z, even if configured to
    with pytest.warns(UserWarning, match='frame averaging'):
        assert thor.get_thorimage_fps_xml(_layout_xml(average_num=2)) == 30

    assert thor.get_thorimage_fps_xml(_layout_xml(average_num=2),
        before_averaging=True
    ) == 30
