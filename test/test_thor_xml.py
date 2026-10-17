
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
