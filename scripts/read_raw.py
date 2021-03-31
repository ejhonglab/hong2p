#!/usr/bin/env python3

import tempfile
from pprint import pprint as pp

import numpy as np
import tifffile
import matplotlib.pyplot as plt

from hong2p.util import write_tiff
from hong2p.thor import read_movie
from hong2p.matlab import matlab_engine


# TODO factor this whole test script into a unit test of reading / saving fns

def main():
    show_frames = True

    # Since despite python rt checking out, normcorre output still seemed weird
    # w/ TIFFs generated this way.
    evil = matlab_engine()

    # so that i know where to look for the code if that would help troubleshoot
    evil.evalc("clear; tp = '{}'; p1 = which('imread_big(tp)')".format(
        'test.tif'))
    p1 = evil.eval('p1')
    p2 = evil.eval("which('imread_big')")
    # "limit to fns named x on search path"
    p3 = evil.eval("which('/imread_big')")
    assert p1 == p2
    assert p2 == p3

    #raw = '/mnt/nas/mb_team/raw_data/2019-01-23/6/_002/Image_0001_0001.raw'
    raw_dir = '/mnt/nas/mb_team/raw_data/2019-01-23/6/_002'
    from_raw = read_movie(raw_dir)

    # This TIFF should have been created with ImageJ manually (this manual
    # process produces TIFFs that yield reasonable looking output from
    # normcorre. tiffs produced as-of-now, w/ write_tiff, do not seem to.)
    tiff = '/mnt/nas/mb_team/raw_data/2019-01-23/6/tif_stacks/_002.tif'
    with tifffile.TiffFile(tiff) as tif:
        from_tiff = tif.asarray()
        ij_md = tif.imagej_metadata
        #pp(dir(tif))
        print(tif.is_imagej)
        print(ij_md)
        print(tif.offsetformat)
        print(tif.offsetsize)
        print(tif.byteorder)

    assert from_tiff.shape == from_raw.shape
    assert np.array_equal(from_tiff, from_raw)
    # dtype are also equal

    '''
    def m2p_array(matlab_array, transpose=True):
        # stackoverflow.com/questions/34155829
        # TODO has need for this just gone away w/ newer version of matlab
        # engine?  (for the order='F' / possible transpose bits?)
        ret = np.array(matlab_array._data).reshape(matlab_array.size, order='C')
        if transpose:
            return ret.T
        else:
            return ret
    '''

    # not doing it this way b/c evil.size(m_imrb_ij_tiff) was taking forever
    #m_imrb_ij_tiff = evil.imread_big(tiff)

    frame = 500
    evil.evalc("ij = imread_big('{}');".format(tiff))
    m_imrb_ij_tiff = evil.eval('ij')

    if show_frames:
        plt.imshow(from_tiff[frame, :, :])

    # not the comparison that really matters... should be matlab-to-matlab
    '''
    imrb_ij_tiff = m2p_array(m_imrb_ij_tiff)
    import ipdb; ipdb.set_trace()
    assert from_tiff.shape == imrb_ij_tiff.shape
    assert from_tiff.dtype == imrb_ij_tiff.dtype

    if show_frames:
        plt.figure()
        plt.imshow(imrb_ij_tiff[frame, :, :])
    '''

    # if this looks normal (even if transposed) normcorre output probably won't
    # be weird (if the problem is a C/F or tranpose thing like it think it might
    # be)
    # The fact that frames plotted in matlab, both from test and ij tiff,
    # did not seem transposed / C/F mangled wrt each other, means that (if the
    # problem is in imread_big) the problem must be something else.
    evil.evalc('frame = ij(:, :, {});'.format(frame + 1))

    if show_frames:
        evil.eval('imshow(double(frame) / double(max(max(frame))))')

    with tempfile.NamedTemporaryFile() as temp:
        test_tiff = temp.name

        # TODO need other args to save metadata same way ij does?
        # and do we actually use that metadata anywhere?
        #tifffile.imsave(test_tiff, from_raw)
        write_tiff(test_tiff, from_raw)

        #from_test_tiff = tifffile.imread(test_tiff)
        with tifffile.TiffFile(test_tiff) as tif:
            from_test_tiff = tif.asarray()
            test_ij_md = tif.imagej_metadata
            #pp(dir(tif))
            print(tif.is_imagej)
            print(test_ij_md)
            print(tif.offsetformat)
            print(tif.offsetsize)
            print(tif.byteorder)

        evil.evalc("tt = imread_big('{}');".format(test_tiff))

    assert from_tiff.shape == from_test_tiff.shape
    assert np.array_equal(from_tiff, from_test_tiff)

    #m_imrb_test_tiff = evil.eval('tt')

    evil.evalc('tt_frame = tt(:, :, {}); figure;'.format(frame + 1))

    if show_frames:
        plt.figure()
        plt.imshow(from_test_tiff[frame, :, :])

        evil.eval('imshow(double(tt_frame) / double(max(max(tt_frame))))')

    # This seems sufficient to check the MATLAB dtypes are the same.
    assert type(evil.eval('frame')) == type(evil.eval('tt_frame'))
    assert evil.eval('frame').size == evil.eval('tt_frame').size
    assert evil.eval('size(ij)') == evil.eval('size(tt)')

    '''
    print('max of frame:', evil.eval('max(max(frame))'))
    print('max of tt_frame:', evil.eval('max(max(tt_frame))'))

    print('min of frame:', evil.eval('min(min(frame))'))
    print('min of tt_frame:', evil.eval('min(min(tt_frame))'))
    '''
    assert evil.eval('min(min(frame))') == evil.eval('min(min(tt_frame))')
    assert evil.eval('max(max(frame))') == evil.eval('max(max(tt_frame))')

    assert evil.eval('isequal(ij, tt)')

    if show_frames:
        plt.show()

    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
