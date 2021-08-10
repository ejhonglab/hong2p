"""
Functions for working with suite2p as part of analysis.
"""

from hong2p import thor


def suite2p_params(thorimage_dir):
    # From: https://suite2p.readthedocs.io/en/latest/settings.html

    single_plane_fps, xy, z, c, n_flyback, _ = thor.load_thorimage_metadata(
        thorimage_dir
    )

    # "(int, default: 1) each tiff has this many planes in sequence"
    # wait, is this =z, or =z*t or just =t?
    nplanes = z

    # "(int, default: 1) each tiff has this many channels per plane"
    #nchannels = 

    # "(int, default: 1) this channel is used to extract functional ROIs (1-based, so 1
    # means first channel, and 2 means second channel)"
    #functional_chan = 

    # "(float, default: 1.0) The timescale of the sensor (in seconds), used for
    # deconvolution kernel. The kernel is fixed to have this decay and is not fit to the
    # data. We recommend: 0.7 for GCaMP6f, 1.0 for GCaMP6m, 1.25-1.5 for GCaMP6s"
    #tau = 

    # "(float, default: 10.0) Sampling rate (per plane). For instance, if you have a 10
    # plane recording acquired at 30Hz, then the sampling rate per plane is 3Hz,
    # so set ops[‘fs’] = 3 "
    fs = single_plane_fps / (z + n_flyback)

    # TODO TODO how much of a problem is it to ignore the fact that there are flyback
    # frames contributing to time before a plane is returned to? not sure i'm
    # understanding 'fs' correctly to begin with...

    # TODO TODO maybe use ops['ignore_flyback'] (== indices of each flyback frame?
    # or is there a more concise option?) to make fs more meaningful?

    # TODO TODO can we pick a better spatial_scale than suite2p does automatically?
    # does its optimal value even differ across the types of data we have in the lab?

    ops = {
        'fs': fs,
        'nplanes': nplanes,
    }
    return ops


def print_suite2p_params(thorimage_dir):
    ops = suite2p_params(thorimage_dir)
    print('nplanes:', ops['nplanes'])
    print(f'fs: {ops["fs"]:.2f}')
    print('tau: 0.7 (recommended for GCaMP6f)')

