"""
Functions for working with suite2p as part of analysis.
"""

from hong2p import thor


def print_suite2p_params(thorimage_dir):
    # From: https://suite2p.readthedocs.io/en/latest/settings.html

    single_plane_fps, xy, z, c, n_flyback, _ = thor.load_thorimage_metadata(
        thorimage_dir
    )

    # TODO TODO TODO is this per plane or per volume in the volumetric case? might need
    # to read code...
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

    # TODO TODO TODO is this per plane or per volume in the volumetric case? might need
    # to read code...
    # "(float, default: 10.0) Sampling rate (per plane). For instance, if you have a 10
    # plane recording acquired at 30Hz, then the sampling rate per plane is 3Hz,
    # so set ops[‘fs’] = 3 "
    fs = single_plane_fps / (z + n_flyback)

    # TODO TODO can we pick a better spatial_scale than suite2p does automatically?
    # does its optimal value even differ across the types of data we have in the lab?

    print('nplanes:', nplanes, '(assuming nplanes == z)')
    print('tau: 0.7 (recommended for GCaMP6f)')
    print('fs:', fs)

