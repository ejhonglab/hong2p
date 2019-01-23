#!/usr/bin/env python3

"""
Traverses analysis output and loads traces and odor information into database.
"""

import os
import glob
import pprint

from sqlalchemy import create_engine
import h5py
import numpy as np
import pandas as pd

url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
conn = create_engine(url)

#matlab_output_file = 'test_data/struct_no_sparsearray.mat'
matlab_output_file = 'test_data/_007_cnmf.mat'

'''
import hdf5storage
print('loading .mat file...')
data = hdf5storage.loadmat(matlab_output_file)
print('done loading mat file')
import ipdb; ipdb.set_trace()
'''

with h5py.File(matlab_output_file, 'r') as data:
    print(list(data.keys()))

    pprint.pprint(list(data['sCNM'].items()))
    pprint.pprint(list(data['S'].items()))

    # block_{f/i}ct - i=initial f=final cross time (scope trigger?)
    # TODO get 'ti' for PID trials too?
    ti = data['ti']
    frame_times = np.array(ti['frame_times'])
    print(np.array(data['sCNM']['file']))
    #print(np.array(data['sCNM']['file']).tostring().decode())

    print(int(np.array(ti['num_stim'])[0,0]))
    print(int(np.array(ti['num_trials'])[0,0]))
    pprint.pprint(list(data['ti'].items()))
    print('')
    # stim_... are similar. of length = to num of stimuli. not grouped by
    # block.

    # spt / fpt = seconds and frames per trial
    # so far, w/ frame averaging, get exactly requested num frames per block
    # (scalar)

    # stim_on/off 

    # TODO index (adjust for diff indexing) into pin_odors w/ pin (- 1) to get
    # odor names

    # not sure what value of si is...

    # TODO how does remy enter odor information now?
    # reformat my pickles for her automated use? just convert that part of
    # analysis to python?
    pin2odors = dict()
    # TODO list of sets / tuples (of names?) to table to sql?
    # TODO out-of-band concentration info?
    # TODO default volume / flows?

    #print(ti['pin_odors'])
    #print(type(ti['pin_odors']))
    
    print(ti['struct_pin_odors'])
    print(list(ti['struct_pin_odors'].items()))
    print(np.array(ti['struct_pin_odors']['odor']))
    print(np.array(ti['struct_pin_odors']['odor'][0]))
    '''
    print([dir(x) for x in ti['pin_odors']])
    print(type(ti['pin_odors'][0]))
    # TODO assert this is length 1 if indexing this way
    pin_odor_refs = ti['pin_odors'][0]
    r = pin_odor_refs[0]
    # TODO it seems i need to convert this back to strings...
    print('dereferenced:', [ti[r] for r in pin_odor_refs])
    '''
    # TODO assert x is actually always len 1?
    '''
    channel_A_pins = [int(x[0]) for x in ti['channelA']]
    channel_B_pins = [int(x[0]) for x in ti['channelB']]

    channel_A_odors = [pin2odors[p] for p in channel_A_pins]
    channel_B_odors = [pin2odors[p] for p in channel_B_pins]

    odors = [set(x) for x in zip(channel_A_odors, channel_B_odors)]
    '''

    # TODO how to get odor pid
    # A has footprints
    # dims=dimensions of image (256x256)
    # T is # of timestamps
    #print(data['sCNM'])
    #print('')

    # DFF - traces
    # sDFF - filtered traces (filtered how?)
    # (what she did w/ jon. see source)
    # k - # components (redundant)
    # F0 - background fluorescence (dims?)
    # TODO TODO subset to stuff just starting from onset
    traces = np.array(data['S']['sDFF'])
    #print(traces)
    # TODO get from_onset array of equal length
    # (seconds from odor valve on) (include negative values?)

