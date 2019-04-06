#!/usr/bin/env python3

import pickle
import sys

import numpy as np
#from scipy.sparse import coo_matrix
import pandas as pd
import matplotlib
matplotlib.use('TkAGG')
import matplotlib.pyplot as plt
#b1 = matplotlib.get_backend()
#print(b1)

#import caiman
#from caiman.source_extraction.cnmf import params, cnmf
#import caiman.utils.visualization
from caiman.source_extraction.cnmf.utilities import extract_DF_F, detrend_df_f


def plot_ex(data):
    # This works if the first dimension of the array is the cell axis, and the
    # second is time.
    pC_df = pd.DataFrame(C_df)
    pC_df.sample(n=100, random_state=7).T.plot()


with open('cnmf_state.p', 'rb') as f:
    state = pickle.load(f)

# Yr, A, C, bl, b, f
Yr = state['Yr']
A = state['A']
C = state['C']
bl = state['bl']
b = state['b']
f = state['f']

# From self.cnm.estimates.detrend_df_f() -> self.cnm.estimates.F_dff.T
df_over_f = state['df_over_f']

fig = plt.figure()
plt.plot(np.mean(Yr, axis=0))

# defaults: quantileMin=8, frames_window=200, block_size=400
C_df = extract_DF_F(Yr, A, C, bl)

# TODO TODO why is this failing? (for lack of YrA?)
# TODO is it called w/ self.YrA not None? b/c if so, it would seem to behave as
# if use_residuals intended, indep. of that flag.
#F_df = detrend_df_f(A, b, C, f) #, None)
#sys.exit()

# TODO title them?
plot_ex(C_df)
#plot_ex(F_df)
plot_ex(df_over_f.T)

# TODO to the extent that trace extraction method was causing poor correlations,
# need to find method that produces good correlations
# TODO if something else (some small / generally bad quality ROIs) were causing
# them, need to determine the cause and fix it
# TODO to the extent that the data was causing the problem, also need to
# determine that
# TODO matlab pipeline produce reasonable correlations? is manual editing very
# crucial to getting good correlations?

# TODO TODO TODO where are these really thin peaks (some negative) coming from?
# are those the cause of the problem?

plt.show(block=True)

import ipdb; ipdb.set_trace()
