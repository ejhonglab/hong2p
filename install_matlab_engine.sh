#!/usr/bin/env bash

# Open MATLAB and enter 'matlabroot' (without quotes). Run this script with the
# path it outputs.

# Example:
# >> matlabroot
# ans = '/usr/local/MATLAB/R2018b'
#
# tom@atlas:~/src/python_2p_analysis$
# ./install_matlab_engine.sh /usr/local/MATLAB/R2018b

matlabroot="$1"
cd ${matlabroot}/extern/engines/python
# TODO may need root if in sys dir?
sudo python3 setup.py install
