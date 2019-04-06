#!/usr/bin/env bash

# Open MATLAB and enter 'matlabroot' (without quotes). Run this script with the
# path it outputs.

# Example:
# >> matlabroot
# ans = '/usr/local/MATLAB/R2018b'
#
# tom@atlas:~/src/python_2p_analysis$
# ./install_matlab_engine.sh /usr/local/MATLAB/R2018b

LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
sudo apt-add-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install gcc-6 g++-6 -y

# TODO support installing in anaconda, which seems to require chmod -R 777
# on MATLAB install dir (build at least), then calling setup.py w/ anaconda
# python as non-root user
#matlabroot="$1"
#cd ${matlabroot}/extern/engines/python
## TODO may need root if in sys dir?
#sudo python3 setup.py install
