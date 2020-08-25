#!/usr/bin/env bash

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6 python3.6-dev python3.6-venv
curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

# After the above (on atlas 2020-06-02), this seemed to work to install
# requirements.txt :
# (deactivate if need be)
# rm -rf venv
# python3.6 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# CC=gcc pip install -r requirements.txt

# Specifying CC=gcc really did seem necessary, though I couldn't figure out how
# to tell what compiler pip was using otherwise, though 'x86_64-linux-gnu-gcc'
# was its name in the output (so diff version? not sure)

# From bash: gcc --version
# gcc (Ubuntu 7.4.0-1ubuntu1~16.04~ppa1) 7.4.0

