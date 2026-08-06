#!/bin/bash

PREFIX=$HOME/.local/opt/labelme/
rm -rf $PREFIX
mkdir -p $PREFIX
virtualenv $PREFIX

source $PREFIX/bin/activate

# install in editable mode
pip install -e .
