#!/bin/bash
source /software/miniconda/etc/profile.d/conda.sh
conda activate rawdata

export PETALO_CALIB=$PWD
export PYTHONPATH=$PETALO_CALIB:$PYTHONPATH
