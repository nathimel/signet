#!/bin/sh
source ~/miniforge3/etc/profile.d/conda.sh # Local
conda activate signet

# main command
python3 src/main.py configs/dev.yml
