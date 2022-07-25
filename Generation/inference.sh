#!/bin/bash

version="03_epoch50"

export CUDA_VISIBLE_DEVICES='0,1'

mkdir -p pred/val

python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
        --generation_params_file baseline_ch/configs/generation/generation_params.json \
        --eval_dataset test \
        --dataroot data/ \
        --output_file test_evl.json
