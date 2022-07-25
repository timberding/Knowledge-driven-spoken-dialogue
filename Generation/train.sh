#!/bin/bash

version="03_epoch50"
dataroot=""
num_gpus=2


# Response generation
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline_ch/configs/generation/params.json \
    --dataroot ${dataroot} \
    --exp_name rg-hml128-kml128-${version}
