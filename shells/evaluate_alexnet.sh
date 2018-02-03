#!/bin/bash
set -x

python scripts/evaluate_flic.py \
    --model results/AlexNet_2017-12-19_23-32-266yqnij7i/AlexNet.py \
    --param results/AlexNet_2017-12-19_23-32-266yqnij7i/epoch-1.model \
    --batchsize 18 \
    --gpu 1 \
    --datadir data/FLIC-full \
    --mode test \
    --n_imgs 9 \
    --resize -1 \
    --seed 9 \
    --draw_limb True \
    --text_scale 1.0
