#!/bin/bash
# Copyright (c) 2016 Shunta Saito
set -x

CHAINER_TYPE_CHECK=0 \
python scripts/train_single.py \
--model models/VGG_BN.py \
--gpus 1 \
--epoch 200 \
--batchsize 8 \
--snapshot 20 \
--valid_freq 5 \
--train_csv_fn data/FLIC-full/train_joints.csv \
--test_csv_fn data/FLIC-full/test_joints.csv \
--img_dir data/FLIC-full/images \
--test_freq 1 \
--seed 1701 \
--im_size 220 \
--fliplr \
--rotate \
--rotate_range 10 \
--zoom \
--zoom_range 0.2 \
--translate \
--translate_range 5 \
--coord_normalize \
--gcn \
--n_joints 7 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[2, 4], [1, 5], [0, 6]]" \
--opt Adam
