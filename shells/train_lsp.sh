#!/bin/bash
# Copyright (c) 2016 Shunta Saito

DATASET_ROOT="deeppose_tf/datasets/lsp_ext"
PYTHONPATH="`pwd`" \
HAINER_TYPE_CHECK=0 \
time python scripts/train_single.py \
--model models/AlexNet.py \
--gpu 2 \
--epoch 1500 \
--batchsize 128 \
--snapshot 20 \
--valid_freq 200 \
--train_csv_fn $DATASET_ROOT/train_joints.csv \
--test_csv_fn $DATASET_ROOT/test_joints.csv \
--img_dir '' \
--test_freq 200 \
--seed 1701 \
--im_size 227 \
--fliplr \
--rotate_range 10 \
--zoom \
--zoom_range 0.2 \
--translate \
--translate_range 5 \
--shift 0.1 \
--min_dim 5 \
--bbox_extension_min 1.2 \
--bbox_extension_max 2.0 \
--coord_normalize \
--gcn \
--n_joints 14 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]" \
--opt Adam
