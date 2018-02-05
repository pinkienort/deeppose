#!/bin/bash
# Copyright (c) 2016 Shunta Saito
set -x

DATASET_ROOT='/lhome/hisakazu-fu/datasets/lsp_ext'
PYTHONPATH="`pwd`" \
HAINER_TYPE_CHECK=0 \
#time python -m pdb scripts/train_single.py \
time python scripts/train_single.py \
--model models/AlexNet.py \
--gpus 0 \
--epoch 100 \
--batchsize 2 \
--snapshot 10 \
--valid_freq 5 \
--train_csv_fn $DATASET_ROOT/example_train_joints.csv \
--test_csv_fn $DATASET_ROOT/example_test_joints.csv \
--img_dir '' \
--test_freq 10 \
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
