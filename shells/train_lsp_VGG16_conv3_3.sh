#!/bin/bash
# Copyright (c) 2016 Shunta Saito
set -x

DATASET_ROOT='/lhome/hisakazu-fu/datasets/lsp_ext'
PYTHONPATH="`pwd`" \
HAINER_TYPE_CHECK=0 \
#time python -m pdb scripts/train_single.py \
time python scripts/train_single.py \
--model models/VGG16_conv3_3.py \
--resume_model results2/VGG16_conv3_3/VGG16_conv3_3.npy \
--gpu 1 \
--epoch 10000 \
--batchsize 64 \
--snapshot 100 \
--valid_freq 1000 \
--train_csv_fn $DATASET_ROOT/train_joints.csv \
--test_csv_fn $DATASET_ROOT/test_joints.csv \
--img_dir '' \
--test_freq 1000 \
--seed 1701 \
--im_size 224 \
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
--show_log_iter 50 \
--coord_normalize \
--gcn \
--n_joints 14 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]" \
--opt AdaGrad \
--lr 0.0001
