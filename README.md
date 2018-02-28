# DeepPose

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

This programs in this repository use [deeppose_tf](https://github.com/asanakoy/deeppose_tf) to evaluate, to get (x,t) from dataset.

# Requirements

- Python 3.5.1+
  - [Chainer 1.13.0+](https://github.com/pfnet/chainer)
  - numpy 1.9+
  - scikit-image 0.11.3+
  - OpenCV 3.1.0+

I strongly recommend to use Anaconda environment. This repo may be able to be used in Python 2.7 environment, but I haven't tested.

## Installation of dependencies

```
pip install chainer
pip install numpy
pip install scikit-image
pip install scipy
pip install tqdm
# for python3
conda install -c https://conda.binstar.org/menpo opencv3
# for python2
conda install opencv
```

### Dataset preparation

```sh
cd deeppose_tf/datasets
bash download.sh
cd ..
python datasets/lsp_dataset.py
python datasets/mpii_dataset.py
```

- [LSP dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) (1000 tran / 1000 test images)
- [LSP Extended dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip) (10000 train images)
- **MPII dataset** (use original train set and split it into 17928 train / 1991 test images)
    - [Annotation](http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz)
    - [Images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

Tested dataset is LSP-ext only.

## MPII Dataset

- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)
- training images: 18079, test images: 6908
  - test images don't have any annotations
  - so we split trining imges into training/test joint set
  - each joint set has
- training joint set: 17928, test joint set: 1991

# Start training

Starting with the prepared shells is the easiest way. If you want to run `train-single.py` with your own settings,
please check the options first by `python scripts/train-single.py --help` and modify one of the following shells to customize training settings.

## For LSP Dataset

```
bash shells/train_lsp.sh
```

### Fine tuning of VGG16

A neural network model before conv3 or conv4 is frozen, and 
remain a part of neural network is trained.

1. Create the model which a part of conv-net is frozen. Two models(.npy) are created.
2. Move these model to target result directory.
  When you use `--resume-model` option, `train-single.py` use the directory which the resume model is stored, as the result directory.
3. Run training script.

```
# In examples, training the NN after conv3.
python scripts/initmodel.py
mv <conv-net> <result-directory>
# Change --resume-model to <result-directory>
$EDITOR shells/train_lsp_VGG16_conv3_3.sh
bash shells/train_lsp_VGG16_conv3_3.sh
```

## For MPII Dataset

```
bash shells/train_mpii.sh
```

# Prediction

Will add some tools soon
