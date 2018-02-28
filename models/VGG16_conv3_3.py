from __future__ import print_function
import collections
import os

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links import BatchNormalization
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable


class VGG16_conv3_3(link.Chain):

    """A pre-trained CNN model with 16 layers provided by VGG team.

    During initialization, this chain model automatically downloads
    the pre-trained caffemodel, convert to another chainer model,
    stores it on your local directory, and initializes all the parameters
    with it. This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.
    Note that this pre-trained model is released under Creative Commons
    Attribution License.

    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.

    See: K. Simonyan and A. Zisserman, `Very Deep Convolutional Networks
    for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`_

    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable. The converted chainer model is
            automatically used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.Normal(scale=0.01)``.

    Attributes:
        ~VGG16Layers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.

    """

    def __init__(self, n_joints):
        super(VGG16_conv3_3, self).__init__()

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1)
            self.bn1     = BatchNormalization(512)
            self.fc6     = Linear(None, 4096)
            self.bn2     = BatchNormalization(4096)
            self.fc7     = Linear(4096, 4096)
            self.bn3     = BatchNormalization(4096)
            self.fc8     = Linear(4096, n_joints * 2)

    def __call__(self, x):
        h = x

        h = relu(self.conv1_1(h))
        h = relu(self.conv1_2(h))
        h = _max_pooling_2d(h)

        h = relu(self.conv2_1(h))
        h = relu(self.conv2_2(h))
        h = _max_pooling_2d(h)

        h = relu(self.conv3_1(h))
        h = relu(self.conv3_2(h))
        h = relu(self.conv3_3(h))
        h = _max_pooling_2d(h)

        h = relu(self.conv4_1(h))
        h = relu(self.conv4_2(h))
        h = relu(self.conv4_3(h))
        h = _max_pooling_2d(h)

        h = relu(self.conv5_1(h))
        h = relu(self.conv5_2(h))
        h = relu(self.bn1(self.conv5_3(h)))
        h = _max_pooling_2d(h)

        h = dropout(relu(self.bn2(self.fc6(h))))
        h = dropout(relu(self.bn3(self.fc7(h))))
        h = self.fc8(h)

        return h

def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)
