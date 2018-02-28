

from chainer.links import VGG16Layers
from VGG16_conv3_3 import VGG16_conv3_3
from VGG16_conv4_3 import VGG16_conv4_3
from chainer.serializers import save_npz

n_joints = 14
org_model = VGG16Layers()

dst_model = VGG16_conv3_3(n_joints)
dst_model.conv1_1.W.data = org_model.conv1_1.W.data
dst_model.conv1_2.W.data = org_model.conv1_2.W.data
dst_model.conv2_1.W.data = org_model.conv2_1.W.data
dst_model.conv2_2.W.data = org_model.conv2_2.W.data
dst_model.conv3_1.W.data = org_model.conv3_1.W.data
dst_model.conv3_2.W.data = org_model.conv3_2.W.data
dst_model.conv3_3.W.data = org_model.conv3_3.W.data
dst_model.conv4_1.W.data = org_model.conv4_1.W.data
dst_model.conv4_2.W.data = org_model.conv4_2.W.data
dst_model.conv4_3.W.data = org_model.conv4_3.W.data
dst_model.conv5_1.W.data = org_model.conv5_1.W.data
dst_model.conv5_2.W.data = org_model.conv5_2.W.data
dst_model.conv5_3.W.data = org_model.conv5_3.W.data
dst_model.conv1_1.b.data = org_model.conv1_1.b.data
dst_model.conv1_2.b.data = org_model.conv1_2.b.data
dst_model.conv2_1.b.data = org_model.conv2_1.b.data
dst_model.conv2_2.b.data = org_model.conv2_2.b.data
dst_model.conv3_1.b.data = org_model.conv3_1.b.data
dst_model.conv3_2.b.data = org_model.conv3_2.b.data
dst_model.conv3_3.b.data = org_model.conv3_3.b.data
dst_model.conv4_1.b.data = org_model.conv4_1.b.data
dst_model.conv4_2.b.data = org_model.conv4_2.b.data
dst_model.conv4_3.b.data = org_model.conv4_3.b.data
dst_model.conv5_1.b.data = org_model.conv5_1.b.data
dst_model.conv5_2.b.data = org_model.conv5_2.b.data
dst_model.conv5_3.b.data = org_model.conv5_3.b.data
save_npz('VGG16_conv3_3.npy', dst_model)


dst2_model = VGG16_conv4_3(n_joints)
dst2_model.conv1_1.W.data = org_model.conv1_1.W.data
dst2_model.conv1_2.W.data = org_model.conv1_2.W.data
dst2_model.conv2_1.W.data = org_model.conv2_1.W.data
dst2_model.conv2_2.W.data = org_model.conv2_2.W.data
dst2_model.conv3_1.W.data = org_model.conv3_1.W.data
dst2_model.conv3_2.W.data = org_model.conv3_2.W.data
dst2_model.conv3_3.W.data = org_model.conv3_3.W.data
dst2_model.conv4_1.W.data = org_model.conv4_1.W.data
dst2_model.conv4_2.W.data = org_model.conv4_2.W.data
dst2_model.conv4_3.W.data = org_model.conv4_3.W.data
dst2_model.conv5_1.W.data = org_model.conv5_1.W.data
dst2_model.conv5_2.W.data = org_model.conv5_2.W.data
dst2_model.conv5_3.W.data = org_model.conv5_3.W.data
dst2_model.conv1_1.b.data = org_model.conv1_1.b.data
dst2_model.conv1_2.b.data = org_model.conv1_2.b.data
dst2_model.conv2_1.b.data = org_model.conv2_1.b.data
dst2_model.conv2_2.b.data = org_model.conv2_2.b.data
dst2_model.conv3_1.b.data = org_model.conv3_1.b.data
dst2_model.conv3_2.b.data = org_model.conv3_2.b.data
dst2_model.conv3_3.b.data = org_model.conv3_3.b.data
dst2_model.conv4_1.b.data = org_model.conv4_1.b.data
dst2_model.conv4_2.b.data = org_model.conv4_2.b.data
dst2_model.conv4_3.b.data = org_model.conv4_3.b.data
dst2_model.conv5_1.b.data = org_model.conv5_1.b.data
dst2_model.conv5_2.b.data = org_model.conv5_2.b.data
dst2_model.conv5_3.b.data = org_model.conv5_3.b.data
save_npz('VGG16_conv4_3.npy', dst2_model)
