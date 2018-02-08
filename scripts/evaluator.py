
from chainer import Variable, Chain
from chainer import report
from chainer.cuda import get_device
from chainer.functions import mean_squared_error
import numpy as np

## deeppose_tf
from deeppose_tf.scripts.regressionnet import calculate_metric
from deeppose_tf import poseevaluation


class PoseEvaluateModel(Chain):

    def __init__(self, predictor, dataset_name):
        super(PoseEvaluateModel, self).__init__(predictor=predictor)
        self.dataset_name = dataset_name

    def __call__(self, image, joints, is_valid_joints, crop_bbox, orig_bbox):

        ## predict joints, and calculate loss
        y = self.predictor(image)
        with get_device(image):
            t = (joints * is_valid_joints).astype(joints.dtype)
        loss = mean_squared_error(y, t)

        joints = Variable(joints)
        joints.to_cpu()
        joints = joints.data.reshape(-1, 14, 2)
        crop_bbox = Variable(crop_bbox)
        crop_bbox.to_cpu()
        crop_bbox = crop_bbox.data
        y.to_cpu()
        y = y.data.reshape(-1, 14, 2)

        ## evaluate with PCP, mPCP metrics
        pcp_per_stick = calculate_metric(joints, y, crop_bbox,
                self.dataset_name, 'PCP')
        pcp_per_part, pcp_part_names = \
                poseevaluation.pcp.average_pcp_left_right_limbs(pcp_per_stick)

        relaxed_pcp_per_stick = calculate_metric(joints, y, crop_bbox,
                self.dataset_name, 'RelaxedPCP')
        relaxed_pcp_per_part, relaxed_pcp_part_names = \
                poseevaluation.pcp.average_pcp_left_right_limbs(relaxed_pcp_per_stick)

        ## Report above meansured value
        r['loss'] = loss
        r['PCP']  = np.mean(pcp_per_stick)
        r['mPCP'] = np.mean(relaxed_pcp_per_stick)
        for name, score in zip(pcp_pert_names, pcp_per_part):
            name = "PCP/{}". format(name)
            r[name] = score
        for name, score in zip(relaxed_pcp_part_names, relaxed_pcp_per_part):
            name = "mPCP/{}". format(name)
            r[name] = score
        report(r, self)
