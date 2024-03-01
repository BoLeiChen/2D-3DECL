import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from habitat_baselines.common.rednet import SemanticPredRedNet

class Obs_Preprocess():

    def __init__(self, args):
        self.args = args
        self.sem_pred_model = SemanticPredRedNet(args)
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
        self.rgb_vis = None

    def _get_sem_pred(self, rgb, depth, cat_goal):

        semantic_pred, sem_entropy, sem_goal_prob = self.sem_pred_model.get_prediction \
            (rgb, depth, cat_goal)
        self.rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return semantic_pred, sem_entropy, sem_goal_prob

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        mask1 = depth >0.99
        mask2 = depth == 0
        depth =  depth * (max_d - min_d) * 100 + min_d * 100
        depth[mask1] = 0
        depth[mask2] = 0

        return depth

    def _preprocess_obs(self, obs, cat_goal_id):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred, sem_seg_entropy, sem_goal_pred = self._get_sem_pred \
            (rgb.astype(np.uint8), depth, cat_goal_id)


        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor

        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]
            sem_seg_entropy = sem_seg_entropy[ds // 2::ds, ds // 2::ds]
            sem_goal_pred = sem_goal_pred[ds // 2::ds, ds // 2::ds]


        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred, sem_seg_entropy[:, :, None], sem_goal_pred[:, :, None]),
                               axis=2).transpose(2, 0, 1)

        return state
