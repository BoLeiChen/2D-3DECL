#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from habitat_baselines.rl.models.resnet import DetectronResNet50
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.common.utils import CategoricalNet
import habitat_baselines.rl.models.vision_transformer as vits
from einops import rearrange

import numpy as np
import cv2
from habitat_baselines.rl.ppo.utils.distributions import Categorical, DiagGaussian
from habitat_baselines.rl.ppo.utils.model import get_grid, ChannelPool, Flatten, NNBase
import habitat_baselines.rl.ppo.utils.depth_utils as du
from habitat_baselines.rl.ppo.utils.pointnet import PointNetEncoder
from habitat_baselines.rl.ppo.utils.ply import write_ply_xyz, write_ply_xyz_rgb
from habitat_baselines.rl.ppo.utils.img_save import save_semantic, save_KLdiv
import os



class ExplorePolicy(nn.Module):
    def __init__(self, policy_net, dim_actions):
        super().__init__()

        # CNN encoder to extract features from Policy network, NOT Visual encoder
        self.policy_net = policy_net

        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.policy_net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.policy_net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def forward_visual(self, observations):
        return self.visual_net(observations)

    def act(
        self,
        observations,
        obs_points,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states


    def get_value(self, observations, obs_points, rnn_hidden_states, prev_actions, masks):
        features, _ = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, obs_points, rnn_hidden_states, prev_actions, masks, action
    ):
        if isinstance(observations, dict):
            observations = observations['rgb']
        features, rnn_hidden_states = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action.argmax(dim=1))
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class ExploreBaselinePolicy(ExplorePolicy):
    def __init__(self,
                observation_space,
                action_space,
                hidden_size=512,
                num_recurrent_layers=1,
                rnn_type="LSTM",
                device=torch.device("cpu")):
        super().__init__(
            ExplorePolicyNet(
                observation_space, hidden_size, num_recurrent_layers, rnn_type, device
            ),
            action_space.n
        )

class ExploreBaselinePolicyRollout(ExplorePolicy):
    def __init__(self,
                action_dim,
                hidden_size=512,
                num_recurrent_layers=1,
                rnn_type="LSTM",
                device=torch.device("cpu")):
        super().__init__(
            ExplorePolicyNet(
                None, hidden_size, num_recurrent_layers, rnn_type, device
            ),
            action_dim
        )


class ExploreNet(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass


class ExplorePolicyNet(ExploreNet):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, num_recurrent_layers, rnn_type, device):
        super().__init__()

        self._hidden_size = hidden_size

        self.policy_encoder = DetectronResNet50(downsample=True, device_id=torch.cuda.current_device())
        #self.policy_encoder = vits.vit_base(patch_size = 16)
        #self.policy_encoder.to(device)
        self.state_encoder = RNNStateEncoder(
            #self.policy_encoder.embed_dim,
            self.policy_encoder.output_size,
            self._hidden_size,
            num_recurrent_layers,
            rnn_type
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        #observations = rearrange(observations, 'b h w c -> b c h w')
        perception_embed = self.policy_encoder(observations)
        x, rnn_hidden_states = self.state_encoder(perception_embed, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class Semantic_Mapping(nn.Module):
    """
    Semantic_Mapping
    """

    def __init__(self, args, device):
        super(Semantic_Mapping, self).__init__()
        # print(args.device)
        # exit(0)
        self.device = device
        self.step_size = args.step_size
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
                                self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
                                self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, observations, obs, pose_obs, poses_last, origins,
                observation_points, gl_tree_list, infos,
                args):

        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        point_cloud_t_3d = point_cloud_t.clone()

        agent_view_t_3d = point_cloud_t.clone().to(torch.float32)

        '''
        from matplotlib import pyplot as plt
        from PIL import  Image
        ax1 = plt.axes(projection='3d')
        ax1.set_xlabel('x', size=20)
        ax1.set_ylabel('y', size=20)
        ax1.set_zlabel('z', size=20)
        ax1.scatter3D(agent_view_t_3d.cpu()[0, :, :, 0], agent_view_t_3d.cpu()[0, :, :, 1], agent_view_t_3d.cpu()[0, :, :, 2], cmap='Blues')
        plt.show()

        rgb_vis = Image.fromarray(observations[0]['rgb'])
        rgb_vis.show(rgb_vis)
        '''

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                          torch.sin(pose[:, 2] / 57.29577951308232) \
                          + rel_pose_change[:, 1] * \
                          torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                          torch.cos(pose[:, 2] / 57.29577951308232) \
                          - rel_pose_change[:, 1] * \
                          torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)

        points_pose = current_poses.clone()
        points_pose[:, :2] = points_pose[:, :2] + torch.from_numpy(
            origins[:, :2]).to(self.device).float()

        points_pose[:, 2] = points_pose[:, 2] * np.pi / 180
        points_pose[:, :2] = points_pose[:, :2] * 100

        for e in range(bs):

            world_view_t = du.transform_pose_t2(
                agent_view_t_3d[e, ...], points_pose[e, ...].cpu().numpy(),
                self.device).reshape(-1, 3)

            world_view_sem_t = obs[e, 4:4 + (self.num_sem_categories), :,
                               :].reshape((self.num_sem_categories),
                                          -1).transpose(0, 1)

            non_zero_row_1 = torch.abs(
                point_cloud_t_3d[e, ...].reshape(-1, 3)).sum(dim=1) > 0
            #non_zero_row_2 = torch.abs(world_view_sem_t).sum(dim=1) > 0
            #non_zero_row_3 = torch.argmax(world_view_sem_t,
            #                              dim=1) != self.num_sem_categories - 1

            #non_zero_row = non_zero_row_1 & non_zero_row_2 & non_zero_row_3
            non_zero_row = non_zero_row_1
            non_zero_row = [ True for i in range(len(non_zero_row))]
            world_view_sem = world_view_sem_t[non_zero_row].cpu().numpy()

            if world_view_sem.shape[0] < 50:
                continue

            world_view_label = np.argmax(world_view_sem, axis=1)  #

            world_view_rgb = obs[e, :3, :, :].permute(1, 2, 0).reshape(-1, 3)[
                non_zero_row].cpu().numpy()
            world_view_t = world_view_t[non_zero_row].cpu().numpy()

            if world_view_t.shape[0] >= 1024:   #512
                indx = np.random.choice(world_view_t.shape[0], 1024, replace=False)     # 512
            else:
                indx = np.linspace(0, world_view_t.shape[0] - 1, world_view_t.shape[0]).astype(np.int32)

            gl_tree = gl_tree_list[e]
            gl_tree.init_points_node(world_view_t[indx])
            per_frame_nodes = gl_tree.add_points(world_view_t[indx],
                                                 world_view_sem[indx],
                                                 world_view_rgb[indx],
                                                 world_view_label[indx],
                                                 infos[e]['timestep'])
            scene_nodes = gl_tree.all_points()
            '''
            import open3d
            pos = []
            color = []
            for node in scene_nodes:
                pos.append(node.point_coor)
                color.append(node.point_color / 255)
            pos = np.array(pos)
            color = np.array(color)

            pt = open3d.geometry.PointCloud()

            pt.points = open3d.utility.Vector3dVector(pos)
            pt.colors = open3d.utility.Vector3dVector(color)
            open3d.visualization.draw_geometries([pt])
            '''

            gl_tree.update_neighbor_points(per_frame_nodes)

            #sample_points_tensor = torch.tensor(gl_tree.sample_points())  # local map
            sample_points_tensor = torch.tensor((gl_tree.sliding_window_points(
                          infos[e]['timestep'], self.step_size + 1)))

            sample_points_tensor[:, :2] = sample_points_tensor[:,
                                          :2] - origins[e, :2] * 100
            sample_points_tensor[:, 2] = sample_points_tensor[:,
                                         2] - 0.88 * 100
            sample_points_tensor[:, :3] = sample_points_tensor[:,
                                          :3] / args.map_resolution

            observation_points[e] = sample_points_tensor.transpose(1, 0)

        return current_poses, observation_points
