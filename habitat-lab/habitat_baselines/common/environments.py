#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import os
import random
from datetime import datetime
import gym
import numpy as np
import quaternion
import torch
from gym.spaces.dict_space import Dict as SpaceDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.constants import scenes, master_scene_dir
from habitat.core.simulator import Observations, Simulator, AgentState
from habitat.sims import make_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
import cv2
from PIL import Image
from torchvision import transforms
from habitat_baselines.common.rednet import SemanticPredRedNet
from habitat_baselines.common.maskrcnn import SemanticPredMaskRCNN
import habitat_baselines.common.pose as pu

# adapted from detectron2
def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
        )
        seed = int(seed % 1e4)
    # np.random.seed(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _sim: Simulator
    _agent_state: AgentState
    _max_episode_steps: int
    _elapsed_steps: int

    def __init__(self, config):
        self._config = config
        self._sim = None
        self._agent = None

        # specify episode information
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        # reward range
        self.reward_range = self.get_reward_range()
        if self._config.ENVIRONMENT.dataset == 'mp3d':
            self.sem_pred_model = SemanticPredRedNet(config)
        elif self._config.ENVIRONMENT.dataset == 'gibson':
            self.sem_pred_model = SemanticPredMaskRCNN(config)
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((config.ENVIRONMENT.frame_height, config.ENVIRONMENT.frame_width),
                               interpolation=Image.NEAREST)])
        self.rgb_vis = None

        obs_after_process, observations, infos = self.reset()

        # specify action and observation space
        self.observation_space = self._sim.sensor_suite.observation_spaces # 256 x 256 x 3
        self.action_space = self._sim.action_space # Discrete(3) starting from 1
        self.last_sim_location = None

    def _get_sem_pred(self, rgb, depth, cat_goal):
        if self._config.ENVIRONMENT.dataset == "mp3d":
            semantic_pred, sem_entropy, sem_goal_prob = self.sem_pred_model.get_prediction \
                (rgb, depth, cat_goal)
        elif self._config.ENVIRONMENT.dataset == "gibson":
            semantic_pred, sem_entropy, sem_goal_prob = self.sem_pred_model.get_prediction \
                (rgb, cat_goal)
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
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
    
        sem_seg_pred, sem_seg_entropy, sem_goal_pred = self._get_sem_pred \
            (rgb.astype(np.uint8), depth, cat_goal_id)

        depth = self._preprocess_depth(depth, self._config.ENVIRONMENT.min_depth, self._config.ENVIRONMENT.max_depth)

        ds = self._config.ENVIRONMENT.env_frame_width // self._config.ENVIRONMENT.frame_width  # Downscaling factor

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

    def step(self, action):
        # self._sim._prev_sim_obs record previous timestep observation
        observations = self._sim.step(action)
        reward = self.get_reward(observations)
        done = self.get_done(observations)

        # increment single timestep
        self._elapsed_steps += 1

        info = self.get_info(observations)
        goal_cat_id = 0
        obs_after_process = np.concatenate((observations["rgb"], observations["depth"]), axis=2).transpose(2,0,1)
        obs_after_process = self._preprocess_obs(obs_after_process, goal_cat_id)

        return obs_after_process, observations, reward, done, info

    def reset(self):
        # cannot use parent reset: self._env.reset()
        # return: init_obs

        # change config node
        self._config.defrost()

        if self._config.ENVIRONMENT.SCENE == "none":
            # random select a scene
            scene_str = self.random_scene() # default training scenes
            # print("initializing scene {}".format(scene_str))
            self._config.SIMULATOR.SCENE = master_scene_dir + scene_str + '.glb'
        else:
            self._config.SIMULATOR.SCENE = master_scene_dir + self._config.ENVIORNMENT.SCENE + '.glb'

        self._config.freeze()

        if self._sim is None:
            # construct habitat simulator
            self._sim = make_sim(
                id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
            )
        else:
            self._sim.reconfigure(self._config.SIMULATOR)

        # initialize scene simulator
        observations = self._sim.reset()

        # initialize the agent at a random start state
        self._agent_state = self._sim.get_agent_state()
        self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
        random_rotation = np.random.rand(4) # rotation has dimension 4
        random_rotation[1] = 0.0
        random_rotation[3] = 0.0
        self._agent_state.rotation = quaternion.as_quat_array(random_rotation)
        self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)
        observations = self._sim.get_observations_at(self._agent_state.position, self._agent_state.rotation, True)
        color_density = 1 - np.count_nonzero(observations['rgb']) / np.prod(observations['rgb'].shape)
        valid_start = color_density < 0.05

        while not valid_start:
            # initialize the agent at a random start state
            self._agent_state = self._sim.get_agent_state()
            self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
            random_rotation = np.random.rand(4) # rotation has dimension 4
            random_rotation[1] = 0.0
            random_rotation[3] = 0.0
            self._agent_state.rotation = quaternion.as_quat_array(random_rotation)
            self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)
            observations = self._sim.get_observations_at(self._agent_state.position, self._agent_state.rotation, True)
            color_density = 1 - np.count_nonzero(observations['rgb']) / np.prod(observations['rgb'].shape)
            valid_start = color_density < 0.05

        # reset count
        self._elapsed_steps = 0

        self.last_sim_location = self.get_sim_location()
        info = self.get_info(observations)
        goal_cat_id = 0
        obs_after_process = np.concatenate((observations["rgb"], observations["depth"]), axis=2).transpose(2,0,1)
        obs_after_process = self._preprocess_obs(obs_after_process, goal_cat_id)

        return obs_after_process, observations, info


    def reset_scene(self, scene_str, seed=None):
        # change config node
        self._config.defrost()

        # set specific random seed
        self._config.SIMULATOR.SEED = seed
        random.seed(self._config.SIMULATOR.SEED)

        # set specific scene
        self._config.SIMULATOR.SCENE = master_scene_dir + scene_str + '.glb'

        self._config.freeze()

        if self._sim is None:
            # construct habitat simulator
            self._sim = make_sim(
                id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
            )
        else:
            self._sim.reconfigure(self._config.SIMULATOR)

        # initialize scene simulator
        observations = self._sim.reset()

        # initialize the agent at a random start state
        self._agent_state = self._sim.get_agent_state()
        self._agent_state.position = self._sim.sample_navigable_point() # location has dimension 3
        self._sim.set_agent_state(self._agent_state.position, self._agent_state.rotation)

        # reset count
        self._elapsed_steps = 0

        return observations

    # not used actually, since we have intrinsic reward
    def get_reward_range(self):
        return (-1.0, 1.0)

    # no task reward here
    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._elapsed_steps + 1 >= self._max_episode_steps

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = self._sim.get_agent_state()
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose

        return dx, dy, do

    def get_info(self, observations):
        dx, dy, do = self.get_pose_change()
        return {"sensor_pose": [dx, dy, do], "timestep": self._elapsed_steps}

    def close(self):
        self._sim.close()

    def random_scene(self, mode='train'):
        return random.choice(scenes[mode])
