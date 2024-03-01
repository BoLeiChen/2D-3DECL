#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs, construct_envs_simple
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.constants import scenes
from habitat_baselines.common.rollout_storage import RolloutStorage, SimpleRolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    generate_map_image
)
from habitat_baselines.rl.ppo import PPO, ExploreBaselinePolicy

# exploration agent
from habitat_baselines.rl.exploration.rnd import RND
from habitat_baselines.rl.exploration.crl import CRL
# representation learning model
from habitat_baselines.rl.representation.invdyn import InverseDynamics
from habitat_baselines.rl.ppo.utils.constants import color_palette_array

@baseline_registry.register_trainer(name="ppo_reward_actpred")
class PPOTrainer_RewardActPred(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ExploreBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = rollouts.observations[rollouts.step]
            step_obs_points = rollouts.obs_points[rollouts.step]

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states
            ) = self.actor_critic.act(
                step_observation,
                step_obs_points,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        #for a in actions:
        #    a[0] = torch.randint(0, 3, (1,))
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", actions)
        # step in the enviorment: need to +1 to account for index difference
        outputs = self.envs.step([a[0].item() + 1 for a in actions])

        obs_after_process, observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        obs_after_process = [torch.from_numpy(x) for x in obs_after_process]
        obs_after_process = torch.stack(obs_after_process).to(self.device)

        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx in range(self.config.NUM_PROCESSES)])
        ).float().to(self.device)

        self.local_pose, self.observation_points = \
            self.sem_map_module(observations, obs_after_process, poses, self.local_pose, self.origins,
                           self.observation_points, self.gl_tree_list,
                           infos, self.config.RL.ThreeD_MAP)
        '''
        import open3d
        pt = open3d.geometry.PointCloud()

        points_vis = self.observation_points[0][0:3, :].transpose(1, 0).numpy()
        points_vis_semantics = torch.argmax(self.observation_points[0][3:3+self.config.RL.ThreeD_MAP.num_sem_categories, :], dim=0) + 2
        #points_vis_semantics[points_vis_semantics == 3] = 0
        points_vis_colors = color_palette_array[points_vis_semantics.numpy()]

        pt.points = open3d.utility.Vector3dVector(points_vis)
        pt.colors = open3d.utility.Vector3dVector(points_vis_colors)
        open3d.visualization.draw_geometries([pt])

        #point_cloud = open3d.geometry.PointCloud()
        #point_cloud.points = open3d.utility.Vector3dVector(points_vis)
        #open3d.visualization.draw_geometries([point_cloud])
        '''
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        #print(infos[0]["timestep"])
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)
        # target shape of rewards tensor: num_envs x 1

        def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if self.config.RL.ThreeD_MAP.global_downscaling > 1:
                gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
                gx2, gy2 = gx1 + local_w, gy1 + local_h
                if gx1 < 0:
                    gx1, gx2 = 0, local_w
                if gx2 > full_w:
                    gx1, gx2 = full_w - local_w, full_w

                if gy1 < 0:
                    gy1, gy2 = 0, local_h
                if gy2 > full_h:
                    gy1, gy2 = full_h - local_h, full_h
            else:
                gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

            return [gx1, gx2, gy1, gy2]

        def init_map_and_pose_for_env(e):
            self.observation_points[e].fill_(0.)
            self.full_pose[e].fill_(0.)
            self.full_pose[e, :2] = self.config.RL.ThreeD_MAP.map_size_cm / 100.0 / 2.0
            self.gl_tree_list[e].reset_gltree()

            locs = self.full_pose[e].cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / self.config.RL.ThreeD_MAP.map_resolution),
                            int(c * 100.0 / self.config.RL.ThreeD_MAP.map_resolution)]


            self.lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (self.local_w, self.local_h),
                                              (self.full_w, self.full_h))

            self.origins[e] = [self.lmb[e][2] * self.config.RL.ThreeD_MAP.map_resolution / 100.0,
                          self.lmb[e][0] * self.config.RL.ThreeD_MAP.map_resolution / 100.0, 0.]

            self.local_pose[e] = self.full_pose[e] - \
                            torch.from_numpy(self.origins[e]).to(self.device).float()

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        for e, x in enumerate(dones):
            if x:
                init_map_and_pose_for_env(e)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        # one hot encode actions
        actions = torch.nn.functional.one_hot(actions, num_classes=3)
        actions = torch.squeeze(actions).float()

        # try to understand (s, a, s')
        # print("At timestep {}".format(rollouts.step))
        # print("Previous observation {} saved at index {}".format(step_observation, rollouts.step))
        # print("Take action {} saved at index {}".format(actions, rollouts.step))
        # print("Next observation {} saved at index {}".format(observations, rollouts.step + 1))
        # input("Press enter to continue")

        rollouts.insert(
            batch,
            self.observation_points,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = rollouts.observations[rollouts.step]
            last_obs_points = rollouts.obs_points[rollouts.step]
            next_value = self.actor_critic.get_value(
                last_observation,
                last_obs_points,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )


    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs_simple(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        rl_cfg = self.config.RL.REWARD
        dyn_cfg = self.config.RL.ACTION
        maskrcnn_cfg = self.config.RL.MASKRCNN

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        if dyn_cfg.invdyn_mlp:
            self.dynamics_agent = InverseDynamics(
                self.actor_critic.policy_net.policy_encoder,
                dyn_cfg.action_dim,
                dyn_cfg.proj_dim,
                dyn_cfg.hidden_dim,
                dyn_cfg.num_steps,
                dyn_cfg.mini_batch_size,
                dyn_cfg.lr, # by default use same learning rate as policy learning
                dyn_cfg.gradient_updates,
                self.device
            )
            logger.info(
                "idm number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.dynamics_agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        if rl_cfg.rnd:
            self.reward_agent = RND(
                # encoder = self.actor_critic.policy_net.policy_encoder, # deprecated rnd_momentum
                hidden_dim = rl_cfg.rnd_hidden_dim,
                rnd_repr_dim = rl_cfg.rnd_repr_dim,
                learning_rate = rl_cfg.rnd_lr,
                device = self.device
            )
        elif rl_cfg.crl:
            self.reward_agent = CRL(
                # encoder = self.actor_critic.policy_net.policy_encoder, # deprecated crl_momentum
                proj_dim = rl_cfg.crl_repr_dim,
                hidden_dim = rl_cfg.crl_hidden_dim,
                simclr_lr = rl_cfg.crl_lr,
                temperature = rl_cfg.temperature,
                device = self.device
            )
        else:
            raise NotImplementedError

        rollouts = SimpleRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        # env reset obs
        rollouts.observations.copy_(batch['rgb'])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                # FIRST: update visual encoder of RL policy with inverse dynamics objective
                if dyn_cfg.invdyn_mlp:
                    inv_dyn_loss = self.dynamics_agent.update_invdyn(rollouts.observations, rollouts.actions) # multiple iterations adopted in nn.Module

                    writer.add_scalar("inv_dyn_loss", inv_dyn_loss["inv_dyn_loss"], count_steps)
                    writer.add_scalar("inv_dyn_acc", inv_dyn_loss["pred_acc"], count_steps)

                # SECOND: update exploration agent
                if rl_cfg.rnd:
                    for _ in range(rl_cfg.gradient_updates):
                        rnd_loss = self.reward_agent.update(rollouts.observations[1:])
                    writer.add_scalar("rnd_loss", rnd_loss["rnd_loss"], count_steps)

                elif rl_cfg.crl:
                    for _ in range(rl_cfg.gradient_updates):
                        crl_loss = self.reward_agent.update(rollouts.observations[1:])
                    writer.add_scalar("crl_loss", crl_loss["crl_loss"], count_steps)

                else:
                    raise NotImplementedError

                # compute intrinsic reward
                with torch.no_grad():
                    if rl_cfg.rnd:
                        intr_reward = self.reward_agent.compute_rnd_reward(rollouts.observations[1:])
                    elif rl_cfg.crl:
                        intr_reward = self.reward_agent.compute_simclr_reward(rollouts.observations[1:])
                    else:
                        raise NotImplementedError
                # reset rollouts rewards
                rollouts.rewards.copy_(intr_reward.detach())

                # recompute episode reward based on intrinsic reward
                for step in range(ppo_cfg.num_steps):
                    step = torch.tensor(step, dtype=torch.int)
                    # rollouts.rewards has shape (num_step, num_env, obs_shape)
                    # current_episode_reward is on cpu; rollouts is on gpu
                    current_episode_reward += torch.index_select(rollouts.rewards.cpu(), 0, step).squeeze(0)
                    running_episode_stats["reward"] += (1 - torch.index_select(rollouts.masks.cpu(), 0, step).squeeze(0)) * current_episode_reward
                    running_episode_stats["count"] += 1 - torch.index_select(rollouts.masks.cpu(), 0, step).squeeze(0)
                    current_episode_reward *= torch.index_select(rollouts.masks.cpu(), 0, step).squeeze(0)

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    self.reward_agent.save(self.config.CHECKPOINT_FOLDER, count_checkpoints) # have an ["encoder"] entry
                    self.dynamics_agent.save(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                    count_checkpoints += 1

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

            self.envs.close()


    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()
        config.defrost()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)

        config.ENV_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config


    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        logger.info(f"Eval ckpt path at {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = batch["rgb"]

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.policy_net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        map_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            os.makedirs(self.config.MAP_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            batch = batch["rgb"]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        generate_map_image(
                            video_option=self.config.VIDEO_OPTION,
                            map_dir=self.config.MAP_DIR,
                            images=map_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                        map_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame, map_frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    map_frames[i].append(map_frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

