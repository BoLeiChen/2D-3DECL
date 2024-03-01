#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
import copy
import cv2
import gym.spaces
from PIL import Image

import numpy as np
import torch
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs, construct_envs_simple
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage, SimpleRolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, linear_decay
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.rl.models.resnet import DetectronResNet50
from habitat_baselines.rl.exploration.rnd import RND
from habitat_baselines.rl.exploration.crl import CRL
from habitat_baselines.rl.representation.invdyn import InverseDynamics
from habitat_baselines.rl.ppo.explore_policy import ExploreBaselinePolicy
from habitat_baselines.rl.ppo.ppo_trainer_reward_actpred import PPOTrainer_RewardActPred
from habitat_baselines.common.semantic_utils import main
from habitat_baselines.rl.ppo.explore_policy import Semantic_Mapping
from habitat_baselines.GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
from habitat_baselines.GLtree.octree import GL_tree

@baseline_registry.register_trainer(name="ddppo_reward_actpred")
class DDPPOTrainer_RewardActPred(PPOTrainer_RewardActPred):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

        PathManager.mkdirs(os.path.dirname(config.LOG_FILE))

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

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
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            device=self.device,
        )
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
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

    def save_image(self, rollouts, img_idx_init):
        global_img_dir = os.path.join(self.config.LOG_DIR, "ddppo_image")
        if img_idx_init == 0:
            os.makedirs(os.path.join(global_img_dir, "process_" + str(self.world_rank)), exist_ok = True)

        total_step = rollouts.observations.shape[0]
        for step in range(1, total_step):
            step_obs = rollouts.observations[step].detach().cpu().numpy()
            # check shape of paralleled observation
            img_idx = img_idx_init + step - 1
            local_img_dir = os.path.join(os.path.join(global_img_dir, "process_" + str(self.world_rank)), "%06d.png" % (img_idx))
            local_step_obs = step_obs[0].astype(np.uint8) # by default parallelize 4 process per GPU, only save first paralleled process
            color_image = Image.fromarray(local_step_obs, mode="RGB")
            color_image.save(local_img_dir)

        return img_idx


    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.ENV_CONFIG.SIMULATOR.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        if self.config.SAVE_PPO_IMAGE == True:
            image_index_initial = 0

        # print("random seed value at rank {} is {}".format(self.world_rank, self.config.ENV_CONFIG.SIMULATOR.SEED))

        random.seed(self.config.ENV_CONFIG.SIMULATOR.SEED)
        np.random.seed(self.config.ENV_CONFIG.SIMULATOR.SEED)
        torch.manual_seed(self.config.ENV_CONFIG.SIMULATOR.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs_simple(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        rl_cfg = self.config.RL.REWARD
        dyn_cfg = self.config.RL.ACTION
        maskrcnn_cfg = self.config.RL.MASKRCNN
        threedmap_cfg = self.config.RL.ThreeD_MAP

        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        if dyn_cfg.invdyn_mlp: # by default MLP arch
            self.dynamics_agent = InverseDynamics(
                self.actor_critic.policy_net.policy_encoder,
                dyn_cfg.action_dim,
                dyn_cfg.proj_dim,
                dyn_cfg.hidden_dim,
                dyn_cfg.num_steps,
                dyn_cfg.mini_batch_size,
                dyn_cfg.lr, # by default use same learning rate as policy learning
                dyn_cfg.gradient_updates,
                self.device,
                threedmap_cfg.num_sem_categories,
                rl_cfg.temperature
            )
            self.dynamics_agent = DistributedDataParallel(
                self.dynamics_agent, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=True
                )

            if self.world_rank == 0:
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
                #encoder = self.actor_critic.policy_net.policy_encoder, # deprecated crl_momentum
                proj_dim = rl_cfg.crl_repr_dim,
                hidden_dim = rl_cfg.crl_hidden_dim,
                simclr_lr = rl_cfg.crl_lr,
                temperature = rl_cfg.temperature,
                device = self.device
            )
        else:
            raise NotImplementedError
        self.reward_agent = DistributedDataParallel(
            self.reward_agent, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False
        )
################################################################################################

        self.gl_tree_list = []
        for e in range(self.config.NUM_PROCESSES):
            self.gl_tree_list.append(GL_tree(threedmap_cfg))

        # Calculating full and local map sizes
        self.map_size = threedmap_cfg.map_size_cm // threedmap_cfg.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / threedmap_cfg.global_downscaling)
        self.local_h = int(self.full_h / threedmap_cfg.global_downscaling)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.config.NUM_PROCESSES, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.config.NUM_PROCESSES, 3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((self.config.NUM_PROCESSES, 3))
        # Local Map Boundaries
        self.lmb = np.zeros((self.config.NUM_PROCESSES, 4)).astype(int)

        self.observation_points = torch.zeros(self.config.NUM_PROCESSES,
                                         threedmap_cfg.points_channel_num,
                                         threedmap_cfg.map_point_size)

        self.sem_map_module = Semantic_Mapping(threedmap_cfg, self.device).to(self.device)
        self.sem_map_module.eval()

        results = self.envs.reset()
        obs_after_process, observations, infos = [list(x) for x in zip(*results)]
        obs_after_process = [torch.from_numpy(x) for x in obs_after_process]
        obs_after_process = torch.stack(obs_after_process).to(self.device)

        batch = batch_obs(observations, device=self.device)


        def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if threedmap_cfg.global_downscaling > 1:
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

        def init_map_and_pose():
            self.observation_points.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:, :2] = threedmap_cfg.map_size_cm / 100.0 / 2.0

            locs = self.full_pose.cpu().numpy()
            for e in range(self.config.NUM_PROCESSES):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / threedmap_cfg.map_resolution),
                                int(c * 100.0 / threedmap_cfg.map_resolution)]

                self.lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (self.local_w, self.local_h),
                                                  (self.full_w, self.full_h))

                self.origins[e] = [self.lmb[e][2] * threedmap_cfg.map_resolution / 100.0,
                              self.lmb[e][0] * threedmap_cfg.map_resolution / 100.0, 0.]

            for e in range(self.config.NUM_PROCESSES):

                self.local_pose[e] = self.full_pose[e] - \
                                torch.from_numpy(self.origins[e]).to(self.device).float()

            for e in range(self.config.NUM_PROCESSES):
                self.gl_tree_list[e].reset_gltree()


        init_map_and_pose()

        obs_space = self.envs.observation_spaces[0]
        points_observation_space = gym.spaces.Box(0, 1, (
        threedmap_cfg.points_channel_num, threedmap_cfg.map_point_size),
                                                  dtype='float32')

        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = SpaceDict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

##############################################################################################
        rollouts = SimpleRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            points_observation_space.shape,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            self.agent.actor_critic.policy_net.num_recurrent_layers
        )
        rollouts.to(self.device)


        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx in range(self.config.NUM_PROCESSES)])
        ).float().to(self.device)


        self.local_pose, self.observation_points = \
            self.sem_map_module(observations, obs_after_process, poses, self.local_pose, self.origins,
                           self.observation_points, self.gl_tree_list,
                           infos, threedmap_cfg)
        '''
        import open3d
        points_vis = self.observation_points[0][0:3, :].transpose(1, 0).numpy()
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points_vis)
        open3d.visualization.draw_geometries([point_cloud])
        '''
        # env reset obs
        rollouts.observations.copy_(batch['rgb'])
        rollouts.obs_points.copy_(self.observation_points)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:

            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        )

                    requeue_job()
                    return

                count_steps_delta = 0
                # no need to input model.eval() since it will be in mode torch.no_grad()
                # self.agent.eval()
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
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                # ABLATION: save all images in ddppo for independent self-supervised training
                if self.config.SAVE_PPO_IMAGE:
                    image_index_initial = self.save_image(rollouts, image_index_initial)

                # update visual encoder of RL policy with inverse dynamics objective
                if dyn_cfg.invdyn_mlp:
                    inv_dyn_loss = self.dynamics_agent.module.update_invdyn(rollouts.observations, rollouts.obs_points, rollouts.actions)
                    # average across multiple gpus
                    inv_dyn_loss = comm.reduce_dict(inv_dyn_loss)

                    if self.world_rank == 0:
                        #writer.add_scalar("crl_x_total_loss", inv_dyn_loss["crl_x_total_loss"], count_steps)
                        writer.add_scalar("simclr_with_inv_dyn_loss", inv_dyn_loss["simclr_with_inv_dyn_loss"], count_steps)
                        writer.add_scalar("inv_dyn_loss", inv_dyn_loss["inv_dyn_loss"], count_steps)
                        writer.add_scalar("inv_dyn_acc", inv_dyn_loss["pred_acc"], count_steps)

                # update exploration agent with RND curiosity objective
                if rl_cfg.rnd:
                    for _ in range(rl_cfg.gradient_updates):
                        rnd_loss = self.reward_agent.module.update(rollouts.observations[1:])
                    # average across multiple gpus
                    rnd_loss = comm.reduce_dict(rnd_loss)

                    if self.world_rank == 0:
                        writer.add_scalar("rnd_loss", rnd_loss["rnd_loss"], count_steps)

                elif rl_cfg.crl:
                    for _ in range(rl_cfg.gradient_updates):
                        crl_loss = self.reward_agent.module.update(rollouts.observations[1:], rollouts.actions)
                    # average across multiple gpus
                    crl_loss = comm.reduce_dict(crl_loss)

                    if self.world_rank == 0:
                        writer.add_scalar("crl_loss", crl_loss["crl_loss"], count_steps)
                else:
                    raise NotImplementedError

                # THIRD: compute intrinsic reward with unsup RL objective
                with torch.no_grad():
                    if rl_cfg.rnd:
                        intr_reward = self.reward_agent.module.compute_rnd_reward(rollouts.observations[1:])
                    elif rl_cfg.crl:
                        intr_reward = self.reward_agent.module.compute_simclr_reward(rollouts.observations[1:], rollouts.actions)

                # reset rollouts rewards
                rollouts.rewards.copy_(intr_reward.detach())

                # recompute episode reward based on intrinsic reward
                for step in range(ppo_cfg.num_steps):
                    step = torch.tensor(step, dtype=torch.int).to(self.device)
                    # rollouts.rewards has shape (num_step, num_env, obs_shape)
                    # current_episode_reward is on cpu; rollouts is on gpu
                    current_episode_reward += torch.index_select(rollouts.rewards, 0, step).squeeze(0)
                    running_episode_stats["reward"] += (1 - torch.index_select(rollouts.masks, 0, step).squeeze(0)) * current_episode_reward
                    running_episode_stats["count"] += 1 - torch.index_select(rollouts.masks, 0, step).squeeze(0)
                    current_episode_reward *= torch.index_select(rollouts.masks, 0, step).squeeze(0)
                del step

                # remove bc we don't change mode anyway
                # self.agent.train()
                if self._static_encoder:
                    self._encoder.eval()

                # FINALLY: update actor critic agent using PPO
                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time
                
                # random sample data for downstream Mask-RCNN training
                if update > 0 and maskrcnn_cfg.rollout == True:
                    if update % maskrcnn_cfg.rollout_between_step == 0:
                        with torch.no_grad():

                            # save images and labels
                            main(
                                # os.path.join(self.config.CHECKPOINT_FOLDER, "ckpt.temp.pth"),
                                copy.deepcopy(self.config),
                                self.agent.state_dict(),
                                self.config.ROLLOUT_DIR,
                                maskrcnn_cfg.rollout_length,
                                maskrcnn_cfg.rollout_prob,
                                int((update // maskrcnn_cfg.rollout_between_step - 1) * maskrcnn_cfg.rollout_length),
                                self.world_rank,
                                self.world_size
                            )
                            distrib.barrier()
                
                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                        [value_loss, action_loss, count_steps_delta],
                        device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[2].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                            stats[0].item() / self.world_size,
                            stats[1].item() / self.world_size,
                        ]

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar("reward", deltas["reward"] / deltas["count"], count_steps,)

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        writer.add_scalars("metrics", metrics, count_steps)

                    writer.add_scalars(
                            "losses",
                            {k: l for l, k in zip(losses, ["value", "policy"])},
                            count_steps,
                        )

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                count_steps
                                / ((time.time() - t_start) + prev_time),
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
                        # save agent and RL config
                        self.save_checkpoint(
                            f"exploration.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        #self.reward_agent.module.save(self.config.CHECKPOINT_FOLDER, count_checkpoints) # have an ["encoder"] entry
                        # save additional models, otherwise exploration baselines
                        if dyn_cfg.invdyn_mlp:
                            #self.dynamics_agent.module.save(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                            self.dynamics_agent.module.save_encoder(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                            self.dynamics_agent.module.save_pcl_encoder(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                        count_checkpoints += 1

            self.envs.close()
