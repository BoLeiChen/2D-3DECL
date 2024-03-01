import contextlib
import os 
import random
import numpy as np
import time 
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from gym import spaces
from detectron2.utils import comm 

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.rollout_storage import RandomRolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.common.env_utils import construct_envs, construct_envs_simple
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.rl.models.resnet import DetectronResNet50
from habitat_baselines.rl.representation.cpc import CPC 
from habitat_baselines.rl.representation.simclr import SimCLR 
from habitat_baselines.rl.representation.invdyn import InverseDynamics 


class RandomTrainer(BaseTrainer):

    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30

        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")
    
    @property
    def flush_secs(self):
        return self._flush_secs
    
    def _collect_rollout_step(self, rollouts):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        num_actions = self.envs.action_spaces[0].n # list of Discrete(num_actions)
        actions = torch.randint(num_actions, size=(len(self.envs.action_spaces), 1), dtype=torch.long, device=torch.device(self.local_rank))
        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        # step in the enviorment: need to +1 to account for index difference
        outputs = self.envs.step([a[0].item() + 1 for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=torch.device(self.local_rank),
        )

        # one hot encode actions
        actions = torch.nn.functional.one_hot(actions, num_classes=3)
        actions = actions.squeeze().float()

        rollouts.insert(
            batch,
            actions,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs
    
    def train(self) -> None:
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

        random.seed(self.config.ENV_CONFIG.SIMULATOR.SEED)
        np.random.seed(self.config.ENV_CONFIG.SIMULATOR.SEED)
        torch.manual_seed(self.config.ENV_CONFIG.SIMULATOR.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        ppo_cfg = self.config.RL.PPO # keep same hyperparameters as RL training
        # add unsup RL objective
        unsup_cfg = self.config.RL.UNSUP 
        # add dynamics oriented objective
        dyn_cfg = self.config.RL.ACTION

        self.envs = construct_envs_simple(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        logger.add_filehandler(self.config.LOG_FILE)

        # visual encoder
        self.visual_encoder = DetectronResNet50(pretrained=False, downsample=True, device_id=self.local_rank)
        self.visual_encoder = DistributedDataParallel(self.visual_encoder, device_ids=[self.local_rank], output_device=self.local_rank)
        
        # set up unsup RL objective
        if unsup_cfg.cpc:
            self.unsup_agent = CPC(
                self.visual_encoder, # visual encoder
                unsup_cfg.proj_dim,
                unsup_cfg.hidden_dim,
                unsup_cfg.future_num_steps,
                unsup_cfg.lr, # by default use same learning rate as policy learning
                self.device
            )
            self.unsup_agent = DistributedDataParallel(
                self.unsup_agent, device_ids=[self.local_rank], output_device=self.local_rank
                )
        elif unsup_cfg.simclr:
            self.unsup_agent = SimCLR(
                self.visual_encoder, # visual encoder
                unsup_cfg.proj_dim,
                unsup_cfg.hidden_dim,
                unsup_cfg.lr, # by default use same learning rate as policy learning
                unsup_cfg.mini_batch_size,
                unsup_cfg.temperature,
                self.device
            )
            self.unsup_agent = DistributedDataParallel(
                self.unsup_agent, device_ids=[self.local_rank], output_device=self.local_rank
                )
        
        # set up action oriented RL objective
        if dyn_cfg.invdyn_mlp:
            self.dynamics_agent = InverseDynamics(
                self.visual_encoder, # visual encoder
                dyn_cfg.action_dim,
                dyn_cfg.proj_dim,
                dyn_cfg.hidden_dim,
                dyn_cfg.num_steps,
                dyn_cfg.mini_batch_size,
                dyn_cfg.lr, # by default use same learning rate as policy learning
                dyn_cfg.gradient_updates,
                self.device
            )
            self.dynamics_agent = DistributedDataParallel(
                self.dynamics_agent, device_ids=[self.local_rank], output_device=self.local_rank
                )
        
        rollouts = RandomRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        # env reset obs
        rollouts.observations.copy_(batch['rgb'])
        batch = None
        observations = None

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        with (TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) if self.world_rank == 0 else contextlib.suppress()) as writer:

            for update in range(self.config.NUM_UPDATES):

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

                # collect rollout frames
                count_steps_delta = 0
                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(rollouts)
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

                if unsup_cfg.cpc:
                    for _ in range(unsup_cfg.gradient_updates):
                        cpc_loss = self.unsup_agent.module.update_cpc(rollouts.observations)
                    cpc_loss = comm.reduce_dict(cpc_loss)

                    if self.world_rank == 0:
                        writer.add_scalar("cpc_loss", cpc_loss["cpc_loss"], count_steps)

                elif unsup_cfg.simclr:
                    for _ in range(unsup_cfg.gradient_updates):
                        simclr_loss = self.unsup_agent.module.update_simclr(rollouts.observations)
                    simclr_loss = comm.reduce_dict(simclr_loss)

                    if self.world_rank == 0:
                        writer.add_scalar("simclr_loss", simclr_loss["simclr_loss"], count_steps)
    
                if dyn_cfg.invdyn_mlp:
                    inv_dyn_loss = self.dynamics_agent.module.update_invdyn(rollouts.observations, rollouts.actions)
                    # average across multiple gpus
                    inv_dyn_loss = comm.reduce_dict(inv_dyn_loss)

                    if self.world_rank == 0:
                        writer.add_scalar("inv_dyn_loss", inv_dyn_loss["inv_dyn_loss"], count_steps)
                        writer.add_scalar("inv_dyn_acc", inv_dyn_loss["pred_acc"], count_steps)
                
                rollouts.clear_storage()
                
                stats = torch.tensor(
                    count_steps_delta, device=self.device
                )
                distrib.all_reduce(stats)
                count_steps += stats.item()

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0 and self.world_rank == 0:
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

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0 and self.world_rank == 0:
                    self.unsup_agent.module.save(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                    self.dynamics_agent.module.save(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                    self.visual_encoder.module.save(self.config.CHECKPOINT_FOLDER, count_checkpoints)
                    count_checkpoints += 1
        
        self.envs.close()