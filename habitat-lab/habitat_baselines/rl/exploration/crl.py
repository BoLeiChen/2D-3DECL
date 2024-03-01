import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
from PIL import Image
from torchvision import transforms
from habitat_baselines.rl.exploration import utils
from habitat_baselines.rl.models.resnet import DetectronResNet50
from einops import rearrange


class CRL(nn.Module):
    def __init__(
        self,
        # encoder,
        proj_dim,  # output low dim feature space to compute similarity
        hidden_dim,  # hidden dimension of projection head
        simclr_lr,  # learning rate to optimize
        temperature,  # temperature to compute SimCLR objective
        device,
        image_size=256,
    ):
        super(CRL, self).__init__()

        self.reward_rms = utils.RMS(device=device)

        self.encoder = DetectronResNet50(device_id=torch.cuda.current_device())
        self.proj_head = nn.Sequential(nn.Linear(
            # self.encoder.embed_dim,
            self.encoder.output_size,
            hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)).to(device)
        self.proj_head.apply(utils.weight_init)

        self.actions_proj_head = nn.Sequential(nn.Linear(3, 8),
                                               nn.ReLU(),
                                               nn.Linear(8, 16)).to(device)
        self.actions_proj_head.apply(utils.weight_init)

        self.temperature = temperature

        # optimizers
        self.crl_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj_head.parameters())
            + list(self.actions_proj_head.parameters()), lr=simclr_lr)

        self.train()

    def save(self, folder_path, step):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "projection": self.proj_head.state_dict(),
            "projection_a": self.actions_proj_head.state_dict()
        }
        torch.save(
            checkpoint,
            os.path.join(folder_path, "crl_param_{}.pth".format(int(step)))
        )

    def forward_simclr(self, obs, actions):
        obs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3],
                       obs.shape[4])
        feature_repr = self.encoder(obs)
        prediction = self.proj_head(feature_repr)

        actions_ = self.actions_proj_head(actions.squeeze(1))

        return prediction, actions_

    # need to compute in the mode with "torch.no_grad()"
    def compute_simclr_reward(self, obs, actions):
        prediction_v, prediction_a = self.forward_simclr(obs,
                                                         actions)  # bs x proj_dim

        neg_dot = (prediction_v[:, None] * prediction_v[None, :]).sum(dim=-1)
        neg_dot = torch.sum(neg_dot, dim=1)
        reward = - neg_dot.view(obs.shape[0], obs.shape[1], 1)
        _, intr_reward_var = self.reward_rms(reward)
        reward = reward / (torch.sqrt(intr_reward_var) + 1e-8)

        neg_dot_a = (prediction_a[:, None] * prediction_a[None, :]).sum(dim=-1)
        neg_dot_a = torch.sum(neg_dot_a, dim=1)
        reward_a = - neg_dot_a.view(actions.shape[0], actions.shape[1], 1)
        _, intr_reward_var_a = self.reward_rms(reward_a)
        reward_a = reward_a / (torch.sqrt(intr_reward_var_a) + 1e-8)

        reward = reward + reward_a

        return reward  # should be in same shape as obs

    def update(self, obs, actions):
        prediction_v, prediction_a = self.forward_simclr(obs,
                                                         actions)  # bs x proj_dim

        neg_dot = (prediction_v[:, None] * prediction_v[None, :]).sum(dim=-1)
        neg_dot = torch.sum(neg_dot, dim=1)
        loss_v = neg_dot.view(obs.shape[0], obs.shape[1], 1)

        neg_dot_a = (prediction_a[:, None] * prediction_a[None, :]).sum(dim=-1)
        neg_dot_a = torch.sum(neg_dot_a, dim=1)
        loss_a = neg_dot_a.view(actions.shape[0], actions.shape[1], 1)

        loss = loss_v + loss_a
        loss = loss.mean()

        self.crl_opt.zero_grad()
        loss.backward()
        self.crl_opt.step()

        return {"crl_loss": loss}

    # check data aug works correctly
    def save_image(self, obs, view_1, view_2):
        os.makedirs("./tmp", exist_ok=True)

        for step in range(10):
            curr_obs = obs[step].detach().cpu().numpy().astype(np.uint8)
            img_file = os.path.join("./tmp", "obs_%02d.png" % (step))
            color_image = Image.fromarray(curr_obs, mode="RGB")
            color_image.save(img_file)

            curr_obs = view_1[step].detach().cpu().numpy().astype(np.uint8)
            img_file = os.path.join("./tmp", "view1_%02d.png" % (step))
            color_image = Image.fromarray(curr_obs, mode="RGB")
            color_image.save(img_file)

            curr_obs = view_2[step].detach().cpu().numpy().astype(np.uint8)
            img_file = os.path.join("./tmp", "view2_%02d.png" % (step))
            color_image = Image.fromarray(curr_obs, mode="RGB")
            color_image.save(img_file)
