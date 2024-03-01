import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
from habitat_baselines.rl.representation.utils import spatial_sampling
from habitat_baselines.rl.exploration.utils import weight_init
from habitat_baselines.rl.models.resnet import DetectronResNet50
from habitat_baselines.rl.ppo.utils.pointnet import PointNetEncoder
from einops import rearrange
from PIL import Image
#torch.autograd.set_detect_anomaly(True)
class InverseDynamics(nn.Module):
    def __init__(
        self,
        encoder, # visual encoder from resnet policy
        action_dim,
        proj_dim,
        hidden_dim,
        num_steps,
        batch_size,
        idm_lr,
        num_update,
        device,
        num_sem_categories,
        temperature
    ):
        super(InverseDynamics, self).__init__()

        self.encoder = encoder # assume already on cuda
        self.temperature = temperature
        self.point_encoder = PointNetEncoder(global_feat=True, channel = num_sem_categories + 5).to(device)
        self.image_feat_proj = nn.Sequential(
            nn.Linear(5 * 512, 2048),          #2560 --> 4608
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(device)
        self.image_feat_proj.apply(weight_init)
        self.repr_dim =self. encoder.output_size    #encoder.embed_dim      #encoder.output_size # should be 2048 from resnet50
        self.action_dim = action_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        assert self.proj_dim <= self.repr_dim, "can only project to lower dimension space"

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device

        # projection head: repr_dim -> proj_dim
        if self.proj_dim < self.repr_dim:
            self.proj_head = nn.Sequential(
                nn.Linear(self.repr_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.proj_dim)
            ).to(self.device)
            self.proj_head.apply(weight_init)

        self.act_pred = nn.Sequential(
            nn.Linear(self.proj_dim * (self.num_steps + 1), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * self.num_steps)
        ).to(self.device)
        self.act_pred.apply(weight_init)

        self.ce_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.num_update = num_update

        if self.proj_dim < self.repr_dim:
            self.invdyn_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.proj_head.parameters()) +
                                               list(self.act_pred.parameters()), lr=idm_lr, eps=1e-5)
            self.crl_opt = torch.optim.Adam(list(self.point_encoder.parameters()) +
                                               list(self.image_feat_proj.parameters()), lr=idm_lr, eps=1e-5)
        else:
            self.invdyn_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.act_pred.parameters()), lr=idm_lr, eps=1e-5)


        self.train()

    def save(self, folder_path, step):
        if self.proj_dim < self.repr_dim:
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "proj_head": self.proj_head.state_dict(),
                "act_pred": self.act_pred.state_dict(),
                "pcl_encoder": self.point_encoder.state_dict(),
                "image_feat_proj": self.image_feat_proj.state_dict(),
            }
        else:
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "act_pred": self.act_pred.state_dict(),
                "pcl_encoder": self.point_encoder.state_dict(),
                "image_feat_proj": self.image_feat_proj.state_dict(),
            }
        torch.save(
            checkpoint, os.path.join(folder_path, "idm_param_{}.pth".format(int(step)))
        )

    def save_encoder(self, folder_path, step):
        torch.save(
            self.encoder.state_dict(), os.path.join(folder_path, "resnet_{}.pth".format(int(step)))
        )
    def save_pcl_encoder(self, folder_path, step):
        torch.save(
            self.point_encoder.state_dict(), os.path.join(folder_path, "PNet_{}.pth".format(int(step)))
        )

    def forward(self, img, obs_points_batch, action_target):
        repr_feat = self.encoder(img)
        #print(obs_points_batch.shape)
        img_b = img.shape[0]
        pt_b = obs_points_batch.shape[0]
        env_num = int(img_b / pt_b)
        #print(img_b, pt_b, env_num)
        obs_points_match_all = []
        for i in range(0, env_num):
            obs_points_match = []
            for j in range(4, pt_b, 2):     # 8
                obs_points_match.append(obs_points_batch[j, i, ...].unsqueeze(0))
            obs_points_match = torch.stack(obs_points_match)
            obs_points_match_all.append(obs_points_match)
        obs_points_match_all = torch.cat(obs_points_match_all, dim = 1)
        obs_points_match_all = rearrange(obs_points_match_all, 'b n c l -> (b n) c l')
        repr_1 = self.point_encoder(obs_points_match_all)
        #print(repr_1.size())

        repr_feat = self.proj_head(repr_feat).detach()
        repr_feat = repr_feat.view(self.batch_size, -1, repr_feat.shape[1]) # env_steps x num_envs x proj_dim
        # env_steps x num_envs x proj_dim x t ->  env_steps x num_envs x t x proj_dim
        repr_cat = repr_feat.unfold(dimension=0, size=(self.num_steps + 1), step=2).permute(0, 1, 3, 2).contiguous()
        repr_cat = repr_cat.view(repr_cat.shape[0], repr_cat.shape[1], -1) # concatenate (num_steps x proj_sim)

        repr_2 = self.image_feat_proj(repr_cat)
        repr_2 = rearrange(repr_2, 'b n l -> (b n) l')
        #print(repr_2.size())
        assert repr_cat.shape[0] == action_target.shape[0] and repr_cat.shape[1] == action_target.shape[2], "shape must match between input and target!"

        # prediction logits
        pred_logits = self.act_pred(repr_cat)
        pred_logits = pred_logits.view(pred_logits.shape[0], pred_logits.shape[1], self.num_steps, self.action_dim).permute(0, 3, 1, 2)
        pred_logits = F.normalize(pred_logits, p=2, dim=1)
        x_loss = self.ce_loss(pred_logits, action_target.argmax(dim=1)) # env_steps x action_dim x num_envs x t

        pos_dot = (repr_1 * repr_2 / self.temperature).sum(dim=-1)
        neg_dot= (repr_1[:, None] * repr_2[None, :] / self.temperature).sum(dim=-1)
        denom_loss = torch.logsumexp(neg_dot, dim=1)
        simclr_loss = (- pos_dot + denom_loss).view(action_target.shape[0], obs_points_batch.shape[1], 1)
        simclr_loss = simclr_loss.mean()
        #print(simclr_loss)

        # prediction accuracy
        with torch.no_grad():
            pred_probs = self.softmax(pred_logits)
            pred_actions = torch.argmax(pred_probs, dim=1) # 0-1 probabilities
            true_actions = torch.argmax(action_target, dim=1) # 0,1 entries
            assert pred_actions.shape == true_actions.shape, "prediction and true label shape must match!"
            pred_acc = (pred_actions == true_actions).sum() / true_actions.numel()
            # print("Acc: {}".format(pred_acc))

        return simclr_loss, x_loss, pred_acc

    def update_invdyn(self, obs_batch, obs_points_batch, action_batch):
        r"""
        Function to update inverse dynamics prediction model.

        Input:
        prev_obs_batch: batch observation of previous time step
        next_obs_batch: batch observation of next time step
        action_batch: batch of action taken - (s, a, s')

        Output:
        cross_entropy_loss
        prediction_accuracy
        """
        # env_steps x num_envs x action_dim -> env_steps x num_envs x action_dim x t -> env_steps x action_dim x num_envs x t
        action_mini_batch = action_batch.unfold(dimension=0, size=self.num_steps, step=2).permute(0, 2, 1, 3)
        # env_steps x num_envs x img_size
        obs_mini_batch = obs_batch.view(-1, obs_batch.shape[2], obs_batch.shape[3], obs_batch.shape[4])

        '''
        obs_mini_batch_0 = spatial_sampling(obs_mini_batch.permute(0, 3, 1, 2), spatial_idx = 0).permute(0, 2, 3, 1).unsqueeze(1)
        obs_mini_batch_1 = spatial_sampling(obs_mini_batch.permute(0, 3, 1, 2), spatial_idx = 1).permute(0, 2, 3, 1).unsqueeze(1)
        obs_mini_batch_2 = spatial_sampling(obs_mini_batch.permute(0, 3, 1, 2), spatial_idx = 2).permute(0, 2, 3, 1).unsqueeze(1)
        obs_mini_batch_ = torch.cat([obs_mini_batch_0, obs_mini_batch_1, obs_mini_batch_2], dim=1)   #(65,3,256,256,3)
        x1 = obs_mini_batch[20, ...]
        x_0 = obs_mini_batch_0[20, ...].permute(1, 2, 0)
        x_1 = obs_mini_batch_1[20, ...].permute(1, 2, 0)
        x_2 = obs_mini_batch_2[20, ...].permute(1, 2, 0)
        img1 = x1.cpu().numpy().astype(np.uint8)
        img_0 = x_0.cpu().numpy().astype(np.uint8)
        img_1 = x_1.cpu().numpy().astype(np.uint8)
        img_2 = x_2.cpu().numpy().astype(np.uint8)

        a = Image.fromarray(img1, mode='RGB')
        a.show()
        b = Image.fromarray(img_0, mode='RGB')
        b.show()
        c = Image.fromarray(img_1, mode='RGB')
        c.show()
        d = Image.fromarray(img_2, mode='RGB')
        d.show()
        '''
        for _ in range(self.num_update):
            simclr_loss, x_loss, pred_acc = self.forward(obs_mini_batch, obs_points_batch, action_mini_batch)

            simclr_loss_ = simclr_loss * 0.2
            x_loss_ = simclr_loss_ + x_loss
            self.invdyn_opt.zero_grad()
            x_loss_.backward(retain_graph=True)
            self.invdyn_opt.step()
            
            self.crl_opt.zero_grad()
            simclr_loss_.backward()
            self.crl_opt.step()

        return {"simclr_with_inv_dyn_loss": simclr_loss_, "inv_dyn_loss": x_loss, "pred_acc": pred_acc}
