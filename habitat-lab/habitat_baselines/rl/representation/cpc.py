import torch
import torch.nn as nn
import numpy as np
import random
import copy
import os

from habitat_baselines.rl.exploration.utils import weight_init


class CPC(nn.Module):
    def __init__(
        self, 
        encoder, # visual encoder from resnet policy
        proj_dim, # projection dimension for contrastive loss
        hidden_dim, # hidden dimension of projection head
        num_steps, # number of temporal steps to select positive samples
        cpc_lr,
        device
        ):
        super(CPC, self).__init__()

        # useful parameters
        self.encoder = encoder # assume already on cuda
        self.repr_dim = encoder.output_size # should be 2048 from resnet50
        self.proj_dim = proj_dim 
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.device = device 

        # useful architecture modules
        self.compressor = nn.Sequential(nn.Linear(self.repr_dim, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.proj_dim))
        self.predictor = nn.Sequential(nn.Linear(self.proj_dim, self.hidden_dim), 
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_dim, self.proj_dim))
        self.compressor.apply(weight_init)
        self.predictor.apply(weight_init)

        # prediction matrix onto future
        self.W = nn.Linear(proj_dim, proj_dim, bias=False)
        self.W.apply(weight_init)

        # move to cuda
        self.to(self.device)

        self.cpc_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.compressor.parameters()) + list(self.predictor.parameters()) + list(self.W.parameters()), 
                                        lr=cpc_lr, eps=1e-5)
        
        self.train()
    
    def to(self, device):
        self.encoder.to(device)
        self.compressor.to(device)
        self.predictor.to(device)
        self.W.to(device)
    
    def save(self, folder_path, step):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "compressor": self.compressor.state_dict(),
            "predictor": self.predictor.state_dict(),
            "w": self.W.state_dict()
        }
        torch.save(
            checkpoint, os.path.join(folder_path, "cpc_param_{}.pth".format(int(step)))
        )
    
    def load(self, folder_path, step):
        checkpoint = torch.load(os.path.join(folder_path, "cpc_param_{}.pth".format(int(step))), map_location="cpu")
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.compressor.load_state_dict(checkpoint["compressor"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.W.load_state_dict(checkpoint["w"])

    def forward_cpc(self, anchor_obs, other_obs):
        # process anchor
        anchor_repr = self.encoder(anchor_obs)
        anchor_proj = self.compressor(anchor_repr)
        # anchor_proj = F.normalize(anchor_proj, p=2, dim=-1) # L2 norm of latent representation
        anchor_proj = anchor_proj + self.predictor(anchor_proj)

        # process other examples
        other_repr = self.encoder(other_obs)
        other_proj = self.compressor(other_repr)
        # other_proj = F.normalize(other_proj, p=2, dim=-1) # L2 norm of latent representation

        return anchor_proj, other_proj
    

    def loss_cpc(self, anchor, other):
        # anchor shape: num_env x proj_dim
        # other shape: num_env x proj_dim
        # bs = num_env
        anchor_pred = self.W(anchor) # shape: num_env x proj_dim
        # Initialize log-Softmax to compute loss
        log_softmax = nn.LogSoftmax(dim=1)

        attention = torch.mm(anchor_pred, other.T)
        # print("attention value ", log_softmax(attention))
        # num_env x num_env
        # attention[i][j] = Pred_i * Other_j
        attention = attention - torch.max(attention, dim=1, keepdim=True)[0] # normalize
        # print(log_softmax(attention))
        info_nce = torch.mean(torch.diag(log_softmax(attention)))
        # print(info_nce)

        return info_nce


    def update_cpc(self, observations):
        # observation shape: num_step, num_envs, 3 x 256 x 256
        # need to remove index 0 from rollout storage, which is previous last obs
        num_updates = int(observations.shape[0] // self.num_steps)
        nce_loss = 0.0

        for iter in range(num_updates):
            rand_idx = np.random.choice(np.arange(self.num_steps), size=2, replace=False)
            rand_idx = np.sort(rand_idx) # sort array in ascending order
            # similarity within num_steps
            anchor_idx, close_idx = iter * self.num_steps + rand_idx[0], iter * self.num_steps + rand_idx[1]

            anchor_obs, other_obs = observations.select(dim=0, index=anchor_idx), observations.select(dim=0, index=close_idx)
            assert anchor_obs.shape == other_obs.shape
            anchor_proj, other_proj = self.forward_cpc(anchor_obs, other_obs)
            nce_loss += self.loss_cpc(anchor_proj, other_proj)
        
        # take mean as loss
        nce_loss /= -1 * num_updates

        # update unsup parameters
        self.cpc_opt.zero_grad()
        nce_loss.backward()
        self.cpc_opt.step()

        return {"cpc_loss": nce_loss}
    