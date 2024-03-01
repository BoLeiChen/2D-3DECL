import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.exploration import utils
from habitat_baselines.rl.models.resnet import DetectronResNet50


class RND(nn.Module):
    def __init__(self, hidden_dim, rnd_repr_dim, learning_rate, device):
        super(RND, self).__init__()
        self.reward_rms = utils.RMS(device=device)

        # feature encoder
        self.encoder = DetectronResNet50(device_id = torch.cuda.current_device())
            
        self.encoder.to(device)
        # self.target_encoder = utils.make_target(self.encoder).to(device)

        # predictor network
        self.predictor = nn.Sequential(nn.Linear(self.encoder.output_size, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, rnd_repr_dim)).to(device)
        
        # target network
        self.target = nn.Sequential(nn.Linear(self.encoder.output_size, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, rnd_repr_dim)).to(device)
        # target network is freezed, not trainable
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.apply(utils.weight_init)

        # optimizers
        self.rnd_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=learning_rate)

        self.train()
    
    def save(self, folder_path, step):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            # "target_encoder": self.target_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict()
        }
        torch.save(
            checkpoint, os.path.join(folder_path, "rnd_param_{}.pth".format(int(step)))
        )
    
    def load(self, folder_path, step):
        checkpoint = torch.load(os.path.join(folder_path, "rnd_param_{}.pth".format(int(step))), map_location="cpu")
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.target.load_state_dict(checkpoint["target"])
    

    def forward_rnd(self, obs):
        # check obs shape and .view(shape[0] * shape[1], ...) if needed
        shape_0, shape_1 = obs.shape[0], obs.shape[1]
        obs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        # feature_repr, target_feature_repr = self.encoder(obs), self.target_encoder(obs) # bs x 2048
        feature_repr = self.encoder(obs)
        prediction, target = self.predictor(feature_rsepr), self.target(feature_repr) # bs x 64
        prediction_error = torch.square(target.detach() - prediction).mean(dim=-1, keepdim=True) # bs x 1
        return prediction_error

    # need to compute in the mode with "torch.no_grad()"
    def compute_rnd_reward(self, obs):
        prediction_error = self.forward_rnd(obs) # bs x 1
        prediction_error = prediction_error.view(obs.shape[0], obs.shape[1], 1)
        _, intr_reward_var = self.reward_rms(prediction_error)
        reward = prediction_error / (torch.sqrt(intr_reward_var) + 1e-8)
        return reward # should be in same shape as obs
    
    def update(self, obs):
        prediction_error = self.forward_rnd(obs) # bs x 1
        prediction_error = prediction_error.view(obs.shape[0], obs.shape[1], 1)
        loss = prediction_error.mean()
        
        # optimize and update predictor network
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

        return {"rnd_loss": loss}
