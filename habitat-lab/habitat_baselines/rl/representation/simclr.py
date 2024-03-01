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


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

def map_dataaug(args):
    # image shape: H x W x C
    image, transform = args 
    if transform is not None:
        image = image.permute(2, 0, 1)
        image = transform(image)
        image *= 255.0
        image = image.permute(1, 2, 0).numpy()
    return image


class SimCLR(nn.Module):
    def __init__(
        self,
        encoder, # visual encoder from resnet policy
        proj_dim, # output low dim feature space to compute similarity
        hidden_dim, # hidden dimension of projection head
        simclr_lr, # learning rate to optimize 
        batch_size, # batch size to optimize
        temperature, # temperature to compute SimCLR objective
        device,
        image_size = 256
    ):
        super(SimCLR, self).__init__()
        self.device = device

        # data augmentation
        # INPUT: PIL image
        # OUTPUT: normalized [0, 1] in RGB order
        self.data_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            get_color_distortion(1.2), 
            transforms.ToTensor()
            ])
        
        self.encoder = encoder
        self.encoder.to(device)
        # encoder is trainable
        for param in self.encoder.parameters():
            assert param.requires_grad == True
        self.proj_head = nn.Sequential(nn.Linear(encoder.output_size, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, proj_dim)).to(device)
        self.proj_head.apply(utils.weight_init)

        self.temperature = temperature
        self.batch_size = batch_size
        self.simclr_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.proj_head.parameters()), lr=simclr_lr)

        self.train()
    
    def forward_simclr(self, obs):
        shape_0, shape_1 = obs.shape[0], obs.shape[1]
        obs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])

        # data augmentation
        obs_view_1 = []
        obs_view_2 = []
        args = [(obs[i], self.data_aug) for i in range(obs.shape[0])]
        for elem in args:
            image_1 = map_dataaug(elem)
            image_2 = map_dataaug(elem)
            obs_view_1.append(image_1)
            obs_view_2.append(image_2)
        obs_view_1 = torch.Tensor(np.array(obs_view_1, dtype=float)).cuda()
        obs_view_2 = torch.Tensor(np.array(obs_view_2, dtype=float)).cuda()

        repr_1, repr_2 = self.encoder(obs_view_1), self.encoder(obs_view_2)
        repr_1, repr_2 = self.proj_head(repr_1), self.proj_head(repr_2)
        repr_1, repr_2 = F.normalize(repr_1, p=2, dim=-1), F.normalize(repr_2, p=2, dim=-1)

        return repr_1, repr_2
    
    def update_simclr(self, obs):
        if self.batch_size < obs.shape[0]:
            mini_batch_index = torch.LongTensor(np.random.choice(obs.shape[0], self.batch_size, replace=False)).to(self.device)
            obs = torch.index_select(obs, 0, mini_batch_index)
        repr_1, repr_2 = self.forward_simclr(obs) # bs x proj_dim

        pos_dot = (repr_1 * repr_2 / self.temperature).sum(dim=-1)
        neg_dot = (repr_1[:, None] * repr_2[None, :] / self.temperature).sum(dim=-1)
        denom_loss = torch.logsumexp(neg_dot, dim=1)
        simclr_loss = (- pos_dot + denom_loss).view(obs.shape[0], obs.shape[1], 1) 
        simclr_loss = simclr_loss.mean()

        self.simclr_opt.zero_grad()
        simclr_loss.backward()
        self.simclr_opt.step()

        return {"simclr_loss": simclr_loss}
    
    def save(self, folder_path, step):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "projection_head": self.proj_head.state_dict(),
        }
        torch.save(
            checkpoint, os.path.join(folder_path, "simclr_param_{}.pth".format(int(step)))
        )