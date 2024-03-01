import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.utils import comm
from habitat import logger
from habitat_baselines.common.utils import Flatten

scratch_cfg_file = '/home/cbl/alp/configs/mask_rcnn/scratch_mask_rcnn_R_50_FPN_3x_syncbn.yaml'


class SelectStage(nn.Module):
    """Selects features from a given stage."""

    def __init__(self, stage: str = 'res5'):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]


class DetectronResNet50(nn.Module):
    def __init__(self, downsample=True, distributed=True, device_id=0):
        super(DetectronResNet50, self).__init__()

        self.resnet_layer_size = 2048
        self.downsample = downsample

        cfg = get_cfg()
        cfg.merge_from_file(scratch_cfg_file)

        # use for feature encoder of policy training
        # random initialized weights and training mode
        cfg.MODEL.DEVICE = "cuda:{}".format(device_id)

        mask_rcnn = build_model(cfg)

        self.cnn = nn.Sequential(mask_rcnn.backbone.bottom_up, # resnet50 from mask_rcnn for multiple gpus
                                    SelectStage('res5'),
                                    # torch.nn.AdaptiveAvgPool2d(1), # res5 has shape bsz x 2048 x 8 x 8
                                    )
        self.pooling_layer = torch.nn.AdaptiveAvgPool2d(1)
        # input order and normalization
        self.input_order = cfg.INPUT.FORMAT
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

        # model is trainable
        for param in self.cnn.parameters():
            assert param.requires_grad
        
        """if comm.is_main_process() and pretrained:
            # sanity check: print model architecture
            print("Load resnet weight from {}".format(cfg.MODEL.WEIGHTS))
            print("Input channel order: {}".format(self.input_order))
            print("Normalize mean: {}".format(self.pixel_mean))
            print("Normalize std: {}".format(self.pixel_std))"""

    
    def forward(self, observations, pooling=True):
        device = observations.device
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        # observations from rollout buffer already query "rgb" key value item
        rgb_observations = observations.permute(0, 3, 1, 2).contiguous()

        # downsample for faster compute
        if self.downsample:
            rgb_observations = F.avg_pool2d(rgb_observations, 2)

        if self.input_order == "BGR":
            # flip into BGR order
            rgb_observations = torch.flip(rgb_observations, dims=(1,)).contiguous()
        else:
            assert self.input_order == "RGB"
        
        # normalize
        rgb_observations = (rgb_observations - self.pixel_mean) / self.pixel_std 

        # resnet forward -> last layer repr
        resnet_output = self.cnn(rgb_observations)
        if pooling:
            resnet_output = self.pooling_layer(resnet_output)
            # flatten dimension
            resnet_output = resnet_output.view(resnet_output.shape[0], resnet_output.shape[1])
        return resnet_output
    
    def save(self, folder_path, step, prefix="none"):
        if prefix == "none":
            torch.save(self.cnn[0].state_dict(), os.path.join(folder_path, "resnet_{}.pth".format(int(step))))
        else:
            torch.save(self.cnn[0].state_dict(), os.path.join(folder_path, "{}_resnet_{}.pth".format(prefix, int(step))))
        
    
    @property
    def output_size(self):
        return self.resnet_layer_size


if __name__ == "__main__":
    rand_input = torch.zeros(20, 256, 256, 3).float()
    model = DetectronResNet50()
    output = model(rand_input)
    print(output.shape)