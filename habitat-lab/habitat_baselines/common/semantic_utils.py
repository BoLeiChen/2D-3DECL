import os
import cv2
import pickle
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools
import torch
import quaternion

import habitat_sim

import habitat
from ..config.default import get_config_simple
from ..rl.ppo import PPO
from habitat.utils.visualizations.utils import observations_to_image_custom
from ..common.utils import generate_video_custom
from habitat_baselines.common.sim_settings import default_sim_settings, make_cfg
from habitat_baselines.common.environments import seed_all_rng
from habitat_baselines.common.constants import scenes, master_scene_dir, coco_categories, coco_categories_mapping, action_mapping, action_decode
from habitat_baselines.rl.ppo.explore_policy import ExploreNet, ExploreBaselinePolicy, ExplorePolicy, ExploreBaselinePolicyRollout

import detectron2
from detectron2.structures import BoxMode


def make_custom_cfg(scene, gpu_id):
    default_sim_settings['scene'] = master_scene_dir + scene + '.glb'
    default_sim_settings['width'] = 640
    default_sim_settings['height'] = 480
    default_sim_settings['sensor_height'] = 0.88
    default_sim_settings['semantic_sensor'] = True
    default_sim_settings['depth_sensor'] = False
    default_sim_settings["seed"] = seed_all_rng()

    cfg = make_cfg(default_sim_settings, gpu_id)
    return cfg


def initialize_scene_agent(cfg):
    sim = habitat_sim.Simulator(cfg)
    random.seed(default_sim_settings["seed"])
    sim.seed(default_sim_settings["seed"])

    # initialize the agent at a random start state
    agent = sim.initialize_agent(default_sim_settings["default_agent"])
    start_state = agent.get_state()

    start_state.position = sim.pathfinder.get_random_navigable_point() # location has dimension 3
    random_rotation = np.random.rand(4) # rotation has dimension 4
    random_rotation[1] = 0.0
    random_rotation[3] = 0.0
    start_state.rotation = quaternion.as_quat_array(random_rotation)

    agent.set_state(start_state)
    return sim, agent


def print_scene_objects(sim):
    scene = sim.semantic_scene

    # key: category_id as specified in coco_categories
    #      "15" is for background
    # values: list of instance_id falling into this category_id
    category_instance_lists = {}

    # key: instance_id
    # value: category_id as specified in coco_categories
    #        "6" is for background
    instance_category_lists = {}

    for obj in scene.objects:
        if obj is None or obj.category is None:
            continue
        obj_class = obj.category.name()
        if obj_class in coco_categories.keys():
            cat_id = coco_categories[obj_class]
            obj_id = int(obj.id.split("_")[-1])
            if cat_id not in category_instance_lists:
                category_instance_lists[cat_id] = [obj_id]
            else:
                category_instance_lists[cat_id].append(obj_id)
            if obj_id not in instance_category_lists:
                instance_category_lists[obj_id] = cat_id

    # print(category_instance_lists)
    # print(instance_category_lists)
    # try to compute bounding box

    # input("Press Enter to continue...")
    return category_instance_lists, instance_category_lists


def save_color_observation(color_obs, save_dir, scene, total_frames):
    color_img = Image.fromarray(color_obs, mode="RGBA")
    color_img.save(save_dir + "/%s/rgba/%05d.png" % (scene, total_frames))


def area_filter(mask, bounding_box, img_height, img_width, size_tol=0.05):
    """
    Function to filter out masks that contain sparse instances
    for example:
        0 0 0 0 0 0
        1 0 0 0 0 0
        1 0 0 0 1 0    This is a sparse mask
        0 0 0 0 1 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        1 1 1 1 1 0
        1 1 1 1 1 1    This is not a sparse mask
        0 0 0 1 1 1
        0 0 0 0 0 0
    """
    xmin, ymin, xmax, ymax = bounding_box
    num_positive_pixels = np.sum(mask[ymin:ymax, xmin:xmax])
    num_total_pixels = (xmax - xmin) * (ymax - ymin)
    big_enough = (xmax - xmin) >= size_tol * img_width and (
        ymax - ymin
    ) >= size_tol * img_height
    if big_enough:
        not_sparse = num_positive_pixels / num_total_pixels >= 0.3
    else:
        not_sparse = False
    return not_sparse and big_enough


def save_semantic_observation(
    color_obs, semantic_obs, action, agent_state,
    instance_to_category, category_to_instance,
    total_frames, save_dir, scene, rollout_prob
):
    record = {}

    img_dir = save_dir + "/%s/rgba/%05d.png" % (scene, total_frames)

    color_density = 1 - np.count_nonzero(color_obs) / np.prod(color_obs.shape)
    semantic_density = np.count_nonzero(semantic_obs) / np.prod(semantic_obs.shape) # shape: height x width

    # process one hot encode version
    max_inst_id = max(list(instance_to_category.keys()))

    seg_obs_one_hot = np.arange(max_inst_id + 1) # shape: (max(instance_id) + 1) x height x width
    seg_obs_one_hot = (seg_obs_one_hot[:, np.newaxis, np.newaxis] == semantic_obs).astype(int)

    if color_density < 0.05 and semantic_density > 0.05 ** 2:

        idx = "%s_%05d" % (scene, total_frames)
        height, width = default_sim_settings['height'], default_sim_settings['width']

        record["file_name"] = img_dir
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        for instance_id in list(instance_to_category.keys()):

            # query one hot vector as object mask
            obj_mask = seg_obs_one_hot[instance_id]
            # get object mask
            # select bounding box numbers
            nonzero_index = np.argwhere(obj_mask == 1)

            # nontrivial object mask
            if nonzero_index.shape[0] > 0:

                # query category of object
                category = instance_to_category[instance_id]

                py_min, px_min = tuple(np.amin(nonzero_index, axis=0))
                py_max, px_max = tuple(np.amax(nonzero_index, axis=0))

                obj = {
                        "bbox": [px_min, py_min, px_max, py_max],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": pycocotools.mask.encode(np.asarray(obj_mask, order="F").astype('uint8')),
                        "category_id": category,
                        }

                filter_obj = area_filter(obj_mask, obj['bbox'], height, width)
                if filter_obj:
                    objs.append(obj)

        record["annotations"] = objs

        # keep track of action information
        record['action'] = np.zeros(len(action_mapping.keys()))
        record['action'][action_mapping[action]] = 1
        # keep track of agent information
        record['agent_pos'] = agent_state.position
        record['agent_rot'] = agent_state.rotation

        if np.random.uniform() < rollout_prob:
            with open(save_dir + "/%s/dict/%05d.pkl" % (scene, total_frames), 'wb') as f:
                pickle.dump(record, f)
                f.close()
            return True , record

        else:
            return False, record

    else:
        return False, record

def save_observation(observations, action, agent_state, total_frames,
                        instance_to_category, category_to_instance,
                        save_dir, scene, rollout_prob):
    os.makedirs(os.path.join(save_dir, scene, "rgba"), exist_ok=True) # image
    os.makedirs(os.path.join(save_dir, scene, "dict"), exist_ok=True) # label

    semantic_obs = observations["semantic_sensor"]
    color_obs = observations["color_sensor"]

    valid_sample, record = save_semantic_observation(color_obs, semantic_obs, action, agent_state,
                                                        instance_to_category, category_to_instance,
                                                        total_frames, save_dir, scene, rollout_prob)
    if valid_sample:
        save_color_observation(color_obs, save_dir, scene, total_frames)
        # print(total_frames)
    return valid_sample, record


def run_env(
            sim, agent, custom_cfg, actor_critic,
            device, config, save_dir, scene,
            rollout_length, rollout_prob, start_index
    ):
    if actor_critic is not None:
        # config information for PPO agent and REWARD agent
        ppo_cfg = config.RL.PPO

    # get instance_id to category_id mapping
    # get category_id to instance_id mapping
    category_instance_lists, instance_category_lists = print_scene_objects(sim)

    # keep track of total env steps taken
    total_frames = 0
    # keep track od valid frames useful for training
    valid_frames = start_index

    if actor_critic is not None:

        test_recurrent_hidden_states = torch.zeros(
            actor_critic.policy_net.num_recurrent_layers,
            config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES,
            1, device=device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            config.NUM_PROCESSES,
            1, device=device
        )

    action_names = list(
        custom_cfg.agents[
            default_sim_settings["default_agent"]
        ].action_space.keys()
    )[:3]

    if actor_critic is not None:
        actor_critic.eval()

    # reset enviornment
    prev_observations = None

    while valid_frames < start_index + rollout_length:

        # rollout policy
        if prev_observations is not None and actor_critic is not None:

            # convert RGBA to RGB
            rgba_img = Image.fromarray(prev_observations["color_sensor"], mode="RGBA")
            rgb_img = rgba_img.convert("RGB")
            rgb_array = np.array(rgb_img)
            obs_tensor = torch.from_numpy(rgb_array).float().unsqueeze(0).to(device)
            obs_points = None      # no use
            # forward pass in eval mode
            with torch.no_grad():
                _, action, action_log_probs, test_recurrent_hidden_states = actor_critic.act(obs_tensor, obs_points,
                                                                                    test_recurrent_hidden_states,
                                                                                    prev_actions,
                                                                                    not_done_masks)
            # decode action
            action = action_decode[action.item()]
        else:
            # uniform random sample
            action = random.choice(action_names)
        # print("action", action)

        next_observations = sim.step(action) # contains rgb and semantic, but we only record video for rgb

        agent_state = agent.get_state()
        # print("action", action, "agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        next_color_obs = next_observations["color_sensor"]
        next_semantic_obs = next_observations["semantic_sensor"]

        if prev_observations is not None:
            valid_sample, record = save_observation(next_observations, action, agent_state, valid_frames,
                                                    instance_category_lists, category_instance_lists,
                                                    save_dir, scene, rollout_prob)
        else:
            # account for first frame after initialization without action
            valid_sample = False

        if valid_sample:
            valid_frames += 1

        prev_observations = next_observations

        total_frames += 1

        # epsiode end, record whole trajectory
        assert config.ENV_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS == 512, "max episode length not correct!"
        if total_frames % config.ENV_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS == 0:
            # initialize the agent at a random start state
            agent_state = agent.get_state()
            agent_state.position = sim.pathfinder.get_random_navigable_point() # location has dimension 3
            random_rotation = np.random.rand(4) # rotation has dimension 4
            random_rotation[1] = 0.0
            random_rotation[3] = 0.0
            agent_state.rotation = quaternion.as_quat_array(random_rotation)
            agent.set_state(agent_state)

            test_recurrent_hidden_states.fill_(0.0)
            prev_actions.fill_(0.0)
            not_done_masks.fill_(0.0)

            prev_observations = None
            total_frames = 0

    sim.close()
    del sim


def setup_env(scene, gpu_id):
    custom_cfg = make_custom_cfg(scene, gpu_id)
    sim, agent = initialize_scene_agent(custom_cfg)
    # print("agent_state: position", agent.get_state().position, "rotation", agent.get_state().rotation)
    return sim, agent, custom_cfg


def main(config, param, save_dir, rollout_length, rollout_prob, start_index, gpu_id, num_gpu):

    # ckpt_dict = torch.load(model_dir, map_location='cpu')
    # config = ckpt_dict["config"]

    device = torch.cuda.current_device()

    config.defrost()
    config.NUM_PROCESSES = 1
    config.freeze()

    ppo_cfg = config.RL.PPO
    ddppo_cfg = config.RL.DDPPO

    # set up PPO agent and Actor-Critic model
    actor_critic = ExploreBaselinePolicyRollout(
            action_dim=3,
            hidden_size=ppo_cfg.hidden_size,
            num_recurrent_layers=ddppo_cfg.num_recurrent_layers,
            rnn_type=ddppo_cfg.rnn_type,
            device=None
        )
    actor_critic.to(device)
    policy_agent = PPO(
        actor_critic=actor_critic,
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
    # policy_agent.load_state_dict(ckpt_dict['state_dict'], strict=False)
    policy_agent.load_state_dict(param)
    actor_critic = policy_agent.actor_critic

    # rollout policy
    scene_length = int(len(scenes["train"]) // num_gpu)
    machine_index = gpu_id * scene_length
    # handle non-5 gpus training
    if len(scenes["train"]) % num_gpu != 0 and gpu_id == num_gpu - 1:
        scene_length += 1

    for scene in scenes["train"][machine_index : machine_index + scene_length]:
        sim, agent, custom_cfg = setup_env(scene, gpu_id)

        run_env(
            sim, agent, custom_cfg, actor_critic,
            device, config, save_dir, scene,
            rollout_length, rollout_prob, start_index
            )

    del actor_critic


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process random data collection arguments.')

    # related to simulator and enviornmnet
    parser.add_argument('--scene', type=str, default="", help='to only collect one scene with specified name')
    parser.add_argument('--seed', type=int, default=1, help='random seed to set up simulator enviornment')

    # related to policy checkpoint load
    parser.add_argument('--policy', type=str, default="random", help='policy rollouts in enviornment to collect data')
    parser.add_argument('--model-dir', type=str, help='checkpoint directory to load policy model')
    parser.add_argument('--config', type=str, default='none', help='config file to initialize RL env and policy architecture') # will use config saved during training by default

    # related to policy rollouts save and visualize
    parser.add_argument('--date', type=str, help='date of RL policy training experiments, for correspondence')
    parser.add_argument('--save-dir', type=str, help='directory to save frames of policy rollouts')
    parser.add_argument('--video-name', type=str, default="none", help='whether to save episode trajectory in video format and provide name for logging, none if no need to save video by default')
    parser.add_argument('--reset-episode', action='store_true', help='whether to reset recurrent states every new epsiode length')

    args = parser.parse_args()

    print("Command Line Args:", args)
    main(args)
