# 2D-3DECL

## Installation

Create conda environment with python>=3.7 and cmake>=3.10:
```bash
conda create -n alp python=3.7 cmake=3.14.0
conda activate alp
```

We use pytorch v1.10.0 and cuda 11.3:
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

We use earlier version of [habitat-sim](https://github.com/facebookresearch/habitat-sim):
```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/v0.1.5; 
pip install -r requirements.txt 
python setup.py install --headless
```

Install [detectron2](https://github.com/facebookresearch/detectron2) from released package:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Preparing training dataset

Download the Gibson dataset using the instructions [here](https://github.com/facebookresearch/habitat-lab#scenes-datasets) (download the 11GB file `gibson_habitat_trainval.zip`)

Move the Gibson scene dataset or create a symlink at `data/scene_datasets/gibson_semantics`.

Download the tiny partition of the 3DSceneGraph dataset which contains the semantic information for the Gibson dataset.

The semantic data will need to be converted before it can be used within Habitat, follow instructions to use `gen_gibson_semantics.sh` script from the [habitat-sim](https://github.com/facebookresearch/habitat-sim#datasets).

Download the tiny partition (`gibson_tiny.tar.gz`) of the gibson dataset and extract it into the `data/scene_datasets/gibson_tiny` folder.

Run script to generate semantic annotations:
```sh
bash habitat-sim/datatool/tools/gen_gibson_semantics.sh data/scene_datasets/3DSceneGraphTiny/automated_graph data/scene_datasets/gibson_tiny data/scene_datasets/gibson_semantics
```

## Setting up training dataset

The code requires setting up following formats in the `data` folder:
```
2D-3DECL/
  data/
    scene_datasets/
      gibson_semantics/
        # for non-semantic scenes
        Airport.glb
        Airport.navmesh
        # for semantic scenes
        Allensville_semantic.ply
        Allensville.glb
        Allensville.ids
        Allensville.navmesh
        Allensville.scn
        ...

```

To test training datasets are downloaded in correct formats, run `python data/example_test.py` should print out set of objects in Allensville scene in Gibson dataset, mappings from category_id to instance_id, and mappings from instance_id to category_id. You could use scripts `python data/example_test.py --scene [SCENE_NAME]` to test in other Gibson scenes.

## Running the code

### Training visual representations and exploration policy

Please use following scripts:
```bash
bash scripts/pretrain/run_crl_alp.sh [CUDA_VISIBLE_DEVICES] [num_gpus_per_node]
```

For compute requirements, we use 5 GPUs each training 4 environment processes with 24GB memory. You could change `NUM_PROCESSES` in config.yaml files to training different numbers of environment processes in parallel.

To generate videos and maps of agent behaviors, please use following scripts:
```bash
CUDA_VISIBLE_DEVICES=[CUDA_VISIBLE_DEVICES] python src/pretrain/run_ddppo.py --exp-config configs/visualize/[AGENT].yaml --run-type eval
```
You may need to download PointNav-v1 dataset in Gibson scenes following [here](https://github.com/xinranliang/habitat-lab/tree/alp-pkg#data), and modify relative path to dataset directory and model checkpoint.

### Training and evaluating downstream perception models

Please use following scripts to train Mask-RCNN models:

```bash
CUDA_VISIBLE_DEVICES=[CUDA_VISIBLE_DEVICES] python src/mask_rcnn/plain_train_net.py \
--date [exp_date] --dataset /path/to/samples_dir/ \
--pretrain-weights {random, imagenet-sup, sim-pretrain} --pretrain-path /path/to/simulator_trained_repr/ \
--num-gpus [num_gpus] --max-iter [total_train_iters] --batch-size [batch_size]
```

More specific settings and default values are listed below:
- `--dataset`: relative path to folder directory that saves labeled samples from a small random subset of actively explored frames.
- `--pretrain-weights`: weight init of downstream perception model. `random` refers to random initialization and `imagenet-sup` refers to init from ImageNet supervised model trained for classification task. `sim-pretrain` initializes from visual representation pretrained in simulator; you need to provide relative path to pretrained model weights using argument `--pretrain-path`.
- `--max-iter` and `--batch-size`: we use `batch_size=32` to train on 4 GPUs each with 12GB memory.

To evaluate Mask-RCNN models, in addition to above training script, add `--eval-only` argument and provide relative path to downstream model weights using argument `--model-path`.

