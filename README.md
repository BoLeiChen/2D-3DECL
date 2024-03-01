# 2D-3DECL

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

Please refer to `data/README.md` for details in setting up training and evaluation dataset.
