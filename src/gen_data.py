import blenderproc as bproc

import os
import sys
import time

if "INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT" in os.environ:
    SRC_DIR = "/home/mujin/sandbox/mujin/repos/synthetic-data-generator/src"
    sys.path.append(SRC_DIR)
    CONF_DIR = os.path.join(SRC_DIR, "../conf")
else:
    CONF_DIR = "../conf"

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from scene import Scene, load_object_model


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:

    bproc.init()
    scene = instantiate(cfg.scene)
    obj = load_object_model(cfg.object)

    if cfg.pack_type == "random":
        scene.drop_objs_into_container(obj, cfg.num_objs, cfg.batch_size)


if __name__ == "__main__":
    main()
