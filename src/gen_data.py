import blenderproc as bproc

import os
import sys
from threading import Thread
import time

if "INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT" in os.environ:
    SRC_DIR = "/home/kln/sandbox/cmu/repos/capstone/synthetic-data-generator/src"
    sys.path.append(SRC_DIR)
    CONF_DIR = os.path.join(SRC_DIR, "../conf")
else:
    CONF_DIR = "../conf"

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from scene import Scene, load_object_model
from data_writer import DataWriter

import numpy as np
from scipy.spatial.transform import Rotation as R

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


def write_data(cfg, data):

    bproc.writer.write_hdf5(
        os.path.join(cfg.output_dir, "hdf5"), data, append_to_existing_output=True
    )

    # Write data in bop format
    # bproc.writer.write_bop(
    #     os.path.join(cfg.output_dir, "bop_data"),
    #     dataset=cfg.name,
    #     depths=data["depth"],
    #     colors=data["colors"],
    #     color_file_format=cfg.rgb_img_format,
    #     ignore_dist_thres=cfg.ignore_dist_thres,
    # )

    # bproc.writer.write_coco_annotations(
    #     os.path.join(cfg.output_dir, "coco_data"),
    #     instance_segmaps=data["instance_segmaps"],
    #     instance_attribute_maps=data["instance_attribute_maps"],
    #     colors=data["instance_segmaps"],
    #     color_file_format=cfg.rgb_img_format,
    #     mask_encoding_format="rle",
    # )

    return


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:

    bproc.init()
    scene = Scene(cfg.scene)
    obj = load_object_model(cfg.object)
    data_writer = DataWriter(cfg.dataset)

    for i in range(cfg.dataset.num_scenes):

        start = time.time()
        if cfg.pack_type == "random":
            # num_objs = np.random.randint(cfg.batch_size, cfg.num_objs)
            num_objs = cfg.num_objs
            objs_in_container = scene.drop_objs_into_container(
                obj, num_objs, cfg.batch_size
            )
        elif cfg.pack_type == "ordered":
            # num_objs = np.random.randint(cfg.batch_size, cfg.num_objs)
            num_objs = cfg.num_objs
            objs_in_container = scene.order_objs_in_container(obj, num_objs)
        else:
            objs_in_container = []

        if len(objs_in_container) > 0:
            data = scene.render(num_poses=cfg.dataset.num_poses_per_scene)
            metadata = {
                "obj_id": cfg.object.bop_dataset_name + "_" + str(cfg.object.ID),
                "scene": i,
                "pack_type": cfg.pack_type,
            }
            data_writer.write_data_hdf5(data, metadata)
            # data_writer.write_data(data)
            # write_data(cfg.dataset, data)
            # thread = Thread(target=data_writer.write_data, args=(data,))
            # thread = Thread(target=write_data, args=(cfg.dataset,data))
            # thread.start()
            if i + 1 != cfg.dataset.num_scenes:
                scene.empty_container()
            print("rendered ", i)
        else:
            print("no objects in container!!")

        end = time.time()
        print("KLN: time per scene = ", end - start)


if __name__ == "__main__":
    main()
