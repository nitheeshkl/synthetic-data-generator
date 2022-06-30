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


def render(cfg):
    ## render the whole pipeline
    data = bproc.renderer.render()

    ## Render segmentation masks (per class and per instance)
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    ## Write data in bop format
    bproc.writer.write_bop(
        os.path.join(cfg.output_dir, "bop_data"),
        dataset=cfg.name,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format=cfg.rgb_img_format,
        ignore_dist_thres=cfg.ignore_dist_thres,
    )

    bproc.writer.write_coco_annotations(
        os.path.join(cfg.output_dir, "coco_data"),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["instance_segmaps"],
        color_file_format=cfg.rgb_img_format,
        mask_encoding_format="rle",
    )


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:

    bproc.init()
    scene = instantiate(cfg.scene)
    obj = load_object_model(cfg.object)

    ## activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    for i in range(5):
        if cfg.pack_type == "random":
            scene.drop_objs_into_container(obj, cfg.num_objs, cfg.batch_size)

        render(cfg.dataset)
        scene.empty_container()
        print("iter ", i)


if __name__ == "__main__":
    main()
