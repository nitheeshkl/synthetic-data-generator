import blenderproc as bproc

import os
import glob
from omegaconf import DictConfig, OmegaConf
import hydra
import json
import numpy as np
import imageio
import bpy
import blenderproc.python.camera.CameraUtility as CameraUtility


class DataWriter:
    def __init__(self, cfg: DictConfig) -> None:

        self._cfg = cfg
        self._hdf5_dir = os.path.join(cfg.output_dir, "hdf5")
        self._color_dir = os.path.join(cfg.output_dir, "rgb")
        self._depth_dir = os.path.join(cfg.output_dir, "depth")
        self._mask_dir = os.path.join(cfg.output_dir, "mask")
        self._normals_dir = os.path.join(cfg.output_dir, "normals")
        self._cam_pose_dir = os.path.join(cfg.output_dir, "cam_pose")

        for d in [
            self._color_dir,
            self._depth_dir,
            self._mask_dir,
            self._normals_dir,
            self._cam_pose_dir,
        ]:
            if not os.path.exists(d):
                print("creating ", d)
                os.makedirs(d)

        self._file_idx = len(glob.glob(os.path.join(self._color_dir, "*.png")))

    def write_data_hdf5(self, data, metadata):

        data_dir = os.path.join(
            self._hdf5_dir,
            metadata["obj_id"],
            metadata["pack_type"],
            "scene_{:04d}".format(metadata["scene"]),
        )
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        bproc.writer.write_hdf5(data_dir, data, append_to_existing_output=True)

        self.write_cam_data()

    def write_data(self, data):

        num_imgs = len(data["colors"])

        for i in range(num_imgs):
            print("writing index ", self._file_idx)
            rgb = data["colors"][i]
            depth = data["depth"][i]
            mask = data["instance_segmaps"][i]
            # normals = data["normals"][i]
            cam_pose = data["cam_pose"][i]

            # partial_depth = self.process_depth(depth, normals, mask, cam_pose)

            # meter to mm
            depth = depth * 1000.0
            # partial_depth = partial_depth * 1000.0

            rgb_img = os.path.join(self._color_dir, "{:06d}.png".format(self._file_idx))
            depth_img = os.path.join(
                self._depth_dir, "{:06d}.npy".format(self._file_idx)
            )
            partial_depth_img = os.path.join(
                self._depth_dir, "partial_{:06d}.npy".format(self._file_idx)
            )
            mask_img = os.path.join(self._mask_dir, "{:06d}.png".format(self._file_idx))
            normals_img = os.path.join(
                self._normals_dir, "{:06d}.npy".format(self._file_idx)
            )
            cam_pose_file = os.path.join(
                self._cam_pose_dir, "{:06d}.npy".format(self._file_idx)
            )

            imageio.imwrite(rgb_img, rgb)
            imageio.imwrite(mask_img, mask)
            np.save(depth_img, depth)
            # np.save(partial_depth_img, partial_depth)
            # np.save(normals_img, normals)
            np.save(cam_pose_file, cam_pose)

            self._file_idx += 1

        self.write_cam_data()

    def write_cam_data(self):

        cam_K = CameraUtility.get_intrinsics_as_K_matrix()
        camera = {
            "cx": cam_K[0][2],
            "cy": cam_K[1][2],
            "depth_scale": 1.0,
            "fx": cam_K[0][0],
            "fy": cam_K[1][1],
            "height": bpy.context.scene.render.resolution_y,
            "width": bpy.context.scene.render.resolution_x,
        }

        cam_info_file = os.path.join(self._cfg.output_dir, "camera.json")
        with open(cam_info_file, "w") as f:
            json.dump(camera, f)

    def process_depth(self, depth, normals, segmask, cam_pose):

        cam_loc = cam_pose[:3, 3]
        unit_norms = normals / np.expand_dims(np.linalg.norm(normals, axis=-1), -1)
        unit_vec = cam_loc / np.linalg.norm(cam_loc)

        n = np.abs(np.dot(unit_norms, unit_vec))
        theta = np.radians(45)
        mask = (n > np.cos(theta)).astype(float)

        processed_depth = depth * mask
        processed_depth[np.where(segmask[:, :] == 1.0)] = 0.0

        return processed_depth
