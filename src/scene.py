from cgitb import text
from hashlib import new
from logging.handlers import RotatingFileHandler
from typing import Dict, NamedTuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from collections import namedtuple
import os
import json
import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.LightUtility import Light
import blenderproc.python.camera.CameraUtility as CameraUtility
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.loader.CCMaterialLoader import CCMaterialLoader
import numpy as np

Extents = namedtuple("Extents", ["x", "y", "z"])
Location = namedtuple("Location", ["x", "y", "z"])
Color = namedtuple("Color", ["r", "g", "b"])
ImageSize = namedtuple("ImageSize", ["w", "h"])
Intrinsics = namedtuple("Intrinsics", ["fx", "fy", "cx", "cy"])


class Container:

    _extents: Extents
    _thickness: float
    _bproc_obj: MeshObject

    def __init__(self, extents: Extents, thickness: float, texture_dir: str = None):
        self._extents = extents
        self._thickness = thickness
        self._create(texture_dir)

    def _create(self, texture_dir=None):
        # Create container as looking at xz plane. (x=right, z=top, y=inwards) The sides
        # are created by first placing a cube flat (on xy plane) and rotating abound the
        # x or y axis

        # bottom side, no rotation
        c_bottom = bproc.object.create_primitive(
            "CUBE",
            scale=[self._extents.x / 2, self._extents.y / 2, self._thickness / 2],
            location=[
                0,
                0,
                -self._thickness / 2,
            ],  # ensure the inside bottom plane alings with xy plane at z=0
        )

        # front side, plane rotated along x-axis
        c_front = bproc.object.create_primitive(
            "CUBE",
            scale=[
                (self._extents.x + 2 * self._thickness) / 2,
                (self._extents.z + self._thickness) / 2,
                self._thickness / 2,
            ],
            location=[
                0,
                -(self._extents.y + self._thickness) / 2,
                (self._extents.z - self._thickness) / 2,
            ],
            rotation=[-np.pi / 2, 0, 0],
        )

        # back side, plane rotated along x-axis
        c_back = bproc.object.create_primitive(
            "CUBE",
            scale=[
                (self._extents.x + 2 * self._thickness) / 2,
                (self._extents.z + self._thickness) / 2,
                self._thickness / 2,
            ],
            location=[
                0,
                (self._extents.y + self._thickness) / 2,
                (self._extents.z - self._thickness) / 2,
            ],
            rotation=[np.pi / 2, 0, 0],
        )

        # left side, plane rotated along y-axis
        c_left = bproc.object.create_primitive(
            "CUBE",
            scale=[
                (self._extents.z + self._thickness) / 2,
                self._extents.y / 2,
                self._thickness / 2,
            ],
            location=[
                -(self._extents.x + self._thickness) / 2,
                0,
                (self._extents.z - self._thickness) / 2,
            ],
            rotation=[0, -np.pi / 2, 0],
        )

        # right side, plane rotated along y-axis
        c_right = bproc.object.create_primitive(
            "CUBE",
            scale=[
                (self._extents.z + self._thickness) / 2,
                self._extents.y / 2,
                self._thickness / 2,
            ],
            location=[
                (self._extents.x + self._thickness) / 2,
                0,
                (self._extents.z - self._thickness) / 2,
            ],
            rotation=[0, np.pi / 2, 0],
        )

        self._bproc_obj = bproc.python.types.MeshObjectUtility.create_with_empty_mesh(
            "container"
        )
        self._bproc_obj.join_with_other_objects(
            [c_bottom, c_front, c_back, c_left, c_right]
        )
        # enable rigid body physics for container and floor
        self._bproc_obj.enable_rigidbody(
            False,
            collision_shape="MESH",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
        if texture_dir:
            asset = os.path.basename(texture_dir)
            if asset == "":
                asset = texture_dir.split("/")[-2]

            base_image_path = os.path.join(texture_dir, "{}_2K_Color.jpg".format(asset))
            if not os.path.exists(base_image_path):
                print("Texture not found for container. Continuing without texture.")
                return
            else:
                print("Loading container texture from ", texture_dir)

            # construct all image paths
            ambient_occlusion_image_path = base_image_path.replace(
                "Color", "AmbientOcclusion"
            )
            metallic_image_path = base_image_path.replace("Color", "Metalness")
            roughness_image_path = base_image_path.replace("Color", "Roughness")
            alpha_image_path = base_image_path.replace("Color", "Opacity")
            normal_image_path = base_image_path.replace("Color", "Normal")
            displacement_image_path = base_image_path.replace("Color", "Displacement")

            new_mat = MaterialLoaderUtility.create_new_cc_material(
                asset, add_custom_properties={}
            )
            CCMaterialLoader.create_material(
                new_mat,
                base_image_path,
                ambient_occlusion_image_path,
                metallic_image_path,
                roughness_image_path,
                alpha_image_path,
                normal_image_path,
                displacement_image_path,
            )
            material = Material(new_mat)
            self._bproc_obj.replace_materials(material)

    def print(self):
        print(self._extents)


class Light:

    _energy: float
    _color: Color
    _loc: Location
    _bproc_obj: Light

    def __init__(self, energy: float, color: Color, location: Location):
        self._energy = energy
        self._color = color
        self._loc = location
        self._create()

    def _create(self):
        self._bproc_obj = bproc.types.Light()
        self._bproc_obj.set_energy(self._energy)
        self._bproc_obj.set_color(self._color)
        self._bproc_obj.set_location(self._loc)


class LightPlane:

    _emission_strength: float
    _emission_color: Color
    _extents: Extents
    _loc: Location
    _bproc_obj: MeshObject

    def __init__(
        self,
        emission_strength: float,
        emission_color: Color,
        extents: Extents,
        location: Location,
    ) -> None:
        self._emission_strength = emission_strength
        self._emission_color = emission_color
        self._extents = extents
        self._loc = location

        self._create()

    def _create(self) -> None:

        self._bproc_obj = bproc.object.create_primitive(
            "PLANE", scale=self._extents, location=self._loc
        )
        self._bproc_obj.set_name("light_plane")
        material = bproc.material.create("light_material")
        r, g, b = self._emission_color
        material.make_emissive(
            emission_strength=self._emission_strength, emission_color=[r, g, b, 1.0],
        )
        self._bproc_obj.replace_materials(material)


class Object:

    _ID: int
    _bop_dataset_path: str
    _extents: Extents
    _bproc_obj: MeshObject

    def __init__(
        self, ID: int, bop_dataset_path: str, bop_dataset_name: str, mm2m: bool = True,
    ):
        self._ID = ID
        self._bop_dataset_path = os.path.join(bop_dataset_path, bop_dataset_name)

        self._create(mm2m)

    def _create(self, mm2m=True):

        obj = bproc.loader.load_bop_objs(
            bop_dataset_path=self._bop_dataset_path,
            mm2m=mm2m,
            sample_objects=False,
            obj_ids=[self._ID],
        )[0]

        obj.enable_rigidbody(
            True, friction=100.0, linear_damping=0.99, angular_damping=0.99
        )
        obj.set_shading_mode("auto")
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

        models_info_file = os.path.join(
            self._bop_dataset_path, "models", "models_info.json"
        )
        with open(models_info_file) as f:
            models_info = json.load(f)
            model_info = models_info[str(self._ID)]
            self._extents = Extents(
                model_info["size_x"], model_info["size_y"], model_info["size_z"]
            )

        self._extents = Extents(0.1, 0.1, 0.1)
        return

    def print(self):
        print(self._extents)


class Camera:

    _loc: Location
    _rot: np.ndarray
    _K: np.ndarray
    _img_size: ImageSize

    def __init__(
        self, location: Location, rot=np.eye(3), intrinsics=None, image_size=None
    ):
        self._loc = location
        self._rot = rot
        self._K = intrinsics
        self._img_size = image_size

        self._create()

    def _create(self):

        """
        The K matrix should have the format:
            [[fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1]]
        """
        k_mat = np.eye(3)
        k_mat[0][0], k_mat[1][1] = self._K.fx, self._K.fy
        k_mat[0][2], k_mat[1][2] = self._K.cx, self._K.cy

        CameraUtility.set_intrinsics_from_K_matrix(
            k_mat, self._img_size.w, self._img_size.h
        )

        cam2world_matrix = bproc.math.build_transformation_mat(
            np.array(self._loc), self._rot
        )

        bproc.camera.add_camera_pose(cam2world_matrix)


class Floor:

    _extents: Extents
    _loc: Location
    _bproc_obj: MeshObject

    def __init__(self, extents, location):
        self._extents = extents
        self._loc = location
        self._create()

    def _create(self, texture=None):
        # Create a floor plane to collect the object that fall off the container and it
        # at some negative depth below th container. This is done because later, any
        # objects below z = 0 (i.e, outside container and falled on floor) will be
        # filtered.
        self._bproc_obj = bproc.object.create_primitive(
            "PLANE", scale=self._extents, location=self._loc
        )
        self._bproc_obj.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
        if texture:
            self._bproc_obj.replace_materials(texture)


class Scene:

    _cfg: DictConfig
    _container: Container
    _object: Object
    _floor: Floor
    _light: Light
    _light_plane: LightPlane
    _camera: Camera

    def __init__(
        self,
        cfg: DictConfig,
        container: Container,
        object: Object,
        floor: Floor,
        light: Light,
        light_plane: LightPlane,
        camera: Camera,
    ) -> None:
        self._cfg = cfg
        self._container = container
        self._object = object
        self._floor = floor
        self._light = light
        self._light_plane = light_plane
        self._camera = camera

    def print(self):
        print("num_objs : ", self._cfg.num_objs)
        self._container.print()
        self._object.print()
