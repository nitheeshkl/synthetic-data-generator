from omegaconf import DictConfig, OmegaConf
import hydra
from collections import namedtuple
import os
import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.types.MaterialUtility import Material
import blenderproc.python.camera.CameraUtility as CameraUtility
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.loader.CCMaterialLoader import CCMaterialLoader
import numpy as np

Extents = namedtuple("Extents", ["x", "y", "z"])
Location = namedtuple("Location", ["x", "y", "z"])
Color = namedtuple("Color", ["r", "g", "b"])
ImageSize = namedtuple("ImageSize", ["w", "h"])
Intrinsics = namedtuple("Intrinsics", ["fx", "fy", "cx", "cy"])


class Scene:

    _cfg: DictConfig
    _container: MeshObject
    _floor: MeshObject
    _light_plane: MeshObject
    _light: bproc.types.Light

    _objs_in_container: list[MeshObject, ...] = None

    def __init__(self, cfg: DictConfig,) -> None:
        self._cfg = cfg

        self.__create()

        return

    def __create(self):
        self.__create_floor()
        self.__create_lighting()
        self.__init_camera()
        self.__create_container()

        return

    def reset(self):
        objs = [
            self._container,
            self._floor,
            self._light,
            self._light_plane,
        ] + self._objs_in_container

        bproc.object.delete_multiple(objs)

        self.__create()

    def __init_camera(self):

        loc = Location(*self._cfg.camera.location)
        fx, fy, cx, cy = Intrinsics(**self._cfg.camera.intrinsics)
        img_w, img_h = ImageSize(**self._cfg.camera.image_size)

        """
        The K matrix should have the format:
            [[fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1]]
        """
        k_mat = np.eye(3)
        k_mat[0][0], k_mat[1][1] = fx, fy
        k_mat[0][2], k_mat[1][2] = cx, cy

        CameraUtility.set_intrinsics_from_K_matrix(k_mat, img_w, img_h)
        cam2world_matrix = bproc.math.build_transformation_mat(np.array(loc), np.eye(3))
        bproc.camera.add_camera_pose(cam2world_matrix)

    def __create_floor(self):
        # Create a floor plane to collect the object that fall off the container and it
        # at some negative depth below th container. This is done because later, any
        # objects below z = 0 (i.e, outside container and falled on floor) will be
        # filtered.
        self._floor = bproc.object.create_primitive(
            "PLANE", scale=self._cfg.floor.extents, location=self._cfg.floor.location
        )
        self._floor.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

    def __create_lighting(self):
        self._light = bproc.types.Light()
        self._light.set_energy(self._cfg.light.energy)
        self._light.set_color(self._cfg.light.color)
        self._light.set_location(self._cfg.light.location)

        self._light_plane = bproc.object.create_primitive(
            "PLANE",
            scale=self._cfg.light_plane.extents,
            location=self._cfg.light_plane.location,
        )
        self._light_plane.set_name("light_plane")
        material = bproc.material.create("light_material")
        material.make_emissive(
            emission_strength=self._cfg.light_plane.emission_strength,
            emission_color=self._cfg.light_plane.emission_color,
        )
        self._light_plane.replace_materials(material)

    def __create_container(self):

        if "model_file" in self._cfg.container:

            self._container = bproc.loader.load_obj(self._cfg.container.model_file)[0]

            # update extents to represent inside space of the container
            c_thickness = self._cfg.container.thickness
            c_x, c_y, c_z = self._container.blender_obj.dimensions
            cont_x = c_x - (2 * c_thickness)
            cont_y = c_y - (2 * c_thickness)
            cont_z = c_z - (0.013)

            self._cfg.container.extents = [cont_x, cont_y, cont_z]

            # move container center to origin
            self._container.set_location([-c_x / 2, -c_y / 2, -0.013])

        else:
            self._container = build_container(
                Extents(*self._cfg.container.extents), self._cfg.container.thickness
            )

        # enable rigid body physics for container and floor
        self._container.enable_rigidbody(
            False,
            collision_shape="MESH",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

        if self._cfg.container.texture_dir:
            texture = load_cc_texture(self._cfg.container.texture_dir)
            self._container.replace_materials(texture)

    def empty_container(self):
        bproc.object.delete_multiple(self._objs_in_container)

    def drop_objs_into_container(self, sample_obj, num_objs, batch_size):
        sample_obj.set_location([0, 0, -0.2])  # place sample obj outside container
        objs_to_keep = []

        # Define a function that samples 6-DoF poses
        def random_pose_func(obj: bproc.types.MeshObject):
            loc = np.random.normal(
                [0, 0, 2 * self._cfg.container.extents[2]], [0.06, 0.02, 0.1]
            )
            obj.set_location(loc)
            return

        for i in range(num_objs // batch_size):
            objs = [sample_obj.duplicate() for i in range(batch_size)]

            bproc.object.sample_poses(
                objects_to_sample=objs,
                sample_pose_func=random_pose_func,
                max_tries=1000,
            )

            # Physics Positioning
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=self._cfg.simulation.min_sim_time,
                max_simulation_time=self._cfg.simulation.max_sim_time,
                check_object_interval=self._cfg.simulation.check_object_interval,
                substeps_per_frame=self._cfg.simulation.substeps_per_frame,
                solver_iters=self._cfg.simulation.solver_iters,
            )

            objs_to_delete = []
            for obj in objs:
                objx, objy, objz = obj.get_location()

                if (
                    objz < 0
                    or (objz + obj.blender_obj.dimensions[2] / 2)
                    > self._cfg.container.extents[2]
                ):
                    objs_to_delete.append(obj)
                else:
                    objs_to_keep.append(obj)

            print("deleting {} objs outside container".format(len(objs_to_delete)))
            bproc.object.delete_multiple(objs_to_delete)

        # bproc.object.delete_multiple([sample_obj])

        self._objs_in_container = objs_to_keep

        return self._objs_in_container


def build_container(cont_extents, cont_thickness):
    # Create container as looking at xz plane. (x=right, z=top, y=inwards) The sides
    # are created by first placing a cube flat (on xy plane) and rotating abound the
    # x or y axis

    # bottom side, no rotation
    c_bottom = bproc.object.create_primitive(
        "CUBE",
        scale=[cont_extents.x / 2, cont_extents.y / 2, cont_thickness / 2],
        location=[
            0,
            0,
            -cont_thickness / 2,
        ],  # ensure the inside bottom plane alings with xy plane at z=0
    )

    # front side, plane rotated along x-axis
    c_front = bproc.object.create_primitive(
        "CUBE",
        scale=[
            (cont_extents.x + 2 * cont_thickness) / 2,
            (cont_extents.z + cont_thickness) / 2,
            cont_thickness / 2,
        ],
        location=[
            0,
            -(cont_extents.y + cont_thickness) / 2,
            (cont_extents.z - cont_thickness) / 2,
        ],
        rotation=[-np.pi / 2, 0, 0],
    )

    # back side, plane rotated along x-axis
    c_back = bproc.object.create_primitive(
        "CUBE",
        scale=[
            (cont_extents.x + 2 * cont_thickness) / 2,
            (cont_extents.z + cont_thickness) / 2,
            cont_thickness / 2,
        ],
        location=[
            0,
            (cont_extents.y + cont_thickness) / 2,
            (cont_extents.z - cont_thickness) / 2,
        ],
        rotation=[np.pi / 2, 0, 0],
    )

    # left side, plane rotated along y-axis
    c_left = bproc.object.create_primitive(
        "CUBE",
        scale=[
            (cont_extents.z + cont_thickness) / 2,
            cont_extents.y / 2,
            cont_thickness / 2,
        ],
        location=[
            -(cont_extents.x + cont_thickness) / 2,
            0,
            (cont_extents.z - cont_thickness) / 2,
        ],
        rotation=[0, -np.pi / 2, 0],
    )

    # right side, plane rotated along y-axis
    c_right = bproc.object.create_primitive(
        "CUBE",
        scale=[
            (cont_extents.z + cont_thickness) / 2,
            cont_extents.y / 2,
            cont_thickness / 2,
        ],
        location=[
            (cont_extents.x + cont_thickness) / 2,
            0,
            (cont_extents.z - cont_thickness) / 2,
        ],
        rotation=[0, np.pi / 2, 0],
    )

    container = bproc.python.types.MeshObjectUtility.create_with_empty_mesh("container")

    container.join_with_other_objects([c_bottom, c_front, c_back, c_left, c_right])

    return container


def load_cc_texture(texture_dir):
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
    ambient_occlusion_image_path = base_image_path.replace("Color", "AmbientOcclusion")
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

    return material


def load_object_model(cfg):
    bop_dataset_path = os.path.join(cfg.bop_dataset_path, cfg.bop_dataset_name)
    obj = bproc.loader.load_bop_objs(
        bop_dataset_path=bop_dataset_path,
        mm2m=cfg.mm2m,
        sample_objects=False,
        obj_ids=[cfg.ID],
    )[0]

    obj.enable_rigidbody(
        True, friction=100.0, linear_damping=0.99, angular_damping=0.99
    )
    obj.set_shading_mode("auto")
    mat = obj.get_materials()[0]
    # TODO: check if it is required to modify object roughness and specularity
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

    return obj
