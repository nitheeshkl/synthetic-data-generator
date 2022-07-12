from ctypes import set_errno
from random import sample
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
from scipy.spatial.transform import Rotation as R

Extents = namedtuple("Extents", ["x", "y", "z"])
Location = namedtuple("Location", ["x", "y", "z"])
Color = namedtuple("Color", ["r", "g", "b"])
ImageSize = namedtuple("ImageSize", ["w", "h"])
Intrinsics = namedtuple("Intrinsics", ["fx", "fy", "cx", "cy"])


class Scene:
    """
    This class represents the scene to be rendered to generate the dataset and
    contains all the required elements (container, camera, lights, etc).
    """

    _cfg: DictConfig  # contains all the config parameters parsed by Hydra
    _container: MeshObject  # container into which the objects will be packed
    _floor: MeshObject  # used to collect object that falls off the container
    _light_plane: MeshObject  # placed above the light source to act as a reflection/emission plane
    _light: bproc.types.Light  # a light source
    _objs_in_container: list[
        MeshObject, ...
    ] = None  # holds all the object packed in the container

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        # create the scene upon instatntiation
        self.__create()

    def __create(self) -> None:
        """
        Creates all the elements in the scene
        """
        self.__create_floor()
        self.__create_lighting()
        self.__create_container()
        self.__init_camera()
        self.__init_renderer()

    def reset(self) -> None:
        """
        Resets the scene by deleting all the elements and creating them again
        """
        objs = [
            self._container,
            self._floor,
            self._light,
            self._light_plane,
        ] + self._objs_in_container

        bproc.object.delete_multiple(objs)

        self.__create()

    def __init_camera(self) -> None:
        """ Initialize camera for the scene. """

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

        # set initial random pose
        self.add_rand_cam_pose()


    def __init_renderer(self) -> None:
        """ Initalize render settings. """

        # activate depth rendering
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        # activate normals
        bproc.renderer.enable_normals_output()
    
        bproc.renderer.set_max_amount_of_samples(50)

    def add_rand_cam_pose(self, frame: int = None) -> None:
        """ Add a rander camera pose looking at the container.

        :param frame: keyframe to add the pose into.
        """
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 1.0,
                                radius_max = 1.6,
                                elevation_min = 70,
                                elevation_max = 95,
                                azimuth_min = 0,
                                azimuth_max = 180,
                                uniform_volume = True)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi([self._container])
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # loc = Location(*self._cfg.camera.location)
        # cam2world_matrix = bproc.math.build_transformation_mat(loc, np.eye(3))


        # cam2world_matrix = bproc.math.build_transformation_mat(np.array(loc), np.eye(3))
        bproc.camera.add_camera_pose(cam2world_matrix, frame=frame)

    def __create_floor(self) -> None:
        """
        Create a floor plane to collect the object that fall off the container.
        It is placed at some negative depth below th container. This is done
        because later, any objects below z = 0 (i.e, outside container and
        falled on floor) will be filtered.
        """

        self._floor = bproc.object.create_primitive(
            "PLANE", scale=self._cfg.floor.extents, location=self._cfg.floor.location
        )
        # enable rigid body physics to collect the falled container
        self._floor.enable_rigidbody(
            False,  # does not actively particiate in simulation
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

    def __create_lighting(self) -> None:
        """ Create scene light and light_plane """

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

    def __create_container(self) -> None:
        """" Create container either by building a custom one or loading a container model. """

        # if there is a model_file defined in container config section
        if "model_file" in self._cfg.container:
            # then load the model from file

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
            # else build a new custom container
            self._container = build_container(
                Extents(*self._cfg.container.extents), self._cfg.container.thickness
            )

        # enable rigid body physics for container
        self._container.enable_rigidbody(
            False,  # does not actively participate in simulation, i.e, does not fall/move. Only act as static walls.
            collision_shape="MESH",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

        if self._cfg.container.texture_dir:
            texture = load_cc_texture(self._cfg.container.texture_dir)
            self._container.replace_materials(texture)

    def empty_container(self) -> None:
        """ Deletes all objects in container. """
        if self._objs_in_container:
            bproc.object.delete_multiple(self._objs_in_container)

    def render(self, num_poses: int=1) -> dict:
        """ Render the scene.
        
        :param num_poses: number of random poses to render from the scene.

        :return: data dict returned from blender render
        """

        # remove all existing keyframes
        bproc.utility.reset_keyframes()

        cam_poses = []
        # generate camera poses
        for i in range(num_poses):
            self.add_rand_cam_pose()
            cam_poses.append(CameraUtility.get_camera_pose())
        # render the whole scene
        data = bproc.renderer.render()

        # Render segmentation masks (per class and per instance)
        data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

        data['cam_pose'] = cam_poses

        return data 

    def drop_objs_into_container(
        self, sample_obj: MeshObject, num_objs: int, batch_size: int
    ) -> list[MeshObject, ...]:
        """ Drop objects into the container with physics simulation.

        :param sample_obj: The object to be replicated and dropped into the
                           container. This object instance will not be deleted, but will be moved
                           below the container to be avoideed from camera view.
        :param num_objs: Number of sample objects to be dropped into the container.
        :param batch_size: batch_size objects will be dropped at a time into the container.

        :return: list of objects in the container.
        """
        sample_obj.set_location([0, 0, 5])  # place sample obj outside container
        sample_obj = random_pose(sample_obj)
        obj_x, obj_y, obj_z = sample_obj.blender_obj["extents"]
        objs_to_keep = []  # holds the objects in the container

        # Define a function that samples 6-DoF poses
        def random_pose_func(obj: bproc.types.MeshObject) -> None:
            # TODO: improve object placing above container before dropping
            loc = np.random.normal(
                [0, 0, 2 * self._cfg.container.extents[2]], [0.06, 0.02, 0.1],
            )
            obj.set_location(loc)
            return

        for i in range(num_objs // batch_size):
            # create objects to be dropped
            objs = [sample_obj.duplicate() for i in range(batch_size)]

            # sample initial poses before dropping
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

            # collect objects outside the container to delete
            objs_to_delete = []
            for obj in objs:
                try:
                    objx, objy, objz = obj.get_location()
                except Exception as e:
                    print("Exception:\n", e)
                    objs_to_delete.append(obj)
                    continue

                if (
                    objz < -0.001
                    or objz > self._cfg.container.extents[2]
                    or (objz + obj_z / 2) > self._cfg.container.extents[2]
                ):
                    objs_to_delete.append(obj)
                else:
                    objs_to_keep.append(obj)

            print("deleting {} objs outside container".format(len(objs_to_delete)))
            bproc.object.delete_multiple(objs_to_delete)

        # TODO: find a better way to handle sample object
        # bproc.object.delete_multiple([sample_obj])

        self._objs_in_container = objs_to_keep

        return self._objs_in_container

    def order_objs_in_container(self, sample_object, num_objs):
        """  Pack objects in the container in an orderly fashion.
 
        :param sample_obj: The object to be replicated and dropped into the
                           container. This object instance will not be deleted, but will be moved
                           below the container to be avoideed from camera view.
        :param num_objs: Number of sample objects to be placed in the container.

        :return: list of objects in the container.
        """
        # TODO: cleanup code to order objects in container.

        # objects are places starting from the back right corner from bottom.
        # objects are stacked on top rows until max_z (container height) is
        # reached, and then moves to the row in front.

        sample_object.set_location([0, 0, 5])  # place sample obj outside container
        obj1 = sample_object.duplicate()
        sample_obj = random_pose(obj1)

        cont_x, cont_y, cont_z = self._cfg.container.extents
        start_x, start_y, start_z = cont_x / 2, cont_y / 2, 0
        obj_x, obj_y, obj_z = sample_obj.blender_obj["extents"]
        dx, dy, dz = self._cfg.container.pack_object_spacing
        rot_x, rot_y, rot_z = 0, 0, 0
        r = R.from_euler("xyz", sample_obj.get_rotation())
        new_obj_x, new_obj_y, new_obj_z = obj_x + dx, obj_y + dy, obj_z + dz
        obj_stack = [num_objs - i for i in range(num_objs)]
        objs_to_keep = []
        objs_to_del = []

        while obj_stack:

            if (start_x - new_obj_x) < (-cont_x / 2):
                # move to next row on top
                print("reached max x")
                start_x = cont_x / 2
                # start_y = cont_y/2
                start_z += new_obj_z

            if (start_z + new_obj_z) > cont_z:
                print("reached z, checking for better pose")

                # check for best pose to fit z
                x_objs = (cont_z - start_z) // new_obj_x
                y_objs = (cont_z - start_z) // new_obj_y
                print("x_objs={}, y_objs={}".format(x_objs, y_objs))
                prev_state = (new_obj_x, new_obj_y, new_obj_z, r)
                if x_objs > y_objs:  # then rotate around y
                    if (start_x - new_obj_x) > (-cont_x / 2):
                        print("rotating around y to fit z")
                        r = R.from_euler("y", np.pi / 2) * r
                        _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                        new_obj_x, new_obj_y, new_obj_z = _obj_z, _obj_y, _obj_x
                else:  # rotate around around x
                    if (start_y - new_obj_y) < (-cont_y / 2):
                        print("rotating around x to fit z")
                        rot_x = np.pi / 2
                        r = R.from_euler("x", np.pi / 2) * r
                        _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                        new_obj_x, new_obj_y, new_obj_z = _obj_x, _obj_z, _obj_y
                    else:
                        if new_obj_y > new_obj_x:
                            print("rot around z first and then y to fit x along z")
                            r = (
                                R.from_euler("y", np.pi / 2)
                                * R.from_euler("z", np.pi / 2)
                                * r
                            )
                            _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                            new_obj_x, new_obj_y, new_obj_z = _obj_z, _obj_x, _obj_y
                        else:
                            print("rot around around y to better fit z and x")
                            r = R.from_euler("y", np.pi / 2) * r
                            _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                            new_obj_x, new_obj_y, new_obj_z = _obj_z, _obj_y, _obj_x

                if (start_z + new_obj_z) > cont_z:
                    print("reached max z")
                    # move to next row in front
                    # and start from bottom with previous pose
                    new_obj_x, new_obj_y, new_obj_z, r = prev_state
                    start_x = cont_x / 2
                    start_y -= new_obj_y
                    start_z = 0
                print(
                    "start_x={}, start_y={}, start_z={}".format(
                        start_x, start_y, start_z
                    )
                )
                print(
                    "new obj_x={}, obj_y={}, obj_z={}".format(
                        new_obj_x, new_obj_y, new_obj_z
                    )
                )
                print("cont_x={}, cont_y={}, cont_z={}".format(cont_x, cont_y, cont_z))

            if (start_y - new_obj_y) < (-cont_y / 2):
                print("reached max y")
                # check for best pose to fit
                y_objs = cont_x // new_obj_y
                z_objs = cont_x // new_obj_z
                print("y_objs={}, z_objs={}".format(y_objs, z_objs))
                if y_objs > z_objs:  # then rotate around z
                    rot_z = np.pi / 2
                    r = R.from_euler("z", np.pi / 2) * r
                    _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                    new_obj_x, new_obj_y, new_obj_z = _obj_y, _obj_x, _obj_z
                else:  # rotate around around x
                    rot_x = np.pi / 2
                    r = R.from_euler("x", np.pi / 2) * r
                    _obj_x, _obj_y, _obj_z = new_obj_x, new_obj_y, new_obj_z
                    new_obj_x, new_obj_y, new_obj_z = _obj_x, _obj_z, _obj_y
                print(
                    "start_x={}, start_y={}, start_z={}".format(
                        start_x, start_y, start_z
                    )
                )
                print(
                    "new obj_x={}, obj_y={}, obj_z={}".format(
                        new_obj_x, new_obj_y, new_obj_z
                    )
                )

            if (start_y - new_obj_y) < (-cont_y / 2):
                # can't fit any more poses in front
                break

            x = start_x - new_obj_x / 2
            y = start_y - new_obj_y / 2
            z = start_z + new_obj_z / 2

            i = obj_stack.pop()
            obj = sample_object.duplicate()
            print(
                "placing obj {}. r=[{},{},{}] e=[{}, {}, {}]".format(
                    i, rot_x, rot_y, rot_z, new_obj_x, new_obj_y, new_obj_z
                )
            )
            # obj.set_rotation_euler([rot_x, rot_y, rot_z])
            obj.set_location([x, y, z])
            # t = bproc.math.build_transformation_mat([x, y, z], [rot_x, rot_y, rot_z])
            t = bproc.math.build_transformation_mat([x, y, z], r.as_matrix())
            obj.apply_T(t)
            objx, objy, objz = obj.get_location()
            if (objz - new_obj_z / 2) < -0.001 or (objz + new_obj_z / 2) > cont_z:
                objs_to_del.append(obj)
                print(
                    "removing obj ",
                    i,
                    (objz - new_obj_z / 2),
                    (objz + new_obj_z / 2),
                    cont_z,
                )
            else:
                objs_to_keep.append(obj)

            start_x -= new_obj_x

        bproc.object.delete_multiple([obj1,sample_obj] + objs_to_del)

        self._objs_in_container = objs_to_keep

        return self._objs_in_container


def random_pose(obj):
    obj_x, obj_y, obj_z = obj.blender_obj.dimensions
    loc = obj.get_location()
    r = R.identity()
    combinations = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    choice = np.random.randint(0, 8)
    rand_comb = combinations[choice]
    print("generating randome pose with choice = ", choice)
    for i in range(3):
        if rand_comb[i] == 1:
            if i == 2:  # rot x
                print("rotating x")
                r = R.from_euler("x", np.random.choice([-np.pi / 2, np.pi / 2])) * r
                _obj_x, _obj_y, _obj_z = obj_x, obj_y, obj_z
                obj_x, obj_y, obj_z = _obj_x, _obj_z, _obj_y
            if i == 1:  # rot y
                print("rotating y")
                r = R.from_euler("y", np.random.choice([-np.pi / 2, np.pi / 2])) * r
                _obj_x, _obj_y, _obj_z = obj_x, obj_y, obj_z
                obj_x, obj_y, obj_z = _obj_z, _obj_y, _obj_x
            if i == 0:  # rot z
                print("rotating z")
                r = R.from_euler("z", np.random.choice([-np.pi / 2, np.pi / 2])) * r
                _obj_x, _obj_y, _obj_z = obj_x, obj_y, obj_z
                obj_x, obj_y, obj_z = _obj_y, _obj_x, _obj_z
    t = bproc.math.build_transformation_mat(loc, r.as_matrix())
    obj.apply_T(t)
    # FIXME: find a better way to handle object extents
    obj.blender_obj["extents"] = [obj_x, obj_y, obj_z]
    return obj


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
