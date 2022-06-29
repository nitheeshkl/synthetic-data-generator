
from cgitb import text
from typing import Dict, NamedTuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from collections import namedtuple

import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.types.LightUtility import Light
import numpy as np

Extents = namedtuple('Extents', ['x', 'y', 'z'])
Location = namedtuple('Location', ['x', 'y', 'z'])
Color = namedtuple('Color', ['r', 'g', 'b'])
ImageSize = namedtuple('ImageSize', ['w', 'h'])
Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'cx', 'cy'])

class Container:

    _extents : Extents
    _thickness: float
    _bproc_obj: MeshObject

    def __init__(self, extents: Extents, thickness: float):
        self._extents = extents
        self._thickness = thickness
        self._create()

    def _create(self, texture=None):
        # Create container as looking at xz plane. (x=right, z=top, y=inwards) The sides
        # are created by first placing a cube flat (on xy plane) and rotating abound the
        # x or y axis

        # bottom side, no rotation
        c_bottom = bproc.object.create_primitive('CUBE', 
                        scale=[self._extents.x/2, self._extents.y/2, self._thickness/2],
                        location=[0, 0, -self._thickness/2] # ensure the inside bottom plane alings with xy plane at z=0
                        )

        # front side, plane rotated along x-axis
        c_front = bproc.object.create_primitive('CUBE',
                        scale=[(self._extents.x + 2*self._thickness)/2, (self._extents.z + self._thickness)/2, self._thickness/2],
                        location=[0, -(self._extents.y + self._thickness)/2, (self._extents.z - self._thickness)/2],
                        rotation=[-np.pi/2, 0, 0]) 

        # back side, plane rotated along x-axis
        c_back = bproc.object.create_primitive('CUBE',
                        scale=[(self._extents.x + 2*self._thickness)/2, (self._extents.z + self._thickness)/2, self._thickness/2],
                        location=[0, (self._extents.y + self._thickness)/2, (self._extents.z - self._thickness)/2],
                        rotation=[np.pi/2, 0, 0]) 

        # left side, plane rotated along y-axis
        c_left = bproc.object.create_primitive('CUBE',
                        scale=[(self._extents.z + self._thickness)/2, self._extents.y/2, self._thickness/2],
                        location=[-(self._extents.x + self._thickness)/2, 0, (self._extents.z - self._thickness)/2],
                        rotation=[0, -np.pi/2, 0]) 

        # right side, plane rotated along y-axis
        c_right = bproc.object.create_primitive('CUBE',
                        scale=[(self._extents.z + self._thickness)/2, self._extents.y/2, self._thickness/2],
                        location=[(self._extents.x + self._thickness)/2, 0, (self._extents.z - self._thickness)/2],
                        rotation=[0, np.pi/2, 0]) 

        self._bproc_obj = bproc.python.types.MeshObjectUtility.create_with_empty_mesh("container")
        self._bproc_obj.join_with_other_objects([c_bottom, c_front, c_back, c_left, c_right])
        # enable rigid body physics for container and floor
        self._bproc_obj.enable_rigidbody(False, collision_shape='MESH', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        if texture:
            self._bproc_obj.replace_materials(texture)

    def print(self):
        print(self._extents)

class Light:

    _energy : float
    _color : Color
    _loc : Location
    _bproc_obj : Light 

    def __init__(self, energy : float, color : Color, location : Location):
        self._energy = energy
        self._color = color
        self._loc = location
        self._create()

    def _create(self):
        self._bproc_obj = bproc.types.Light()
        self._bproc_obj.set_energy(self._energy)
        self._bproc_obj.set_color(self._color)
        self._bproc_obj.set_location(self._loc)

class Object:

    _ID : int
    _extents : Extents

    def __init__(self, ID, extents: Extents):
        self._ID = ID
        self._extents = extents

    def print(self):
        print(self._extents)

class Camera:

    _loc : Location
    _rot : np.ndarray
    _K : np.ndarray
    _img_w : ImageSize

    def __init__(self):
        return

class Floor:

    _extents: Extents
    _loc : Location
    _bproc_obj : MeshObject

    def __init__(self, extents, location):
        self._extents = extents
        self._loc = location
        self._create() 

    def _create(self, texture=None):
        # Create a floor plane to collect the object that fall off the container and it
        # at some negative depth below th container. This is done because later, any
        # objects below z = 0 (i.e, outside container and falled on floor) will be
        # filtered.
        self._bproc_obj = bproc.object.create_primitive('PLANE', scale=self._extents, location=self._loc)
        self._bproc_obj.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        if texture:
            self._bproc_obj.replace_materials(texture)
        

class Scene:

    _cfg: DictConfig
    _container : Container
    _object : Object
    _floor : Floor
    _light : Light

    def __init__(self, cfg : DictConfig, container : Container, object: Object, floor: Floor, light: Light) -> None:
        self._cfg = cfg
        self._container = container
        self._object = object
        self._floor = floor
        self._light = light


    def print(self):
        print("num_objs : ", self._cfg.num_objs)
        self._container.print()
        self._object.print()