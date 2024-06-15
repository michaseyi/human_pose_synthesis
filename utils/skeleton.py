
from typing import Dict, Optional, List
from utils.custom_types import Vec3
import numpy as np


class Bone:
    parent: Optional['Bone']
    children: List['Bone']
    name: str
    length: float
    direction: Optional[Vec3]
    position: Optional[Vec3]
    orientation: Optional[Vec3]
    dof: List[int]
    bind_matrix: np.ndarray
    inverse_bind_matrix: np.ndarray
    bone_transform: np.ndarray

    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.length = 0
        self.bind_matrix = np.eye(3)
        self.inverse_bind_matrix = np.eye(3)
        self.bone_transform = np.eye(3)


class Skeleton:
    root: Optional[Bone]
    bone_map: Dict[str, Bone]

    mass: float
    length: float
    angle: str

    def __init__(self):
        self.root = None
        self.bone_map = {}
        self.mass = 0
        self.length = 0
        self.angle = "deg"
