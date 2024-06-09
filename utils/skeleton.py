
from typing import Dict, Optional, List
from utils.custom_types import Vec3


class Bone:
    parent: Optional['Bone']
    children: List['Bone']
    name: str
    length: float
    direction: Optional[Vec3]
    position: Optional[Vec3]
    orientation: Vec3
    dof: List[str]

    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.length = 0


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
