from typing import Dict, Optional, List
from transforms3d.euler import euler2mat
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

MotionFrame = Dict[str, List[float]]

class Bone:
    parent: Optional['Bone']
    children: List['Bone']
    name: str
    length: float
    direction: Optional[List[float]]
    position: Optional[List[float]]
    orientation: Optional[List[float]]
    dof: List[int]
    bind_matrix: np.ndarray
    inverse_bind_matrix: np.ndarray
    bone_transform: np.ndarray
    axis: Optional[List[float]]
    limits: Optional[List[List[float]]]

    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.length = 0
        self.bind_matrix = np.eye(3)
        self.inverse_bind_matrix = np.eye(3)
        self.bone_transform = np.eye(3)
        self.limits = None
        self.direction = None
        self.position = None
        self.orientation = None
        self.dof = []
        self.axis = None


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


def set_motion(skeleton: Skeleton, motion: MotionFrame):
    root_motion = motion.get("root")
    assert root_motion is not None
    assert skeleton.root is not None
    skeleton.root.position = root_motion[:3]
    skeleton.root.orientation = root_motion[3: 6]

    for [name, bone] in skeleton.bone_map.items():
        if name == "root":
            continue
        bone_motion = motion.get(name)

        if bone_motion is None:
            continue

        rotation: list[float] = [0, 0, 0]

        for i, dof in enumerate(bone.dof):
            rotation[dof] = bone_motion[i]

        bone.bone_transform = (
            bone.bind_matrix @ euler2mat(*np.deg2rad(rotation))) @ bone.inverse_bind_matrix


def orientation_to_matrix(orientation: List[float]) -> np.ndarray:
    rotation = euler2mat(*np.deg2rad(orientation))
    return np.vstack([np.hstack([rotation, np.zeros((3, 1))]),
                      np.array([0, 0, 0, 1])])


def get_lines(skeleton: Skeleton) -> List[List[List[float]]]:

    def get_lines_recursive(bone: Bone, lines: List[List[List[float]]], parent_transform: np.ndarray = np.eye(3), parent_tail: np.ndarray = np.array([0, 0, 0])):
        head_position = parent_tail

        assert bone.direction is not None

        transform = parent_transform @ bone.bone_transform

        direction = transform @ np.array(bone.direction)

        tail_position = head_position + direction * bone.length

        lines.append([head_position.tolist(),
                     tail_position.tolist()])

        for child in bone.children:
            get_lines_recursive(child, lines, transform, tail_position)

    lines: List[List[List[float]]] = []

    assert skeleton.root is not None and skeleton.root.position is not None and skeleton.root.orientation is not None

    root_transform = euler2mat(*np.deg2rad(skeleton.root.orientation))
    root_position = np.array(skeleton.root.position)
    for child in skeleton.root.children:
        get_lines_recursive(child, lines, root_transform, root_position)

    return lines


def get_bones_head_position(skeleton: Skeleton) -> dict[str, List[float]]:
    def get_lines_recursive(bone: Bone, lines: dict[str, List[float]], parent_transform: np.ndarray = np.eye(3), parent_tail: np.ndarray = np.array([0, 0, 0])):
        head_position = parent_tail

        assert bone.direction is not None

        transform = parent_transform @ bone.bone_transform

        direction = transform @ np.array(bone.direction)

        tail_position = head_position + direction * bone.length

        lines[bone.name] = head_position.tolist()

        for child in bone.children:
            get_lines_recursive(child, lines, transform, tail_position)

    lines: dict[str, List[float]] = {}

    assert skeleton.root is not None and skeleton.root.position is not None and skeleton.root.orientation is not None

    root_transform = euler2mat(*np.deg2rad(skeleton.root.orientation))
    root_position = np.array(skeleton.root.position)
    lines['root'] = skeleton.root.position
    for child in skeleton.root.children:
        get_lines_recursive(child, lines, root_transform, root_position)

    return lines


def plot_skeleton(skeleton: Skeleton, ax):
    assert isinstance(ax, Axes3D)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_xlim3d(-20, 20)
    # ax.set_ylim3d(-20, 20)
    # ax.set_zlim3d(-20, 20)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    lines = get_lines(skeleton)

    for line in lines:
        xs, ys, zs = zip(*line)
        ax.plot(zs, xs, ys, "r", linewidth=3)

    for line in lines:
        xs, ys, zs = zip(*line)
        ax.plot(zs, xs, ys, "b.", markersize=5)


def is_same_skeleton(reference: Skeleton, other: Skeleton) -> bool:
    for name, structure in reference.bone_map.items():
        structure_other = other.bone_map.get(name)
        assert structure_other is not None

        if structure.limits is not None:
            assert structure_other.limits is not None
            if not np.allclose(structure.limits, structure_other.limits):
                print(structure.limits, structure_other.limits)
                return False

        if structure.axis is not None:
            assert structure_other.axis is not None
            if not np.allclose(np.cos(structure.axis), np.cos(structure_other.axis)):
                print(structure.axis, structure_other.axis)
                return False
    return True
