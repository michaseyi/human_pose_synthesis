from utils.skeleton import Skeleton, Bone
from typing import List, Optional
from utils.custom_types import Vec3

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2mat


def numpy_to_vec3(np_array: np.ndarray) -> Vec3:
    return (np_array[0], np_array[1], np_array[2])


def orientation_to_matrix(orientation: Vec3) -> np.ndarray:
    rotation = euler2mat(*np.deg2rad(orientation))
    return np.vstack([np.hstack([rotation, np.zeros((3, 1))]),
                      np.array([0, 0, 0, 1])])



def get_lines(skeleton: Skeleton) -> List[List[Vec3]]:

    def get_lines_recursive(bone: Bone, lines: List[List[Vec3]], parent_transform: np.ndarray = np.eye(3), parent_tail: np.ndarray = np.array([0, 0, 0])):
        head_position = parent_tail

        assert bone.direction is not None

        transform = parent_transform @ bone.bone_transform

        direction = transform @ np.array(bone.direction)

        tail_position = head_position + direction * bone.length

        lines.append([numpy_to_vec3(head_position),
                     numpy_to_vec3(tail_position)])

        for child in bone.children:
            get_lines_recursive(child, lines, transform, tail_position)

    lines: List[List[Vec3]] = []

    assert skeleton.root is not None and skeleton.root.position is not None and skeleton.root.orientation is not None

    root_transform = euler2mat(*np.deg2rad(skeleton.root.orientation))
    root_position = np.array(skeleton.root.position)
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
        ax.plot(zs, xs, ys, "b." , markersize=5)
