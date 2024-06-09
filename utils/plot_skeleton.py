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
    return euler2mat(*np.deg2rad(orientation))


def get_lines(skeleton: Skeleton) -> List[List[Vec3]]:

    def get_lines_recursive(bone: Bone, lines: List[List[Vec3]]):
        # apply transformation
        parent_position = np.array(bone.position)

        assert parent_position is not None

        for child in bone.children:
            child_position = parent_position
            assert child.direction is not None
            assert child.orientation is not None

            direction = np.array(child.direction)
            orientation = orientation_to_matrix(child.orientation)

            child_position = parent_position + \
                (orientation @ (direction * child.length))

            child.position = numpy_to_vec3(child_position)
            lines.append([numpy_to_vec3(parent_position), child.position])

            get_lines_recursive(child, lines)

    lines: List[List[Vec3]] = []
    assert skeleton.root is not None
    get_lines_recursive(skeleton.root, lines)

    return lines


def plot_skeleton(skeleton: Skeleton, fig: Optional[Figure] = None):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    assert isinstance(ax, Axes3D)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = get_lines(skeleton)

    for line in lines:
        xs, ys, zs = zip(*line)
        ax.plot(xs, ys, zs)

    plt.show()
