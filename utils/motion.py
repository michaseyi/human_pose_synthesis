from typing import List, Dict
from utils.skeleton import Skeleton
from matplotlib.axes import Axes

MotionFrame = Dict[str, List[float]]


class Motion:
    def __init__(self, frame_rate: int = 120):
        self.frames = [] 
        self.frame_rate = frame_rate



def plot_bone_motion_graph(bone_name: str, motion_frames: list[MotionFrame], skeleton: Skeleton, ax: Axes):
    plot_data = [[], [], []]  # rx, ry, rz
    for frame_index in motion_frames:
        bone_motion = frame_index.get(bone_name)
        bone_data = skeleton.bone_map.get(bone_name)

        assert bone_data is not None

        if bone_motion is None:
            break

        if bone_name == "root":
            for i, rot in enumerate(bone_motion[3:]):
                plot_data[i].append(rot)
        else:
            for i, dof in enumerate(bone_data.dof):
                plot_data[dof].append(bone_motion[i])

    ax.plot(range(len(plot_data[0])), plot_data[0],
            label="rx", color="red", linewidth=1)
    ax.plot(range(len(plot_data[1])), plot_data[1],
            label="ry", color="green", linewidth=1)
    ax.plot(range(len(plot_data[2])), plot_data[2],
            label="rz", color="blue", linewidth=1)
    ax.set_title('Motion Graph for ' + bone_name, fontsize=10)
    ax.set_xlabel('Frame Number', fontsize=9)
    ax.set_ylabel('Motion Value (Degrees)', fontsize=9)
    ax.legend()
    ax.grid(True)
