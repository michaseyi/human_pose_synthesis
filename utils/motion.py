from typing import List, Dict
from utils.skeleton import Skeleton
from matplotlib.axes import Axes

MotionFrame = Dict[str, List[float]]


class Motion:
    frame_rate: int
    frames: List[MotionFrame]

    def __init__(self):
        self.frames = []
        self.frame_rate = 120


def add_motion_frame(dst: MotionFrame, src: MotionFrame):
    for key in src.keys():
        dst[key] = [d + s for d, s in zip(dst[key], src[key])]


def divide_motion_frame(dst: MotionFrame, divisor: int):
    for key in dst.keys():
        dst[key] = [d / divisor for d in dst[key]]


def blank_motion_frame(sample_frame: MotionFrame) -> MotionFrame:
    return {key: [-1] * len(sample_frame[key]) for key in sample_frame.keys()}


def downsample_motion(raw_motion: Motion, target_frame_rate: int, use_average: bool = True) -> Motion:
    assert target_frame_rate > 0 and target_frame_rate <= raw_motion.frame_rate
    downsampled_motion = Motion()
    downsampled_motion.frame_rate = target_frame_rate

    downsample_factor = raw_motion.frame_rate // target_frame_rate

    for start_frame_index in range(0, len(raw_motion.frames), downsample_factor):
        motion_frame = blank_motion_frame(raw_motion.frames[start_frame_index])

        if use_average:
            total_frames = 0
            for frame_index in range(start_frame_index, min(start_frame_index + downsample_factor, len(raw_motion.frames))):
                total_frames += 1
                add_motion_frame(motion_frame, raw_motion.frames[frame_index])
                pass
            divide_motion_frame(motion_frame, total_frames)
        else:
            add_motion_frame(
                motion_frame, raw_motion.frames[start_frame_index])
        downsampled_motion.frames.append(motion_frame)
        pass

    return downsampled_motion


def smooth_motion(raw_motion: Motion, window_size: int) -> Motion:
    smoothed_motion = Motion()
    smoothed_motion.frame_rate = raw_motion.frame_rate

    for motion_frame_index, motion_frame in enumerate(raw_motion.frames):
        smoothed_frame = blank_motion_frame(motion_frame)
        total_frame_count = 0
        for frame in raw_motion.frames[max(0, motion_frame_index - window_size): motion_frame_index + 1]:
            total_frame_count += 1
            add_motion_frame(smoothed_frame, frame)
        divide_motion_frame(smoothed_frame, total_frame_count)
        smoothed_motion.frames.append(smoothed_frame)
    return smoothed_motion


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
