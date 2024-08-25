import torch
from utils.motion import MotionFrame, Motion
from transforms3d.euler import mat2euler
from data import bone_sequence, default_skeleton, hierarchical_order


def add_motion_frame(dst: MotionFrame, src: MotionFrame):
    for key in src.keys():
        dst[key] = [d + s for d, s in zip(dst[key], src[key])]


def divide_motion_frame(dst: MotionFrame, divisor: int):
    for key in dst.keys():
        dst[key] = [d / divisor for d in dst[key]]


def blank_motion_frame(sample_frame: MotionFrame) -> MotionFrame:
    return {key: [-1] * len(sample_frame[key]) for key in sample_frame.keys()}


def downsample_motion(raw_motion: Motion, target_frame_rate: int) -> Motion:
    assert target_frame_rate > 0 and target_frame_rate <= raw_motion.frame_rate

    if target_frame_rate == raw_motion.frame_rate:
        return raw_motion

    downsample_factor = raw_motion.frame_rate // target_frame_rate

    motion = Motion(target_frame_rate)
    motion.frames = [raw_motion.frames[i]
                     for i in range(0, len(raw_motion.frames), downsample_factor)]
    return motion


def smooth_motion(raw_motion: Motion, window_size: int) -> Motion:
    smoothed_motion = Motion(frame_rate=raw_motion.frame_rate)

    for motion_frame_index, motion_frame in enumerate(raw_motion.frames):
        smoothed_frame = blank_motion_frame(motion_frame)
        total_frame_count = 0
        for frame in raw_motion.frames[max(0, motion_frame_index - window_size): motion_frame_index + 1]:
            total_frame_count += 1
            add_motion_frame(smoothed_frame, frame)
        divide_motion_frame(smoothed_frame, total_frame_count)
        smoothed_motion.frames.append(smoothed_frame)
    return smoothed_motion


def motion_frames_to_tensor(frames: list[MotionFrame]):
    data = []
    for frame in frames:
        pose = []
        for bone in hierarchical_order:
            rotation = torch.zeros(3)

            if bone == 'root':
                position = torch.tensor(frame['root'][:3])
                rotation = torch.tensor(frame['root'][3:]).deg2rad()
                pose.append(position)
                pose.append(rotation)
            elif bone in frame:
                for axis, angle in zip(default_skeleton.bone_map[bone].dof, frame[bone]):
                    rotation[axis] = angle
                if bone in ['rthumb', 'lthumb', 'rfingers', 'lfingers']:
                    rotation = torch.zeros(3)
                pose.append(rotation.deg2rad())
            else:
                pose.append(torch.zeros(3))
        data.append(torch.cat(pose).to(torch.float32))
    return torch.stack(data)



def tensor_to_motion_frames(tensor: torch.Tensor) -> list[MotionFrame]:
    assert len(tensor.shape) == 2
    frames = []

    for i in range(tensor.size(0)):
        frame = {}
        pose = tensor[i]

        step = 0
        for i in range(len(hierarchical_order)):
            bone = hierarchical_order[i]

            if bone == 'root':
                frame['root'] = [*pose[step:step + 3].tolist(), *pose[step + 3:step + 6].rad2deg()]
                step += 6
            else:
                rotation = pose[step:step + 3].rad2deg().tolist()
                step += 3
                frame[bone] = rotation
                dof = default_skeleton.bone_map[bone].dof
                rotation = [rotation[axis] for axis in dof]

        frame['rfingers'] = [7.12502]
        frame['lfingers'] = [7.12502]
        frame['rthumb'] = [15.531, -12.8745]
        frame['lthumb'] = [24.8333, 61.5281]

        frames.append(frame)

    return frames
