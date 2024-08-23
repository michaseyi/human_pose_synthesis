import torch
from utils.motion import MotionFrame, Motion
from transforms3d.euler import euler2mat, mat2euler
from data import bone_sequence, default_skeleton


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
        position = torch.tensor(frame['root'][:3])
        # pose.append(position.to(torch.float32))

        # rotation = torch.tensor(frame['root'][3:6])
        # rotation = rotation.deg2rad()

        # matrix = torch.tensor(euler2mat(*(rotation.tolist())))
        # pose.append(matrix.view(-1).to(torch.float32))

        for bone in bone_sequence:
            rotation = torch.zeros(3)
            for axis, angle in zip(default_skeleton.bone_map[bone].dof, frame[bone]):
                rotation[axis] = angle
            rotation = rotation.deg2rad()
            matrix = torch.tensor(euler2mat(*(rotation.tolist())))
            pose.append(matrix.view(-1).to(torch.float32))
        data.append(torch.cat(pose))
    return torch.stack(data)


def tensor_to_motion_frames(tensor: torch.Tensor) -> list[MotionFrame]:
    assert len(tensor.shape) == 2
    frames = []

    for i in range(tensor.size(0)):
        frame = {}

        pose = tensor[i]

        # position = torch.tensor([0] * 3)
        # rotation = pose[:9].view(3, 3)

        # frame['root'] = position.tolist(
        # ) + torch.tensor(mat2euler(rotation.numpy())).rad2deg().tolist()
        frame['root'] = [0] * 6

        frame['rfingers'] = [7.12502]
        frame['lfingers'] = [7.12502]
        frame['rthumb'] = [15.531, -12.8745]
        frame['lthumb'] = [24.8333, 61.5281]

        for (index, bone) in zip(range(0, len(pose), 9), bone_sequence):
            rotation = pose[index:index + 9].view(3, 3)
            euler = torch.tensor(
                mat2euler(rotation.numpy())).rad2deg().tolist()
            dof = default_skeleton.bone_map[bone].dof
            frame[bone] = [euler[axis] for axis in dof]

        frames.append(frame)

    return frames
