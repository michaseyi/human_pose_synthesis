import torch
from utils.motion import MotionFrame, Motion
from transforms3d.euler import euler2mat
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
    for (prev_frame, current_frame) in zip(frames[:-1], frames[1:]):
        pose = []
        prev_position = torch.tensor(
            prev_frame['root'][:3], dtype=torch.float64)
        current_position = torch.tensor(
            current_frame['root'][:3], dtype=torch.float64)
        prev_rotation_mat = torch.tensor(
            euler2mat(*torch.tensor(prev_frame['root'][3:6]).deg2rad().tolist()))
        current_rotation_mat = torch.tensor(
            euler2mat(*torch.tensor(current_frame['root'][3:6]).deg2rad().tolist()))
        rotation_change_mat = current_rotation_mat @ prev_rotation_mat.inverse()
        position_change = prev_rotation_mat @ (
            current_position - prev_position)

        pose.append(position_change)
        pose.append(rotation_change_mat.view(-1))

        for bone in bone_sequence:
            rotation = torch.zeros(3)
            for axis, angle in zip(default_skeleton.bone_map[bone].dof, current_frame[bone]):
                rotation[axis] = angle
            rotation = rotation.deg2rad()
            matrix = torch.tensor(euler2mat(*(rotation.tolist())))
            pose.append(matrix.view(-1))
        data.append(torch.cat(pose))
    return torch.stack(data)


def tensor_to_motion_frames(tensor: torch.Tensor, start_position: list[float], start_rotation: list[float]) -> list[MotionFrame]:
    frames = []

    return frames


# original_orientations = torch.deg2rad(torch.tensor(list(map(lambda frame: frame['root'][3:6], motion_file.frames))))
# original_positions = torch.tensor(list(map(lambda frame: frame['root'][:3], motion_file.frames)), dtype=torch.float64)
# computed_orientations = [original_orientations[0]]
# computed_positions = [original_positions[0]]

# changes = []
# for (prev_orientation, current_orientation, prev_position, current_position) in zip(original_orientations[:-1], original_orientations[1:], original_positions[:-1], original_positions[1:]):
#     m1 = torch.tensor(euler2mat(*prev_orientation.tolist()))
#     m2 = torch.tensor(euler2mat(*current_orientation.tolist()))
#     orientation_change = m2 @ m1.inverse()
#     position_change = current_position - prev_position
#     changes.append((orientation_change, m1 @ position_change))


# for (orientation_change, position_change) in changes:
#     prev_orientation = computed_orientations[-1]
#     prev_position = computed_positions[-1]
#     m1 = torch.tensor(euler2mat(*prev_orientation.tolist()))
#     m2 = orientation_change @ m1

#     computed_positions.append(prev_position + m1.inverse() @ position_change)
#     computed_orientations.append(torch.tensor(mat2euler(m2.numpy())))


# original_orientations = torch.stack([torch.tensor(euler2mat(*orientation.tolist())) for orientation in original_orientations])
# computed_orientations = torch.stack([torch.tensor(euler2mat(*orientation.tolist())) for orientation in computed_orientations])
# computed_positions = torch.stack(computed_positions)

# torch.allclose(original_orientations, computed_orientations), torch.allclose(original_positions, computed_positions)

# (original_orientations - computed_orientations).abs().max(), (original_positions - computed_positions).abs().max()

