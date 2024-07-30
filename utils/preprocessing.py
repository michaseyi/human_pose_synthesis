import torch
from utils.motion import MotionFrame
from transforms3d.euler import euler2mat
from data import bone_sequence, default_skeleton


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


# def to_target_frame_rate(motion: torch.Tensor, current_frame_rate: int, target_frame_rate, average: bool = False) -> torch.Tensor:
#     assert current_frame_rate >= target_frame_rate
#     if current_frame_rate == target_frame_rate:
#         return motion
#     factor = current_frame_rate // target_frame_rate
#     idx = torch.arange(0, len(motion), factor)
#     if average:
#         return torch.stack([motion[i: min(len(motion), i + factor)].mean(0)  for i in idx])
#     else: 
#         return motion[idx]