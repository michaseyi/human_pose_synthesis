import os
import torch
from data import metadata, MotionInfo
from utils.parser import parse_motion_file
from utils.preprocessing import downsample_motion, motion_frames_to_tensor, smooth_motion


def is_amc_file(file: str) -> bool:
    return file.endswith('.amc')


def generate_dataset(subjects_dir: str, block_size: int, frame_rate: int, window_size: int = 5):
    dataset = []
    for root, dirs, files in os.walk(subjects_dir):
        subject = str(int(os.path.basename(root))) if os.path.basename(
            root).isdigit() else root

        for file in files:
            if not is_amc_file(file):
                continue

            index = str(int(file.split(".")[0].split("_")[1]))

            # frame rate defaults to 120 if not present in metadata
            info = MotionInfo(120, "")

            if subject in metadata and index in metadata[subject]:
                info = metadata[subject][index]

            motion_data = parse_motion_file(os.path.join(root, file))
            motion_data.frame_rate = info.frame_rate

            motion_data = downsample_motion(motion_data, frame_rate)
            motion_data = smooth_motion(motion_data, window_size)
            if len(motion_data.frames) < block_size:
                continue

            motion_tensor = motion_frames_to_tensor(motion_data.frames)

            idx = torch.randint(0, (len(motion_tensor) - (block_size)) + 1,
                                (max(1, (len(motion_tensor) // block_size) * 3),))

            dataset.extend([motion_tensor[i: i + block_size] for i in idx])

    return torch.stack(dataset)
