from typing import List, Optional

from utils.custom_types import MotionFrame


class Motion:
    frame_rate: int
    frames: List[MotionFrame]

    def __init__(self):
        self.frames = []
        self.frame_rate = 120


def parse_motion_file(file_path: str) -> Motion:
    with open(file_path, "r") as file:
        current_motion_frame: Optional[MotionFrame] = None
        motion = Motion()
        for line in file.readlines():
            line = line.strip()

            if line.isnumeric():
                if current_motion_frame is not None:
                    motion.frames.append(current_motion_frame)
                current_motion_frame = {}
                continue

            if current_motion_frame is not None:
                tokens = line.split(" ")

                assert len(tokens) > 1

                current_motion_frame[tokens[0]] = [
                    float(token) for token in tokens[1:]]

        if current_motion_frame:
            motion.frames.append(current_motion_frame)
    return motion

