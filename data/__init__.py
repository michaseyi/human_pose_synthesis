
from utils.parser import parse_asf
import pandas as pd
from dataclasses import dataclass
import re
import os

'''
    The bone names commented out are not captured as part of the mocap data
'''
bone_sequence = [
    "lowerback",
    "upperback",
    "thorax",
    "lowerneck",
    "upperneck",
    "head",
    "rclavicle",
    "rhumerus",
    "rradius",
    "rwrist",
    "rhand",
    #     "rfingers",
    #     "rthumb",
    "lclavicle",
    "lhumerus",
    "lradius",
    "lwrist",
    "lhand",
    #     "lfingers",
    #     "lthumb",
    "rfemur",
    "rtibia",
    "rfoot",
    "rtoes",
    "lfemur",
    "ltibia",
    "lfoot",
    "ltoes"
]

script_directory = os.path.dirname(os.path.abspath(__file__))

default_skeleton = parse_asf(os.path.join(script_directory, "default_skeleton.asf"))

raw_metadata = pd.read_html(os.path.join(script_directory, 'mocap-index.html'))[5]


@dataclass
class MotionInfo:
    frame_rate: int
    description: str


metadata: dict[str, dict[str, MotionInfo]] = {}

current_subject = None

for row in range(raw_metadata.shape[0]):
    subject = raw_metadata.iloc[row, 0]
    if isinstance(subject, str):
        match = re.search(r"Subject #(\d+)", subject)
        if match:
            current_subject = match.group(1)
            metadata[current_subject] = {}
            continue

    frame_rate = raw_metadata.iloc[row, 8]

    if isinstance(frame_rate, str) and frame_rate.isdigit():
        description = raw_metadata.iloc[row, 2]
        description = description if isinstance(description, str) else ""
        index = raw_metadata.iloc[row, 1]
        assert isinstance(current_subject, str)
        assert isinstance(index, str)
        metadata[current_subject][index] = MotionInfo(
            int(frame_rate), description)
