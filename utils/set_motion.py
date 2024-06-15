from utils.skeleton import Skeleton
from utils.custom_types import MotionFrame
from transforms3d.euler import euler2mat
import numpy as np


def set_motion(skeleton: Skeleton, motion: MotionFrame):
    root_motion = motion.get("root")
    assert root_motion is not None
    assert skeleton.root is not None
    skeleton.root.position = (root_motion[0], root_motion[1], root_motion[2])
    skeleton.root.orientation = (root_motion[3], root_motion[4], root_motion[5])
    
    for [name, bone] in skeleton.bone_map.items():
        if name == "root":
            continue
        bone_motion = motion.get(name)

        if bone_motion is None:
            continue

        rotation: list[float] = [0, 0, 0]

        for i, dof in enumerate(bone.dof):
            rotation[dof] = bone_motion[i]

  
        bone.bone_transform = (bone.bind_matrix @ euler2mat(*np.deg2rad(rotation))) @ bone.inverse_bind_matrix
