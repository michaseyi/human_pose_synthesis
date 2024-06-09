
from typing import Optional
from enum import Enum
from skeleton import Skeleton, Bone


class Section(Enum):
    Units = 1
    Root = 2
    Bonedata = 3
    Hierarchy = 4


def get_section(section_name: str) -> Optional[Section]:
    match section_name:
        case ":units":
            return Section.Units
        case ":root":
            return Section.Root
        case ":bonedata":
            return Section.Bonedata
        case ":hierarchy":
            return Section.Hierarchy
        case default:
            return None


def extract_units_data(line: str, skeleton: Skeleton):
    line_tokens = line.split(" ")

    match line_tokens[0]:
        case "mass":
            skeleton.mass = float(line_tokens[1])
        case "length":
            skeleton.length = float(line_tokens[1])
        case "angle":
            skeleton.angle = line_tokens[1]
    pass


def extract_root_data(line: str, skeleton: Skeleton):
    if skeleton.root is None:
        skeleton.root = Bone("root")
        skeleton.bone_map[skeleton.root.name] = skeleton.root

    line_tokens = line.split(" ")

    match line_tokens[0]:
        case "position":
            skeleton.root.position = (float(line_tokens[1]), float(
                line_tokens[2]), float(line_tokens[3]))
        case "orientation":
            skeleton.root.orientation = (float(line_tokens[1]), float(
                line_tokens[2]), float(line_tokens[3]))
            pass


def extract_heirarchy_data(line: str, skeleton: Skeleton):
    if "begin" in line or "end" in line:
        return

    line_tokens = line.split(" ")

    parent_bone = skeleton.bone_map.get(line_tokens[0])

    assert parent_bone is not None

    for child in line_tokens[1:]:
        child_bone = skeleton.bone_map.get(child)

        assert child_bone is not None

        child_bone.parent = parent_bone
        parent_bone.children.append(child_bone)


def extract_bone_data(line: str, skeleton: Skeleton, current_bone: Optional[Bone]) -> Optional[Bone]:

    line_tokens = line.split(" ")

    match line_tokens[0]:
        case "begin":
            current_bone = Bone("")
        case "name":
            assert current_bone is not None
            current_bone.name = line_tokens[1]
            skeleton.bone_map[current_bone.name] = current_bone
        case "direction":
            assert current_bone is not None
            current_bone.direction = (float(line_tokens[1]), float(
                line_tokens[2]), float(line_tokens[3]))

        case "length":
            assert current_bone is not None
            current_bone.length = float(line_tokens[1])

        case "axis":
            assert current_bone is not None
            current_bone.orientation = (float(line_tokens[1]), float(
                line_tokens[2]), float(line_tokens[3]))

        case "dof":
            assert current_bone is not None
            current_bone.dof = line_tokens[1:]

        case "limits":
            pass

        case "end":
            current_bone = None
    return current_bone


def parse_asf(file_path: str) -> Skeleton:

    skeleton = Skeleton()
    with open(file_path, 'r') as f:
        current_section: Optional[Section] = None
        current_bone: Optional[Bone] = None

        for line in f.readlines():
            line = line.strip()
            if line.startswith(":"):
                current_section = get_section(line)
                continue

            match current_section:
                case Section.Root:
                    extract_root_data(line, skeleton)

                case Section.Units:
                    extract_units_data(line, skeleton)

                case Section.Bonedata:
                    current_bone = extract_bone_data(
                        line, skeleton, current_bone)

                case Section.Hierarchy:
                    extract_heirarchy_data(line, skeleton)

    return skeleton
