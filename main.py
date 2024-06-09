from utils.asf_parser import parse_asf
from utils.plot_skeleton import plot_skeleton


def main():
    skeleton = parse_asf("mocap/02.asf")
    plot_skeleton(skeleton, None)


if __name__ == "__main__":
    main()
