"""
Convert pose list in transformation.json to TUM format
"""
import sys
import json
import argparse
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from typing import Dict, List


def load_timestamp(timestamp_file: Path):
    assert timestamp_file.exists()
    timestamps = []
    with open(timestamp_file, "r") as f:
        for line in f.readlines():
            timestamps.append(float(line))
    return timestamps


def parse_transform_json(json_data: Dict):
    poses_dict = {}
    for frame in json_data["frames"]:
        key = frame["colmap_im_id"]
        val = frame["transform_matrix"]
        poses_dict[key] = pose_to_SE3(val)

    return dict(sorted(poses_dict.items()))


def load_transform_json(json_file: Path):
    with open(json_file, "r") as f:
        json_data = json.loads(f.read())

    return parse_transform_json(json_data)


def pose_to_SE3(pose: List):
    return pp.mat2SE3(torch.FloatTensor(pose).cuda())


def write_tum_poses(file_path: Path, timestamp: float, poses: pp.SE3_type):
    poses_ = poses

    def stringify_pose(pose: pp.SE3_type):
        assert pose.ndim == 1
        return " ".join([str(x.item()) for x in pose.data])

    with open(file_path, "a+", buffering=1) as f:
        if poses_.ndim == 1:
            poses_ = poses_[None,]
        assert poses_.ndim == 2
        for pose_ in poses_:
            f.write(f"{timestamp} {stringify_pose(pose_)}\n")


def runner(
    input: Path, output: Path, timestamp_file: Path | None, timestamp_offset: int
):
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        f.write("#timestamp tx ty tz qx qy qz qw\n")

    pose_dict_ordered = load_transform_json(input)

    if timestamp_file is not None:
        n_frames = len(pose_dict_ordered)
        timestamps = load_timestamp(timestamp_file)
        # timestamps = timestamps[-n_frames:]
        timestamps = timestamps[27:]

        for ts, pose in zip(timestamps, pose_dict_ordered.values()):
            write_tum_poses(output, ts, pose)
    else:
        for index, pose in pose_dict_ordered.items():
            write_tum_poses(output, float(index), pose)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeRF's transforms.json to TUM formatted txt."
    )
    parser.add_argument("--input", type=str, default="transforms.json", required=False)
    parser.add_argument("--output", type=str, default="transforms.txt", required=False)
    parser.add_argument("--timestamps", type=str, required=False)
    parser.add_argument("--timestamp_offset", type=int, default=27, required=False)

    args = parser.parse_args()

    INPUT = Path(args.input)
    OUTPUT = Path(args.output)
    TIMESTAMPS = Path(args.timestamps) if args.timestamps is not None else None
    TIMESTAMP_OFFSET = args.timestamp_offset

    print(f"INPUT: {INPUT}")
    print(f"OUTPUT: {OUTPUT}")
    print(f"TIMESTAMPS: {TIMESTAMPS}")
    print(f"TIMESTAMP_OFFSET: {TIMESTAMP_OFFSET}")

    runner(INPUT, OUTPUT, TIMESTAMPS, TIMESTAMP_OFFSET)


if __name__ == "__main__":
    main()
