"""
Generate TUM formatted trajectory for blender to export blur sequence (with optional upsampling)
"""
import argparse
from pathlib import Path
import spline
import math
from typing import IO


def create_spline_trajectory(path_to_ctrl_knots: str):
    trajectory = spline.CubicSplineSE3()
    trajectory.loadFromFile(path_to_ctrl_knots)
    trajectory.setSplineDegK(2)
    trajectory.setStartTime(0)
    return trajectory


def trajectory_upsample(trajectory, n_upsample: float):
    tstart = trajectory.getStartTime()
    dt = trajectory.getSamplingDuration()
    num_knots = trajectory.getNumCtrlKnots()

    dt_new = dt / n_upsample
    num_knots_new = math.ceil((num_knots - 1) * n_upsample)
    print(f"Upsampling: num_knots: {num_knots} to {num_knots_new}")

    timestamps = []
    for i in range(num_knots_new):
        t = tstart + dt_new * i
        timestamps.append(t)
    return timestamps


def save_blur_image_poses(
    trajectory, capture_t: float, exposure_t: float, n_virt_cams: int, fstream: IO
):
    half_exposure = 0.5 * exposure_t
    start_time = capture_t - half_exposure
    end_time = capture_t + half_exposure
    dt = exposure_t / (n_virt_cams - 1 + 1e-8)

    for i in range(n_virt_cams):
        t = start_time + i * dt
        t_b2w = trajectory.getTranslation(t)
        R_b2w = trajectory.getRotation(t)
        fstream.write(
            "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n"
            % (t, t_b2w[0], t_b2w[1], t_b2w[2], R_b2w[0], R_b2w[1], R_b2w[2], R_b2w[3])
        )


def runner(
    input: Path, output: Path, exposure_time: float, n_virt_cams: int, n_upsample: float
):
    assert input.exists()

    output.parent.mkdir(exist_ok=True)
    trajectory = create_spline_trajectory(str(input.resolve()))
    timestamps = trajectory_upsample(trajectory, n_upsample)
    with open(output, "w") as f:
        f.write("#timestamp tx ty tz qx qy qz qw\n")
        for capture_t in timestamps:
            save_blur_image_poses(trajectory, capture_t, exposure_time, n_virt_cams, f)


def main():
    parser = argparse.ArgumentParser(
        description="Upsample / downsample TUM formatted trajectory."
    )
    parser.add_argument("--input", type=str, default="groundtruth.txt", required=False)
    parser.add_argument(
        "--output", type=str, default="groundtruth_blur.txt", required=False
    )
    parser.add_argument(
        "--exposure_time", type=float, default=0.0156288 * 5, required=False
    )
    parser.add_argument("--n_virt_cams", type=int, default=31, required=False)
    parser.add_argument("--n_upsample", type=float, default=4.0, required=False)

    args = parser.parse_args()

    INPUT = Path(args.input)
    OUTPUT = Path(args.output)
    EXPOSURE_TIME = args.exposure_time
    N_UPSAMPLE = args.n_upsample
    N_VIRT_CAMS = args.n_virt_cams

    print(f"INPUT: {INPUT}")
    print(f"OUTPUT: {OUTPUT}")
    print(f"EXPOSURE_TIME: {EXPOSURE_TIME}")
    print(f"N_VIRT_CAMS: {N_VIRT_CAMS}")
    print(f"N_UPSAMPLE: {N_UPSAMPLE}")

    runner(INPUT, OUTPUT, EXPOSURE_TIME, N_VIRT_CAMS, N_UPSAMPLE)


if __name__ == "__main__":
    main()
