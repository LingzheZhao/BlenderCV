"""
Upsample / downsample TUM formated pose
"""
import argparse
from pathlib import Path
import spline
import math

def create_spline_trajectory(path_to_ctrl_knots: str):
    trajectory = spline.CubicSplineSE3()
    trajectory.loadFromFile(path_to_ctrl_knots)
    trajectory.setSplineDegK(2)
    trajectory.setStartTime(0)
    return trajectory

def save_poses(trajectory, path_to_file: Path, n_upsample: float):
    tstart = trajectory.getStartTime()
    dt = trajectory.getSamplingDuration()
    num_knots = trajectory.getNumCtrlKnots()

    dt_new = dt / n_upsample
    num_knots_new = math.ceil((num_knots - 1) * n_upsample)
    print(f"num_knots: {num_knots} to {num_knots_new}")

    with open(path_to_file, 'w') as f:
        f.write('#timestamp tx ty tz qx qy qz qw\n')
        for i in range(num_knots_new):
            t = tstart + dt_new * i
            print(f"knot: {i}, t = {t}")
            t_b2w = trajectory.getTranslation(t)
            R_b2w = trajectory.getRotation(t)
            f.write('%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n' % (
                t, t_b2w[0], t_b2w[1], t_b2w[2], R_b2w[1], R_b2w[2], R_b2w[3], R_b2w[0]
            ))

def runner(input: Path, output: Path, n_upsample: float):
    assert input.exists()

    output.parent.mkdir(exist_ok=True)
    with open(output, 'w') as f:
        f.write("#timestamp tx ty tz qx qy qz qw\n")

    trajectory = create_spline_trajectory(str(input.resolve()))
    save_poses(trajectory, output, n_upsample)

def main():
    parser = argparse.ArgumentParser(
        description='Upsample / downsample TUM formatted trajectory.'
    )
    parser.add_argument('--input', type=str, default="groundtruth.txt", required=False)
    parser.add_argument('--output', type=str, default="groundtruth_interpolated.txt", required=False)
    parser.add_argument('--n_upsample', type=float, default=10.0, required=False)

    args = parser.parse_args()

    INPUT = Path(args.input)
    OUTPUT = Path(args.output)
    N_UPSAMPLE = args.n_upsample

    print(f"INPUT: {INPUT}")
    print(f"OUTPUT: {OUTPUT}")
    print(f"N_UPSAMPLE: {N_UPSAMPLE}")

    runner(INPUT, OUTPUT, N_UPSAMPLE)

if __name__ == "__main__":
    main()
