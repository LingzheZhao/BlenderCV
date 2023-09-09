"""
A simple script that used in blender to render with TUM formatted poses.

Created by LingzheZhao (zhaolingzhe AT westlake.edu.cn)
Created Date: 2023-05-25
"""
import os
import binascii
import csv
import json
import bpy
from mathutils import Quaternion
import numpy as np

from typing import Tuple, List

Camera = "Camera"
RESOLUTION_X = 600  # MALI: change resolution
RESOLUTION_Y = 400
FORMAT = "PNG"
WORK_DIR_ROOT = "/home/cvgluser/data/blender_ws/BlenderCV/data/deblur_nerf"
SEQUENCE = "tanabata_blur"
WORK_DIR = os.path.join(WORK_DIR_ROOT, SEQUENCE)
OUT_PREFIX = os.path.join(WORK_DIR, "raw")
POSE_FILE = os.path.join(WORK_DIR, "groundtruth.txt")

# Scaling factor on translation of input TUM trajectory
SCALING_FACTOR = 1.0

# Enable jump here to quickly get a glimpse of the outputs
ENABLE_JUMP = False
JUMP_NUM = 500

# Common Settings
# print(f'Active Devices: {bpy.context.preferences.addons["cycles"].preferences.has_active_device()}')
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION_X
scene.render.resolution_y = RESOLUTION_Y

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Background
bpy.context.scene.render.dither_intensity = 0.0
# bpy.context.scene.render.film_transparent = False


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def has_utf8_bom(file_path):
    """
    Checks if the given file starts with a UTF8 BOM
    wikipedia.org/wiki/Byte_order_mark
    """
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 3:
        return False
    with open(file_path, "rb") as f:
        return not int(binascii.hexlify(f.read(3)), 16) ^ 0xEFBBBF


def csv_read_matrix(file_path: str, delim=",", comment_str="#") -> List[List[str]]:
    """
    directly parse a csv-like file into a matrix
    :param file_path: path of csv file (or file handle)
    :param delim: delimiter character
    :param comment_str: string indicating a comment line to ignore
    :return: 2D list with raw data (string)
    """
    if hasattr(file_path, "read"):  # if file handle
        generator = (line for line in file_path if not line.startswith(comment_str))
        reader = csv.reader(generator, delimiter=delim)
        mat = [row for row in reader]
    else:
        assert os.path.isfile(file_path)
        skip_3_bytes = has_utf8_bom(file_path)
        with open(file_path) as f:
            if skip_3_bytes:
                f.seek(3)
            generator = (line for line in f if not line.startswith(comment_str))
            reader = csv.reader(generator, delimiter=delim)
            mat = [row for row in reader]
    return mat


def read_tum_trajectory_file(file_path: str):
    """
    parses trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)
    :param file_path: the trajectory file path (or file handle)
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    error_msg = (
        "TUM trajectory files must have 8 entries per row "
        "and no trailing delimiter at the end of the rows (space)"
    )
    assert raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) != 8)
    mat = np.array(raw_mat).astype(float)
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:]  # n x 4
    xyz *= SCALING_FACTOR
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    print(f"Loaded {len(stamps)} stamps and poses from: {file_path}")
    return (stamps, xyz, quat)


scene.render.image_settings.file_format = "PNG"  # set output format to .png

# Create collection for objects not to render with background

cam = bpy.context.scene.objects.get(Camera, None)
cam.rotation_mode = "QUATERNION"
target = bpy.context.scene.objects.get("Target", None)
if cam is None or target is None:
    print("doesn't find object named 'Target' and 'Camera'")

scene.frame_set(0)
scene.camera = cam  # set rendering camera

if not os.path.exists(OUT_PREFIX):
    os.makedirs(OUT_PREFIX)

out_data = {
    "path": OUT_PREFIX,
    "fov": cam.data.angle,
    "w": RESOLUTION_X,
    "h": RESOLUTION_Y,
    "frames": [],
}

ts, translations, quat_wxyz = read_tum_trajectory_file(POSE_FILE)
N = len(ts)
assert len(translations) == N
assert len(quat_wxyz) == N

quat_blender = []
for q in quat_wxyz:
    quat_blender.append(Quaternion(q))

for framei in range(N):
    if ENABLE_JUMP and 0 != framei % JUMP_NUM:
        continue
    print(f"Rendering frame {framei}")
    new_cam = cam.copy()
    new_cam.animation_data_clear()
    scene.collection.objects.link(new_cam)
    scene.camera = new_cam  # set rendering camera

    new_cam.matrix_world = quat_blender[framei].normalized().to_matrix().to_4x4()
    new_cam.matrix_world.translation = translations[framei]
    frame_data = {"transform_matrix": listify_matrix(new_cam.matrix_world)}

    # render
    scene.render.filepath = OUT_PREFIX + f"/{framei:06d}"
    bpy.ops.render.render(write_still=True)  # render still
    bpy.data.objects.remove(new_cam, do_unlink=True)
    out_data["frames"].append(frame_data)

transforms_json = os.path.join(OUT_PREFIX, "transforms.json")
with open(transforms_json, "w") as out_file:
    json.dump(out_data, out_file, indent=4)
