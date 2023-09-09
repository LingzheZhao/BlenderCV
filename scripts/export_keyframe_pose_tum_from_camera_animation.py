import bpy
from pathlib import Path

WORK_DIR = Path("/home/cvgluser/data/blender_ws/BlenderCV/data/deblur_nerf")
SEQUENCE = "tanabata"
CAMERA = "Camera"

OUTPUT_DIR = WORK_DIR / SEQUENCE
OUTPUT_FILE = OUTPUT_DIR / "groundtruth.txt"
OUTPUT_FILE.parent.mkdir(exist_ok=True)

scene = bpy.context.scene
cam = scene.objects.get(CAMERA, None)

def stringify_tum_pose(i, translation, quaternion):
    t = translation
    q = quaternion
    return f"{i} {t.x} {t.y} {t.z} {q.x} {q.y} {q.z} {q.w}\n"

with open (OUTPUT_FILE, "w") as f:
    f.write("#timestamp tx ty tz qx qy qz qw\n")
    for framei in range(scene.frame_end):
        scene.frame_set(framei)
        f.write(stringify_tum_pose(
            framei,
            cam.matrix_world.translation,
            cam.matrix_world.to_quaternion(),
        ))
