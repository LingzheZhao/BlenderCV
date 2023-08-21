import bpy
from pathlib import Path

OUTPUT_DIR = Path("/home/lzzhao/data/ws_tencent-blender")
OUTPUT_FILE = OUTPUT_DIR / "gt_tabataba.txt"

OUTPUT_FILE.parent.mkdir(exist_ok=True)

scene = bpy.context.scene
cam = bpy.context.scene.objects.get('Camera.001', None)

def stringify_tum_pose(i, translation, quaternion):
    t = translation
    q = quaternion
    return f"{i} {t.x} {t.y} {t.z} {q.x} {q.y} {q.z} {q.w}\n"

with open (OUTPUT_FILE, "w") as f:
    f.write("#timestamp tx ty tz qx qy qz qw\n")
    for framei in range(scene.frame_end):
        cam.rotation_mode = 'XYZ'
        scene.frame_set(framei)
        cam.rotation_mode = 'QUATERNION'
        f.write(stringify_tum_pose(
            framei,
            cam.location,
            cam.rotation_quaternion,
        ))
