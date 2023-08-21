# BlenderCV

A collection of scripts used in [Blender](https://www.blender.org/) to generate various computer vision datasets, including motion-blur, rolling-shutter, event, etc.

## Install dependency

### 1. [quaternions in numpy](https://quaternion.readthedocs.io/en/latest/)

```sh
conda install -c conda-forge quaternion
```

or

```sh
python -m pip install --upgrade --force-reinstall numpy-quaternion
```

### 2. spline

- You can download our pre-built python wheels from [here](https://github.com/WU-CVGL/spline/releases), and install it with `pip`, for example:
  - `wget https://github.com/WU-CVGL/spline/releases/download/1.0.4/spline-1.0.4-cp310-cp310-win_amd64.whl`
  - `pip install ./spline-1.0.4-cp310-cp310-win_amd64.whl`
  - P.S. Remember to change `cp310` to your current python version
- You can also build by yourself:
  - For Windows, you can use [chocolatey](https://chocolatey.org/) to install [cmake](https://community.chocolatey.org/packages/cmake) and [Eigen](https://community.chocolatey.org/packages/eigen)
  - `cd dependency/spline`
  - `pip install .`

## Example workflow (generating motion-blur dataset)

### Download
Download [dataset from Deblur-NeRF](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EqB3QrnNG5FMpGzENQq_hBMBSaCQiZXP7yGCVlBHIGuSVA?e=UaSQCC)

### Extract trajectory from Blender
1. Open `tanabata.blend` with Blender
2. Select the `scripting` workspace in Blender, and drag the `export_keyframe_pose_tum_from_camera_animation.py` into Blender. Remember to change the output path to your own folder.
3. Hit Run (Alt+P) to run the script. You will get a TUM-formatted trajectory file `gt_tanabata.txt`.

### Interpolate
Interpolate the exported TUM-formatted trajectory with:

```sh
python generate_blur_trajectory_from_tum.py --input gt_tanabata.txt --output gt_tanabata_blur.txt --n_upsample 4
```

Then you will get the interpolated `gt_tanabata_blur.txt` file.

### Render

Drag and drop `export_by_tum_pose.py` into Blender. Change the `WORK_DIR` to the current working directory with your `gt_tanabata_blur.txt` file.
Hit Run (Alt+P) to run the script. Now you will get the rendered images.
