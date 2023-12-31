# BlenderCV

A collection of scripts used in [Blender](https://www.blender.org/) to generate various computer vision datasets, including motion-blur, rolling-shutter, event, etc.

## Install dependency

### 0. Clone this repo

```sh
git clone --recursive https://github.com/LingzheZhao/BlenderCV
```

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
  - For linux, e.g. Ubuntu, install dependencies with: `sudo apt install -y cmake build-essential libeigen3-dev`
  - For Windows, you can use [chocolatey](https://chocolatey.org/) to install [cmake](https://community.chocolatey.org/packages/cmake) and [Eigen](https://community.chocolatey.org/packages/eigen)
  - `cd dependency/spline`
  - `pip install .`

## Example workflow (generating motion-blur dataset)

### 0. Download
Download [dataset from Deblur-NeRF](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EqB3QrnNG5FMpGzENQq_hBMBSaCQiZXP7yGCVlBHIGuSVA?e=UaSQCC)

### 1. Extract trajectory from Blender
1. Open `tanabata.blend` with Blender
2. Select the `scripting` workspace in Blender, and drag the `export_keyframe_pose_tum_from_camera_animation.py` into Blender. Remember to change the output path to your own folder.
3. Hit Run (Alt+P) to run the script. You will get a TUM-formatted trajectory file `tanabata/groundtruth.txt`.

### 2. Interpolate
Interpolate the exported TUM-formatted trajectory with:

```sh
mkdir ../data/deblur_nerf/tanabata_blur
python generate_blur_trajectory_from_tum.py --input ../data/deblur_nerf/tanabata/groundtruth.txt --output ../data/deblur_nerf/tanabata_blur/groundtruth.txt --n_upsample 4
```

Then you will get the interpolated `tanabata_blur/groundtruth.txt` file.

### 3. Render

Drag and drop `render_by_tum_pose.py` into Blender. Change the `WORK_DIR` to the current working directory with your `tanabata_blur/groundtruth.txt` file.
Hit Run (Alt+P) to run the script. Now you will get the rendered images.

### 4. Generate blurry images

Now you can generate blurry images with the rendered virtual sharp images:

```sh
python images_avg.py /path/to/rendered/
```

> P.S. Note that the folder `rendered` should contain the rendered `raw` folder.
> 
> P.P.S. Also, you need to change `blur_num` in `images_avg.py` if you changed `N_VIRT_CAMS` in `generate_blur_trajectory_from_tum.py`.

## Acknowledgments

- `render_by_tum_pose.py` is derived from [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF/)
- Depth rendering is borrowed from [PyBlend](https://github.com/anyeZHY/PyBlend)
