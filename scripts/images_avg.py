"""
Batch average virtual sharp images to blurry images
@author: Wang Peng & Lingzhe Zhao
"""
import imageio.v2 as imageio
import numpy as np
import os
import sys
from PIL import Image
from joblib import Parallel, delayed

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def imread(f):
    if f.endswith("png"):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def load_imgs(path):
    imgfiles = [
        os.path.join(path, f)
        for f in sorted(os.listdir(path))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    imgs = Parallel(n_jobs=24, verbose=5)(
        delayed(lambda f: imread(f)[..., :3] / 255.0)(f) for f in imgfiles
    )
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)

    return imgs, imgfiles


def avg_image(path, skip, blur_num, with_sharp=False):
    # weight_raw = np.linspace(0.1, 0.4, 71)
    # weight_sum = np.sum(weight_raw)
    # weight = weight_raw/weight_sum
    raw_path = os.path.join(path, "raw")
    imgs, imgfiles = load_imgs(raw_path)
    imgs_list = []
    imgs_ = 0
    save_dir = os.path.join(path, "images_reblur")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(imgs.shape[0]):
        if with_sharp:
            if i % (blur_num * skip + 1) != 0:
                imgs_list.append(imgs[i])
            else:
                print(imgfiles[i])
        else:
            imgs_list.append(imgs[i])
    for i in range(len(imgs_list)):
        # imgs_ += imgs_list[i] * weight[i%blur_num]
        imgs_ += imgs_list[i]
        if (i + 1) % blur_num == 0:
            # img_blur = imgs_
            img_blur = imgs_ / blur_num
            # img_blur_array = Image.fromarray(img_blur)
            # img_blur_array.save(os.path.join(save_dir, 'blur_{:03d}.png'.format(i//blur_num)))
            img_blur8 = to8b(img_blur)
            imageio.imwrite(
                os.path.join(save_dir, "rgb_blur_{:03d}.png".format(i // blur_num)),
                img_blur8,
            )
            imgs_ = 0


def avg_image_part(path, skip, blur_num, avg_num=10, with_sharp=False):
    raw_path = os.path.join(path, "raw")
    imgs, img_files = load_imgs(raw_path)
    imgs_list = []
    imgs_ = 0
    save_dir = os.path.join(path, "blur_10")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(imgs.shape[0]):
        if with_sharp:
            if i % (blur_num * skip + 1) != 0:
                imgs_list.append(imgs[i])
            else:
                print(img_files[i])
        else:
            imgs_list.append(imgs[i])
    for i in range(len(imgs_list)):
        if i % avg_num == 0 and i != (blur_num - 1):
            imgs_ += imgs_list[i]
        if (i + 1) % blur_num == 0:
            # img_blur = imgs_/(blur_num//avg_num + 1)
            img_blur = imgs_ / (blur_num // avg_num)
            # img_blur_array = Image.fromarray(img_blur)
            # img_blur_array.save(os.path.join(save_dir, 'blur_{:03d}.png'.format(i//blur_num)))
            img_blur8 = to8b(img_blur)
            imageio.imwrite(
                os.path.join(save_dir, "rgb_blur_{:03d}.png".format(i // blur_num)),
                img_blur8,
            )
            imgs_ = 0


if __name__ == "__main__":
    path = sys.argv[1]
    avg_image(path, skip=999999999, blur_num=31, with_sharp=False)
