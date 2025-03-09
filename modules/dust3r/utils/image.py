# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(images, cog_seg_maps, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    # if isinstance(folder_or_list, str):
    #     if verbose:
    #         print(f'>> Loading images from {folder_or_list}')
    #     root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    # elif isinstance(folder_or_list, list):
    #     if verbose:
    #         print(f'>> Loading a list of {len(folder_or_list)} images')
    #     root, folder_content = '', folder_or_list

    # else:
    #     raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    # supported_images_extensions = ['.jpg', '.jpeg', '.png']
    # if heif_support_enabled:
    #     supported_images_extensions += ['.heic', '.heif']
    # supported_images_extensions = tuple(supported_images_extensions)
    pil_images = images.pil_images

    mean_colors = {}
    mean_colors_cnt = {}
    for i, img in enumerate(pil_images):
 
        img_np = np.array(img)
        seg_map = cog_seg_maps[i]
        unique_labels = np.unique(seg_map)
        for label in unique_labels:
            if label == -1:
                continue
            mask = (seg_map == label)
            mean_color = img_np[mask].mean(axis=0)
            if label in mean_colors.keys():
                mean_colors[label] += mean_color
                mean_colors_cnt[label] += 1
            else:
                mean_colors[label] = mean_color
                mean_colors_cnt[label] = 1
                
    for key in mean_colors.keys():
        mean_colors[key] /= mean_colors_cnt[key]

    imgs = []
    for i, img in enumerate(pil_images):
        img = pil_images[i]

        img_np = np.array(img)
        smoothed_image = np.zeros_like(img_np)
        seg_map = cog_seg_maps[i]
        unique_labels = np.unique(seg_map)
        for label in unique_labels:
            mask = (seg_map == label)
            if label == -1:
                smoothed_image[mask] = img_np[mask]
                continue
            smoothed_image[mask] = mean_colors[label]
        smoothed_image = cv2.addWeighted(img_np, 0.05, smoothed_image, 0.95, 0)
        smoothed_image = PIL.Image.fromarray(smoothed_image)

        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            smoothed_image = _resize_pil_image(smoothed_image, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
            smoothed_image = _resize_pil_image(smoothed_image, size)

        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
            smoothed_image = smoothed_image.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
            smoothed_image = smoothed_image.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        # W2, H2 = img.size
        # if verbose:
        #     print(f' - adding image {i} with resolution {W1}x{H1} --> {W2}x{H2}')

        imgs.append(dict(img=ImgNorm(img)[None], ori_img=ImgNorm(img)[None], smoothed_img=ImgNorm(smoothed_image)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs
