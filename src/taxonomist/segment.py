"""Defines functions for handling segmentation masks and bounding boxes"""

import numpy as np
import cv2
import pycocotools
import matplotlib.pyplot as plt

from PIL import Image
from pycocotools import mask as mask_utils
from skimage.filters import gaussian, threshold_triangle, unsharp_mask
from skimage import exposure
from skimage.color import rgb2hsv
from scipy import ndimage as ndi
from tqdm import tqdm


def get_bbox(mask):
    """Calculates a bounding box for a binary mask. The bounding box is calculated for the maximum blob"""
    mask = np.array(mask)
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def segment(img, c):
    """A basic segmentation pipeline of sharpening, histogram eq, blur, thresholding, hole filling and maximum contour finding"""
    if c == "hsv":
        B = rgb2hsv(np.array(img))[:, :, 2]
    else:
        B = np.array(img)[:, :, c]

    B = unsharp_mask(B, radius=2, amount=3)

    B = exposure.equalize_adapthist(B, clip_limit=0.01)

    # Blur
    B_blur = gaussian(B, sigma=1)

    # Threshold
    thresh = threshold_triangle(B_blur)
    mask = B_blur < thresh
    if mask.sum().sum() / mask.size > 0.5:
        mask = B_blur > thresh

    # Fill holes
    fill = ndi.binary_fill_holes(mask)

    # Find contours
    cnts, _ = cv2.findContours(
        fill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cnt = max(cnts, key=cv2.contourArea)

    # Select max contour
    out = np.zeros(fill.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(fill.astype(np.uint8), out)

    # Find bounding box
    x, y, w, h = get_bbox(out)
    return out, (x, y, w, h)


def segment_multi(img):
    """Combines segmetations for a rgb image by their union"""
    out_f = np.zeros_like(np.array(img)[:, :, 0]).astype(bool)
    for c in [0, 1, 2]:
        out, _ = segment(img, 0)
        out_f = (out_f | out.astype(bool)).astype(bool)
    bbox_f = get_bbox(out_f)
    return out_f, bbox_f


def does_bbox_hit_border(bbox, img):
    """Returns true if the bounding box hits any of the four borders of img"""
    bbox = xywh2xyxy(bbox)
    if bbox[0] == 0:
        return True
    elif bbox[1] == 0:
        return True
    elif bbox[2] == img.shape[0]:
        return True
    elif bbox[3] == img.shape[1]:
        return True
    return False


def add_bbox(img, bbox):
    """Adds a xyxy bounding box to an image"""
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    rec = cv2.rectangle(img, bbox, (255, 0, 0), 1)
    return rec


def xywh2xyxy(bbox):
    """Converts XYWH bounding box to an XYXY bounding box"""
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def xyxy2xywh(bbox):
    """Converts XYXY bounding box to an XYWH bounding box"""
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


def coco_fname_anns(coco: pycocotools.coco.COCO, fname):
    """Takes a pytotocools.coco"""
    imgs = []
    for img in coco.imgs.values():
        if fname == img["file_name"]:
            imgs.append(img)
    return imgs


def show_mask(img, mask, ax):
    """Shows a binary mask on an image
    https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
    """

    ax.imshow(np.array(img))
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.axis("off")


def show_ann(coco: pycocotools.coco.COCO, fname):
    """Displays COCO annotations (mask, bbox), of a filename"""
    img = Image.open(fname)
    img_coco = coco_fname_anns(coco, fname.name)[0]
    ann = coco.loadAnns(coco.getAnnIds(imgIds=[img_coco["id"]]))[0]
    mask = mask_utils.decode(ann["segmentation"])
    bbox = ann["bbox"]

    ax = plt.gca()
    show_mask(add_bbox(img, bbox), mask, ax)
