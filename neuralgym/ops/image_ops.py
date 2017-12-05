"""image related ops."""
import math

import numpy as np
import cv2
import tensorflow as tf


def np_scale_to_shape(image, shape, align=True):
    """Scale the image.

    The minimum side of height or width will be scaled to or
    larger than shape.

    Args:
        image: numpy image, 2d or 3d
        shape: (height, width)

    Returns:
        numpy image
    """
    height, width = shape
    imgh, imgw = image.shape[0:2]
    if imgh < height or imgw < width or align:
        scale = np.maximum(height/imgh, width/imgw)
        image = cv2.resize(
            image,
            (math.ceil(imgh*scale), math.ceil(imgw*scale)))
    return image


def np_random_crop(image, shape, align=True):
    """Random crop.

    shape from image.

    Args:
        image: numpy image, 2d or 3d
        shape: (height, width)

    Returns:
        numpy image
    """
    height, width = shape
    image = np_scale_to_shape(image, shape, align=align)
    imgh, imgw = image.shape[0:2]
    h = np.random.randint(imgh-height+1)
    w = np.random.randint(imgw-width+1)
    return image[h:h+height, w:w+width, :]
