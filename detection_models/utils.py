# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from PIL import Image


def load_image_as_array(image_path: Path) -> np.ndarray:
    """Loads an image into memory as a numpy array
    
    Args:
        image_path (pathlib.Path): the filepath at which the image is stored
    
    Returns:
        np.ndarray: an image loaded into memory as a numpy array in the RGB
            colorspace (height, width, 3)
    """

    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
        np.uint8)