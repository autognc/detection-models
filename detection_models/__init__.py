# -*- coding: utf-8 -*-
"""Top-level package for detection-models."""

__author__ = """Gavin C. Martin"""
__email__ = 'gavinmartin@utexas.edu'
__version__ = '0.1.2'

import sys

try:
    import object_detection
except ModuleNotFoundError:
    raise ImportError(
        """ You must have the TensorFlow Object Detection API installed to use
            detection_models. Follow the instructions at:
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md"""
    )

from .object_detector import ObjectDetector
from .bbox_detector import BBoxDetector
