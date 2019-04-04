#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `detection_models` package."""

import os
from pathlib import Path

import pytest

import detection_models
import detection_models.utils

TESTS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def model():
    test_data_dir = TESTS_DIR / "test_data"
    return detection_models.BBoxDetector(
        model_path=test_data_dir / "inception_graph_boxes.pb",
        label_map_path=test_data_dir / "mscoco_label_map.pbtxt")


def test_detect(model):
    sample_image_path = TESTS_DIR / "test_data" / "image.jpg"
    image = detection_models.utils.load_image_as_array(sample_image_path)
    _ = model.detect(image)
