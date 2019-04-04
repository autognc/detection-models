#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pytest

import detection_models
import detection_models.results
import detection_models.utils

TESTS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def detection_results():
    results = detection_models.results.DetectionResults()
    results["person"] = [
        detection_models.results.DetectedBBox(
            "person", 0.9601364135742188,
            np.array([0.56533706, 0.3948621, 0.592293, 0.40808806])),
        detection_models.results.DetectedBBox(
            "person", 0.9334038496017456,
            np.array([0.5651202, 0.05923552, 0.61881506, 0.07810305])),
    ]
    results["kite"] = [
        detection_models.results.DetectedBBox(
            "kite", 0.9470244646072388,
            [0.0890673, 0.43968114, 0.16936198, 0.49565384]),
        detection_models.results.DetectedBBox(
            "kite", 0.9046292304992676,
            [0.37981525, 0.3519255, 0.4022239, 0.36464125]),
    ]
    return results


def test_overlay_all_on_image(detection_results):
    sample_image_path = TESTS_DIR / "test_data" / "image.jpg"
    image = detection_models.utils.load_image_as_array(sample_image_path)
    _ = detection_results.overlay_all_on_image(image)


@pytest.fixture
def detected_bbox():
    return detection_models.results.DetectedBBox(
        "person", 0.9601364135742188,
        np.array([0.56533706, 0.3948621, 0.592293, 0.40808806]))


def test_bbox_overlay_on_image(detected_bbox):
    sample_image_path = TESTS_DIR / "test_data" / "image.jpg"
    image = detection_models.utils.load_image_as_array(sample_image_path)
    _ = detected_bbox.overlay_on_image(image)
