# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from object_detection.utils import visualization_utils as vis_util


class DetectionResults(OrderedDict):
    """Stores the results of object detection and provides utility functions
    
    `DetectionResults` is a subclass of the `collections.OrderedDict` class.
    The dictionary is keyed with the detected labels (from the label map)
    ordered in descending order of labels with the highest detection score. The
    value at each key is a list of all of the instances of the detected label
    within the image. Each instance in the list is an object subclassed from
    `DetectedObject`.
    """

    def overlay_all_on_image(self,
                             image: np.ndarray,
                             inplace=True,
                             score_threshold: float = 0.5,
                             max_detections: int = 20) -> np.ndarray:
        """Overlays deteced objects and their associated visuals on an image

        Args:
            image (np.ndarray): the image on which to overlay results; loaded
                into memory as a numpy array in the RGB colorspace
                (height, width, 3)
            inplace (bool, optional): Defaults to True. Whether to modify the
                input image directly or make a copy before adding visualized
                results. The function will return an image with overlaid
                results either way.
            score_threshold (float, optional): Defaults to 0.5. A threshold
                with to ignore the visualization of low-confidence detections.
            max_detections (int, optional): Defaults to 20. The maximum number
                of detected objects to display on the image.

        Returns:
            np.ndarray: the image with the associated visuals overlaid
        """

        if not inplace:
            image = image.copy()
        all_detected_objects = []
        for detections in self.values():
            all_detected_objects.extend(detections)
        all_detected_objects.sort(key=lambda obj: obj.confidence, reverse=True)

        # construct a category index (this does not necessarily match that
        # of the detection model; we can just construct an equivalent mapping
        # such that the visualization utils associate the proper strings with
        # each detected object), and construct a dict for quick lookups in the
        # subsequent section
        category_index = {}
        label_to_int = {}
        for i, label in enumerate(self.keys()):
            category_index[i + 1] = {"id": i + 1, "name": label}
            label_to_int[label] = i + 1

        # TODO: handle empty list
        if isinstance(all_detected_objects[0], DetectedBBox):
            boxes = np.zeros((len(all_detected_objects), 4), dtype=np.float32)
            classes = np.zeros((len(all_detected_objects), ), dtype=np.uint8)
            scores = np.zeros((len(all_detected_objects), ), dtype=np.float32)

            for i, obj in enumerate(all_detected_objects):
                classes[i] = label_to_int[obj.label]
                scores[i] = obj.confidence
                boxes[i] = np.array([obj.ymin, obj.xmin, obj.ymax, obj.xmax])

            return vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_detections,
                min_score_thresh=score_threshold,
            )
        else:
            return image


class DetectedObject(ABC):
    """An abstract base class for representing objects detected in an image

    Attributes:
        label (str): the label (class name) for the detected object
        confidence (float): the detection score for the detected object
    """

    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

    @abstractmethod
    def overlay_on_image(self, image: np.ndarray, inplace=True) -> np.ndarray:
        pass

    @abstractmethod
    def denormalize(self, image_height: int, image_width: int):
        pass


class DetectedBBox(DetectedObject):
    """Represents detected objects and their bounding boxes in an image

    Attributes:
        label (str): the label (class name) for the detected object
        confidence (float): the detection score for the detected object
        ymin (float): the top boundary of the bounding box in normalized
            pixel coordinates
        xmin (float): the left boundary of the bounding box in normalized
            pixel coordinates
        ymin (float): the bottom boundary of the bounding box in normalized
            pixel coordinates
        xmax (float): the right boundary of the bounding box in normalized
            pixel coordinates
    """

    def __init__(self, label: str, confidence: float, box: np.ndarray) -> None:
        super().__init__(label, confidence)
        self.ymin = box[0]
        self.xmin = box[1]
        self.ymax = box[2]
        self.xmax = box[3]

    def overlay_on_image(self, image: np.ndarray, inplace=True) -> np.ndarray:
        """Overlays the bounding box, class label, and score on an image

        Args:
            image (np.ndarray): the image on which to overlay results; loaded
                into memory as a numpy array in the RGB colorspace
                (height, width, 3)
            inplace (bool, optional): Defaults to True. Whether to modify the
                input image directly or make a copy before adding visualized
                results. The function will return an image with overlaid
                results either way.

        Returns:
            np.ndarray: the image with the box, label, and score overlaid
        """

        if not inplace:
            image = image.copy()
        return vis_util.visualize_boxes_and_labels_on_image_array(
            image=image,
            boxes=np.array(
                [[self.ymin, self.xmin, self.ymax, self.xmax]],
                dtype=np.float32),
            classes=np.array([1], dtype=np.uint8),
            scores=np.array([self.confidence], dtype=np.float32),
            category_index={1: {
                "id": 1,
                "name": self.label
            }},
            use_normalized_coordinates=True,
            line_thickness=8,
        )

    def denormalize(self, image_height: int, image_width: int):
        """Converts this objects normalized coordinates into pixel coordinates

        Args:
            image_height (int): the height of the image in pixels
            image_width (int): the width of the image in pixels

        Returns:
            dict: a dictionary with the following keys and values:
                ymin (float): the top boundary of the bounding box in pixel
                    coordinates
                xmin (float): the left boundary of the bounding box in pixel
                    coordinates
                ymin (float): the bottom boundary of the bounding box in pixel
                    coordinates
                xmax (float): the right boundary of the bounding box in pixel
                    coordinates
        """

        return {
            "ymin": int(self.ymin * image_height),
            "xmin": int(self.xmin * image_width),
            "ymax": int(self.ymax * image_height),
            "xmax": int(self.xmax * image_width),
        }

    def __repr__(self) -> str:
        return (
            "detection_models.results.DetectedBBox({label}, {confidence}, {box})".
            format(
                label=self.label,
                confidence=self.confidence,
                box=np.array([self.ymin, self.xmin, self.ymax, self.xmax])))

    def __str__(self) -> str:
        return ("label: {} | confidence: {:.4f} | ".format(
            self.label, self.confidence) + " | ".join([
                "{}: {:.4f}".format(l, v)
                for l, v in zip(["ymin", "xmin", "ymax", "xmax"],
                                [self.ymin, self.xmin, self.ymax, self.xmax])
            ]))
