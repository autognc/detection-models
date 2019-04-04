# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from object_detection.utils import visualization_utils as vis_util


class DetectionResults(OrderedDict):
    def overlay_all_on_image(self,
                             image: np.ndarray,
                             inplace=True,
                             score_threshold: float = 0.5,
                             max_detections: int = 20):
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
            pass


class DetectedObject(ABC):
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
    def __init__(self, label: str, confidence: float, box: np.ndarray) -> None:
        super().__init__(label, confidence)
        self.ymin = box[0]
        self.xmin = box[1]
        self.ymax = box[2]
        self.xmax = box[3]

    def overlay_on_image(self, image: np.ndarray, inplace=True) -> np.ndarray:
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
        return ("label: {}\nconfidence: {:.4}\ndimensions:\n\t".format(
            self.label, self.confidence) + "\n\t".join([
                "{}: {:.4}".format(l, v)
                for l, v in zip(["ymin", "xmin", "ymax", "xmax"],
                                [self.ymin, self.xmin, self.ymax, self.xmax])
            ]))
