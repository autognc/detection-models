# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

import detection_models
import detection_models.results


class BBoxDetector(detection_models.ObjectDetector):
    def detect(self, image: np.ndarray, detection_threshold: float = 0.5
               ) -> detection_models.results.DetectionResults:
        output_dict = self._session.run(
            fetches=self._tensor_dict,
            feed_dict={self._image_tensor: np.expand_dims(image, 0)})

        # get rid of extra dimensions, convert to appropriate types
        num_detections = int(output_dict['num_detections'][0])
        detection_classes = output_dict['detection_classes'][0].astype(
            np.uint8)[0:num_detections]
        detection_boxes = output_dict['detection_boxes'][0][0:num_detections]
        detection_scores = output_dict['detection_scores'][0][0:num_detections]

        detection_results = detection_models.results.DetectionResults()
        for (label_id, box, score) in zip(
                detection_classes,
                detection_boxes,
                detection_scores,
        ):
            if score < detection_threshold:
                break

            label = self._category_index[label_id]["name"]
            detected_bbox = detection_models.results.DetectedBBox(
                label=label, confidence=score, box=box)

            # if label does not yet exist in DetectionResults, add the
            # DetectedBBox as a list (of only one element) to DetectionResults
            # at that label
            if label not in detection_results.keys():
                detection_results[label] = [detected_bbox]

            # else, append the DetectedBboxto the list at that label since
            # multiple instances of that category have been found
            else:
                detection_results[label].append(detected_bbox)

        return detection_results
