# -*- coding: utf-8 -*-

import numpy as np

import detection_models
import detection_models.results


class BBoxDetector(detection_models.ObjectDetector):
    """A class for representing bounding box detectors

    This class inherits from `detection_models.ObjectDetector` and abstracts
    out many of the underlying operations necessary for loading a TensorFlow
    Object Detection API model and performing object detection on a given
    image. This class is specifically made for models that supply bounding
    boxes for detected classes within an image.
    
    Attributes:
        _graph (tf.Graph): the TensorFlow Graph object that represents the
            frozen inference graph
        _category_index (dict): a dictionary that stores the model's ID->label
            associations from the input label map; the keys are the class IDs
            (stored as `int`s), and each value is a dict with:
                "id": the class ID
                "name": the class name
        _session (tf.Session): the running TensorFlow session that represents
            the connection between the Python runtime and underlying C++ engine
        _tensor_dict (dict): a dictionary that stores the tensor names (as
            `str` keys) and tf.Tensor objects for the given model that will
            be supplied as "fetches" to a tf.Session.run() call; these are the
            return values of the session run
        _image_tensor (tf.Tensor): the image tensor that constitutes the
            tf.Session.run() feed_dict when paired with input images
    """

    def detect(self, image: np.ndarray, detection_threshold: float = 0.5
               ) -> detection_models.results.DetectionResults:
        """Performs object detection on a given image
    
        Args:
            image (np.ndarray): an image loaded into memory as a numpy array in
                the RGB colorspace (height, width, 3)
            detection_threshold (float, optional): Defaults to 0.5. A threshold
                with which to discard detected objects that have a low
                detection score

        Returns:
            detection_models.results.DetectionResults: the set of prediction
                results for a given image; see 
                `detection_models.results.DetectionResults` for a description
                of this object type
        """

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

            # else, append the DetectedBBox to the list at that label since
            # multiple instances of that category have been found
            else:
                detection_results[label].append(detected_bbox)

        return detection_results
