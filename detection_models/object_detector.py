# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util

import detection_models.results


class ObjectDetector(ABC):
    def __init__(self, model_path: Path, label_map_path: Path):
        self._graph = self._load_graph(str(model_path.absolute()))
        self._category_index = label_map_util.create_category_index_from_labelmap(
            str(label_map_path.absolute()))
        self._session = tf.Session(graph=self._graph)
        self._tensor_dict = self._get_tensor_dict()
        self._image_tensor = self._graph.get_tensor_by_name("image_tensor:0")

    def _load_graph(self, model_path: Path) -> tf.Graph:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _get_tensor_dict(self) -> Dict[str, tf.Tensor]:
        ops = self._graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self._graph.get_tensor_by_name(tensor_name)
        return tensor_dict

    @abstractmethod
    def detect(self, image: np.ndarray, detection_threshold: float = 0.5
               ) -> detection_models.results.DetectionResults:
        pass
