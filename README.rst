
================
detection-models
================


.. image:: https://img.shields.io/pypi/v/detection_models.svg
        :target: https://pypi.python.org/pypi/detection_models

.. image:: https://img.shields.io/travis/gavincmartin/detection_models.svg
        :target: https://travis-ci.org/gavincmartin/detection_models

.. image:: https://readthedocs.org/projects/detection-models/badge/?version=latest
        :target: https://detection-models.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




This library abstracts out many of the low-level operations of the TensorFlow
Object Detection API and provides an object-oriented approach to detecting
objects with frozen inference graphs.

* Free software: MIT license
* Documentation: https://detection-models.readthedocs.io.
* Demo: https://github.com/autognc/detection-models/blob/master/detection_models_demo.ipynb

Installation
------------
This package is heavily dependent upon the TensorFlow Object Detection API. Unfortunately, the OD API is not packaged for install on PyPI and therefore cannot be installed automatically as a dependency when installing `detection-models` from PyPI. To successfully install and use this package, users must:

1. Install the TensorFlow Object Detection API (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

2. Install `detection-models` from PyPI.
.. code::
        pip install detection-models
