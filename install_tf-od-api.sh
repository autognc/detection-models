#!/usr/bin/env bash

mkdir .deps && cd .deps
git clone https://github.com/tensorflow/models.git

cd .deps/models/research
protoc object_detection/protos/*.proto --python_out=.
