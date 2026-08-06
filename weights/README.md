# weights/

This directory is where the gaze-estimation model files belong. It is empty in the public repository — the trained checkpoint and exported ONNX model are part of ongoing, unpublished research and are intentionally **not included**.

## What is not included

- The trained gaze checkpoint (e.g. `model_epoch_100.pth`)
- The exported gaze ONNX model (e.g. `gaze_model.onnx`)

## What you need to provide

To run any of the inference or tooling scripts in this repository, place your own compatible model file(s) in this directory:

```text
weights/model_epoch_100.pth   # PyTorch checkpoint
weights/gaze_model.onnx       # ONNX export, if using the ONNX comparison tool
```

## Expected model interface

A compatible model must satisfy:

- **Input**: tensor of shape `[1, 3, 224, 224]` (batch, RGB channels, height, width)
- **Output**: tensor of shape `[1, 2]` — index `0` is yaw, index `1` is pitch, both in radians

The backbone used by this project is a ResNet18 with its final fully-connected layer replaced by a 2-output linear layer (see [`model.py`](../model.py)).

## Public model

The face-detection model (YuNet) is public and already included at [`models/face_detection_yunet_2023mar.onnx`](../models/README.md) — no action needed for that one.
