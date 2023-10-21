# Smart Home Surveillance using Computer Vision

## Examples of how to use (YOLO model):

1. Webcam
```py main.py```
2. Images
```py main.py images/dog.jpg```
3. Videos
```py main.py videos/kitten.mp4```

## Supplementary

To use models other than YOLO, use the `nonyolo.py` instead of `main.py`. The format is the same.

Models available:
1. TensorFlow - object detection
2. Caffe - face detection

*uses TensorFlow by default

*to use the Caffe model, replace the `MODEL_TYPE` variable with `caffe` inside the `nonyolo.py` file
