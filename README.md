# YOU-ONLY-LOOK-ONCE-YOLOV3-with-Streamlit

# YOLOv3 Object Detection with Streamlit Deployment

## Overview
This project implements real-time object detection using the YOLOv3 (You Only Look Once) model. The YOLOv3 model is pre-trained on the COCO dataset, capable of detecting various objects in images and videos. The model is deployed using Streamlit to provide a user-friendly interface for real-time detection.

## Features

- Real-time object detection on uploaded images.
- Detects multiple objects in images with bounding boxes and class labels.
- Interactive user interface with Streamlit.

## Installation

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you have the YOLOv3 pre-trained weights file. You can download the weights from [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights).

## Dependencies

The required libraries for this project are:

- `streamlit`: For creating the web app interface.
- `opencv-python`: For image and video processing.
- `tensorflow` or `keras`: For loading the YOLOv3 model.
- `numpy`: For numerical operations.

Install the required dependencies with:
```bash
pip install streamlit opencv-python tensorflow numpy
