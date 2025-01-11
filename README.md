# 🚗 Car Counter with YOLOv8 and SORT Tracker

This project utilizes YOLOv8 for real-time object detection and SORT (Simple Online and Realtime Tracking) for tracking and counting cars in a video. The system detects and counts the number of cars that cross a predefined line in the video, using both detection and tracking to maintain accuracy.

![Car Counter Demo](https://media.giphy.com/media/3o6ZtffBFeWzfrW7aI/giphy.gif)

## 🧑‍💻 Overview

This program processes a video file, detects cars using YOLOv8, and tracks them using the SORT tracker. It counts the number of cars that cross a defined counting line and displays the car count in real-time.

### Features:
- **Real-time Car Detection** using YOLOv8
- **Car Tracking** with the SORT algorithm
- **Car Count Display** in the video stream
- **Counting Line** to detect when cars cross a predefined position
- **High Detection Confidence Filter** for accurate results

## 🛠️ Requirements

- Python 3.x
- OpenCV
- YOLOv8 model weights (`yolov8l.pt`)
- SORT tracker library

### Install the necessary libraries:
```bash
pip install opencv-python ultralytics numpy sort
