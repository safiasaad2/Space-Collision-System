---

## 🛰️ Space Collision Detection & Surveillance System

[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Project-purple)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

### 🎯 Overview

This project presents a complete computer vision pipeline for detecting and tracking space objects (Satellites & Space Rocks) and identifying potential collision risks in video sequences.

The system combines object detection, multi-object tracking, ID stabilization, and collision analysis into a unified real-time surveillance framework.

---

### 🚀 Features

✅ Object detection using fine-tuned YOLOv26l model on custom dataset
✅ Multi-object tracking using ByteTrack
✅ Custom Stable ID system to prevent ID switching
✅ Trajectory visualization for motion tracking
✅ Collision detection based on spatial proximity
✅ Alert system for potential collisions
✅ Automatic saving of collision frames

---

### 🧠 Pipeline

1. Data Collection

   * Space dataset from Roboflow: https://universe.roboflow.com/space-0zlim/space-uxe02/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
   * Additional videos from NASA Image and Video Library (converted to frames): https://images.nasa.gov/

2. Annotation

   * Initial auto-annotation using baseline model
   * Manual refinement using CVAT

3. Data Analysis

   * Dataset distribution analysis
   * Split strategy selection
   *Data Augmentation for Classes Balancing

4. Model Training

   * Baseline YOLO model
   * Fine-tuning on custom dataset
   * Data augmentation applied

5. Evaluation

   * Precision, Recall, mAP metrics

6. System Integration

   * Detection + Tracking
   * Stable ID Mapping
   * Collision Logic

---

### 📊 Model Performance

| Class      | Precision | Recall | mAP50 | mAP50-95 |
| ---------- | --------- | ------ | ----- | -------- |
| Satellite  | 0.735     | 0.738  | 0.807 | 0.547    |
| Space-Rock | 0.969     | 0.957  | 0.982 | 0.833    |

> Note: Performance may vary under different environmental conditions such as lighting, resolution, motion blur, and object occlusion.

---

### 💥 Collision Logic

The system detects potential collisions when the distance between a satellite and a space rock falls below a defined threshold.

---

### 🎥 Output Preview

<p align="center">
  <img src="assets/preview.gif" width="800"/>
</p>

---

### ⚙️ Requirements

```bash
pip install ultralytics opencv-python numpy
```

---

### ▶️ Usage

```bash
python collision.py --video input.mp4
```

---

### 📁 Outputs

* Annotated video with tracking & alerts
* Saved frames for collision events

---

### ⚠️ Notes

* Model weights are not included due to size

---

### 💡 Future Improvements

* Continuous model training with new space data 
* Improving tracking with ReID-based models (e.g., StrongSORT / DeepSORT++)
* Adding temporal prediction using motion models or lightweight transformers

---

### 💬 Credits

* YOLO (Ultralytics)
* ByteTrack
* CVAT
* Roboflow
* NASA Open Data

---

### 👩‍💻 Author

Safia Saad
AI Engineer | Computer Vision

---
