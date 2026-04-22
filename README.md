# 🛰️ Space Collision Detection & Surveillance System

[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Project-purple)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLO26-Ultralytics-orange)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 🎯 Overview

This project presents a complete end-to-end computer vision pipeline for detecting and tracking space objects — Satellites and Space Rocks — and identifying potential collision risks in video sequences.

The system combines object detection, multi-object tracking, stable ID assignment, trajectory visualization, and collision analysis into a unified real-time surveillance framework.

---

## 🚀 Features

- Object detection using fine-tuned **YOLO26 Large** (Ultralytics) on a custom space dataset
- Multi-object tracking using **ByteTrack**
- Custom **Stable ID Mapper** to prevent ID switching across frames
- **Trajectory visualization** with motion trails
- **Collision warning system** based on spatial proximity
- Automatic saving of collision alert frames

---

## 🧠 Pipeline

### 1. Data Collection
- Space imagery dataset from **Roboflow Universe**:
  [universe.roboflow.com/space-0zlim/space-uxe02](https://universe.roboflow.com/space-0zlim/space-uxe02)
- Real footage frames extracted from **NASA Image and Video Library**:
  [images.nasa.gov](https://images.nasa.gov/)

### 2. Annotation
- Auto-annotation using baseline model on NASA frames
- Manual refinement and correction using **CVAT**

### 3. Data Analysis & Preprocessing
- Class distribution analysis (4.8x imbalance detected)
- 80 / 10 / 10 train / val / test split
- Augmentation on train set
- Techniques: flip, HSV jitter, noise, random crop, cutout, mosaic

### 4. Model Training
- **Round 1:** Baseline YOLO26 Large on Roboflow dataset (used for auto-annotation)
- **Round 2:** Fine-tuning on combined dataset (Roboflow + NASA-annotated frames)

### 5. Evaluation
- Precision, Recall, mAP50, mAP50-95 on held-out test set

### 6. System Integration
- Real-time object detection + tracking using ByteTrack
- Stable ID mapping across frames
- Trajectory visualization with motion trails
- Collision distance monitoring and alerting

---

## 📊 Model Performance

| Class       | Precision | Recall | mAP50 | mAP50-95 |
|-------------|-----------|--------|-------|----------|
| All         | 0.852     | 0.847  | 0.894 | 0.69     |
| Satellite   | 0.735     | 0.738  | 0.807 | 0.547    |
| Space-Rock  | 0.969     | 0.957  | 0.982 | 0.833    |

> Performance may vary depending on video resolution, lighting, motion blur, and object scale.

---

## 💥 Collision Logic

The system raises a warning when the Euclidean distance between a detected Satellite and a Space-Rock falls below a configurable threshold. Alert frames are automatically saved for review.

---

## 🎥 Output Preview

<p align="center">
  <img src="assets/preview.gif" width="800"/>
</p>

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python src/collision.py
```

Edit the top of `collision.py` to set your paths:

```python
MODEL_PATH = "your_model.pt"
VIDEO_PATH = "your_video.mp4"
```

---

## 📁 Project Structure

```
Space-Collision-System/
│
├── src/
│   └── collision.py         # Detection, tracking & collision surveillance
│
├── outputs/
│   └── demo.mp4             # Demo preview
│   └── alert_frames/        # Saved frames for collision events
│
├── assets/
│   └── preview.gif          # Demo preview
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚠️ Notes

- Model weights are not included due to file size. Training scripts and data pipeline are available upon request.
- Works best with space footage where objects appear against a dark background.

---

## 💡 Future Improvements

- Continuous retraining with new space footage
- ReID-based tracking (StrongSORT / DeepSORT) for more stable IDs
- Temporal trajectory prediction using motion models or lightweight transformers
- Expanded dataset with more samples

---

## 💬 Credits

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Roboflow Universe](https://universe.roboflow.com/space-0zlim/space-uxe02)
- [NASA Image and Video Library](https://images.nasa.gov/)
- [CVAT](https://www.cvat.ai/)
- ByteTrack

---

## 👩‍💻 Author

**Safia Saad**
AI Engineer | Computer Vision & Deep Learning
