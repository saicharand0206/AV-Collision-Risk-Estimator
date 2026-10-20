---
title: AV Collision Risk Estimator
emoji: 🚗
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🚗 Real-time Collision Risk Estimation for Autonomous Vehicles

**MT310 Autonomous Vehicles | Assignment II | IcfaiTech IFHE Hyderabad**

> Final Year Level Project — Sensor Fusion + ML + Computer Vision + ADAS Pipeline

---

## 🎯 What This System Does

This end-to-end ADAS (Advanced Driver Assistance System) prototype estimates **real-time collision risk** by fusing data from multiple autonomous vehicle sensors using machine learning and computer vision.

---

## 🏗️ System Architecture

```
INPUT SENSORS (13 channels)
├── LiDAR   → Obstacle distance
├── Radar   → Closing speed
├── Camera  → Lane offset + lead distance
├── GPS     → Ego speed
├── CAN Bus → Brake + Steering
└── IMU     → Acceleration + Yaw rate

FEATURE ENGINEERING (+6 features)
├── Kinetic Energy Proxy
├── Danger Zone Score
├── Required Deceleration
├── Visibility-Speed Risk
├── Lateral Instability
└── Brake Threat Ratio

ML PIPELINE
└── Random Forest (300 trees) / TensorFlow DNN
    ├── 3-class output: Safe | Medium | High Risk
    └── ~95% accuracy on 10,000-sample KITTI-calibrated dataset

COMPUTER VISION
├── Image: Lane detection (Hough) + Obstacle detection (Contours)
└── Video: Frame-by-frame risk timeline + Annotated output

GRADIO UI — 6 Tabs
├── ⚡ Live Risk Predictor (13 sensor sliders + 3 charts)
├── 🖼️ Image Analysis (dashcam CV pipeline)
├── 🎬 Video Analysis (frame-by-frame timeline)
├── 📊 EDA Dashboard (KITTI-calibrated dataset exploration)
├── 📈 Model Metrics (Confusion matrix, ROC, PR curves, Feature Importance)
└── ℹ️ About (architecture, references)
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Weighted F1 | ~95% |
| AUC-ROC (High Risk) | ~0.99 |
| Dataset Size | 10,000 samples |
| Classes | Safe / Medium Risk / High Risk |

---

## 🔬 Dataset

KITTI-inspired synthetic dataset with feature distributions calibrated against:
- **KITTI Vision Benchmark Suite** (Geiger et al., CVPR 2012)
- **nuScenes Dataset** (Caesar et al., CVPR 2020)
- **Waymo Open Dataset** (Sun et al., CVPR 2021)

---

## 📚 References

1. Geiger et al. (2012). *KITTI Vision Benchmark Suite*. CVPR 2012.
2. Caesar et al. (2020). *nuScenes: A multimodal dataset*. CVPR 2020.
3. Sun et al. (2021). *Waymo Open Dataset*. CVPR 2021.
4. Janai et al. (2020). *Computer Vision for Autonomous Vehicles*. ECCV 2020.
5. Gruyer et al. (2017). *Perception for autonomous driving*. IEEE IV 2017.
6. Breiman (2001). *Random Forests*. Machine Learning.
7. Goodfellow et al. (2016). *Deep Learning*. MIT Press.

---

*Built with Python 3.10 | OpenCV | Scikit-learn | Matplotlib | Gradio 4.x*
