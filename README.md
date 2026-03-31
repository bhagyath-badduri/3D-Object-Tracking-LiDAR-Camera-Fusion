# SFND 3D Object Tracking

This project implements a complete 3D object tracking pipeline using camera and LiDAR data as part of the Udacity Sensor Fusion Nanodegree.

The system combines 2D object detection using a deep learning model with LiDAR point cloud processing to track vehicles over time and estimate Time-to-Collision (TTC) using both sensors.

---

## Project Overview

The pipeline performs the following steps:

- Object detection in camera images using the YOLO deep learning framework  
- Association of LiDAR points with camera bounding boxes  
- Tracking of detected objects across successive frames  
- Estimation of Time-to-Collision (TTC) using both LiDAR and camera data  

The project follows the complete perception pipeline taught in the course and integrates both geometric and feature-based techniques for robust object tracking.

---

## Implemented Tasks

All required final project tasks have been successfully completed:

### FP.1 – Match 3D Objects  
Matched 3D bounding boxes across successive frames by counting keypoint correspondences between bounding boxes and selecting the pairing with the highest number of matches.

### FP.2 – Compute LiDAR-based TTC  
LiDAR-based TTC was computed using the change in distance to the object between successive frames. To improve robustness, outlier LiDAR points were filtered before estimating the distance.

### FP.3 – Associate Keypoint Correspondences with Bounding Boxes  
Keypoint matches were filtered by checking whether they lie inside the bounding box region of interest. Outlier matches were removed based on distance statistics.

### FP.4 – Compute Camera-based TTC  
Camera-based TTC was computed using the distance ratio between matched keypoints across frames, assuming constant velocity and using the median ratio to improve robustness.

---

## FP.5 – Performance Evaluation 1 (LiDAR-based TTC)

During testing, several frames were observed where the LiDAR-based TTC estimate did not appear plausible when compared to the visual change in distance of the preceding vehicle.

### Observations
- Sudden jumps or drops in TTC values occurred even though the distance between vehicles changed smoothly.
- Sparse LiDAR returns on the rear of the preceding vehicle led to unstable distance estimates.
- Outlier LiDAR points closer to the ego vehicle occasionally affected the TTC calculation.

### Root Causes
- The LiDAR-based TTC calculation assumes a **constant velocity model**, meaning it assumes the relative velocity between vehicles remains constant between frames.
- In real-world driving scenarios, vehicles frequently accelerate or decelerate, violating this assumption.
- Measurement noise and uneven LiDAR point distribution further amplify TTC inconsistencies when using a simple distance-based approach.

### Conclusion
The primary limitation of the LiDAR-based TTC approach is the constant velocity assumption combined with sensitivity to outliers. These factors explain why TTC estimates can become inconsistent in dynamic traffic conditions.

---

## FP.6 – Performance Evaluation 2 (Camera-based TTC)

Different detector and descriptor combinations were evaluated to analyze their effect on camera-based TTC estimation.

### Detector / Descriptor Comparison

| Detector   | Descriptor | TTC Stability | Notes |
|-----------|------------|---------------|------|
| SHITOMASI | BRISK      | High          | Stable keypoints on vehicle edges |
| ORB       | ORB        | Medium        | Occasional mismatches |
| FAST      | BRIEF      | Low           | Sensitive to noise and motion |
| SIFT      | SIFT       | High          | Robust but computationally expensive |

### Observations
- Combinations producing fewer but more stable keypoints resulted in more reliable TTC estimates.
- Camera-based TTC was highly sensitive to incorrect keypoint matches.
- Sudden TTC spikes were observed when outliers passed the matching stage.

### Reasons for Camera TTC Errors
- Perspective effects amplify small pixel errors into large TTC variations.
- Camera-based TTC depends entirely on accurate feature matching.
- Unlike LiDAR, camera TTC has no direct depth measurement and is therefore more sensitive to noise.

### Conclusion
Camera-based TTC estimation is more sensitive to noise and matching errors than LiDAR-based TTC. Robust keypoint detection and filtering are critical for stable performance.

---

## Dependencies

- cmake ≥ 2.8  
- make  
- gcc / g++  
- OpenCV ≥ 4.1  
- Git LFS (for YOLO weight files)  

---

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./3D_object_tracking
