# Optichrom

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-yellow)

This repository provides a collection of Python tools and applications for comparing images using various algorithms such as ORB (Oriented FAST and Rotated BRIEF) and SSIM (Structural Similarity Index). The tools include both command-line scripts and graphical user interface (GUI) applications for ease of use.

## Key Features

- **ORB Feature Matching**: Identifies and matches features between two images using the ORB algorithm.
- **SSIM Comparison**: Computes the Structural Similarity Index (SSIM) between two images and visualizes the similarity map.
- **Graphical User Interface (GUI)**: Tkinter-based applications allow users to compare images visually using either SSIM or ORB.
- **Object Detection**: Supports object detection using the YOLO (You Only Look Once) model for images compared using ORB.
  
## Tools Included

1. **ORB Feature Matcher (`orb_feature_matcher.py`)**: 
   - Command-line tool to match features between two images using ORB.
   - Saves an image with the top 50 matched features highlighted.

2. **SSIM Visualizer (`ssim_visualizer.py`)**: 
   - Command-line tool for calculating the SSIM score between two images and generating a similarity map.

3. **Image Comparison App (`image_comparison_app.py`)**: 
   - Tkinter-based GUI for comparing images using either SSIM or ORB. 
   - Includes object detection using the YOLOv3 model.

4. **OptiChrom App (`optichrom_app.py`)**: 
   - Another GUI application for comparing images with support for SSIM and ORB comparison.
   - Detects objects using YOLOv3 in the compared images.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Solrikk/Optichrom.git
   cd image-comparison-tools


