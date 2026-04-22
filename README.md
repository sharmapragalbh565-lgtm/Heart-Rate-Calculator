# Heart Rate Detector using Webcam
Contactless Pulse Estimation with Computer Vision (rPPG)

This project estimates heart rate using a standard webcam by analyzing subtle color variations in facial skin caused by blood flow.

It uses Remote Photoplethysmography (rPPG) — a non-contact method for pulse detection.

# How It Works

The system:

Detects the face using OpenCV

Extracts the forehead region

Tracks changes in the green color channel

Applies bandpass filtering

Performs FFT frequency analysis

Converts dominant frequency to BPM

# Technologies Used

OpenCV – Face detection & webcam handling

NumPy – Signal processing

SciPy – Filtering & FFT

# System Architecture
Webcam Frame
     ↓
Face Detection
     ↓
Forehead ROI Extraction
     ↓
Green Channel Intensity Tracking
     ↓
Bandpass Filtering (0.8–3.0 Hz)
     ↓
FFT Frequency Analysis
     ↓
BPM Estimation

# Installation
git clone https://github.com/yourusername/heart-rate-detector.git
cd heart-rate-detector
pip install -r requirements.txt
python heart_rate_detector.py

# Requirements
opencv-python
numpy
scipy


# Possible Improvements

Use MediaPipe Face Mesh instead of Haar Cascade

Implement moving average smoothing

Add real-time graph visualization

Add signal quality index

Add GUI interface

Use deep learning–based rPPG (PhysNet)

Save heart rate data to CSV
