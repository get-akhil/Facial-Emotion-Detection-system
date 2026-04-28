# Facial Emotion Detection System

A real-time Computer Vision application that detects human faces and classifies their emotions into seven categories using a Convolutional Neural Network (CNN).

---

## Project Overview
It uses a custom-built CNN trained on the **FER-2013** dataset to identify emotions from live webcam feeds with high stability and accuracy.

### Key Features:
* **Real-time Detection:** Processes video frames at 30+ FPS.
* **Temporal Smoothing:** Uses a sliding-window average to prevent UI flickering.
* **Optimized Pipeline:** Efficient preprocessing using OpenCV and NumPy.

---

## The AI Model (`train.py`)
The core of this project is a Sequential CNN designed to extract spatial features from grayscale facial images.

### Data Processing
* **Dataset:** FER-2013 (35,887 images).
* **Normalization:** Pixel values scaled from **0–255** to **0–1** for mathematical stability.
* **One-Hot Encoding:** Categorical labels converted to vectors to ensure unbiased classification.

### Architecture
* **Conv2D Layers:** Uses **3x3 kernels** to detect edges, curves, and facial textures.
* **Activation:** **ReLU** is used to introduce non-linearity while maintaining speed.
* **Regularization:** **Dropout (0.5)** is applied to prevent overfitting (memorization).
* **Final Layer:** **Softmax** activation outputs a probability distribution across 7 classes.

---

## Live Inference (`detect_emotion.py`)
The detection script implements a multi-stage vision pipeline to transform raw webcam frames into emotion predictions.

### The Pipeline
1.  **Face Localization:** Uses a **Haar Cascade Classifier** to find coordinates $(x, y, w, h)$.
2.  **ROI Extraction:** The face is cropped as a "Region of Interest" to eliminate background noise.
3.  **Preprocessing:** Resized to **48x48** and reshaped into a **4D Tensor** $(1, 48, 48, 1)$.
4.  **Smoothing Logic:** A history buffer of 10 frames is maintained. We display the **mean probability**, making the green bars move smoothly.

---

## Tech Stack
* **Python 3.11**
* **TensorFlow/Keras:** Model building and training.
* **OpenCV:** Video stream handling and image processing.
* **NumPy:** High-speed matrix calculations.
* **Pandas:** Dataset management.

---

## Emotion Classes
The model classifies faces into one of the following:
* Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

---

## How to Run

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy pandas
```
### 2. Activate virtual environment
```bash
source venv/bin/activate
```
### 3. Run
```bash
python detect_emotion.py
