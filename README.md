# Facial Emotion Detection system
1. Project Overview
This project uses a Convolutional Neural Network (CNN) to identify human
emotions in real-time. By processing video from a webcam, the system can detect
faces and classify their expressions into one of seven categories: Angry, Disgust, Fear,
Happy, Sad, Surprise, or Neutral.
TensorFlow / Keras OpenCV Python

2. How the Model Works (The Brain)
The system is trained using the FER-2013 dataset, which contains over 35,000
labeled images of facial expressions.

The Architecture:
Feature Extraction: Convolutional layers use 3x3 filters to find patterns like
eye shapes and lip curves.
Downsampling: Max-pooling layers shrink the image to help the model
focus on core features.
Classification: Dense layers with Softmax activation output a probability
percentage for each emotion.

3. The Real-Time Pipeline
When you run the detection script, the following sequence occurs every millisecond:
Frame Capture: OpenCV grabs a frame from the webcam.
•

•

•

1.

Face Detection: A Haar Cascade classifier finds the (x, y) coordinates of faces in
the frame.
Preprocessing: The face is cropped, converted to grayscale, resized to 48x48, and
normalized (divided by 255).
Inference: The processed face is fed into the saved model (.h5 file).
Smoothing: The last 10 predictions are averaged to prevent flickering in the UI.
Visualization: The result and probability bars are drawn onto the screen.

4. Key Files
train.py
Handles data loading, preprocessing, model architecture design, and the training
loop.
detect_emotion.py
The "Live" script that uses the webcam and the trained model for real-time
interaction.
best_emotion_model.h5
The final "learned" weights of the AI, exported for use in the detection script.

5. Why it is Effective
This project uses Dropout (0.5) to prevent the model from simply memorizing the
training data, ensuring it performs well on new, unseen faces. The addition of
Temporal Smoothing in the live demo provides a professional, stable user
experience.

# SetUp:
  To install dependencies, run: pip install -r requirements.txt

# To create and activate virtual environment:
  python3 -m venv venv
  source venv/bin/activate

# To Run:
  python detect_emotion.py
