import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

print("Loading model...")
model = load_model('best_emotion_model.h5')

#Haar cascade from opencv for detecting face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) # starts video capture

history_length = 10  # For Averaging the last 10 frames
emotion_history = []

while True:
    ret, frame = cap.read() #capture frame(Numpy array representing BGR)
    if not ret: break #didnt capture
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_reshaped = np.reshape(roi_resized, (1, 48, 48, 1))
        
        # Get prediction
        raw_prediction = model.predict(roi_reshaped, verbose=0)[0]
        
        # Adding it to history and maintaining size
        emotion_history.append(raw_prediction)
        if len(emotion_history) > history_length:
            emotion_history.pop(0)
            
        # Calculating the Avg of the history
        smoothed_prediction = np.mean(emotion_history, axis=0)
        
        max_index = np.argmax(smoothed_prediction)
        predicted_emotion = EMOTIONS[max_index]
        
        #UI
        for i, prob in enumerate(smoothed_prediction):
            label = EMOTIONS[i]
            # Draw black background
            cv2.rectangle(frame, (5, 5 + i*35), (250, 35 + i*35), (0,0,0), -1)
            # Draw smooth green bar
            bar_width = int(prob * 150)
            cv2.rectangle(frame, (10, 25 + i*35), (10 + bar_width, 30 + i*35), (0, 255, 0), -1)
            # Text
            cv2.putText(frame, f"{label}: {prob*100:.1f}%", (10, 20 + i*35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_emotion}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Smoothed Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break #quit

cap.release()
cv2.destroyAllWindows()
