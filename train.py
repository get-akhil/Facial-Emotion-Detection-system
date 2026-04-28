import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def load_and_preprocess_data(csv_path):
    
    # Loads fer2013.csv, parses the pixel strings, and prepares the data for training.
    print("Loading dataset...")
    data = pd.read_csv(csv_path)
    
    X_train, y_train = [], [] #for training
    X_val, y_val = [], []       #for validation
    X_test, y_test = [], []     #for Test
    
    for index,row in data.iterrows():
    
        # Parse pixel string into a NumPy array (2304 values)
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.float32)
        
        # Reshape array into 48x48x1 and normalize pixels to [0, 1]
        pixels = pixels.reshape((48, 48, 1)) / 255.0
        emotion = row['emotion']
        usage = row['Usage']
        
        if usage == 'Training':
            X_train.append(pixels)
            y_train.append(emotion)
        elif usage == 'PublicTest':
            X_val.append(pixels)
            y_val.append(emotion)
        elif usage == 'PrivateTest':
            X_test.append(pixels)
            y_test.append(emotion)
            
    # Converting lists to numpy arrays for calculations
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # We One-hot encode the 7 emotion labels to prevent assuming a numerical hierarchy between emotions
    # One-hot encoding turns a single category number into a list of 0 and 1, where a 1 marks the correct category and 0 marks everything else.
    y_train = to_categorical(y_train, num_classes=7)
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    
    #Constructs the Sequential CNN model
    model = Sequential([
    
        # Block 1 (simple edges)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2 (eye-corners and lip curves etc)
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3 (complex features)
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Classification Block
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Higher dropout before the final layer to prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    
    return model

#Execution:
CSV_PATH = 'fer2013.csv'
MODEL_SAVE_PATH = 'best_emotion_model.h5'

# Load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(CSV_PATH)
print(f"Training data shape: {X_train.shape}")

# Building and compiling model
model = build_cnn_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, # Stop if validation loss doesn't improve for 10 epochs
    restore_best_weights=True,
    verbose=1
)

# Training
print("Starting training...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

print("Training complete. Best model saved to", MODEL_SAVE_PATH)
