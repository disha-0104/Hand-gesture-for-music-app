import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, concatenate, Reshape
from tensorflow.keras.utils import to_categorical

# Function to load data from a JSON file
def load_json_data(json_file_path):
    data = []
    labels = []
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        for entry in json_data:
            landmarks = entry['landmarks']
            if landmarks and len(landmarks) == 21:  # Check if we have 21 landmarks
                data.append(np.array(landmarks).flatten())  # Flatten landmarks to a single array
                labels.append(entry['label'])
    return np.array(data), np.array(labels)

# Load data from files
train_data_file = '/Users/dishagundarpi/Desktop/processed_data/train_data.json'
val_data_file = '/Users/dishagundarpi/Desktop/processed_data/val_data.json'
test_data_file = '/Users/dishagundarpi/Desktop/processed_data/test_data.json'

X_train, y_train = load_json_data(train_data_file)
X_val, y_val = load_json_data(val_data_file)
X_test, y_test = load_json_data(test_data_file)

# Print the shapes of the loaded data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Reshape the input for CNN (N, height, width, channels)
# Assuming you have 21 landmarks, each with x and y coordinates
# Reshaping to (N, 21, 2, 1) - 21 landmarks with (x, y) coordinates
X_train = X_train.reshape(-1, 21, 2, 1)
X_val = X_val.reshape(-1, 21, 2, 1)
X_test = X_test.reshape(-1, 21, 2, 1)

# Define the hybrid model
input_layer = Input(shape=(21, 2, 1))

# CNN branch
cnn = Conv2D(32, (2, 2), activation='relu', padding='same')(input_layer)
cnn = MaxPooling2D((2, 2))(cnn)  # Changed to a 2x2 pooling layer
cnn = Flatten()(cnn)  # Flatten to feed into dense layers

# LSTM branch
lstm_input = Reshape((21, 2))(input_layer)  # Reshape to fit LSTM input
lstm = LSTM(64)(lstm_input)

# Concatenate both branches
combined = concatenate([cnn, lstm])

# Fully connected layers
fc = Dense(128, activation='relu')(combined)
fc = Dropout(0.5)(fc)
output_layer = Dense(y_train.shape[1], activation='softmax')(fc)

# Build and compile the model
hybrid_model = Model(inputs=input_layer, outputs=output_layer)
hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = hybrid_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Save the trained model to an .h5 file
hybrid_model.save('hybrid_model.h5')

# Test the model
y_pred = hybrid_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Load the model later if needed
model = load_model('hybrid_model.h5')
