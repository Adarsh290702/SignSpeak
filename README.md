Overview
inference_classifier.py is a Python script designed for real-time hand gesture and emotion detection. It leverages MediaPipe for hand landmark detection, a machine learning model for gesture recognition, and the FER library for emotion analysis.

Features
Hand Gesture Recognition:

Detects hand gestures using MediaPipe's hand landmarks.
Recognizes gestures and maps them to characters (A-Z) and numbers (1-10).
Implements a buffer to ensure stable predictions over consecutive frames.
Emotion Detection:

Identifies emotions from facial expressions using the FER library.
Displays the detected emotion and bounding box for the face.
Real-Time Operation:

Captures video input from a webcam.
Processes frames to display predictions for gestures and emotions dynamically.
Performance Metrics:

Includes functionality for calculating real-time accuracy for gesture predictions against ground truth.
Dependencies
Ensure you have the following Python libraries installed before running the script:

opencv-python (for video capture and processing)
mediapipe (for hand landmark detection)
numpy (for numerical operations)
pickle (for loading the trained model)
fer (for emotion recognition)
collections (for managing buffers)
Install the dependencies with:

pip install opencv-python mediapipe numpy fer
Usage
Prerequisites:

A trained machine learning model (model.p) must be saved in the specified path (/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/).
Ensure your webcam is functional.
Running the Script: Execute the script in a terminal or an IDE:

python inference_classifier.py
Controls:

Press q to exit the application.
Key Functions
process_predicted_characters(predicted_characters):

Filters and returns the most significant characters from predictions.
calculate_real_time_accuracy(predicted_characters, ground_truth):

Computes the prediction accuracy compared to the ground truth.
predict_characters():

Core function for capturing video input, processing frames for gesture and emotion detection, and displaying results.
Output
The application displays:

Real-time video feed with:
Recognized gestures for each detected user (e.g., "User 1: A").
Detected emotions with a bounding box around the face.
Console logs may include predictions and accuracy metrics.
Customization
Update labels_dict to modify or extend the range of recognizable gestures.
Adjust buffer_size to control the stability of predictions.
Replace model.p with your trained model as necessary.
Future Enhancements
Add support for more gestures and emotions.
Optimize the model for higher accuracy and lower latency.
Include logging or saving of detected gestures and emotions.
