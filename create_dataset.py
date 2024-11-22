import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/Data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            img_path_full = os.path.join(DATA_DIR, dir_, img_path)
            print(f"Processing image: {img_path_full}")

            img = cv2.imread(img_path_full)
            if img is None:
                print(f"WARNING: Unable to load image {img_path_full}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("INFO: Created TensorFlow Lite XNNPACK delegate for CPU.")