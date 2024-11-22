import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from fer import FER

def process_predicted_characters(predicted_characters):
    filtered_characters = predicted_characters[5:-5]
    unique_characters = []
    seen_characters = set()
    for char in filtered_characters:
        if char not in seen_characters:
            unique_characters.append(char)
            seen_characters.add(char)
    return unique_characters[:2]

def calculate_real_time_accuracy(predicted_characters, ground_truth):
    correct_predictions = sum([1 for pc, gt in zip(predicted_characters, ground_truth) if pc == gt])
    total_predictions = len(predicted_characters)
    return correct_predictions / total_predictions if total_predictions > 0 else 0

def predict_characters():
    model_dict = pickle.load(open('/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    emotion_detector = FER()

    labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
    labels_dict.update({i: str(i - 25) for i in range(26, 36)})  # 26-35 to 1-10

    buffer_size = 5
    prediction_buffers = {}
    emotion_buffers = {}
    ground_truth = []
    accuracy_list = []

    while True:
        data_aux = {}
        x_ = {}
        y_ = {}

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Emotion detection
        emotion_results = emotion_detector.detect_emotions(frame_rgb)
        for emotion_result in emotion_results:
            if emotion_result:
                dominant_emotion = emotion_result['emotions']
                emotion_text = max(dominant_emotion, key=dominant_emotion.get)
                face_box = emotion_result['box']
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion_text}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Emotion buffer logic for tracking
                user_id = f"user_{len(emotion_buffers)}"
                if user_id not in emotion_buffers:
                    emotion_buffers[user_id] = []
                emotion_buffers[user_id].append(emotion_text)
                if len(emotion_buffers[user_id]) > buffer_size:
                    emotion_buffers[user_id].pop(0)

        # Hand landmark detection
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                user_id = f"user_{idx}"
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if user_id not in data_aux:
                    data_aux[user_id] = []
                    x_[user_id] = []
                    y_[user_id] = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_[user_id].append(x)
                    y_[user_id].append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux[user_id].append(x_[user_id][i] - min(x_[user_id]))
                    data_aux[user_id].append(y_[user_id][i] - min(y_[user_id]))

                prediction = model.predict([np.asarray(data_aux[user_id][:42])])
                try:
                    predicted_index = int(prediction[0])
                    if predicted_index in labels_dict:
                        predicted_character = labels_dict[predicted_index]
                    else:
                        continue
                except ValueError:
                    continue

                if user_id not in prediction_buffers:
                    prediction_buffers[user_id] = []
                prediction_buffers[user_id].append(predicted_character)
                if len(prediction_buffers[user_id]) > buffer_size:
                    prediction_buffers[user_id].pop(0)

                most_common_prediction = Counter(prediction_buffers[user_id]).most_common(1)[0][0]
                cv2.putText(frame, f'User {idx + 1}: {most_common_prediction}', (10, 50 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_characters()
