import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, defaultdict
from fer import FER
import time
import pyttsx3

def predict_characters():
    # Load the trained model
    try:
        model_dict = pickle.load(open('/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/model.p', 'rb'))
        model = model_dict['model']
    except FileNotFoundError:
        print("Error: Model file not found.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    emotion_detector = FER()
    engine = pyttsx3.init()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    labels_dict = {i: chr(65 + i) for i in range(26)}
    labels_dict.update({i: str(i - 25) for i in range(26, 36)})

    buffer_size = 5
    prediction_buffers = defaultdict(list)
    user_words = defaultdict(str)
    user_last_time = defaultdict(lambda: time.time())

    show_help = False  # Toggle help guide
    help_icon = cv2.imread("help_icon.png")  # 32x32 icon for help
    if help_icon is not None:
        help_icon = cv2.resize(help_icon, (40, 40))
    else:
        print("Warning: 'help_icon.png' not found. Help icon will not display.")

    # Load ASL alphabet guide
    help_image = cv2.imread("/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/ASL_Alphabet.jpg")
    if help_image is not None:
        help_image = cv2.resize(help_image, (600, 800))  # Adjust dimensions as needed
    else:
        print("Warning: 'ASL_Alphabet.jpg' not found. Help image won't show.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_rgb = frame_rgb[y:y + h, x:x + w]
            results = emotion_detector.detect_emotions(face_rgb)
            if results:
                emotions = results[0]['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Emotion: {dominant_emotion}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                user_id = f"user_{idx}"
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_, y_, data_aux = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))

                try:
                    prediction = model.predict([np.asarray(data_aux[:42])])
                    predicted_index = int(prediction[0])
                    predicted_character = labels_dict.get(predicted_index, None)
                    if not predicted_character:
                        continue
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue

                prediction_buffers[user_id].append(predicted_character)
                if len(prediction_buffers[user_id]) > buffer_size:
                    prediction_buffers[user_id].pop(0)

                most_common_char = Counter(prediction_buffers[user_id]).most_common(1)[0][0]
                current_time = time.time()

                if current_time - user_last_time[user_id] >= 5:
                    user_words[user_id] += most_common_char
                    user_last_time[user_id] = current_time

                time_left = int(5 - (current_time - user_last_time[user_id]))
                timer_text = f'Time left: {max(time_left, 0)}s'

                base_y = 60 + idx * 100
                cv2.putText(frame, f'User {idx + 1}: {most_common_char}', (10, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Word: {user_words[user_id]}', (10, base_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, timer_text, (10, base_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Show help icon in top-right
        if help_icon is not None:
            frame[10:50, W - 50:W - 10] = help_icon
            cv2.putText(frame, "Help", (W - 85, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        # Instructions
        cv2.putText(frame, 'Press R: Reset | S: Save | Q: Quit | H: Help', (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        if show_help and help_image is not None:
            cv2.imshow("Sign Language Guide", help_image)

        cv2.imshow('Sign Language Detector', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            for uid in user_words:
                user_words[uid] = ""
        elif key == ord('s'):
            for uid, word in user_words.items():
                if word:
                    with open("saved_words.txt", "a") as f:
                        f.write(f"{uid}: {word}\n")
                    print(f"Saved word for {uid}: {word}")
                    engine.say(f"{word} saved")
                    engine.runAndWait()
                user_words[uid] = ""
        elif key == ord('h'):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()

predict_characters()
