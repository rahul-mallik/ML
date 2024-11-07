import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import tempfile
import pygame

# Custom LSTM class to load the pre-trained model
from tensorflow.keras.layers import LSTM

class CustomLSTM(LSTM):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

# Load the pre-trained model using the custom LSTM class
custom_objects = {'LSTM': CustomLSTM}
model = load_model('action.h5', custom_objects=custom_objects)

# Actions and corresponding colors
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Update with actual actions
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def play_odia_audio(action):
    audio_file = f"adudioLib/odia_{action}.mpeg"
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.music.unload()

def text_to_speech(text, language, action):
    if language == "or":  # Odia
        play_odia_audio(action)
    else:
        tts = gTTS(text=text, lang=language)
        temp_filename = next(tempfile._get_candidate_names()) + ".mp3"
        tts.save(temp_filename)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.music.unload()
        os.remove(temp_filename)

def main():
    st.title("Real-time Action Detection with Text-to-Speech")

    # Language selection dropdown
    language = st.selectbox(
        "Select Language",
        ("en", "es", "fr", "de", "zh", "or", "hi", "bn", "gu", "kn", "ml", "mr", "pa", "ta", "te", "ur")
    )
    
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    sequence = []
    sentence = []
    predictions = []
    translated_text = ""
    threshold = 0.5
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        action = actions[np.argmax(res)]
                        print(f"Detected Action: {action}")  # Debugging print
                        print(f"Language: {language}")  # Debugging print
                        translated_text = action  # Directly use the action as text for TTS
                        if len(sentence) > 0:
                            if action != sentence[-1]:
                                sentence.append(action)
                                text_to_speech(translated_text, language, action)
                        else:
                            sentence.append(action)
                            text_to_speech(translated_text, language, action)
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            stframe.image(image, channels="BGR", use_column_width=True)
            
            # Print detected action and translated text in Streamlit
            if len(sentence) > 0:
                st.text(f"Detected Action: {sentence[-1]}")
                st.text(f"Translated Text: {translated_text}")
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()