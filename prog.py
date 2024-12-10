import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import os

TF_ENABLE_ONEDNN_OPTS=0

# Определение имени файла с параметрами масштабирования данных
scaler_filename = "scaler.save"

# Загрузка модели и параметров масштабирования данных
model = load_model('emotion_recognition_model.h5')
scaler = joblib.load(scaler_filename)

def map_emotions(emotion):
    emotion_map = {
        'anger': 0, 'disgust': 0, 'fear': 0, 'sadness': 0,  # Негативные эмоции
        'happiness': 1, 'neutral': 1, 'enthusiasm': 1       # Позитивные эмоции
    }
    return emotion_map.get(emotion)

def extract_features(audio_path, sr=16000):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        features = np.hstack([mfccs, chroma, spectral_contrast, spectral_centroid, spectral_bandwidth])
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    if features is not None:
        features = scaler.transform([features])
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        return 'Positive' if predicted_class[0] == 1 else 'Negative'
    else:
        return 'Error in feature extraction'

def start_analyse(file_path):
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python predict_emotion.py <audio_file_path>")
    #     sys.exit(1)

    emotion = predict_emotion(file_path)
    print(file_path, emotion)
    # print(f"Predicted emotion: {emotion}")
    return emotion
