import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from data.feature_extraction import extract_features
from sklearn.preprocessing import OneHotEncoder


@st.cache(allow_output_mutation=True)
def load_emotion_model():
    return load_model('path/to/your/saved_model.h5')


@st.cache(allow_output_mutation=True)
def load_encoder():
    encoder = OneHotEncoder()
    encoder.fit(np.array(['neutral', 'calm', 'happy', 'sad',
                'angry', 'fear', 'disgust', 'surprise']).reshape(-1, 1))
    return encoder


model = load_emotion_model()
encoder = load_encoder()

st.title("Emotion Recognition from Speech")

uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])

if uploaded_file is not None:
    data, sample_rate = librosa.load(uploaded_file, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    predictions = model.predict(features)
    predicted_emotion = encoder.inverse_transform(predictions)[0][0]

    st.write(f'Predicted Emotion: {predicted_emotion}')
