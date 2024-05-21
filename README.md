# Emotion Recognition from Audio

## Project Description

This project aims to develop a robust system for recognizing emotions from audio files. Leveraging advanced deep learning techniques, the project processes audio data, extracts relevant features, and trains a model to accurately predict the emotional state conveyed in the audio. The ultimate goal is to provide a user-friendly interface through a Streamlit app where users can upload their audio files and receive real-time emotion predictions.

## Models Utilized

The project uses Convolutional Neural Networks (CNNs) to classify emotions from audio features. The model architecture includes:

- **Conv1D Layers**: Used to capture the temporal features from the audio data.
- **MaxPooling1D Layers**: For down-sampling the input representations and reducing dimensionality.
- **Dense Layers**: Fully connected layers to interpret the features learned by the convolutional layers.
- **Dropout Layers**: To prevent overfitting by randomly setting a fraction of input units to zero at each update during training.

The model is trained on a diverse dataset comprising audio samples from multiple datasets, including RAVDESS, CREMA-D, TESS, and SAVEE. Data augmentation techniques such as noise injection, time-stretching, and pitch shifting are employed to enhance the model's robustness.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/shekharkalshetti/emotion_recognition.git
cd emotion_recognition
pip install -r requirements.txt

```

## Streamlit App

The Streamlit app allows users to upload audio files and get predictions for the emotion present in the audio.

To run the Streamlit app:

```bash
streamlit run streamlit_app.py

```
