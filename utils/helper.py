import numpy as np


def predict_emotion(model, features, encoder):
    features = np.expand_dims(features, axis=2)
    predictions = model.predict(features)
    predictions = encoder.inverse_transform(predictions)
    return predictions
