import pickle
import numpy as np
from extract_features import extract_features

# Load saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


def predict_vehicle(audio_file):
    features, _, _, _, _ = extract_features(audio_file)
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = max(probabilities) * 100

    print(f"\nFile       : {audio_file}")
    print(f"Prediction : {prediction.upper()}")
    print(f"Confidence : {confidence:.1f}%")
    return prediction


# Test both
predict_vehicle("dataset/car/car1.wav")
predict_vehicle("dataset/bike/bike1.wav")