import librosa
import numpy as np
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3.0)

    # FFT
    fft_spectrum = np.abs(np.fft.fft(y))
    frequencies  = np.fft.fftfreq(len(fft_spectrum), d=1/sr)
    pos_freqs    = frequencies[:len(frequencies)//2]
    pos_spectrum = fft_spectrum[:len(fft_spectrum)//2]

    # Original 4 features
    spectral_centroid = np.sum(pos_freqs * pos_spectrum) / (np.sum(pos_spectrum) + 1e-10)
    total_energy      = np.sum(pos_spectrum ** 2)
    dominant_freq     = pos_freqs[np.argmax(pos_spectrum)]
    low_band  = pos_spectrum[pos_freqs < 300]
    high_band = pos_spectrum[pos_freqs >= 300]
    band_ratio = np.sum(low_band**2) / (np.sum(high_band**2) + 1e-10)

    # NEW — librosa extra features
    spectral_rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth  = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    zero_crossing_rate  = librosa.feature.zero_crossing_rate(y).mean()
    rms_energy          = librosa.feature.rms(y=y).mean()

    # NEW — MFCCs (13 coefficients — captures timbre/texture of engine)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1).tolist()

    # Combine all features into one list
    features = [
        spectral_centroid,
        total_energy,
        dominant_freq,
        band_ratio,
        spectral_rolloff,
        spectral_bandwidth,
        zero_crossing_rate,
        rms_energy,
        *mfcc_means   # adds 13 more values
    ]

    return features, y, sr, pos_freqs, pos_spectrum


def build_dataset(dataset_folder):
    features_list = []
    labels_list   = []

    for vehicle_type in os.listdir(dataset_folder):
        folder = os.path.join(dataset_folder, vehicle_type)
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                file_path = os.path.join(folder, filename)
                try:
                    features, _, _, _, _ = extract_features(file_path)
                    features_list.append(features)
                    labels_list.append(vehicle_type.upper())
                    print(f"  Processed: {filename}")
                except Exception as e:
                    print(f"  Error with {filename}: {e}")

    return np.array(features_list), np.array(labels_list)


if __name__ == "__main__":
    test_file = "dataset/car/car1.wav"
    features, _, _, _, _ = extract_features(test_file)
    print(f"Total features extracted: {len(features)}")