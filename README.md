# Acoustic Vehicle Classifier — Engine Identification Using Signal Processing & ML

A machine learning web application that identifies vehicle types from engine audio signals using **FFT-based feature extraction** and a **Support Vector Machine (SVM)** classifier. Upload any `.wav` engine recording and the system predicts the vehicle class with high accuracy.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey) ![Accuracy](https://img.shields.io/badge/Accuracy-98.4%25-brightgreen) ![Classes](https://img.shields.io/badge/Classes-9-orange)

---

## Live Demo

> Run locally — see setup instructions below.

---

## Project Overview

This project classifies vehicle engine sounds into **9 distinct vehicle categories**:

| Class | Description |
|---|---|
| CAR | Standard passenger car |
| BIKE | Motorcycle |
| BUS | Full-size bus |
| MINIBUS | Minibus / shuttle |
| PICKUP | Pickup truck |
| CROSSOVER | Crossover SUV |
| JEEP | Off-road vehicle |
| SPORTS_CAR | High-performance car |
| TRUCK | Heavy commercial truck |

---

## How It Works
```
Audio (.wav)
    ↓
librosa.load()         — load raw signal
    ↓
FFT (numpy)            — convert time → frequency domain
    ↓
Feature Extraction     — 21 features extracted
    ↓
StandardScaler         — normalize features
    ↓
SVM Classifier         — RBF kernel, C=10
    ↓
Vehicle Prediction     — with certainty %
```

---

## Features Extracted (21 total)

| # | Feature | Description |
|---|---|---|
| 1 | Spectral Centroid | Center of mass of frequency spectrum |
| 2 | Total Energy | Overall signal power |
| 3 | Dominant Frequency | Peak frequency in spectrum |
| 4 | Band Energy Ratio | Low vs high frequency energy ratio |
| 5 | Spectral Rolloff | Frequency below which 85% of energy exists |
| 6 | Spectral Bandwidth | Spread of the frequency spectrum |
| 7 | Zero Crossing Rate | Signal roughness indicator |
| 8 | RMS Energy | Root mean square amplitude |
| 9–21 | MFCC 1–13 | Mel-frequency cepstral coefficients — engine timbre fingerprint |

---

## Model Performance
```
Accuracy     : 98.4%
CV Score     : 97.9% ± 0.4%

              precision  recall  f1-score
BIKE            1.00      1.00      1.00
BUS             1.00      1.00      1.00
CAR             0.95      0.97      0.96
CROSSOVER       0.96      0.97      0.97
JEEP            0.98      0.97      0.98
MINIBUS         1.00      0.97      0.99
PICKUP          0.98      0.96      0.97
SPORTS_CAR      1.00      1.00      1.00
TRUCK           1.00      1.00      1.00
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Signal Processing | NumPy (FFT), Librosa |
| Machine Learning | Scikit-learn (SVM, StandardScaler) |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, Vanilla JS |
| Fonts | Bebas Neue, Share Tech Mono |

---

## Project Evolution

| Phase | What Changed |
|---|---|
| Phase 1 | Started with synthetic generated audio — 2 classes (Car vs Bike) — 4 features — ~70% accuracy |
| Phase 2 | Switched to real VISC8 dataset — expanded to 9 vehicle classes — 4 features |
| Phase 3 | Upgraded to 21 features including MFCCs — accuracy jumped to **98.4%** |

---

## Dataset

This project uses the **VISC8 Dataset** — a benchmark dataset of 5,980 vehicle interior engine sounds recorded at 48kHz.

**Download:** [https://zenodo.org/records/5606504](https://zenodo.org/records/5606504)

After downloading, extract and organize the files like this:
```
vehicle_classifier/
└── dataset/
    ├── car/
    ├── bike/
    ├── bus/
    ├── minibus/
    ├── pickup/
    ├── crossover/
    ├── jeep/
    ├── sports_car/
    └── truck/
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/thavasix-gr8/engine-identification-acoustic-ml.git
cd engine-identification-acoustic-ml/vehicle_classifier
```

### 2. Install dependencies
```bash
pip install flask librosa numpy scikit-learn matplotlib soundfile
```

### 3. Download and prepare the dataset

Download VISC8 from [zenodo.org/records/5606504](https://zenodo.org/records/5606504) and place audio files into the folder structure shown above.

### 4. Train the model
```bash
python train_model.py
```

This generates `model.pkl` and `scaler.pkl` in the project root.

### 5. Run the web app
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Usage

1. Open the web app in your browser
2. Drag and drop any `.wav` engine audio file into the upload zone
3. The system will display:
   - Vehicle type prediction with certainty percentage
   - 4 key spectral feature values
   - Time domain waveform graph
   - Frequency domain FFT spectrum graph
4. Click **ANALYZE_NEW_FILE** to test another file

---

## File Structure
```
vehicle_classifier/
├── dataset/              ← audio files (not included, download separately)
├── static/
│   ├── main.js           ← frontend logic
│   └── style.css         ← styling
├── templates/
│   └── index.html        ← web interface
├── app.py                ← Flask backend + feature extraction
├── extract_features.py   ← feature extraction module
├── train_model.py        ← model training script
├── predict.py            ← terminal prediction script
└── generate_data.py      ← synthetic data generator (for testing)
```

---

## License

MIT License — free to use for academic and research purposes.

---

*Built as a Signal Processing & Machine Learning mini project.*
