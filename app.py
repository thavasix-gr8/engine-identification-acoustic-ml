from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, io, base64, tempfile

app = Flask(__name__)

VEHICLE_ICONS = {
    "BUS": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <rect x="5" y="10" width="90" height="35" rx="6"/>
      <line x1="5" y1="28" x2="95" y2="28"/>
      <line x1="35" y1="10" x2="35" y2="45"/>
      <line x1="65" y1="10" x2="65" y2="45"/>
      <rect x="10" y="14" width="18" height="11" rx="2"/>
      <rect x="40" y="14" width="18" height="11" rx="2"/>
      <rect x="70" y="14" width="18" height="11" rx="2"/>
      <rect x="10" y="31" width="18" height="10" rx="2"/>
      <rect x="40" y="31" width="18" height="10" rx="2"/>
      <rect x="70" y="31" width="18" height="10" rx="2"/>
      <circle cx="20" cy="50" r="6"/>
      <circle cx="80" cy="50" r="6"/>
    </svg>''',

    "MINIBUS": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10 40 L10 18 Q10 10 20 10 L75 10 Q88 10 90 22 L90 40 Z"/>
      <line x1="10" y1="27" x2="90" y2="27"/>
      <line x1="50" y1="10" x2="50" y2="40"/>
      <rect x="15" y="13" width="18" height="11" rx="2"/>
      <rect x="55" y="13" width="18" height="11" rx="2"/>
      <rect x="15" y="30" width="18" height="7" rx="2"/>
      <rect x="55" y="30" width="18" height="7" rx="2"/>
      <circle cx="25" cy="50" r="7"/>
      <circle cx="75" cy="50" r="7"/>
    </svg>''',

    "PICKUP": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M5 42 L5 30 L20 30 L35 12 L65 12 L65 30 L95 30 L95 42 Z"/>
      <line x1="65" y1="30" x2="5" y2="30"/>
      <rect x="36" y="14" width="27" height="15" rx="2"/>
      <rect x="68" y="20" width="22" height="10" rx="2"/>
      <circle cx="22" cy="50" r="7"/>
      <circle cx="78" cy="50" r="7"/>
    </svg>''',

    "SPORTS_CAR": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 40 L8 28 L25 28 L40 14 L70 14 L85 28 L97 28 L97 40 Z"/>
      <path d="M40 14 L38 28"/>
      <path d="M70 14 L72 28"/>
      <rect x="42" y="16" width="26" height="11" rx="2"/>
      <line x1="3" y1="35" x2="97" y2="35"/>
      <circle cx="22" cy="49" r="8"/>
      <circle cx="78" cy="49" r="8"/>
      <line x1="85" y1="28" x2="97" y2="30"/>
    </svg>''',

    "JEEP": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <rect x="8" y="16" width="84" height="26" rx="4"/>
      <rect x="14" y="20" width="22" height="14" rx="2"/>
      <rect x="42" y="20" width="22" height="14" rx="2"/>
      <rect x="70" y="20" width="16" height="14" rx="2"/>
      <line x1="8" y1="32" x2="92" y2="32"/>
      <circle cx="22" cy="50" r="8"/>
      <circle cx="78" cy="50" r="8"/>
      <line x1="36" y1="16" x2="36" y2="42"/>
      <line x1="64" y1="16" x2="64" y2="42"/>
      <rect x="3" y="22" width="5" height="8" rx="1"/>
    </svg>''',

    "TRUCK": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="18" width="55" height="26" rx="3"/>
      <rect x="58" y="8" width="39" height="36" rx="3"/>
      <line x1="58" y1="26" x2="97" y2="26"/>
      <rect x="62" y="12" width="15" height="12" rx="2"/>
      <rect x="80" y="12" width="13" height="12" rx="2"/>
      <circle cx="18" cy="50" r="7"/>
      <circle cx="45" cy="50" r="7"/>
      <circle cx="82" cy="50" r="7"/>
    </svg>''',

    "CROSSOVER": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M5 40 L5 28 L18 28 L32 14 L68 14 L82 28 L95 28 L95 40 Z"/>
      <rect x="33" y="16" width="34" height="11" rx="2"/>
      <line x1="5" y1="34" x2="95" y2="34"/>
      <line x1="32" y1="14" x2="32" y2="40"/>
      <line x1="68" y1="14" x2="68" y2="40"/>
      <rect x="8" y="30" width="18" height="7" rx="1"/>
      <rect x="74" y="30" width="18" height="7" rx="1"/>
      <circle cx="23" cy="50" r="8"/>
      <circle cx="77" cy="50" r="8"/>
    </svg>''',

    "CAR": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M5 40 L5 30 L15 30 L30 16 L70 16 L85 30 L95 30 L95 40 Z"/>
      <rect x="31" y="18" width="38" height="11" rx="2"/>
      <line x1="5" y1="35" x2="95" y2="35"/>
      <line x1="30" y1="16" x2="30" y2="40"/>
      <line x1="70" y1="16" x2="70" y2="40"/>
      <circle cx="22" cy="50" r="8"/>
      <circle cx="78" cy="50" r="8"/>
    </svg>''',

    "BIKE": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60" width="90" height="54" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="22" cy="42" r="14"/>
      <circle cx="78" cy="42" r="14"/>
      <circle cx="22" cy="42" r="4"/>
      <circle cx="78" cy="42" r="4"/>
      <path d="M22 42 L45 20 L60 20 L78 42"/>
      <path d="M45 20 L50 42"/>
      <path d="M60 20 L70 10 L80 10"/>
      <path d="M38 20 L45 20"/>
    </svg>'''
}


def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


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
    low_band          = pos_spectrum[pos_freqs < 300]
    high_band         = pos_spectrum[pos_freqs >= 300]
    band_ratio        = np.sum(low_band**2) / (np.sum(high_band**2) + 1e-10)

    # New features
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    rms_energy         = librosa.feature.rms(y=y).mean()

    # MFCCs — 13 coefficients
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1).tolist()

    features = [
        spectral_centroid,
        total_energy,
        dominant_freq,
        band_ratio,
        spectral_rolloff,
        spectral_bandwidth,
        zero_crossing_rate,
        rms_energy,
        *mfcc_means
    ]

    return features, y, sr, pos_freqs, pos_spectrum


def generate_waveform(y, sr, color='#00fff7'):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    time = np.linspace(0, len(y)/sr, len(y))
    ax.plot(time, y, color=color, linewidth=0.6, alpha=0.9)
    ax.fill_between(time, y, alpha=0.15, color=color)
    ax.set_xlabel('Time (s)', color=color, fontsize=9, labelpad=6)
    ax.set_ylabel('Amplitude', color=color, fontsize=9, labelpad=6)
    ax.tick_params(colors=color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(0.5)
    ax.grid(True, color=color, alpha=0.08, linewidth=0.4)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64


def generate_fft(pos_freqs, pos_spectrum, color='#00fff7'):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    mask  = (pos_freqs >= 0) & (pos_freqs <= 2000)
    freqs = pos_freqs[mask]
    spec  = pos_spectrum[mask]
    ax.plot(freqs, spec, color=color, linewidth=0.7)
    ax.fill_between(freqs, spec, alpha=0.15, color='#a855f7')
    ax.set_xlabel('Frequency (kHz)', color=color, fontsize=9, labelpad=6)
    ax.set_ylabel('Magnitude', color=color, fontsize=9, labelpad=6)
    ax.set_xticks(np.arange(0, 2001, 500))
    ax.set_xticklabels([f'{x/1000:.1f}' for x in np.arange(0, 2001, 500)], fontsize=8)
    ax.tick_params(colors=color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(0.5)
    ax.grid(True, color=color, alpha=0.08, linewidth=0.4)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files supported'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        model, scaler    = load_model()
        features, y, sr, pos_freqs, pos_spectrum = extract_features(tmp_path)
        features_scaled  = scaler.transform([features])
        prediction       = str(model.predict(features_scaled)[0]).upper()
        probabilities    = model.predict_proba(features_scaled)[0]
        certainty        = float(round(max(probabilities) * 100, 1))
        icon             = VEHICLE_ICONS.get(prediction, VEHICLE_ICONS["CAR"])

        # Match graph color to vehicle color
        color_map = {
            'BIKE':       '#a855f7',
            'BUS':        '#f97316',
            'CAR':        '#00fff7',
            'CROSSOVER':  '#22c55e',
            'JEEP':       '#84cc16',
            'MINIBUS':    '#fb923c',
            'PICKUP':     '#eab308',
            'SPORTS_CAR': '#ef4444',
            'TRUCK':      '#3b82f6',
        }
        color = color_map.get(prediction, '#00fff7')

        waveform_img = generate_waveform(y, sr, color)
        fft_img      = generate_fft(pos_freqs, pos_spectrum, color)

        return jsonify({
            'prediction': prediction,
            'icon':       icon,
            'certainty':  certainty,
            'color':      color,
            'features': {
                'spectral_centroid': float(round(float(features[0]), 1)),
                'total_energy':      float(round(float(features[1]), 4)),
                'dominant_freq':     float(round(float(features[2]), 1)),
                'band_ratio':        float(round(float(features[3]), 4)),
            },
            'waveform': waveform_img,
            'fft':      fft_img,
        })

    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Run train_model.py first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(debug=False, threaded=True)