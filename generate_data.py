import numpy as np
import soundfile as sf
import os

os.makedirs("dataset/car", exist_ok=True)
os.makedirs("dataset/bike", exist_ok=True)

def generate_car_sound(filename, duration=3, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    sound = (
        0.5 * np.sin(2 * np.pi * 80 * t) +
        0.3 * np.sin(2 * np.pi * 120 * t) +
        0.2 * np.sin(2 * np.pi * 150 * t) +
        0.05 * np.random.randn(len(t))
    )
    sound = sound / np.max(np.abs(sound))
    sf.write(filename, sound, sr)

def generate_bike_sound(filename, duration=3, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    sound = (
        0.5 * np.sin(2 * np.pi * 250 * t) +
        0.3 * np.sin(2 * np.pi * 350 * t) +
        0.2 * np.sin(2 * np.pi * 480 * t) +
        0.08 * np.random.randn(len(t))
    )
    sound = sound / np.max(np.abs(sound))
    sf.write(filename, sound, sr)

for i in range(1, 16):
    generate_car_sound(f"dataset/car/car{i}.wav")
    generate_bike_sound(f"dataset/bike/bike{i}.wav")

print("Done! Generated 15 car + 15 bike .wav files.")