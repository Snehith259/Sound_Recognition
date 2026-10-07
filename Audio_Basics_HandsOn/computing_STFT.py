import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np 
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
# --- Experiment with different n_fft and hop_length --- 
configs = [ 
    {"n_fft": 512,  "hop_length": 128,  "label": "Short window (n_fft=512, hop=128)"}, 
    {"n_fft": 2048, "hop_length": 512,  "label": "Standard (n_fft=2048, hop=512)"}, 
    {"n_fft": 4096, "hop_length": 1024, "label": "Long window (n_fft=4096, hop=1024)"}, 
] 
 
fig, axes = plt.subplots(len(configs), 3, figsize=(20, 14)) 
fig.suptitle("STFT Analysis — Effect of n_fft and hop_length", fontsize=14, fontweight="bold") 
 
for row, cfg in enumerate(configs): 
    n_fft      = cfg["n_fft"] 
    hop_length = cfg["hop_length"] 
 
    # Compute STFT (complex matrix) 
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length) 
 
    # Magnitude spectrogram (|STFT|) 
    magnitude = np.abs(D) 
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max) 
 
    # Phase spectrogram (angle of STFT) 
    phase = np.angle(D) 
 
    # --- Col 1: Magnitude --- 
    img1 = librosa.display.specshow( 
        magnitude_db, sr=sr, hop_length=hop_length, 
        x_axis="time", y_axis="hz", ax=axes[row, 0], cmap="magma" 
    ) 
    axes[row, 0].set_title(f"{cfg['label']}\nMagnitude (dB)") 
    fig.colorbar(img1, ax=axes[row, 0], format="%+2.0f dB") 

     
    # --- Col 2: Log-scale magnitude --- 
    img2 = librosa.display.specshow( 
        magnitude_db, sr=sr, hop_length=hop_length, 
        x_axis="time", y_axis="log", ax=axes[row, 1], cmap="magma" 
    ) 
    axes[row, 1].set_title(f"Magnitude (dB) — Log Freq Scale") 
    fig.colorbar(img2, ax=axes[row, 1], format="%+2.0f dB") 
 
    # --- Col 3: Phase --- 
    img3 = librosa.display.specshow( 
        phase, sr=sr, hop_length=hop_length, 
        x_axis="time", y_axis="hz", ax=axes[row, 2], cmap="twilight" 
    ) 
    axes[row, 2].set_title(f"Phase (radians)") 
    fig.colorbar(img3, ax=axes[row, 2]) 
 
plt.tight_layout() 
plt.savefig("stft_analysis.png", dpi=150, bbox_inches="tight") 
plt.show() 
print("STFT analysis plots saved.") 
