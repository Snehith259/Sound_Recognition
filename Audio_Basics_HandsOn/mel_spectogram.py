
import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np 
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
# Standard Mel Spectrogram 
mel_spec = librosa.feature.melspectrogram( 
    y=y, sr=sr, 
    n_fft=2048, 
    hop_length=512, 
    n_mels=128,       # Number of Mel filter banks 
    fmin=20,          # Minimum frequency (Hz) 
    fmax=8000         # Maximum frequency (Hz) 
) 
 
# Log-Mel Spectrogram (apply log to amplitude) 
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
 
print(f"Mel spectrogram shape  : {mel_spec.shape}") 
print(f"  Rows (Mel bins)      : {mel_spec.shape[0]}") 
print(f"  Columns (time steps) : {mel_spec.shape[1]}") 
print(f"  Min value (dB)       : {log_mel_spec.min():.2f}") 
print(f"  Max value (dB)       : {log_mel_spec.max():.2f}") 
 
# --- Visualize --- 
fig, axes = plt.subplots(2, 1, figsize=(14, 8)) 
fig.suptitle("Mel Spectrogram Analysis", fontsize=14, fontweight="bold") 
 
img1 = librosa.display.specshow( 
    mel_spec, sr=sr, hop_length=512, 
    x_axis="time", y_axis="mel", 
 fmin=20, fmax=8000, ax=axes[0], cmap="viridis" 
) 
axes[0].set_title("Mel Spectrogram (linear power)") 
fig.colorbar(img1, ax=axes[0], format="%+.0f") 
 
img2 = librosa.display.specshow( 
    log_mel_spec, sr=sr, hop_length=512, 
    x_axis="time", y_axis="mel", 
    fmin=20, fmax=8000, ax=axes[1], cmap="magma" 
) 
axes[1].set_title("Log-Mel Spectrogram (dB scale)") 
fig.colorbar(img2, ax=axes[1], format="%+2.0f dB") 
 
plt.tight_layout() 
plt.savefig("mel_spectrogram.png", dpi=150, bbox_inches="tight") 
plt.show() 
