import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np 
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
fig, axes = plt.subplots(3, 1, figsize=(14, 9)) 
fig.suptitle("Waveform Analysis", fontsize=16, fontweight="bold") 
 
# --- Plot 1: Full waveform --- 
time_axis = np.linspace(0, len(y) / sr, num=len(y)) 
axes[0].plot(time_axis, y, color="steelblue", linewidth=0.5, alpha=0.8) 
axes[0].set_title("Full Waveform") 
axes[0].set_xlabel("Time (seconds)") 
axes[0].set_ylabel("Amplitude") 
axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--") 
axes[0].set_xlim([0, len(y) / sr]) 
 
# --- Plot 2: RMS Energy over time --- 
frame_length = 2048 
hop_length   = 512 
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0] 
times = librosa.times_like(rms, sr=sr, hop_length=hop_length) 
axes[1].plot(times, librosa.amplitude_to_db(rms, ref=np.max), 
             color="darkorange", linewidth=1.2) 
axes[1].set_title("RMS Energy (dB)") 

axes[1].set_xlabel("Time (seconds)") 
axes[1].set_ylabel("Energy (dB)") 
axes[1].set_xlim([0, len(y) / sr]) 
 
# --- Plot 3: Zoomed view (first 1 second) --- 
zoom_samples = min(sr, len(y)) 
axes[2].plot(time_axis[:zoom_samples], y[:zoom_samples], 
             color="mediumseagreen", linewidth=0.8) 
axes[2].set_title("Zoomed Waveform (First 1 Second)") 
axes[2].set_xlabel("Time (seconds)") 
axes[2].set_ylabel("Amplitude") 
axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--") 
 
plt.tight_layout() 
plt.savefig("waveform_analysis.png", dpi=150, bbox_inches="tight") 
plt.show() 
print("Waveform plots saved.")
