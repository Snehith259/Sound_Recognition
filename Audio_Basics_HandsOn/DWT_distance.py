import librosa
import numpy as np
from dtaidistance import dtw # pip install dtaidistance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def extract_mel_sequence(filepath, sr=22050, n_mels=64, max_frames=200):
 """Extract Mel spectrogram as a time sequence (truncated/padded)."""
 y, _ = librosa.load(filepath, sr=sr, duration=5.0)
 mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
 log_mel = librosa.power_to_db(mel, ref=np.max)
 # Average across Mel bins → 1D time series for DTW
 return log_mel.mean(axis=0)[:max_frames]

# Collect audio files
audio_files = glob.glob("*.wav")
if not audio_files:
    raise FileNotFoundError("No .wav files found in current directory.")

sample_files = audio_files[:10]
labels = [os.path.splitext(os.path.basename(f))[0][:15] for f in sample_files]
sequences = [extract_mel_sequence(f) for f in sample_files]
n = len(sequences)
# Pairwise DTW distance
dtw_matrix = np.zeros((n, n))
for i in range(n):
 for j in range(i+1, n):
    d = dtw.distance_fast(sequences[i].astype(np.double),
 sequences[j].astype(np.double))
 dtw_matrix[i, j] = d
 dtw_matrix[j, i] = d
# Normalize for visualization
dtw_norm = dtw_matrix / (dtw_matrix.max() + 1e-10)
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(dtw_norm, annot=True, fmt=".2f", cmap="Blues_r",
 xticklabels=labels, yticklabels=labels,
 linewidths=0.5, ax=ax)
ax.set_title("Pairwise DTW Distance Matrix (normalized)", fontsize=13, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig("dtw_distance_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("DTW distance matrix saved.")