import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def cross_correlation_similarity(y1, y2, sr=22050, duration=3.0):
    """Compute peak normalized cross-correlation between two audio clips."""
    n_samples = int(duration * sr)
    a = y1[:n_samples] if len(y1) >= n_samples else np.pad(y1, (0, n_samples - len(y1)))
    b = y2[:n_samples] if len(y2) >= n_samples else np.pad(y2, (0, n_samples - len(y2)))
    # Normalize
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    # Cross-correlation
    xcorr = np.correlate(a, b, mode="full")
    return float(np.max(np.abs(xcorr)))

# Fix: finding audio files
audio_files = glob.glob("*.wav")

sample_files = audio_files[:10]
labels = [os.path.splitext(os.path.basename(f))[0][:15] for f in sample_files]
waveforms = [librosa.load(f, sr=22050, mono=True, duration=5.0)[0] for f in sample_files]
n = len(waveforms)
xcorr_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        xcorr_matrix[i, j] = cross_correlation_similarity(waveforms[i], waveforms[j])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(xcorr_matrix, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax)
ax.set_title("Pairwise Cross-Correlation Similarity Matrix", fontsize=13, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("xcorr_similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
