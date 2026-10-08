import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def extract_mel_feature_vector(filepath, sr=22050, n_mels=128, n_mfcc=40):
 """Extract a compact feature vector (mean + std of MFCC) for a clip."""
 y, _ = librosa.load(filepath, sr=sr, duration=10.0)
 mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
 # Mean and std across time → fixed-size vector regardless of duration
 return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

# Collect audio files
audio_files = glob.glob("*.wav")
if not audio_files:
    raise FileNotFoundError("No .wav files found in current directory.")

# Use first 10 files from your dataset
sample_files = audio_files[:10]
labels = [os.path.splitext(os.path.basename(f))[0][:15] for f in sample_files]
# Extract features
feature_matrix = np.array([extract_mel_feature_vector(f) for f in sample_files])
print(f"Feature matrix shape: {feature_matrix.shape}")
# Pairwise cosine similarity
cos_sim_matrix = cosine_similarity(feature_matrix)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
 xticklabels=labels, yticklabels=labels,
 linewidths=0.5, vmin=0, vmax=1, ax=ax)
ax.set_title("Pairwise Cosine Similarity — MFCC Features", fontsize=13, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig("cosine_similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Cosine similarity matrix saved.")