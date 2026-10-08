import torchaudio
import torch
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import glob

# Load pre-trained HuBERT model (downloads ~360MB on first run)
model_name = "facebook/hubert-base-ls960"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)
model.eval()
TARGET_SR = 16000 # HuBERT requires 16kHz input

def get_hubert_embedding(filepath, model, feature_extractor, target_sr=16000):
    """Extract mean-pooled HuBERT embedding from an audio file."""
    waveform, orig_sr = torchaudio.load(filepath)
    # Resample to 16kHz if necessary
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze().numpy()
    
    # Feature extraction
    inputs = feature_extractor(
        waveform, sampling_rate=target_sr,
        return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool over time dimension -> fixed 768-dim vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Fix: define audio_files
audio_files = glob.glob("*.wav")

# Use 5 specific clips (or any 5 from your dataset)
clips_5 = audio_files[:5]
clip_labels = [os.path.splitext(os.path.basename(f))[0][:20] for f in clips_5]
print("Extracting HuBERT embeddings...")
embeddings = []

for i, filepath in enumerate(clips_5):
    emb = get_hubert_embedding(filepath, model, feature_extractor)
    embeddings.append(emb)
    print(f" [{i+1}/5] {clip_labels[i]} — embedding shape: {emb.shape}")

embeddings_matrix = np.array(embeddings)
print(f"\nFull embedding matrix shape: {embeddings_matrix.shape}")

# Save embeddings and labels so they can be loaded by compare_cosine_similarity.py
np.savez("hubert_embeddings.npz", embeddings_matrix=embeddings_matrix, clip_labels=clip_labels)
print("Saved embeddings to hubert_embeddings.npz for later use.")

# --- t-SNE Visualization ---
# With only 5 points, use perplexity < n_samples
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
embeddings_2d = tsne.fit_transform(embeddings_matrix)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set1(np.linspace(0, 1, len(clips_5)))

for i, (x, y_coord) in enumerate(embeddings_2d):
    ax.scatter(x, y_coord, color=colors[i], s=200, zorder=3, edgecolors="black", linewidths=1.5)
    ax.annotate(
        clip_labels[i],
        (x, y_coord),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
    )

ax.set_title("t-SNE Visualization of HuBERT Embeddings (5 Audio Clips)", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.grid(True, alpha=0.3)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig("hubert_tsne_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nt-SNE visualization saved.")