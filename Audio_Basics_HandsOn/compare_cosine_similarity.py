from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Fix: Load the embeddings saved by Hubert_t-Sne_visu.py
if not os.path.exists("hubert_embeddings.npz"):
    raise FileNotFoundError("hubert_embeddings.npz not found. Please run Hubert_t-Sne_visu.py first to extract and save the embeddings.")

data = np.load("hubert_embeddings.npz")
embeddings_matrix = data['embeddings_matrix']
clip_labels = data['clip_labels']

cos_sim = cosine_similarity(embeddings_matrix)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cos_sim, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=clip_labels, yticklabels=clip_labels,
            linewidths=0.5, vmin=-1, vmax=1, ax=ax)
ax.set_title("HuBERT Embedding Cosine Similarity (5 Clips)", fontsize=12, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig("hubert_cosine_similarity.png", dpi=150, bbox_inches="tight")
plt.show()
print("HuBERT similarity matrix saved.")
