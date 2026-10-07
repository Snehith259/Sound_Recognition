import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
# CQT uses logarithmically spaced frequency bins (ideal for music) 
C = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz("C1"), n_bins=84, 
bins_per_octave=12) 
 
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max) 
 
fig, ax = plt.subplots(figsize=(14, 5)) 
img = librosa.display.specshow( 
    C_db, sr=sr, hop_length=512, 
    x_axis="time", y_axis="cqt_note", ax=ax, cmap="magma" 
) 
ax.set_title("Constant-Q Transform (CQT) — Musical Notes on Y-axis", fontsize=13) 
fig.colorbar(img, ax=ax, format="%+2.0f dB") 
plt.tight_layout() 
plt.savefig("cqt_analysis.png", dpi=150, bbox_inches="tight") 
plt.show()