import librosa 
import librosa.display 
import matplotlib.pyplot as plt 
import numpy as np 
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
# Compute 13 MFCC coefficients (standard for speech) 
mfccs = librosa.feature.mfcc( 
    y=y, sr=sr, 
    n_mfcc=13,        # Number of coefficients 
    n_fft=2048, 
    hop_length=512, 
    n_mels=128 
) 
 
# Compute delta (velocity) and delta-delta (acceleration) features 
mfcc_delta  = librosa.feature.delta(mfccs) 
mfcc_delta2 = librosa.feature.delta(mfccs, order=2) 
 
print(f"MFCC shape             : {mfccs.shape}") 
print(f"  Rows (coefficients)  : {mfccs.shape[0]}") 
print(f"  Columns (time steps) : {mfccs.shape[1]}") 
print(f"Delta MFCC shape       : {mfcc_delta.shape}") 
print(f"Delta-Delta MFCC shape : {mfcc_delta2.shape}") 
print(f"Full feature vector    : {np.vstack([mfccs, mfcc_delta, mfcc_delta2]).shape}") 
 
# Visualize

fig, axes = plt.subplots(3, 1, figsize=(14, 10)) 
fig.suptitle("MFCC Features", fontsize=14, fontweight="bold") 
 
for ax, data, title, cmap in zip( 
    axes, 
    [mfccs, mfcc_delta, mfcc_delta2], 
    ["MFCCs (13 coefficients)", "Delta MFCCs (velocity)", "Delta-Delta MFCCs (acceleration)"], 
    ["coolwarm", "PuOr", "RdYlGn"] 
): 
    img = librosa.display.specshow(data, sr=sr, hop_length=512, x_axis="time", ax=ax, cmap=cmap) 
    ax.set_title(title) 
    ax.set_ylabel("MFCC Coefficient") 
    fig.colorbar(img, ax=ax) 
 
plt.tight_layout() 
plt.savefig("mfcc_features.png", dpi=150, bbox_inches="tight") 
plt.show() 