import librosa 
import soundfile as sf 
import numpy as np 
from IPython.display import Audio, display 
 
y, sr = librosa.load("1-137-A-32.wav", sr=22050) 
 
# Compute STFT 
D = librosa.stft(y, n_fft=2048, hop_length=512) 
 
# Perfect reconstruction (keep all information) 
y_reconstructed = librosa.istft(D, hop_length=512, length=len(y)) 
 
# Magnitude-only reconstruction (zero phase — lower quality) 
mag_only = np.abs(D) 
y_mag_only = librosa.istft(mag_only, hop_length=512, length=len(y)) 
 
# Save both 
sf.write("reconstructed_perfect.wav", y_reconstructed, sr) 
sf.write("reconstructed_mag_only.wav", y_mag_only, sr) 
 
# Compute reconstruction error 
error = np.mean((y - y_reconstructed) ** 2)

print(f"Original duration            : {len(y)/sr:.3f}s") 
print(f"Reconstructed duration       : {len(y_reconstructed)/sr:.3f}s") 
print(f"MSE (perfect reconstruction) : {error:.2e}") 
 
print("\n--- Listen to compare ---") 
print("Original:") 
display(Audio(data=y, rate=sr)) 
print("Perfect iSTFT reconstruction:") 
display(Audio(data=y_reconstructed, rate=sr)) 
print("Magnitude-only iSTFT (phase destroyed):") 
display(Audio(data=y_mag_only, rate=sr))