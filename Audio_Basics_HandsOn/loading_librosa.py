import librosa 
import numpy as np 
 
# librosa.load() always returns: 
#   y  : audio time series as a 1D NumPy float32 array (mono by default) 
#   sr : sample rate (default resamples to 22050 Hz) 
 
audio_path = "1-137-A-32.wav"   # Replace with your file path 
 
# Load with default resampling to 22050 Hz 
y, sr = librosa.load(audio_path, sr=22050) 
 
print(f"Audio array shape      : {y.shape}") 
print(f"Sample rate            : {sr} Hz") 
print(f"Duration               : {len(y) / sr:.2f} seconds") 
print(f"Data type              : {y.dtype}") 
print(f"Min value              : {y.min():.4f}")
print(f"Max value              : {y.max():.4f}") 
print(f"Total samples          : {len(y)}") 