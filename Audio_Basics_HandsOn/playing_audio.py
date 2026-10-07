from pathlib import Path

import librosa
from IPython.display import Audio, display


audio_path = Path(__file__).with_name("1-137-A-32.wav")
y, sr = librosa.load(audio_path, sr=22050)

# Play directly from the file.
print("Playing from file:")
display(Audio(filename=str(audio_path)))

# Play from the NumPy array, which is useful after processing.
print("Playing from NumPy array:")
display(Audio(data=y, rate=sr))

# Play up to five seconds, even when the audio is shorter.
start_sec = 0
end_sec = min(5, len(y) / sr)
y_slice = y[int(start_sec * sr) : int(end_sec * sr)]
print(f"Playing slice [{start_sec}s - {end_sec:.2f}s]:")
display(Audio(data=y_slice, rate=sr))