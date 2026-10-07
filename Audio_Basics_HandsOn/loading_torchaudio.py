import wave

import numpy as np
import torchaudio


def load_wav_fallback(path):
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw_audio = wav_file.readframes(frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    int_samples = np.frombuffer(raw_audio, dtype=dtype_map[sample_width])
    samples = int_samples.reshape(-1, channels).astype(np.float32)

    if np.issubdtype(int_samples.dtype, np.unsignedinteger):
        info = np.iinfo(int_samples.dtype)
        samples = (samples - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
    elif np.issubdtype(int_samples.dtype, np.signedinteger):
        info = np.iinfo(int_samples.dtype)
        samples = samples / float(abs(info.min))

    return samples.T, sample_rate


file_path = "1-137-A-32.wav"

try:
    waveform, sample_rate = torchaudio.load(file_path)
except Exception as exc:
    print(f"torchaudio failed: {exc}")
    print("Falling back to Python's built-in WAV loader...")
    waveform, sample_rate = load_wav_fallback(file_path)

# torchaudio returns shape: [channels, samples]
print(f"Waveform shape         : {waveform.shape}")
print(f"Sample rate            : {sample_rate} Hz")
print(f"Number of channels     : {waveform.shape[0]}")
print(f"Number of samples      : {waveform.shape[1]}")
print(f"Duration               : {waveform.shape[1] / sample_rate:.2f} seconds")
print(f"Dtype                  : {waveform.dtype}")
print(f"Max amplitude          : {np.abs(waveform).max():.4f}")

# Convert stereo to mono if needed
if waveform.shape[0] == 2:
    mono = waveform.mean(axis=0, keepdims=True)
    print(f"\nConverted to mono shape: {mono.shape}")
