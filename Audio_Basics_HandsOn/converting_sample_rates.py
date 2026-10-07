from pathlib import Path

import librosa
import soundfile as sf
import torch
import torchaudio.transforms as T


audio_path = Path(__file__).with_name("1-137-A-32.wav")
y, sr = librosa.load(audio_path, sr=None)  # Load at original rate

print(f"Original sample rate   : {sr} Hz  |  Samples: {len(y)}")

target_rates = [8000, 16000, 22050, 44100, 48000]

for target_sr in target_rates:
    if target_sr == sr:
        print(f"Target {target_sr:>6} Hz  |  (same as original, skipped)")
        continue

    # --- Method 1: librosa resample ---
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    output_path = audio_path.with_name(f"resampled_{target_sr}Hz.wav")
    sf.write(output_path, y_resampled, target_sr)

    print(
        f"Target {target_sr:>6} Hz  |  Samples: {len(y_resampled):>8,}"
        f"  |  Saved: {output_path.name}"
    )

# --- Method 2: torchaudio Resample transform ---
# Use librosa and soundfile for file I/O; torchaudio.load/save require TorchCodec.
waveform = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
resampler = T.Resample(orig_freq=sr, new_freq=16000)
resampled_torch = resampler(waveform).squeeze(0).cpu().numpy()
torch_output_path = audio_path.with_name("resampled_torchaudio_16000Hz.wav")
sf.write(torch_output_path, resampled_torch, 16000)
print(f"\ntorchaudio resampled to 16000 Hz and saved to {torch_output_path.name}.")
