from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 22050
MAX_DURATION_SECONDS = 5.0

# Speech: Marion Bryce (author), Garth Comira (reader), LibriSpeech SLR12,
# CC BY 4.0; https://www.openslr.org/12/
# Music: P. I. Tchaikovsky (composer), Kevin MacLeod (arranger),
# CC BY 4.0; https://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler/Dance_of_the_Sugar_Plum_Fairy
# Noise: deterministic synthetic white noise generated for this comparison.
audio_files = {
    "Speech": ROOT / "speech_sample.wav",
    "Music": ROOT / "music_sample.wav",
    "Noise (synthetic)": ROOT / "noise_sample.wav",
}

missing_files = [path.name for path in audio_files.values() if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        f"Missing audio sample(s): {', '.join(missing_files)}. "
        "Place the sample WAV files next to this script."
    )

fig, axes = plt.subplots(3, 3, figsize=(18, 11))
fig.suptitle(
    "Audio Type Comparison: Speech vs. Music vs. Noise",
    fontsize=14,
    fontweight="bold",
)

for col, (label, path) in enumerate(audio_files.items()):
    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )

    times = np.arange(len(y)) / sr
    axes[0, col].plot(times, y, color="steelblue", linewidth=0.5)
    axes[0, col].set_title(f"{label} — Waveform")
    axes[0, col].set_xlabel("Time (s)")
    axes[0, col].set_ylabel("Amplitude")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    image = librosa.display.specshow(
        log_mel,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, col],
        cmap="magma",
    )
    axes[1, col].set_title(f"{label} — Log-Mel Spectrogram")
    fig.colorbar(image, ax=axes[1, col], format="%+2.0f dB")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    image = librosa.display.specshow(
        mfccs,
        sr=sr,
        hop_length=512,
        x_axis="time",
        ax=axes[2, col],
        cmap="coolwarm",
    )
    axes[2, col].set_title(f"{label} — MFCCs")
    fig.colorbar(image, ax=axes[2, col])

fig.tight_layout()
output_path = ROOT / "three_audio_types_comparison.png"
fig.savefig(output_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"3-audio-type comparison saved to {output_path.name}.")