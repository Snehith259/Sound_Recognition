import numpy as np
import librosa
import soundfile as sf


def explore_audio(filepath):
    """Comprehensive audio property explorer."""
    info = sf.info(filepath)

    subtype_to_bits = {
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
        "PCM_S8": 8,
        "PCM_U8": 8,
    }
    bit_depth = subtype_to_bits.get(info.subtype, "Unknown")

    y, sr = librosa.load(filepath, sr=None, mono=False)
    if y.ndim == 1:
        channels = "Mono"
        y_mono = y
    else:
        channels = f"Stereo ({y.shape[0]} channels)"
        y_mono = librosa.to_mono(y)

    duration = info.frames / info.samplerate
    rms = np.sqrt(np.mean(y_mono**2))
    peak_amp = np.abs(y_mono).max()
    dynamic_range = 20 * np.log10(peak_amp / (rms + 1e-10))
    db_rms = 20 * np.log10(rms + 1e-10)

    print("=" * 50)
    print("  AUDIO FILE PROPERTIES")
    print("=" * 50)
    print(f"  File Format    : {info.format}")
    print(f"  Subtype        : {info.subtype}")
    print(f"  Bit Depth      : {bit_depth} bits")
    print(f"  Sample Rate    : {info.samplerate} Hz")
    print(f"  Channels       : {channels}")
    print(f"  Duration       : {duration:.3f} seconds")
    print(f"  Total Samples  : {info.frames:,}")
    print(f"  Peak Amplitude : {peak_amp:.4f}")
    print(f"  RMS Amplitude  : {rms:.4f}")
    print(f"  RMS Level (dB) : {db_rms:.2f} dBFS")
    print(f"  Dynamic Range  : {dynamic_range:.2f} dB")
    print("=" * 50)


if __name__ == "__main__":
    explore_audio("1-137-A-32.wav")