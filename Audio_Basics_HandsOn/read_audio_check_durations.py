from pathlib import Path
import soundfile as sf
import librosa


def get_audio_duration(file_path: Path | str) -> tuple[float, int, int]:
    """
    Get audio duration using soundfile (fast, doesn't load whole file into memory),
    falling back to librosa if needed.

    Returns:
        tuple of (duration_seconds, sample_rate, num_frames)
    """
    path_obj = Path(file_path)
    try:
        info = sf.info(str(path_obj))
        duration = info.duration
        samplerate = info.samplerate
        frames = info.frames
    except Exception:
        # Fallback to librosa if format is unsupported by soundfile
        duration = librosa.get_duration(path=str(path_obj))
        samplerate = librosa.get_samplerate(str(path_obj))
        frames = int(duration * samplerate)

    return duration, samplerate, frames


def check_audio_durations(folder_path: Path | str, extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg")):
    """
    Read audio files in the folder and print their durations, sample rates, and total samples.
    """
    folder = Path(folder_path).resolve()
    audio_files = sorted(
        p for p in folder.glob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not audio_files:
        print(f"No audio files found in: {folder}")
        return

    print(f"Found {len(audio_files)} audio file(s) in: {folder}\n")
    print(f"{'Filename':<35} | {'Duration (s)':<12} | {'Sample Rate (Hz)':<16} | {'Frames'}")
    print("-" * 80)

    total_duration = 0.0
    for file in audio_files:
        try:
            duration, sr, frames = get_audio_duration(file)
            total_duration += duration
            print(f"{file.name:<35} | {duration:<12.2f} | {sr:<16} | {frames:,}")
        except Exception as e:
            print(f"{file.name:<35} | Error reading file: {e}")

    print("-" * 80)
    print(f"Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    check_audio_durations(current_dir)
