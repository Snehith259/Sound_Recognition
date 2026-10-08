import math
import os
from pathlib import Path
import librosa
import soundfile as sf


def split_audio(
    input_path: Path | str,
    output_dir: Path | str,
    chunk_duration: float = 1.0,
    overlap: float = 0.0,
) -> list[str]:
    """
    Split an audio file into fixed-duration chunks and save them.

    Parameters:
        input_path: Path to the source audio file.
        output_dir: Directory where chunk files will be saved.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap duration in seconds between consecutive chunks.

    Returns:
        List of generated chunk file paths.
    """
    input_file = Path(input_path).resolve()
    output_folder = Path(output_dir).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_file}")

    # Load audio preserving original sample rate
    y, sr = librosa.load(str(input_file), sr=None, mono=False)

    total_samples = y.shape[-1]
    chunk_samples = int(chunk_duration * sr)
    step_samples = int((chunk_duration - overlap) * sr)

    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("chunk_duration must be greater than overlap and non-zero.")

    stem = input_file.stem
    chunk_paths = []
    chunk_index = 0

    for start_idx in range(0, total_samples, step_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        
        # Extract chunk (handles both mono [samples] and stereo [channels, samples])
        chunk = y[..., start_idx:end_idx]

        # Ignore tiny trailing snippets (less than 0.1s)
        if chunk.shape[-1] < int(0.1 * sr):
            break

        out_name = output_folder / f"{stem}_chunk_{chunk_index:03d}.wav"
        
        # soundfile expects (samples, channels) for stereo
        data_to_write = chunk.T if chunk.ndim > 1 else chunk
        sf.write(str(out_name), data_to_write, sr)

        chunk_paths.append(str(out_name))
        chunk_index += 1

    return chunk_paths


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    sample_file = current_dir / "1-137-A-32.wav"

    if sample_file.exists():
        out_folder = current_dir / "audio_chunks"
        print(f"Splitting audio file: {sample_file.name}")
        chunks = split_audio(sample_file, out_folder, chunk_duration=1.0, overlap=0.0)
        print(f"Created {len(chunks)} chunk(s) in: {out_folder}")
        for p in chunks[:5]:
            print(f"  {Path(p).name}")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more")
    else:
        print(f"Sample audio {sample_file.name} not found.")
