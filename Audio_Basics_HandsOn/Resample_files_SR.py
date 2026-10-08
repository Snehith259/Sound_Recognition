import os
from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 16000
OUTPUT_FOLDER = "./resampled_16k"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def list_audio_files(folder_path=".", extensions=(".wav", ".mp3", ".flac", ".ogg")):
    """Find audio files in directory."""
    folder = Path(folder_path).resolve()
    supported = {ext.lower() for ext in extensions}
    return sorted(
        str(p)
        for p in folder.glob("*")
        if p.is_file() and p.suffix.lower() in supported and "resampled_16k" not in p.parts
    )


def resample_file(input_path, output_folder, target_sr=16000):
    """Resample a single audio file and save."""
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, filename.rsplit(".", 1)[0] + ".wav")
    try:
        y, orig_sr = librosa.load(input_path, sr=None, mono=True)
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        sf.write(output_path, y, target_sr)
        return {"status": "success", "orig_sr": orig_sr, "output": output_path}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# Find audio files in current folder
audio_files = list_audio_files(".")

# Batch resample
success_count = 0
fail_count = 0

for filepath in audio_files:
    result = resample_file(filepath, OUTPUT_FOLDER, target_sr=TARGET_SR)
    if result["status"] == "success":
        success_count += 1
        print(f" SUCCESS: {os.path.basename(filepath)} ({result['orig_sr']} Hz -> {TARGET_SR} Hz)")
    else:
        fail_count += 1
        print(f" FAILED: {filepath} — {result['error']}")

print(f"\nResampling complete.")
print(f" Success: {success_count} files | Failed: {fail_count} files")
print(f" Output folder: {OUTPUT_FOLDER}")
