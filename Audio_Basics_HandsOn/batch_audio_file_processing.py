from pathlib import Path


def list_audio_files(
    folder_path: str | Path,
    extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg"),
) -> list[Path]:
    """Return sorted audio files under a directory, excluding virtual environments."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Audio folder does not exist: {folder}")

    supported_extensions = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
        and ".venv" not in path.parts
        and "venv" not in path.parts
    )


folder = Path(__file__).resolve().parent
audio_files = list_audio_files(folder)

print(f"Searching: {folder}")
print(f"Found {len(audio_files)} audio files:")
for audio_file in audio_files[:10]:
    print(f"  {audio_file}")
if len(audio_files) > 10:
    print(f"  ... and {len(audio_files) - 10} more")