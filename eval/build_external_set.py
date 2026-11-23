"""Download and normalize public external audio used for blind evaluation.

The builder never feeds external audio into training. It assigns whole source
recordings to dev/final before making any predictions. Final evaluation is an
explicit opt-in in eval_external.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "external"
MANIFEST = ROOT / "eval" / "external_manifest.csv"
USER_AGENT = "SoundSenseEvaluation/1.0 (research benchmark; no training use)"

ANIMAL_PREFIXES = {
    "kedi": ("cat", [5], True),
    "kopek": ("dog", [0], True),
    "inek": ("cow", [3], True),
    "koyun": ("sheep", [8], True),
    "kurbaga": ("frog", [4], True),
    "kus": ("chirping_birds", [14, 9], True),
    "tavuk": ("hen", [6, 1], True),
    "aslan": ("__ood__", [], False),
    "esek": ("__ood__", [], False),
    "maymun": ("__ood__", [], False),
}
IBM_LABELS = {
    "applause": ("clapping", [22], True),
    "birds1": ("chirping_birds", [14], True),
    "birds2": ("chirping_birds", [14], True),
    "car-horn": ("car_horn", [43], True),
    "car_engine": ("engine", [44], True),
    "dog": ("dog", [0], True),
    "fireworks": ("fireworks", [48], True),
    "rain": ("rain", [10], True),
    "siren": ("siren", [42], True),
    "snoring": ("snoring", [28], True),
    "thunder": ("thunderstorm", [19], True),
    "train": ("train", [45], True),
    "typing": ("keyboard_typing", [32], True),
    "gunshots": ("__ood__", [], False),
    "guitar": ("__ood__", [], False),
    "jazz_guitar": ("__ood__", [], False),
    "piano": ("__ood__", [], False),
    "arcade_music": ("__ood__", [], False),
    "cinematic": ("__ood__", [], False),
    "crowd_cheering": ("__ood__", [], False),
}


def _request(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _github_tree(owner: str, repo: str) -> list[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    data = json.loads(_request(url))
    return data.get("tree", [])


def _transcode(data: bytes, source_name: str) -> tuple[np.ndarray, int]:
    suffix = Path(source_name).suffix or ".wav"
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", "pipe:1"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode {source_name}: {proc.stderr.decode(errors='replace')[-400:]}")
    y, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 22050:
        raise RuntimeError(f"Normalization failed for {source_name}: expected 22050 Hz, got {sr} Hz")
    return np.asarray(y, dtype=np.float32), sr


def _safe_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _assign_splits(records: list[dict], dev_fraction: float, seed: int) -> None:
    sources = sorted({item["source_recording"] for item in records})
    random.Random(seed).shuffle(sources)
    dev_count = max(1, round(len(sources) * dev_fraction)) if sources else 0
    dev_sources = set(sources[:dev_count])
    for item in records:
        item["split"] = "dev" if item["source_recording"] in dev_sources else "final"


def _write_audio(path: Path, audio: np.ndarray, sr: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def build(dev_fraction: float = 0.30, seed: int = 0, animals_per_folder: int = 10) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    animal_tree = _github_tree("YashNita", "Animal-Sound-Dataset")
    animal_files: dict[str, list[str]] = {}
    for item in animal_tree:
        path = item.get("path", "")
        if path.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
            animal_files.setdefault(path.split("/")[0], []).append(path)
    rng = random.Random(seed)
    for folder, paths in sorted(animal_files.items()):
        folder_key = folder.lower()
        mapping = next((v for prefix, v in ANIMAL_PREFIXES.items() if folder_key.startswith(prefix)), None)
        if mapping is None:
            continue
        label, allowed, in_scope = mapping
        rng.shuffle(paths)
        for source_path in paths[:animals_per_folder]:
            encoded_path = urllib.parse.quote(source_path, safe="/")
            url = f"https://raw.githubusercontent.com/YashNita/Animal-Sound-Dataset/HEAD/{encoded_path}"
            raw = _request(url)
            y, sr = _transcode(raw, source_path)
            stem = Path(source_path).stem
            source_id = f"animals:{_safe_id(source_path)}"
            filename = f"animals_{_safe_id(source_path)}.wav"
            _write_audio(OUT_DIR / "animals" / filename, y, sr)
            records.append({
                "file": f"animals/{filename}", "source_url": url, "source_dataset": "YashNita/Animal-Sound-Dataset",
                "source_recording": source_id, "source_name": stem, "label": label,
                "allowed_labels": json.dumps(allowed), "in_scope": str(in_scope).lower(),
            })

    ibm_tree = _github_tree("IBM", "MAX-Audio-Classifier")
    for item in ibm_tree:
        path = item.get("path", "")
        if not path.lower().startswith("samples/") or not path.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
            continue
        stem = Path(path).stem.lower()
        mapping = IBM_LABELS.get(stem)
        if mapping is None:
            continue
        label, allowed, in_scope = mapping
        encoded_path = urllib.parse.quote(path, safe="/")
        url = f"https://raw.githubusercontent.com/IBM/MAX-Audio-Classifier/HEAD/{encoded_path}"
        y, sr = _transcode(_request(url), path)
        source_id = f"ibm:{_safe_id(path)}"
        filename = f"ibm_{_safe_id(path)}.wav"
        _write_audio(OUT_DIR / "ibm" / filename, y, sr)
        records.append({
            "file": f"ibm/{filename}", "source_url": url, "source_dataset": "IBM/MAX-Audio-Classifier",
            "source_recording": source_id, "source_name": stem, "label": label,
            "allowed_labels": json.dumps(allowed), "in_scope": str(in_scope).lower(),
        })

    soundscape_path = "birdnet_analyzer/example/soundscape.wav"
    soundscape_url = f"https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/HEAD/{soundscape_path}"
    try:
        y, sr = _transcode(_request(soundscape_url, timeout=120), soundscape_path)
        source_id = "birdnet:example-soundscape"
        # One source recording cannot be divided across splits without leakage.
        # Keep this single ambient source in dev for threshold selection.
        for idx, start in enumerate(range(0, len(y), 5 * sr)):
            chunk = y[start:start + 5 * sr]
            if len(chunk) < sr:
                continue
            filename = f"birdnet_window_{idx:03d}.wav"
            _write_audio(OUT_DIR / "ambient" / filename, chunk, sr)
            records.append({
                "file": f"ambient/{filename}", "source_url": soundscape_url, "source_dataset": "kahst/BirdNET-Analyzer",
                "source_recording": source_id, "source_name": f"window_{idx:03d}",
                "label": "chirping_birds", "allowed_labels": json.dumps([14, 7, 13, 16]),
                "in_scope": "true", "split": "dev",
            })
    except Exception as exc:
        print(f"Ambient soundscape unavailable: {exc}")

    owner_manifest = ROOT / "eval" / "owner_recorded_manifest.csv"
    if owner_manifest.exists():
        with owner_manifest.open(newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                file_path = (ROOT / item["file"]).resolve()
                if not file_path.is_file():
                    print(f"Skipping missing owner recording: {item['file']}")
                    continue
                y, sr = _transcode(file_path.read_bytes(), file_path.name)
                filename = f"owner_{_safe_id(str(file_path))}.wav"
                _write_audio(OUT_DIR / "owner" / filename, y, sr)
                records.append({
                    "file": f"owner/{filename}", "source_url": "owner_recorded", "source_dataset": "owner_recorded",
                    "source_recording": f"owner:{_safe_id(str(file_path))}", "source_name": file_path.stem,
                    "label": item.get("label", "__ood__"), "allowed_labels": item.get("allowed_labels", "[]"),
                    "in_scope": str(item.get("in_scope", "false")).lower(),
                })

    # Assign source groups as units; the ambient soundscape remains wholly in dev.
    automatic = [r for r in records if "split" not in r]
    _assign_splits(automatic, dev_fraction, seed)
    records.sort(key=lambda item: item["file"])
    fieldnames = ["file", "source_url", "source_dataset", "source_recording", "source_name", "label", "allowed_labels", "in_scope", "split"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--animals-per-folder", type=int, default=10)
    args = parser.parse_args()
    built = build(args.dev_fraction, args.seed, args.animals_per_folder)
    counts = {split: sum(row["split"] == split for row in built) for split in ("dev", "final")}
    print(f"Built {len(built)} clips: {counts}; manifest={MANIFEST}")
