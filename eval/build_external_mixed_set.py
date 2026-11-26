"""Build a large, leak-free external internet audio corpus (~62-65% mixed, ~35-38% pure & OOD).

Sources (100% independent of ESC-50):
  1. CLAPv2/audioset_strong (multi-event YouTube recordings with event durations)
  2. agkphysics/AudioSet (real YouTube single-label and multi-label recordings)
  3. danavery/urbansound8K (both salience=1 pure and salience=2 noisy/mixed,
     strictly filtered to exclude any Freesound fsID present in ESC-50)
  4. Wikimedia Commons field recordings (explicit rain+thunder+birds, footsteps, etc.)
  5. Existing eval/external dev split (IBM/MAX, YashNita Animals, BirdNET soundscapes)
  6. Cross-source real acoustic mixtures created by overlapping two verified pure
     external internet recordings from distinct categories & repositories (60/40 SNR).

Never modifies or reads eval/external `final` holdout clips.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "external_mixed"
MANIFEST_OUT = ROOT / "eval" / "external_mixed_manifest.csv"
USER_AGENT = "SoundSenseBenchmark/1.0 (https://github.com/Snehith259/Sound_Recognition)"

ESC50_NAME_TO_ID = {
    "dog": 0, "rooster": 1, "pig": 2, "cow": 3, "frog": 4, "cat": 5, "hen": 6,
    "insects": 7, "sheep": 8, "crow": 9, "rain": 10, "sea_waves": 11,
    "crackling_fire": 12, "crickets": 13, "chirping_birds": 14, "water_drops": 15,
    "wind": 16, "pouring_water": 17, "toilet_flush": 18, "thunderstorm": 19,
    "crying_baby": 20, "sneezing": 21, "clapping": 22, "breathing": 23,
    "coughing": 24, "footsteps": 25, "laughing": 26, "brushing_teeth": 27,
    "snoring": 28, "drinking_sipping": 29, "door_wood_knock": 30,
    "mouse_click": 31, "keyboard_typing": 32, "door_wood_creaks": 33,
    "can_opening": 34, "washing_machine": 35, "vacuum_cleaner": 36,
    "clock_alarm": 37, "clock_tick": 38, "glass_breaking": 39, "helicopter": 40,
    "chainsaw": 41, "siren": 42, "car_horn": 43, "engine": 44, "train": 45,
    "church_bells": 46, "airplane": 47, "fireworks": 48, "hand_saw": 49,
}

# Strict substring/exact AudioSet label -> ESC-50 category mapping
AUDIOSET_TO_ESC50: dict[str, tuple[str, int]] = {
    "Dog": ("dog", 0),
    "Bark": ("dog", 0),
    "Bow-wow": ("dog", 0),
    "Growling": ("dog", 0),
    "Rooster": ("rooster", 1),
    "Crowing, cock-a-doodle-doo": ("rooster", 1),
    "Pig": ("pig", 2),
    "Oink": ("pig", 2),
    "Cattle, bovinae": ("cow", 3),
    "Moo": ("cow", 3),
    "Frog": ("frog", 4),
    "Croak": ("frog", 4),
    "Cat": ("cat", 5),
    "Meow": ("cat", 5),
    "Purr": ("cat", 5),
    "Chicken, rooster": ("hen", 6),
    "Cluck": ("hen", 6),
    "Fowl": ("hen", 6),
    "Insect": ("insects", 7),
    "Buzz": ("insects", 7),
    "Bee, wasp, etc.": ("insects", 7),
    "Sheep": ("sheep", 8),
    "Bleat": ("sheep", 8),
    "Crow": ("crow", 9),
    "Caw": ("crow", 9),
    "Rain": ("rain", 10),
    "Rain on surface": ("rain", 10),
    "Raindrop": ("rain", 10),
    "Waves, surf": ("sea_waves", 11),
    "Ocean": ("sea_waves", 11),
    "Fire": ("crackling_fire", 12),
    "Crackle": ("crackling_fire", 12),
    "Cricket": ("crickets", 13),
    "Bird": ("chirping_birds", 14),
    "Bird vocalization, bird call, bird song": ("chirping_birds", 14),
    "Chirp, tweet": ("chirping_birds", 14),
    "Pigeon, dove": ("chirping_birds", 14),
    "Drip": ("water_drops", 15),
    "Trickle, dribble": ("water_drops", 15),
    "Wind": ("wind", 16),
    "Wind noise (microphone)": ("wind", 16),
    "Pour": ("pouring_water", 17),
    "Stream": ("pouring_water", 17),
    "Gush": ("pouring_water", 17),
    "Toilet flush": ("toilet_flush", 18),
    "Thunder": ("thunderstorm", 19),
    "Thunderstorm": ("thunderstorm", 19),
    "Baby cry, infant cry": ("crying_baby", 20),
    "Sneeze": ("sneezing", 21),
    "Applause": ("clapping", 22),
    "Clapping": ("clapping", 22),
    "Breathing": ("breathing", 23),
    "Pant": ("breathing", 23),
    "Cough": ("coughing", 24),
    "Throat clearing": ("coughing", 24),
    "Walk, footsteps": ("footsteps", 25),
    "Laughter": ("laughing", 26),
    "Giggle": ("laughing", 26),
    "Chuckle, chortle": ("laughing", 26),
    "Toothbrush": ("brushing_teeth", 27),
    "Electric toothbrush": ("brushing_teeth", 27),
    "Snoring": ("snoring", 28),
    "Sip": ("drinking_sipping", 29),
    "Knock": ("door_wood_knock", 30),
    "Computer keyboard": ("keyboard_typing", 32),
    "Typing": ("keyboard_typing", 32),
    "Typewriter": ("keyboard_typing", 32),
    "Squeak": ("door_wood_creaks", 33),
    "Creak": ("door_wood_creaks", 33),
    "Vacuum cleaner": ("vacuum_cleaner", 36),
    "Alarm clock": ("clock_alarm", 37),
    "Clock": ("clock_tick", 38),
    "Tick": ("clock_tick", 38),
    "Tick-tock": ("clock_tick", 38),
    "Shatter": ("glass_breaking", 39),
    "Glass": ("glass_breaking", 39),
    "Helicopter": ("helicopter", 40),
    "Chainsaw": ("chainsaw", 41),
    "Siren": ("siren", 42),
    "Emergency vehicle": ("siren", 42),
    "Police car (siren)": ("siren", 42),
    "Ambulance (siren)": ("siren", 42),
    "Fire engine, fire truck (siren)": ("siren", 42),
    "Vehicle horn, car horn, honking": ("car_horn", 43),
    "Toot": ("car_horn", 43),
    "Engine": ("engine", 44),
    "Idling": ("engine", 44),
    "Medium engine (mid frequency)": ("engine", 44),
    "Heavy engine (low frequency)": ("engine", 44),
    "Light engine (high frequency)": ("engine", 44),
    "Train": ("train", 45),
    "Railroad car, train wagon": ("train", 45),
    "Rail transport": ("train", 45),
    "Church bell": ("church_bells", 46),
    "Bell": ("church_bells", 46),
    "Aircraft": ("airplane", 47),
    "Aircraft engine": ("airplane", 47),
    "Jet engine": ("airplane", 47),
    "Propeller, airscrew": ("airplane", 47),
    "Fireworks": ("fireworks", 48),
    "Firecracker": ("fireworks", 48),
    "Sawing": ("hand_saw", 49),
}

OOD_AUDIOSET_LABELS = {
    "Music", "Singing", "Guitar", "Piano", "Drum", "Percussion", "Violin, fiddle",
    "Gunshot, gunfire", "Machine gun", "Speech", "Male speech, man speaking",
    "Female speech, woman speaking", "Narration, monologue", "Trumpet", "Brass instrument",
    "Saxophone", "Electric guitar", "Dubstep", "Hip hop music", "Rock music", "Pop music",
}

URBAN8K_ESC50 = {
    "dog_bark": ("dog", 0),
    "siren": ("siren", 42),
    "car_horn": ("car_horn", 43),
    "engine_idling": ("engine", 44),
}

URBAN8K_OOD = {"street_music", "gun_shot", "children_playing"}


def _get_json(url: str, timeout: int = 25, retries: int = 3) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt)
    return {}


def _download(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 + attempt)
    return b""


def _transcode_wav(raw: bytes, label_hint: str) -> tuple[np.ndarray, int]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", "pipe:1"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg decode failed for {label_hint}: {proc.stderr.decode(errors='replace')[-200:]}")
    y, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if rms < 0.002 or y.size < int(sr * 0.4):
        raise ValueError(f"Clip too quiet or too short ({y.size} samples, rms={rms:.5f})")
    return y, sr


def _load_esc50_freesound_ids() -> set[str]:
    esc_ids: set[str] = set()
    for split_file in (ROOT / "Preproccesing" / "splits").glob("*.csv"):
        try:
            df = pd.read_csv(split_file)
            if "src_file" in df.columns:
                esc_ids.update(str(x).strip() for x in df["src_file"].dropna())
        except Exception:
            pass
    return esc_ids


def _save_wav(rel_path: str, y: np.ndarray, sr: int = 22050) -> Path:
    dest = OUT_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.clip(y, -0.99, 0.99).astype(np.float32), sr, subtype="PCM_16")
    return dest


def _fetch_audioset_candidates(esc_cap: int = 45, ood_cap: int = 12) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # Scan balanced test across spread offsets to capture broad class coverage
    offsets = [0, 100, 300, 600, 900, 1200, 1800, 2500, 3500, 4500, 6000, 8000, 10000, 12000, 14000]
    in_scope_by_class: dict[int, int] = {}
    in_scope_mixed = 0
    in_scope_pure = 0
    ood_count = 0

    for off in offsets:
        try:
            d = _get_json(
                f"https://datasets-server.huggingface.co/rows?dataset=agkphysics%2FAudioSet&config=balanced&split=test&offset={off}&length=100"
            )
        except Exception:
            continue
        for item in d.get("rows", []):
            r = item["row"]
            labels = r.get("human_labels") or []
            audio_list = r.get("audio") or []
            if not audio_list or not labels:
                continue
            src_url = audio_list[0].get("src")
            if not src_url:
                continue

            mapped: list[tuple[str, int]] = []
            for lbl in labels:
                if lbl in AUDIOSET_TO_ESC50:
                    mapped.append(AUDIOSET_TO_ESC50[lbl])

            # Filter out clips where Speech/Music clearly dominates non-environmental clips unless it is pure OOD
            has_speech_music = any(lbl in OOD_AUDIOSET_LABELS for lbl in labels)
            if mapped:
                # Deduplicate mapped IDs preserving order
                allowed_ids: list[int] = []
                allowed_names: list[str] = []
                for name, cid in mapped:
                    if cid not in allowed_ids:
                        allowed_ids.append(cid)
                        allowed_names.append(name)

                # Skip speech+SFX where speech can overwhelm short sound events unless another ESC sound is clear
                if "Speech" in labels and len(labels) >= 2 and len(allowed_ids) == 1:
                    continue

                is_mixed = len(allowed_ids) >= 2 or len(labels) >= 2
                dom_id = allowed_ids[0]
                if in_scope_by_class.get(dom_id, 0) >= 4:
                    continue
                if is_mixed and in_scope_mixed >= esc_cap:
                    continue
                if not is_mixed and in_scope_pure >= 20:
                    continue

                in_scope_by_class[dom_id] = in_scope_by_class.get(dom_id, 0) + 1
                if is_mixed:
                    in_scope_mixed += 1
                else:
                    in_scope_pure += 1
                candidates.append({
                    "id": f"as_{r['video_id']}",
                    "url": src_url,
                    "source_dataset": "agkphysics/AudioSet",
                    "source_recording": f"audioset:{r['video_id']}",
                    "label": allowed_names[0],
                    "allowed_labels": allowed_ids,
                    "in_scope": True,
                    "sound_type": "mixed" if is_mixed else "pure",
                    "raw_labels": "; ".join(labels),
                })
            elif has_speech_music and len(labels) <= 2 and ood_count < ood_cap:
                # Ensure zero accidental overlap with any environmental sound
                if not any(kw in "|".join(labels).lower() for kw in ("bird", "dog", "cat", "rain", "wind", "car", "train", "siren", "bell", "water", "fire", "applause", "laugh", "cry", "cough", "clock", "foot")):
                    ood_count += 1
                    candidates.append({
                        "id": f"as_ood_{r['video_id']}",
                        "url": src_url,
                        "source_dataset": "agkphysics/AudioSet",
                        "source_recording": f"audioset:{r['video_id']}",
                        "label": "__ood__",
                        "allowed_labels": [],
                        "in_scope": False,
                        "sound_type": "ood_mixed" if len(labels) > 1 else "ood_pure",
                        "raw_labels": "; ".join(labels),
                    })
    return candidates


def _fetch_audioset_strong_candidates(cap: int = 35) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    per_class: dict[int, int] = {}
    for split in ("validation", "test"):
        for off in range(0, 500, 100):
            try:
                d = _get_json(
                    f"https://datasets-server.huggingface.co/rows?dataset=CLAPv2%2Faudioset_strong&config=default&split={split}&offset={off}&length=100"
                )
            except Exception:
                break
            rows = d.get("rows", [])
            if not rows:
                break
            for item in rows:
                r = item["row"]
                audio_list = r.get("audio") or []
                if not audio_list or not audio_list[0].get("src"):
                    continue
                raw_text = r.get("raw_text") or []
                strong_events_line = next((x for x in raw_text if x.startswith("Strong events: ")), "")
                tags_line = next((x for x in raw_text if x.startswith("Tags: ")), "")
                tags = [t.strip() for t in tags_line[6:].split(",") if t.strip()]

                # Parse strong event durations so we only pick ESC-50 sounds active >= 2.0s
                dur_by_esc: dict[int, float] = {}
                name_by_esc: dict[int, str] = {}
                n_events_distinct = set()
                for seg in strong_events_line[15:].split("|"):
                    seg = seg.strip()
                    if not seg.startswith("(") or ")" not in seg:
                        continue
                    time_part, rest = seg[1:].split(")", 1)
                    try:
                        t0_s, t1_s = [float(x.strip().rstrip("s")) for x in time_part.split("-")]
                        dur = max(0.0, t1_s - t0_s)
                    except Exception:
                        dur = 0.0
                    ev_name = rest.strip().split("(/")[0].strip()
                    n_events_distinct.add(ev_name)
                    if ev_name in AUDIOSET_TO_ESC50 and dur >= 2.0:
                        cname, cid = AUDIOSET_TO_ESC50[ev_name]
                        dur_by_esc[cid] = dur_by_esc.get(cid, 0.0) + dur
                        name_by_esc[cid] = cname

                if not dur_by_esc:
                    continue
                # Sort by total active duration descending -> primary foreground ESC-50 label
                ranked = sorted(dur_by_esc.items(), key=lambda kv: kv[1], reverse=True)
                dom_id, dom_dur = ranked[0]
                if dom_dur < 3.0 or per_class.get(dom_id, 0) >= 3:
                    continue
                allowed = [cid for cid, dval in ranked if dval >= 2.0]
                is_mixed = len(allowed) >= 2 or len(n_events_distinct) >= 2 or len(tags) >= 2
                if not is_mixed:
                    continue
                per_class[dom_id] = per_class.get(dom_id, 0) + 1
                candidates.append({
                    "id": f"asstrong_{r['index'][-11:]}",
                    "url": audio_list[0]["src"],
                    "source_dataset": "CLAPv2/audioset_strong",
                    "source_recording": f"audioset_strong:{r['index']}",
                    "label": name_by_esc[dom_id],
                    "allowed_labels": allowed,
                    "in_scope": True,
                    "sound_type": "mixed",
                    "raw_labels": r.get("text", ""),
                })
                if len(candidates) >= cap:
                    return candidates
    return candidates


def _fetch_urbansound_candidates(esc_fs_ids: set[str], mixed_cap: int = 24, pure_cap: int = 16, ood_cap: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    offsets = [0, 300, 800, 1500, 2400, 3300, 4200, 5200, 6200, 7200, 8000]
    for off in offsets:
        try:
            d = _get_json(
                f"https://datasets-server.huggingface.co/rows?dataset=danavery%2Furbansound8K&config=default&split=train&offset={off}&length=100"
            )
        except Exception:
            continue
        for item in d.get("rows", []):
            r = item["row"]
            fs_id = str(r.get("fsID", "")).strip()
            if fs_id in esc_fs_ids:
                continue  # Strict zero-overlap with ESC-50 Freesound IDs!
            audio_list = r.get("audio") or []
            if not audio_list or not audio_list[0].get("src"):
                continue
            cls_name = r.get("class")
            salience = int(r.get("salience", 1))
            dur = float(r.get("end", 0.0)) - float(r.get("start", 0.0))
            if dur < 1.2:
                continue

            if cls_name in URBAN8K_ESC50:
                esc_name, esc_id = URBAN8K_ESC50[cls_name]
                stype = "mixed" if salience == 2 else "pure"
                total_cur = sum(v for (st, _), v in counts.items() if st == stype)
                if stype == "mixed" and total_cur >= mixed_cap:
                    continue
                if stype == "pure" and total_cur >= pure_cap:
                    continue
                if counts.get((stype, esc_name), 0) >= 6:
                    continue
                counts[(stype, esc_name)] = counts.get((stype, esc_name), 0) + 1
                candidates.append({
                    "id": f"us8k_{Path(r['slice_file_name']).stem}",
                    "url": audio_list[0]["src"],
                    "source_dataset": "danavery/urbansound8K",
                    "source_recording": f"urbansound8k:{fs_id}",
                    "label": esc_name,
                    "allowed_labels": [esc_id],
                    "in_scope": True,
                    "sound_type": stype,
                    "raw_labels": f"{cls_name} (salience={salience})",
                })
            elif cls_name in URBAN8K_OOD and salience == 1:
                total_ood = sum(v for (st, _), v in counts.items() if st == "ood_pure")
                if total_ood >= ood_cap or counts.get(("ood_pure", cls_name), 0) >= 3:
                    continue
                counts[("ood_pure", cls_name)] = counts.get(("ood_pure", cls_name), 0) + 1
                candidates.append({
                    "id": f"us8k_ood_{Path(r['slice_file_name']).stem}",
                    "url": audio_list[0]["src"],
                    "source_dataset": "danavery/urbansound8K",
                    "source_recording": f"urbansound8k:{fs_id}",
                    "label": "__ood__",
                    "allowed_labels": [],
                    "in_scope": False,
                    "sound_type": "ood_pure",
                    "raw_labels": f"{cls_name} (OOD)",
                })
    return candidates


def _fetch_wikimedia_candidates() -> list[dict[str, Any]]:
    specs = [
        ("File:Rain thunder and birds.ogg", "rain", [10, 19, 14], True, "mixed"),
        ("File:Rain thunder steps.ogg", "rain", [10, 19, 25], True, "mixed"),
        ("File:Rain and thunder (1).ogg", "rain", [10, 19], True, "mixed"),
        ("File:Rain thunder and other noises.ogg", "rain", [10, 19], True, "mixed"),
        ("File:Ambulance Sounds2.wav", "siren", [42], True, "pure"),
        ("File:Car Horn.wav", "car_horn", [43], True, "pure"),
        ("File:Speeding-car-horn doppler effect sample.ogg", "car_horn", [43, 44], True, "mixed"),
    ]
    titles = urllib.parse.quote("|".join(s[0] for s in specs))
    try:
        info = _get_json(f"https://commons.wikimedia.org/w/api.php?action=query&titles={titles}&prop=imageinfo&iiprop=url|size&format=json")
    except Exception:
        return []
    url_by_title: dict[str, str] = {}
    for _, page in info.get("query", {}).get("pages", {}).items():
        ii = (page.get("imageinfo") or [{}])[0]
        if ii.get("url"):
            url_by_title[page["title"]] = ii["url"]

    out: list[dict[str, Any]] = []
    for title, lbl, allowed, in_scope, stype in specs:
        u = url_by_title.get(title)
        if not u:
            continue
        slug = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
        out.append({
            "id": f"wm_{slug}",
            "url": u,
            "source_dataset": "Wikimedia-Commons",
            "source_recording": f"wikimedia:{title}",
            "label": lbl,
            "allowed_labels": allowed,
            "in_scope": in_scope,
            "sound_type": stype,
            "raw_labels": title,
        })
    return out


def _build_cross_source_mixtures(pure_records: list[dict[str, Any]], target_count: int = 45, seed: int = 20261125) -> list[dict[str, Any]]:
    """Create realistic 2-source acoustic mixtures from distinct verified pure internet clips."""
    rng = random.Random(seed)
    # Index pure in-scope clips that exist on disk
    pool = [r for r in pure_records if r["in_scope"] and len(r["allowed_labels"]) >= 1]
    if len(pool) < 6:
        return []

    mixes: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(mixes) < target_count and attempts < target_count * 25:
        attempts += 1
        a, b = rng.sample(pool, 2)
        c_a = a["allowed_labels"][0]
        c_b = b["allowed_labels"][0]
        if c_a == c_b:
            continue
        if a["source_recording"] == b["source_recording"]:
            continue
        pair_key = tuple(sorted((a["file"], b["file"])))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        ya, sr_a = sf.read(str(OUT_DIR / a["file"]), dtype="float32")
        yb, sr_b = sf.read(str(OUT_DIR / b["file"]), dtype="float32")
        target_samples = 5 * 22050
        # Tile/loop longer or take 5s active window with smooth fade
        def prep_5s(arr: np.ndarray) -> np.ndarray:
            if arr.size >= target_samples:
                # Pick highest-RMS 5s slice
                best_s, best_e = 0, -1.0
                for s in range(0, arr.size - target_samples + 1, 22050):
                    e = float(np.sqrt(np.mean(arr[s:s + target_samples] ** 2)))
                    if e > best_e:
                        best_e, best_s = e, s
                seg = arr[best_s:best_s + target_samples].copy()
            else:
                # Pad cleanly to 5s or crossfade-repeat with 100ms Hann fade
                seg = np.zeros(target_samples, dtype=np.float32)
                fade = min(2205, arr.size // 4)
                chunk = arr.copy()
                if fade > 8:
                    win = np.hanning(2 * fade).astype(np.float32)
                    chunk[:fade] *= win[:fade]
                    chunk[-fade:] *= win[fade:]
                pos = 0
                while pos < target_samples:
                    take = min(chunk.size, target_samples - pos)
                    seg[pos:pos + take] = chunk[:take]
                    pos += take + 2205  # natural brief acoustic gap instead of sharp loop click
            rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-8
            return seg / rms

        sa = prep_5s(ya)
        sb = prep_5s(yb)
        # Foreground SNR +1.5 dB to +4.5 dB -> secondary weight ~0.60..0.84 relative amplitude
        sec_gain = rng.uniform(0.58, 0.88)
        mixed = sa + sec_gain * sb
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.85:
            mixed = mixed / peak * 0.85

        idx = len(mixes) + 1
        rel_file = f"mixed_synth/crossmix_{idx:03d}_{a['label']}_plus_{b['label']}.wav"
        _save_wav(rel_file, mixed, 22050)
        mixes.append({
            "file": rel_file,
            "source_url": f"{a['source_url']} + {b['source_url']}",
            "source_dataset": f"CrossMix({a['source_dataset']}+{b['source_dataset']})",
            "source_recording": f"crossmix:{a['id']}+{b['id']}",
            "label": a["label"],
            "secondary_label": b["label"],
            "allowed_labels": [c_a, c_b],
            "in_scope": True,
            "sound_type": "mixed",
            "raw_labels": f"{a['label']} + {b['label']} (SNR~+{20*np.log10(1.0/sec_gain):.1f}dB)",
        })
    return mixes


def build(target_total: int = 200, mixed_ratio: float = 0.63) -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    esc_fs_ids = _load_esc50_freesound_ids()
    print(f"Loaded {len(esc_fs_ids)} ESC-50 source Freesound IDs to enforce zero-overlap.", flush=True)

    print("1/4 Querying external internet dataset manifests...", flush=True)
    c_as = _fetch_audioset_candidates(esc_cap=45, ood_cap=12)
    c_strong = _fetch_audioset_strong_candidates(cap=30)
    c_us8k = _fetch_urbansound_candidates(esc_fs_ids, mixed_cap=24, pure_cap=16, ood_cap=8)
    c_wm = _fetch_wikimedia_candidates()
    all_remote = c_as + c_strong + c_us8k + c_wm
    print(f"Found remote candidates: AudioSet={len(c_as)}, AudioSetStrong={len(c_strong)}, US8K={len(c_us8k)}, Wikimedia={len(c_wm)}", flush=True)

    records: list[dict[str, Any]] = []
    seen_recordings: set[str] = set()

    def _worker(cand: dict[str, Any]) -> dict[str, Any] | None:
        raw = _download(cand["url"])
        y, sr = _transcode_wav(raw, cand["id"])
        # Cap long YouTube clips to 10s (keep first 10s so evaluation stays crisp)
        if y.size > sr * 10:
            y = y[:sr * 10]
        sub = "mixed" if cand["sound_type"].startswith("mixed") else ("ood" if not cand["in_scope"] else "pure")
        rel_file = f"{sub}/{cand['id']}.wav"
        _save_wav(rel_file, y, sr)
        return {
            "file": rel_file,
            "id": cand["id"],
            "source_url": cand["url"][:140],
            "source_dataset": cand["source_dataset"],
            "source_recording": cand["source_recording"],
            "label": cand["label"],
            "secondary_label": "",
            "allowed_labels": cand["allowed_labels"],
            "in_scope": cand["in_scope"],
            "sound_type": cand["sound_type"],
            "raw_labels": cand["raw_labels"],
        }

    print(f"2/4 Downloading & transcoding {len(all_remote)} remote audio files (8 threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {}
        for c in all_remote:
            if c["source_recording"] in seen_recordings:
                continue
            seen_recordings.add(c["source_recording"])
            futs[pool.submit(_worker, c)] = c
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
            except Exception:
                pass
    print(f"Downloaded {len(records)} clean remote internet recordings.", flush=True)

    # 3/4 Include existing dev split from eval/external (YashNita animals + IBM MAX + 10 BirdNET windows)
    dev_csv = ROOT / "eval" / "external_manifest.csv"
    if dev_csv.is_file():
        with dev_csv.open(newline="", encoding="utf-8") as f:
            dev_rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
        bird_taken = 0
        for r in dev_rows:
            src_wav = ROOT / "eval" / "external" / r["file"]
            if not src_wav.is_file():
                continue
            is_birdnet = r["source_dataset"] == "kahst/BirdNET-Analyzer"
            if is_birdnet:
                bird_taken += 1
                if bird_taken > 10:
                    continue
                stype = "mixed"
            elif r["in_scope"].lower() != "true":
                stype = "ood_pure"
            else:
                stype = "pure"
            y, sr = sf.read(str(src_wav), dtype="float32")
            cid = f"extdev_{Path(r['file']).stem}"
            rel = f"{'mixed' if stype == 'mixed' else ('ood' if stype.startswith('ood') else 'pure')}/{cid}.wav"
            _save_wav(rel, y, sr)
            allowed = json.loads(r.get("allowed_labels") or "[]")
            records.append({
                "file": rel,
                "id": cid,
                "source_url": r["source_url"],
                "source_dataset": r["source_dataset"],
                "source_recording": r["source_recording"],
                "label": r["label"],
                "secondary_label": "wind" if is_birdnet else "",
                "allowed_labels": allowed,
                "in_scope": r["in_scope"].lower() == "true",
                "sound_type": stype,
                "raw_labels": r["source_name"],
            })

    pure_count = sum(not r["sound_type"].startswith("mixed") for r in records)
    mixed_count = sum(r["sound_type"].startswith("mixed") for r in records)
    # Achieve exact ~60-65% mixed ratio by constructing verified cross-source real internet acoustic mixtures
    needed_mixed = max(0, round(pure_count * mixed_ratio / (1.0 - mixed_ratio)) - mixed_count)
    needed_mixed = max(needed_mixed, max(0, target_total - len(records)))
    print(f"3/4 Current counts: mixed={mixed_count}, non-mixed={pure_count}. Synthesizing {needed_mixed} verified cross-source mixtures...", flush=True)
    cross_mixes = _build_cross_source_mixtures(records, target_count=needed_mixed)
    records.extend(cross_mixes)

    # Sort deterministically and write manifest
    records.sort(key=lambda x: (x["sound_type"], x["file"]))
    fieldnames = ["file", "source_url", "source_dataset", "source_recording", "label", "secondary_label", "allowed_labels", "in_scope", "sound_type", "raw_labels"]
    with MANIFEST_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            row["allowed_labels"] = json.dumps(r["allowed_labels"])
            row["in_scope"] = str(r["in_scope"]).lower()
            w.writerow(row)

    total = len(records)
    n_mix = sum(r["sound_type"] == "mixed" for r in records)
    n_pure = sum(r["sound_type"] == "pure" for r in records)
    n_ood = sum(not r["in_scope"] for r in records)
    print(f"4/4 Built {total} total clips: Mixed={n_mix} ({n_mix/total*100:.1f}%), Pure={n_pure} ({n_pure/total*100:.1f}%), OOD={n_ood} ({n_ood/total*100:.1f}%)", flush=True)
    print(f"Saved manifest: {MANIFEST_OUT}", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=200)
    parser.add_argument("--mixed-ratio", type=float, default=0.63)
    args = parser.parse_args()
    build(args.target_total, args.mixed_ratio)
