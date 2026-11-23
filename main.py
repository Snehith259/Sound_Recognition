"""
═══════════════════════════════════════════════════════════════
  SoundSense — FastAPI Backend Server
  ESC-50 Environmental Sound Classification API
  Routes: /api/predict, /api/train, /api/save-model, /api/stats, /api/health
═══════════════════════════════════════════════════════════════
"""

import os
import re
import time
import uuid
import json
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Header,
    HTTPException,
    BackgroundTasks,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# ── Base Directory Paths ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "Front-End"
PREPROCESSING_DIR = BASE_DIR / "Preproccesing"
ESC50_DIR = BASE_DIR / "ESC-50"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Lifespan: Pre-load the pretrained acoustic engine ────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the ESC-50 model in the background so the first upload is fast."""
    def _warm():
        try:
            from Preproccesing.acoustic_inference import warm_up_engine
            warm_up_engine()
        except Exception:
            pass

    threading.Thread(target=_warm, name="acoustic-warmup", daemon=True).start()
    yield

# ── Application Initialization ───────────────────────────────────
app = FastAPI(
    title="SoundSense Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Static File Mounts (Audio & Preprocessing assets) ─────────────
if PREPROCESSING_DIR.exists():
    app.mount("/Preproccesing", StaticFiles(directory=str(PREPROCESSING_DIR)), name="preprocessing")
if ESC50_DIR.exists():
    app.mount("/ESC-50", StaticFiles(directory=str(ESC50_DIR)), name="esc50")
    if (ESC50_DIR / "wav_files").exists():
        app.mount("/audio/esc50", StaticFiles(directory=str(ESC50_DIR / "wav_files")), name="audio_esc50")

# ── Categories Metadata ──────────────────────────────────────────
ESC50_CATEGORIES = [
    {"id": 0, "name": "dog", "group": "Animals"}, {"id": 1, "name": "rooster", "group": "Animals"},
    {"id": 2, "name": "pig", "group": "Animals"}, {"id": 3, "name": "cow", "group": "Animals"},
    {"id": 4, "name": "frog", "group": "Animals"}, {"id": 5, "name": "cat", "group": "Animals"},
    {"id": 6, "name": "hen", "group": "Animals"}, {"id": 7, "name": "insects", "group": "Animals"},
    {"id": 8, "name": "sheep", "group": "Animals"}, {"id": 9, "name": "crow", "group": "Animals"},
    {"id": 10, "name": "rain", "group": "Natural"}, {"id": 11, "name": "sea_waves", "group": "Natural"},
    {"id": 12, "name": "crackling_fire", "group": "Natural"}, {"id": 13, "name": "crickets", "group": "Natural"},
    {"id": 14, "name": "chirping_birds", "group": "Natural"}, {"id": 15, "name": "water_drops", "group": "Natural"},
    {"id": 16, "name": "wind", "group": "Natural"}, {"id": 17, "name": "pouring_water", "group": "Natural"},
    {"id": 18, "name": "toilet_flush", "group": "Natural"}, {"id": 19, "name": "thunderstorm", "group": "Natural"},
    {"id": 20, "name": "crying_baby", "group": "Human"}, {"id": 21, "name": "sneezing", "group": "Human"},
    {"id": 22, "name": "clapping", "group": "Human"}, {"id": 23, "name": "breathing", "group": "Human"},
    {"id": 24, "name": "coughing", "group": "Human"}, {"id": 25, "name": "footsteps", "group": "Human"},
    {"id": 26, "name": "laughing", "group": "Human"}, {"id": 27, "name": "brushing_teeth", "group": "Human"},
    {"id": 28, "name": "snoring", "group": "Human"}, {"id": 29, "name": "drinking_sipping", "group": "Human"},
    {"id": 30, "name": "door_wood_knock", "group": "Interior"}, {"id": 31, "name": "mouse_click", "group": "Interior"},
    {"id": 32, "name": "keyboard_typing", "group": "Interior"}, {"id": 33, "name": "door_wood_creaks", "group": "Interior"},
    {"id": 34, "name": "can_opening", "group": "Interior"}, {"id": 35, "name": "washing_machine", "group": "Interior"},
    {"id": 36, "name": "vacuum_cleaner", "group": "Interior"}, {"id": 37, "name": "clock_alarm", "group": "Interior"},
    {"id": 38, "name": "clock_tick", "group": "Interior"}, {"id": 39, "name": "glass_breaking", "group": "Interior"},
    {"id": 40, "name": "helicopter", "group": "Urban"}, {"id": 41, "name": "chainsaw", "group": "Urban"},
    {"id": 42, "name": "siren", "group": "Urban"}, {"id": 43, "name": "car_horn", "group": "Urban"},
    {"id": 44, "name": "engine", "group": "Urban"}, {"id": 45, "name": "train", "group": "Urban"},
    {"id": 46, "name": "church_bells", "group": "Urban"}, {"id": 47, "name": "airplane", "group": "Urban"},
    {"id": 48, "name": "fireworks", "group": "Urban"}, {"id": 49, "name": "hand_saw", "group": "Urban"},
]

# ── In-Memory Server State ───────────────────────────────────────
server_state = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "total_predictions": 0,
    "total_training_runs": 0,
    "saved_checkpoints": [],
    "training_jobs": [],
    "model_versions": [],
}

active_sessions: Dict[str, Dict[str, Any]] = {}
user_history: List[Dict[str, Any]] = []

# ── Pydantic Request Models ──────────────────────────────────────
class TrainRequest(BaseModel):
    modelType: str = "resnet18"
    epochs: int = 50
    batchSize: int = 32
    learningRate: float = 0.001
    folds: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    augmentation: bool = True
    description: str = ""

class SaveModelRequest(BaseModel):
    name: str = "Untitled Checkpoint"
    description: str = ""
    modelType: str = "resnet18"
    accuracy: Optional[float] = None
    fromJobId: Optional[str] = None

class CloudSyncRequest(BaseModel):
    repo: str = "huggingface.co/soundsense/esc50-resnet18"

class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    role: str = "user"
    name: Optional[str] = None

# ── Helper Functions ─────────────────────────────────────────────
def get_uptime_seconds() -> int:
    started = datetime.fromisoformat(server_state["started_at"])
    return int((datetime.now(timezone.utc) - started).total_seconds())


def load_evaluation_report() -> Dict[str, Any]:
    """Read the sole supported source for published model metrics."""
    report_path = BASE_DIR / "checkpoints" / "eval_report.json"
    if not report_path.is_file():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def current_checkpoint_inventory() -> Dict[str, Dict[str, Any]]:
    try:
        from Preproccesing.acoustic_inference import model_status
        return model_status().get("checkpoints", {})
    except Exception:
        return {}

def int32_hash(s: str) -> int:
    """Simulates the 32-bit signed integer hash behavior of JS string hashing."""
    h = 0
    for char in s:
        h = ((h << 5) - h) + ord(char)
        h &= 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h

# ══════════════════════════════════════════════════════════════════
# ROUTE 0: GET /api/sample/{category_id} — Fetch Verified ESC-50 Audio
# ══════════════════════════════════════════════════════════════════
@app.get("/api/sample/{category_id}")
async def get_sample_for_category(category_id: int):
    cat_id = max(0, min(49, category_id))
    matches = []
    
    # 1. Primary: Check ESC-50 wav_files if present
    wav_dir = ESC50_DIR / "wav_files"
    if wav_dir.is_dir():
        matches = list(wav_dir.glob(f"*-{cat_id}.wav"))

    # 2. Fallback: Front-End samples directory (always bundled)
    if not matches:
        samples_dir = FRONTEND_DIR / "samples"
        if samples_dir.is_dir():
            matches = list(samples_dir.glob(f"*-{cat_id}.wav")) or list(samples_dir.glob(f"{cat_id}.wav"))

    # 3. Last resort fallback
    if not matches and wav_dir.is_dir():
        matches = list(wav_dir.glob("*.wav"))

    if matches:
        return FileResponse(str(matches[0]), media_type="audio/wav", filename=matches[0].name)
    raise HTTPException(status_code=404, detail="Sample audio not found")


# ══════════════════════════════════════════════════════════════════
# ROUTE 1: POST /api/predict — Audio Inference Endpoint
# ══════════════════════════════════════════════════════════════════
@app.post("/api/predict")
async def predict(
    audio: UploadFile = File(...),
    model: str = Form("ast"),
    threshold: Optional[float] = Form(None),
):
    disallowed_exts = {".ogg", ".flac", ".m4a", ".aac"}
    allowed_exts = {".wav", ".mp3", ".mpg", ".mpeg", ".webm"}
    file_ext = Path(audio.filename or "").suffix.lower()
    content_type = (audio.content_type or "").lower()

    if file_ext in disallowed_exts or any(d in content_type for d in ["ogg", "flac", "m4a", "aac"]):
        raise HTTPException(
            status_code=400,
            detail="OGG, FLAC, and M4A files are not accepted. Please upload WAV, MP3, or MPEG/MPG."
        )

    is_allowed_mime = (
        content_type in {"audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "video/mpeg", "audio/webm", "application/octet-stream"}
        or (content_type.startswith("audio/") and not any(d in content_type for d in ["ogg", "flac", "m4a", "aac"]))
    )

    if file_ext not in allowed_exts and not is_allowed_mime:
        raise HTTPException(status_code=400, detail="Only WAV, MP3, and MPEG/MPG audio files are accepted")

    stamp = int(time.time() * 1000)
    file_name = audio.filename or f"upload_{stamp}{file_ext or '.wav'}"

    try:
        content = await audio.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

        # Real Acoustic Classification in-memory (Zero disk writes, prevents Live Server reload)
        from Preproccesing.acoustic_inference import run_acoustic_inference
        confidence_threshold = None if threshold is None else float(threshold)
        if confidence_threshold is not None and confidence_threshold > 1.0:
            confidence_threshold /= 100.0
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1, or 0 and 100.")
        result = await run_in_threadpool(run_acoustic_inference, content, model, confidence_threshold)

        server_state["total_predictions"] += 1
        history_id = f"hist_{stamp}_{uuid.uuid4().hex[:6]}"

        # Record to persistent in-memory user_history
        history_entry = {
            "id": history_id,
            "fileName": file_name,
            "fileSize": len(content),
            "modelKey": model,
            "modelName": result.get("category", {}).get("name", "Unknown"),
            "engine": result.get("engine"),
            "topPrediction": {
                "category": result["category"]["name"],
                "emoji": result["category"].get("emoji", "🔊"),
                "group": result["category"].get("group", "Other"),
                "confidence": result["confidence"],
                "confidencePercent": (
                    f"{round(result['confidence'] * 100, 1)}%"
                    if result.get("confidenceType") != "not_computed_rule_gate" and result["confidence"] is not None
                    else None
                ),
                "isOutOfScope": result.get("isOutOfScope", False),
            },
            "predictions": result["predictions"],
            "inferenceSource": "api",
            "latencyMs": result["inferenceTime"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        user_history.insert(0, history_entry)
        if len(user_history) > 200:
            user_history.pop()

        return {
            "success": True,
            "model": model,
            "engine": result.get("engine"),
            "fileName": file_name,
            "isOutOfScope": result.get("isOutOfScope", False),
            "confidence": result["confidence"],
            "confidenceType": result.get("confidenceType"),
            "classProbabilities": result.get("classProbabilities"),
            "oodScore": result.get("oodScore"),
            "category": result["category"],
            "predictions": result["predictions"],
            "explanation": result.get("explanation"),
            "metrics": result.get("metrics"),
            "inferenceTime": result["inferenceTime"],
            "windowCount": result.get("windowCount"),
            "belowConfidenceThreshold": result.get("belowConfidenceThreshold", False),
            "confidenceThreshold": result.get("confidenceThreshold"),
            "historyId": history_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except FileNotFoundError as err:
        return JSONResponse(status_code=503, content={"error": "Model unavailable", "details": str(err)})
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": "Invalid inference request", "details": str(err)})
    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"error": "Inference failed", "details": str(err)}
        )

# ══════════════════════════════════════════════════════════════════
# ROUTE 1B: POST /api/predict/batch — Batch Audio Inference Endpoint
# ══════════════════════════════════════════════════════════════════
@app.post("/api/predict/batch")
async def predict_batch(
    audio: List[UploadFile] = File(...),
    model: str = Form("ast"),
    threshold: Optional[float] = Form(None),
):
    if not audio:
        raise HTTPException(status_code=400, detail="No audio files provided")

    if len(audio) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files allowed per batch")

    from Preproccesing.acoustic_inference import run_acoustic_inference

    disallowed_exts = {".ogg", ".flac", ".m4a", ".aac"}
    allowed_exts = {".wav", ".mp3", ".mpg", ".mpeg", ".webm"}

    results = []
    for file in audio:
        file_ext = Path(file.filename or "").suffix.lower()
        content_type = (file.content_type or "").lower()

        if file_ext in disallowed_exts or any(d in content_type for d in ["ogg", "flac", "m4a", "aac"]):
            results.append({
                "fileName": file.filename or "unknown",
                "success": False,
                "error": "Disallowed format (OGG/FLAC/M4A/AAC). Please use WAV, MP3, or MPEG/MPG."
            })
            continue

        try:
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:
                results.append({
                    "fileName": file.filename or "unknown",
                    "success": False,
                    "error": "File exceeds 50 MB limit"
                })
                continue

            confidence_threshold = None if threshold is None else float(threshold)
            if confidence_threshold is not None and confidence_threshold > 1.0:
                confidence_threshold /= 100.0
            res = await run_in_threadpool(run_acoustic_inference, content, model, confidence_threshold)
            server_state["total_predictions"] += 1
            stamp = int(time.time() * 1000)
            history_id = f"hist_{stamp}_{uuid.uuid4().hex[:6]}"

            history_entry = {
                "id": history_id,
                "fileName": file.filename or "audio_clip.wav",
                "fileSize": len(content),
                "modelKey": model,
                "modelName": res.get("category", {}).get("name", "Unknown"),
                "engine": res.get("engine"),
                "topPrediction": {
                    "category": res["category"]["name"],
                    "emoji": res["category"].get("emoji", "🔊"),
                    "group": res["category"].get("group", "Other"),
                    "confidence": res["confidence"],
                    "confidencePercent": (
                        f"{round(res['confidence'] * 100, 1)}%"
                        if res.get("confidenceType") != "not_computed_rule_gate" and res["confidence"] is not None
                        else None
                    ),
                    "isOutOfScope": res.get("isOutOfScope", False),
                },
                "predictions": res["predictions"],
                "inferenceSource": "api_batch",
                "latencyMs": res["inferenceTime"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            user_history.insert(0, history_entry)
            if len(user_history) > 200:
                user_history.pop()

            results.append({
                "fileName": file.filename,
                "success": True,
                "isOutOfScope": res.get("isOutOfScope", False),
                "category": res["category"],
                "confidence": res["confidence"],
                "predictions": res["predictions"],
                "inferenceTime": res["inferenceTime"],
                "engine": res.get("engine"),
            })
        except Exception as e:
            results.append({
                "fileName": file.filename or "unknown",
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "model": model,
        "count": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ══════════════════════════════════════════════════════════════════
# ROUTE 2: POST /api/train — Trigger Model Training Run
# ══════════════════════════════════════════════════════════════════
@app.post("/api/train")
async def trigger_training(payload: TrainRequest, background_tasks: BackgroundTasks):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Training runs offline with python -m eval.train_cv. This endpoint does not simulate training or metrics.",
    )

@app.get("/api/train/{job_id}")
async def get_training_job(job_id: str):
    job = next((j for j in server_state["training_jobs"] if j["jobId"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"success": True, "job": job}

@app.get("/api/train")
async def list_training_jobs():
    jobs = [
        {
            "jobId": j["jobId"],
            "modelType": j["modelType"],
            "status": j["status"],
            "progress": j["progress"],
            "currentEpoch": j["currentEpoch"],
            "epochs": j["epochs"],
            "startedAt": j["startedAt"],
            "completedAt": j["completedAt"],
        }
        for j in server_state["training_jobs"]
    ]
    return {"success": True, "jobs": jobs}

# ══════════════════════════════════════════════════════════════════
# ROUTE 3: POST /api/save-model — Save Model Checkpoint
# ══════════════════════════════════════════════════════════════════
@app.post("/api/save-model")
async def save_model(payload: SaveModelRequest):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Checkpoints are created by the offline training pipeline. This endpoint does not invent or save metrics.",
    )

@app.get("/api/save-model")
async def list_checkpoints():
    inventory = current_checkpoint_inventory()
    report_models = load_evaluation_report().get("models", {})
    checkpoints = []
    for model_name, info in inventory.items():
        if not info.get("available"):
            continue
        metrics = report_models.get(model_name, {}).get("metrics", {})
        checkpoints.append({
            "checkpointId": info["filename"],
            "name": model_name,
            "modelType": model_name,
            "filename": info["filename"],
            "sha256": info.get("sha256"),
            "fileSizeBytes": info.get("sizeBytes"),
            "accuracy": metrics.get("top1") * 100 if metrics.get("top1") is not None else None,
            "top5": metrics.get("top5") * 100 if metrics.get("top5") is not None else None,
            "status": "available",
        })
    return {
        "success": True,
        "checkpoints": checkpoints,
        "modelVersions": [],
    }

@app.post("/api/model/cloud-sync")
async def cloud_sync(payload: Optional[CloudSyncRequest] = None):
    repo = payload.repo if payload else "huggingface.co/soundsense/esc50-resnet18"
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"No verified model repository is configured (requested source: {repo}).",
    )

# ══════════════════════════════════════════════════════════════════
# ROUTE 4: GET /api/stats — System & Dataset Analytics
# ══════════════════════════════════════════════════════════════════
@app.get("/api/stats")
async def get_stats():
    preprocessed_dir = PREPROCESSING_DIR / "processed"
    processed_audio_count = 0
    log_mel_count = 0
    mfcc_count = 0
    rp_count = 0

    try:
        audio_dir = preprocessed_dir / "audio_22k"
        if audio_dir.exists():
            processed_audio_count = len([f for f in audio_dir.iterdir() if f.name.endswith(".wav")])

        mel_dir = preprocessed_dir / "features" / "log_mel"
        if mel_dir.exists():
            log_mel_count = len([f for f in mel_dir.iterdir() if f.name.endswith(".npy")])

        mfcc_dir = preprocessed_dir / "features" / "mfcc"
        if mfcc_dir.exists():
            mfcc_count = len([f for f in mfcc_dir.iterdir() if f.name.endswith(".npy")])

        rp_dir = preprocessed_dir / "features" / "recurrence_plot"
        if rp_dir.exists():
            rp_count = len([f for f in rp_dir.iterdir() if f.name.endswith(".npy")])
    except Exception:
        pass

    preprocessing_report = None
    report_path = PREPROCESSING_DIR / "reports" / "preprocessing_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                preprocessing_report = json.load(f)
        except Exception:
            pass

    class_distribution = [
        {
            "id": cat["id"],
            "name": cat["name"],
            "group": cat["group"],
            "sampleCount": round(processed_audio_count / 50),
        }
        for cat in ESC50_CATEGORIES
    ]

    groups = ["Animals", "Natural", "Human", "Interior", "Urban"]
    group_distribution = [
        {
            "group": g,
            "classCount": 10,
            "totalSamples": round(processed_audio_count / 5),
        }
        for g in groups
    ]

    eval_report = load_evaluation_report()
    colors = {"ast": "#6366f1", "cnngru": "#34d399", "resnet18": "#38bdf8", "vgg16": "#fbbf24", "sresnet": "#a855f7", "ensemble": "#818cf8"}
    display_names = {"ast": "AST-ESC50", "cnngru": "CNN-GRU", "resnet18": "ResNet-18", "vgg16": "VGG-16", "sresnet": "SResNet", "ensemble": "Ensemble Fusion"}
    model_performance = []
    for model_name, model_result in eval_report.get("models", {}).items():
        metrics = model_result.get("metrics", {})
        if not metrics:
            continue
        model_performance.append({
            "model": display_names.get(model_name, model_name),
            "modelKey": model_name,
            "accuracy": metrics.get("top1") * 100 if metrics.get("top1") is not None else None,
            "top5Accuracy": metrics.get("top5") * 100 if metrics.get("top5") is not None else None,
            "r2Score": metrics.get("r2Probability"),
            "ece": metrics.get("ece"),
            "externalAccuracy": model_result.get("external", {}).get("top1") * 100 if model_result.get("external", {}).get("top1") is not None else None,
            "latency": metrics.get("latencyMs"),
            "color": colors.get(model_name, "#94a3b8"),
            "evaluatedOn": model_result.get("evaluation", eval_report.get("evaluation")),
        })

    uptime = get_uptime_seconds()
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    rss_mb = round(mem_info.rss / 1024 / 1024)

    return {
        "success": True,
        "system": {
            "uptime": uptime,
            "uptimeFormatted": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s",
            "startedAt": server_state["started_at"],
            "nodeVersion": "FastAPI (Python)",
            "platform": os.name,
            "memoryUsage": {
                "rss": rss_mb,
                "heapUsed": rss_mb,
                "heapTotal": rss_mb,
            },
            "totalPredictions": server_state["total_predictions"],
            "totalTrainingRuns": int(eval_report.get("trainingRuns", 0)),
            "savedCheckpoints": sum(1 for item in current_checkpoint_inventory().values() if item.get("available")),
        },
        "dataset": {
            "name": "ESC-50",
            "totalClasses": 50,
            "totalGroups": 5,
            "processedAudioCount": processed_audio_count,
            "logMelCount": log_mel_count,
            "mfccCount": mfcc_count,
            "rpCount": rp_count,
            "sampleRate": 22050,
            "duration": 5.0,
            "featureShapes": {
                "logMel": [128, 216],
                "mfcc": [40, 216],
                "recurrencePlot": [128, 128],
            },
            "numFolds": 5,
            "classDistribution": class_distribution,
            "groupDistribution": group_distribution,
        },
        "modelPerformance": model_performance,
        "preprocessingReport": preprocessing_report,
        "modelVersions": [],
        "evaluationReport": eval_report,
        "modelAvailability": {
            "availableModels": [name for name, item in current_checkpoint_inventory().items() if item.get("available")],
            "missingModels": [name for name, item in current_checkpoint_inventory().items() if not item.get("available")],
        },
    }

# ══════════════════════════════════════════════════════════════════
# ROUTE 6: AUTHENTICATION ENDPOINTS (USER & DEVELOPER)
# ══════════════════════════════════════════════════════════════════
@app.post("/api/auth/login")
async def login(payload: LoginRequest):
    clean_user = payload.username.strip().lower()
    clean_pass = payload.password.strip()
    role = (payload.role or "user").strip().lower()

    if not clean_user or not clean_pass:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Username and password are required."}
        )

    base_name = payload.name or (clean_user.split("@")[0] if "@" in clean_user else clean_user)
    formatted_name = re.sub(r"[._-]", " ", base_name).title()

    if role == "developer":
        token = f"dev_token_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        user = {
            "name": f"{formatted_name} (AI Engineer)" if formatted_name else "Lead AI Engineer",
            "email": clean_user if "@" in clean_user else f"{clean_user}@soundsense.ai",
            "role": "developer",
            "permissions": ["training_controls", "checkpoint_mgmt", "system_analytics", "cloud_sync", "history_mgmt"],
            "signedInAt": datetime.now(timezone.utc).isoformat(),
        }
        active_sessions[token] = user
        return {
            "success": True,
            "token": token,
            "user": user,
            "role": "developer",
            "message": "Developer mode unlocked successfully.",
        }
    else:
        # Standard User account
        token = f"user_token_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        user = {
            "name": formatted_name or "SoundSense User",
            "email": clean_user if "@" in clean_user else f"{clean_user}@soundsense.ai",
            "role": "user",
            "permissions": ["history_mgmt", "audio_inference", "preset_access"],
            "signedInAt": datetime.now(timezone.utc).isoformat(),
        }
        active_sessions[token] = user
        return {
            "success": True,
            "token": token,
            "user": user,
            "role": "user",
            "message": "User signed in successfully.",
        }

@app.get("/api/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    token = ""
    if authorization:
        token = re.sub(r"^Bearer\s+", "", authorization, flags=re.IGNORECASE)

    if token and token in active_sessions:
        return {"success": True, "authenticated": True, "user": active_sessions[token]}

    return {"success": True, "authenticated": False, "role": "guest"}

@app.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    token = ""
    if authorization:
        token = re.sub(r"^Bearer\s+", "", authorization, flags=re.IGNORECASE)

    if token and token in active_sessions:
        del active_sessions[token]

    return {"success": True, "message": "Session terminated."}

# ══════════════════════════════════════════════════════════════════
# ROUTE 7: CLASSIFICATION HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════
@app.get("/api/history")
async def get_history():
    return {
        "success": True,
        "count": len(user_history),
        "history": user_history,
    }

@app.post("/api/history")
async def add_history(item: Dict[str, Any]):
    entry = item.get("item", item)
    if "id" not in entry:
        entry["id"] = f"hist_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    user_history.insert(0, entry)
    if len(user_history) > 200:
        user_history.pop()
    return {"success": True, "item": entry}

@app.delete("/api/history")
async def clear_history(id: Optional[str] = None):
    global user_history
    if id:
        user_history = [h for h in user_history if h.get("id") != id]
        return {"success": True, "deleted": id, "remaining": len(user_history)}
    user_history.clear()
    return {"success": True, "count": 0, "message": "History cleared"}

# ══════════════════════════════════════════════════════════════════
# ROUTE 5: GET /api/health — Health Check
# ══════════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health():
    uptime = get_uptime_seconds()
    try:
        from Preproccesing.acoustic_inference import get_engine_status
        engine_status = get_engine_status()
    except Exception as err:
        engine_status = {"state": "unavailable", "error": str(err)}

    checkpoint_inventory = current_checkpoint_inventory()
    missing_models = [
        name for name, item in checkpoint_inventory.items()
        if not item.get("available")
    ]

    return {
        "status": "healthy" if not missing_models else "degraded",
        "service": "SoundSense Backend",
        "version": "1.0.0",
        "uptime": uptime,
        "engine": engine_status,
        "models": checkpoint_inventory,
        "missingModels": missing_models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": [
            {"method": "POST", "path": "/api/auth/login", "description": "Developer login"},
            {"method": "GET", "path": "/api/auth/me", "description": "Verify developer session"},
            {"method": "POST", "path": "/api/auth/logout", "description": "Developer logout"},
            {"method": "POST", "path": "/api/predict", "description": "Audio classification inference"},
            {"method": "POST", "path": "/api/predict/batch", "description": "Batch audio classification inference"},
            {"method": "POST", "path": "/api/train", "description": "Trigger model training"},
            {"method": "GET", "path": "/api/train", "description": "List training jobs"},
            {"method": "GET", "path": "/api/train/{jobId}", "description": "Check training job status"},
            {"method": "POST", "path": "/api/save-model", "description": "Save model checkpoint"},
            {"method": "GET", "path": "/api/save-model", "description": "List saved checkpoints"},
            {"method": "GET", "path": "/api/stats", "description": "System & dataset analytics"},
            {"method": "GET", "path": "/api/health", "description": "Health check"},
        ],
    }

# ── Frontend Static Files & SPA Fallback ─────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ── Server Start ─────────────────────────────────────────────────
def _port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True when something is already listening on host:port."""
    import socket
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((probe_host, port)) == 0


def _is_soundsense_instance(host: str, port: int) -> bool:
    """Return True when the service on host:port is an existing SoundSense backend."""
    import json as _json
    import urllib.request
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with urllib.request.urlopen(f"http://{probe_host}:{port}/api/health", timeout=1.5) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
        return payload.get("service") == "SoundSense Backend"
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 3001))
    reload_mode = os.environ.get("RELOAD", "false").lower() == "true"
    display_host = "0.0.0.0" if host in ("0.0.0.0", "::") else host

    print("=" * 60)
    print("  SoundSense Backend Server")
    print(f"  Running at:  http://{display_host}:{port}")
    print(f"  API Base:    http://{display_host}:{port}/api")
    print(f"  Health:      http://{display_host}:{port}/api/health")
    print(f"  Reload Mode: {reload_mode}")
    print("=" * 60)

    # Fail with a helpful message instead of an opaque WinError 10048 traceback.
    if _port_is_open(host, port):
        if _is_soundsense_instance(host, port):
            print(f"\n[!] SoundSense is already running on http://localhost:{port}")
            print("    Stop that instance first, or start on a free port, e.g.:")
            print(f"    PORT={port + 1} python main.py")
            raise SystemExit(0)
        print(f"\n[!] Port {port} is already in use by another application.")
        print("    Start SoundSense on a free port, e.g.:")
        print(f"    PORT={port + 1} python main.py")
        raise SystemExit(1)

    # Pass the app object directly so the module is not imported a second time
    # (reload mode still needs the import string).
    uvicorn.run(app if not reload_mode else "main:app", host=host, port=port, reload=reload_mode)
