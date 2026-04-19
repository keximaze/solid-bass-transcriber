from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
import uuid
from pathlib import Path
from typing import Any, Callable

# -----------------------------------------------------------------------------
# Optional FastAPI imports
# -----------------------------------------------------------------------------
# This file stays importable in limited environments where FastAPI is missing.
# In a real backend environment, the actual FastAPI classes are used.
FASTAPI_AVAILABLE = True
FASTAPI_IMPORT_ERROR: str | None = None

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except Exception as exc:  # pragma: no cover - limited environments only
    FASTAPI_AVAILABLE = False
    FASTAPI_IMPORT_ERROR = str(exc)

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:
        def __init__(self, method: str = "GET", url_path: str = "/"):
            self.method = method
            self.url = type("URL", (), {"path": url_path})()

    class UploadFile:
        def __init__(self, filename: str | None = None):
            self.filename = filename

        async def read(self, size: int = -1) -> bytes:
            return b""

    def File(default: Any = None) -> Any:
        return default

    def Form(default: Any = None) -> Any:
        return default

    class CORSMiddleware:
        pass

    class FileResponse:
        def __init__(self, path: str | Path, filename: str | None = None):
            self.path = str(path)
            self.filename = filename

    class JSONResponse(dict):
        def __init__(self, content: dict[str, Any], status_code: int = 200):
            super().__init__(content)
            self.content = content
            self.status_code = status_code

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any):
            self.routes: list[tuple[str, str, Callable[..., Any]]] = []
            self.exception_handlers: dict[type[Exception], Callable[..., Any]] = {}

        def add_middleware(self, *args: Any, **kwargs: Any) -> None:
            return None

        def exception_handler(self, exc_type: type[Exception]):
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.exception_handlers[exc_type] = func
                return func
            return decorator

        def get(self, path: str, **kwargs: Any):
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(("GET", path, func))
                return func
            return decorator

        def post(self, path: str, **kwargs: Any):
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(("POST", path, func))
                return func
            return decorator


try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - limited environments only
    class BaseModel:
        def __init__(self, **kwargs: Any):
            annotations = getattr(self.__class__, "__annotations__", {})
            for key in annotations:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif hasattr(self.__class__, key):
                    setattr(self, key, getattr(self.__class__, key))
                else:
                    setattr(self, key, None)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
def resolve_base_dir() -> Path:
    """Resolve a stable project directory in both script and interactive runs."""
    file_value = globals().get("__file__")
    return Path(file_value).resolve().parent if file_value else Path.cwd()


APP_NAME = "Solid Bass Transcriber API"
BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
EXPORT_DIR = DATA_DIR / "exports"

for directory in [DATA_DIR, UPLOAD_DIR, RUNS_DIR, EXPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME, version="1.0.0")
logger = logging.getLogger("bass_transcriber")
logging.basicConfig(level=logging.INFO)

if FASTAPI_AVAILABLE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class PipelineSettings(BaseModel):
    detected_key: str = "E minor"
    strict_scale: bool = True
    keep_human_feel: bool = True
    export_tab: bool = True
    quantize_percent: int = 15
    min_note_ms: int = 70
    range_mode: str = "standard4"
    time_signature: str = "4/4"
    tempo_override: int | None = None
    correction_mode: str = "key_aware"  # chromatic | key_aware | chord_aware
    key_root: str = "E"
    key_mode: str = "natural_minor"  # major | natural_minor


class PipelineResult(BaseModel):
    run_id: str
    input_file: str
    tempo_bpm: float
    detected_key: str
    time_signature: str
    note_count: int
    corrections: list[dict[str, Any]]
    exports: dict[str, str]
    diagnostics: dict[str, Any]
    tab_preview: list[dict[str, Any]] | None = None


# -----------------------------------------------------------------------------
# Music helpers
# -----------------------------------------------------------------------------
NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def require_fastapi_runtime() -> None:
    if not FASTAPI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "FastAPI is not available in this environment. "
                f"Original import error: {FASTAPI_IMPORT_ERROR}"
            ),
        )


def make_json_error(status_code: int, detail: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        content={"detail": detail, "error_type": error_type},
        status_code=status_code,
    )


def run_self_tests() -> unittest.result.TestResult:
    """Run tests without unittest.main(), which can raise SystemExit in sandboxes."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackendUtilityTests)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def safe_filename(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", " ")).strip()
    return cleaned or f"audio_{uuid.uuid4().hex}.wav"


def validate_upload_filename(filename: str) -> None:
    allowed_suffixes = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a"}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix or 'no extension'}. Allowed: {', '.join(sorted(allowed_suffixes))}",
        )


def ensure_file_not_empty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")


def parse_scale(key_name: str) -> tuple[int | None, set[int] | None]:
    try:
        parts = key_name.split()
        tonic = NOTE_TO_PC[parts[0]]
        mode = parts[1].lower() if len(parts) > 1 else "major"
        template = MINOR_SCALE if "min" in mode else MAJOR_SCALE
        return tonic, {(tonic + interval) % 12 for interval in template}
    except Exception:
        return None, None


def build_scale_from_settings(settings: PipelineSettings) -> tuple[int | None, set[int] | None, str]:
    """Resolve the active musical scale from explicit mode settings.

    `detected_key` is kept as a backward-compatible fallback so older frontend
    payloads still work.
    """
    correction_mode = (settings.correction_mode or "key_aware").lower()
    if correction_mode == "chromatic":
        return None, None, "Chromatic"

    root = (settings.key_root or "").strip()
    mode = (settings.key_mode or "").strip().lower()

    if root and mode in {"major", "natural_minor"}:
        display_mode = "major" if mode == "major" else "natural minor"
        scale_name = f"{root} {display_mode}"
        tonic = NOTE_TO_PC.get(root)
        template = MAJOR_SCALE if mode == "major" else MINOR_SCALE
        if tonic is not None:
            return tonic, {(tonic + interval) % 12 for interval in template}, scale_name

    tonic, allowed = parse_scale(settings.detected_key)
    return tonic, allowed, settings.detected_key


def range_limits(range_mode: str) -> tuple[int, int]:
    if range_mode == "five":
        return 23, 67
    if range_mode == "drop":
        return 26, 67
    return 28, 67


def nearest_pitch_in_scale(midi_pitch: int, allowed_pcs: set[int]) -> int:
    candidates: list[tuple[int, int]] = []
    for offset in range(-6, 7):
        candidate = midi_pitch + offset
        if candidate % 12 in allowed_pcs:
            candidates.append((abs(offset), candidate))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1] if candidates else midi_pitch


def quantize_seconds(value: float, beat_seconds: float, amount_percent: int) -> float:
    if amount_percent <= 0:
        return value
    grid = beat_seconds / 4.0
    snapped = round(value / grid) * grid
    blend = amount_percent / 100.0
    return value * (1.0 - blend) + snapped * blend


def maybe_run_ffmpeg_convert(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".wav":
        return input_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg is required to convert non-WAV input files.")

    output_path = input_path.with_suffix(".wav")
    cmd = [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "44100", str(output_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg conversion failed: {completed.stderr[:500]}")
    return output_path


def require_runtime_dependency(package_name: str, install_hint: str) -> None:
    raise HTTPException(status_code=500, detail=f"Missing optional dependency: {package_name}. {install_hint}")


def estimate_tempo(audio_path: Path) -> float:
    """Estimate tempo from audio.

    This works best on full mixes or stems with clear transients. Bass-only stems
    can be less reliable, so this now acts as a secondary fallback.
    """
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, tuple)):
            return float(tempo[0])
        return float(tempo)
    except Exception:
        return 120.0


def estimate_tempo_from_midi(midi_path: Path) -> float | None:
    """Estimate tempo from MIDI note onsets.

    Strategy:
    - collect sorted note start times from the first instrument
    - compute positive inter-onset intervals
    - keep a musical range of roughly 40 to 240 BPM
    - choose the BPM candidate that appears most often after rounding

    This is more stable than audio beat tracking for isolated bass stems.
    """
    try:
        pm, _ = load_pretty_midi(midi_path)
    except Exception:
        return None

    if not pm.instruments:
        return None

    starts = sorted(float(note.start) for note in pm.instruments[0].notes)
    if len(starts) < 4:
        return None

    intervals: list[float] = []
    for previous, current in zip(starts, starts[1:]):
        delta = current - previous
        if delta > 0.05:
            intervals.append(delta)

    if not intervals:
        return None

    candidates: dict[int, int] = {}
    for interval in intervals:
        for multiplier in (1.0, 2.0, 0.5):
            beat_seconds = interval * multiplier
            if beat_seconds <= 0:
                continue
            bpm = 60.0 / beat_seconds
            if 40 <= bpm <= 240:
                rounded = int(round(bpm))
                candidates[rounded] = candidates.get(rounded, 0) + 1

    if not candidates:
        return None

    best_bpm = max(candidates.items(), key=lambda item: (item[1], -abs(item[0] - 120)))[0]
    return float(best_bpm)


def resolve_tempo(audio_path: Path, raw_midi_path: Path, settings: PipelineSettings) -> tuple[float, str, dict[str, float | None]]:
    """Resolve the tempo using override > MIDI > audio > fallback."""
    if settings.tempo_override and settings.tempo_override > 0:
        return float(settings.tempo_override), "tempo_override", {
            "tempo_override": float(settings.tempo_override),
            "midi_estimate": None,
            "audio_estimate": None,
        }

    midi_estimate = estimate_tempo_from_midi(raw_midi_path)
    if midi_estimate is not None:
        audio_estimate = estimate_tempo(audio_path)
        return float(midi_estimate), "midi_estimate", {
            "tempo_override": None,
            "midi_estimate": float(midi_estimate),
            "audio_estimate": float(audio_estimate),
        }

    audio_estimate = estimate_tempo(audio_path)
    if audio_estimate and audio_estimate > 0:
        return float(audio_estimate), "audio_estimate", {
            "tempo_override": None,
            "midi_estimate": None,
            "audio_estimate": float(audio_estimate),
        }

    return 120.0, "fallback_120", {
        "tempo_override": None,
        "midi_estimate": None,
        "audio_estimate": None,
    }


def transcribe_with_basic_pitch(audio_path: Path, output_dir: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Run Basic Pitch using the packaged ICASSP 2022 model.

    Newer basic-pitch versions require an explicit `model_or_model_path`
    argument when calling `predict_and_save`.
    """
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict_and_save
    except Exception:
        require_runtime_dependency("basic-pitch", "Install it with: pip install basic-pitch")

    model_output_dir = output_dir / "basic_pitch"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        audio_path_list=[str(audio_path)],
        output_directory=str(model_output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    midi_files = sorted(model_output_dir.glob("*.mid"))
    if not midi_files:
        raise HTTPException(status_code=500, detail="Basic Pitch did not generate a MIDI file.")

    return midi_files[0], [], {"engine": "basic_pitch", "model": str(ICASSP_2022_MODEL_PATH)}


def load_pretty_midi(path: Path):
    try:
        import pretty_midi
    except Exception:
        require_runtime_dependency("pretty_midi", "Install it with: pip install pretty_midi")
    return pretty_midi.PrettyMIDI(str(path)), pretty_midi


def cleanup_bass_midi(
    midi_path: Path,
    output_path: Path,
    settings: PipelineSettings,
    tempo_bpm: float,
) -> tuple[int, list[dict[str, Any]], str]:
    pm, pretty_midi = load_pretty_midi(midi_path)
    low_limit, high_limit = range_limits(settings.range_mode)
    _, allowed_pcs, resolved_scale_name = build_scale_from_settings(settings)

    beat_seconds = 60.0 / max(tempo_bpm, 1.0)
    min_note_seconds = settings.min_note_ms / 1000.0
    corrections: list[dict[str, Any]] = []

    if not pm.instruments:
        raise HTTPException(status_code=500, detail="No MIDI instruments found in transcription output.")

    target_instrument = pm.instruments[0]
    notes = sorted(target_instrument.notes, key=lambda note: (note.start, note.pitch))
    cleaned_notes: list[Any] = []
    previous_end = 0.0

    for idx, note in enumerate(notes):
        original_pitch = int(note.pitch)
        original_start = float(note.start)
        original_end = float(note.end)
        duration = original_end - original_start

        if duration < min_note_seconds:
            corrections.append(
                {
                    "index": idx,
                    "issue": "very_short_note",
                    "action": "removed false trigger",
                    "confidence": 0.90,
                }
            )
            continue

        new_pitch = original_pitch
        new_start = original_start
        new_end = original_end
        actions: list[str] = []

        while new_pitch < low_limit:
            new_pitch += 12
            actions.append("octave_up_to_bass_range")
        while new_pitch > high_limit:
            new_pitch -= 12
            actions.append("octave_down_to_bass_range")

        if settings.correction_mode.lower() == "key_aware" and settings.strict_scale and allowed_pcs and new_pitch % 12 not in allowed_pcs:
            snapped_pitch = nearest_pitch_in_scale(new_pitch, allowed_pcs)
            if snapped_pitch != new_pitch:
                new_pitch = snapped_pitch
                actions.append("scale_snap")

        if settings.quantize_percent > 0:
            new_start = quantize_seconds(original_start, beat_seconds, settings.quantize_percent)
            new_end = quantize_seconds(original_end, beat_seconds, settings.quantize_percent)
            if new_end <= new_start:
                new_end = new_start + max(min_note_seconds, 0.05)
            if abs(new_start - original_start) > 1e-4 or abs(new_end - original_end) > 1e-4:
                actions.append("light_quantize")

        new_velocity = 90 if not settings.keep_human_feel else int(max(45, min(120, note.velocity)))

        if new_start < previous_end:
            new_start = previous_end + 0.003
            if new_end <= new_start:
                new_end = new_start + max(min_note_seconds, 0.05)
            actions.append("monophonic_overlap_fix")

        previous_end = new_end

        cleaned_notes.append(
            pretty_midi.Note(
                velocity=new_velocity,
                pitch=int(new_pitch),
                start=float(new_start),
                end=float(new_end),
            )
        )

        if actions:
            corrections.append(
                {
                    "index": idx,
                    "issue": "note_adjusted",
                    "from_pitch": original_pitch,
                    "to_pitch": int(new_pitch),
                    "actions": actions,
                    "confidence": 0.82,
                }
            )

    cleaned_pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    bass_program = pretty_midi.instrument_name_to_program("Electric Bass (finger)")
    cleaned_instrument = pretty_midi.Instrument(program=bass_program, is_drum=False, name="Bass Clean")
    cleaned_instrument.notes = cleaned_notes
    cleaned_pm.instruments.append(cleaned_instrument)
    cleaned_pm.write(str(output_path))
    return len(cleaned_notes), corrections, resolved_scale_name


def midi_to_musicxml(clean_midi_path: Path, output_musicxml: Path, settings: PipelineSettings) -> None:
    try:
        from music21 import converter, key, meter
    except Exception:
        require_runtime_dependency("music21", "Install it with: pip install music21")

    score = converter.parse(str(clean_midi_path))

    try:
        score.insert(0, meter.TimeSignature(settings.time_signature))
    except Exception:
        pass

    tonic, _ = parse_scale(settings.detected_key)
    if tonic is not None:
        try:
            mode = "minor" if "min" in settings.detected_key.lower() else "major"
            tonic_name = settings.detected_key.split()[0]
            score.insert(0, key.Key(tonic_name, mode))
        except Exception:
            pass

    score.write("musicxml", fp=str(output_musicxml))


def string_layout_for_range(range_mode: str) -> list[tuple[str, int]]:
    """Return bass strings from lowest to highest as (name, open_midi)."""
    if range_mode == "five":
        return [("B", 23), ("E", 28), ("A", 33), ("D", 38), ("G", 43)]
    if range_mode == "drop":
        return [("D", 26), ("A", 33), ("D", 38), ("G", 43)]
    return [("E", 28), ("A", 33), ("D", 38), ("G", 43)]


def choose_string_and_fret(midi_pitch: int, strings: list[tuple[str, int]], previous_string: str | None) -> tuple[str, int]:
    """Pick a playable string/fret, preferring lower fret numbers and phrase continuity."""
    candidates: list[tuple[int, int, str, int]] = []
    for order, (string_name, open_pitch) in enumerate(strings):
        fret = midi_pitch - open_pitch
        if 0 <= fret <= 24:
            continuity_penalty = 0 if previous_string == string_name else 1
            candidates.append((continuity_penalty, fret, string_name, order))

    if not candidates:
        nearest = min(strings, key=lambda item: abs(midi_pitch - item[1]))
        return nearest[0], max(0, midi_pitch - nearest[1])

    candidates.sort(key=lambda item: (item[0], item[1], item[3]))
    _, fret, string_name, _ = candidates[0]
    return string_name, fret


def parse_time_signature_denominator(time_signature: str) -> int:
    try:
        return max(1, int(time_signature.split("/")[1]))
    except Exception:
        return 4


def midi_to_tab_preview(clean_midi_path: Path, settings: PipelineSettings, tempo_bpm: float, limit: int = 64) -> list[dict[str, Any]]:
    """Map cleaned MIDI notes to bass strings/frets for preview and export."""
    pm, _ = load_pretty_midi(clean_midi_path)
    if not pm.instruments:
        return []

    strings = string_layout_for_range(settings.range_mode)
    denominator = parse_time_signature_denominator(settings.time_signature)
    numerator = int(settings.time_signature.split("/")[0]) if "/" in settings.time_signature else 4
    beat_seconds = 60.0 / max(tempo_bpm, 1.0)
    quarter_ratio = 4 / denominator
    bar_seconds = beat_seconds * quarter_ratio * numerator

    preview: list[dict[str, Any]] = []
    previous_string: str | None = None
    notes = sorted(pm.instruments[0].notes, key=lambda note: (note.start, note.pitch))

    for idx, note in enumerate(notes[:limit]):
        string_name, fret = choose_string_and_fret(int(note.pitch), strings, previous_string)
        previous_string = string_name

        bar_number = int(note.start // max(bar_seconds, 1e-6)) + 1
        beat_in_bar = ((note.start % max(bar_seconds, 1e-6)) / max(beat_seconds * quarter_ratio, 1e-6)) + 1

        preview.append(
            {
                "index": idx,
                "pitch": int(note.pitch),
                "start": round(float(note.start), 4),
                "end": round(float(note.end), 4),
                "duration": round(float(note.end - note.start), 4),
                "bar": bar_number,
                "beat": round(beat_in_bar, 3),
                "string": string_name,
                "fret": int(fret),
            }
        )

    return preview


def save_tab_json(tab_preview: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps(tab_preview, indent=2), encoding="utf-8")


def musicxml_to_pdf(musicxml_path: Path, output_pdf_path: Path) -> None:
    candidates = ["musescore", "mscore", "mscore4portable", "MuseScore"]
    exe = next((shutil.which(name) for name in candidates if shutil.which(name)), None)
    if not exe:
        raise HTTPException(
            status_code=500,
            detail="MuseScore CLI not found. Install MuseScore to export PDF automatically.",
        )

    cmd = [exe, str(musicxml_path), "-o", str(output_pdf_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise HTTPException(status_code=500, detail=f"MuseScore PDF export failed: {completed.stderr[:500]}")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "HTTP error on %s %s: %s",
        getattr(request, "method", "?"),
        getattr(getattr(request, "url", None), "path", "?"),
        exc.detail,
    )
    return make_json_error(exc.status_code, str(exc.detail), exc.__class__.__name__)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled error on %s %s",
        getattr(request, "method", "?"),
        getattr(getattr(request, "url", None), "path", "?"),
    )
    return make_json_error(500, str(exc), exc.__class__.__name__)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "app": APP_NAME,
        "status": "ok" if FASTAPI_AVAILABLE else "degraded",
        "fastapi_available": FASTAPI_AVAILABLE,
        "fastapi_import_error": FASTAPI_IMPORT_ERROR,
        "base_dir": str(BASE_DIR),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if FASTAPI_AVAILABLE else "degraded",
        "upload_dir": str(UPLOAD_DIR),
        "runs_dir": str(RUNS_DIR),
        "exports_dir": str(EXPORT_DIR),
        "fastapi_available": FASTAPI_AVAILABLE,
    }


@app.post("/api/transcribe", response_model=PipelineResult if FASTAPI_AVAILABLE else None)
async def transcribe_bass(file: UploadFile = File(...), settings_json: str = Form(...)):
    require_fastapi_runtime()
    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    uploaded_path: Path | None = None

    try:
        settings = PipelineSettings(**json.loads(settings_json))

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file")
        validate_upload_filename(file.filename)

        run_dir.mkdir(parents=True, exist_ok=True)
        upload_name = safe_filename(file.filename)
        uploaded_path = run_dir / upload_name

        with uploaded_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        ensure_file_not_empty(uploaded_path)

        wav_path = maybe_run_ffmpeg_convert(uploaded_path)

        raw_midi_path, raw_issues, transcription_meta = transcribe_with_basic_pitch(wav_path, run_dir)
        tempo_bpm, tempo_source, tempo_diagnostics = resolve_tempo(wav_path, raw_midi_path, settings)

        clean_midi_path = run_dir / "bass_clean.mid"
        note_count, corrections, resolved_scale_name = cleanup_bass_midi(
            midi_path=raw_midi_path,
            output_path=clean_midi_path,
            settings=settings,
            tempo_bpm=tempo_bpm,
        )

        musicxml_path = run_dir / "bass_score.musicxml"
        midi_to_musicxml(clean_midi_path, musicxml_path, settings)

        pdf_path = run_dir / "bass_score.pdf"
        pdf_error: str | None = None
        try:
            musicxml_to_pdf(musicxml_path, pdf_path)
        except HTTPException as exc:
            pdf_error = str(exc.detail)

        tab_preview = midi_to_tab_preview(clean_midi_path, settings, tempo_bpm)
        tab_json_path = run_dir / "bass_tab_preview.json"
        save_tab_json(tab_preview, tab_json_path)

        exports: dict[str, str] = {
            "midi": f"/api/download/{run_id}/bass_clean.mid",
            "musicxml": f"/api/download/{run_id}/bass_score.musicxml",
            "tab_json": f"/api/download/{run_id}/bass_tab_preview.json",
        }
        if pdf_path.exists():
            exports["pdf"] = f"/api/download/{run_id}/bass_score.pdf"

        diagnostics = {
            "raw_transcription_issues": raw_issues,
            "transcription_meta": transcription_meta,
            "pdf_export_error": pdf_error,
            "saved_source": str(uploaded_path.name) if uploaded_path else None,
            "converted_wav": str(wav_path.name),
            "correction_mode": settings.correction_mode,
            "key_root": settings.key_root,
            "key_mode": settings.key_mode,
            "tempo_source": tempo_source,
            "tempo_diagnostics": tempo_diagnostics,
        }

        return {
            "run_id": run_id,
            "input_file": file.filename,
            "tempo_bpm": tempo_bpm,
            "detected_key": resolved_scale_name,
            "time_signature": settings.time_signature,
            "note_count": note_count,
            "corrections": corrections,
            "exports": exports,
            "diagnostics": diagnostics,
            "tab_preview": tab_preview[:24],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Run %s failed with unexpected error: %s", run_id, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Transcription pipeline failed: {exc}",
        )


@app.get("/api/download/{run_id}/{filename}")
def download_export(run_id: str, filename: str):
    require_fastapi_runtime()
    target = RUNS_DIR / run_id / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Requested export file not found.")
    return FileResponse(path=target, filename=target.name)


@app.get("/api/run/{run_id}")
def inspect_run(run_id: str):
    require_fastapi_runtime()
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    files = [
        {
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "download_url": f"/api/download/{run_id}/{item.name}",
        }
        for item in sorted(run_dir.iterdir())
        if item.is_file()
    ]
    return JSONResponse(content={"run_id": run_id, "files": files})


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class BackendUtilityTests(unittest.TestCase):
    def test_resolve_base_dir_returns_path(self) -> None:
        self.assertIsInstance(resolve_base_dir(), Path)

    def test_safe_filename_keeps_valid_chars(self) -> None:
        self.assertEqual(safe_filename("bass stem 01.wav"), "bass stem 01.wav")

    def test_safe_filename_removes_invalid_chars(self) -> None:
        self.assertEqual(safe_filename("../../bad:*?name.wav"), "....badname.wav")

    def test_parse_scale_minor(self) -> None:
        tonic, scale = parse_scale("E minor")
        self.assertEqual(tonic, 4)
        self.assertEqual(scale, {4, 6, 7, 9, 11, 0, 2})

    def test_parse_scale_invalid(self) -> None:
        tonic, scale = parse_scale("not_a_real_key")
        self.assertIsNone(tonic)
        self.assertIsNone(scale)

    def test_build_scale_from_settings_key_aware_major(self) -> None:
        settings = PipelineSettings(correction_mode="key_aware", key_root="Bb", key_mode="major")
        tonic, scale, label = build_scale_from_settings(settings)
        self.assertEqual(tonic, 10)
        self.assertEqual(label, "Bb major")
        self.assertIn(10, scale or set())

    def test_build_scale_from_settings_key_aware_minor(self) -> None:
        settings = PipelineSettings(correction_mode="key_aware", key_root="G", key_mode="natural_minor")
        tonic, scale, label = build_scale_from_settings(settings)
        self.assertEqual(tonic, 7)
        self.assertEqual(label, "G natural minor")
        self.assertIn(10, scale or set())

    def test_build_scale_from_settings_chromatic(self) -> None:
        settings = PipelineSettings(correction_mode="chromatic")
        tonic, scale, label = build_scale_from_settings(settings)
        self.assertIsNone(tonic)
        self.assertIsNone(scale)
        self.assertEqual(label, "Chromatic")

    def test_range_limits(self) -> None:
        self.assertEqual(range_limits("standard4"), (28, 67))
        self.assertEqual(range_limits("five"), (23, 67))
        self.assertEqual(range_limits("drop"), (26, 67))

    def test_nearest_pitch_in_scale_prefers_closest_candidate(self) -> None:
        _, c_major = parse_scale("C major")
        assert c_major is not None
        self.assertEqual(nearest_pitch_in_scale(61, c_major), 60)

    def test_quantize_seconds_zero_percent_keeps_value(self) -> None:
        self.assertAlmostEqual(quantize_seconds(0.37, 0.5, 0), 0.37)

    def test_quantize_seconds_full_percent_snaps_to_grid(self) -> None:
        self.assertAlmostEqual(quantize_seconds(0.37, 0.5, 100), 0.375)

    def test_estimate_tempo_from_midi_returns_none_for_missing_file(self) -> None:
        self.assertIsNone(estimate_tempo_from_midi(Path("missing.mid")))

    def test_resolve_tempo_prefers_override(self) -> None:
        settings = PipelineSettings(tempo_override=98)
        tempo, source, details = resolve_tempo(Path("missing.wav"), Path("missing.mid"), settings)
        self.assertEqual(tempo, 98.0)
        self.assertEqual(source, "tempo_override")
        self.assertEqual(details["tempo_override"], 98.0)

    def test_make_json_error_shape(self) -> None:
        response = make_json_error(418, "teapot", "ExampleError")
        if hasattr(response, "content"):
            self.assertEqual(response.content["detail"], "teapot")
            self.assertEqual(response.content["error_type"], "ExampleError")
            self.assertEqual(response.status_code, 418)
        else:
            self.assertEqual(response["detail"], "teapot")

    def test_validate_upload_filename_rejects_txt(self) -> None:
        with self.assertRaises(HTTPException) as context:
            validate_upload_filename("notes.txt")
        self.assertEqual(context.exception.status_code, 400)

    def test_ensure_file_not_empty_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty = Path(tmpdir) / "empty.wav"
            empty.write_bytes(b"")
            with self.assertRaises(HTTPException) as context:
                ensure_file_not_empty(empty)
            self.assertEqual(context.exception.status_code, 400)

    def test_run_self_tests_returns_result(self) -> None:
        result = run_self_tests()
        self.assertTrue(hasattr(result, "testsRun"))

    def test_string_layout_for_standard4(self) -> None:
        self.assertEqual(string_layout_for_range("standard4"), [("E", 28), ("A", 33), ("D", 38), ("G", 43)])

    def test_choose_string_and_fret_prefers_continuity(self) -> None:
        string_name, fret = choose_string_and_fret(40, string_layout_for_range("standard4"), "D")
        self.assertEqual((string_name, fret), ("D", 2))

    def test_choose_string_and_fret_finds_playable_position(self) -> None:
        string_name, fret = choose_string_and_fret(45, string_layout_for_range("standard4"), None)
        self.assertIn(string_name, {"A", "D", "G"})
        self.assertGreaterEqual(fret, 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        if not FASTAPI_AVAILABLE:
            raise SystemExit(
                "FastAPI is not installed in this environment. "
                "Run the server where FastAPI is available."
            )
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        # Do not call unittest.main() here. Some environments pass extra argv or
        # treat 'no tests' as SystemExit(5), which breaks code preview execution.
        run_self_tests()
