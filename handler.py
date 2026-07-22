"""
handler.py — RunPod serverless entry point for the Parakeet batch engine.

The engine (parakeet_engine.py) is the exact code validated on an A40 pod
across ~625 audio-hours of real-world conversational speech (2026-07-22).

Input schema (inside "input"):
  audios       list      REQUIRED. Either plain URL strings, or objects
                         with "audio_url" plus any metadata keys
                         (chunk_id, session_id, speaker, ...) which are
                         passed through to the corresponding result.
  timestamps   bool      include word + segment timestamps (default true)

Top-level "webhook" works as with any RunPod endpoint: the full response
is POSTed to your URL on completion.

Response:
  {"results": [...per-file dicts...], "counts": {...},
   "model": ..., "total_audio_sec": ..., "wall_time_sec": ...,
   "gpu_time_sec": ..., "rtfx_end_to_end": ..., "rtfx_compute_only": ...,
   "success": true}
Per-file dicts carry text, duration_sec, segments[], words[], all input
metadata, or an "error" string (with ffmpeg stderr) on per-file failure.
"""

import os
import resource

# No core dumps: a crash of a ~40GB process must not fill container disk
# (learned the hard way on the pod).
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import runpod  # noqa: E402

from parakeet_engine import ParakeetEngine  # noqa: E402

MAX_FILES_PER_JOB = int(os.environ.get("MAX_FILES_PER_JOB", 500))

# Loaded once per worker; reused across jobs (warm starts). The engine's
# full<->local attention handling is warm-reuse safe (validated across 21
# consecutive waves on one process).
ENGINE = ParakeetEngine()


def _normalize(audios) -> list[dict]:
    items = []
    for i, a in enumerate(audios):
        if isinstance(a, str):
            items.append({"audio_url": a, "_order": i})
        elif isinstance(a, dict) and a.get("audio_url"):
            item = dict(a)
            item["_order"] = i
            items.append(item)
    return items


def handler(job):
    inp = job.get("input") or {}
    audios = inp.get("audios")
    if not isinstance(audios, list) or not audios:
        return {"error": "input.audios must be a non-empty list of URLs "
                         "or {audio_url, ...} objects", "success": False}
    if len(audios) > MAX_FILES_PER_JOB:
        return {"error": f"{len(audios)} files > MAX_FILES_PER_JOB="
                         f"{MAX_FILES_PER_JOB}; split into multiple jobs "
                         f"(recommended: 200-500 per job, sent in "
                         f"parallel)", "success": False}

    items = _normalize(audios)
    if not items:
        return {"error": "no valid audio_url entries found",
                "success": False}

    timestamps = bool(inp.get("timestamps", True))
    result = ENGINE.transcribe(items, timestamps=timestamps)
    result["results"] = result.pop("files")
    result["success"] = True
    return result


runpod.serverless.start({"handler": handler})
