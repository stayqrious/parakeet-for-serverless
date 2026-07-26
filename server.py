"""
server.py — long-running HTTP server for POD-mode Parakeet transcription.

Wire-compatible with the RunPod serverless endpoint so a caller can point at
either surface by changing only the base URL:

  POST   /run           -> {"id", "status": "IN_QUEUE"}         (non-blocking)
  GET    /status/<id>   -> {"id", "status", "output"?}          (404 if unknown)
  GET    /requests      -> {"requests": [{"id", "status"}, ...]}
  GET    /health        -> {"ready", "jobs", "workers", ...}
  DELETE /status/<id>   -> {"deleted": true}

One GPU => exactly one worker thread drains a FIFO queue. ParakeetEngine is not
safe for concurrent transcribe() calls (TDT CUDA graphs, and _set_attention()
mutates the model globally), so jobs are strictly serialized.

Serverless entry point (handler.py) is untouched and still works; a pod selects
this file via RunPod's dockerStartCmd:

  "dockerStartCmd": ["python", "-u", "server.py"]

Stdlib only by design — the image installs nemo_toolkit[asr], runpod and
requests and nothing else, so there is no FastAPI/uvicorn here.
"""

import json
import os
import queue
import re
import resource
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# No core dumps: a crash of a ~40GB process must not fill container disk.
# (Same guard as handler.py — learned on the A40 pod.)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

from parakeet_engine import ParakeetEngine  # noqa: E402

PORT = int(os.environ.get("PORT", 8000))
MAX_FILES_PER_JOB = int(os.environ.get("MAX_FILES_PER_JOB", 500))
# How long a terminal job's output is retained for collection. Generous: the
# caller polls every ~30s and may be mid-batch. Eviction is lazy.
RESULT_TTL_S = int(os.environ.get("RESULT_TTL_S", 6 * 3600))

_JOBS = {}                      # job_id -> {"status", "output", "created", "finished"}
_LOCK = threading.Lock()
_QUEUE = queue.Queue()
_READY = threading.Event()
_LOAD_ERROR = None              # set if the engine never reaches VRAM
ENGINE = None


def _normalize(audios):
    """Mirror of handler.py's _normalize. Kept local so importing this module
    never pulls in handler.py (which calls runpod.serverless.start() at import
    time). If handler.py's version changes, change this one too."""
    items = []
    for i, a in enumerate(audios):
        if isinstance(a, str):
            items.append({"audio_url": a, "_order": i})
        elif isinstance(a, dict) and a.get("audio_url"):
            item = dict(a)
            item["_order"] = i
            items.append(item)
    return items


def _evict_expired():
    now = time.time()
    with _LOCK:
        for jid in [j for j, v in _JOBS.items()
                    if v.get("finished") and now - v["finished"] > RESULT_TTL_S]:
            del _JOBS[jid]


def _worker():
    """Single consumer. Serializes all GPU work."""
    global ENGINE, _LOAD_ERROR
    print("[server] loading engine ...", flush=True)
    try:
        ENGINE = ParakeetEngine()
    except Exception as e:
        # Never leave the caller polling a silent /health forever: keep the
        # socket up, stay ready:false, and report why (no GPU, bad CUDA
        # version, model download failure, ...).
        import traceback
        traceback.print_exc()
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[server] FATAL: engine load failed: {_LOAD_ERROR}", flush=True)
        return
    _READY.set()
    print("[server] engine ready; listening for jobs", flush=True)

    while True:
        job_id, items, timestamps = _QUEUE.get()
        with _LOCK:
            if job_id not in _JOBS:      # cancelled before it started
                _QUEUE.task_done()
                continue
            _JOBS[job_id]["status"] = "IN_PROGRESS"
        print(f"[server] job {job_id}: START ({len(items)} files)", flush=True)
        try:
            result = ENGINE.transcribe(items, timestamps=timestamps)
            result["results"] = result.pop("files")
            result["success"] = True
            status = "COMPLETED"
        except Exception as e:
            # Job-level failure only. Per-file errors are already isolated
            # inside the engine and surface as an "error" key per result.
            import traceback
            traceback.print_exc()
            result = {"success": False, "error": f"{type(e).__name__}: {e}"}
            status = "FAILED"
        with _LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(status=status, output=result,
                                     finished=time.time())
        print(f"[server] job {job_id}: {status}", flush=True)
        _QUEUE.task_done()
        _evict_expired()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # quieter default access log
        print(f"[http] {fmt % args}", flush=True)

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        """Always drain the request body before replying. Under HTTP/1.1
        keep-alive an undrained body is parsed as the next request line, so a
        requests.Session that gets an early 503/404 on /run would see garbage
        on its next call."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        return self.rfile.read(n) if n > 0 else b""

    # ---------------- GET ----------------
    def do_GET(self):
        if self.path.startswith("/health"):
            with _LOCK:
                counts = {}
                for v in _JOBS.values():
                    counts[v["status"]] = counts.get(v["status"], 0) + 1
            ready = _READY.is_set()
            payload = {
                "ready": ready,
                "model": os.environ.get("PARAKEET_MODEL",
                                        "nvidia/parakeet-tdt-0.6b-v2"),
                "jobs": {"inQueue": counts.get("IN_QUEUE", 0),
                         "inProgress": counts.get("IN_PROGRESS", 0),
                         "completed": counts.get("COMPLETED", 0),
                         "failed": counts.get("FAILED", 0)},
                "workers": {"ready": 1 if ready else 0,
                            "running": counts.get("IN_PROGRESS", 0)},
            }
            if _LOAD_ERROR:
                payload["error"] = _LOAD_ERROR
            return self._send(200, payload)

        if self.path.startswith("/requests"):
            with _LOCK:
                return self._send(200, {"requests": [
                    {"id": j, "status": v["status"]} for j, v in _JOBS.items()]})

        m = re.match(r"^/status/([^/?]+)", self.path)
        if m:
            with _LOCK:
                job = _JOBS.get(m.group(1))
                if job is None:
                    return self._send(404, {"status": 404, "title": "Not Found",
                                            "detail": "job not found"})
                out = {"id": m.group(1), "status": job["status"]}
                if job["status"] in ("COMPLETED", "FAILED"):
                    out["output"] = job.get("output")
                return self._send(200, out)

        return self._send(404, {"error": "unknown path"})

    # ---------------- POST ----------------
    def do_POST(self):
        raw = self._read_body()
        if not self.path.startswith("/run"):
            return self._send(404, {"error": "unknown path"})
        if not _READY.is_set():
            if _LOAD_ERROR:
                return self._send(503, {"error": f"engine failed to load: "
                                                 f"{_LOAD_ERROR}"})
            return self._send(503, {"error": "engine still loading"})

        try:
            body = json.loads(raw or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"invalid JSON: {e}"})

        inp = body.get("input") or {}
        audios = inp.get("audios")
        if not isinstance(audios, list) or not audios:
            return self._send(400, {"error": "input.audios must be a non-empty "
                                             "list of URLs or {audio_url,...} objects"})
        if len(audios) > MAX_FILES_PER_JOB:
            return self._send(400, {"error": f"{len(audios)} files > "
                                             f"MAX_FILES_PER_JOB={MAX_FILES_PER_JOB}"})
        items = _normalize(audios)
        if not items:
            return self._send(400, {"error": "no valid audio_url entries found"})

        job_id = str(uuid.uuid4())
        with _LOCK:
            _JOBS[job_id] = {"status": "IN_QUEUE", "output": None,
                             "created": time.time(), "finished": None}
        _QUEUE.put((job_id, items, bool(inp.get("timestamps", True))))
        return self._send(200, {"id": job_id, "status": "IN_QUEUE"})

    # ---------------- DELETE ----------------
    def do_DELETE(self):
        self._read_body()
        m = re.match(r"^/status/([^/?]+)", self.path)
        if not m:
            return self._send(404, {"error": "unknown path"})
        with _LOCK:
            existed = _JOBS.pop(m.group(1), None) is not None
        return self._send(200 if existed else 404, {"deleted": existed})


if __name__ == "__main__":
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"[server] HTTP up on 0.0.0.0:{PORT} (engine loading in background)",
          flush=True)
    srv.serve_forever()
