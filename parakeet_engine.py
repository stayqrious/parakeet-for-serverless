"""
parakeet_engine.py — batch transcription engine for Parakeet TDT 0.6B v2 (v2).

Changes from v1 (informed by the 20-file A40 run):
- FLUSH_SIZE decoupled from batch_size: NeMo's transcribe() batches
  internally, so we accumulate large flushes (fewer dataloader setups,
  each ~1s of overhead) while keeping the memory-safe batch size.
- Medium lane widened to 600s at batch 4 (memory ~ bs * dur^2; 10-min
  files at bs 4 need ~8GB, safe on 48GB). Long lane (>600s) now batch 2
  under local attention instead of 1. The long lane dominated wall time
  in v1 (23s of 32s).
- WAV files deleted after their flush (200-file jobs would otherwise
  hold multiple GB of temp audio until job end).
- verbose=False on transcribe; single [progress] line per flush instead.
- v3: MEDIUM_MAX reverted 600->300 (600s at bs4 OOM'd when stacked on
  retained memory); 300-600s files go to the long lane, which is safe
  because local attention memory is ~linear in duration. Added per-flush
  GPU memory release (synchronize + del hyps + gc + empty_cache).
  Do NOT run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —
  it interacts badly with empty_cache on torch 2.6 (illegal memory
  access during tensor dealloc, observed on the A40).
"""

import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ----------------------------- tunables ------------------------------------
# All env-overridable. Defaults = the configuration validated on the A40
# pod (full-day run, ~306x e2e). Tuned profile per GPU-utilization
# profiling (avg ~73% util, peak 18GB/46GB => headroom):
#   SHORT_BS=64 SHORT_FLUSH=128 MEDIUM_BS=8 MEDIUM_FLUSH=32 LONG_BS=4
import os as _os

def _env_int(name, default):
    return int(_os.environ.get(name, default))

MODEL_NAME = _os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")

DOWNLOAD_WORKERS = _env_int("DOWNLOAD_WORKERS", 10)
DOWNLOAD_TIMEOUT = _env_int("DOWNLOAD_TIMEOUT", 180)
DOWNLOAD_RETRIES = _env_int("DOWNLOAD_RETRIES", 2)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                         "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}

# lane: (max_duration_sec, batch_size, flush_size)
SHORT_MAX = _env_int("SHORT_MAX", 60)
SHORT_BS = _env_int("SHORT_BS", 16)
SHORT_FLUSH = _env_int("SHORT_FLUSH", 64)
MEDIUM_MAX = _env_int("MEDIUM_MAX", 300)
MEDIUM_BS = _env_int("MEDIUM_BS", 4)
MEDIUM_FLUSH = _env_int("MEDIUM_FLUSH", 16)
LONG_BS = _env_int("LONG_BS", 2)  # local attention, end-of-job pass
LOCAL_ATTN_CONTEXT = [256, 256]
MAX_DURATION_SEC = _env_int("MAX_DURATION_SEC", 3 * 3600)

# ----------------------------------------------------------------------------


class ParakeetEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        import nemo.collections.asr as nemo_asr
        print("[engine] version v4.0")
        print(f"[engine] lanes: short<={SHORT_MAX}s "
              f"bs{SHORT_BS}/fl{SHORT_FLUSH} | "
              f"medium<={MEDIUM_MAX}s "
              f"bs{MEDIUM_BS}/fl{MEDIUM_FLUSH} | "
              f"long bs{LONG_BS}")
        print(f"[engine] loading {model_name} ...")
        t = time.time()
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name)
        self.model.eval()
        # Subsampling conv chunking: set to AUTO once and never toggle.
        # NeMo's change_subsampling_conv_chunking_factor(-1) revert leaves
        # the swapped MaskedConvSequential in a broken half-state (crash:
        # "forward() missing 1 required positional argument: 'lengths'"
        # on the first full-attention transcribe after a local round-trip,
        # hit by wave 2 of the full-day run). Auto mode chunks only when a
        # file actually needs it, so short batches are unaffected.
        self.model.change_subsampling_conv_chunking_factor(1)
        self._attn = "full"
        # VRAM guard: tuned lane profiles (SHORT_BS=64 etc.) were validated
        # on 46GB (peak 29.4GB). Warn loudly if someone runs them on less.
        import torch as _torch
        vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[engine] GPU: {_torch.cuda.get_device_name(0)} "
              f"({vram_gb:.0f}GB)")
        if vram_gb < 39 and (SHORT_BS > 16 or MEDIUM_BS > 4 or LONG_BS > 2):
            print("[engine] WARNING: tuned lane profile on <40GB GPU -- "
                  "peak demand ~29GB will OOM; the retry net will degrade "
                  "batches, but you should unset the lane env vars on "
                  "this hardware.")
        print(f"[engine] model ready in {time.time() - t:.1f}s")

    def _set_attention(self, mode: str) -> None:
        if mode == self._attn:
            return
        if mode == "local":
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=LOCAL_ATTN_CONTEXT)
        else:
            self.model.change_attention_model(self_attention_model="rel_pos")
        self._attn = mode
        print(f"[engine] attention -> {mode}")

    @staticmethod
    def _fetch_and_convert(item: dict, workdir: str) -> dict:
        uid = uuid.uuid4().hex[:8]
        raw = Path(workdir) / f"raw_{uid}"
        wav = Path(workdir) / f"wav_{uid}.wav"
        try:
            src = item.get("audio_url") or item.get("path")
            last_err = None
            if src.startswith("http"):
                for _ in range(DOWNLOAD_RETRIES + 1):
                    try:
                        with requests.get(src, headers=HEADERS, stream=True,
                                          timeout=DOWNLOAD_TIMEOUT) as r:
                            r.raise_for_status()
                            with open(raw, "wb") as f:
                                for chunk in r.iter_content(1 << 20):
                                    f.write(chunk)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                if last_err:
                    raise last_err
                src = str(raw)
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", src,
                 "-ac", "1", "-ar", "16000", str(wav)],
                capture_output=True, timeout=600)
            if proc.returncode != 0:
                stderr = (proc.stderr or b"").decode(
                    "utf-8", "replace").strip()
                raise RuntimeError(f"ffmpeg failed: {stderr[:300]}")
            raw.unlink(missing_ok=True)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
                capture_output=True, text=True, timeout=60, check=True)
            dur = float(probe.stdout.strip())
            if dur > MAX_DURATION_SEC:
                raise ValueError(
                    f"duration {dur:.0f}s exceeds {MAX_DURATION_SEC}s; "
                    f"split upstream")
            item["wav"] = str(wav)
            item["duration_sec"] = round(dur, 2)
        except Exception as e:
            item["error"] = str(e)[:400]
            raw.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        return item

    @staticmethod
    def _ts(hyp, level: str) -> list:
        out = []
        for t in (getattr(hyp, "timestamp", None) or {}).get(level, []):
            out.append({
                "start": round(float(t.get("start", 0.0)), 3),
                "end": round(float(t.get("end", 0.0)), 3),
                "text": t.get(level) or t.get("word") or t.get("segment")
                        or "",
            })
        return out

    def _run_batch(self, group: list, batch_size: int,
                   timestamps: bool) -> None:
        if not group:
            return
        import torch as _torch
        bs = batch_size
        while True:
            try:
                hyps = self.model.transcribe(
                    [g["wav"] for g in group], batch_size=bs,
                    timestamps=timestamps, verbose=False)
                break
            except _torch.OutOfMemoryError:
                # Safety net for tuned/unusual configs: degrade this flush
                # to a smaller batch instead of failing the job.
                _torch.cuda.synchronize()
                import gc as _gc
                _gc.collect()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[engine] OOM -> retrying flush at batch_size={bs}")
        for g, h in zip(group, hyps):
            g["text"] = h.text
            if timestamps:
                g["segments"] = self._ts(h, "segment")
                g["words"] = self._ts(h, "word")
            Path(g["wav"]).unlink(missing_ok=True)  # free disk per flush
        # Memory hygiene: hypothesis objects hold GPU tensors (alignments
        # from timestamp decoding); release them NOW, not at GC's leisure,
        # or allocations stack across flushes (observed: 39GB retained
        # after 2 flushes on the 200-file A40 run).
        # Memory hygiene v3.2: del + gc returns hypothesis tensors (GPU
        # alignments from timestamp decoding) to torch's cache so the next
        # flush reuses that memory -- this is what prevents the 39GB
        # retention OOM. Do NOT call torch.cuda.empty_cache() here: NeMo's
        # TDT decoder uses CUDA graphs whose captured pointers reference the
        # cached pool; releasing it causes an illegal memory access on the
        # next transcribe call (observed twice, deterministic, on the A40).
        del hyps
        import gc
        gc.collect()

    def transcribe(self, items: list[dict], timestamps: bool = True) -> dict:
        t0 = time.time()
        total_n = len(items)
        short_q, medium_q, long_q, done = [], [], [], []
        gpu_time = 0.0
        audio_done = 0.0

        def flush(queue: list, bs: int) -> None:
            nonlocal gpu_time, audio_done
            if not queue:
                return
            t = time.time()
            self._run_batch(queue, bs, timestamps)
            gpu_time += time.time() - t
            audio_done += sum(x["duration_sec"] for x in queue)
            done.extend(queue)
            queue.clear()
            print(f"[progress] {len(done)}/{total_n} files, "
                  f"{audio_done:.0f}s audio, gpu {gpu_time:.1f}s")

        with tempfile.TemporaryDirectory() as workdir, \
             ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futs = [pool.submit(self._fetch_and_convert, dict(it), workdir)
                    for it in items]

            for fut in as_completed(futs):
                it = fut.result()
                if "error" in it:
                    done.append(it)
                    continue
                d = it["duration_sec"]
                if d <= SHORT_MAX:
                    short_q.append(it)
                    if len(short_q) >= SHORT_FLUSH:
                        flush(short_q, SHORT_BS)
                elif d <= MEDIUM_MAX:
                    medium_q.append(it)
                    if len(medium_q) >= MEDIUM_FLUSH:
                        flush(medium_q, MEDIUM_BS)
                else:
                    long_q.append(it)

            flush(short_q, SHORT_BS)
            flush(medium_q, MEDIUM_BS)

            if long_q:
                self._set_attention("local")
                flush(long_q, LONG_BS)
                self._set_attention("full")

        for it in done:
            it.pop("wav", None)

        ok = [x for x in done if "error" not in x]
        total_audio = sum(x["duration_sec"] for x in ok)
        wall = time.time() - t0
        return {
            "model": MODEL_NAME,
            "files": sorted(done, key=lambda x: x.get("_order", 0)),
            "counts": {"ok": len(ok), "failed": len(done) - len(ok)},
            "total_audio_sec": round(total_audio, 1),
            "wall_time_sec": round(wall, 1),
            "gpu_time_sec": round(gpu_time, 1),
            "rtfx_end_to_end": round(total_audio / wall, 1) if wall else 0,
            "rtfx_compute_only": round(total_audio / gpu_time, 1)
                                 if gpu_time else 0,
        }
