# Parakeet TDT 0.6B v2 — Serverless Batch Transcription Worker

English-only ASR worker for RunPod Serverless, optimized for cost per
audio-hour. Every configuration value below was validated on an A40 pod
across **~625 audio-hours of real production classroom audio** (two full
school days, 2026-07-22).

## Measured performance (A40, 48GB, baseline profile)

| Metric | Value |
| --- | --- |
| RTFx compute-only | **306x** (with word+segment timestamps) |
| RTFx end-to-end (incl. downloads) | **293x** |
| Full day (334 audio-hr, 8,047 files) | 68.4 min wall |
| Cost for 500 audio-hr/day | ~1.7 GPU-hr ≈ **$0.75/day** |
| Per-file failure rate (corrupt sources) | 0.3–0.6%, isolated per file |

## Deploy (GitHub integration — no registry needed, $0)

1. Push this folder to a GitHub repo.
2. RunPod console → Settings → Connections → GitHub → authorize the repo.
3. Serverless → **New Endpoint** → GitHub Repo → select repo, branch
   `main`. RunPod builds the Dockerfile on their infra (~20–30 min: the
   model bake makes the image ~12GB — intentional; workers cold-start
   with weights on disk, no network volume, no bootstrap).
4. Endpoint settings (the validated envelope):
   - **GPU**: A40 / A6000 / L40S (48GB class). **CUDA filter: >= 12.4**
     (image is torch 2.6+cu124; older hosts will fail to initialize).
   - **Workers**: min 1 during your processing window (avoids cold
     starts), max = desired parallelism (5 workers clear a full day in
     ~15 min of wall time).
   - **Execution timeout**: 1800s covers a 400-file job with margin.
   - No network volume.
5. New GitHub Release tag → automatic rebuild/redeploy. Rollback from
   the endpoint's Builds tab.

## API

```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
{
  "input": {
    "audios": [
      {"audio_url": "https://...", "chunk_id": "395366",
       "session_id": "TST...", "speaker": "Aelish Mahajan"},
      "https://plain-url-also-works/..."
    ],
    "timestamps": true
  },
  "webhook": "https://your-server.com/hooks/parakeet-done"
}
```

- **200–500 files per job** is the sweet spot (amortizes overhead,
  bounds failure blast radius, parallelizes across workers). Hard cap:
  `MAX_FILES_PER_JOB` env (default 500).
- Any HTTP(S) URL: S3/R2 presigned, ActivePieces file links (worker
  sends a browser UA and follows redirects — required for AP), CDN.
- Formats: webm/wav/mp3/m4a/flac/ogg/opus (ffmpeg converts internally).
- Metadata keys on input objects pass through to the matching result.
- Per-file failures return `"error"` with ffmpeg stderr (e.g. the
  production `matroska ... 0x00 at pos 36` corrupt-capture signature)
  without failing the job.
- `webhook` receives the full response; key your DB on the job `id`.

Per-file result:

```json
{"chunk_id": "395366", "speaker": "...", "duration_sec": 187.4,
 "text": "...",
 "segments": [{"start": 0.0, "end": 4.2, "text": "..."}],
 "words":    [{"start": 0.0, "end": 0.6, "text": "..."}]}
```

## Lane configuration

Duration-bucketed batching (attention memory ~ batch x duration^2; files
>300s run under local attention, which is ~linear in duration). Defaults
are the validated production configuration — do not change without a
controlled A/B on identical data:

| Env | Default | Notes |
| --- | --- | --- |
| SHORT_BS / SHORT_FLUSH | 16 / 64 | files <= 60s |
| MEDIUM_BS / MEDIUM_FLUSH | 4 / 16 | 60–300s |
| LONG_BS | 2 | >300s, local attention |

Peak VRAM ~18GB. Other env: `PARAKEET_MODEL`, `DOWNLOAD_WORKERS` (10),
`DOWNLOAD_TIMEOUT`, `DOWNLOAD_RETRIES`, `MAX_DURATION_SEC` (3h guard —
split longer files upstream), `MAX_FILES_PER_JOB` (500).

Note: a higher-batch profile (SHORT_BS=64, MEDIUM_BS=8, LONG_BS=4) was
trialed on 2026-07-22; it raised GPU utilization (73% -> ~85%, peak
29.4GB) but showed no clear aggregate throughput win, and was rejected.
The engine warns if these values are set on a <40GB GPU.

## Reliability features (each one earned by a pod crash)

- Torch pinned 2.6+cu124 with build-time assertion (version-cascade
  guard).
- Per-flush GPU memory release via del+gc — **without**
  `torch.cuda.empty_cache()`, which invalidates NeMo's TDT CUDA graphs
  (illegal memory access).
- Subsampling conv chunking set to auto once at init — NeMo's toggle
  revert corrupts the module on warm full<->local<->full reuse.
- OOM on a flush -> halve batch and retry, down to 1.
- Concurrent downloads (10 threads) pipelined with transcription;
  browser UA; 2 retries.
- Per-flush WAV deletion; core dumps disabled; per-file error isolation.

## Known limitations

- English-only (route code-switched Hindi/Kannada audio to a Whisper
  fallback endpoint).
- No `initial_prompt` equivalent (transducer): keep the proper-noun
  find/replace post-pass for names/StayQrious terms.
- Files > 3h rejected by guard — split upstream.
- Speaker diarization not included; upstream per-speaker chunking + the
  metadata passthrough covers turn attribution.
