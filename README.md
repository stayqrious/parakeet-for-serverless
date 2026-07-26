# Parakeet TDT 0.6B v2 — Serverless Batch Transcription Worker

English-only ASR worker for RunPod Serverless, optimized for cost per
audio-hour. Every configuration value below was validated on an A40 pod
across **~625 audio-hours of real-world conversational speech** .

## Measured performance (A40, 48GB, baseline profile)

| Metric | Value |
| --- | --- |
| RTFx compute-only | **306x** (with word+segment timestamps) |
| RTFx end-to-end (incl. downloads) | **293x** |
| (334 audio-hr, 8,047 files) | 68.4 min wall |
| Cost for 500 audio-hr/day | ~1.7 GPU-hr  |
| Per-file failure rate (corrupt sources) | 0.3–0.6%, isolated per file |

## Deploy (GitHub integration — no registry needed, $0)

1. Push this folder to a GitHub repo.
2. RunPod console → Settings → Connections → GitHub → authorize the repo.
3. Serverless → **New Endpoint** → GitHub Repo → select repo, branch
   `main`. RunPod builds the Dockerfile on their infra (~20–30 min: the
   model bake makes the image ~12GB — intentional; workers cold-start
   with weights on disk, nothing to fetch at boot).
4. Endpoint settings (the validated envelope):
   - **GPU**: A40 / A6000 / L40S (48GB class). **CUDA filter: >= 12.4**
     (image is torch 2.6+cu124; older hosts will fail to initialize).
   - **Workers**: min 1 during your processing window (avoids cold
     starts), max = desired parallelism (5 workers clear a full day in
     ~15 min of wall time).
   - **Execution timeout**: 1800s covers a 400-file job with margin.
   5. New GitHub Release tag → automatic rebuild/redeploy. Rollback from
   the endpoint's Builds tab.

## API

```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
{
  "input": {
    "audios": [
      {"audio_url": "https://...", "chunk_id": "0001",
       "session_id": "S-0001", "speaker": "Speaker A"},
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
- Any HTTP(S) URL: S3/R2 presigned, file services that need a browser UA or redirect following, CDN.
- Formats: webm/wav/mp3/m4a/flac/ogg/opus (ffmpeg converts internally).
- Metadata keys on input objects pass through to the matching result.
- Per-file failures return `"error"` with ffmpeg stderr (e.g. the
  production `matroska ... 0x00 at pos 36` corrupt-capture signature)
  without failing the job.
- `webhook` receives the full response; key your DB on the job `id`.

Per-file result:

```json
{"chunk_id": "0001", "speaker": "...", "duration_sec": 187.4,
 "text": "...",
 "segments": [{"start": 0.0, "end": 4.2, "text": "..."}],
 "words":    [{"start": 0.0, "end": 0.6, "text": "..."}]}
```

## Pod mode (for once-daily batch jobs)

Serverless bills a standby fee on top of GPU time — it is a substantial share of total spend.
That fee buys ~10 hosts keeping the 12GB image warm: real value for
latency-sensitive traffic, poor value for a once-daily batch. A serverless
day cost a large bill a comparable pod day cost $2.13.

So a batch caller can instead start **one pod**, run every batch against
it, and terminate it. `server.py` is that surface — a stdlib-only
`ThreadingHTTPServer`, **wire-compatible with the serverless API**, so an
existing client switches by changing only the base URL:

| Method | Path | Returns |
| --- | --- | --- |
| `POST` | `/run` | `{"id", "status": "IN_QUEUE"}` — returns immediately; body identical to the serverless `input` payload above |
| `GET` | `/status/<id>` | `{"id","status","output"?}`; status ∈ `IN_QUEUE`\|`IN_PROGRESS`\|`COMPLETED`\|`FAILED`; `output` only when terminal; **404** if unknown |
| `GET` | `/requests` | `{"requests": [{"id","status"}, ...]}` |
| `GET` | `/health` | `{"ready", "jobs", "workers", "model"}` (plus `error` if the engine failed to load) |
| `DELETE` | `/status/<id>` | `{"deleted": true}` — release a retained result |

`output` on a `COMPLETED` job is byte-shape-identical to the serverless
response. One GPU means **one job at a time**: a single worker thread
drains a FIFO queue, because `ParakeetEngine` is not safe for concurrent
`transcribe()` calls (TDT CUDA graphs, and the full↔local attention switch
mutates the model globally). The HTTP socket opens *before* the model
loads so the caller can poll — `/health` reports `ready: false` and `/run`
returns **503** until weights are in VRAM.

Pods **cannot** pull from `registry.runpod.net` (that registry is bound to
the serverless build system: *"Failed to get Hub registry auth"*), so
`.github/workflows/build-pod-image.yml` publishes the same Dockerfile to
GHCR on every push to `main`, on release, or on demand:

```
ghcr.io/stayqrious/parakeet-for-serverless:sha-<commit>   # pin this
ghcr.io/stayqrious/parakeet-for-serverless:latest
```

**One-time setup after the first successful build:** repo → Packages →
`parakeet-for-serverless` → Package settings → **Change visibility →
Public**. GHCR packages default to private even on a public repo, and a
private package fails the pod pull exactly like `registry.runpod.net`
does. (Alternative: a PAT with `read:packages` registered via RunPod's
container-registry-auth API, passed as `containerRegistryAuthId`.)

Create the pod with the **v1** API — v2 lacks `allowedCudaVersions`, which
this image requires (torch 2.6+cu124):

```jsonc
POST https://rest.runpod.io/v1/pods
{
  "name": "parakeet-batch",
  "imageName": "ghcr.io/stayqrious/parakeet-for-serverless:sha-<commit>",
  "cloudType": "COMMUNITY",
  "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"],
  "gpuCount": 1,
  "containerDiskInGb": 30,   // long lane holds ~10GB of WAV per 450-file batch
  "volumeInGb": 0,           // the image is self-contained
  "ports": ["8000/http"],
  "allowedCudaVersions": ["12.4","12.5","12.6","12.7","12.8","12.9","13.0"],
  "dockerStartCmd": ["python", "-u", "server.py"]
}
```

Then poll `https://{podId}-8000.proxy.runpod.net/health` until `ready`,
submit batches, and **`DELETE /v1/pods/{podId}`** when done — terminate,
never *stop*: a stopped pod keeps its disk and keeps billing.

Serverless is unaffected and remains the fallback when pod provisioning
fails: the image `CMD` is still `handler.py`, `handler.py` and
`parakeet_engine.py` are untouched, `server.py` is imported only when a
pod's `dockerStartCmd` selects it, and the RunPod GitHub integration keeps
building the serverless image as before.

### Pod-mode env

| Env | Default | Notes |
| --- | --- | --- |
| `POD_AUTH_TOKEN` | unset | **Set this.** When set, `/run`, `/status`, `/requests` and `DELETE` require `Authorization: Bearer <token>`; `/health` stays open so a readiness poller needs no secret. The pod proxy URL is public and unauthenticated, and `/run` makes the pod fetch caller-supplied URLs on a paid GPU. |
| `MAX_RETAINED_RESULTS` | 32 | Hard ceiling on retained terminal outputs, oldest evicted first. Word timestamps for one 450-file batch are tens of MB of Python objects, so age alone is not a bound. |
| `RESULT_TTL_S` | 21600 | Age limit for the same. |
| `MAX_BODY_BYTES` | 8388608 | Request-body cap; a 500-file job body is ~150KB. |
| `PORT` | 8000 | Must match the pod's exposed `ports` entry. |

### Startup time

`/health` reports the boot breakdown so provisioning can be budgeted from
measurement rather than guesswork:

```json
{"ready": true, "boot": {"engine_load_s": 41.3, "ready_s": 43.0}}
```

`ready_s` is measured from process start, so **(pod time-to-ready) −
`ready_s` = RunPod scheduling + image pull**.

**A pod is billed from GPU allocation, so the image pull and the model load
are billed idle GPU time.** The quantity to minimize is therefore total
*pod-seconds*, which has a consequence worth stating plainly: creating the
pod earlier (so boot overlaps the caller's prep work) improves latency but
**does not reduce cost** — it starts the meter sooner and bills the same
boot, or more. Only three things actually reduce the bill: paying boot
fewer times, pulling fewer bytes, and loading faster.

Ranked by billed seconds saved:

1. **One pod for all of the day's batches** — already the design, and by
   far the largest effect: boot is paid once per day instead of once per
   batch. Everything below is an optimization of that single boot.
2. **Fewer bytes to pull.** Pull scales with image size, and for a ~12GB
   image it should dominate the rest combined. Landed so far: packaged test
   suites pruned in the install layer (deleting in a later layer would
   shrink nothing). The build now logs `pip freeze` and the 20 largest
   `site-packages` entries, and the job's last step prints per-layer
   compressed sizes — read those before cutting anything else.
   `nemo_toolkit[asr]` carries a training-oriented tree, so there is likely
   more to remove, but each cut needs a real transcription run to trust:
   the build only proves that imports and the model load survive.
3. **zstd layer compression** — decompresses several times faster than
   gzip, and a large pull is often decompression-bound rather than
   network-bound. Published under `zstd-latest` / `zstd-sha-<sha>` tags,
   with the default tags left on gzip until a pod confirms its host Docker
   accepts zstd layers (an older daemon rejects the media type). Compare
   the two by booting one pod on each tag and reading `boot.ready_s`
   against total time-to-ready.
4. **Faster load.** `HF_HUB_OFFLINE=1` is baked, so boot no longer
   round-trips to huggingface.co for a revision check (a build-time
   assertion proves the weights still resolve offline), and all bytecode is
   precompiled so no boot writes `.pyc`.
5. **Split the single large pip layer** so layers download in parallel —
   Docker pulls 3 concurrently, so one ~5GB layer is a single-stream
   straggler. Not landed: splitting a `pip install` across RUNs can resolve
   different versions, so it needs the pinned `pip freeze` (now in the build
   log) plus evidence from the per-layer report that one layer really is the
   straggler. Do this only if the report shows it.

Note that pre-warming the model or CUDA graphs before answering `ready` is
*not* on this list: it moves the same seconds earlier in a pod that is
already billing, so it changes latency and not cost.

Engine-side options (would require changing `parakeet_engine.py`, which is
frozen — listed for completeness): load weights directly to GPU via
`map_location`, and pre-extract the `.nemo` archive at build time so boot
skips a ~2.5GB tar extraction on every start.

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

- English-only (route non-English or code-switched audio to a Whisper
  fallback endpoint).
- No `initial_prompt` equivalent (transducer): keep the proper-noun
  find/replace post-pass for names and domain terms.
- Files > 3h rejected by guard — split upstream.
- Speaker diarization not included; upstream per-speaker chunking + the
  metadata passthrough covers turn attribution.
