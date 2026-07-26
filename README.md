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
latency-sensitive traffic, poor value for a once-daily batch. Serverless day cost is modest; the structural problem is not the daily total
but that ~60% of every dollar is standby, not compute. Pod mode pays only
for GPU actually used, so it removes that 60% at the cost of one boot per
day.

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

A pod is billed from GPU allocation, so pull and model load are billed idle
GPU. Note that creating the pod earlier so boot overlaps the caller's prep
work improves *latency* but **not cost** — it starts the meter sooner.
Likewise, pre-warming the model before answering `ready` moves the same
seconds around inside an already-billing pod.

### Measured on real pods (2026-07-26, RTX 4090 Secure, `sha-c12aac2`)

Two pods, one per image variant. Both passed all 20 functional checks and
were terminated.

| Phase | gzip (6.19GB) | zstd (5.06GB) |
| --- | --- | --- |
| pod create → `ready: true` | 210.9s | 39.3s |
| ├─ RunPod scheduling + image pull | 190.9s | 18.8s |
| └─ process start → ready (`boot.ready_s`) | 20.0s | 20.5s |
| &nbsp;&nbsp;&nbsp;&nbsp;of which engine load | 20.0s | 20.4s |

**Do not read that as a zstd win.** 18% fewer bytes cannot make a pull 10x
faster; the two pods landed on different hosts and pulled at 32 MB/s and
269 MB/s respectively. Pull time is dominated by host and network variance,
which swamps anything the image can do about it — attributing a difference
to compression would need repeated runs of both tags. What these runs *do*
establish: zstd layers pull correctly on RunPod (the reason the variant was
published behind separate tags), and engine load is a stable ~20s.

Image is **6.19GB compressed** (the ~12GB figure elsewhere is
*uncompressed*), across 10 layers:

| Layer | Compressed | Share |
| --- | --- | --- |
| pytorch base (torch + CUDA + cuDNN) | 3.28GB | 53% |
| baked model | 2.30GB | 37% |
| `nemo_toolkit[asr]` + deps | 0.43GB | 7% |
| ffmpeg, `server.py`, metadata | 0.19GB | 3% |

**Boot is a large fraction of pod time, but a small number of cents.** Both
things are true and they pull in opposite directions, so be careful which
one you are optimizing.

A small daily batch is a few audio-hours. At ~306x that is
**~3.6 min of GPU for a normal day**, so a boot of 0.7–3.5 min is **16–49%
of pod time**, not the rounding error a backlog run suggests.

In money, though, it is still tiny: 3.6 min of work plus boot is ~4–7 min of
pod time, i.e. **a fraction of a cent per day.05–0.08
on a Secure 4090. Halving the pull saves well under a cent a day. And since
pull time swung 10x on host luck between two identical runs, image-side
tuning cannot even be measured reliably at this volume.

So: worth removing boot *repetition* (below), not worth further engineering
on boot *duration*.

What is worth doing, in order:

1. **One pod for all of the day's batches** — already the design, and the
   only startup lever that matters. On a normal day boot is 16–49% of pod
   time, so paying it once instead of once per batch is the whole game. If
   the caller ever provisions per batch, boot would dominate the bill.
2. **Use the zstd tags.** Level-10 zstd is **5.06GB vs 6.19GB gzip, 18%
   fewer bytes**, because it compresses the torch/CUDA base far better
   (3.28 → 2.30GB); the model layer barely moves, as compressed weights do
   not recompress. Verified to pull on RunPod. Strictly fewer bytes at zero
   runtime cost, so take it — but expect the benefit to be invisible next to
   host variance, and do not spend time tuning it further.
3. **Faster load** (landed, and already cheap at 20s): `HF_HUB_OFFLINE=1`
   so boot never round-trips to huggingface.co, guarded by a build-time
   assertion that the weights still resolve offline, plus precompiled
   bytecode so no boot writes `.pyc`.

Deliberately **not** done, with reasons:

- **Splitting the pip layer for parallel download.** The premise was wrong:
  that layer is 0.43GB (7%), not the ~5GB straggler assumed. The two layers
  that matter are already separate and Docker pulls 3 concurrently, so the
  parallelism is already there. No gain available.
- **Further `site-packages` pruning.** It can only touch the 7% layer. The
  test-suite prune that is landed is worth ~1% of the pull.
- **fp16/bf16 model weights** — the largest remaining lever by far (~18% of
  the pull, since the `.nemo` holds fp32 weights: 0.6B × 4 bytes ≈ 2.4GB).
  Needs `parakeet_engine.py` to `restore_from` a converted local archive,
  which is frozen, and a WER check against the validated baseline. Not worth
  it for a sub-cent daily saving.
- **A slimmer base image.** 53% of the pull, but it is the validated
  torch 2.6+cu124 environment, and `nvidia/cuda` + pip torch is likely a
  wash since cu124 wheels bundle their own CUDA libs.

### Capacity, which matters more than boot time

At test time **no 24GB-class community GPU was available** — two
`COMMUNITY` attempts across 11 GPU types both returned HTTP 500 *"does not
have the resources to deploy your pod"*, and the pod only scheduled on
`SECURE` (RTX 4090, $0.69/hr). A daily job that requests one GPU type on
community will simply fail to provision. Give the caller a fallback ladder:
several community GPU types first, then Secure Cloud. Secure pricing still
lands far under the serverless bill for the same work, so falling back is
much better than not running.

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
