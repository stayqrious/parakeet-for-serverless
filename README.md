# Parakeet TDT 0.6B v2 — batch speech-to-text worker

A GPU container that transcribes **large batches of audio URLs** in one
request, built around NVIDIA's [`parakeet-tdt-0.6b-v2`][model] English ASR
model. Point it at a few hundred audio files, get back text with word- and
segment-level timestamps.

It is optimised for **cost per audio-hour on batch workloads**, not for
single-clip latency. The model is baked into the image, so a container
starts transcribing about 20 seconds after the process starts, with no
model download and no external dependency at boot.

[model]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2

## Two ways to run it

| Mode | Entry point | Where it runs |
| --- | --- | --- |
| **HTTP server** | `server.py` | **Any** machine with an NVIDIA GPU and Docker |
| **RunPod Serverless** | `handler.py` | RunPod Serverless only |

`server.py` is a stdlib-only `ThreadingHTTPServer` with no cloud SDK and no
provider API calls — it binds a port and serves HTTP. That is the portable
surface, and it is what you want anywhere other than RunPod Serverless.

`handler.py` is a thin adapter for RunPod's Serverless runtime (it imports
the `runpod` SDK). The two speak the **same wire format**, so a client moves
between them by changing only the base URL.

> **Provenance, stated honestly:** this was built and validated on RunPod —
> both Serverless and on-demand GPU pods — and the deployment numbers below
> were measured there. Nothing in `server.py` or the engine is RunPod-specific,
> so it should run on any GPU host that can run the container, but other
> providers have not been tested. If you run it elsewhere, the
> [generic deployment](#deploying-on-any-gpu-host) section is the path.

## Measured performance

Validated on an A40 (48GB) across **~625 audio-hours of real-world
conversational speech**, with word and segment timestamps enabled throughout.

| Metric | Value |
| --- | --- |
| RTFx, compute only | **306x** |
| RTFx, end to end (including downloads) | **293x** |
| 334 audio-hours / 8,047 files | 68.4 min wall |
| GPU time per 500 audio-hours | ~1.7 GPU-hours |
| Per-file failure rate on real-world sources | 0.3–0.6%, isolated per file |

RTFx is audio-seconds processed per wall-second: 306x means one hour of audio
transcribes in about twelve seconds of GPU time.

**Scope of that measurement.** These figures come from the A40 batch
validation. Long-running HTTP-server mode has been verified functionally but
not separately benchmarked at that scale, so treat the throughput numbers as a
property of the engine rather than a guarantee for a given deployment. If
you depend on them, re-measure with a realistic mix of durations — in
particular files over 300s, which trigger the local-attention switch.

## Quick start

Requirements: an NVIDIA GPU with **≥ 24GB VRAM** (peak usage is ~18GB on the
default profile) and a **CUDA 12.4 or newer** driver — the image ships
torch 2.6+cu124 and will fail to initialise on older hosts.

```bash
docker build -t parakeet-batch .

# HTTP server mode. --gpus all requires the NVIDIA container toolkit.
docker run --rm --gpus all -p 8000:8000 \
  -e POD_AUTH_TOKEN="$(openssl rand -hex 32)" \
  parakeet-batch python -u server.py
```

The HTTP socket opens *before* the model loads, so you can poll immediately:

```bash
curl -s localhost:8000/health          # {"ready": false, ...} while loading
```

Once `"ready": true`, submit a batch:

```bash
curl -s -X POST localhost:8000/run \
  -H "Authorization: Bearer $POD_AUTH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"input": {"audios": ["https://example.com/a.wav"], "timestamps": true}}'
# -> {"id": "...", "status": "IN_QUEUE"}

curl -s localhost:8000/status/<id> -H "Authorization: Bearer $POD_AUTH_TOKEN"
```

`test_input.json` holds a minimal single-file payload for a smoke test.

## API

Both modes accept the same `input` object.

```jsonc
{
  "input": {
    "audios": [
      // Objects: audio_url plus ANY metadata keys you want echoed back
      {"audio_url": "https://...", "id": "0001", "speaker": "Speaker A"},
      // Or a plain URL string
      "https://example.com/clip.wav"
    ],
    "timestamps": true    // word + segment timestamps (default true)
  }
}
```

- **200–500 files per request** is the sweet spot: it amortises model and
  batching overhead, and bounds the blast radius of a failure. Hard cap:
  `MAX_FILES_PER_JOB` (default 500).
- **Sources:** any HTTP(S) URL — presigned S3/R2 links, CDN URLs, and file
  services that require following redirects or a browser `User-Agent` (the
  downloader sends one and follows redirects).
- **Formats:** webm, wav, mp3, m4a, flac, ogg, opus — ffmpeg converts
  internally.
- **Metadata passthrough:** every key you set on an input object is copied
  onto its result, so you can correlate without relying on ordering.
- **Per-file errors are isolated.** A corrupt file returns an `"error"`
  string carrying ffmpeg's stderr; the rest of the batch still completes.

Per-file result:

```json
{"id": "0001", "speaker": "Speaker A", "duration_sec": 187.4,
 "text": "...",
 "segments": [{"start": 0.0, "end": 4.2, "text": "..."}],
 "words":    [{"start": 0.0, "end": 0.6, "text": "..."}]}
```

The response also carries `counts`, `model`, `total_audio_sec`,
`wall_time_sec`, `gpu_time_sec`, `rtfx_end_to_end` and `rtfx_compute_only`.

### HTTP server endpoints (`server.py`)

| Method | Path | Returns |
| --- | --- | --- |
| `POST` | `/run` | `{"id", "status": "IN_QUEUE"}` — returns immediately |
| `GET` | `/status/<id>` | `{"id", "status", "output"?}`; status ∈ `IN_QUEUE`\|`IN_PROGRESS`\|`COMPLETED`\|`FAILED`; `output` only when terminal; **404** if unknown |
| `GET` | `/requests` | `{"requests": [{"id", "status"}, ...]}` |
| `GET` | `/health` | `{"ready", "jobs", "workers", "boot", ...}` (plus `error` if the engine failed to load) |
| `DELETE` | `/status/<id>` | `{"deleted": true}` — release a retained result |

**One GPU means one job at a time.** A single worker thread drains a FIFO
queue, because the engine is not safe for concurrent `transcribe()` calls —
TDT CUDA graphs and the full↔local attention switch both mutate the model
globally. Submitting more jobs is fine; they queue.

`/run` returns **503** until the model is in VRAM. `/health` is always open
so a readiness poller needs no credentials; everything else requires the
bearer token when `POD_AUTH_TOKEN` is set.

**Readiness means `/health` returning 200 with `"ready": true` in the body**
— nothing weaker. If you put the container behind a proxy or ingress, that
edge may answer before the container exists, so a check that only asks "did
a response arrive?" will report ready almost immediately and be wrong.

## Deploying on any GPU host

The image is self-contained: model weights are baked in, so there is no
external storage to attach and nothing to fetch on first request.

1. Build and push the image to a registry your host can pull from, or use
   the GitHub Actions workflow in `.github/workflows/build-pod-image.yml`,
   which publishes to GHCR as `ghcr.io/<owner>/<repo>` on every push to
   `main`, on release, and on demand.
2. Run the container with **the command overridden to `python -u server.py`**.
   The image `CMD` is `handler.py` (the RunPod Serverless adapter), which is
   not what you want off RunPod.
3. Expose `PORT` (default 8000).
4. **Set `POD_AUTH_TOKEN`** if the port is reachable by anyone but you.
   `/run` makes the container fetch caller-supplied URLs on a GPU you are
   paying for.
5. Poll `/health` until `"ready": true`, submit batches, and shut the
   container down when the batch is done — it holds no state worth keeping.

Anything that runs an OCI image on an NVIDIA GPU works: a bare VM with the
container toolkit, Kubernetes with a GPU node pool, or another provider's
on-demand GPU product. Only the RunPod-specific details in the next section
change.

## Deploying on RunPod

This is the reference deployment, and the one the numbers above came from.

### Serverless

1. Push this repository to GitHub.
2. RunPod console → Settings → Connections → GitHub → authorise the repo.
3. Serverless → **New Endpoint** → GitHub Repo → pick the repo and branch.
   RunPod builds the Dockerfile on their infrastructure (~20–30 min; the
   baked model makes the image large on purpose, so workers cold-start with
   weights already on disk).
4. Endpoint settings:
   - **GPU:** 48GB class (A40 / A6000 / L40S) or any ≥24GB card.
   - **CUDA filter:** ≥ 12.4.
   - **Workers:** min 1 during a processing window to avoid cold starts;
     max sets your parallelism.
   - **Execution timeout:** 1800s comfortably covers a 400-file job.
5. Tagging a new release rebuilds and redeploys; roll back from the
   endpoint's Builds tab.

Call it at `POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run` with the `input`
body above. A top-level `"webhook"` URL receives the full response on
completion.

### On-demand pods

Serverless bills a standby fee on top of GPU time, which is good value for
latency-sensitive traffic and poor value for a batch job that runs once a
day. An on-demand pod pays only for the GPU it uses, at the cost of one boot
per run — so for batch work, start **one** pod, run every batch through it,
and terminate it.

Pods cannot pull from `registry.runpod.net` (that registry is bound to the
Serverless build system and returns *"Failed to get Hub registry auth"*),
which is why the GHCR workflow exists. After the first successful build,
make the package public: repo → Packages → Package settings → **Change
visibility → Public**. GHCR packages default to private even on a public
repo, and a private package fails the pull the same way. Alternatively
register a PAT with `read:packages` through RunPod's container-registry-auth
API and pass `containerRegistryAuthId`.

Create the pod with the **v1** API — v2 lacks `allowedCudaVersions`, which
this image needs:

```jsonc
POST https://rest.runpod.io/v1/pods
{
  "name": "parakeet-batch",
  "imageName": "ghcr.io/<owner>/<repo>:sha-<commit>",   // pin an immutable tag
  "cloudType": "COMMUNITY",
  "gpuTypeIds": ["NVIDIA GeForce RTX 3090"],
  "gpuCount": 1,
  "containerDiskInGb": 30,
  "volumeInGb": 0,           // the image is self-contained; a volume only adds cost
  "ports": ["8000/http"],
  "allowedCudaVersions": ["12.4","12.5","12.6","12.7","12.8","12.9","13.0"],
  "dockerStartCmd": ["python", "-u", "server.py"]
}
```

Then poll `https://{podId}-8000.proxy.runpod.net/health` until ready, submit
batches, and **`DELETE /v1/pods/{podId}`** when done. Terminate, never
*stop*: a stopped pod keeps its disk and keeps billing for it.

Two things worth knowing before you write the provisioning loop:

- **`gpuTypeIds` does not fall through.** It accepts a list and reads like a
  priority list, but a request naming several types can be refused while a
  single-type request for one of those same types succeeds at the same
  moment. Ask for one type per request and walk your own preference list.
- **Cheap capacity is scarce and bursty.** Community GPUs frequently report
  no availability, and RunPod returns that as an HTTP **500** — otherwise the
  most retryable status there is — so match on the message rather than
  retrying blindly. Plan a fallback ladder across several GPU types, then
  Secure Cloud, rather than a single type.
- **`dockerStartCmd` must be a file path in argv form.** A multi-line
  `python -c "<script>"` fails silently: the pod starts, never serves, and
  bills while you poll it. Use `["python", "-u", "server.py"]`.

Note also that GHCR packages default to private even on a public repo, and an
organisation policy may hide the "Public" option until an owner enables it
org-wide — a private package fails the pod pull the same way a missing one
does.

## Configuration

All settings are environment variables.

### Server

| Env | Default | Notes |
| --- | --- | --- |
| `POD_AUTH_TOKEN` | unset | **Set this** on any reachable port. When set, `/run`, `/status`, `/requests` and `DELETE` require `Authorization: Bearer <token>`; `/health` stays open. |
| `PORT` | 8000 | Must match the port you publish. |
| `MAX_RETAINED_RESULTS` | 32 | Ceiling on retained terminal outputs, oldest evicted first. Word timestamps for one 450-file batch are tens of MB of Python objects, so age alone is not a sufficient bound. |
| `RESULT_TTL_S` | 21600 | Age limit for the same. |
| `MAX_BODY_BYTES` | 8388608 | Request-body cap; a 500-file job body is ~150KB. |

### Engine

| Env | Default | Notes |
| --- | --- | --- |
| `PARAKEET_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | Only the baked model resolves offline. |
| `MAX_FILES_PER_JOB` | 500 | Rejects oversized batches rather than running out of memory. |
| `MAX_DURATION_SEC` | 10800 | 3h guard; split longer files upstream. |
| `DOWNLOAD_WORKERS` | 10 | Downloads run concurrently with transcription. |
| `DOWNLOAD_TIMEOUT` / `DOWNLOAD_RETRIES` | — | Per-file fetch behaviour. |

### Batching lanes

Files are bucketed by duration, because attention memory grows with
`batch × duration²`. Files over 300s run under local attention, which is
roughly linear in duration instead.

| Env | Default | Applies to |
| --- | --- | --- |
| `SHORT_BS` / `SHORT_FLUSH` | 16 / 64 | files ≤ 60s |
| `MEDIUM_BS` / `MEDIUM_FLUSH` | 4 / 16 | 60–300s |
| `LONG_BS` | 2 | > 300s, local attention |

**The defaults are the validated configuration and peak at ~18GB VRAM.** A
higher-batch profile (`SHORT_BS=64`, `MEDIUM_BS=8`, `LONG_BS=4`) was trialled:
it raised GPU utilisation from 73% to ~85% and peak VRAM to 29.4GB, but showed
no clear throughput win, so it was rejected. Those values will OOM any card
under 40GB, and the engine warns if you set them on one. Do not change these
without an A/B on identical data.

## Image and startup

`/health` reports its own boot breakdown, so provisioning can be budgeted
from measurement rather than guesswork:

```json
{"ready": true, "boot": {"engine_load_s": 20.4, "ready_s": 20.5}}
```

`ready_s` is measured from process start, so **(time to ready) − `ready_s`**
is your host's scheduling and image pull.

Engine load is a stable **~20s**. The image is **6.19GB compressed** (gzip;
~12GB uncompressed) across 10 layers:

| Layer | Compressed | Share |
| --- | --- | --- |
| pytorch base (torch + CUDA + cuDNN) | 3.28GB | 53% |
| baked model | 2.30GB | 37% |
| `nemo_toolkit[asr]` + deps | 0.43GB | 7% |
| ffmpeg, application code, metadata | 0.19GB | 3% |

The workflow also publishes **zstd-compressed tags** (`zstd-latest`,
`zstd-sha-<commit>`) at 5.06GB — 18% fewer bytes, because zstd compresses the
torch/CUDA base far better; the model layer barely moves, since compressed
weights do not recompress. They are strictly fewer bytes at no runtime cost.

Expect the benefit to be invisible in practice, though: in two measured pulls
of otherwise identical images, host and network variance produced 32 MB/s and
269 MB/s. **Pull time is dominated by which host you land on**, which swamps
anything the image can do about it. Older Docker daemons can also reject zstd
media types, which is why the default tags stay gzip.

If you are optimising startup, the only lever that reliably matters is
**running many batches through one container instead of paying boot per
batch**.

## Reliability

Each of these exists because something failed without it:

- Torch pinned to 2.6+cu124, asserted at build time — NeMo's install can
  otherwise silently cascade the version.
- Per-flush GPU memory release via `del` + `gc`, deliberately **without**
  `torch.cuda.empty_cache()`, which invalidates NeMo's TDT CUDA graphs and
  causes an illegal memory access.
- Subsampling conv chunking set to auto once at init; NeMo's toggle revert
  corrupts the module on warm full↔local↔full reuse.
- OOM on a flush halves the batch and retries, down to 1.
- Downloads run concurrently with transcription, with retries and a browser
  `User-Agent`.
- WAV files deleted per flush; core dumps disabled (a crashing multi-GB
  process must not fill the container disk); per-file error isolation.
- The build asserts the baked weights resolve with `HF_HUB_OFFLINE=1`, so a
  future dependency that needs the network to locate a cached model fails the
  build instead of shipping an image that boots slowly or not at all.

## Limitations

- **English only.** Route non-English or code-switched audio to a multilingual
  model such as Whisper.
- **No `initial_prompt` equivalent.** This is a transducer model, so there is
  no prompt to bias decoding toward domain vocabulary. Proper nouns and jargon
  are best handled with a find/replace pass over the output.
- **No speaker diarization.** Chunk per speaker upstream and use the metadata
  passthrough to attribute turns.
- **Files over 3 hours are rejected** by the `MAX_DURATION_SEC` guard. Split
  them upstream.
- Timestamps are word- and segment-level; there is no phoneme alignment.

## Licence and attribution

The model is NVIDIA's [`parakeet-tdt-0.6b-v2`][model]; consult its model card
for licence terms and intended use. This repository is the packaging and batch
serving layer around it.
