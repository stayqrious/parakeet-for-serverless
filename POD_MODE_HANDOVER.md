# Pod mode handover

Everything a caller needs to run batches against a RunPod pod instead of the
serverless endpoint. Written for the daily job in
`process-transcript-feedback-sprints`.

Status: **verified end to end on real pods on 2026-07-26.** All 20 functional
checks passed on two separate pods, one per image variant.

---

## 1. The image

Pin one of these. Both were built from commit `c12aac2` by
`.github/workflows/build-pod-image.yml` and both were verified on a live pod.

| Tag | Compressed | Verified on | Use |
| --- | --- | --- | --- |
| `ghcr.io/stayqrious/parakeet-for-serverless:sha-c12aac21a1decaa24455f9ffdf20b9a71f91d2a6` | 6.19 GB | RTX 4090 (Secure) | **Recommended for the unattended daily job** — gzip layers, pull works on any Docker version |
| `ghcr.io/stayqrious/parakeet-for-serverless:zstd-sha-c12aac21a1decaa24455f9ffdf20b9a71f91d2a6` | 5.06 GB | RTX 4090 (Secure) | 18% fewer bytes; zstd layers |

**Recommendation: pin the gzip `sha-` tag.** The zstd variant is verified on a
Secure 4090, but Community hosts could not be tested (no capacity at test
time) and an older Docker daemon rejects zstd layer media types. If a pod ever
fails to pull with a media-type error, that is why. Try zstd once against your
real provisioning path; switch only if it pulls cleanly.

Do **not** use `:latest` in production — it moves on every push to `main`.

The GHCR package is **public**, so no registry credentials are needed. If it
is ever set back to private, pods fail to pull with
`Failed to get Hub registry auth`; the fix is a classic PAT with
`read:packages` registered via `POST https://rest.runpod.io/v1/containerregistryauth`
(`{name, username, password}` → returns an id) and passed to pod creation as
`containerRegistryAuthId`.

Nothing is installed or downloaded at pod start: `pip install`, the 2.5 GB
model bake, and bytecode precompilation are all build-time layers, and
`HF_HUB_OFFLINE=1` prevents even a metadata call to huggingface.co. A pod does
only *pull → start process → load weights*.

---

## 2. Creating the pod

Use the **v1** REST API. v2 lacks `allowedCudaVersions`, which this image
needs (torch 2.6+cu124). Every field below was validated against the live
OpenAPI schema.

```jsonc
POST https://rest.runpod.io/v1/pods
Authorization: Bearer <RUNPOD_API_KEY>
{
  "name": "parakeet-batch",
  "imageName": "ghcr.io/stayqrious/parakeet-for-serverless:sha-c12aac21a1decaa24455f9ffdf20b9a71f91d2a6",
  "cloudType": "COMMUNITY",
  "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                 "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4500"],
  "gpuCount": 1,
  "containerDiskInGb": 30,
  "volumeInGb": 0,
  "ports": ["8000/http"],
  "allowedCudaVersions": ["12.4","12.5","12.6","12.7","12.8","12.9","13.0"],
  "dockerStartCmd": ["python", "-u", "server.py"],
  "env": {"POD_AUTH_TOKEN": "<generate a random token per run>"}
}
```

Defaults that will hurt if omitted (confirmed from the live schema):

- **`volumeInGb` defaults to 20** — must be explicitly `0` or you are billed
  for a persistent volume you never use.
- `containerDiskInGb` defaults to 50; 30 is enough (the long lane holds
  ~10 GB of WAV per 450-file batch).
- `cloudType` defaults to `SECURE`, `ports` defaults to `8888/http,22/tcp`.
- `dockerStartCmd` is what selects pod mode. Without it the image runs its
  `CMD` (`handler.py`, the serverless worker) and nothing will answer HTTP.

### Capacity: you need a fallback ladder

On 2026-07-26 **no 24 GB-class Community GPU was available.** Eleven GPU types
across two `COMMUNITY` attempts all returned HTTP 500 *"This machine does not
have the resources to deploy your pod"*. Every test pod ended up on Secure.

A single-GPU-type request will intermittently fail to provision. Use a
widening ladder and stop at the first success:

```python
ATTEMPTS = [
    ("COMMUNITY", ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                   "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4090",
                   "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation"]),
    ("COMMUNITY", ["NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S",
                   "NVIDIA RTX 5000 Ada Generation", "NVIDIA GeForce RTX 5090"]),
    ("SECURE",    ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090",
                   "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S"]),
]
```

Note the 48 GB cards (A6000/L40S/A40) are safe with the **default** lane
profile only. Do not set `SHORT_BS`/`MEDIUM_BS`/`LONG_BS` env vars — the tuned
profile peaks ~29 GB and will OOM a 24 GB card.

---

## 3. Waiting for readiness — read this before writing the poll loop

**RunPod's edge proxy answers before your container exists.** A GET to
`https://{podId}-8000.proxy.runpod.net/` returns **404 in ~0.3s** for a pod ID
that never existed, for a terminated pod, and for a pod still pulling its
image. Any readiness check that accepts "an HTTP response arrived", or treats
`status_code < 500` as success, will report ready almost instantly and be
wrong. (This bit the first version of our own probe.)

Poll for the actual payload:

```python
while time.time() - t0 < 600:
    try:
        r = session.get(f"{base}/health", timeout=15)
        if r.status_code == 200 and r.json().get("ready") is True:
            break
        h = r.json() if r.status_code == 200 else {}
        if h.get("error"):                     # engine failed to load
            raise RuntimeError(h["error"])     # do not keep polling
    except Exception:
        pass                                   # proxy 404 / conn reset while pulling
    time.sleep(5)
```

`/health` needs no auth token, so a readiness poller needs no secret. It
returns:

```json
{"ready": true,
 "model": "nvidia/parakeet-tdt-0.6b-v2",
 "jobs": {"inQueue": 0, "inProgress": 0, "completed": 0, "failed": 0},
 "workers": {"ready": 1, "running": 0, "alive": true},
 "boot": {"engine_load_s": 20.0, "ready_s": 20.0}}
```

- `workers.alive` false means the single worker thread died — jobs will sit in
  `IN_QUEUE` forever. Treat it as fatal: terminate and re-provision.
- `error` present means the model never reached VRAM. Fatal, do not poll on.
- `boot.ready_s` is measured from process start, so
  `(your wall-clock time to ready) − boot.ready_s` = RunPod scheduling + image
  pull. Log it; it is the only way to see whether a slow start was infra or us.

---

## 4. Submitting work

Identical to the serverless API except for the base URL and the auth header.
Set `POD_AUTH_TOKEN` at pod creation and send it on every call except
`/health` — the pod proxy URL is public and unauthenticated, and `/run` makes
the pod fetch caller-supplied URLs on a paid GPU.

```
Authorization: Bearer <POD_AUTH_TOKEN>
```

| Method | Path | Returns |
| --- | --- | --- |
| `POST` | `/run` | `{"id", "status": "IN_QUEUE"}` — returns immediately |
| `GET` | `/status/<id>` | `{"id","status","output"?}`; **404** if unknown |
| `GET` | `/requests` | `{"requests": [{"id","status"}, ...]}` |
| `GET` | `/health` | as above (no auth) |
| `DELETE` | `/status/<id>` | `{"deleted": true}` — release a retained result |

`POST /run` body is byte-identical to the serverless input:

```json
{"input": {"audios": [{"audio_url": "https://...", "chunk_id": "395366",
                       "session_id": "TST...", "speaker": "..."}],
           "timestamps": true}}
```

`status` ∈ `IN_QUEUE` | `IN_PROGRESS` | `COMPLETED` | `FAILED`. `output`
appears only on a terminal status and is shape-identical to what `handler.py`
returns, so `apply_job_output()` needs no changes.

Behaviour the caller must handle:

- **One job at a time.** One GPU means one worker thread draining a FIFO
  queue (`ParakeetEngine` is unsafe for concurrent `transcribe()` — TDT CUDA
  graphs, and the full↔local attention switch mutates the model globally).
  Submitting 5 batches at once is fine; they run sequentially. Verified live:
  job 2 reported `IN_QUEUE` while job 1 was `IN_PROGRESS`.
- **404 on `/status/<id>` means gone, unrecoverable** — never a transient.
- **Results are retained, then evicted.** `MAX_RETAINED_RESULTS=32` (oldest
  first) and `RESULT_TTL_S=21600`. `DELETE /status/<id>` after you have stored
  a result; do not rely on the TTL if you run many batches.
- **`503` on `/run`** means the engine is still loading. Poll `/health` first.
- **`400`** for: empty/missing `input.audios`, no valid `audio_url` entries,
  more than `MAX_FILES_PER_JOB` (500) files, malformed JSON.
- **`413`** for a body over `MAX_BODY_BYTES` (8 MB) or chunked encoding. Send
  `Content-Length` (any normal JSON client does). A 500-file body is ~150 KB.
- Keep 200–500 files per job, as with serverless.

### Finishing

```
DELETE https://rest.runpod.io/v1/pods/{podId}
```

**Terminate, never `stop`.** A stopped pod keeps its disk and keeps billing.
Put the delete in a `finally` block and verify it: re-`GET` the pod and expect
404. Our verification script does both, plus an account-wide sweep for strays.

---

## 5. Measured numbers

Two pods, RTX 4090 Secure, image `c12aac2`, both terminated after.

| Phase | gzip 6.19 GB | zstd 5.06 GB |
| --- | --- | --- |
| create → `ready: true` | 210.9 s | 39.3 s |
| ├─ scheduling + image pull (derived) | 190.9 s | 18.8 s |
| └─ process start → ready (`boot.ready_s`) | 20.0 s | 20.5 s |
| &nbsp;&nbsp;&nbsp;&nbsp;of which engine load | 20.0 s | 20.4 s |

**The 10x pull difference is not a zstd effect.** 18% fewer bytes cannot
explain it: the two pods pulled at 32 MB/s and 269 MB/s on different hosts.
Pull time is dominated by host and network luck. Budget for the slow case
(~3.5 min) and log `boot.ready_s` so you can tell infra from application.

Image composition (10 layers, compressed): pytorch base 3.28 GB (53%), baked
model 2.30 GB (37%), `nemo_toolkit[asr]` + deps 0.43 GB (7%), rest 0.19 GB.

Load time is ~20 s and cannot be baked away. A container image is a
filesystem, not a memory snapshot: the weights are on disk in the image, but
every process start must still `import torch`/`import nemo`, read the 2.5 GB
`.nemo`, deserialize it, and copy it into VRAM.

---

## 6. Not verified — do not assume

- **Throughput on a pod.** The functional test used five 10-second clips, so
  its `rtfx_end_to_end=13.9` is fixed overhead, not throughput, and it says
  nothing about the 306x figure. The 306x/293x numbers still come only from
  the original A40 serverless validation. **A realistic multi-lane batch
  (short/medium/long, including >300 s files that trigger the local-attention
  switch) has not been run on a pod.** That switch has historically been a
  crash source, so validate it before trusting a production batch.
- **zstd on Community hosts** — untested, no capacity available.
- **Community provisioning** — never succeeded during testing.
- **A 450-file batch on a pod**, and the 30 GB container disk under a real
  long-lane load.

## 7. Traps already hit, so you don't repeat them

- The RunPod proxy 404 issue in §3 — the single most likely thing to make a
  caller think a pod is ready when it is not.
- `dockerStartCmd` with a multi-line `python -c "<script>"` silently fails;
  the pod runs, never serves, and you pay while polling. Pass a file path
  (`["python","-u","server.py"]`) or base64 the script into one argv element.
- Pushing to `main` **does** trigger the image build, but the run appears in
  the Actions API ~40 s later. Do not conclude it did not fire and dispatch a
  duplicate 6 GB build.
- GHCR packages default to private even on a public repo, and org policy may
  disable the Public option until an owner enables it org-wide.

## 8. Repo facts

- `server.py` is pod mode; `handler.py` is unchanged and still the image
  `CMD`, so the serverless endpoint remains a working fallback.
- `parakeet_engine.py` and `handler.py` were not modified by this work.
- Env knobs: `POD_AUTH_TOKEN`, `MAX_RETAINED_RESULTS` (32), `RESULT_TTL_S`
  (21600), `MAX_BODY_BYTES` (8388608), `PORT` (8000), plus the existing engine
  vars. Leave the lane vars alone.
- Verification scripts used for the numbers above are not in the repo (they
  were session scratch); §2–§4 contain everything needed to rebuild them.
