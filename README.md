# llamacpp-infer

A self-contained [llama.cpp](https://github.com/ggml-org/llama.cpp) inference server, run as a Docker Compose service.

Point it at any GGUF model on Hugging Face via a `.env` file — it **downloads the model on first run**, **caches it** in a persistent volume (so it's never re-downloaded), and serves an **OpenAI-compatible API** on GPU.

---

## Requirements

- Docker + Docker Compose v2
- NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed (so `docker run --gpus all` works)

## Quick start

```bash
# 1. Create your .env from the template, then edit it
cp .env.example .env

# 2. Start the service
docker compose up -d

# 3. Watch it download + load the model on first run
docker compose logs -f

# 4. Once healthy, it serves on http://localhost:8080
curl http://localhost:8080/health          # -> {"status":"ok"}
```

Stop it with `docker compose down` — the downloaded models stay in the cache volume.

## Configuration (`.env`)

All configuration is done through environment variables in `.env`:

| Variable     | Meaning                                                                 | Example                          |
|--------------|-------------------------------------------------------------------------|----------------------------------|
| `HF_REPO`    | Hugging Face repo id (do **not** append a quant tag)                    | `unsloth/gemma-4-E4B-it-GGUF`    |
| `HF_FILE`    | Exact `.gguf` filename inside that repo                                 | `gemma-4-E4B-it-UD-IQ2_M.gguf`   |
| `HF_TOKEN`   | Hugging Face token — only for private/gated repos (optional)           | `hf_xxx`                         |
| `PORT`       | Host port to expose the server on                                       | `8080`                           |
| `PARALLEL`   | Number of concurrent request slots                                      | `4`                              |
| `CTX_SIZE`   | Total context window, **shared across all slots** (`0` = model max)     | `4096`                           |
| `NGL`        | Layers offloaded to GPU (`99` = all)                                    | `99`                             |
| `FLASH_ATTN` | Flash attention: `on` / `off` / `auto` (reduces KV-cache memory)        | `on`                             |
| `EXTRA_ARGS` | Extra `llama-server` CLI flags (see below)                              | `--no-mmproj`                    |

> **Changing a model:** edit `HF_REPO` / `HF_FILE`, then `docker compose up -d --force-recreate`. Each model is cached independently, so switching back is instant.

### `HF_FILE` must match exactly

`HF_FILE` has to be a real file in `HF_REPO`, character-for-character, or the download 404s. Check the repo's "Files" tab on Hugging Face.

## How the pieces work

### Model download & persistent cache
The server pulls the model from Hugging Face using `LLAMA_ARG_HF_REPO` + `LLAMA_ARG_HF_FILE`. Downloads land in `/models` inside the container (`LLAMA_CACHE`), which is backed by the named Docker volume **`llama-models`**. Because the volume persists across `docker compose down`/`up`, a model is downloaded only once.

```bash
docker volume inspect llamacpp-infer_llama-models   # see where the cache lives
docker volume rm llamacpp-infer_llama-models        # wipe all downloaded models
```

### GPU
The `deploy.resources.reservations.devices` block is the Compose equivalent of `docker run --gpus all`. Confirm the model is actually on the GPU:

```bash
nvidia-smi                                          # llama-server should hold VRAM
docker logs llama-server 2>&1 | grep -i offload     # "offloaded N/N layers to GPU"
```

### `EXTRA_ARGS` (e.g. disabling multimodal)
Some `llama-server` flags are presence-only (no value), so they can't be a simple env boolean. `EXTRA_ARGS` is appended verbatim to the server command:

- `EXTRA_ARGS=--no-mmproj` — **text-only**, skips the multimodal projector and saves ~1.2 GB VRAM.
- `EXTRA_ARGS=` (empty) — full **multimodal** (image/audio) support, if the model has it.

You can pass multiple space-separated flags here too (e.g. `--no-mmproj --mlock`).

## Using the API

The server is OpenAI-compatible. Base URL: **`http://localhost:8080/v1`**

### Basic chat
```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi in 3 words"}],"max_tokens":200}'
```

### Streaming (token by token)
```bash
curl -N http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":200,"stream":true}'
```

Clean text-only stream (strip the SSE/JSON envelope with `jq`):
```bash
curl -sN http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":200,"stream":true}' \
  | grep '^data: ' | sed 's/^data: //' | grep -v '^\[DONE\]' \
  | jq -j '.choices[0].delta.content // empty'
```

### Reasoning ("thinking") models
Some models (e.g. Gemma 3n) emit a private reasoning phase before the answer. In responses the thinking appears in `reasoning_content` and the final answer in `content` (streaming: `delta.reasoning_content` then `delta.content`). To skip thinking for faster replies:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

## Tuning for limited VRAM

The model weights + KV cache + compute buffers must all fit in VRAM. If the container crashes with a **CUDA out-of-memory** error, reduce footprint in `.env` (cheapest wins first):

1. **`EXTRA_ARGS=--no-mmproj`** — frees ~1.2 GB if you don't need image/audio.
2. **Lower `CTX_SIZE`** — the KV cache is sized for the *total* context. `4096` instead of `16384` cuts it ~4×. This is the main lever.
3. **Lower `PARALLEL`** — fewer concurrent slots. (Note: KV cache tracks total `CTX_SIZE`, so lowering `PARALLEL` alone doesn't shrink it unless you also lower `CTX_SIZE`.)
4. **Lower `NGL`** — offload fewer layers to GPU; the rest run on CPU (slower, but fits).

Keep `FLASH_ATTN=on` — it meaningfully reduces KV-cache memory.

## Common commands

```bash
docker compose up -d                     # start / apply .env changes
docker compose up -d --force-recreate    # force reload after env changes
docker compose logs -f                   # follow logs
docker compose ps                        # status (look for "healthy")
docker compose down                      # stop (keeps the model cache)
docker compose restart                   # restart the container
```
