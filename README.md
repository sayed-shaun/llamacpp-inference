# llamacpp-infer

A self-contained [llama.cpp](https://github.com/ggml-org/llama.cpp) inference server, run as a Docker Compose service.

Point it at any GGUF model on Hugging Face via `.env` — it downloads and caches the model on first run, then serves an OpenAI-compatible API.

| Backend    | Compose file                 | Runs on                          |
|------------|-------------------------------|-----------------------------------|
| **CUDA**   | `docker-compose.yml`          | NVIDIA GPUs                       |
| **Vulkan** | `docker-compose.vulkan.yml`   | AMD / Intel GPUs (integrated or discrete) |
| **CPU**    | `docker-compose.cpu.yml`      | Any machine, no GPU needed        |

## Requirements

- Docker + Docker Compose v2
- **CUDA:** NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Vulkan:** AMD/Intel GPU with working Mesa drivers
- **CPU:** nothing extra

## Quick start

```bash
cp .env.example .env                                  # 1. configure your model + settings
docker compose up -d                                   # 2. start (CUDA by default)
# docker compose -f docker-compose.vulkan.yml up -d    #    or: AMD / Intel GPU
# docker compose -f docker-compose.cpu.yml up -d       #    or: CPU only

docker compose logs -f                                 # 3. watch it download + load the model
curl http://localhost:8080/health                      # 4. -> {"status":"ok"}
```

Run **one backend at a time** — they share port `8080`. Append `-f <compose-file>` to any command below to target the vulkan/cpu backend. Stop with `docker compose down`; the model cache persists (it's a shared volume across all three backends, so a model is only ever downloaded once).

## Configuration (`.env`)

| Variable     | Meaning                                                       | Example                         |
|--------------|-----------------------------------------------------------------|----------------------------------|
| `HF_REPO`    | Hugging Face repo id — no quant tag                              | `unsloth/gemma-4-E4B-it-GGUF`    |
| `HF_FILE`    | Exact `.gguf` filename in that repo (case-sensitive, must match) | `gemma-4-E4B-it-UD-IQ2_M.gguf`   |
| `HF_TOKEN`   | HF token, only for private/gated repos                          | `hf_xxx`                         |
| `PORT`       | Host port                                                        | `8080`                           |
| `PARALLEL`   | Concurrent request slots                                        | `4`                               |
| `CTX_SIZE`   | Total context window, shared across slots (`0` = model max)     | `4096`                            |
| `NGL`        | Layers offloaded to GPU (`99` = all)                             | `99`                              |
| `FLASH_ATTN` | `on` / `off` / `auto` — reduces KV-cache memory                  | `on`                              |
| `EXTRA_ARGS` | Extra `llama-server` CLI flags                                   | `--no-mmproj`                    |

To switch models: edit `HF_REPO`/`HF_FILE`, then `docker compose up -d --force-recreate`. Each model caches independently, so switching back is instant.

## How it works

The server pulls the model via `LLAMA_ARG_HF_REPO` + `LLAMA_ARG_HF_FILE` into `/models` (`LLAMA_CACHE`), backed by the named volume `llama-models`, which survives `docker compose down`.

```bash
docker volume inspect llamacpp-infer_llama-models   # see where the cache lives
docker volume rm llamacpp-infer_llama-models         # wipe all downloaded models
```

## Backends

### CUDA (default)
Uses image `:server-cuda`. Just works with `docker compose up -d` given the NVIDIA Container Toolkit.

```bash
nvidia-smi                                          # confirm VRAM is held
docker logs llama-server 2>&1 | grep -i offload     # "offloaded N/N layers to GPU"
```

### Vulkan (AMD / Intel)
Uses image `:server-vulkan`, reaching the GPU via `/dev/dri` + the `video`/`render` groups.

```bash
docker compose -f docker-compose.vulkan.yml up -d
docker logs llama-server-vulkan 2>&1 | grep -i vulkan   # "Vulkan0: <your GPU>"
```

- **AMD:** also uncomment `- /dev/kfd:/dev/kfd` in the compose file.
- **GPU not detected?** Run `getent group render video` on the host and swap the group names in `group_add:` for the numeric GIDs it prints — the container's base image may not have those group names defined. Some driver combos also need extra GL libraries; see [this discussion](https://github.com/ggml-org/llama.cpp/discussions/16138).

### CPU
Uses image `:server`, no GPU block. Tune `THREADS` in `.env` (`0` = auto-detect). `NGL` is ignored.

```bash
docker compose -f docker-compose.cpu.yml up -d
```

Much slower than GPU — prefer smaller models and lower quants (`Q4_K_M` and below).

### `EXTRA_ARGS`
Appended verbatim to the server command, for presence-only flags:

- `--no-mmproj` — text-only, skips the multimodal projector (saves ~1.2 GB VRAM)
- empty — full multimodal (image/audio) support, if the model has it

Space-separate multiple flags, e.g. `--no-mmproj --mlock`.

## Using the API

OpenAI-compatible, base URL `http://localhost:8080/v1`.

```bash
# basic chat
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi in 3 words"}],"max_tokens":200}'

# streaming
curl -N http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":200,"stream":true}'

# streaming, text-only (strips the SSE/JSON envelope)
curl -sN http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":200,"stream":true}' \
  | grep '^data: ' | sed 's/^data: //' | grep -v '^\[DONE\]' \
  | jq -j '.choices[0].delta.content // empty'
```

**Reasoning models** (e.g. Gemma 3n) emit thinking before the answer — it lands in `reasoning_content`, separate from `content`. Skip it with:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

## Tuning for limited VRAM

Model weights + KV cache + compute buffers all need to fit in VRAM. If you hit a CUDA out-of-memory error, in order of cheapest win first:

1. `EXTRA_ARGS=--no-mmproj` — frees ~1.2 GB if you don't need image/audio
2. Lower `CTX_SIZE` — the KV cache scales with total context; `4096` vs `16384` cuts it ~4×. **Main lever.**
3. Lower `PARALLEL` — only helps if `CTX_SIZE` is also lowered (KV cache tracks total context, not per-slot)
4. Lower `NGL` — offload fewer layers to GPU, rest runs on CPU (slower, but fits)

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
