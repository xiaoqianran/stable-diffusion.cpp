# Modal CLI

Run the official `sd-cli` on [Modal](https://modal.com) when the local machine cannot host the model.

This example does not change the C API, local CLI, or server. It treats `sd-cli` as a black box: each GPU container probes `--help` and only forwards flags that exist on that binary.

Weights live on Modal Volume `sdcpp-models`. The default image is `ghcr.io/leejet/stable-diffusion.cpp:master-cuda`.

## Setup

```bash
cd examples/modal
python3 -m pip install 'modal>=0.64' pytest
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

Public Hugging Face files work without a token. For gated models or Civitai:

```bash
export HF_TOKEN=...
export CIVITAI_TOKEN=...
# optional persistent secret
modal secret create sdcpp-tokens HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN"
```

Do not put tokens in git.

## CLI

```bash
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py ls
python3 sdcpp_modal.py probe
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe sd15 -o cat.png --publish
python3 sdcpp_modal.py publish cat.png --recipe sd15 -p "a lovely cat"
python3 sdcpp_modal.py web
python3 sdcpp_modal.py cost
python3 sdcpp_modal.py cost --official
```

| Command | Where it runs | What it does |
| --- | --- | --- |
| `pull` | CPU | download URIs, a `--recipe`, or `--all` recipes onto volume `sdcpp-models` |
| `put` | CPU | upload a small local file (init image, mask) to `uploads/` |
| `ls` | CPU | list files already on that volume |
| `probe` | CUDA image, no GPU | print remote `sd-cli` flags |
| `generate` | CPU, then GPU (`SDCPP_GPU`, default `L40S`) | CPU pulls missing weights; GPU only loads them and runs `sd-cli` |
| `publish` | local + Hugging Face | upload a PNG into the multi-model gallery dataset |
| `web` | local FastAPI | 生成 / 批量 / 任务 / 成本 / 画廊 workbench on `:7860` |
| `cost` | local (+ optional Modal API) | print the call-chain ledger (see [成本教程](#成本教程)) |

`generate` first ensures missing URIs are on the volume **from a CPU container**. The GPU container only reloads those files and runs `sd-cli`. It does not download weights.

CPU pulls use `aria2c` (`-x 16 -s 16 -c -k 1M`) when it is on the image, then the Hugging Face CLI, then urllib. Several missing files download in parallel (`SDCPP_PULL_WORKERS`, default 4). Tokens stay in headers or `CIVITAI_TOKEN` query params and are redacted in logs.

```bash
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py generate -p '{"high_level_description":"A fluffy orange cat"}' --recipe ideogram4 -o ideogram4.png --publish
```

Do not convert Ideogram4 weights yourself. `pull --recipe ideogram4` downloads the prebuilt GGUF pair from [`leejet/ideogram-4-GGUF`](https://huggingface.co/leejet/ideogram-4-GGUF) (`ideogram4-Q4_0.gguf` and `ideogram4_uncond-Q4_0.gguf`), plus the FLUX.2 VAE and Qwen3-VL GGUF. There is no `convert` command in this CLI. The FLUX.2 VAE is gated, so set `HF_TOKEN`. Ideogram4 prompts must be JSON.

The default GPU is `L40S`. `ideogram4` and `flux2-dev` default to `RTX-PRO-6000` (96 GB Blackwell) unless `SDCPP_GPU` is set. A 24 GB `L4` can OOM on the diffusion compute buffer. A10 and A100 are blocked. Modal's live GPU name is `RTX-PRO-6000`; `RTX6000` is accepted as an alias.

## Web

The local workbench follows the [modal-sana](https://github.com/xiaoqianran/modal-sana) split: Interface / Core / Modal. `python3 sdcpp_modal.py web` starts FastAPI on `http://127.0.0.1:7860`. It is **not** `modal serve`. The page owns jobs, SSE progress, and a local gallery. GPU inference still goes through the existing `sdcpp-storage` + `sdcpp-cli` workers and the seven recipes.

```bash
python3 -m pip install 'fastapi>=0.115' 'uvicorn>=0.30' pillow python-multipart
python3 sdcpp_modal.py web
python3 sdcpp_modal.py web --dry-run   # placeholder images, no GPU
```

Pages: **生成**, **批量**, **任务**, **成本**, **画廊**, **设置**. The static UI is plain HTML / CSS / JS talking to FastAPI, in Simplified Chinese. Default recipe is `z-image-turbo`. Selecting Ideogram 4 or FLUX.2 Dev also selects RTX PRO 6000. Job metadata lives in `~/.cache/sdcpp-modal/web/` (override with `SDCPP_WEB_DATA`). `--dry-run` or `SDCPP_WEB_DRY_RUN=1` writes prompt placeholders so the UI can be tested without Modal.

Idle CPU and GPU containers scale to zero after `SDCPP_IDLE_SECONDS` (default **10**). `min_containers=0`, so nothing stays warm when there are no requests.

How to read every Modal charge, the per-second rate, and the matching job: **[成本教程](#成本教程)**.

Common `sd-cli` flags are first-class (`--vae`, `--diffusion-model`, `--init-img`, `--control-net`, `--taesd`, `--sampling-method`, ...). Any other remote flag can be appended and is forwarded if `probe` sees it:

```bash
python3 sdcpp_modal.py generate -p "a cat" --recipe sd15 --offload-to-cpu --type f16
```

Unknown flags are dropped and printed as `dropped_fields` instead of failing the run.

### Model URIs

| URI | Meaning |
| --- | --- |
| `hf://org/repo/file.safetensors` | Hugging Face file at revision `main` |
| `hf://org/repo@rev/file.gguf` | Hugging Face file at `rev` |
| `civitai://128713` | Civitai **model version** id |
| `https://...` | Direct download |

`HF_ENDPOINT` can point at a Hugging Face mirror.

For a multi-gigabyte local checkpoint, use `modal volume put` instead of `put`:

```bash
modal volume put sdcpp-models ./v1-5-pruned-emaonly.safetensors hf/local/v1-5-pruned-emaonly.safetensors
```

Then pass `--model /models/hf/local/v1-5-pruned-emaonly.safetensors`.

## Bundled models

This CLI ships **seven** recipes and nothing else. `pull --all` downloads every file they need onto volume `sdcpp-models`. Other Modal recipes (SD-Turbo, SSD-1B, Dreamlike, FLUX.1, and same-family klein/base variants) were removed.

Artificial Analysis text-to-image Elo is from the [open-weights leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights) on 2026-08-16. SDXL-Turbo / SD 2.1 / SD 1.5 are older checkpoints and are not ranked there the same way.

| Recipe | Model | AA Elo | Defaults | Volume files under `/models/` |
| --- | --- | --- | --- | --- |
| `ideogram4` | Ideogram 4.0 Q4_0 | 1217 | 1024², 20 steps, cfg 4.0 | `hf/leejet/ideogram-4-GGUF/main/ideogram4-Q4_0.gguf`, `ideogram4_uncond-Q4_0.gguf`; shared FLUX.2 VAE; `hf/unsloth/Qwen3-VL-8B-Instruct-GGUF/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf` |
| `flux2-klein` | FLUX.2 [klein] 9B Q4_0 | 1149 | 1024², 4 steps, cfg 1.0 | `hf/leejet/FLUX.2-klein-9B-GGUF/main/flux-2-klein-9b-Q4_0.gguf`; shared FLUX.2 VAE; `hf/unsloth/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf` |
| `flux2-dev` | FLUX.2 [dev] Q4_K_S | 1200 | 1024², 20 steps, cfg 1.0, euler | `hf/city96/FLUX.2-dev-gguf/main/flux2-dev-Q4_K_S.gguf`; shared FLUX.2 VAE; `hf/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` |
| `z-image-turbo` | Z-Image Turbo Q3_K | 1131 | 512×1024, 8 steps, cfg 1.0 | `hf/leejet/Z-Image-Turbo-GGUF/main/z_image_turbo-Q3_K.gguf`; `hf/black-forest-labs/FLUX.1-schnell/main/ae.safetensors`; `hf/unsloth/Qwen3-4B-Instruct-2507-GGUF/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| `sdxl-turbo` | SDXL-Turbo fp16 | — | 512², 4 steps, cfg 1.0, euler | `hf/stabilityai/sdxl-turbo/main/sd_xl_turbo_1.0_fp16.safetensors` |
| `sd2` | Stable Diffusion 2.1 | — | 512², 20 steps, cfg 7.0 | `hf/Manojb/stable-diffusion-2-1-base/main/v2-1_512-ema-pruned.safetensors` |
| `sd15` | Stable Diffusion 1.5 | — | 512², 20 steps, cfg 7.0 | `hf/stable-diffusion-v1-5/stable-diffusion-v1-5/main/v1-5-pruned-emaonly.safetensors` |

Shared FLUX.2 VAE (gated): `hf/black-forest-labs/FLUX.2-dev/main/ae.safetensors`. Used by Ideogram 4.0, FLUX.2 [klein], and FLUX.2 [dev]. Z-Image Turbo uses the Flux.1 schnell VAE, not the Flux.2 VAE. Qwen3-8B (klein) is not Qwen3-VL-8B (Ideogram4).

GGUF diffusion files still need their VAE and text-encoder companions. `pull --all` is the CPU path that fetches the full set.

### Volume inventory (2026-08-16)

CPU `pull --all` on volume `sdcpp-models` left only these weight files (77.60 GiB). Leftover FLUX.1-dev and the old extra recipes were deleted.

| Path under `/models/` | Bytes | Size |
| --- | ---: | ---: |
| `hf/leejet/ideogram-4-GGUF/main/ideogram4-Q4_0.gguf` | 5643820832 | 5.26 GiB |
| `hf/leejet/ideogram-4-GGUF/main/ideogram4_uncond-Q4_0.gguf` | 5643820832 | 5.26 GiB |
| `hf/unsloth/Qwen3-VL-8B-Instruct-GGUF/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf` | 5027785568 | 4.68 GiB |
| `hf/black-forest-labs/FLUX.2-dev/main/ae.safetensors` | 336211292 | 0.31 GiB |
| `hf/leejet/FLUX.2-klein-9B-GGUF/main/flux-2-klein-9b-Q4_0.gguf` | 5616208032 | 5.23 GiB |
| `hf/unsloth/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf` | 5027784512 | 4.68 GiB |
| `hf/city96/FLUX.2-dev-gguf/main/flux2-dev-Q4_K_S.gguf` | 19299128288 | 17.97 GiB |
| `hf/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` | 14333922848 | 13.35 GiB |
| `hf/leejet/Z-Image-Turbo-GGUF/main/z_image_turbo-Q3_K.gguf` | 3143559104 | 2.93 GiB |
| `hf/black-forest-labs/FLUX.1-schnell/main/ae.safetensors` | 335304388 | 0.31 GiB |
| `hf/unsloth/Qwen3-4B-Instruct-2507-GGUF/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497281120 | 2.33 GiB |
| `hf/stabilityai/sdxl-turbo/main/sd_xl_turbo_1.0_fp16.safetensors` | 6938081905 | 6.46 GiB |
| `hf/Manojb/stable-diffusion-2-1-base/main/v2-1_512-ema-pruned.safetensors` | 5214604494 | 4.86 GiB |
| `hf/stable-diffusion-v1-5/stable-diffusion-v1-5/main/v1-5-pruned-emaonly.safetensors` | 4265146304 | 3.97 GiB |

```bash
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe flux2-klein -o klein.png
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe flux2-dev -o flux2.png
python3 sdcpp_modal.py generate -p "a rainy city at night" --recipe z-image-turbo -o zimage.png
```

`ideogram4`, `flux2-klein`, `flux2-dev`, and `z-image-turbo` also pass `--diffusion-fa` and `--offload-to-cpu`.

### Environment

| Variable | Default |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` |
| `SDCPP_GPU` | unset: `L40S`, or `RTX-PRO-6000` for `ideogram4` / `flux2-dev` (also `L4`; A10 and A100 are blocked) |
| `SDCPP_IDLE_SECONDS` | `10` (CPU and GPU scale to zero after this idle window) |
| `SDCPP_SECRET` | `sdcpp-tokens` (used only if that Modal secret exists) |
| `SDCPP_PULL_WORKERS` | `4` (parallel CPU downloads) |
| `HF_ENDPOINT` | `https://huggingface.co` |
| `SDCPP_GALLERY_DATASET` | `seachen/stable-diffusion-cpp-gallery` |
| `SDCPP_GITHUB_REPO` | `xiaoqianran/stable-diffusion.cpp` |
| `SDCPP_COST_LOG` | `~/.cache/sdcpp-modal/cost.jsonl` |

## 成本教程

工作台侧栏的 **成本** 就是 Modal 账本。每一笔 `app.run` / `.remote` 都会留下一行：调用链、精确到秒的费率、`费率 × 秒数 = 这一笔`、以及对应的 `job_…` / `img_…`。计费不进 `sd-cli`，只在 `sdcpp_hooks/cost.py` 和 `sdcpp_hooks/modal_meter.py`。

下面按顺序做一遍：先演练（$0），再真生成，再读调用链。

### 1. 打开工作台

```bash
cd examples/modal
python3 -m pip install 'fastapi>=0.115' 'uvicorn>=0.30' pillow python-multipart 'modal>=0.64'
python3 sdcpp_modal.py web
```

浏览器打开 <http://127.0.0.1:7860>。左侧应有：生成 / 批量 / 任务 / **成本** / 画廊 / 设置。

还没装 Modal token、只想看页面时：

```bash
python3 sdcpp_modal.py web --dry-run
```

演练不会打 GPU，占位图仍会写成一条任务，成本页记 `local:dry_run` · **$0**。

### 2. 演练：确认任务能挂上账本

1. 打开 **生成**。
2. 提示词随便写，例如 `雨夜城市`。
3. 配方用默认 `z-image-turbo`，显卡保持 `L40S`。
4. 展开「尺寸与采样」，勾选 **演练（不调用 Modal / GPU）**。
5. 点 **开始生成**。
6. 打开 **任务**：费用列应是 `演练 · $0`。点任务编号，详情里有调用链。
7. 打开 **成本**，或点费用跳到 `#/cost?job=job_…`。

此时账本里至少有一行：

| 调用链 | 时长 | 费用 | 每秒 | 任务 |
| --- | --- | --- | --- | --- |
| `local:dry_run` | 0.000s | $0 | $0/s | `job_…` |

这只证明页面和任务对得上。**$0 不是 Modal 报价。**

### 3. 真生成：一次完整调用链

先保证 token 和权重在（`modal token set`，`export HF_TOKEN=…`，至少 `python3 sdcpp_modal.py pull --recipe z-image-turbo`）。

1. **生成**页取消「演练」。
2. 仍用 `z-image-turbo` + `L40S`，张数 1，点 **开始生成**。
3. 等任务变成「已完成」，打开 **成本**。

一次工作台出图会写下 **两条 session**，每条下面挂 `.remote`：

```
session:storage                          app.run(sdcpp-storage)  只有 CPU + 内存
  ↳ remote:ensure_artifacts              确认 / 补齐卷上的权重
session:gpu                              app.run(sdcpp-cli)      会话本身仍按 CPU 计价
  ↳ remote:generate                      真正占 GPU 的那一段，挂 img_…
```

命令行同样记账，只是没有 `job_…`：

```bash
python3 sdcpp_modal.py generate -p "a rainy city at night" --recipe z-image-turbo -o /tmp/zimage.png
python3 sdcpp_modal.py cost
```

`ideogram4` / `flux2-dev` 默认 `RTX-PRO-6000`（$3.03/h），其余默认 `L40S`（$1.95/h）。A10 / A100 已禁用。

### 4. 怎么读「成本」这一页

页顶是 **已入账**：去重后的美元、笔数、计费秒数。下面一排是当前费率（L40S / L4 / RTX-PRO-6000 / CPU / 内存）。再下面是调用链表。

| 列 | 含义 |
| --- | --- |
| 调用链 | `phase:name`。无缩进 = `app.run` 会话；`↳` = 这次会话里的 `.remote` |
| 时长 | 这一段墙钟秒数，精确到 0.001s |
| 费用 | `usd_per_second × duration_s`，量化到 $0.000001 |
| 每秒 | 这一段资源的合计费率（GPU + 0.125 CPU + 1 GiB 内存） |
| 拆分 | `gpu $…/s × Ns = $…` · `cpu …` · `memory …` |
| GPU | `L40S` / `L4` / `RTX-PRO-6000`；CPU 会话为 — |
| 任务 | `job_…`，点进任务详情 |
| 图片 | `img_…`，只挂在 `remote:generate` 上 |

同一页下方还有「按任务汇总」。任务详情里的「调用链」按钮等于 `#/cost?job=job_…`。

设置页的「成本账本」是 jsonl 路径，默认 `~/.cache/sdcpp-modal/cost.jsonl`。

### 5. 每秒费率

数字来自 `modal.Workspace.billing.rates()`；Modal 连不上时用 2026-08-15 工作区 `pythonmoive` 的快照。工作台顶栏会写 `modal-billing-rates` 或 `fallback`。

默认容器按 **0.125 CPU + 1 GiB 内存** 计价（Modal 未公布真实 reserved 时的假定）。

| 资源 | 每小时 | 每秒（写入账本的精度） | 1 秒里还有 |
| --- | ---: | ---: | --- |
| L40S GPU | $1.95 | $0.000541667/s | + CPU + 内存 → **$0.000545531/s** |
| L4 GPU | $0.80 | $0.000222222/s | + CPU + 内存 → **$0.000226087/s** |
| RTX-PRO-6000 GPU | $3.03 | $0.000841667/s | + CPU + 内存 → **$0.000845531/s** |
| CPU 0.125 核 | $0.04730 × 0.125 | **$0.000001642/s** | storage / session 会话 |
| 内存 1 GiB | $0.008 | **$0.000002222/s** | 同上 |
| CPU+内存（无 GPU） | — | **$0.000003865/s** | `session:*`、`remote:ensure_artifacts` |

卷存储 $0.09/GiB·月 **不按次**写入这本账，只在官方账单里。

### 6. 一笔钱怎么算

```
这一笔 = (GPU小时价/3600 + CPU小时价×0.125/3600 + 内存小时价×1/3600) × 秒数
```

对照一次真实出图（数字会随墙钟变化，费率不变）：

| 调用 | 卡 | 时长 | 每秒 | 这一笔 |
| --- | --- | ---: | ---: | ---: |
| `remote:ensure_artifacts` | CPU | 8.000s | $0.000003865/s | $0.000031 |
| `remote:generate` | L40S | 47.000s | $0.000545531/s | $0.025640 |
| `remote:generate` | RTX-PRO-6000 | 33.000s | $0.000845531/s | $0.027903 |

拆开 47s 的 L40S：

```
gpu    $0.000541667/s × 47.000s = $0.025458
cpu    $0.000001642/s × 47.000s = $0.000077
memory $0.000002222/s × 47.000s = $0.000104
合计   $0.000545531/s × 47.000s = $0.025640
```

所以：**Pro 6000 更快，但不一定更便宜。** 同样一张图，33s × $3.03/h ≈ $0.0279，47s × $1.95/h ≈ $0.0256。

### 7. 为什么不能把 session 和 remote 加在一起

`app.run()` 从进到出是一条 **session**。里面的 `.remote()` 是同一段墙钟上的 **remote**。两段重叠。把两行的 `费用` 列加起来会把 GPU 时间算两次。

已入账只加：

1. 所有 `remote`（以及演练的 `local`）
2. `session` 比 remote 多出来的那几秒，按 **CPU+内存** 计价（连上 Modal、镜像拉取、会话收尾）

GPU 容器 `@modal.enter` → `@modal.exit`（含空闲到 `SDCPP_IDLE_SECONDS` 缩成 0）写在 worker 账本，默认 `/models/.sdcpp-cost/events.jsonl`，需要卷可写。工作台「已入账」看的是本机 `SDCPP_COST_LOG`，不含那段空闲，除非你自己把 worker 行合进来。

### 8. 命令行和 API

```bash
# 本机账本：调用树 + 每秒费率 + 累计已入账
python3 sdcpp_modal.py cost

# 再加 Modal 官方账期汇总（metered / billed / 分项）
python3 sdcpp_modal.py cost --official
```

CLI 示例：

```
trace a1b2c3d4e5f6  0.025648  jobs job_ab12cd34ef  4 calls
  session:storage  9.200s  $0.000036  $0.000003865/s  job job_ab12cd34ef
    remote:ensure_artifacts  8.100s  $0.000031  $0.000003865/s  job job_ab12cd34ef
  session:gpu  48.400s  $0.000187  $0.000003865/s  job job_ab12cd34ef
    remote:generate  47.000s  $0.025640  $0.000545531/s  gpu L40S  job job_ab12cd34ef  img img_ff11aa22bb
ledger /home/you/.cache/sdcpp-modal/cost.jsonl
all-time billed estimate $0.025648
per-second cpu $0.000001642/s  mem $0.000002222/s
per-second gpu L40S $0.000545531/s  (1.95000/h)
per-second gpu L4 $0.000226087/s  (0.80000/h)
per-second gpu RTX-PRO-6000 $0.000845531/s  (3.03000/h)
```

工作台读同一本账。`GET /api/cost` 是全部 traces；`GET /api/cost?job_id=` 只看一个任务：

```bash
curl -s 'http://127.0.0.1:7860/api/cost' | python3 -m json.tool
curl -s 'http://127.0.0.1:7860/api/cost?job_id=job_ab12cd34ef'
```

`GET /api/jobs` / `GET /api/jobs/{id}` 带 `cost_usd`、`cost_events`、`cost_chain`。换账本路径：

```bash
export SDCPP_COST_LOG=$PWD/cost.jsonl
python3 sdcpp_modal.py web
```

### 9. 对不上官方账单时

| 现象 | 原因 |
| --- | --- |
| 成本页是空的 | 还没跑过生成 / 演练，或 `SDCPP_COST_LOG` 指到了另一份文件 |
| 只有 `local:dry_run` · $0 | 勾了演练，或 `SDCPP_WEB_DRY_RUN=1` |
| 任务费用是 $0，但 Modal 控制台有钱 | 看的是旧任务；打开 **成本** 看全部 traces |
| 自己把表里两行加起来比「已入账」大 | session 与 remote 重叠，见第 7 节 |
| 和 `cost --official` 对不上 | 官方含本账期所有 app、卷存储、别人的 run；本页只估计 **这台机器记下的 sdcpp session/remote** |
| Pro 6000 出图却按 L40S 计价 | 旧版本的 bug，当前工作台会在生成前写入 `SDCPP_GPU` |
| 想清空 | 删掉 `SDCPP_COST_LOG` 那个 jsonl（不可恢复） |

## Gallery dataset and Pages

Generated images are stored on the public Hugging Face dataset
[`seachen/stable-diffusion-cpp-gallery`](https://huggingface.co/datasets/seachen/stable-diffusion-cpp-gallery),
one folder per bundled recipe (`images/ideogram4`, `images/flux2-klein`,
`images/sd15`, ...). A new `--model-id` creates a new folder, so later models
do not need a schema change.

```bash
python3 -m pip install 'huggingface_hub>=0.26'
export HF_TOKEN=...
python3 sdcpp_modal.py publish cat.png --model-id sd15 -p "a lovely cat" --seed 42
```

`generate --publish` writes the prompt plus run facts into the sidecar: duration,
GPU name, CUDA version, torch version (or a note that sd-cli uses ggml), NVIDIA
driver, sd-cli version, Python, Modal GPU type, and the container image. Pages
cards show those fields under the prompt.

GitHub Actions workflow `.github/workflows/gallery-pages.yml` downloads that
dataset and deploys a paginated gallery to GitHub Pages (12 images per page,
filters for current and future model families):
https://xiaoqianran.github.io/stable-diffusion.cpp/

## Limits

- First-class path is `img_gen`. `vid_gen`, `adetailer`, `convert`, and `metadata` are not wrapped; some of their flags can still be forwarded if the remote binary accepts them.
- `put` is for small files (64 MiB). Weights should use `pull` or `modal volume put`.
- Size defaults match local `sd-cli` (512x512, 20 steps, cfg 7.0) unless you override them or use a recipe.

## Tests

```bash
python3 -m pytest
```

These tests do not download weights and do not need a GPU.
