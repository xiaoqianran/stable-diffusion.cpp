# Modal CLI / Image Lab

`examples/modal` 把本仓库的 `sd-cli` / `sd-server` 放到 Modal 上运行，同时把**模型下载、GPU 推理、本地任务状态**分开：

```text
Create / CLI
    │
    ├─ CPU storage worker ──> Modal Volume: sdcpp-models
    │                         原子下载 + SHA256 manifest + source index
    │
    └─ durable Job / FunctionCall IDs
              │
              └─ SDEngine(recipe, gpu)
                     └─ @modal.enter -> sd-server -> 模型常驻到 scale-down
```

默认 CUDA 镜像来自本 fork：

```text
ghcr.io/xiaoqianran/stable-diffusion.cpp:master-cuda
```

CI/CD 部署使用不可变标签 `sha-<git-sha>-cuda`，并把 Web / Modal deployment / image revision 放进 runtime identity，避免“本地代码已更新、远程 worker 还是旧版”。

## Setup

```bash
cd examples/modal
uv sync
uv run modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

公共 Hugging Face 文件不需要 token。gated 模型或 Civitai：

```bash
export HF_TOKEN=...
export CIVITAI_TOKEN=...
# 可选：持久 Modal Secret
modal secret create sdcpp-tokens HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN"
```

不要把 token 写入 git。

## CLI

```bash
uv run python sdcpp_modal.py prefetch --all
uv run python sdcpp_modal.py prefetch --status
uv run python sdcpp_modal.py ls
uv run python sdcpp_modal.py probe
uv run python sdcpp_modal.py generate -p "a rainy city at night" --recipe z-image-turbo -o zimage.png
uv run python sdcpp_modal.py generate -p "A fluffy orange cat" --recipe ideogram4 -o ideogram4.png
uv run python sdcpp_modal.py web
uv run python sdcpp_modal.py cost
uv run python sdcpp_modal.py cost --official
```

| Command | Where | Behavior |
| --- | --- | --- |
| `prefetch` / `pull` | CPU | 下载并验证模型，写入 `sdcpp-models` |
| `ls` | CPU | 查看 Volume artifact |
| `probe` | CUDA image / CPU | 查看远程 `sd-cli` 能力 |
| `generate` | CPU → GPU | CPU 先保证 artifact 完整，再启动 GPU |
| `web` | local FastAPI | **创建 / 运行 / 图库** Image Lab |
| `cost` | local | 本地成本**估算**；`--official` 另读 Modal 官方账期 summary |

## Artifact integrity

模型缓存不是“文件存在就算成功”。生产路径执行：

1. 下载到同目录唯一 `.partial-*` 文件；
2. 下载完整后用 `os.replace()` 原子发布；
3. 写 `.sdcpp.json`，记录 source、实际 resolved URL、bytes、SHA256；
4. 写 `.sdcpp-index/<sha256(source)>.json`，把原始 URI 映射到已验证的 immutable artifact；
5. GPU / `sd-server` 使用 `allow_download=False`，只读 source index，不联网下载。

`hf://org/repo/file` 第一次在 CPU 端会把移动的 `main` 解析成 Hugging Face commit SHA。之后同一 Volume 默认一直复用这个 immutable snapshot。若明确要刷新 `main`：

```bash
export SDCPP_REFRESH_MOVING_HF=1
uv run python sdcpp_modal.py prefetch --all
unset SDCPP_REFRESH_MOVING_HF
```

直接 HTTP URL 的缓存 key 包含完整 URL 的 SHA256 与 hostname，因此两个相同 basename 不会互相覆盖。

Modal Volume 上的 artifact writer 默认 `max_containers=1`。下载器内部仍可用 `SDCPP_PULL_WORKERS=4` 并行拉不同文件，但不会让多个 Modal writer container 同时提交同一路径。

## Persistent model pools

Web 的 bundled recipe 使用参数化：

```text
SDEngine(recipe=<recipe>, gpu_name=<gpu>)
```

`@modal.enter()` 启动 `sd-server` 并加载模型一次。同一 warm container 后续 Prompt 走本机 HTTP，不再每张图新启 `sd-cli`。如果 child `sd-server` 崩溃，worker 会检查进程状态并重启一次；第二次仍失败则让该 Modal input 失败，而不是把一个坏 container 永久留在池里。

通用 CLI 仍保留 `sd-cli` 路径，以兼容任意 CLI flags。

CPU/GPU idle window 分开：

```text
SDCPP_CPU_IDLE_SECONDS=10
SDCPP_GPU_IDLE_SECONDS=60
```

`min_containers=0`，所以最终仍会 scale to zero。60 秒 GPU 窗口是交互延迟与空闲费用之间的折中。

## Batch scheduling

Web batch 有三个档位：

| UI | parallelism | 最大 active GPU calls |
| --- | ---: | ---: |
| 省钱 | 1 | 1 |
| 平衡 | 2 | 2 |
| 最快 | 4 | 4 |

独立 Job 默认仍由本地 GPU job scheduler 串行拥有 GPU stage，因此两个 batch 不会把 `4 × 4` 叠成不可控的 fan-out。同 recipe/GPU 使用 affinity 调度，提高 warm pool 命中率，并有 streak/window 上限防止 starvation。

批量实现使用 Modal `Function.spawn()`，只维持 1/2/4 个 active `FunctionCall`；不会先创建整个 10000 张 batch 的 Future。每个 call ID 都写入 SQLite，便于取消和恢复。

## Durable jobs, cancel, restart recovery

Web job metadata 保存在：

```text
~/.cache/sdcpp-modal/web/sdcpp-web.db
```

可用 `SDCPP_WEB_DATA` 改路径。数据库启用 WAL、`busy_timeout` 和常用索引。

任务不再依赖“daemon thread 活着才算活着”：

- 本地 CPU staging 使用有界 `ThreadPoolExecutor`，默认 `SDCPP_CPU_JOB_WORKERS=4`；
- Modal GPU call ID 持久化到 `modal_calls`；
- Web 进程重启会把非终态 Job 标为 recovering，并通过 `FunctionCall.from_id()` 重新挂回尚未完成的远程 call；
- 点击 **停止** 会调用 `FunctionCall.cancel()`，不仅仅是把本地 UI 改成 cancelled；
- 远程 call 完成后，在 PNG 原子写盘前保留 call ID，因此本地恰好崩溃也可以再次读取 Modal 已完成结果，而不必重复生成。

历史 Job 不会全部塞给顶栏：`GET /api/jobs` 默认只返回最近 50 条，可用 `limit/offset/status` 分页。

终态数据清理：

```bash
curl -X POST 'http://127.0.0.1:7863/api/maintenance/cleanup?keep_days=30'
```

会删除过期终态 Job 及本地 output 文件。

## Web

```bash
uv run python sdcpp_modal.py web
uv run python sdcpp_modal.py web --dry-run
```

打开 <http://127.0.0.1:7863>。

一级入口只有：

- **创建**：单张与批量；
- **运行**：真实 Job phase、GPU queue、结果与估算成本；
- **图库**：50 / 100 / 200 分页、Prompt 搜索、模型筛选、图片详情。

运行详情通过 SSE `GET /api/jobs/{id}/events` 更新，不再每 1.8 秒整页轮询。顶栏只每 5 秒刷新最近任务和全局 GPU queue。

### TXT / JSONL

TXT：每行一个 Prompt。

JSONL 是真正解析后的结构，而不是把整行 JSON 当 Prompt：

```jsonl
{"prompt":"a cat","seed":42}
{"prompt":"a dog","seed":99,"count":2}
```

当前 JSONL per-line 字段为 `prompt`、`seed`、`count`。模型/GPU/尺寸/steps 是整个 Job 级配置。

Ideogram 4 用户可以直接输入普通文本；API 会自动转换为：

```json
{"high_level_description":"A fluffy orange cat"}
```

也可以自己传合法 JSON object。

### API limits

后端本身会验证资源上限，不依赖 HTML `max=`：

| Variable | Default |
| --- | ---: |
| `SDCPP_MAX_PROMPT_CHARS` | 20000 |
| `SDCPP_MAX_PROMPTS` | 5000 |
| `SDCPP_MAX_TOTAL_IMAGES` | 10000 |
| `SDCPP_MAX_COUNT` | 100 |
| `SDCPP_MAX_DIMENSION` | 4096 |
| `SDCPP_MAX_STEPS` | 200 |
| `SDCPP_MAX_UPLOAD_BYTES` | 10 MiB |

### Public binding / authentication

默认 `127.0.0.1` 不需要认证。如果把 Web bind 到 `0.0.0.0`、局域网地址或公网地址，CLI 会拒绝启动，除非设置：

```bash
export SDCPP_WEB_TOKEN='a-long-random-secret'
uv run python sdcpp_modal.py web --host 0.0.0.0
```

浏览器使用 HTTP Basic auth（密码填这个 token）；API 也接受：

```text
Authorization: Bearer <SDCPP_WEB_TOKEN>
```

这避免公开 Web 入口被陌生人直接调用你的付费 Modal GPU。

## Deployment identity and CI/CD

仓库构建 CUDA image 时同时发布：

```text
master-cuda
sha-<full-git-sha>-cuda
```

`.github/workflows/modal-deploy.yml` 会在 master 的 CI 成功后（且仓库配置了 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` secrets）部署：

```text
SDCPP_DEPLOY_SHA=<commit>
SDCPP_IMAGE=ghcr.io/xiaoqianran/stable-diffusion.cpp:sha-<commit>-cuda
```

部署后会调用 storage/GPU identity endpoint 验证 SHA 一致。不配置 secrets 时 workflow 明确 skip，不会把 token 硬编码到仓库。

本地 `ensure_deployed()` 也检查 deployment identity。旧版没有 identity endpoint 时会滚动升级；如果已有 identity endpoint 但 Modal 控制面临时不可达，会把错误上抛，**不会因为一次网络错误擅自重新 deploy**。

手工强制部署仍可用：

```bash
SDCPP_FORCE_DEPLOY=1 uv run python -c \
  'from sdcpp_hooks.deployed import ensure_deployed; ensure_deployed(force=True)'
```

## Model URIs

| URI | Meaning |
| --- | --- |
| `hf://org/repo/file.safetensors` | CPU 首次解析 `main` -> commit SHA，并在 Volume 固定 snapshot |
| `hf://org/repo@rev/file.gguf` | 明确 revision |
| `civitai://128713` | Civitai model version id |
| `https://...` | 直接下载；full URL 哈希防 basename collision |
| `/models/...` | Volume 上显式本地路径 |

七个 bundled recipes：`z-image-turbo`、`ideogram4`、`flux2-klein`、`flux2-dev`、`sdxl-turbo`、`sd2`、`sd15`。`ideogram4` / `flux2-dev` 默认 `RTX-PRO-6000`，其余默认 `L40S`。A10 / A100 在本 wrapper 中被阻止；L4 可选但大模型可能 OOM。

## Environment

| Variable | Default / meaning |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/xiaoqianran/stable-diffusion.cpp:master-cuda`; CI deploy uses immutable SHA tag |
| `SDCPP_DEPLOY_SHA` | 当前部署 commit identity |
| `SDCPP_GPU` | CLI override；Web 把实际 GPU 作为 SDEngine 参数传递 |
| `SDCPP_CPU_IDLE_SECONDS` | `10` |
| `SDCPP_GPU_IDLE_SECONDS` | `60` |
| `SDCPP_GPU_MAX_CONTAINERS` | generic CLI pool 默认 `1` |
| `SDCPP_WEB_GPU_POOL_MAX` | Web recipe pool 最大 `4` |
| `SDCPP_GPU_JOB_MAX_ACTIVE` | 独立 Web Job 同时拥有 GPU stage 的数量，默认 `1` |
| `SDCPP_CPU_JOB_WORKERS` | 本地 CPU job workers，默认 `4` |
| `SDCPP_PULL_WORKERS` | 单个 CPU writer 内并行下载数量，默认 `4` |
| `SDCPP_WEB_TOKEN` | 非 loopback Web 必需 |
| `SDCPP_WEB_DATA` | 本地 Job/图库数据目录 |
| `SDCPP_COST_LOG` | `~/.cache/sdcpp-modal/cost.jsonl` |
| `SDCPP_COST_WORKER_LOG` | 默认关闭；仅在你明确给安全的 container-local 路径时记录 worker lifetime |
| `SDCPP_REFRESH_MOVING_HF` | `1` 时主动重新解析 HF `main` |

## 成本教程

**重要：Image Lab 展示的是“估算成本”，不是 Modal 官方实际账单。**

本地 estimator 使用静态 rate snapshot，根据本机观察到的 `.remote()` 墙钟时间做近似。它无法精确包含所有 container idle tail、retry、credit、折扣和账单调整。真实账期请使用：

```bash
python3 sdcpp_modal.py cost --official
```

当前 SDK 读取 `Workspace.billing.summary()`；本仓库不再调用不存在的 `Workspace.billing.rates()`。

### 1. Dry run

```bash
python3 sdcpp_modal.py web --dry-run
```

浏览器：<http://127.0.0.1:7863>。演练生成 placeholder，不调用 GPU，Job 的估算成本为 `$0`。

### 2. 本地调用链

```bash
python3 sdcpp_modal.py cost
```

典型链仍可看到：

```text
session:storage
  remote:ensure_artifacts
session:gpu
  remote:generate
```

Web API：

```text
GET /api/cost
GET /api/cost?job_id=job_…
```

`GET /api/jobs` 和 `GET /api/jobs/{id}` 中的 `cost_usd` 为兼容字段；新的明确字段是 `estimated_cost_usd` / `cost_kind="estimate"`。

### 3. 静态估算费率示例

当前 fallback snapshot 中：

- L40S 约 `$0.000545531/s`（GPU + 假定 CPU/内存）；
- RTX-PRO-6000 约 `$0.000845531/s`。

这些数字用于**本地估算**，不代表 Modal 向当前账号最终结算的保证价格。

### 4. 为什么不能把 session 和 remote 加在一起

`session` 是本地调用窗口，`remote` 是窗口内真正远程调用的墙钟时间，两者会重叠。**不能把 session 和 remote 加在一起**，否则同一段时间会重复计算。

worker container lifetime 默认不再 append 到共享 Volume 的 `/models/...events.jsonl`，因为多个 container 对同一 Volume 文件并发 append 不可靠。若确实需要 worker-local 估算日志，显式设置 `SDCPP_COST_WORKER_LOG` 到安全路径。

### 5. Official billing

```bash
python3 sdcpp_modal.py cost --official
```

官方 summary 的 `metered` / `billed` 才来自 Modal Workspace billing。它与本地 `job_…` estimator 用途不同：前者回答“账期最终记了多少”，后者回答“这个 Job 大约消耗了多少 GPU 墙钟资源”。

## Tests

本地：

```bash
cd examples/modal
uv run pytest
node --check web/static/ux-core.js
node --check web/static/ux-create.js
node --check web/static/ux-runs.js
node --check web/static/ux-gallery.js
node --check web/static/ux-system.js
node --check web/static/ux-main.js
```

GitHub `Modal Python` workflow 会执行上述 JS syntax check + pytest。可靠性测试覆盖：原子下载、URL collision、未验证缓存拒绝、source index offline GPU resolution、SQLite WAL/index、FunctionCall cancel、JSONL parsing、`model` alias、API limits、bounded EventBus。

## Gallery dataset / Pages

CLI `--publish` 仍可把生成结果写到 Hugging Face gallery dataset；`.github/workflows/gallery-pages.yml` 负责 Pages。Web Image Lab 的本地 Gallery 则直接读取 `SDCPP_WEB_DATA/outputs` 与 SQLite，不依赖公开 dataset。

## Limits

- persistent `sd-server` path 当前针对 bundled text-to-image recipes；init image / control net 会退回 generic `sd-cli` path；
- `put` 适合小文件（64 MiB），大权重使用 `pull` 或 `modal volume put`；
- 本层 wrapper 的重点是可靠地调度 stable-diffusion.cpp，不改变 C API。
