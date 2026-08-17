export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const workspace = $("#workspace");
export const imageDialog = $("#image-dialog");
export const systemDialog = $("#system-dialog");
export const queueButton = $("#queue-button");
export const queueLabel = $("#queue-label");
export const runCount = $("#run-count");

export const state = {
  meta: null,
  doctor: null,
  queue: null,
  jobs: [],
  createMode: localStorage.getItem("sdcpp:create-mode") || "single",
  createDraft: { recipe: null, gpu: null, parallelism: 1 },
  gallery: {
    page: 1,
    perPage: Number(localStorage.getItem("sdcpp:gallery-per-page") || 50),
    q: "",
    recipe: "",
    sort: "newest",
  },
};

export const PHASE = {
  pending: "已提交",
  preparing: "正在准备模型",
  recovering: "正在恢复任务",
  gpu_queued: "等待 GPU",
  gpu_running: "正在生成",
  running: "正在运行",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

export function money(value) {
  const n = Number(String(value ?? 0).replace("$", ""));
  return Number.isFinite(n) ? `$${n.toFixed(n < 0.01 ? 4 : 2)}` : "—";
}

export function timeAgo(iso) {
  if (!iso) return "";
  const delta = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(delta)) return "";
  const sec = Math.max(0, Math.floor(delta / 1000));
  if (sec < 60) return `${sec} 秒前`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} 分钟前`;
  const hour = Math.floor(min / 60);
  if (hour < 24) return `${hour} 小时前`;
  return `${Math.floor(hour / 24)} 天前`;
}

export function toast(message, kind = "ok") {
  const region = $("#toast-region");
  const node = document.createElement("div");
  node.className = `toast ${kind === "bad" ? "bad" : ""}`;
  node.textContent = message;
  region.append(node);
  setTimeout(() => node.remove(), 3500);
}

export async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

export function parseRoute() {
  const raw = (location.hash || "#/create").slice(1);
  const [pathRaw, queryRaw = ""] = raw.split("?");
  const parts = pathRaw.split("/").filter(Boolean);
  return { page: parts[0] || "create", id: parts[1] ? decodeURIComponent(parts[1]) : "", params: new URLSearchParams(queryRaw) };
}

export function setNav(page) {
  const top = page === "create" ? "create" : page === "gallery" ? "gallery" : "runs";
  $$('[data-route]').forEach((link) => {
    if (link.dataset.route === top) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

export function modelById(id) {
  return state.meta?.models?.find((model) => model.id === id) || state.meta?.models?.[0] || {};
}

export function gpuById(id) {
  return state.meta?.gpus?.find((gpu) => gpu.id === id) || state.meta?.gpus?.[0] || {};
}

export function phaseText(job) {
  if (!job) return "";
  if (job.phase === "gpu_queued") {
    const ahead = Number(job.queue?.ahead || 0);
    return ahead ? `等待 GPU · 前方 ${ahead} 个任务` : "等待 GPU · 即将开始";
  }
  if (job.phase === "gpu_running") {
    const p = Number(job.config?.parallelism || 1);
    return p > 1 ? `正在生成 · ${p} 路并行` : "正在生成";
  }
  return PHASE[job.phase] || PHASE[job.status] || job.status || "";
}

export function phaseClass(job) {
  const phase = job?.phase || job?.status || "pending";
  if (phase === "gpu_running" || phase === "running") return "running";
  if (phase === "gpu_queued") return "queued";
  if (phase === "preparing" || phase === "recovering" || phase === "pending") return "preparing";
  return phase;
}

export function progress(job) {
  const total = Number(job?.total_images || 0);
  const done = Number(job?.completed_images || 0) + Number(job?.failed_images || 0);
  return { total, done, percent: total ? Math.min(100, Math.round(done / total * 100)) : 0 };
}

export function pageTitle(eyebrow, title, subtitle, action = "") {
  return `<header class="page-title"><div class="page-title-copy"><p class="eyebrow">${escapeHtml(eyebrow)}</p><h1>${escapeHtml(title)}</h1><p>${escapeHtml(subtitle)}</p></div>${action}</header>`;
}

export function updateHeader() {
  const active = state.jobs.filter((job) => !["completed", "failed", "cancelled"].includes(job.status));
  runCount.hidden = active.length === 0;
  runCount.textContent = String(active.length);
  const q = state.queue;
  if (!q) {
    queueButton.dataset.state = "offline";
    queueLabel.textContent = "GPU 状态未知";
    return;
  }
  queueButton.dataset.state = q.state || "idle";
  if (q.running_count > 0) {
    const job = state.jobs.find((item) => item.id === q.running_job_ids?.[0]);
    const p = Number(job?.config?.parallelism || 1);
    queueLabel.textContent = `GPU 生成中${p > 1 ? ` · ×${p}` : ""}`;
  } else if (q.queue_length > 0) {
    queueLabel.textContent = `GPU 等待 ${q.queue_length}`;
  } else {
    queueLabel.textContent = "GPU 空闲";
  }
}

export async function refreshRuntime() {
  try {
    const [queuePayload, jobs] = await Promise.all([api("/api/runtime/queue"), api("/api/jobs?limit=50")]);
    state.queue = queuePayload.gpu || queuePayload;
    state.jobs = jobs || [];
    updateHeader();
    return true;
  } catch {
    queueButton.dataset.state = "offline";
    queueLabel.textContent = "本地 API 离线";
    return false;
  }
}
