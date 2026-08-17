const REFRESH_MS = 900;

const phaseLabels = {
  pending: "已提交",
  preparing: "CPU 准备模型中",
  gpu_queued: "GPU 排队中",
  gpu_running: "GPU 生成中",
  running: "进行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

let latestQueue = null;
let latestJobs = new Map();
let rendering = false;

async function json(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function jobPhaseText(job) {
  if (!job) return "";
  if (job.phase === "gpu_queued") {
    const ahead = Number(job.queue?.ahead || 0);
    return ahead > 0 ? `GPU 排队中 · 前方 ${ahead} 个任务` : "GPU 排队中 · 即将开始";
  }
  return phaseLabels[job.phase] || phaseLabels[job.status] || job.phase || job.status || "";
}

function ensureGlobalPanel() {
  const main = document.getElementById("main");
  const head = main?.querySelector(".page-head");
  if (!main || !head) return null;

  let panel = document.getElementById("gpu-queue-panel");
  if (panel && panel.parentElement !== main) panel.remove();
  panel = document.getElementById("gpu-queue-panel");
  if (!panel) {
    panel = document.createElement("section");
    panel.id = "gpu-queue-panel";
    panel.className = "gpu-queue-panel";
    panel.setAttribute("aria-live", "polite");
    panel.innerHTML = `
      <div class="gpu-queue-main">
        <span class="gpu-queue-dot" aria-hidden="true"></span>
        <div>
          <strong id="gpu-queue-title">GPU 状态</strong>
          <div class="gpu-queue-sub" id="gpu-queue-sub"></div>
        </div>
      </div>
      <div class="gpu-queue-metrics">
        <span><b id="gpu-running-count">0</b> 运行</span>
        <span><b id="gpu-waiting-count">0</b> 排队</span>
        <span><b id="gpu-slot-count">1</b> 并发上限</span>
      </div>`;
    head.insertAdjacentElement("afterend", panel);
  }
  return panel;
}

function renderGlobalQueue(queue) {
  const panel = ensureGlobalPanel();
  if (!panel || !queue) return;

  const title = panel.querySelector("#gpu-queue-title");
  const sub = panel.querySelector("#gpu-queue-sub");
  const running = panel.querySelector("#gpu-running-count");
  const waiting = panel.querySelector("#gpu-waiting-count");
  const slots = panel.querySelector("#gpu-slot-count");

  panel.dataset.state = queue.state || "idle";
  running.textContent = String(queue.running_count || 0);
  waiting.textContent = String(queue.queue_length || 0);
  slots.textContent = String(queue.max_active || 1);

  if ((queue.running_count || 0) > 0) {
    title.textContent = "GPU 生成中";
    const ids = queue.running_job_ids || [];
    sub.textContent = `${queue.queue_length || 0} 个任务等待 · 当前 ${ids.join(", ") || "远程任务"}`;
  } else if ((queue.queue_length || 0) > 0) {
    title.textContent = "GPU 排队中";
    sub.textContent = `${queue.queue_length} 个任务等待 GPU 槽位`;
  } else {
    title.textContent = "GPU 空闲";
    sub.textContent = "没有任务等待 GPU";
  }
}

function decorateJob(job, anchor) {
  if (!job || !anchor) return;
  const row = anchor.closest("tr");
  const pill = row?.querySelector(".pill");
  if (pill) {
    pill.textContent = jobPhaseText(job);
    [...pill.classList]
      .filter((name) => name.startsWith("phase-"))
      .forEach((name) => pill.classList.remove(name));
    pill.classList.add(`phase-${job.phase || job.status}`);

    let note = pill.parentElement?.querySelector(".queue-note");
    if (!note) {
      note = document.createElement("div");
      note.className = "queue-note";
      pill.insertAdjacentElement("afterend", note);
    }
    if (job.phase === "gpu_queued") {
      const position = job.queue?.position;
      note.textContent = position ? `执行顺序 #${position}` : "等待 GPU";
    } else if (job.phase === "preparing") {
      note.textContent = "CPU / Volume 阶段";
    } else if (job.phase === "gpu_running") {
      note.textContent = "占用 GPU 槽位";
    } else {
      note.textContent = "";
    }
  }
}

function decorateJobTable() {
  const seen = new Set();
  document.querySelectorAll('#main a[href^="#/job/"]').forEach((anchor) => {
    const id = anchor.getAttribute("href").slice("#/job/".length).split("?")[0];
    if (!id || seen.has(id)) return;
    seen.add(id);
    decorateJob(latestJobs.get(id), anchor);
  });
}

function decorateJobDetail() {
  const match = location.hash.match(/^#\/job\/([^?]+)/);
  if (!match) return;
  const job = latestJobs.get(decodeURIComponent(match[1]));
  if (!job) return;
  const panel = document.querySelector("#main .panel");
  if (!panel) return;

  let row = panel.querySelector(".gpu-job-queue-row");
  if (!row) {
    row = document.createElement("div");
    row.className = "row gpu-job-queue-row";
    row.innerHTML = '<span>GPU 调度</span><span class="gpu-job-queue-value"></span>';
    const statusRow = panel.querySelector(".row:nth-child(2)");
    if (statusRow) statusRow.insertAdjacentElement("afterend", row);
    else panel.prepend(row);
  }
  const value = row.querySelector(".gpu-job-queue-value");
  value.textContent = jobPhaseText(job);
}

function decorateActiveSubmission() {
  const idNode = document.getElementById("job-id");
  const jobId = idNode?.textContent?.trim();
  if (!idNode || !jobId) return;
  const job = latestJobs.get(jobId);
  if (!job) return;

  let note = document.getElementById("active-job-queue");
  if (!note) {
    note = document.createElement("span");
    note.id = "active-job-queue";
    note.className = "active-job-queue";
    idNode.insertAdjacentElement("afterend", note);
  }
  note.textContent = jobPhaseText(job);
}

function renderCached() {
  if (rendering) return;
  rendering = true;
  try {
    renderGlobalQueue(latestQueue);
    decorateJobTable();
    decorateJobDetail();
    decorateActiveSubmission();
  } finally {
    rendering = false;
  }
}

async function refresh() {
  try {
    const [queuePayload, jobs] = await Promise.all([
      json("/api/runtime/queue"),
      json("/api/jobs"),
    ]);
    latestQueue = queuePayload.gpu || queuePayload;
    latestJobs = new Map((jobs || []).map((job) => [job.id, job]));
    renderCached();
  } catch {
    const panel = ensureGlobalPanel();
    if (panel) {
      panel.dataset.state = "offline";
      const title = panel.querySelector("#gpu-queue-title");
      const sub = panel.querySelector("#gpu-queue-sub");
      if (title) title.textContent = "GPU 队列状态不可用";
      if (sub) sub.textContent = "等待本地 API 恢复";
    }
  }
}

const observer = new MutationObserver(() => renderCached());
observer.observe(document.getElementById("main"), { childList: true, subtree: true });
window.addEventListener("hashchange", () => setTimeout(renderCached, 0));

refresh();
setInterval(refresh, REFRESH_MS);
