const main = document.getElementById("main");
const lightbox = document.getElementById("lightbox");
const railStatus = document.getElementById("rail-status");
const notice = document.getElementById("notice");

const state = { meta: null, source: null };

const STATUS = {
  queued: "排队",
  running: "进行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

const DOCTOR = {
  recipes: "配方",
  data_dir: "数据目录",
  modal: "Modal",
  pillow: "Pillow",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function showNotice(message, kind = "ok") {
  notice.hidden = !message;
  notice.dataset.kind = kind;
  notice.textContent = message || "";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  if (response.headers.get("content-type")?.includes("text/event-stream")) return response;
  return response.json();
}

function defaultGpuFor(meta, recipeId) {
  const model = (meta.models || []).find((item) => item.id === recipeId);
  return model?.default_gpu || meta.defaults?.gpu || "L40S";
}

function defaultsFrom(meta, recipeId) {
  const recipe = recipeId || meta.defaults?.recipe || "z-image-turbo";
  const model = (meta.models || []).find((item) => item.id === recipe) || {};
  return {
    recipe,
    gpu: defaultGpuFor(meta, recipe),
    count: 1,
    width: model.width || 512,
    height: model.height || 1024,
    steps: model.default_steps || 8,
    cfg_scale: model.cfg_scale || 1,
    seed: 101,
    dry_run: false,
  };
}

function gpuOptionLabel(gpu) {
  const bits = [gpu.label || gpu.name || gpu.id];
  if (gpu.vram_gb) bits.push(`${gpu.vram_gb}GB`);
  if (gpu.usd_per_hour != null) bits.push(`$${Number(gpu.usd_per_hour).toFixed(2)}/时`);
  return bits.join(" · ");
}

function field(id, name, label, value, type = "text") {
  if (type === "checkbox") {
    return `<label class="check" for="${id}"><span>${label}</span><input id="${id}" type="checkbox" name="${name}" ${value ? "checked" : ""} /></label>`;
  }
  const extra = type === "number" ? `inputmode="decimal"` : "";
  return `<div class="field"><label class="field-label" for="${id}">${label}</label><input id="${id}" name="${name}" type="${type}" value="${escapeHtml(value ?? "")}" ${extra} /></div>`;
}

function select(id, name, label, options, value) {
  const opts = options.map((item) => {
    const optionId = item.id || item;
    const title = item.name || item.id || item;
    return `<option value="${escapeHtml(optionId)}" ${optionId === value ? "selected" : ""}>${escapeHtml(title)}</option>`;
  }).join("");
  return `<div class="field"><label class="field-label" for="${id}">${label}</label><select id="${id}" name="${name}">${opts}</select></div>`;
}

function recipeFieldset(meta, selected, prefix) {
  const cards = meta.models || [];
  return `
    <fieldset>
      <legend>配方</legend>
      <div class="choice-list">
        ${cards.map((model) => `
          <label class="choice">
            <input type="radio" name="recipe" id="${prefix}-recipe-${escapeHtml(model.id)}" value="${escapeHtml(model.id)}" ${model.id === selected ? "checked" : ""} />
            <span>
              <span class="choice-name">${escapeHtml(model.label_zh || model.label || model.name)}</span>
              <span class="choice-id">${escapeHtml(model.id)}</span>
            </span>
            <span class="choice-meta">${model.width}×${model.height} · ${model.default_steps} 步</span>
            ${model.default_gpu === "RTX-PRO-6000" ? `<span class="tag">PRO 6000</span>` : `<span></span>`}
          </label>`).join("")}
      </div>
    </fieldset>`;
}

function composerFields(prefix, defaults, meta, submitLabel) {
  return `
    <div class="composer">
      ${select(`${prefix}-gpu`, "gpu", "显卡", (meta.gpus || []).map((gpu) => ({
        id: gpu.id,
        name: gpuOptionLabel(gpu),
      })), defaults.gpu)}
      ${field(`${prefix}-count`, "count", "张数", defaults.count, "number")}
      <div class="field">
        <label class="field-label" for="${prefix}-go">出图</label>
        <button id="${prefix}-go" type="submit">${submitLabel}</button>
      </div>
    </div>`;
}

function settingsFields(prefix, defaults, meta) {
  return `
    ${recipeFieldset(meta, defaults.recipe, prefix)}
    <details class="advanced">
      <summary>尺寸与采样</summary>
      <div class="grid">
        ${field(`${prefix}-width`, "width", "宽度", defaults.width, "number")}
        ${field(`${prefix}-height`, "height", "高度", defaults.height, "number")}
        ${field(`${prefix}-steps`, "steps", "步数", defaults.steps, "number")}
        ${field(`${prefix}-cfg`, "cfg_scale", "CFG", defaults.cfg_scale, "number")}
        ${field(`${prefix}-seed`, "seed", "种子", defaults.seed, "number")}
      </div>
      ${field(`${prefix}-dry`, "dry_run", "演练（不调用 Modal / GPU）", defaults.dry_run, "checkbox")}
    </details>
  `;
}

function formPayload(form) {
  const data = new FormData(form);
  const num = (key) => {
    const value = data.get(key);
    return value === "" || value == null ? null : Number(value);
  };
  return {
    recipe: data.get("recipe"),
    gpu: data.get("gpu"),
    count: Number(data.get("count") || 1),
    width: num("width"),
    height: num("height"),
    steps: num("steps"),
    cfg_scale: num("cfg_scale"),
    seed: num("seed"),
    dry_run: data.get("dry_run") === "on",
  };
}

function money(value) {
  if (value == null || value === "") return "—";
  const text = String(value).replace(/^\$/, "");
  if (Number(text) === 0) return "$0";
  return `$${text}`;
}

function jobCostLabel(job) {
  if (job?.config?.dry_run) return "演练 · $0";
  if (job?.cost_usd == null) return "计费中";
  return money(job.cost_usd);
}

function flattenChain(input) {
  if (!input || !input.length) return [];
  if (input[0].chain) {
    return input.flatMap((trace) => trace.chain || []);
  }
  return input;
}

function renderCostRates(rates) {
  const cards = rates?.cards || [];
  if (!cards.length) return "";
  return `<div class="rate-strip">${cards.map((card) => `
    <span class="rate-chip">
      <strong>${escapeHtml(card.label)}</strong>
      ${card.note ? `<span class="muted">${escapeHtml(card.note)}</span>` : ""}
      ${escapeHtml(money(card.usd_per_second))}/s
      ${card.usd_per_hour ? ` · ${escapeHtml(money(card.usd_per_hour))}/h` : ""}
    </span>`).join("")}</div>`;
}

function renderCostChain(input, caption = "Modal 调用链") {
  const events = flattenChain(input);
  if (!events.length) {
    return `<p class="muted">还没有计费记录。演练任务记 $0；真实生成会在这里挂上 <code>app.run</code> 和 <code>.remote</code>。</p>`;
  }
  const rows = events.map((event) => {
    const depth = Number(event.depth || 0);
    const prefix = depth ? `${"　".repeat(depth)}↳ ` : "";
    const job = event.job_id
      ? `<a href="#/job/${escapeHtml(event.job_id)}">${escapeHtml(event.job_id)}</a>`
      : "—";
    const parts = (event.breakdown_lines || []).map((line) => escapeHtml(line)).join("<br />");
    return `<tr>
      <td class="mono chain-name" style="padding-left:${0.4 + depth * 1.1}rem">${prefix}${escapeHtml(event.phase)}:${escapeHtml(event.name)}</td>
      <td class="mono">${escapeHtml(String(event.duration_s ?? (event.duration_ms / 1000).toFixed(3)))}s</td>
      <td class="mono">${escapeHtml(money(event.usd))}</td>
      <td class="mono">${escapeHtml(money(event.usd_per_second))}/s</td>
      <td class="mono breakdown">${parts || escapeHtml(event.line || "—")}</td>
      <td>${escapeHtml(event.gpu || "—")}</td>
      <td class="mono">${job}</td>
      <td class="mono">${escapeHtml(event.image_id || "—")}</td>
    </tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="cost-table">
    <caption>${escapeHtml(caption)}</caption>
    <thead><tr>
      <th scope="col">调用链</th>
      <th scope="col">时长</th>
      <th scope="col">费用</th>
      <th scope="col">每秒</th>
      <th scope="col">拆分</th>
      <th scope="col">GPU</th>
      <th scope="col">任务</th>
      <th scope="col">图片</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}

function renderJobRollup(jobs) {
  const rows = Object.values(jobs || {});
  if (!rows.length) return "";
  return `<div class="panel">
    <table>
      <caption>按任务汇总（session 与 remote 重叠时间只计一次）</caption>
      <thead><tr>
        <th scope="col">任务</th>
        <th scope="col">费用</th>
        <th scope="col">笔数</th>
      </tr></thead>
      <tbody>
        ${rows.map((job) => `
          <tr>
            <td class="mono"><a href="#/job/${escapeHtml(job.job_id)}">${escapeHtml(job.job_id)}</a></td>
            <td class="mono">${escapeHtml(money(job.billed_usd))}</td>
            <td>${escapeHtml(String(job.event_count))}</td>
          </tr>`).join("")}
      </tbody>
    </table>
  </div>`;
}

async function costPage(params) {
  const jobId = params.get("job") || "";
  document.title = "成本 · sdcpp-modal";
  main.innerHTML = `<p class="muted">读取账本…</p>`;
  try {
    const query = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
    const report = await api(`/api/cost${query}`);
    const billed = report.billed || {};
    main.innerHTML = `
      <header class="page-head">
        <h1>成本</h1>
        <p class="lede">每一笔 Modal <code>app.run</code> / <code>.remote</code> 的调用链，按秒计价，挂到对应任务。重叠的 session 与 remote 只计一次。</p>
      </header>
      <div class="cost-hero panel">
        <div>
          <p class="run-kicker">已入账</p>
          <p class="cost-total">${escapeHtml(billed.display || money(billed.usd || "0"))}</p>
          <p class="muted">${escapeHtml(String(billed.event_count || 0))} 笔 · ${escapeHtml(String(billed.duration_s || 0))}s</p>
        </div>
        <div>
          <p class="hint">费率来源 ${escapeHtml(report.rates?.source || "fallback")}。账本 <span class="mono">${escapeHtml(report.ledger_path || "")}</span></p>
          ${jobId ? `<p class="hint">只看任务 <a href="#/job/${escapeHtml(jobId)}">${escapeHtml(jobId)}</a> · <a href="#/cost">全部</a></p>` : ""}
        </div>
      </div>
      ${renderCostRates(report.rates)}
      <div class="panel">
        ${renderCostChain(report.traces, jobId ? `任务 ${jobId} 的调用链` : "全部调用链")}
      </div>
      ${jobId ? "" : renderJobRollup(report.jobs)}
    `;
  } catch (error) {
    main.innerHTML = `<p class="bad">${escapeHtml(error.message)}</p>`;
  }
}

function progressBox() {
  return `
    <div class="progress" id="progress">
      <label class="field-label" for="progress-bar">进度</label>
      <div id="progress-text">0 / 0</div>
      <progress id="progress-bar" max="100" value="0">0%</progress>
    </div>`;
}

function setProgress(completed, total, extra = "") {
  const text = document.getElementById("progress-text");
  const bar = document.getElementById("progress-bar");
  if (!text || !bar) return;
  text.textContent = `${completed} / ${total}${extra ? ` · ${extra}` : ""}`;
  bar.value = total ? Math.min(100, Math.round((completed / total) * 100)) : 0;
}

function listenJob(jobId) {
  if (state.source) state.source.close();
  state.source = new EventSource(`/api/jobs/${jobId}/events`);
  state.source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    const progress = event.payload || {};
    if (event.type === "job.snapshot" || event.type === "job.started") {
      setProgress(progress.completed_images || 0, progress.total_images || 0);
    }
    if (event.type === "image.completed" || event.type === "image.failed") {
      setProgress(progress.completed || 0, progress.total || 0, event.type === "image.failed" ? "失败" : "");
    }
    if (["job.completed", "job.failed", "job.cancelled"].includes(event.type)) {
      const status = STATUS[event.payload.status] || event.payload.status;
      setProgress(event.payload.completed_images || 0, event.payload.total_images || 0, status);
      state.source.close();
      if (event.type === "job.failed") showNotice(event.payload.error || "任务失败", "bad");
      if (event.type === "job.completed") {
        showNotice("生成完成，已写入本地画廊。", "ok");
        render("gallery");
      }
    }
  };
}

function applyRecipeDefaults(form, meta) {
  const recipeId = new FormData(form).get("recipe");
  const recipe = (meta.models || []).find((item) => item.id === recipeId);
  if (!recipe) return;
  form.width.value = recipe.width;
  form.height.value = recipe.height;
  form.steps.value = recipe.default_steps;
  form.cfg_scale.value = recipe.cfg_scale;
  if (form.gpu && recipe.default_gpu) form.gpu.value = recipe.default_gpu;
  const hint = document.getElementById("recipe-hint");
  if (hint) hint.textContent = recipe.hint_zh || recipe.hint || "";
  const prompt = form.querySelector("[name=prompt]");
  if (prompt && recipe.id === "ideogram4" && !prompt.value.trim()) {
    prompt.placeholder = '{"high_level_description":"一只毛茸茸的橘猫"}';
  }
}

function generatePage(meta) {
  const defaults = defaultsFrom(meta);
  document.title = "生成 · sdcpp-modal";
  main.innerHTML = `
    <header class="page-head">
      <h1>生成</h1>
      <p class="lede">写提示词，选配方。Ideogram 4 与 FLUX.2 Dev 默认走 RTX PRO 6000，其余默认 L40S。</p>
    </header>
    <div class="studio">
      <form class="sheet" id="gen-form" method="post" action="/api/jobs" autocomplete="off">
        <div class="field prompt-block">
          <label class="field-label" for="gen-prompt">提示词</label>
          <textarea id="gen-prompt" name="prompt" required placeholder="雨夜城市，电影感摄影"></textarea>
        </div>
        ${composerFields("gen", defaults, meta, "开始生成")}
        <span class="mono" id="job-id"></span>
        ${settingsFields("gen", defaults, meta)}
      </form>
      <aside class="panel" aria-label="这次请求">
        <p class="run-kicker">这次请求</p>
        <p class="hint" id="recipe-hint"></p>
        <p class="hint mono" id="will-apply"></p>
        ${progressBox()}
      </aside>
    </div>
  `;
  const form = document.getElementById("gen-form");
  const refresh = () => {
    const payload = formPayload(form);
    const recipe = (meta.models || []).find((item) => item.id === payload.recipe);
    const gpu = (meta.gpus || []).find((item) => item.id === payload.gpu);
    document.getElementById("will-apply").textContent =
      `${gpuOptionLabel(gpu || { id: payload.gpu })}  ·  ${payload.recipe}  ·  ${payload.width}×${payload.height}  ·  ${payload.steps} 步  ·  ${payload.count} 张`;
    document.getElementById("recipe-hint").textContent = recipe?.hint_zh || recipe?.hint || "";
  };
  form.addEventListener("change", (event) => {
    if (event.target.name === "recipe") applyRecipeDefaults(form, meta);
    refresh();
  });
  form.addEventListener("input", refresh);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const payload = formPayload(event.target);
      payload.prompt = new FormData(event.target).get("prompt");
      const job = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      document.getElementById("job-id").textContent = job.id;
      setProgress(0, job.total_images);
      showNotice(`任务已排队：${job.id}`);
      listenJob(job.id);
    } catch (error) {
      showNotice(error.message, "bad");
    } finally {
      button.disabled = false;
    }
  });
  applyRecipeDefaults(form, meta);
  refresh();
}

function batchPage(meta) {
  const defaults = defaultsFrom(meta);
  document.title = "批量 · sdcpp-modal";
  main.innerHTML = `
    <header class="page-head">
      <h1>批量</h1>
      <p class="lede">每行一条提示词，或放下一个 txt。张数会对每一行生效。</p>
    </header>
    <form class="sheet" id="batch-form" method="post" action="/api/jobs" enctype="multipart/form-data" autocomplete="off">
      <div class="drop" id="drop">
        <label class="field-label" for="batch-file">提示词文件</label>
        <span>把 prompts.txt 拖到这里，或选择文件</span>
        <input id="batch-file" type="file" name="file" accept=".txt,.jsonl" />
      </div>
      <div class="field">
        <label class="field-label" for="batch-text">或直接粘贴</label>
        <textarea id="batch-text" name="text" placeholder="一座美丽的森林&#10;一座未来都市"></textarea>
      </div>
      ${composerFields("batch", defaults, meta, "开始批量")}
      <span class="mono" id="job-id"></span>
      ${settingsFields("batch", defaults, meta)}
      <p class="hint" id="recipe-hint"></p>
      ${progressBox()}
    </form>
  `;
  const form = document.getElementById("batch-form");
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("batch-file");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    if (event.dataTransfer.files[0]) fileInput.files = event.dataTransfer.files;
  });
  form.addEventListener("change", (event) => {
    if (event.target.name === "recipe") applyRecipeDefaults(form, meta);
  });
  applyRecipeDefaults(form, meta);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const payload = formPayload(event.target);
      const file = fileInput.files[0];
      let job;
      if (file) {
        const body = new FormData();
        body.append("file", file);
        const query = new URLSearchParams({
          recipe: payload.recipe,
          gpu: payload.gpu,
          count: String(payload.count),
          dry_run: String(payload.dry_run),
        });
        if (payload.width) query.set("width", String(payload.width));
        if (payload.height) query.set("height", String(payload.height));
        if (payload.steps) query.set("steps", String(payload.steps));
        if (payload.cfg_scale != null) query.set("cfg_scale", String(payload.cfg_scale));
        if (payload.seed != null) query.set("seed", String(payload.seed));
        job = await api(`/api/jobs/from-file?${query}`, { method: "POST", body });
      } else {
        payload.text = new FormData(event.target).get("text");
        job = await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      document.getElementById("job-id").textContent = job.id;
      setProgress(0, job.total_images);
      showNotice(`批量任务已排队：${job.id}`);
      listenJob(job.id);
    } catch (error) {
      showNotice(error.message, "bad");
    } finally {
      button.disabled = false;
    }
  });
}

async function jobsPage() {
  const jobs = await api("/api/jobs");
  document.title = "任务 · sdcpp-modal";
  main.innerHTML = `
    <header class="page-head">
      <h1>任务</h1>
      <p class="lede">每一次生成或批量都是一条任务。费用来自挂在该任务上的 Modal 调用链。</p>
    </header>
    <div class="panel">
      <table>
        <caption>本地任务</caption>
        <thead>
          <tr>
            <th scope="col">编号</th>
            <th scope="col">状态</th>
            <th scope="col">图片</th>
            <th scope="col">配方</th>
            <th scope="col">显卡</th>
            <th scope="col">费用</th>
            <th scope="col">操作</th>
          </tr>
        </thead>
        <tbody>
          ${jobs.map((job) => `
            <tr>
              <td class="mono"><a href="#/job/${escapeHtml(job.id)}">${escapeHtml(job.id)}</a></td>
              <td><span class="pill ${escapeHtml(job.status)}">${STATUS[job.status] || escapeHtml(job.status)}</span></td>
              <td>${job.completed_images}/${job.total_images}</td>
              <td>${escapeHtml(job.recipe)}</td>
              <td>${escapeHtml(job.gpu)}</td>
              <td class="mono"><a href="#/cost?job=${escapeHtml(job.id)}">${escapeHtml(jobCostLabel(job))}</a></td>
              <td>
                <button type="button" class="ghost" data-gallery="${escapeHtml(job.id)}">画廊</button>
                <button type="button" class="ghost" data-resume="${escapeHtml(job.id)}">续跑</button>
              </td>
            </tr>`).join("") || `<tr><td colspan="7">还没有任务。先去生成一页。</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
  main.querySelectorAll("[data-gallery]").forEach((button) => {
    button.addEventListener("click", () => {
      location.hash = `#/gallery?job=${button.dataset.gallery}`;
    });
  });
  main.querySelectorAll("[data-resume]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/api/jobs/${button.dataset.resume}/resume`, { method: "POST" });
        showNotice("已续跑未完成的帧。");
        render("jobs");
      } catch (error) {
        showNotice(error.message, "bad");
      }
    });
  });
}

async function jobDetailPage(jobId) {
  const detail = await api(`/api/jobs/${jobId}`);
  const job = detail.job;
  document.title = `任务 ${job.id} · sdcpp-modal`;
  main.innerHTML = `
    <header class="page-head">
      <h1>任务详情</h1>
    </header>
    <div class="panel">
      <div class="row"><span>编号</span><span class="mono">${escapeHtml(job.id)}</span></div>
      <div class="row"><span>状态</span><span class="pill ${escapeHtml(job.status)}">${STATUS[job.status] || escapeHtml(job.status)}</span></div>
      <div class="row"><span>配方 / 显卡</span><span>${escapeHtml(job.recipe)} · ${escapeHtml(job.gpu)}</span></div>
      <div class="row"><span>图片</span><span>${job.completed_images}/${job.total_images}</span></div>
      <div class="row"><span>费用</span><span class="mono">${escapeHtml(jobCostLabel(job))} · ${escapeHtml(String(job.cost_events || 0))} 笔</span></div>
      <div class="actions" style="margin-block-start:1rem">
        <button type="button" class="ghost" id="to-gallery">查看画廊</button>
        <button type="button" class="ghost" id="to-jobs">全部任务</button>
        <button type="button" class="ghost" id="to-cost">调用链</button>
      </div>
    </div>
    <h2 class="section">调用链</h2>
    <div class="panel">
      ${renderCostChain(job.cost_chain, "该任务的 Modal / 本地计费")}
    </div>
    <h2 class="section">帧</h2>
    <div class="panel">
      <table>
        <caption>该任务生成的帧</caption>
        <thead>
          <tr>
            <th scope="col">编号</th>
            <th scope="col">状态</th>
            <th scope="col">种子</th>
            <th scope="col">尺寸</th>
            <th scope="col">耗时</th>
            <th scope="col">费用</th>
          </tr>
        </thead>
        <tbody>
          ${(detail.images || []).map((item) => `
            <tr>
              <td class="mono">${escapeHtml(item.id)}</td>
              <td><span class="pill ${escapeHtml(item.status)}">${STATUS[item.status] || escapeHtml(item.status)}</span></td>
              <td class="mono">${escapeHtml(item.seed)}</td>
              <td>${item.width}×${item.height}</td>
              <td>${item.duration_ms != null ? (item.duration_ms / 1000).toFixed(2) + " 秒" : "—"}</td>
              <td class="mono">${item.cost_usd != null ? `${escapeHtml(money(item.cost_usd))}${item.usd_per_second ? ` · ${escapeHtml(money(item.usd_per_second))}/s` : ""}` : "—"}</td>
            </tr>`).join("") || `<tr><td colspan="6">还没有帧。</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
  document.getElementById("to-gallery").addEventListener("click", () => {
    location.hash = `#/gallery?job=${job.id}`;
  });
  document.getElementById("to-jobs").addEventListener("click", () => {
    location.hash = "#/jobs";
  });
  document.getElementById("to-cost").addEventListener("click", () => {
    location.hash = `#/cost?job=${job.id}`;
  });
}

async function galleryPage(params) {
  const page = Number(params.get("page") || 1);
  const per = Number(params.get("per") || 50);
  const job = params.get("job") || "";
  const q = params.get("q") || "";
  const recipe = params.get("recipe") || "";
  const query = new URLSearchParams({ page, per_page: per, sort: params.get("sort") || "newest" });
  if (job) query.set("job_id", job);
  if (q) query.set("q", q);
  if (recipe) query.set("recipe", recipe);
  const data = await api(`/api/gallery?${query}`);
  document.title = "画廊 · sdcpp-modal";
  main.innerHTML = `
    <header class="page-head">
      <h1>画廊</h1>
      <p class="lede">本地已有 ${data.total} 张图。点开卡片可看提示词和种子。</p>
    </header>
    <form class="toolbar" id="filters" method="get" action="#/gallery">
      <div class="field">
        <label class="field-label" for="filter-q">搜索提示词</label>
        <input id="filter-q" name="q" value="${escapeHtml(q)}" />
      </div>
      <div class="field">
        <label class="field-label" for="filter-job">任务编号</label>
        <input id="filter-job" name="job" value="${escapeHtml(job)}" />
      </div>
      <div class="field">
        <label class="field-label" for="filter-recipe">配方</label>
        <input id="filter-recipe" name="recipe" value="${escapeHtml(recipe)}" />
      </div>
      <button type="submit" class="ghost">筛选</button>
    </form>
    <div class="gallery">
      ${data.items.map((image) => `
        <article>
          <button type="button" class="shot" data-id="${escapeHtml(image.id)}">
            <img src="/api/images/${escapeHtml(image.id)}/file" alt="${escapeHtml(image.prompt)}" width="${image.width || 512}" height="${image.height || 1024}" />
            <span class="cap">${escapeHtml(image.prompt)}</span>
          </button>
        </article>`).join("") || `<p class="empty">还没有图片。<a href="#/generate">去生成</a></p>`}
    </div>
    <nav class="pager" aria-label="画廊分页">
      <button type="button" class="ghost" ${page <= 1 ? "disabled" : ""} id="prev">上一页</button>
      <span>第 ${data.page} 页</span>
      <button type="button" class="ghost" ${page * per >= data.total ? "disabled" : ""} id="next">下一页</button>
    </nav>
  `;
  document.getElementById("filters").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    location.hash = `#/gallery?${new URLSearchParams({
      q: form.get("q") || "",
      job: form.get("job") || "",
      recipe: form.get("recipe") || "",
      page: "1",
    })}`;
  });
  document.getElementById("prev").addEventListener("click", () => {
    params.set("page", String(page - 1));
    location.hash = `#/gallery?${params}`;
  });
  document.getElementById("next").addEventListener("click", () => {
    params.set("page", String(page + 1));
    location.hash = `#/gallery?${params}`;
  });
  main.querySelectorAll(".shot").forEach((card) => {
    card.addEventListener("click", () => openLightbox(card.dataset.id));
  });
}

async function openLightbox(imageId) {
  const image = await api(`/api/images/${imageId}`);
  lightbox.innerHTML = `
    <div class="lightbox-grid">
      <img src="/api/images/${escapeHtml(image.id)}/file" alt="${escapeHtml(image.prompt)}" width="${image.width || 512}" height="${image.height || 1024}" />
      <div class="meta">
        <h2 id="lightbox-title">帧</h2>
        <p>${escapeHtml(image.prompt)}</p>
        <dl>
          <dt>种子</dt><dd class="mono">${escapeHtml(image.seed)}</dd>
          <dt>配方</dt><dd>${escapeHtml(image.recipe)}</dd>
          <dt>步数</dt><dd>${escapeHtml(image.steps)}</dd>
          <dt>尺寸</dt><dd>${image.width} × ${image.height}</dd>
          <dt>耗时</dt><dd>${image.latency_ms ? (image.latency_ms / 1000).toFixed(2) + " 秒" : "—"}</dd>
          <dt>任务</dt><dd class="mono"><a href="#/job/${escapeHtml(image.job_id)}">${escapeHtml(image.job_id)}</a></dd>
        </dl>
        <div class="actions" style="margin-block-start:1rem">
          <button type="button" id="copy">复制提示词</button>
          <button type="button" class="ghost" id="regen">再生成</button>
          <a class="btn ghost" href="/api/images/${escapeHtml(image.id)}/file" download>下载</a>
          <button type="button" class="ghost" id="close" value="close">关闭</button>
        </div>
      </div>
    </div>
  `;
  if (typeof lightbox.showModal === "function") lightbox.showModal();
  document.getElementById("close").addEventListener("click", () => lightbox.close());
  document.getElementById("copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(image.prompt);
    showNotice("提示词已复制。");
  });
  document.getElementById("regen").addEventListener("click", async () => {
    try {
      const job = await api(`/api/images/${image.id}/regenerate`, { method: "POST" });
      lightbox.close();
      location.hash = "#/jobs";
      showNotice(`已排队 ${job.id}`);
    } catch (error) {
      showNotice(error.message, "bad");
    }
  });
}

async function settingsPage(meta) {
  const doctor = await api("/api/doctor");
  document.title = "设置 · sdcpp-modal";
  main.innerHTML = `
    <header class="page-head">
      <h1>设置</h1>
      <p class="lede">本地工作台。权重留在卷 <code>sdcpp-models</code>。Ideogram 4 与 FLUX.2 Dev 默认 <code>RTX-PRO-6000</code>，其余默认 L40S。A10 与 A100 已禁用。</p>
    </header>
    <div class="panel">
      <div class="row"><span>数据目录</span><span class="mono">${escapeHtml(meta.defaults.data_dir)}</span></div>
      <div class="row"><span>默认配方</span><span>${escapeHtml(meta.defaults.recipe)}</span></div>
      <div class="row"><span>默认显卡</span><span>${escapeHtml(meta.defaults.gpu)}</span></div>
      <div class="row"><span>成本账本</span><span class="mono">${escapeHtml(meta.defaults.cost_log || "")}</span></div>
      <h2 class="section">配方</h2>
      ${(meta.models || []).map((model) => `
        <div class="row">
          <span>${escapeHtml(model.label_zh || model.label)}</span>
          <span class="mono">${escapeHtml(model.id)} · ${model.width}×${model.height} · ${escapeHtml(model.default_gpu)}</span>
        </div>`).join("")}
      <h2 class="section">自检</h2>
      <div>
        ${doctor.checks.map((check) => `
          <div class="row">
            <span>${DOCTOR[check.name] || escapeHtml(check.name)}</span>
            <span class="${check.ok ? "ok" : "bad"}">${check.ok ? "通过" : "失败"} · ${escapeHtml(check.detail)}</span>
          </div>`).join("")}
      </div>
    </div>
  `;
}

async function render(forced) {
  const hash = location.hash.replace(/^#\/?/, "") || "generate";
  const [pageName, query] = hash.split("?");
  const page = forced || pageName || "generate";
  const params = new URLSearchParams(query || "");
  document.querySelectorAll("nav a").forEach((link) => {
    const current = link.dataset.page === page
      || (page.startsWith("job/") && link.dataset.page === "jobs")
      || (page === "cost" && link.dataset.page === "cost");
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  main.classList.toggle("is-wide", page === "cost" || page.startsWith("job/"));
  if (!state.meta) state.meta = await api("/api/meta");
  if (page === "generate") generatePage(state.meta);
  else if (page === "batch") batchPage(state.meta);
  else if (page === "jobs") await jobsPage();
  else if (page.startsWith("job/")) await jobDetailPage(page.slice(4));
  else if (page === "cost") await costPage(params);
  else if (page === "gallery") await galleryPage(params);
  else if (page === "settings") await settingsPage(state.meta);
  else generatePage(state.meta);
}

window.addEventListener("hashchange", () => render());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

(async () => {
  try {
    const [doctor, meta] = await Promise.all([api("/api/doctor"), api("/api/meta")]);
    state.meta = meta;
    railStatus.textContent = doctor.ready ? "本地就绪" : "设置未完成";
    railStatus.className = doctor.ready ? "rail-foot ok" : "rail-foot bad";
  } catch {
    railStatus.textContent = "接口离线";
  }
  await render();
})();
