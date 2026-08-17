import { $, $$, api, escapeHtml, gpuById, modelById, pageTitle, refreshRuntime, state, toast, workspace } from "./ux-core.js";

function modelOptions(selected) {
  return (state.meta?.models || []).map((model, index) => `
    <button type="button" class="model-option" data-model="${escapeHtml(model.id)}" aria-pressed="${model.id === selected}">
      ${index === 0 ? '<span class="recommended" title="默认推荐"></span>' : ""}
      <span class="model-name">${escapeHtml(model.label_zh || model.label || model.name)}</span>
      <span class="model-hint">${escapeHtml(model.hint_zh || model.hint || "")}</span>
      <span class="model-meta"><span>${escapeHtml(`${model.width}×${model.height}`)}</span><span>${escapeHtml(`${model.default_steps} steps`)}</span></span>
    </button>`).join("");
}

function gpuOptions(selected) {
  return (state.meta?.gpus || []).map((gpu) => `<option value="${escapeHtml(gpu.id)}" ${gpu.id === selected ? "selected" : ""}>${escapeHtml(gpu.label || gpu.id)} · ${gpu.vram_gb}GB · $${Number(gpu.usd_per_hour || 0).toFixed(2)}/h</option>`).join("");
}

function createSummary(mode) {
  const model = modelById(state.createDraft.recipe);
  const gpu = gpuById(state.createDraft.gpu);
  const parallelism = mode === "batch" ? state.createDraft.parallelism : 1;
  return `<span>${escapeHtml(model.label_zh || model.label || model.id || "模型")}</span><span>${escapeHtml(gpu.label || gpu.id || "GPU")} · ${parallelism > 1 ? `并行 ×${parallelism}` : "串行"}</span>`;
}

export function renderCreate() {
  const meta = state.meta;
  const defaults = meta.defaults || {};
  if (!state.createDraft.recipe) state.createDraft.recipe = defaults.recipe || meta.models?.[0]?.id;
  const model = modelById(state.createDraft.recipe);
  if (!state.createDraft.gpu) state.createDraft.gpu = model.default_gpu || defaults.gpu;
  const mode = state.createMode;
  const isBatch = mode === "batch";

  workspace.innerHTML = `${pageTitle(
    "CREATE",
    "把提示词变成结果",
    isBatch ? "一次导入几十或几百条提示词。模型只准备一次，按你选择的速度执行。" : "写下你想生成的画面，选择模型，然后开始。高级参数默认收起来。",
    `<div class="segmented" aria-label="创建模式"><button type="button" data-mode="single" aria-pressed="${!isBatch}">单张</button><button type="button" data-mode="batch" aria-pressed="${isBatch}">批量</button></div>`
  )}
  <form id="create-form" class="create-layout" autocomplete="off">
    <section class="create-main">
      ${isBatch ? `<label class="batch-drop" id="batch-drop"><input id="prompt-file" name="file" type="file" accept=".txt,.jsonl,text/plain" /><span class="drop-copy"><strong>拖入 prompts.txt / jsonl</strong><small id="file-copy">也可以点这里选择文件</small></span></label>` : ""}
      <div class="prompt-label"><strong>${isBatch ? "提示词列表" : "Prompt"}</strong><span>${isBatch ? "每行一条" : "支持自然语言"}</span></div>
      <textarea id="prompt-input" class="prompt-editor" name="prompt" placeholder="${isBatch ? "一座雨夜中的未来城市\n极简产品摄影，白色背景\n穿宇航服的人站在红色沙漠中" : "例如：一座雨夜中的未来东京，电影感，霓虹反射，35mm 摄影"}"></textarea>
      <div class="editor-meta"><span>${isBatch ? "空行会自动忽略" : "Ctrl / ⌘ + Enter 快速生成"}</span><span id="line-count"></span></div>
      <section class="model-section"><div class="section-heading"><div><p class="eyebrow">MODEL</p><h2>选择模型</h2></div><p>选模型，不选底层参数堆栈</p></div><div class="model-grid">${modelOptions(state.createDraft.recipe)}</div></section>
    </section>
    <aside class="create-sidebar"><div class="config-panel">
      <section class="config-block"><div class="config-title"><h3>运行方式</h3><span>${isBatch ? "批量任务" : "单次任务"}</span></div><div class="field"><label for="gpu-select">GPU</label><select id="gpu-select" class="input" name="gpu">${gpuOptions(state.createDraft.gpu)}</select></div><div class="field"><label for="count-input">${isBatch ? "每条生成张数" : "生成张数"}</label><input id="count-input" class="input" name="count" type="number" min="1" max="100" value="1" /></div></section>
      ${isBatch ? `<section class="config-block"><div class="config-title"><h3>生成速度</h3><span>会影响 GPU 数量</span></div><div class="speed-options"><button type="button" class="speed-option" data-parallelism="1" aria-pressed="${state.createDraft.parallelism === 1}"><span class="speed-icon">1×</span><span class="speed-copy"><strong>省钱</strong><small>严格串行，最低额外开销</small></span><small>1 GPU</small></button><button type="button" class="speed-option" data-parallelism="2" aria-pressed="${state.createDraft.parallelism === 2}"><span class="speed-icon">2×</span><span class="speed-copy"><strong>平衡</strong><small>适合中等批量</small></span><small>≤2 GPU</small></button><button type="button" class="speed-option" data-parallelism="4" aria-pressed="${state.createDraft.parallelism === 4}"><span class="speed-icon">4×</span><span class="speed-copy"><strong>最快</strong><small>用更多 GPU 换吞吐</small></span><small>≤4 GPU</small></button></div></section>` : ""}
      <section class="config-block"><button class="advanced-toggle" type="button" id="advanced-toggle" aria-expanded="false"><span>高级设置</span><span>＋</span></button><div id="advanced-fields" hidden><div class="advanced-grid"><div class="field"><label for="width-input">宽度</label><input id="width-input" class="input" name="width" type="number" min="64" step="64" value="${model.width || 512}" /></div><div class="field"><label for="height-input">高度</label><input id="height-input" class="input" name="height" type="number" min="64" step="64" value="${model.height || 512}" /></div><div class="field"><label for="steps-input">Steps</label><input id="steps-input" class="input" name="steps" type="number" min="1" value="${model.default_steps || 20}" /></div><div class="field"><label for="cfg-input">CFG</label><input id="cfg-input" class="input" name="cfg_scale" type="number" min="0" step="0.1" value="${model.cfg_scale ?? 1}" /></div></div><div class="field"><label for="seed-input">Seed</label><input id="seed-input" class="input" name="seed" type="number" placeholder="留空使用默认种子" /></div><label class="check-row"><input name="dry_run" type="checkbox" /> 演练模式（不调用 Modal / GPU）</label></div></section>
      <section class="submit-block"><button id="submit-button" class="primary-button" type="submit">${isBatch ? "开始批量生成" : "生成图片"}</button><div class="submit-summary" id="submit-summary">${createSummary(mode)}</div></section>
    </div></aside>
  </form>`;
  bindCreateEvents();
}

function applyModelDefaults(modelId) {
  const model = modelById(modelId);
  state.createDraft.recipe = modelId;
  state.createDraft.gpu = model.default_gpu || state.meta.defaults.gpu;
  const gpu = $("#gpu-select");
  if (gpu) gpu.value = state.createDraft.gpu;
  const map = { "#width-input": model.width, "#height-input": model.height, "#steps-input": model.default_steps, "#cfg-input": model.cfg_scale };
  Object.entries(map).forEach(([selector, value]) => { const input = $(selector); if (input && value != null) input.value = value; });
  $$('[data-model]').forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.model === modelId)));
  $("#submit-summary").innerHTML = createSummary(state.createMode);
}

function bindCreateEvents() {
  $$('[data-mode]').forEach((button) => button.addEventListener("click", () => { state.createMode = button.dataset.mode; localStorage.setItem("sdcpp:create-mode", state.createMode); renderCreate(); }));
  $$('[data-model]').forEach((button) => button.addEventListener("click", () => applyModelDefaults(button.dataset.model)));
  $$('[data-parallelism]').forEach((button) => button.addEventListener("click", () => { state.createDraft.parallelism = Number(button.dataset.parallelism); $$('[data-parallelism]').forEach((item) => item.setAttribute("aria-pressed", String(item === button))); $("#submit-summary").innerHTML = createSummary(state.createMode); }));
  $("#gpu-select")?.addEventListener("change", (event) => { state.createDraft.gpu = event.target.value; $("#submit-summary").innerHTML = createSummary(state.createMode); });
  $("#advanced-toggle")?.addEventListener("click", (event) => { const fields = $("#advanced-fields"); const open = fields.hidden; fields.hidden = !open; event.currentTarget.setAttribute("aria-expanded", String(open)); event.currentTarget.lastElementChild.textContent = open ? "−" : "＋"; });
  const prompt = $("#prompt-input");
  prompt?.addEventListener("input", () => { if (state.createMode === "batch") { const lines = prompt.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean); $("#line-count").textContent = lines.length ? `${lines.length} 条 Prompt` : ""; } else { $("#line-count").textContent = prompt.value ? `${prompt.value.length} 字符` : ""; } });
  prompt?.addEventListener("keydown", (event) => { if (state.createMode === "single" && event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $("#create-form")?.requestSubmit(); } });
  const file = $("#prompt-file");
  file?.addEventListener("change", () => { const selected = file.files?.[0]; $("#file-copy").textContent = selected ? `${selected.name} · ${(selected.size / 1024).toFixed(1)} KB` : "也可以点这里选择文件"; if (selected) selected.text().then((text) => { const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean); $("#line-count").textContent = `${lines.length} 条 Prompt`; }).catch(() => {}); });
  const drop = $("#batch-drop");
  ["dragenter", "dragover"].forEach((name) => drop?.addEventListener(name, () => drop.classList.add("is-over")));
  ["dragleave", "drop"].forEach((name) => drop?.addEventListener(name, () => drop.classList.remove("is-over")));
  $("#create-form")?.addEventListener("submit", submitCreate);
}

async function submitCreate(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const button = $("#submit-button");
  const number = (name) => { const raw = data.get(name); return raw === "" || raw == null ? null : Number(raw); };
  const base = { recipe: state.createDraft.recipe, gpu: state.createDraft.gpu, count: Math.max(1, number("count") || 1), width: number("width"), height: number("height"), steps: number("steps"), cfg_scale: number("cfg_scale"), seed: number("seed"), parallelism: state.createMode === "batch" ? state.createDraft.parallelism : 1, dry_run: data.get("dry_run") === "on" };
  button.disabled = true;
  button.textContent = "正在创建任务…";
  try {
    let job;
    const file = $("#prompt-file")?.files?.[0];
    if (state.createMode === "batch" && file) {
      const body = new FormData(); body.append("file", file);
      const query = new URLSearchParams(); Object.entries(base).forEach(([key, value]) => { if (value != null) query.set(key, String(value)); });
      job = await api(`/api/jobs/from-file?${query}`, { method: "POST", body });
    } else {
      const text = $("#prompt-input").value.trim();
      if (!text) throw new Error(state.createMode === "batch" ? "请粘贴提示词或选择文件" : "请输入 Prompt");
      const payload = state.createMode === "batch" ? { ...base, text } : { ...base, prompt: text };
      job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    }
    toast(`任务已创建：${job.id}`);
    await refreshRuntime();
    location.hash = `#/runs/${job.id}`;
  } catch (error) {
    toast(error.message, "bad");
    button.disabled = false;
    button.textContent = state.createMode === "batch" ? "开始批量生成" : "生成图片";
  }
}
