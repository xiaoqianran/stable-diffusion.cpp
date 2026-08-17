import { $, $$, PHASE, api, escapeHtml, modelById, money, pageTitle, phaseClass, phaseText, progress, refreshRuntime, state, timeAgo, toast, workspace } from "./ux-core.js";
import { openImage } from "./ux-gallery.js";

function runHero(job) {
  const p = progress(job);
  const model = modelById(job.recipe);
  const parallel = Number(job.config?.parallelism || 1);
  return `<article class="run-hero"><div class="run-hero-grid"><div><div class="run-state ${phaseClass(job)}"><i></i><span>${escapeHtml(phaseText(job))}</span></div><h2>${escapeHtml(model.label_zh || model.label || job.recipe)}</h2><p class="run-sub">${job.total_images} 张图片 · ${escapeHtml(job.gpu)}${parallel > 1 ? ` · 并行 ×${parallel}` : " · 串行"}</p></div><div class="run-stat"><strong>${p.done}<span class="muted">/${p.total}</span></strong><small>已处理</small></div></div><div class="progress-track" aria-label="${p.percent}%"><span style="width:${p.percent}%"></span></div><div class="run-inline-meta"><span><b>${p.percent}%</b> 完成</span><span>${escapeHtml(timeAgo(job.updated_at))}</span><span>${job.queue?.ahead ? `前方 ${job.queue.ahead} 个任务` : ""}</span><span>${job.cost_usd != null ? `当前成本 ${money(job.cost_usd)}` : ""}</span></div></article>`;
}

function runRow(job) {
  const model = modelById(job.recipe);
  const p = progress(job);
  return `<a class="run-row" href="#/runs/${encodeURIComponent(job.id)}"><div class="run-row-title"><strong>${escapeHtml(model.label_zh || model.label || job.recipe)}</strong><small>${escapeHtml(job.id)} · ${escapeHtml(timeAgo(job.created_at))}</small></div><div class="run-row-state"><strong>${escapeHtml(phaseText(job))}</strong><span>${escapeHtml(job.gpu)} · ×${Number(job.config?.parallelism || 1)}</span></div><div class="run-row-progress">${p.done} / ${p.total} · ${p.percent}%</div><div class="run-row-cost">${money(job.cost_usd)}</div><div class="chevron">›</div></a>`;
}

export function renderRuns() {
  const jobs = state.jobs || [];
  const active = jobs.find((job) => !["completed", "failed", "cancelled"].includes(job.status));
  const rest = active ? jobs.filter((job) => job.id !== active.id) : jobs;
  workspace.innerHTML = `${pageTitle("RUNS", "运行", "只关注正在发生什么、还要多久，以及结果在哪里。技术细节需要时再展开。")}<div class="runs-layout">${active ? `<a href="#/runs/${encodeURIComponent(active.id)}">${runHero(active)}</a>` : ""}${jobs.length ? `<section><div class="run-section-title"><h2>${active ? "其他任务" : "最近任务"}</h2><span>${jobs.length} 个任务</span></div><div class="run-list">${rest.map(runRow).join("")}</div></section>` : `<div class="empty-state"><div><strong>还没有运行记录</strong><p>从创建页面提交第一条 Prompt。</p><a class="text-button" href="#/create">开始创建 →</a></div></div>`}</div>`;
}

export async function renderRunDetail(jobId) {
  workspace.innerHTML = `<div class="empty-state"><div><strong>正在读取任务…</strong></div></div>`;
  try {
    const [detail, gallery] = await Promise.all([api(`/api/jobs/${encodeURIComponent(jobId)}`), api(`/api/gallery?job_id=${encodeURIComponent(jobId)}&per_page=8&page=1`)]);
    const job = detail.job;
    const p = progress(job);
    const model = modelById(job.recipe);
    const canCancel = !["completed", "failed", "cancelled"].includes(job.status);
    const canResume = ["failed", "cancelled"].includes(job.status) || job.failed_images > 0;
    workspace.innerHTML = `<div class="detail-head"><div><a class="back-link" href="#/runs">← 所有运行</a><p class="eyebrow">RUN DETAIL</p><h1>${escapeHtml(model.label_zh || model.label || job.recipe)}</h1></div><div class="detail-actions"><button class="secondary-button" type="button" data-gallery-job="${escapeHtml(job.id)}">查看图库</button>${canResume ? `<button class="secondary-button" type="button" data-resume="${escapeHtml(job.id)}">继续未完成任务</button>` : ""}${canCancel ? `<button class="danger-button" type="button" data-cancel="${escapeHtml(job.id)}">停止</button>` : ""}</div></div><div class="detail-grid"><section class="detail-panel">${runHero(job)}<div class="run-section-title"><h2>结果</h2><span>${gallery.total || 0} 张已完成</span></div>${gallery.items?.length ? `<div class="result-strip">${gallery.items.slice(0, 8).map((image) => `<button class="result-thumb" type="button" data-image="${escapeHtml(image.id)}"><img src="/api/images/${encodeURIComponent(image.id)}/file" alt="${escapeHtml(image.prompt)}" loading="lazy" /></button>`).join("")}</div>` : `<p class="muted">图片生成完成后会出现在这里。</p>`}<details class="technical-details"><summary>图片明细（${detail.images?.length || 0}）</summary><div class="fact-list">${(detail.images || []).slice(0, 40).map((item) => `<div class="fact"><span>${escapeHtml(item.prompt?.slice(0, 54) || item.id)}</span><b>${escapeHtml(PHASE[item.status] || item.status)}${item.duration_ms ? ` · ${(item.duration_ms / 1000).toFixed(1)}s` : ""}</b></div>`).join("")}</div></details></section><aside><div class="run-section-title"><h2>摘要</h2></div><div class="fact-list"><div class="fact"><span>状态</span><b>${escapeHtml(phaseText(job))}</b></div><div class="fact"><span>模型</span><b>${escapeHtml(job.recipe)}</b></div><div class="fact"><span>GPU</span><b>${escapeHtml(job.gpu)}</b></div><div class="fact"><span>并行</span><b>×${Number(job.config?.parallelism || 1)}</b></div><div class="fact"><span>图片</span><b>${p.done} / ${p.total}</b></div><div class="fact"><span>成本</span><b>${money(job.cost_usd)}</b></div><div class="fact"><span>创建</span><b>${escapeHtml(new Date(job.created_at).toLocaleString())}</b></div><div class="fact"><span>任务 ID</span><b class="mono">${escapeHtml(job.id)}</b></div></div><details class="technical-details"><summary>技术参数</summary><div class="fact-list"><div class="fact"><span>尺寸</span><b>${job.config?.width || "—"} × ${job.config?.height || "—"}</b></div><div class="fact"><span>Steps</span><b>${job.config?.steps ?? "—"}</b></div><div class="fact"><span>CFG</span><b>${job.config?.cfg_scale ?? "—"}</b></div><div class="fact"><span>Seed</span><b>${job.config?.seed ?? "—"}</b></div><div class="fact"><span>队列 affinity</span><b>${escapeHtml(job.queue?.affinity_key || "—")}</b></div></div></details></aside></div>`;
    $$('[data-image]').forEach((button) => button.addEventListener("click", () => openImage(button.dataset.image)));
    $('[data-gallery-job]')?.addEventListener("click", () => { location.hash = `#/gallery?job=${encodeURIComponent(job.id)}`; });
    $('[data-cancel]')?.addEventListener("click", async () => { try { await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" }); toast("任务已停止"); await refreshRuntime(); renderRunDetail(job.id); } catch (error) { toast(error.message, "bad"); } });
    $('[data-resume]')?.addEventListener("click", async () => { try { await api(`/api/jobs/${encodeURIComponent(job.id)}/resume`, { method: "POST" }); toast("已继续未完成任务"); await refreshRuntime(); renderRunDetail(job.id); } catch (error) { toast(error.message, "bad"); } });
  } catch (error) {
    workspace.innerHTML = `<div class="empty-state"><div><strong>任务不存在或读取失败</strong><p>${escapeHtml(error.message)}</p><a class="text-button" href="#/runs">返回运行列表</a></div></div>`;
  }
}
