import { $, $$, api, escapeHtml, imageDialog, pageTitle, state, toast, workspace } from "./ux-core.js";

export async function renderGallery(route) {
  const g = state.gallery;
  if (!route.params.toString()) {
    Object.assign(g, { page: 1, q: "", recipe: "", sort: "newest", job: "" });
  }
  if (route.params.has("job")) g.job = route.params.get("job"); else if (!route.params.has("page")) g.job = "";
  if (route.params.has("page")) g.page = Math.max(1, Number(route.params.get("page")) || 1);
  if (route.params.has("q")) g.q = route.params.get("q") || "";
  if (route.params.has("recipe")) g.recipe = route.params.get("recipe") || "";
  if (route.params.has("sort")) g.sort = route.params.get("sort") || "newest";
  if (route.params.has("per")) g.perPage = [50,100,200].includes(Number(route.params.get("per"))) ? Number(route.params.get("per")) : g.perPage;
  workspace.innerHTML = `${pageTitle("GALLERY", "图库", "结果优先。搜索 Prompt、按模型筛选，点开图片查看完整参数。")}<div class="gallery-toolbar"><div class="search-field"><input id="gallery-search" placeholder="搜索 Prompt" value="${escapeHtml(g.q)}" /></div><select id="gallery-model" class="toolbar-select"><option value="">全部模型</option>${(state.meta?.models || []).map((model) => `<option value="${escapeHtml(model.id)}" ${model.id === g.recipe ? "selected" : ""}>${escapeHtml(model.label_zh || model.label)}</option>`).join("")}</select><select id="gallery-sort" class="toolbar-select"><option value="newest" ${g.sort === "newest" ? "selected" : ""}>最新</option><option value="oldest" ${g.sort === "oldest" ? "selected" : ""}>最早</option><option value="fastest" ${g.sort === "fastest" ? "selected" : ""}>最快</option><option value="slowest" ${g.sort === "slowest" ? "selected" : ""}>最慢</option></select><select id="gallery-per" class="toolbar-select">${[50,100,200].map((n) => `<option value="${n}" ${n === g.perPage ? "selected" : ""}>${n} / 页</option>`).join("")}</select><span class="gallery-count" id="gallery-count">读取中…</span></div><div id="gallery-grid" class="gallery-grid"></div><div id="gallery-pager" class="gallery-pager"></div>`;
  let searchTimer;
  $("#gallery-search").addEventListener("input", (event) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => updateGalleryRoute({ q: event.target.value, page: 1 }), 350); });
  $("#gallery-model").addEventListener("change", (event) => updateGalleryRoute({ recipe: event.target.value, page: 1 }));
  $("#gallery-sort").addEventListener("change", (event) => updateGalleryRoute({ sort: event.target.value, page: 1 }));
  $("#gallery-per").addEventListener("change", (event) => { localStorage.setItem("sdcpp:gallery-per-page", event.target.value); updateGalleryRoute({ per: event.target.value, page: 1 }); });
  await loadGallery();
}

function updateGalleryRoute(changes) {
  const g = state.gallery;
  const params = new URLSearchParams();
  const next = { q: g.q, recipe: g.recipe, sort: g.sort, per: g.perPage, page: g.page, job: g.job || "", ...changes };
  if (next.q) params.set("q", next.q);
  if (next.recipe) params.set("recipe", next.recipe);
  if (next.sort && next.sort !== "newest") params.set("sort", next.sort);
  if (Number(next.per) !== 50) params.set("per", next.per);
  if (Number(next.page) !== 1) params.set("page", next.page);
  if (next.job) params.set("job", next.job);
  location.hash = `#/gallery${params.toString() ? `?${params}` : ""}`;
}

async function loadGallery() {
  const g = state.gallery;
  const query = new URLSearchParams({ page: String(g.page), per_page: String(g.perPage), sort: g.sort });
  if (g.q) query.set("q", g.q);
  if (g.recipe) query.set("recipe", g.recipe);
  if (g.job) query.set("job_id", g.job);
  try {
    const data = await api(`/api/gallery?${query}`);
    $("#gallery-count").textContent = `${data.total} 张图片`;
    const grid = $("#gallery-grid");
    grid.innerHTML = data.items?.length ? data.items.map((image) => { const ratio = image.width && image.height ? `${image.width}/${image.height}` : "1"; return `<button class="image-card" type="button" data-image="${escapeHtml(image.id)}" style="--ratio:${ratio}"><img src="/api/images/${encodeURIComponent(image.id)}/file" alt="${escapeHtml(image.prompt)}" loading="lazy" /><span class="image-info"><strong>${escapeHtml(image.prompt)}</strong><small>${escapeHtml(image.recipe)} · seed ${escapeHtml(image.seed)}</small></span></button>`; }).join("") : `<div class="empty-state" style="grid-column:1/-1"><div><strong>没有匹配的图片</strong><p>换个筛选条件，或先去创建新任务。</p></div></div>`;
    $$('[data-image]', grid).forEach((button) => button.addEventListener("click", () => openImage(button.dataset.image)));
    const pages = Math.max(1, Math.ceil((data.total || 0) / g.perPage));
    $("#gallery-pager").innerHTML = `<button class="secondary-button" type="button" id="gallery-prev" ${g.page <= 1 ? "disabled" : ""}>上一页</button><span>${g.page} / ${pages}</span><button class="secondary-button" type="button" id="gallery-next" ${g.page >= pages ? "disabled" : ""}>下一页</button>`;
    $("#gallery-prev")?.addEventListener("click", () => updateGalleryRoute({ page: g.page - 1 }));
    $("#gallery-next")?.addEventListener("click", () => updateGalleryRoute({ page: g.page + 1 }));
  } catch (error) {
    $("#gallery-grid").innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div><strong>图库读取失败</strong><p>${escapeHtml(error.message)}</p></div></div>`;
  }
}

export async function openImage(imageId) {
  try {
    const image = await api(`/api/images/${encodeURIComponent(imageId)}`);
    imageDialog.innerHTML = `<div class="image-dialog-layout"><div class="image-dialog-media"><img src="/api/images/${encodeURIComponent(image.id)}/file" alt="${escapeHtml(image.prompt)}" /></div><aside class="image-dialog-side"><button class="dialog-close" type="button" aria-label="关闭">×</button><p class="eyebrow">IMAGE</p><p class="image-prompt">${escapeHtml(image.prompt)}</p><div class="fact-list"><div class="fact"><span>模型</span><b>${escapeHtml(image.recipe)}</b></div><div class="fact"><span>Seed</span><b class="mono">${escapeHtml(image.seed)}</b></div><div class="fact"><span>尺寸</span><b>${image.width} × ${image.height}</b></div><div class="fact"><span>Steps / CFG</span><b>${image.steps ?? "—"} / ${image.cfg_scale ?? "—"}</b></div><div class="fact"><span>耗时</span><b>${image.duration_ms ? `${(image.duration_ms / 1000).toFixed(2)} 秒` : "—"}</b></div><div class="fact"><span>任务</span><b class="mono">${escapeHtml(image.job_id)}</b></div></div><div class="image-dialog-actions"><button class="secondary-button" type="button" id="copy-prompt">复制 Prompt</button><button class="secondary-button" type="button" id="regen-image">再生成</button><a class="secondary-button" href="/api/images/${encodeURIComponent(image.id)}/file" download style="display:grid;place-items:center">下载</a><button class="secondary-button" type="button" id="open-run">查看任务</button></div></aside></div>`;
    $(".dialog-close", imageDialog).addEventListener("click", () => imageDialog.close());
    $("#copy-prompt", imageDialog).addEventListener("click", async () => { await navigator.clipboard.writeText(image.prompt); toast("Prompt 已复制"); });
    $("#regen-image", imageDialog).addEventListener("click", async () => { try { const job = await api(`/api/images/${encodeURIComponent(image.id)}/regenerate`, { method: "POST" }); imageDialog.close(); toast("已创建再生成任务"); location.hash = `#/runs/${job.id}`; } catch (error) { toast(error.message, "bad"); } });
    $("#open-run", imageDialog).addEventListener("click", () => { imageDialog.close(); location.hash = `#/runs/${image.job_id}`; });
    imageDialog.showModal();
  } catch (error) { toast(error.message, "bad"); }
}
