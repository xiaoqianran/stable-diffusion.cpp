import { $, api, escapeHtml, imageDialog, modelById, parseRoute, queueButton, queueLabel, refreshRuntime, setNav, state, systemDialog, workspace } from "./ux-core.js";
import { renderCreate } from "./ux-create.js";
import { renderGallery } from "./ux-gallery.js";
import { renderRunDetail, renderRuns } from "./ux-runs.js";
import { openSystem } from "./ux-system.js";

async function renderRoute({ preserveScroll = false } = {}) {
  if (!state.meta) return;
  const scroll = window.scrollY;
  const route = parseRoute();
  setNav(route.page);
  if (route.page === "create") renderCreate();
  else if (route.page === "runs" && route.id) await renderRunDetail(route.id);
  else if (route.page === "runs") renderRuns();
  else if (route.page === "gallery") await renderGallery(route);
  else location.hash = "#/create";
  if (preserveScroll) window.scrollTo({ top: scroll });
  else window.scrollTo({ top: 0 });
}

async function boot() {
  try {
    state.meta = await api("/api/meta");
    state.createDraft.recipe = state.meta.defaults?.recipe || state.meta.models?.[0]?.id;
    const model = modelById(state.createDraft.recipe);
    state.createDraft.gpu = model.default_gpu || state.meta.defaults?.gpu;
    await refreshRuntime();
    if (!location.hash) history.replaceState(null, "", "#/create");
    await renderRoute();
    setInterval(async () => {
      const route = parseRoute();
      const ok = await refreshRuntime();
      if (ok && route.page === "runs") await renderRoute({ preserveScroll: true });
    }, 1800);
  } catch (error) {
    workspace.innerHTML = `<div class="empty-state"><div><strong>工作台启动失败</strong><p>${escapeHtml(error.message)}</p><button class="secondary-button" onclick="location.reload()">重新加载</button></div></div>`;
    queueButton.dataset.state = "offline";
    queueLabel.textContent = "API 离线";
  }
}

window.addEventListener("hashchange", () => renderRoute());
queueButton.addEventListener("click", openSystem);
$("#system-button").addEventListener("click", openSystem);
[imageDialog, systemDialog].forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
boot();
