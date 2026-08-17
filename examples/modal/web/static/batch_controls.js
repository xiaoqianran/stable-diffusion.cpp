const ORIGINAL_FETCH = window.fetch.bind(window);
const OPTIONS = [
  ["1", "严格串行 · 1 个 GPU"],
  ["2", "限制并行 · 2 个 GPU"],
  ["4", "限制并行 · 4 个 GPU"],
];

function selectedParallelism() {
  const select = document.getElementById("batch-parallelism");
  const value = Number(select?.value || 1);
  return [1, 2, 4].includes(value) ? value : 1;
}

function installControl() {
  if (!location.hash.startsWith("#/batch")) return;
  const form = document.getElementById("batch-form");
  if (!form || document.getElementById("batch-parallelism")) return;

  const composer = form.querySelector(".composer");
  if (!composer) return;

  const field = document.createElement("div");
  field.className = "field batch-parallel-field";
  field.innerHTML = `
    <label class="field-label" for="batch-parallelism">GPU 并行</label>
    <select id="batch-parallelism" name="parallelism">
      ${OPTIONS.map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}
    </select>
    <span class="hint">1 最省；2 / 4 会同时启用更多 GPU container。其他 Job 仍在外层排队。</span>`;
  composer.insertAdjacentElement("afterend", field);
}

// app.js owns the existing forms. Add only one transport-level field to its
// requests, so the original form/progress/SSE behavior remains untouched.
window.fetch = async (input, init = {}) => {
  let url = typeof input === "string" ? input : input?.url || "";
  if (!location.hash.startsWith("#/batch") || !init || String(init.method || "GET").toUpperCase() !== "POST") {
    return ORIGINAL_FETCH(input, init);
  }

  const parallelism = selectedParallelism();
  if (url === "/api/jobs" && typeof init.body === "string") {
    try {
      const payload = JSON.parse(init.body);
      payload.parallelism = parallelism;
      init = { ...init, body: JSON.stringify(payload) };
    } catch {
      // Leave unrelated/invalid JSON untouched; backend will report it normally.
    }
  } else if (url.startsWith("/api/jobs/from-file")) {
    const parsed = new URL(url, location.origin);
    parsed.searchParams.set("parallelism", String(parallelism));
    url = `${parsed.pathname}${parsed.search}`;
    input = url;
  }
  return ORIGINAL_FETCH(input, init);
};

function refreshControl() {
  requestAnimationFrame(installControl);
}

window.addEventListener("hashchange", refreshControl);
const observer = new MutationObserver(() => {
  if (!document.getElementById("batch-parallelism")) refreshControl();
});
observer.observe(document.getElementById("main"), { childList: true, subtree: true });
refreshControl();
