const main = document.getElementById("main");
const lightbox = document.getElementById("lightbox");
const railStatus = document.getElementById("rail-status");

const state = { meta: null, source: null };

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

function defaultsFrom(meta) {
  const model = (meta.models || []).find((item) => item.id === (meta.defaults?.recipe || "z-image-turbo")) || {};
  return {
    recipe: meta.defaults?.recipe || "z-image-turbo",
    gpu: meta.defaults?.gpu || "L40S",
    count: 1,
    width: model.width || 512,
    height: model.height || 1024,
    steps: model.default_steps || 8,
    cfg_scale: model.cfg_scale || 1,
    seed: 101,
    dry_run: false,
  };
}

function field(name, label, value, type = "text") {
  if (type === "checkbox") {
    return `<label class="check"><span>${label}</span><input type="checkbox" name="${name}" ${value ? "checked" : ""} /></label>`;
  }
  return `<div><label>${label}</label><input name="${name}" type="${type}" value="${value ?? ""}" /></div>`;
}

function select(name, label, options, value) {
  const opts = options.map((item) => {
    const id = item.id || item;
    const title = item.name || item.id || item;
    return `<option value="${id}" ${id === value ? "selected" : ""}>${title}</option>`;
  }).join("");
  return `<div><label>${label}</label><select name="${name}">${opts}</select></div>`;
}

function settingsGrid(d, meta) {
  return `
    <div class="grid">
      ${select("recipe", "Recipe", meta.models || [], d.recipe)}
      ${select("gpu", "GPU", (meta.gpus || []).map((gpu) => ({
        id: gpu.id,
        name: `${gpu.id} · $${Number(gpu.usd_per_hour).toFixed(2)}/hr`,
      })), d.gpu)}
      ${field("count", "Count", d.count, "number")}
      ${field("width", "Width", d.width, "number")}
      ${field("height", "Height", d.height, "number")}
      ${field("steps", "Steps", d.steps, "number")}
      ${field("cfg_scale", "CFG", d.cfg_scale, "number")}
      ${field("seed", "Seed", d.seed, "number")}
    </div>
    ${field("dry_run", "Dry run (no Modal / no GPU)", d.dry_run, "checkbox")}
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

function progressBox() {
  return `<div class="progress" id="progress"><div>0 / 0</div><div class="bar"><span></span></div></div>`;
}

function setProgress(completed, total, extra = "") {
  const root = document.getElementById("progress");
  if (!root) return;
  root.firstElementChild.textContent = `${completed} / ${total} ${extra}`.trim();
  root.querySelector("span").style.width = total ? `${Math.min(100, (completed / total) * 100)}%` : "0%";
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
      setProgress(progress.completed || 0, progress.total || 0, event.type === "image.failed" ? "failed" : "");
    }
    if (["job.completed", "job.failed", "job.cancelled"].includes(event.type)) {
      setProgress(event.payload.completed_images || 0, event.payload.total_images || 0, event.payload.status);
      state.source.close();
      if (event.type === "job.failed") alert(event.payload.error || "job failed");
      if (event.type === "job.completed") render("gallery");
    }
  };
}

function applyRecipeDefaults(form, meta) {
  const recipe = (meta.models || []).find((item) => item.id === form.recipe.value);
  if (!recipe) return;
  form.width.value = recipe.width;
  form.height.value = recipe.height;
  form.steps.value = recipe.default_steps;
  form.cfg_scale.value = recipe.cfg_scale;
  const hint = document.getElementById("recipe-hint");
  if (hint) hint.textContent = recipe.hint || "";
  const prompt = form.querySelector("[name=prompt]");
  if (prompt && recipe.id === "ideogram4" && !prompt.value.trim()) {
    prompt.placeholder = '{"high_level_description":"A fluffy orange cat"}';
  }
}

function generatePage(meta) {
  const d = defaultsFrom(meta);
  main.innerHTML = `
    <h1>Generate</h1>
    <p class="lede">Local workbench for the seven Modal <code>sd-cli</code> recipes. This is not <code>modal serve</code>. Default recipe is Z-Image Turbo — the stack that already produced the 30-image gallery.</p>
    <form class="panel" id="gen-form">
      <label>Prompt</label>
      <textarea name="prompt" placeholder="a rainy city at night, cinematic photograph" required></textarea>
      ${settingsGrid(d, meta)}
      <p class="apply-line" id="recipe-hint"></p>
      <p class="apply-line mono" id="will-apply"></p>
      <div class="actions">
        <button type="submit">Generate</button>
        <span class="mono" id="job-id"></span>
      </div>
      ${progressBox()}
    </form>
  `;
  const form = document.getElementById("gen-form");
  const refresh = () => {
    const payload = formPayload(form);
    const recipe = (meta.models || []).find((item) => item.id === payload.recipe);
    document.getElementById("will-apply").textContent =
      `will request GPU=${payload.gpu}  recipe=${payload.recipe}  ${payload.width}×${payload.height}  steps=${payload.steps}  cfg=${payload.cfg_scale}  count=${payload.count}`;
    document.getElementById("recipe-hint").textContent = recipe?.hint || "";
  };
  form.recipe.onchange = () => {
    applyRecipeDefaults(form, meta);
    refresh();
  };
  form.addEventListener("input", refresh);
  form.onsubmit = async (event) => {
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
      listenJob(job.id);
    } catch (error) {
      alert(error.message);
    } finally {
      button.disabled = false;
    }
  };
  applyRecipeDefaults(form, meta);
  refresh();
}

function batchPage(meta) {
  const d = defaultsFrom(meta);
  main.innerHTML = `
    <h1>Batch</h1>
    <p class="lede">Paste one prompt per line, or drop a txt file. Count applies to every line.</p>
    <form class="panel" id="batch-form">
      <div class="drop" id="drop">Drop prompts.txt here, or choose a file
        <div style="margin-top:12px"><input type="file" name="file" accept=".txt,.jsonl" /></div>
      </div>
      <label>Or paste</label>
      <textarea name="text" placeholder="a beautiful forest&#10;a futuristic city"></textarea>
      ${settingsGrid(d, meta)}
      <p class="apply-line" id="recipe-hint"></p>
      <div class="actions"><button type="submit">Run batch</button><span class="mono" id="job-id"></span></div>
      ${progressBox()}
    </form>
  `;
  const form = document.getElementById("batch-form");
  const drop = document.getElementById("drop");
  const fileInput = drop.querySelector("input[type=file]");
  drop.ondragover = (event) => { event.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    if (event.dataTransfer.files[0]) fileInput.files = event.dataTransfer.files;
  };
  form.recipe.onchange = () => applyRecipeDefaults(form, meta);
  applyRecipeDefaults(form, meta);
  form.onsubmit = async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button");
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
      listenJob(job.id);
    } catch (error) {
      alert(error.message);
    } finally {
      button.disabled = false;
    }
  };
}

async function jobsPage() {
  const jobs = await api("/api/jobs");
  main.innerHTML = `
    <h1>Jobs</h1>
    <p class="lede">Every generate or batch is a Job. Resume retries unfinished frames.</p>
    <div class="panel">
      <table>
        <thead><tr><th>ID</th><th>STATUS</th><th>IMAGES</th><th>RECIPE</th><th>GPU</th><th></th></tr></thead>
        <tbody>
          ${jobs.map((job) => `
            <tr>
              <td class="mono"><a href="#/job/${job.id}">${job.id}</a></td>
              <td><span class="pill ${job.status}">${job.status}</span></td>
              <td>${job.completed_images}/${job.total_images}</td>
              <td>${job.recipe}</td>
              <td>${job.gpu}</td>
              <td>
                <button class="ghost" data-gallery="${job.id}">Gallery</button>
                <button class="ghost" data-resume="${job.id}">Resume</button>
              </td>
            </tr>`).join("") || `<tr><td colspan="6">No jobs yet.</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
  main.querySelectorAll("[data-gallery]").forEach((button) => {
    button.onclick = () => { location.hash = `#/gallery?job=${button.dataset.gallery}`; };
  });
  main.querySelectorAll("[data-resume]").forEach((button) => {
    button.onclick = async () => {
      await api(`/api/jobs/${button.dataset.resume}/resume`, { method: "POST" });
      render("jobs");
    };
  });
}

async function jobDetailPage(jobId) {
  const detail = await api(`/api/jobs/${jobId}`);
  const job = detail.job;
  main.innerHTML = `
    <h1>Job</h1>
    <div class="panel">
      <div class="check"><span>ID</span><span class="mono">${job.id}</span></div>
      <div class="check"><span>Status</span><span class="pill ${job.status}">${job.status}</span></div>
      <div class="check"><span>Recipe / GPU</span><span>${job.recipe} · ${job.gpu}</span></div>
      <div class="check"><span>Images</span><span>${job.completed_images}/${job.total_images}</span></div>
      <div class="actions" style="margin-top:16px">
        <button class="ghost" id="to-gallery">Gallery</button>
        <button class="ghost" id="to-jobs">All jobs</button>
      </div>
    </div>
    <h2 class="section">Frames</h2>
    <div class="panel">
      <table>
        <thead><tr><th>ID</th><th>STATUS</th><th>SEED</th><th>SIZE</th><th>TIME</th></tr></thead>
        <tbody>
          ${(detail.images || []).map((item) => `
            <tr>
              <td class="mono">${item.id}</td>
              <td><span class="pill ${item.status}">${item.status}</span></td>
              <td class="mono">${item.seed}</td>
              <td>${item.width}×${item.height}</td>
              <td>${item.duration_ms != null ? (item.duration_ms / 1000).toFixed(2) + "s" : "—"}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
  document.getElementById("to-gallery").onclick = () => { location.hash = `#/gallery?job=${job.id}`; };
  document.getElementById("to-jobs").onclick = () => { location.hash = "#/jobs"; };
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
  main.innerHTML = `
    <h1>Gallery</h1>
    <p class="lede">${data.total} local images. Hover a card for prompt and seed.</p>
    <form class="toolbar" id="filters">
      <input name="q" placeholder="search prompt" value="${q}" />
      <input name="job" placeholder="job id" value="${job}" />
      <input name="recipe" placeholder="recipe" value="${recipe}" />
      <button type="submit" class="ghost">Filter</button>
    </form>
    <div class="gallery">
      ${data.items.map((image) => `
        <article class="card" data-id="${image.id}">
          <img src="/api/images/${image.id}/file" alt="" />
          <div class="cap">${image.prompt}</div>
          <div class="hover">
            <div>${image.prompt}</div>
            <div class="mono" style="margin-top:10px">seed ${image.seed}<br>${image.recipe}<br>${image.width}×${image.height}<br>${image.latency_ms ? (image.latency_ms / 1000).toFixed(2) + "s" : ""}</div>
          </div>
        </article>`).join("") || "<p>No images yet. Generate something.</p>"}
    </div>
    <div class="pager">
      <button class="ghost" ${page <= 1 ? "disabled" : ""} id="prev">Prev</button>
      <span>page ${data.page}</span>
      <button class="ghost" ${page * per >= data.total ? "disabled" : ""} id="next">Next</button>
    </div>
  `;
  document.getElementById("filters").onsubmit = (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    location.hash = `#/gallery?${new URLSearchParams({
      q: form.get("q") || "",
      job: form.get("job") || "",
      recipe: form.get("recipe") || "",
      page: "1",
    })}`;
  };
  document.getElementById("prev").onclick = () => {
    params.set("page", String(page - 1));
    location.hash = `#/gallery?${params}`;
  };
  document.getElementById("next").onclick = () => {
    params.set("page", String(page + 1));
    location.hash = `#/gallery?${params}`;
  };
  main.querySelectorAll(".card").forEach((card) => {
    card.onclick = () => openLightbox(card.dataset.id);
  });
}

async function openLightbox(imageId) {
  const image = await api(`/api/images/${imageId}`);
  lightbox.classList.remove("hidden");
  lightbox.innerHTML = `
    <img src="/api/images/${image.id}/file" alt="" />
    <div class="meta">
      <h2>Frame</h2>
      <p>${image.prompt}</p>
      <dl>
        <dt>Seed</dt><dd class="mono">${image.seed}</dd>
        <dt>Recipe</dt><dd>${image.recipe}</dd>
        <dt>Steps</dt><dd>${image.steps}</dd>
        <dt>Size</dt><dd>${image.width} × ${image.height}</dd>
        <dt>Time</dt><dd>${image.latency_ms ? (image.latency_ms / 1000).toFixed(2) + "s" : "—"}</dd>
        <dt>Job</dt><dd class="mono"><a href="#/job/${image.job_id}">${image.job_id}</a></dd>
      </dl>
      <div class="actions" style="margin-top:18px">
        <button id="copy">Copy prompt</button>
        <button class="ghost" id="regen">Regenerate</button>
        <a class="btn ghost" href="/api/images/${image.id}/file" download>Download</a>
        <button class="ghost" id="close">Close</button>
      </div>
    </div>
  `;
  document.getElementById("close").onclick = () => lightbox.classList.add("hidden");
  document.getElementById("copy").onclick = async () => navigator.clipboard.writeText(image.prompt);
  document.getElementById("regen").onclick = async () => {
    const job = await api(`/api/images/${image.id}/regenerate`, { method: "POST" });
    lightbox.classList.add("hidden");
    location.hash = "#/jobs";
    alert(`queued ${job.id}`);
  };
}

async function settingsPage(meta) {
  const doctor = await api("/api/doctor");
  main.innerHTML = `
    <h1>Settings</h1>
    <p class="lede">Local workbench. Weights stay on volume <code>sdcpp-models</code>. GPU default is L40S. A10 and A100 are blocked.</p>
    <div class="panel">
      <div class="check"><span>Data dir</span><span class="mono">${meta.defaults.data_dir}</span></div>
      <div class="check"><span>Default recipe</span><span>${meta.defaults.recipe}</span></div>
      <div class="check"><span>Default GPU</span><span>${meta.defaults.gpu}</span></div>
      <h2 class="section">Recipes</h2>
      ${(meta.models || []).map((model) => `
        <div class="check">
          <span>${model.label}</span>
          <span class="mono">${model.id} · ${model.width}×${model.height} · ${model.default_steps} steps</span>
        </div>`).join("")}
      <h2 class="section">Doctor</h2>
      <div class="checks">
        ${doctor.checks.map((check) => `
          <div class="check">
            <span>${check.name}</span>
            <span class="${check.ok ? "ok" : "bad"}">${check.ok ? "✓" : "✗"} ${check.detail}</span>
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
    link.classList.toggle("active", link.dataset.page === page || (page.startsWith("job/") && link.dataset.page === "jobs"));
  });
  if (!state.meta) state.meta = await api("/api/meta");
  if (page === "generate") generatePage(state.meta);
  else if (page === "batch") batchPage(state.meta);
  else if (page === "jobs") await jobsPage();
  else if (page.startsWith("job/")) await jobDetailPage(page.slice(4));
  else if (page === "gallery") await galleryPage(params);
  else if (page === "settings") await settingsPage(state.meta);
  else generatePage(state.meta);
}

window.addEventListener("hashchange", () => render());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.classList.add("hidden");
});

(async () => {
  try {
    const [doctor, meta] = await Promise.all([api("/api/doctor"), api("/api/meta")]);
    state.meta = meta;
    railStatus.textContent = doctor.ready ? "local web · 7 recipes" : "setup incomplete";
    railStatus.className = doctor.ready ? "rail-foot ok" : "rail-foot bad";
  } catch {
    railStatus.textContent = "api offline";
  }
  await render();
})();
