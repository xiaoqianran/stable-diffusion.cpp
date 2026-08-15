from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


KIND_LABELS = {
    "image": "Image",
    "edit": "Edit",
    "video": "Video",
    "other": "Other",
}

PER_PAGE = 12


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def slugify_model(value: str) -> str:
    text = (value or "other").strip().lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return (text[:64] or "other")


def load_families(path: Path | None = None) -> list[dict[str, str]]:
    models_path = path or Path(__file__).resolve().parent.parent / "gallery" / "models.json"
    payload = json.loads(models_path.read_text(encoding="utf-8"))
    return list(payload["families"])


def family_label(model_id: str, families: Iterable[dict[str, str]] | None = None) -> str:
    families = list(families or load_families())
    for item in families:
        if item["id"] == model_id:
            return item["label"]
    return model_id


def image_id(model_id: str, seed: int, data: bytes, when: datetime | None = None) -> str:
    stamp = (when or utc_now()).strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha256(data).hexdigest()[:8]
    return f"{stamp}-{model_id}-{seed}-{digest}"


@dataclass
class ImageRecord:
    id: str
    model: str
    path: str
    prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    cfg_scale: float | None = None
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    gpu_name: str = ""
    cuda_version: str = ""
    torch_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageRecord:
        known = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        payload["extra"] = dict(extra)
        if payload.get("duration_ms") in (None, "", 0, "0"):
            payload["duration_ms"] = _optional_int(extra.get("duration_ms"))
        payload.setdefault("gpu_name", "")
        payload.setdefault("cuda_version", "")
        payload.setdefault("torch_version", "")
        if not payload.get("gpu_name"):
            payload["gpu_name"] = str(extra.get("gpu_name") or "")
        if not payload.get("cuda_version"):
            payload["cuda_version"] = str(extra.get("cuda_version") or "")
        if not payload.get("torch_version"):
            payload["torch_version"] = str(extra.get("torch_version") or "")
        return cls(**payload)


def record_paths(model_id: str, record_id: str) -> tuple[str, str]:
    model_id = slugify_model(model_id)
    rel = f"images/{model_id}/{record_id}"
    return f"{rel}.png", f"{rel}.json"


def write_sidecar(root: Path, record: ImageRecord, image_bytes: bytes) -> tuple[Path, Path]:
    png_rel, json_rel = record_paths(record.model, record.id)
    png_path = root / png_rel
    json_path = root / json_rel
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(image_bytes)
    json_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return png_path, json_path


def scan_records(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for sidecar in sorted(root.glob("images/*/*.json")):
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        record = ImageRecord.from_dict(data)
        png = sidecar.with_suffix(".png")
        if not png.exists():
            continue
        if not record.path:
            record.path = png.relative_to(root).as_posix()
        records.append(record)
    records.sort(key=lambda item: item.created_at or item.id, reverse=True)
    return records


def paginate(items: list[ImageRecord], per_page: int = PER_PAGE) -> list[list[ImageRecord]]:
    if not items:
        return [[]]
    return [items[index : index + per_page] for index in range(0, len(items), per_page)]


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_duration_ms(duration_ms: int | None) -> str:
    if not duration_ms:
        return ""
    seconds = duration_ms / 1000.0
    return f"{seconds:.1f}s" if seconds >= 10 else f"{seconds:.2f}s"


def card_host_line(record: ImageRecord) -> str:
    extra = record.extra if isinstance(record.extra, dict) else {}
    bits: list[str] = []
    duration = format_duration_ms(record.duration_ms or _optional_int(extra.get("duration_ms")))
    if duration:
        bits.append(duration)
    gpu = record.gpu_name or extra.get("gpu_name")
    if gpu:
        bits.append(str(gpu))
    cuda = record.cuda_version or extra.get("cuda_version")
    if cuda:
        bits.append(f"CUDA {cuda}")
    torch_v = record.torch_version or extra.get("torch_version")
    if torch_v:
        bits.append(f"torch {torch_v}")
    driver = extra.get("driver_version") or extra.get("nvidia_driver")
    if driver:
        bits.append(f"driver {driver}")
    sd_cli = extra.get("sd_cli_version")
    if sd_cli:
        bits.append(f"sd-cli {sd_cli}")
    return " · ".join(bits)


def card_meta_line(record: ImageRecord, families: Iterable[dict[str, str]] | None = None) -> str:
    extra = record.extra if isinstance(record.extra, dict) else {}
    bits = [
        family_label(record.model, families),
        f"seed {record.seed}" if record.seed is not None else "",
        f"{record.steps} steps" if record.steps else "",
        f"cfg {record.cfg_scale:g}" if record.cfg_scale else "",
        f"{record.width}×{record.height}" if record.width and record.height else "",
        f"python {extra['python_version']}" if extra.get("python_version") else "",
        f"modal {extra['modal_gpu']}" if extra.get("modal_gpu") else "",
        record.created_at or "",
    ]
    return " · ".join(part for part in bits if part)


def _rel(depth: int, dest: str) -> str:
    return ("../" * depth) + dest


def _page_href(depth: int, model: str | None, page: int) -> str:
    if model is None:
        return _rel(depth, "index.html") if page <= 1 else _rel(depth, f"page/{page}.html")
    if page <= 1:
        return _rel(depth, f"model/{model}/index.html")
    return _rel(depth, f"model/{model}/page/{page}.html")


def _render_page(
    *,
    title: str,
    records: list[ImageRecord],
    page: int,
    pages: int,
    model: str | None,
    families: list[dict[str, str]],
    counts: dict[str, int],
    depth: int,
) -> str:
    chips = ['<a class="chip{active}" href="{href}">All <span>{n}</span></a>'.format(
        active=" is-active" if model is None else "",
        href=_rel(depth, "index.html"),
        n=sum(counts.values()),
    )]
    for kind in ("image", "edit", "video", "other"):
        group = [item for item in families if item.get("kind") == kind]
        if not group:
            continue
        chips.append(f'<span class="kind">{html.escape(KIND_LABELS.get(kind, kind))}</span>')
        for item in group:
            chips.append(
                '<a class="chip{active}" href="{href}">{label} <span>{n}</span></a>'.format(
                    active=" is-active" if model == item["id"] else "",
                    href=_rel(depth, f"model/{item['id']}/index.html"),
                    label=html.escape(item["label"]),
                    n=counts.get(item["id"], 0),
                )
            )
    extra_models = sorted(name for name in counts if name not in {item["id"] for item in families})
    if extra_models:
        chips.append('<span class="kind">Custom</span>')
        for name in extra_models:
            chips.append(
                '<a class="chip{active}" href="{href}">{label} <span>{n}</span></a>'.format(
                    active=" is-active" if model == name else "",
                    href=_rel(depth, f"model/{name}/index.html"),
                    label=html.escape(name),
                    n=counts[name],
                )
            )

    cards = []
    for record in records:
        prompt = html.escape(record.prompt or "(no prompt)")
        host = html.escape(card_host_line(record))
        meta = html.escape(card_meta_line(record, families))
        host_html = f'<p class="host">{host}</p>' if host else ""
        cards.append(
            f"""<article class="card">
  <a href="{html.escape(_rel(depth, record.path))}" target="_blank" rel="noopener">
    <img src="{html.escape(_rel(depth, record.path))}" alt="{prompt}" loading="lazy">
  </a>
  <div class="meta">
    <p class="prompt">{prompt}</p>
    {host_html}
    <p class="sub">{meta}</p>
  </div>
</article>"""
        )
    if not cards:
        cards.append('<p class="empty">No images for this model yet. Publish a PNG to the dataset and rebuild Pages.</p>')

    nav = []
    if pages > 1:
        if page > 1:
            nav.append(f'<a href="{_page_href(depth, model, page - 1)}">Previous</a>')
        nav.append(f'<span>Page {page} / {pages}</span>')
        if page < pages:
            nav.append(f'<a href="{_page_href(depth, model, page + 1)}">Next</a>')
        numbers = []
        for number in range(1, pages + 1):
            cls = " is-active" if number == page else ""
            numbers.append(f'<a class="num{cls}" href="{_page_href(depth, model, number)}">{number}</a>')
        nav.append('<span class="nums">' + "".join(numbers) + "</span>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{_rel(depth, "assets/style.css")}">
</head>
<body>
  <header>
    <p class="eyebrow">stable-diffusion.cpp</p>
    <h1>Generation gallery</h1>
    <p class="lede">Images are stored on Hugging Face by model family. New families get their own folder automatically.</p>
  </header>
  <nav class="filters">{"".join(chips)}</nav>
  <main class="grid">
    {"".join(cards)}
  </main>
  <nav class="pager">{"".join(nav)}</nav>
</body>
</html>
"""


STYLE = """:root {
  color-scheme: dark;
  --bg: #101218;
  --card: #1a1e27;
  --ink: #f4f1ea;
  --muted: #b7b1a4;
  --line: #2c3140;
  --accent: #e8c07a;
}
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  max-width: 1100px;
  padding: 32px 20px 64px;
  font: 16px/1.5 ui-sans-serif, system-ui, sans-serif;
  background: var(--bg);
  color: var(--ink);
}
.eyebrow { letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); font-size: 12px; }
h1 { margin: 0 0 8px; font-size: 36px; }
.lede, .sub, .empty { color: var(--muted); }
.host { color: #c4b5fd; font-size: 13px; margin: 0 0 6px; line-height: 1.4; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 24px 0; }
.kind { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-left: 8px; }
.chip, .pager a, .num {
  display: inline-flex; gap: 6px; align-items: center;
  padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px;
  color: var(--ink); text-decoration: none; background: #161a22;
}
.chip span { color: var(--muted); font-size: 12px; }
.chip.is-active, .num.is-active { border-color: var(--accent); color: var(--accent); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #000; }
.meta { padding: 12px; }
.prompt { margin: 0 0 6px; }
.pager { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 28px; }
.nums { display: flex; gap: 6px; flex-wrap: wrap; }
.empty { grid-column: 1 / -1; }
"""


def build_site(
    dataset_root: Path,
    output_dir: Path,
    per_page: int = PER_PAGE,
    families: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    families = list(families or load_families())
    records = scan_records(dataset_root)
    counts = {item["id"]: 0 for item in families}
    for record in records:
        counts[record.model] = counts.get(record.model, 0) + 1

    if output_dir.exists():
        for path in output_dir.rglob("*"):
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    (output_dir / "assets" / "style.css").write_text(STYLE, encoding="utf-8")

    images_src = dataset_root / "images"
    if images_src.exists():
        for src in images_src.rglob("*.png"):
            dest = output_dir / src.relative_to(dataset_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

    def write_pages(subset: list[ImageRecord], model: str | None, title: str) -> None:
        chunks = paginate(subset, per_page=per_page)
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            if model is None:
                dest = output_dir / "index.html" if index == 1 else output_dir / "page" / f"{index}.html"
                depth = 0 if index == 1 else 1
            else:
                dest = (
                    output_dir / "model" / model / "index.html"
                    if index == 1
                    else output_dir / "model" / model / "page" / f"{index}.html"
                )
                depth = 2 if index == 1 else 3
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                _render_page(
                    title=title,
                    records=chunk,
                    page=index,
                    pages=total,
                    model=model,
                    families=families,
                    counts=counts,
                    depth=depth,
                ),
                encoding="utf-8",
            )

    write_pages(records, None, "stable-diffusion.cpp gallery")
    seen = {item["id"] for item in families}
    for item in families:
        write_pages(
            [record for record in records if record.model == item["id"]],
            item["id"],
            f"{item['label']} · gallery",
        )
    for model_id in sorted(counts):
        if model_id in seen:
            continue
        write_pages(
            [record for record in records if record.model == model_id],
            model_id,
            f"{model_id} · gallery",
        )
    return {"images": len(records), "models": len(counts)}
