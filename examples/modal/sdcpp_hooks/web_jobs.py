from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cost import default_plan
from .cost_view import job_totals
from .gpu import default_gpu_for_recipe
from .gpu_queue import GPUQueue
from .meter import bind_task, client_ledger, record_event
from .recipes import RECIPES
from .web_catalog import default_recipe, normalize_gpu
from .web_events import Event, EventBus
from .web_generator import ALLOWED_PARALLELISM, MockGenerator, ModalGenerator


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    recipe TEXT NOT NULL,
    gpu TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    total_images INTEGER NOT NULL,
    completed_images INTEGER NOT NULL DEFAULT 0,
    failed_images INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    seed INTEGER NOT NULL,
    recipe TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    steps INTEGER,
    cfg_scale REAL,
    status TEXT NOT NULL,
    path TEXT,
    duration_ms INTEGER,
    error TEXT,
    created_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _gpu_job_queue_limit() -> int:
    try:
        return max(1, int(os.environ.get("SDCPP_GPU_JOB_MAX_ACTIVE", "1")))
    except ValueError:
        return 1


def _affinity_key(job: dict[str, Any]) -> str:
    return f"{job['gpu']}::{job['recipe']}"


class JobService:
    def __init__(self, data_dir: Path, events: EventBus | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.outputs_dir = self.data_dir / "outputs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "sdcpp-web.db"
        self.events = events or EventBus()
        self._lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._cancel: set[str] = set()
        self._threads: dict[str, threading.Thread] = {}
        self._phases: dict[str, str] = {}
        self._model_resident: dict[str, bool] = {}
        self._gpu_queue = GPUQueue(max_active=_gpu_job_queue_limit())
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_job(self, specs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        recipe = config.get("recipe") or default_recipe()
        if recipe not in RECIPES:
            raise KeyError(f"unknown recipe {recipe!r}")
        gpu_raw = config.get("gpu")
        gpu = normalize_gpu(str(gpu_raw) if gpu_raw not in (None, "") else default_gpu_for_recipe(recipe))
        count = max(1, int(config.get("count") or 1))
        parallelism = int(config.get("parallelism") or 1)
        if parallelism not in ALLOWED_PARALLELISM:
            raise ValueError("parallelism must be one of 1, 2, 4")
        base_seed = config.get("seed")
        seed0 = int(base_seed) if base_seed not in (None, "") else 101
        defaults = RECIPES[recipe]
        width = int(config.get("width") or defaults.get("width") or 512)
        height = int(config.get("height") or defaults.get("height") or 512)
        steps = int(config.get("steps") or defaults.get("steps") or 20)
        cfg_scale = float(config.get("cfg_scale") or defaults.get("cfg_scale") or 1.0)
        if not specs:
            raise ValueError("no prompts")

        job_id = _new_id("job")
        now = _utc_now()
        images: list[tuple[str, int, str]] = []
        for spec in specs:
            prompt = str(spec.get("prompt") or "").strip()
            if not prompt:
                continue
            n = max(1, int(spec.get("count") or count))
            start = int(spec["seed"]) if spec.get("seed") not in (None, "") else seed0
            for offset in range(n):
                images.append((prompt, start + offset, spec.get("recipe") or recipe))
        if not images:
            raise ValueError("every prompt was empty")

        payload = {
            **config,
            "recipe": recipe,
            "gpu": gpu,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "count": count,
            "seed": seed0,
            "parallelism": parallelism,
        }
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, status, recipe, gpu, created_at, updated_at, config_json,
                    total_images, completed_images, failed_images)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)""",
                (job_id, "pending", recipe, gpu, now, now, json.dumps(payload), len(images)),
            )
            for prompt, seed, item_recipe in images:
                conn.execute(
                    """INSERT INTO images
                       (id, job_id, prompt, seed, recipe, width, height, steps, cfg_scale,
                        status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (_new_id("img"), job_id, prompt, seed, item_recipe, width, height, steps, cfg_scale, now),
                )
            conn.commit()
        self._set_phase(job_id, "pending")
        summary = self.get_job(job_id)
        self.events.publish(Event("job.snapshot", job_id, summary))
        return summary

    def start(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self._threads[job_id] = thread
            thread.start()

    def resume(self, job_id: str) -> dict[str, Any]:
        self._cancel.discard(job_id)
        self._set_phase(job_id, "pending")
        self.start(job_id)
        return self.get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self._cancel.add(job_id)
        self._gpu_queue.cancel(job_id)
        self._set_phase(job_id, "cancelled")
        self._set_job(job_id, status="cancelled")
        self._publish_queue_state()
        summary = self.get_job(job_id)
        self.events.publish(Event("job.cancelled", job_id, summary))
        return summary

    def queue_snapshot(self) -> dict[str, Any]:
        snapshot = self._gpu_queue.snapshot()
        running_id = snapshot.get("running_job_id")
        if running_id:
            try:
                job = self._raw_job(running_id)
            except KeyError:
                job = None
            if job:
                snapshot["running_recipe"] = job["recipe"]
                snapshot["running_gpu"] = job["gpu"]
                snapshot["running_parallelism"] = int(job["config"].get("parallelism") or 1)
                snapshot["running_model_resident"] = bool(self._model_resident.get(running_id))
        return snapshot

    def _raw_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_row(row)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        costs = job_totals(client_ledger().read())
        return [self._with_cost(self._attach_runtime(self._job_row(row)), costs) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._attach_runtime(self._raw_job(job_id))

    def get_job_detail(self, job_id: str) -> dict[str, Any]:
        job = self._with_cost(self.get_job(job_id))
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM images WHERE job_id = ? ORDER BY created_at", (job_id,)).fetchall()
        images = [self._image_row(row) for row in rows]
        by_image = {item["image_id"]: item for item in job.get("cost_chain") or [] if item.get("image_id")}
        for image in images:
            event = by_image.get(image["id"])
            if event:
                image["cost_usd"] = event["usd"]
                image["usd_per_second"] = event["usd_per_second"]
        return {"job": job, "images": images}

    def list_images(self, *, job_id: str | None = None, recipe: str | None = None, q: str | None = None, sort: str = "newest", page: int = 1, per_page: int = 50) -> dict[str, Any]:
        where = ["status = 'completed'", "path IS NOT NULL"]
        args: list[Any] = []
        if job_id:
            where.append("job_id = ?")
            args.append(job_id)
        if recipe:
            where.append("recipe = ?")
            args.append(recipe)
        if q:
            where.append("prompt LIKE ?")
            args.append(f"%{q}%")
        order = {"oldest": "created_at ASC", "fastest": "duration_ms ASC", "slowest": "duration_ms DESC"}.get(sort, "created_at DESC")
        clause = " AND ".join(where)
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM images WHERE {clause}", args).fetchone()[0]
            offset = max(0, (page - 1) * per_page)
            rows = conn.execute(f"SELECT * FROM images WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?", [*args, per_page, offset]).fetchall()
        return {"items": [self._image_row(row) for row in rows], "total": total, "page": page, "per_page": per_page}

    def get_image(self, image_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(image_id)
        return self._image_row(row)

    def regenerate(self, image_id: str) -> dict[str, Any]:
        image = self.get_image(image_id)
        job = self.get_job(image["job_id"])
        config = dict(job["config"])
        config["count"] = 1
        config["parallelism"] = 1
        config["seed"] = image["seed"]
        config["recipe"] = image["recipe"]
        created = self.create_job([{"prompt": image["prompt"], "seed": image["seed"], "count": 1}], config)
        self.start(created["id"])
        return created

    def _run(self, job_id: str) -> None:
        job = self.get_job(job_id)
        config = job["config"]
        parallelism = int(config.get("parallelism") or 1)
        self._set_job(job_id, status="running", error=None)
        with self._connect() as conn:
            pending = conn.execute("SELECT * FROM images WHERE job_id = ? AND status IN ('pending', 'failed')", (job_id,)).fetchall()
        items = [dict(row) for row in pending]
        dry_run = bool(config.get("dry_run"))
        acquired_gpu = False
        queued_gpu = False

        try:
            if dry_run:
                self._set_phase(job_id, "running")
                self.events.publish(Event("job.started", job_id, self.get_job(job_id)))
                mock = MockGenerator()
                with bind_task(job_id=job_id, recipe=job["recipe"], gpu=job["gpu"]):
                    record_event(default_plan("dry_run"), phase="local", extra={"call": "dry_run"})
                for item in items:
                    if job_id in self._cancel:
                        return
                    self._finish_one(job_id, item, mock.generate_one(prompt=item["prompt"], recipe=item["recipe"], seed=item["seed"], width=item["width"], height=item["height"], steps=item["steps"], cfg_scale=item["cfg_scale"], gpu=job["gpu"]))
            else:
                modal = ModalGenerator()
                generation_items = [{"id": item["id"], "prompt": item["prompt"], "recipe": item["recipe"], "seed": item["seed"], "width": item["width"], "height": item["height"], "steps": item["steps"], "cfg_scale": item["cfg_scale"]} for item in items]
                self._set_phase(job_id, "preparing")
                self.events.publish(Event("job.preparing", job_id, self.get_job(job_id)))
                requests = modal.prepare_job(generation_items, gpu=job["gpu"], job_id=job_id)
                if job_id in self._cancel:
                    return

                self._set_phase(job_id, "gpu_queued")
                self._gpu_queue.enqueue(job_id, affinity_key=_affinity_key(job))
                queued_gpu = True
                self._publish_queue_state()
                self.events.publish(Event("job.gpu_queued", job_id, self.get_job(job_id)))

                acquired_gpu = self._gpu_queue.acquire(job_id, cancelled=lambda: job_id in self._cancel)
                if not acquired_gpu:
                    return
                queued_gpu = False
                self._set_phase(job_id, "gpu_running")
                self._publish_queue_state()
                self.events.publish(Event("job.gpu_started", job_id, self.get_job(job_id)))

                results = modal.generate_prepared_job(generation_items, requests, gpu=job["gpu"], job_id=job_id, parallelism=parallelism)
                by_id = {item["id"]: item for item in items}
                for result in results:
                    if job_id in self._cancel:
                        return
                    if result.get("model_resident"):
                        with self._runtime_lock:
                            self._model_resident[job_id] = True
                    self._finish_one(job_id, by_id[result["id"]], result)
        except Exception as exc:
            self._set_phase(job_id, "failed")
            self._set_job(job_id, status="failed", error=str(exc))
            self.events.publish(Event("job.failed", job_id, {**self.get_job(job_id), "error": str(exc)}))
            return
        finally:
            if acquired_gpu:
                self._gpu_queue.release(job_id)
                self._publish_queue_state()
            elif queued_gpu:
                self._gpu_queue.cancel(job_id)
                self._publish_queue_state()

        if job_id in self._cancel:
            return
        summary = self.get_job(job_id)
        status = "completed" if summary["failed_images"] == 0 else "failed"
        self._set_phase(job_id, status)
        self._set_job(job_id, status=status)
        self.events.publish(Event(f"job.{status}", job_id, self.get_job(job_id)))

    def _publish_queue_state(self) -> None:
        snapshot = self._gpu_queue.snapshot()
        targets = list(dict.fromkeys([*snapshot["running_job_ids"], *snapshot["waiting_job_ids"]]))
        for target in targets:
            try:
                payload = self.get_job(target)
            except KeyError:
                continue
            self.events.publish(Event("gpu.queue", target, payload))

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._runtime_lock:
            self._phases[job_id] = phase

    def _attach_runtime(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._runtime_lock:
            phase = self._phases.get(job["id"])
            model_resident = bool(self._model_resident.get(job["id"]))
        queue = self._gpu_queue.snapshot(job["id"])
        job_queue = queue.get("job") or {}
        if phase is None:
            if job["status"] in {"completed", "failed", "cancelled"}:
                phase = job["status"]
            elif job_queue.get("state") == "running":
                phase = "gpu_running"
            elif job_queue.get("state") == "waiting":
                phase = "gpu_queued"
            else:
                phase = job["status"]
        job["phase"] = phase
        job["queue"] = job_queue
        job["parallelism"] = int(job["config"].get("parallelism") or 1)
        job["affinity_key"] = _affinity_key(job)
        job["model_resident"] = model_resident
        return job

    def _finish_one(self, job_id: str, item: dict[str, Any], result: dict[str, Any]) -> None:
        images = result.get("images") or []
        if not images:
            self._mark_image(item["id"], status="failed", error="no image returned")
            self._bump(job_id, failed=1)
            summary = self.get_job(job_id)
            self.events.publish(Event("image.failed", job_id, {"image_id": item["id"], "error": "no image returned", **self._progress(summary)}))
            return
        dest_dir = self.outputs_dir / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{item['id']}.png"
        dest.write_bytes(images[0])
        self._mark_image(item["id"], status="completed", path=str(dest), duration_ms=result.get("duration_ms"), width=result.get("width"), height=result.get("height"), steps=result.get("steps"), cfg_scale=result.get("cfg_scale"))
        self._bump(job_id, completed=1)
        summary = self.get_job(job_id)
        self.events.publish(Event("image.completed", job_id, {"image_id": item["id"], "path": str(dest), "duration_ms": result.get("duration_ms"), "model_resident": bool(result.get("model_resident")), **self._progress(summary)}))

    def _progress(self, job: dict[str, Any]) -> dict[str, int]:
        return {"completed": job["completed_images"], "failed": job["failed_images"], "total": job["total_images"], "completed_images": job["completed_images"], "total_images": job["total_images"]}

    def _set_job(self, job_id: str, **fields: Any) -> None:
        assignments = ["updated_at = ?"]
        args: list[Any] = [_utc_now()]
        for key, value in fields.items():
            assignments.append(f"{key} = ?")
            args.append(value)
        args.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", args)
            conn.commit()

    def _bump(self, job_id: str, completed: int = 0, failed: int = 0) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET completed_images = completed_images + ?, failed_images = failed_images + ?, updated_at = ? WHERE id = ?", (completed, failed, _utc_now(), job_id))
            conn.commit()

    def _mark_image(self, image_id: str, **fields: Any) -> None:
        assignments = []
        args: list[Any] = []
        for key, value in fields.items():
            assignments.append(f"{key} = ?")
            args.append(value)
        args.append(image_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE images SET {', '.join(assignments)} WHERE id = ?", args)
            conn.commit()

    def _with_cost(self, job: dict[str, Any], costs: dict[str, Any] | None = None) -> dict[str, Any]:
        costs = costs if costs is not None else job_totals(client_ledger().read())
        info = costs.get(job["id"]) or {}
        job["cost_usd"] = info.get("billed_usd") or "0"
        job["cost_events"] = info.get("event_count") or 0
        job["cost_chain"] = info.get("chain") or []
        return job

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        config = json.loads(row["config_json"])
        return {"id": row["id"], "status": row["status"], "recipe": row["recipe"], "model": row["recipe"], "gpu": row["gpu"], "created_at": row["created_at"], "updated_at": row["updated_at"], "config": config, "total_images": row["total_images"], "completed_images": row["completed_images"], "failed_images": row["failed_images"], "error": row["error"], "cost_usd": "0"}

    def _image_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "job_id": row["job_id"], "prompt": row["prompt"], "seed": row["seed"], "recipe": row["recipe"], "model": row["recipe"], "width": row["width"], "height": row["height"], "steps": row["steps"], "cfg_scale": row["cfg_scale"], "status": row["status"], "path": row["path"], "duration_ms": row["duration_ms"], "latency_ms": row["duration_ms"], "error": row["error"], "created_at": row["created_at"], "gpu": None}
