from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .recipes import RECIPES
from .web_catalog import default_recipe, normalize_gpu
from .web_events import Event, EventBus
from .web_generator import MockGenerator, ModalGenerator


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


class JobService:
    def __init__(self, data_dir: Path, events: EventBus | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.outputs_dir = self.data_dir / "outputs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "sdcpp-web.db"
        self.events = events or EventBus()
        self._lock = threading.Lock()
        self._cancel: set[str] = set()
        self._threads: dict[str, threading.Thread] = {}
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
        gpu = normalize_gpu(str(config.get("gpu") or "L40S"))
        count = max(1, int(config.get("count") or 1))
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
        images: list[tuple[str, str, int]] = []
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
                    (
                        _new_id("img"),
                        job_id,
                        prompt,
                        seed,
                        item_recipe,
                        width,
                        height,
                        steps,
                        cfg_scale,
                        now,
                    ),
                )
            conn.commit()
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
        self.start(job_id)
        return self.get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self._cancel.add(job_id)
        self._set_job(job_id, status="cancelled")
        summary = self.get_job(job_id)
        self.events.publish(Event("job.cancelled", job_id, summary))
        return summary

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._job_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_row(row)

    def get_job_detail(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE job_id = ? ORDER BY created_at",
                (job_id,),
            ).fetchall()
        return {"job": job, "images": [self._image_row(row) for row in rows]}

    def list_images(
        self,
        *,
        job_id: str | None = None,
        recipe: str | None = None,
        q: str | None = None,
        sort: str = "newest",
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
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
        order = {
            "oldest": "created_at ASC",
            "fastest": "duration_ms ASC",
            "slowest": "duration_ms DESC",
        }.get(sort, "created_at DESC")
        clause = " AND ".join(where)
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM images WHERE {clause}", args).fetchone()[0]
            offset = max(0, (page - 1) * per_page)
            rows = conn.execute(
                f"SELECT * FROM images WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?",
                [*args, per_page, offset],
            ).fetchall()
        return {
            "items": [self._image_row(row) for row in rows],
            "total": total,
            "page": page,
            "per_page": per_page,
        }

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
        config["seed"] = image["seed"]
        config["recipe"] = image["recipe"]
        created = self.create_job([{"prompt": image["prompt"], "seed": image["seed"], "count": 1}], config)
        self.start(created["id"])
        return created

    def _run(self, job_id: str) -> None:
        job = self.get_job(job_id)
        config = job["config"]
        self._set_job(job_id, status="running")
        self.events.publish(Event("job.started", job_id, self.get_job(job_id)))
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT * FROM images WHERE job_id = ? AND status IN ('pending', 'failed')",
                (job_id,),
            ).fetchall()
        items = [dict(row) for row in pending]
        dry_run = bool(config.get("dry_run"))
        try:
            if dry_run:
                mock = MockGenerator()
                for item in items:
                    if job_id in self._cancel:
                        return
                    self._finish_one(
                        job_id,
                        item,
                        mock.generate_one(
                            prompt=item["prompt"],
                            recipe=item["recipe"],
                            seed=item["seed"],
                            width=item["width"],
                            height=item["height"],
                            steps=item["steps"],
                            cfg_scale=item["cfg_scale"],
                            gpu=job["gpu"],
                        ),
                    )
            else:
                modal = ModalGenerator()
                results = modal.generate_job(
                    [
                        {
                            "id": item["id"],
                            "prompt": item["prompt"],
                            "recipe": item["recipe"],
                            "seed": item["seed"],
                            "width": item["width"],
                            "height": item["height"],
                            "steps": item["steps"],
                            "cfg_scale": item["cfg_scale"],
                        }
                        for item in items
                    ],
                    gpu=job["gpu"],
                )
                by_id = {item["id"]: item for item in items}
                for result in results:
                    if job_id in self._cancel:
                        return
                    self._finish_one(job_id, by_id[result["id"]], result)
        except Exception as exc:
            self._set_job(job_id, status="failed", error=str(exc))
            self.events.publish(Event("job.failed", job_id, {**self.get_job(job_id), "error": str(exc)}))
            return
        summary = self.get_job(job_id)
        status = "completed" if summary["failed_images"] == 0 else "failed"
        self._set_job(job_id, status=status)
        self.events.publish(Event(f"job.{status}", job_id, self.get_job(job_id)))

    def _finish_one(self, job_id: str, item: dict[str, Any], result: dict[str, Any]) -> None:
        images = result.get("images") or []
        if not images:
            self._mark_image(item["id"], status="failed", error="no image returned")
            self._bump(job_id, failed=1)
            summary = self.get_job(job_id)
            self.events.publish(
                Event(
                    "image.failed",
                    job_id,
                    {"image_id": item["id"], "error": "no image returned", **self._progress(summary)},
                )
            )
            return
        dest_dir = self.outputs_dir / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{item['id']}.png"
        dest.write_bytes(images[0])
        self._mark_image(
            item["id"],
            status="completed",
            path=str(dest),
            duration_ms=result.get("duration_ms"),
            width=result.get("width"),
            height=result.get("height"),
            steps=result.get("steps"),
            cfg_scale=result.get("cfg_scale"),
        )
        self._bump(job_id, completed=1)
        summary = self.get_job(job_id)
        self.events.publish(
            Event(
                "image.completed",
                job_id,
                {
                    "image_id": item["id"],
                    "path": str(dest),
                    "duration_ms": result.get("duration_ms"),
                    **self._progress(summary),
                },
            )
        )

    def _progress(self, job: dict[str, Any]) -> dict[str, int]:
        return {
            "completed": job["completed_images"],
            "failed": job["failed_images"],
            "total": job["total_images"],
            "completed_images": job["completed_images"],
            "total_images": job["total_images"],
        }

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
            conn.execute(
                """UPDATE jobs
                   SET completed_images = completed_images + ?,
                       failed_images = failed_images + ?,
                       updated_at = ?
                   WHERE id = ?""",
                (completed, failed, _utc_now(), job_id),
            )
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

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        config = json.loads(row["config_json"])
        return {
            "id": row["id"],
            "status": row["status"],
            "recipe": row["recipe"],
            "model": row["recipe"],
            "gpu": row["gpu"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "config": config,
            "total_images": row["total_images"],
            "completed_images": row["completed_images"],
            "failed_images": row["failed_images"],
            "error": row["error"],
            "cost_usd": None,
        }

    def _image_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "job_id": row["job_id"],
            "prompt": row["prompt"],
            "seed": row["seed"],
            "recipe": row["recipe"],
            "model": row["recipe"],
            "width": row["width"],
            "height": row["height"],
            "steps": row["steps"],
            "cfg_scale": row["cfg_scale"],
            "status": row["status"],
            "path": row["path"],
            "duration_ms": row["duration_ms"],
            "latency_ms": row["duration_ms"],
            "error": row["error"],
            "created_at": row["created_at"],
            "gpu": None,
        }
