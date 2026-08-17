from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .cost import PriceBook, default_plan
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
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    cost_events INTEGER NOT NULL DEFAULT 0,
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
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS modal_calls (
    image_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_job_created ON images(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_images_status_created ON images(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_recipe_status_created ON images(recipe, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_modal_calls_job_status ON modal_calls(job_id, status);
"""

TERMINAL = {"completed", "failed", "cancelled"}
RECOVERABLE_IMAGE_STATES = ("pending", "failed", "running")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _gpu_job_queue_limit() -> int:
    return _positive_int_env("SDCPP_GPU_JOB_MAX_ACTIVE", 1)


def _cpu_job_workers() -> int:
    return _positive_int_env("SDCPP_CPU_JOB_WORKERS", 4)


def _affinity_key(job: dict[str, Any]) -> str:
    return f"{job['gpu']}::{job['recipe']}"


class JobService:
    """Durable local job coordinator around persisted Modal FunctionCall IDs."""

    def __init__(self, data_dir: Path, events: EventBus | None = None, *, auto_recover: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.outputs_dir = self.data_dir / "outputs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "sdcpp-web.db"
        self.events = events or EventBus()
        self._lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._cancel: set[str] = set()
        self._futures: dict[str, Future[Any]] = {}
        self._phases: dict[str, str] = {}
        self._model_resident: dict[str, bool] = {}
        self._gpu_queue = GPUQueue(max_active=_gpu_job_queue_limit())
        self._executor = ThreadPoolExecutor(max_workers=_cpu_job_workers(), thread_name_prefix="sdcpp-job")
        self._init_db()
        if auto_recover:
            self._reconcile_startup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            self._migrate_columns(conn)
            conn.commit()

    @staticmethod
    def _migrate_columns(conn: sqlite3.Connection) -> None:
        jobs = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        images = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
        if "estimated_cost_usd" not in jobs:
            conn.execute("ALTER TABLE jobs ADD COLUMN estimated_cost_usd REAL NOT NULL DEFAULT 0")
        if "cost_events" not in jobs:
            conn.execute("ALTER TABLE jobs ADD COLUMN cost_events INTEGER NOT NULL DEFAULT 0")
        if "estimated_cost_usd" not in images:
            conn.execute("ALTER TABLE images ADD COLUMN estimated_cost_usd REAL NOT NULL DEFAULT 0")

    def _reconcile_startup(self) -> None:
        # A dead local process must never leave a permanently "running" UI row.
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status NOT IN ('completed','failed','cancelled') ORDER BY created_at"
            ).fetchall()
            for row in rows:
                conn.execute("UPDATE jobs SET status='pending', updated_at=? WHERE id=?", (_utc_now(), row["id"]))
            conn.commit()
        for row in rows:
            self._set_phase(row["id"], "recovering")
            self.start(row["id"])

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

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
        cfg_scale = float(config.get("cfg_scale") if config.get("cfg_scale") is not None else defaults.get("cfg_scale") or 1.0)
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
            item_recipe = str(spec.get("recipe") or recipe)
            if item_recipe != recipe:
                raise ValueError("one job must use a single recipe")
            for offset in range(n):
                images.append((prompt, start + offset, item_recipe))
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
                   (id,status,recipe,gpu,created_at,updated_at,config_json,total_images,
                    completed_images,failed_images,estimated_cost_usd,cost_events)
                   VALUES (?,?,?,?,?,?,?,?,0,0,0,0)""",
                (job_id, "pending", recipe, gpu, now, now, json.dumps(payload), len(images)),
            )
            for prompt, seed, item_recipe in images:
                conn.execute(
                    """INSERT INTO images
                       (id,job_id,prompt,seed,recipe,width,height,steps,cfg_scale,status,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,'pending',?)""",
                    (_new_id("img"), job_id, prompt, seed, item_recipe, width, height, steps, cfg_scale, now),
                )
            conn.commit()
        self._set_phase(job_id, "pending")
        summary = self.get_job(job_id)
        self.events.publish(Event("job.snapshot", job_id, summary))
        return summary

    def start(self, job_id: str) -> None:
        self._raw_job(job_id)  # validate before creating a Future
        with self._lock:
            future = self._futures.get(job_id)
            if future is not None and not future.done():
                return
            future = self._executor.submit(self._run, job_id)
            self._futures[job_id] = future
            future.add_done_callback(lambda _f, jid=job_id: self._forget_future(jid))

    def _forget_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
        try:
            terminal = self._raw_job(job_id)["status"] in TERMINAL
        except KeyError:
            terminal = True
        if terminal:
            self._terminal_cleanup(job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        self._cancel.discard(job_id)
        with self._connect() as conn:
            conn.execute("UPDATE images SET status='pending', error=NULL WHERE job_id=? AND status IN ('failed','cancelled')", (job_id,))
            conn.execute("UPDATE jobs SET failed_images=0, status='pending', error=NULL, updated_at=? WHERE id=?", (_utc_now(), job_id))
            conn.commit()
        self._set_phase(job_id, "recovering" if self._active_call_map(job_id) else "pending")
        self.start(job_id)
        return self.get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self._raw_job(job_id)
        self._cancel.add(job_id)
        self._gpu_queue.cancel(job_id)
        errors = self._cancel_remote_calls(job_id)
        self._publish_queue_state()
        with self._lock:
            future = self._futures.get(job_id)
            active_local = future is not None and not future.done()
        if errors:
            message = "Remote cancellation could not be confirmed: " + "; ".join(errors)
            self._set_phase(job_id, "failed")
            self._set_job(job_id, status="failed", error=message)
            summary = self.get_job(job_id)
            summary["cancel_confirmed"] = False
            self.events.publish(Event("job.failed", job_id, summary))
            self._terminal_cleanup(job_id, keep_cancel=active_local)
            return summary
        self._set_phase(job_id, "cancelled")
        self._set_job(job_id, status="cancelled")
        summary = self.get_job(job_id)
        summary["cancel_confirmed"] = True
        self.events.publish(Event("job.cancelled", job_id, summary))
        self._terminal_cleanup(job_id, keep_cancel=active_local)
        return summary

    def _cancel_remote_calls(self, job_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT image_id,call_id FROM modal_calls WHERE job_id=? AND status IN ('running','result_ready')",
                (job_id,),
            ).fetchall()
        if not rows:
            return []
        try:
            import modal
        except ImportError as exc:
            return [f"Modal SDK unavailable: {exc}"]
        errors: list[str] = []
        for row in rows:
            try:
                modal.FunctionCall.from_id(row["call_id"]).cancel()
            except Exception as exc:
                errors.append(f"{row['call_id']}: {exc}")
                continue
            self._set_call_status(row["image_id"], "cancelled")
            self._mark_image(row["image_id"], status="cancelled")
        return errors

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
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_row(row)

    def list_jobs(self, *, limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        limit = min(max(1, int(limit)), 200)
        offset = max(0, int(offset))
        args: list[Any] = []
        where = ""
        if status:
            where = "WHERE status=?"
            args.append(status)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [self._attach_runtime(self._job_row(row)) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._attach_runtime(self._raw_job(job_id))

    def get_job_detail(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        # Detailed trace remains available on demand; list_jobs never scans the ledger.
        costs = job_totals(client_ledger().read()).get(job_id) or {}
        job["cost_chain"] = costs.get("chain") or []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM images WHERE job_id=? ORDER BY created_at", (job_id,)).fetchall()
        return {"job": job, "images": [self._image_row(row) for row in rows]}

    def list_images(self, *, job_id: str | None = None, recipe: str | None = None, q: str | None = None, sort: str = "newest", page: int = 1, per_page: int = 50) -> dict[str, Any]:
        where = ["status='completed'", "path IS NOT NULL"]
        args: list[Any] = []
        if job_id:
            where.append("job_id=?"); args.append(job_id)
        if recipe:
            where.append("recipe=?"); args.append(recipe)
        if q:
            where.append("prompt LIKE ?"); args.append(f"%{q}%")
        order = {"oldest":"created_at ASC","fastest":"duration_ms ASC","slowest":"duration_ms DESC"}.get(sort,"created_at DESC")
        clause = " AND ".join(where)
        per_page = min(max(1, int(per_page)), 200)
        page = max(1, int(page))
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM images WHERE {clause}", args).fetchone()[0]
            offset = (page - 1) * per_page
            rows = conn.execute(f"SELECT * FROM images WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?", [*args, per_page, offset]).fetchall()
        return {"items":[self._image_row(row) for row in rows],"total":total,"page":page,"per_page":per_page}

    def get_image(self, image_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(image_id)
        return self._image_row(row)

    def regenerate(self, image_id: str) -> dict[str, Any]:
        image = self.get_image(image_id)
        job = self.get_job(image["job_id"])
        config = dict(job["config"])
        config.update(count=1, parallelism=1, seed=image["seed"], recipe=image["recipe"])
        created = self.create_job([{"prompt":image["prompt"],"seed":image["seed"],"count":1}], config)
        self.start(created["id"])
        return created

    def _run(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
        except KeyError:
            return
        if job["status"] == "cancelled" or job_id in self._cancel:
            return
        config = job["config"]
        parallelism = int(config.get("parallelism") or 1)
        self._set_job(job_id, status="running", error=None)
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT * FROM images WHERE job_id=? AND status IN ('pending','failed','running') ORDER BY created_at",
                (job_id,),
            ).fetchall()
        items = [dict(row) for row in pending]
        dry_run = bool(config.get("dry_run"))
        acquired_gpu = False
        queued_gpu = False

        try:
            if not items:
                self._finalize_job(job_id)
                return
            if dry_run:
                self._set_phase(job_id, "running")
                self.events.publish(Event("job.started", job_id, self.get_job(job_id)))
                mock = MockGenerator()
                with bind_task(job_id=job_id, recipe=job["recipe"], gpu=job["gpu"]):
                    record_event(default_plan("dry_run"), phase="local", extra={"call":"dry_run"})
                self._add_cost(job_id, Decimal("0"), events=1)
                for item in items:
                    if self._is_cancelled(job_id): return
                    result = mock.generate_one(prompt=item["prompt"],recipe=item["recipe"],seed=item["seed"],width=item["width"],height=item["height"],steps=item["steps"],cfg_scale=item["cfg_scale"],gpu=job["gpu"])
                    self._finish_one(job_id, item, result)
            else:
                modal_gen = ModalGenerator()
                generation_items = [{"id":item["id"],"prompt":item["prompt"],"recipe":item["recipe"],"seed":item["seed"],"width":item["width"],"height":item["height"],"steps":item["steps"],"cfg_scale":item["cfg_scale"]} for item in items]
                existing = self._active_call_map(job_id)
                if not existing:
                    self._set_phase(job_id, "preparing")
                    self.events.publish(Event("job.preparing", job_id, self.get_job(job_id)))
                else:
                    self._set_phase(job_id, "recovering")
                    self.events.publish(Event("job.recovering", job_id, self.get_job(job_id)))
                requests = modal_gen.prepare_job(generation_items, gpu=job["gpu"], job_id=job_id)
                if self._is_cancelled(job_id): return

                self._set_phase(job_id, "gpu_queued")
                self._gpu_queue.enqueue(job_id, affinity_key=_affinity_key(job)); queued_gpu=True
                self._publish_queue_state()
                self.events.publish(Event("job.gpu_queued", job_id, self.get_job(job_id)))
                acquired_gpu = self._gpu_queue.acquire(job_id, cancelled=lambda:self._is_cancelled(job_id))
                if not acquired_gpu: return
                queued_gpu=False
                self._set_phase(job_id, "gpu_running")
                self._publish_queue_state()
                self.events.publish(Event("job.gpu_started", job_id, self.get_job(job_id)))

                by_id = {item["id"]:item for item in items}
                results = modal_gen.generate_prepared_job(
                    generation_items, requests, gpu=job["gpu"], job_id=job_id,
                    parallelism=parallelism, existing_calls=existing,
                    on_spawn=lambda image_id,call_id:self._store_call(job_id,image_id,call_id),
                    on_done=self._remote_call_done,
                    cancelled=lambda:self._is_cancelled(job_id),
                )
                for result in results:
                    if self._is_cancelled(job_id): return
                    if result.get("model_resident"):
                        with self._runtime_lock: self._model_resident[job_id]=True
                    self._finish_one(job_id, by_id[result["id"]], result)
                    self._set_call_status(result["id"], "completed")
        except Exception as exc:
            if self._is_cancelled(job_id):
                return
            self._set_phase(job_id,"failed")
            self._set_job(job_id,status="failed",error=str(exc))
            self.events.publish(Event("job.failed",job_id,{**self.get_job(job_id),"error":str(exc)}))
            return
        finally:
            if acquired_gpu:
                self._gpu_queue.release(job_id); self._publish_queue_state()
            elif queued_gpu:
                self._gpu_queue.cancel(job_id); self._publish_queue_state()

        self._finalize_job(job_id)

    def _finalize_job(self, job_id: str) -> None:
        if self._is_cancelled(job_id): return
        summary = self.get_job(job_id)
        status = "completed" if summary["failed_images"] == 0 and summary["completed_images"] >= summary["total_images"] else "failed"
        self._set_phase(job_id,status)
        self._set_job(job_id,status=status)
        self.events.publish(Event(f"job.{status}",job_id,self.get_job(job_id)))
        self._terminal_cleanup(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        if job_id in self._cancel: return True
        try:
            return self._raw_job(job_id)["status"] == "cancelled"
        except KeyError:
            return True

    def _store_call(self, job_id: str, image_id: str, call_id: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO modal_calls(image_id,job_id,call_id,status,created_at,updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(image_id) DO UPDATE SET call_id=excluded.call_id,status='running',updated_at=excluded.updated_at""",
                (image_id,job_id,call_id,"running",now,now),
            )
            conn.execute("UPDATE images SET status='running', error=NULL WHERE id=?", (image_id,))
            conn.commit()
        self.events.publish(Event("image.remote_started", job_id, {"image_id":image_id,"call_id":call_id}))

    def _remote_call_done(self, image_id: str, status: str) -> None:
        # Keep completed remote results attachable until the PNG is durably written.
        self._set_call_status(image_id, "result_ready" if status == "completed" else status)

    def _set_call_status(self, image_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE modal_calls SET status=?,updated_at=? WHERE image_id=?", (status,_utc_now(),image_id))
            conn.commit()

    def _active_call_map(self, job_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT image_id,call_id FROM modal_calls WHERE job_id=? AND status IN ('running','result_ready')", (job_id,)).fetchall()
        return {row["image_id"]:row["call_id"] for row in rows}

    def _active_call_count(self, job_id: str) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM modal_calls WHERE job_id=? AND status IN ('running','result_ready')", (job_id,)).fetchone()[0])

    def _terminal_cleanup(self, job_id: str, *, keep_cancel: bool = False) -> None:
        with self._runtime_lock:
            self._phases.pop(job_id,None)
            self._model_resident.pop(job_id,None)
        if not keep_cancel:
            self._cancel.discard(job_id)
        self.events.prune(job_id)

    def _publish_queue_state(self) -> None:
        snapshot = self._gpu_queue.snapshot()
        targets = list(dict.fromkeys([*snapshot["running_job_ids"],*snapshot["waiting_job_ids"]]))
        for target in targets:
            try: payload=self.get_job(target)
            except KeyError: continue
            self.events.publish(Event("gpu.queue",target,payload))

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._runtime_lock: self._phases[job_id]=phase

    def _attach_runtime(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._runtime_lock:
            phase=self._phases.get(job["id"]); model_resident=bool(self._model_resident.get(job["id"]))
        queue=self._gpu_queue.snapshot(job["id"]); job_queue=queue.get("job") or {}
        if phase is None:
            if job["status"] in TERMINAL: phase=job["status"]
            elif job_queue.get("state")=="running": phase="gpu_running"
            elif job_queue.get("state")=="waiting": phase="gpu_queued"
            else: phase=job["status"]
        job.update(phase=phase,queue=job_queue,parallelism=int(job["config"].get("parallelism") or 1),affinity_key=_affinity_key(job),model_resident=model_resident,remote_calls_active=self._active_call_count(job["id"]))
        return job

    def _finish_one(self, job_id: str, item: dict[str, Any], result: dict[str, Any]) -> None:
        images=result.get("images") or []
        if not images:
            self._mark_image(item["id"],status="failed",error="no image returned")
            self._bump(job_id,failed=1)
            summary=self.get_job(job_id)
            self.events.publish(Event("image.failed",job_id,{"image_id":item["id"],"error":"no image returned",**self._progress(summary)}))
            return
        dest_dir=self.outputs_dir/job_id; dest_dir.mkdir(parents=True,exist_ok=True)
        dest=dest_dir/f"{item['id']}.png"; temp=dest.with_suffix(".png.partial")
        temp.write_bytes(images[0]); os.replace(temp,dest)
        duration_ms=int(result.get("duration_ms") or 0)
        estimate=Decimal("0")
        if not result.get("dry_run") and duration_ms > 0:
            estimate,_=PriceBook(source="static-estimate").estimate(default_plan("generate",gpu=self._raw_job(job_id)["gpu"]), duration_ms/1000.0)
        self._mark_image(item["id"],status="completed",path=str(dest),duration_ms=duration_ms,width=result.get("width"),height=result.get("height"),steps=result.get("steps"),cfg_scale=result.get("cfg_scale"),estimated_cost_usd=float(estimate),error=None)
        self._bump(job_id,completed=1)
        self._add_cost(job_id,estimate,events=1 if not result.get("dry_run") else 0)
        if not result.get("dry_run"):
            job = self._raw_job(job_id)
            with bind_task(job_id=job_id, image_id=item["id"], recipe=job["recipe"], gpu=job["gpu"]):
                record_event(
                    default_plan("generate", gpu=job["gpu"]),
                    phase="remote",
                    duration_ms=duration_ms,
                    extra={"call": "generate", "estimate": True},
                )
        summary=self.get_job(job_id)
        self.events.publish(Event("image.completed",job_id,{"image_id":item["id"],"path":str(dest),"duration_ms":duration_ms,"estimated_cost_usd":str(estimate),"model_resident":bool(result.get("model_resident")),**self._progress(summary)}))

    @staticmethod
    def _progress(job: dict[str, Any]) -> dict[str,int]:
        return {"completed":job["completed_images"],"failed":job["failed_images"],"total":job["total_images"],"completed_images":job["completed_images"],"total_images":job["total_images"]}

    def _set_job(self, job_id: str, **fields: Any) -> None:
        allowed={"status","error"}
        unknown=set(fields)-allowed
        if unknown: raise ValueError(f"unsupported job fields: {sorted(unknown)}")
        assignments=["updated_at=?"]; args:[Any]=[_utc_now()]
        for key,value in fields.items(): assignments.append(f"{key}=?"); args.append(value)
        args.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?",args); conn.commit()

    def _bump(self, job_id: str, completed: int=0, failed: int=0) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET completed_images=completed_images+?,failed_images=failed_images+?,updated_at=? WHERE id=?",(completed,failed,_utc_now(),job_id)); conn.commit()

    def _add_cost(self, job_id: str, usd: Decimal, *, events: int=1) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET estimated_cost_usd=estimated_cost_usd+?,cost_events=cost_events+?,updated_at=? WHERE id=?",(float(usd),events,_utc_now(),job_id)); conn.commit()

    def _mark_image(self, image_id: str, **fields: Any) -> None:
        allowed={"status","path","duration_ms","error","width","height","steps","cfg_scale","estimated_cost_usd"}
        unknown=set(fields)-allowed
        if unknown: raise ValueError(f"unsupported image fields: {sorted(unknown)}")
        assignments=[]; args:list[Any]=[]
        for key,value in fields.items(): assignments.append(f"{key}=?"); args.append(value)
        if not assignments: return
        args.append(image_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE images SET {', '.join(assignments)} WHERE id=?",args); conn.commit()

    def cleanup(self, *, keep_days: int=30) -> dict[str,int]:
        """Delete terminal jobs older than the retention window and their output files."""
        keep_days=max(1,int(keep_days))
        with self._connect() as conn:
            rows=conn.execute("SELECT id FROM jobs WHERE status IN ('completed','failed','cancelled') AND julianday('now')-julianday(updated_at)>?",(keep_days,)).fetchall()
            ids=[row["id"] for row in rows]
            for job_id in ids: conn.execute("DELETE FROM jobs WHERE id=?",(job_id,))
            conn.commit()
        for job_id in ids:
            import shutil
            shutil.rmtree(self.outputs_dir/job_id,ignore_errors=True)
        return {"deleted_jobs":len(ids)}

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        config=json.loads(row["config_json"])
        cost=str(Decimal(str(row["estimated_cost_usd"] or 0)))
        return {"id":row["id"],"status":row["status"],"recipe":row["recipe"],"model":row["recipe"],"gpu":row["gpu"],"created_at":row["created_at"],"updated_at":row["updated_at"],"config":config,"total_images":row["total_images"],"completed_images":row["completed_images"],"failed_images":row["failed_images"],"error":row["error"],"cost_usd":cost,"estimated_cost_usd":cost,"cost_events":int(row["cost_events"] or 0),"cost_kind":"estimate"}

    def _image_row(self, row: sqlite3.Row) -> dict[str, Any]:
        cost=str(Decimal(str(row["estimated_cost_usd"] or 0)))
        return {"id":row["id"],"job_id":row["job_id"],"prompt":row["prompt"],"seed":row["seed"],"recipe":row["recipe"],"model":row["recipe"],"width":row["width"],"height":row["height"],"steps":row["steps"],"cfg_scale":row["cfg_scale"],"status":row["status"],"path":row["path"],"duration_ms":row["duration_ms"],"latency_ms":row["duration_ms"],"error":row["error"],"created_at":row["created_at"],"gpu":None,"cost_usd":cost,"estimated_cost_usd":cost,"cost_kind":"estimate"}
