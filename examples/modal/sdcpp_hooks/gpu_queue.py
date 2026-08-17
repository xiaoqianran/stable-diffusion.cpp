from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class GPUQueue:
    """FIFO coordinator that mirrors the intended Modal GPU concurrency.

    CPU artifact staging happens before jobs enter this queue. Only jobs that have
    finished staging wait here, so the frontend can report a real GPU wait rather
    than conflating model downloads with GPU contention.
    """

    def __init__(self, max_active: int = 1) -> None:
        self.max_active = max(1, int(max_active))
        self._condition = threading.Condition()
        self._waiting: list[str] = []
        self._running: list[str] = []

    def enqueue(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            if job_id not in self._waiting and job_id not in self._running:
                self._waiting.append(job_id)
                self._condition.notify_all()
            return self._snapshot_unlocked(job_id)

    def acquire(
        self,
        job_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        with self._condition:
            if job_id not in self._waiting and job_id not in self._running:
                self._waiting.append(job_id)
                self._condition.notify_all()

            while True:
                if cancelled is not None and cancelled():
                    if job_id in self._waiting:
                        self._waiting.remove(job_id)
                    self._condition.notify_all()
                    return False

                if job_id in self._running:
                    return True

                if (
                    len(self._running) < self.max_active
                    and self._waiting
                    and self._waiting[0] == job_id
                ):
                    self._waiting.pop(0)
                    self._running.append(job_id)
                    self._condition.notify_all()
                    return True

                self._condition.wait(timeout=0.25)

    def release(self, job_id: str) -> None:
        with self._condition:
            if job_id in self._running:
                self._running.remove(job_id)
            if job_id in self._waiting:
                self._waiting.remove(job_id)
            self._condition.notify_all()

    def cancel(self, job_id: str) -> None:
        """Remove a waiting job; a currently running job releases itself safely."""
        with self._condition:
            if job_id in self._waiting:
                self._waiting.remove(job_id)
            self._condition.notify_all()

    def snapshot(self, job_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_unlocked(job_id)

    def _snapshot_unlocked(self, job_id: str | None = None) -> dict[str, Any]:
        running = list(self._running)
        waiting = list(self._waiting)
        state = "running" if running else ("queued" if waiting else "idle")
        payload: dict[str, Any] = {
            "state": state,
            "max_active": self.max_active,
            "running_count": len(running),
            "running_job_ids": running,
            "running_job_id": running[0] if running else None,
            "queue_length": len(waiting),
            "waiting_job_ids": waiting,
            "total_active": len(running) + len(waiting),
        }

        if job_id is not None:
            if job_id in running:
                job_state = "running"
                ahead = running.index(job_id)
                position = ahead + 1
            elif job_id in waiting:
                job_state = "waiting"
                ahead = len(running) + waiting.index(job_id)
                position = ahead + 1
            else:
                job_state = "idle"
                ahead = 0
                position = None
            payload["job"] = {
                "state": job_state,
                "position": position,
                "ahead": ahead,
                "queue_length": len(waiting),
                "running_count": len(running),
            }
        return payload
