from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class GPUQueue:
    """GPU job coordinator with bounded same-model affinity.

    Jobs only enter after CPU/Volume staging. The scheduler remains job-serial by
    default, but when a warm recipe just finished it may pull another matching
    recipe forward from a small window. A streak cap prevents starvation.
    """

    def __init__(
        self,
        max_active: int = 1,
        *,
        affinity_window: int = 8,
        max_affinity_streak: int = 4,
    ) -> None:
        self.max_active = max(1, int(max_active))
        self.affinity_window = max(1, int(affinity_window))
        self.max_affinity_streak = max(1, int(max_affinity_streak))
        self._condition = threading.Condition()
        self._waiting: list[str] = []
        self._running: list[str] = []
        self._affinity: dict[str, str] = {}
        self._last_affinity = ""
        self._affinity_streak = 0

    def enqueue(self, job_id: str, *, affinity_key: str = "") -> dict[str, Any]:
        with self._condition:
            if affinity_key:
                self._affinity[job_id] = affinity_key
            if job_id not in self._waiting and job_id not in self._running:
                self._waiting.append(job_id)
                self._condition.notify_all()
            return self._snapshot_unlocked(job_id)

    def _preferred_affinity_unlocked(self) -> str:
        if self._running:
            return self._affinity.get(self._running[0], "")
        if self._affinity_streak >= self.max_affinity_streak:
            return ""
        return self._last_affinity

    def _ordered_waiting_unlocked(self) -> list[str]:
        waiting = list(self._waiting)
        if len(waiting) < 2:
            return waiting
        preferred = self._preferred_affinity_unlocked()
        if not preferred:
            return waiting
        limit = min(len(waiting), self.affinity_window)
        for index in range(limit):
            if self._affinity.get(waiting[index], "") == preferred:
                if index:
                    waiting.insert(0, waiting.pop(index))
                break
        return waiting

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
                    self._affinity.pop(job_id, None)
                    self._condition.notify_all()
                    return False

                if job_id in self._running:
                    return True

                ordered = self._ordered_waiting_unlocked()
                if (
                    len(self._running) < self.max_active
                    and ordered
                    and ordered[0] == job_id
                ):
                    self._waiting.remove(job_id)
                    self._running.append(job_id)
                    self._condition.notify_all()
                    return True

                self._condition.wait(timeout=0.25)

    def release(self, job_id: str) -> None:
        with self._condition:
            affinity = self._affinity.get(job_id, "")
            if job_id in self._running:
                self._running.remove(job_id)
            if job_id in self._waiting:
                self._waiting.remove(job_id)
            if affinity:
                if affinity == self._last_affinity:
                    self._affinity_streak += 1
                else:
                    self._last_affinity = affinity
                    self._affinity_streak = 1
            else:
                self._last_affinity = ""
                self._affinity_streak = 0
            self._affinity.pop(job_id, None)
            self._condition.notify_all()

    def cancel(self, job_id: str) -> None:
        """Remove a waiting job; a currently running job releases itself safely."""
        with self._condition:
            if job_id in self._waiting:
                self._waiting.remove(job_id)
                self._affinity.pop(job_id, None)
            self._condition.notify_all()

    def snapshot(self, job_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_unlocked(job_id)

    def _snapshot_unlocked(self, job_id: str | None = None) -> dict[str, Any]:
        running = list(self._running)
        waiting = self._ordered_waiting_unlocked()
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
            "affinity": {
                "preferred": self._preferred_affinity_unlocked(),
                "last": self._last_affinity,
                "streak": self._affinity_streak,
                "streak_limit": self.max_affinity_streak,
                "window": self.affinity_window,
            },
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
                "affinity_key": self._affinity.get(job_id, ""),
            }
        return payload
