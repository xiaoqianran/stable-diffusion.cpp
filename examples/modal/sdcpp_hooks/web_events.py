from __future__ import annotations

import os
import queue
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_limit() -> int:
    try:
        return max(16, int(os.environ.get("SDCPP_EVENT_HISTORY", "256")))
    except ValueError:
        return 256


@dataclass
class Event:
    type: str
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Bounded in-process SSE event fanout.

    Durable state lives in SQLite; event history is deliberately bounded so a
    long-running Web process does not retain every historical job forever.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, deque[Event]] = {}
        self._subs: dict[str, list[queue.Queue[Event]]] = {}
        self._limit = _history_limit()

    def publish(self, event: Event) -> None:
        with self._lock:
            history = self._history.setdefault(event.job_id, deque(maxlen=self._limit))
            history.append(event)
            for sub in list(self._subs.get(event.job_id, [])):
                sub.put(event)

    def history(self, job_id: str) -> list[Event]:
        with self._lock:
            return list(self._history.get(job_id, ()))

    def subscribe(self, job_id: str) -> queue.Queue[Event]:
        sub: queue.Queue[Event] = queue.Queue(maxsize=self._limit)
        with self._lock:
            self._subs.setdefault(job_id, []).append(sub)
        return sub

    def unsubscribe(self, job_id: str, sub: queue.Queue[Event]) -> None:
        with self._lock:
            rows = self._subs.get(job_id, [])
            if sub in rows:
                rows.remove(sub)
            if not rows:
                self._subs.pop(job_id, None)

    def prune(self, job_id: str) -> None:
        """Drop terminal history once no subscriber is attached."""
        with self._lock:
            if not self._subs.get(job_id):
                self._history.pop(job_id, None)
