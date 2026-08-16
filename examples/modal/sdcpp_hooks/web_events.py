from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    type: str
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, list[Event]] = {}
        self._subs: dict[str, list[queue.Queue[Event]]] = {}

    def publish(self, event: Event) -> None:
        with self._lock:
            self._history.setdefault(event.job_id, []).append(event)
            for sub in list(self._subs.get(event.job_id, [])):
                sub.put(event)

    def history(self, job_id: str) -> list[Event]:
        with self._lock:
            return list(self._history.get(job_id, []))

    def subscribe(self, job_id: str) -> queue.Queue[Event]:
        sub: queue.Queue[Event] = queue.Queue()
        with self._lock:
            self._subs.setdefault(job_id, []).append(sub)
        return sub

    def unsubscribe(self, job_id: str, sub: queue.Queue[Event]) -> None:
        with self._lock:
            rows = self._subs.get(job_id, [])
            if sub in rows:
                rows.remove(sub)
