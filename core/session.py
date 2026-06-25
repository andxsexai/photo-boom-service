"""Session management with strict 5-processing limit per photo."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from core.filters import Style


@dataclass
class Session:
    id: str
    limit: int = 5
    used: int = 0
    processed_styles: set[Style] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def can_process(self, style: Style | None = None) -> bool:
        if self.remaining <= 0:
            return False
        if style and style in self.processed_styles:
            return False
        return True

    def consume(self, style: Style) -> None:
        if not self.can_process(style):
            raise SessionLimitError(
                f"Session {self.id}: limit reached or style {style.value} already processed"
            )
        self.used += 1
        self.processed_styles.add(style)


class SessionLimitError(Exception):
    pass


class SessionManager:
    def __init__(self, limit: int = 5, ttl_seconds: int = 3600):
        self.limit = limit
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create(self) -> Session:
        session = Session(id=str(uuid.uuid4()), limit=self.limit)
        with self._lock:
            self._cleanup_expired()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._cleanup_expired()
            return self._sessions.get(session_id)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.created_at > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]
