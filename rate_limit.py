from collections import defaultdict, deque
from dataclasses import dataclass
from random import random
from time import time
from typing import Optional

from sqlalchemy import delete
from sqlalchemy.orm import Session

from .config import settings
from .models import RateLimitEvent


@dataclass(frozen=True)
class RateLimitConfig:
    max_requests: int
    window_seconds: int


_REQUESTS: dict[str, deque[float]] = defaultdict(deque)


def _prune(queue: deque[float], now_ts: float, window_seconds: int) -> None:
    cutoff = now_ts - window_seconds
    while queue and queue[0] < cutoff:
        queue.popleft()


def is_rate_limited(key: str, config: RateLimitConfig) -> bool:
    """In-memory sliding window (resets on process restart)."""
    now_ts = time()
    queue = _REQUESTS[key]
    _prune(queue, now_ts, config.window_seconds)
    if len(queue) >= config.max_requests:
        return True
    queue.append(now_ts)
    return False


def is_rate_limited_persistent(db: Session, key: str, config: RateLimitConfig) -> bool:
    """DB-backed sliding window (survives restarts; best-effort prune)."""
    now_ts = time()
    cutoff = now_ts - config.window_seconds
    if random() < 0.03:
        db.execute(delete(RateLimitEvent).where(RateLimitEvent.ts < now_ts - 86_400))
        db.commit()

    count = (
        db.query(RateLimitEvent)
        .filter(RateLimitEvent.scope_key == key, RateLimitEvent.ts >= cutoff)
        .count()
    )
    if count >= config.max_requests:
        return True
    db.add(RateLimitEvent(scope_key=key, ts=now_ts))
    db.commit()
    return False


def enforce_with_backend(
    db: Optional[Session],
    *,
    use_persistent: bool,
    key: str,
    config: RateLimitConfig,
) -> bool:
    redis_url = (settings.redis_url or "").strip()
    if redis_url:
        from .rate_limit_redis import is_rate_limited_redis

        return bool(is_rate_limited_redis(redis_url, key, config))
    if use_persistent and db is not None:
        return is_rate_limited_persistent(db, key, config)
    return is_rate_limited(key, config)
