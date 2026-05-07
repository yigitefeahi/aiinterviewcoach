"""Optional Redis-backed sliding window rate limiting (multi-instance)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from .rate_limit import RateLimitConfig

_redis: Any = None


def _get_redis(url: str) -> Any:
    global _redis
    if _redis is None:
        import redis as redis_lib

        _redis = redis_lib.Redis.from_url(url, decode_responses=True)
    return _redis


def is_rate_limited_redis(redis_url: str, key: str, config: RateLimitConfig) -> bool:
    try:
        r = _get_redis(redis_url)
    except Exception:
        return False
    now = time.time()
    cutoff = now - config.window_seconds
    rk = f"rl:{key}"
    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(rk, "-inf", cutoff)
        pipe.zcard(rk)
        results = pipe.execute()
        count = int(results[1]) if len(results) > 1 else 0
        if count >= config.max_requests:
            return True
        member = f"{now}:{uuid.uuid4().hex}"
        pipe = r.pipeline()
        pipe.zadd(rk, {member: now})
        pipe.expire(rk, config.window_seconds + 5)
        pipe.execute()
        return False
    except Exception:
        return False


def reset_redis_client() -> None:
    global _redis
    _redis = None
