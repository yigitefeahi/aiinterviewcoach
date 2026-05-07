"""Stateless CSRF tokens (HMAC-signed random payload)."""

from __future__ import annotations

import hashlib
import hmac
import secrets

from .config import settings


def issue_csrf_token() -> str:
    raw = secrets.token_urlsafe(32)
    sig = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{raw}.{sig}"


def verify_csrf_token(token: str) -> bool:
    if not token or "." not in token:
        return False
    raw, sig = token.rsplit(".", 1)
    if len(raw) < 8 or len(sig) != 64:
        return False
    expected = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(sig, expected)
