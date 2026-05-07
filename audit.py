import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


AUDIT_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.log"


def audit_event(
    event: str,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user_id": user_id,
        "session_id": session_id,
        "detail": detail or {},
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
