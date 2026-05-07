from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

security_scheme = HTTPBearer(auto_error=False)


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.jwt_expires_minutes)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def _token_from_request(request: Request, creds: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    if creds and creds.credentials:
        return creds.credentials
    return request.cookies.get("access_token")


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> User:
    token = _token_from_request(request, creds)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
