import base64
import hashlib
import hmac
import os

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

PBKDF2_PREFIX = "pbkdf2_sha256"
PBKDF2_ROUNDS = 120_000


def _hash_password_pbkdf2(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return (
        f"{PBKDF2_PREFIX}${PBKDF2_ROUNDS}$"
        f"{base64.urlsafe_b64encode(salt).decode('utf-8')}$"
        f"{base64.urlsafe_b64encode(dk).decode('utf-8')}"
    )


def _verify_password_pbkdf2(password: str, stored_hash: str) -> bool:
    try:
        prefix, rounds, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if prefix != PBKDF2_PREFIX:
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("utf-8"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("utf-8"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def hash_password(password: str) -> str:
    try:
        return pwd_context.hash(password)
    except Exception:
        return _hash_password_pbkdf2(password)


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith(f"{PBKDF2_PREFIX}$"):
        return _verify_password_pbkdf2(password, password_hash)
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False