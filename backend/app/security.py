import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config
from .db import get_db
from .models import Project, User

ALGORITHM = "HS256"
bearer = HTTPBearer(auto_error=False)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---- passwords (stdlib scrypt, no extra dependency) ----
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except ValueError:
        return False


# ---- tokens ----
def _encode(claims: dict, minutes: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return jwt.encode({**claims, "exp": exp}, config.JWT_SECRET, algorithm=ALGORITHM)


def create_access_token(user_id: int) -> str:
    return _encode({"sub": str(user_id), "scope": "access"}, config.ACCESS_TOKEN_MINUTES)


def create_file_token(project_id: int) -> str:
    """Short-lived token embedded in splat/ply URLs: browser loaders can't set headers."""
    return _encode({"pid": project_id, "scope": "files"}, config.FILE_TOKEN_MINUTES)


def decode_token(token: str, scope: str) -> dict:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    if payload.get("scope") != scope:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token scope")
    return payload


# ---- dependencies ----
def _demo_user(db: Session) -> User:
    user = db.scalar(select(User).where(User.email == "demo@local"))
    if user is None:
        user = User(email="demo@local", password_hash=hash_password(secrets.token_urlsafe(24)))
        db.add(user)
        db.commit()
    return user


def user_from_credentials(creds: HTTPAuthorizationCredentials | None, db: Session) -> User | None:
    if not config.AUTH_REQUIRED:
        return _demo_user(db)
    if creds is None:
        return None
    payload = decode_token(creds.credentials, "access")
    return db.get(User, int(payload["sub"]))


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)
) -> User:
    user = user_from_credentials(creds, db)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_worker(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if creds is None or not secrets.compare_digest(creds.credentials, config.WORKER_TOKEN):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid worker token")


def get_owned_project(db: Session, project_id: int, user: User) -> Project:
    project = db.get(Project, project_id)
    # 404 (not 403) so project ids of other users are not revealed
    if project is None or project.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project
