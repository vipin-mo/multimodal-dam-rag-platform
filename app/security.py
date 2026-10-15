"""JWT issuance/verification and role-based access control."""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app import config
from app.database import User, get_db

SECRET_KEY = config.jwt_secret()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# Checked against when the username is unknown, so login timing doesn't reveal valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt()).decode("utf-8")


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = db.get(User, username)
    if user is None:
        User(hashed_password=_DUMMY_HASH).verify_password(password)
        return None
    return user if user.verify_password(password) else None


def create_access_token(username: str, role: str, minutes: int = config.JWT_EXPIRE_MINUTES) -> str:
    now = datetime.now(UTC)
    claims = {"sub": username, "role": role, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(claims, SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[config.JWT_ALGORITHM], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        raise unauthorized
    # The role is re-read from the database, so demoting a user takes effect immediately.
    user = db.get(User, payload["sub"])
    if user is None:
        raise unauthorized
    return user


class RoleChecker:
    """FastAPI dependency: `Depends(RoleChecker(["admin"]))`."""

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role for this action.")
        return user


require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["admin", "viewer"])
