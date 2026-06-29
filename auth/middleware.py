"""Session middleware using signed cookies."""
import hashlib
import hmac
import os
import time
from fastapi import Request, HTTPException
from database.user_repo import UserRepo

# Random signing key per server start (invalidates all sessions on restart)
_SIGNING_KEY = os.urandom(32).hex()
_COOKIE_NAME = "session_token"
_TOKEN_SEP = ":"


def _sign(payload: str) -> str:
    sig = hmac.new(_SIGNING_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}{_TOKEN_SEP}{sig}"


def _verify(token: str) -> str | None:
    parts = token.rsplit(_TOKEN_SEP, 1)
    if len(parts) != 2:
        return None
    payload, sig = parts
    expected_sig = hmac.new(_SIGNING_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    return payload


def create_session_token(user_id: int) -> str:
    payload = f"{user_id}{_TOKEN_SEP}{int(time.time())}"
    return _sign(payload)


def get_user_id_from_token(token: str) -> int | None:
    payload = _verify(token)
    if payload is None:
        return None
    try:
        uid_str = payload.split(_TOKEN_SEP)[0]
        return int(uid_str)
    except (ValueError, IndexError):
        return None


def get_current_user(request: Request) -> dict:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    user_id = get_user_id_from_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="登录已过期")
    user_repo = UserRepo()
    user = user_repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def get_current_admin_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="权限不足，需要管理员身份")
    return user
