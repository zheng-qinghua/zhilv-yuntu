"""Auth endpoints: register, login, logout, me."""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from auth.service import AuthService
from auth.middleware import create_session_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])
auth_service = AuthService()


class AuthBody(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(body: AuthBody):
    try:
        user = auth_service.register(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_session_token(user["id"])
    resp = JSONResponse(content=user)
    resp.set_cookie("session_token", token, httponly=True, max_age=86400 * 30, samesite="lax")
    return resp


@router.post("/login")
def login(body: AuthBody):
    try:
        user = auth_service.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    token = create_session_token(user["id"])
    resp = JSONResponse(content=user)
    resp.set_cookie("session_token", token, httponly=True, max_age=86400 * 30, samesite="lax")
    return resp


@router.post("/logout")
def logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie("session_token")
    return resp


@router.get("/me")
def me(request: Request):
    try:
        user = get_current_user(request)
    except HTTPException:
        return JSONResponse(content=None, status_code=401)
    return user
