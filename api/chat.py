"""Chat endpoints: send, image upload, sessions CRUD."""
import shutil
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from datamodels.chat_schemas import (
    SendMessageRequest, SendMessageResponse,
    SessionItem, SessionDetail, UpdateSessionRequest,
)
from auth.middleware import get_current_user
from services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])
PHOTO_DIR = Path(__file__).resolve().parent.parent / "photo"


def _get_chat_service(request: Request) -> ChatService:
    return ChatService(
        trip_service=request.app.state.trip_service,
        retriever=request.app.state.retriever,
        chat_repo=request.app.state.chat_repo,
    )


@router.post("/send", response_model=SendMessageResponse)
def send_message(body: SendMessageRequest, request: Request):
    user = get_current_user(request)
    chat_service = _get_chat_service(request)
    try:
        result = chat_service.handle_message(
            user_id=user["id"],
            session_id=body.session_id,
            content=body.content,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-image")
def upload_image(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
    message: str = Form(default=""),
):
    """Upload an image for landmark recognition."""
    user = get_current_user(request)

    # Save image to photo dir
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PHOTO_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sid = int(session_id) if session_id else None
    chat_service = _get_chat_service(request)
    try:
        result = chat_service.handle_message(
            user_id=user["id"],
            session_id=sid,
            content=message or "",
            image_filename=file.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
def list_sessions(request: Request, favorite_only: bool = False):
    user = get_current_user(request)
    chat_service = _get_chat_service(request)
    sessions = chat_service.list_sessions(user["id"], favorite_only=favorite_only)
    return sessions


@router.get("/sessions/{session_id}")
def get_session(session_id: int, request: Request):
    user = get_current_user(request)
    chat_service = _get_chat_service(request)
    detail = chat_service.get_session_detail(session_id, user["id"])
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


@router.patch("/sessions/{session_id}")
def update_session(session_id: int, body: UpdateSessionRequest, request: Request):
    user = get_current_user(request)
    chat_service = _get_chat_service(request)
    detail = chat_service.get_session_detail(session_id, user["id"])
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    chat_service.update_session(session_id, title=body.title, is_favorite=body.is_favorite)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, request: Request):
    user = get_current_user(request)
    chat_service = _get_chat_service(request)
    detail = chat_service.get_session_detail(session_id, user["id"])
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    chat_service.delete_session(session_id)
    return {"ok": True}
