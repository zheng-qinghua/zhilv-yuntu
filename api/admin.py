"""Admin endpoints: view guide chunks."""
from fastapi import APIRouter, Request
from auth.middleware import get_current_admin_user

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/chunks")
def list_chunks(request: Request):
    get_current_admin_user(request)  # raises 403 if not admin
    chat_repo = request.app.state.chat_repo
    return chat_repo.list_all_chunks()
