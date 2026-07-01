"""Admin endpoints: view guide chunks."""
from fastapi import APIRouter, Request, HTTPException
from auth.middleware import get_current_admin_user

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/chunks")
def list_chunks(request: Request):
    get_current_admin_user(request)
    chunk_repo = request.app.state.chunk_repo
    return chunk_repo.get_all_chunks()


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, request: Request):
    get_current_admin_user(request)
    chunk_repo = request.app.state.chunk_repo
    chunks = chunk_repo.get_chunks_by_ids([chunk_id])
    if not chunks:
        raise HTTPException(status_code=404, detail="文档不存在")
    return chunks[0]
