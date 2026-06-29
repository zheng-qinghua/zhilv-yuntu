"""Pydantic models for chat API requests/responses."""
from pydantic import BaseModel


class SendMessageRequest(BaseModel):
    session_id: int | None = None
    content: str


class SendMessageResponse(BaseModel):
    session_id: int
    message_id: int
    role: str
    content: str
    rag_sources: list[str] | None = None


class SessionItem(BaseModel):
    id: int
    title: str
    is_favorite: bool
    last_message_preview: str
    updated_at: str


class SessionDetail(BaseModel):
    id: int
    title: str
    is_favorite: bool
    messages: list[dict]


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    is_favorite: bool | None = None
