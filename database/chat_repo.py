"""Chat session / message data access."""
import json
from sqlalchemy import text
from database.connection import SessionLocal


class ChatRepo:
    # ---- sessions ----

    def create_session(self, user_id: int, title: str = "New Chat") -> dict:
        with SessionLocal() as session:
            result = session.execute(
                text(
                    "INSERT INTO chat_sessions (user_id, title, is_favorite) VALUES (:user_id, :title, 0)"
                ),
                {"user_id": user_id, "title": title},
            )
            session.commit()
            sid = result.lastrowid
        return {"id": sid, "user_id": user_id, "title": title, "is_favorite": False}

    def list_sessions_by_user(self, user_id: int, favorite_only: bool = False) -> list[dict]:
        with SessionLocal() as session:
            where = "WHERE user_id = :user_id"
            if favorite_only:
                where += " AND is_favorite = 1"
            rows = session.execute(
                text(
                    f"SELECT id, user_id, title, is_favorite, created_at, updated_at "
                    f"FROM chat_sessions {where} ORDER BY updated_at DESC"
                ),
                {"user_id": user_id},
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r._mapping)
            d["is_favorite"] = bool(d["is_favorite"])
            # last message preview
            preview = self._last_message_preview(d["id"])
            d["last_message_preview"] = preview or ""
            results.append(d)
        return results

    def get_session(self, session_id: int) -> dict | None:
        with SessionLocal() as session:
            row = session.execute(
                text(
                    "SELECT id, user_id, title, is_favorite, created_at, updated_at "
                    "FROM chat_sessions WHERE id = :id"
                ),
                {"id": session_id},
            ).fetchone()
        if row is None:
            return None
        d = dict(row._mapping)
        d["is_favorite"] = bool(d["is_favorite"])
        return d

    def update_session(self, session_id: int, title: str | None = None, is_favorite: bool | None = None):
        parts = []
        params = {"id": session_id}
        if title is not None:
            parts.append("title = :title")
            params["title"] = title
        if is_favorite is not None:
            parts.append("is_favorite = :is_favorite")
            params["is_favorite"] = int(is_favorite)
        if not parts:
            return
        parts.append("updated_at = NOW()")
        with SessionLocal() as session:
            session.execute(
                text(f"UPDATE chat_sessions SET {', '.join(parts)} WHERE id = :id"),
                params,
            )
            session.commit()

    def delete_session(self, session_id: int):
        with SessionLocal() as session:
            session.execute(text("DELETE FROM chat_messages WHERE session_id = :sid"), {"sid": session_id})
            session.execute(text("DELETE FROM chat_sessions WHERE id = :sid"), {"sid": session_id})
            session.commit()

    # ---- messages ----

    def add_message(self, session_id: int, role: str, content: str, rag_sources: list[str] | None = None) -> dict:
        sources_json = json.dumps(rag_sources, ensure_ascii=False) if rag_sources else None
        with SessionLocal() as session:
            result = session.execute(
                text(
                    "INSERT INTO chat_messages (session_id, role, content, rag_sources) "
                    "VALUES (:session_id, :role, :content, :rag_sources)"
                ),
                {"session_id": session_id, "role": role, "content": content, "rag_sources": sources_json},
            )
            # bump session updated_at
            session.execute(
                text("UPDATE chat_sessions SET updated_at = NOW() WHERE id = :sid"),
                {"sid": session_id},
            )
            session.commit()
            mid = result.lastrowid
        return {"id": mid, "session_id": session_id, "role": role, "content": content}

    def get_messages(self, session_id: int) -> list[dict]:
        with SessionLocal() as session:
            rows = session.execute(
                text(
                    "SELECT id, session_id, role, content, rag_sources, created_at "
                    "FROM chat_messages WHERE session_id = :sid ORDER BY created_at ASC"
                ),
                {"sid": session_id},
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r._mapping)
            if d.get("rag_sources"):
                try:
                    d["rag_sources"] = json.loads(d["rag_sources"])
                except (json.JSONDecodeError, TypeError):
                    d["rag_sources"] = []
            results.append(d)
        return results

    def _last_message_preview(self, session_id: int) -> str | None:
        with SessionLocal() as session:
            row = session.execute(
                text(
                    "SELECT content FROM chat_messages WHERE session_id = :sid "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"sid": session_id},
            ).fetchone()
        if row:
            content = row[0]
            return content[:80] + ("..." if len(content) > 80 else "")
        return None

    # ---- admin ----

    def list_all_chunks(self) -> list[dict]:
        with SessionLocal() as session:
            rows = session.execute(
                text("SELECT id, title, text, source FROM guide_chunks ORDER BY source, id")
            ).fetchall()
        return [dict(r._mapping) for r in rows]
