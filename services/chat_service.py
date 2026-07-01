"""Chat orchestration service — now powered by ReAct Agent."""
import os
import uuid
from pathlib import Path
from database.chat_repo import ChatRepo
from rag.retriever import Retriever
from services.trip_service import TripService

PHOTO_DIR = Path(__file__).resolve().parent.parent / "photo"


class ChatService:
    def __init__(self, trip_service: TripService, retriever: Retriever, chat_repo: ChatRepo):
        self.trip_service = trip_service
        self.retriever = retriever
        self.chat_repo = chat_repo

    def handle_message(self, user_id: int, session_id: int | None, content: str,
                       image_filename: str | None = None) -> dict:
        # Create or reuse session
        if session_id is None:
            title = content[:30] if content else (image_filename or "New Chat")
            title = title + ("..." if len(title) > 30 else "")
            sess = self.chat_repo.create_session(user_id, title=title)
            session_id = sess["id"]

        # Save user message — include image markdown so frontend can render it
        if image_filename:
            display_content = f"![](/photo/{image_filename})\n\n{content or ''}"
        else:
            display_content = content
        self.chat_repo.add_message(session_id, "user", display_content)

        # Build image file path for reverse image search
        image_file_path = None
        if image_filename:
            image_file_path = str(PHOTO_DIR / image_filename)

        # Run ReAct Agent
        from agents.react_agent import ReActAgent
        agent = ReActAgent()
        result = agent.run(
            user_message=content or "请识别这张图片的内容。",
            image_filename=image_filename,
            image_file_path=image_file_path,
        )

        answer = result["answer"]
        sources = result["sources"]

        # Add ReAct info to answer
        if result.get("rounds", 1) > 1:
            answer += f"\n\n> *ReAct Agent 经过 {result['rounds']} 轮推理完成回答*"

        # Save assistant message
        msg = self.chat_repo.add_message(session_id, "assistant", answer, rag_sources=sources)

        # Update session title if needed
        if content:
            title = content[:30] + ("..." if len(content) > 30 else "")
            self.chat_repo.update_session(session_id, title=title)
        elif image_filename:
            self.chat_repo.update_session(session_id, title=f"图片: {image_filename}")

        return {
            "session_id": session_id,
            "message_id": msg["id"],
            "role": "assistant",
            "content": answer,
            "rag_sources": sources,
        }

    def handle_image_upload(self, user_id: int, session_id: int | None,
                            filename: str, content: str) -> dict:
        """Handle image upload: save file, then use ReAct agent."""
        return self.handle_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
            image_filename=filename,
        )

    # ---- session management delegates ----

    def list_sessions(self, user_id: int, favorite_only: bool = False) -> list[dict]:
        sessions = self.chat_repo.list_sessions_by_user(user_id, favorite_only=favorite_only)
        for s in sessions:
            for key in ("updated_at", "created_at"):
                if s.get(key):
                    s[key] = str(s[key])
        return sessions

    def get_session_detail(self, session_id: int, user_id: int) -> dict | None:
        sess = self.chat_repo.get_session(session_id)
        if sess is None or sess["user_id"] != user_id:
            return None
        messages = self.chat_repo.get_messages(session_id)
        for m in messages:
            if m.get("created_at"):
                m["created_at"] = str(m["created_at"])
        sess["messages"] = messages
        for key in ("updated_at", "created_at"):
            if sess.get(key):
                sess[key] = str(sess[key])
        return sess

    def update_session(self, session_id: int, title: str | None = None, is_favorite: bool | None = None):
        self.chat_repo.update_session(session_id, title=title, is_favorite=is_favorite)

    def delete_session(self, session_id: int):
        self.chat_repo.delete_session(session_id)
