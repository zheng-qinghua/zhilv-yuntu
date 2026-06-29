"""User data access."""
from sqlalchemy import text
from database.connection import SessionLocal
from auth.models import User


class UserRepo:
    def create_user(self, username: str, password_hash: str, is_admin: bool = False) -> dict:
        with SessionLocal() as session:
            result = session.execute(
                text(
                    "INSERT INTO users (username, password_hash, is_admin) "
                    "VALUES (:username, :password_hash, :is_admin)"
                ),
                {"username": username, "password_hash": password_hash, "is_admin": int(is_admin)},
            )
            session.commit()
            user_id = result.lastrowid
        return {"id": user_id, "username": username, "is_admin": is_admin}

    def get_by_username(self, username: str) -> dict | None:
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT id, username, password_hash, is_admin FROM users WHERE username = :username"),
                {"username": username},
            ).fetchone()
        if row is None:
            return None
        return dict(row._mapping)

    def get_by_id(self, user_id: int) -> dict | None:
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT id, username, is_admin FROM users WHERE id = :id"),
                {"id": user_id},
            ).fetchone()
        if row is None:
            return None
        return dict(row._mapping)

    def list_all(self) -> list[dict]:
        with SessionLocal() as session:
            rows = session.execute(
                text("SELECT id, username, is_admin, created_at FROM users ORDER BY id")
            ).fetchall()
        return [dict(r._mapping) for r in rows]
