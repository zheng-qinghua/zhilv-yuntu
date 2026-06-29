"""Auth service: register, login, password hashing."""
import bcrypt
from database.user_repo import UserRepo


class AuthService:
    def __init__(self, user_repo: UserRepo | None = None):
        self.user_repo = user_repo or UserRepo()

    def register(self, username: str, password: str, is_admin: bool = False) -> dict:
        existing = self.user_repo.get_by_username(username)
        if existing:
            raise ValueError("用户名已存在")
        if len(password) < 3:
            raise ValueError("密码至少3位")
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return self.user_repo.create_user(username, password_hash, is_admin=is_admin)

    def login(self, username: str, password: str) -> dict:
        user = self.user_repo.get_by_username(username)
        if not user:
            raise ValueError("用户名或密码错误")
        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            raise ValueError("用户名或密码错误")
        return {"id": user["id"], "username": user["username"], "is_admin": bool(user["is_admin"])}
