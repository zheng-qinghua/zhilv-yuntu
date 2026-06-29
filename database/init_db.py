from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy import func
from database.connection import Base, engine


class GuideChunk(Base):
    """攻略文档块表"""
    __tablename__ = "guide_chunks"

    id = Column(String(255), primary_key=True)
    title = Column(String(500), nullable=False)
    text = Column(Text, nullable=False)
    source = Column(String(255))
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ChatSession(Base):
    """聊天会话表"""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New Chat")
    is_favorite = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ChatMessage(Base):
    """聊天消息表"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    rag_sources = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, server_default=func.now())


def init_database():
    """创建所有表（如果已存在则跳过）。"""
    Base.metadata.create_all(bind=engine)
    print("[database] 数据库建表完成")


if __name__ == "__main__":
    init_database()
