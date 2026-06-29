"""智旅云图 — FastAPI Web Server"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from database.init_db import init_database
from database.chunk_repo import ChunkRepository
from database.user_repo import UserRepo
from database.chat_repo import ChatRepo
from rag.embedding import EmbeddingService
from rag.vector_index import VectorIndex
from rag.retriever import Retriever
from services.trip_service import TripService
from auth.service import AuthService

PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"


def _ensure_admin_user():
    """确保至少有一个管理员账号存在。"""
    user_repo = UserRepo()
    auth = AuthService(user_repo)
    existing = user_repo.get_by_username("admin")
    if not existing:
        auth.register("admin", "admin123", is_admin=True)
        print("[app] 已创建默认管理员: admin / admin123")
    else:
        print("[app] 管理员账号已存在")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    print("[app] 正在初始化数据库...")
    init_database()
    _ensure_admin_user()

    chunk_repo = ChunkRepository()
    print(f"[app] guide_chunks 表中有 {chunk_repo.count_chunks()} 条记录")

    vec_index = VectorIndex()
    if vec_index.load():
        print("[app] FAISS 索引加载成功")
    else:
        print("[app] 警告: FAISS 索引不存在，请先运行 python main.py --ingest")

    embed_service = EmbeddingService()
    retriever = Retriever(vec_index, embed_service, chunk_repo)
    trip_service = TripService(retriever, chunk_repo)

    app.state.chunk_repo = chunk_repo
    app.state.vec_index = vec_index
    app.state.embed_service = embed_service
    app.state.retriever = retriever
    app.state.trip_service = trip_service
    app.state.chat_repo = ChatRepo()
    app.state.user_repo = UserRepo()
    app.state.auth_service = AuthService(app.state.user_repo)

    print("[app] 服务初始化完成，启动服务器...")
    yield
    # ---- shutdown ----
    print("[app] 服务器关闭")


app = FastAPI(title="智旅云图", lifespan=lifespan)

# API 路由
from api.auth import router as auth_router
from api.chat import router as chat_router
from api.admin import router as admin_router

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)

# 静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
