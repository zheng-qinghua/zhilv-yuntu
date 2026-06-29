from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from config import DATABASE_URL

#  创建连接池
engine = create_engine(
  DATABASE_URL,
  pool_size=5,
  max_overflow=10,
  pool_pre_ping=True,
  echo=False,
)

# ---- 创建 Session 工厂 ----
# SessionLocal 是一个可调用对象，每次调用创建一个新的数据库会话
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)

Base = declarative_base()