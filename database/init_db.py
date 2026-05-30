from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy import func
from database.connection import Base, engine


class GuideChunk(Base):
    """
    攻略文档的数据库表模型。

    字段说明:
      id          主键，由 source + MD5 生成，如 "dali_guide.md_a1b2c3d4"
      title       文本块所属的标题，如 "大理古城"
      text        文本块正文（去掉了标题行）
      source      来源文件名，如 "dali_guide.md"
      embedding   文本转换成的向量数组，存为 JSON 字符串
                  例如 "[0.0123, -0.0456, 0.0789, ...]"
      created_at  创建时间，自动填
    """
    __tablename__ = "guide_chunks"

    id = Column(String(255), primary_key=True)
    title = Column(String(500), nullable=False)
    text = Column(Text, nullable=False)
    source = Column(String(255))
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


def init_database():
    """
    创建所有表（如果已存在则跳过）。
    应用启动时调用一次即可。
    """
    Base.metadata.create_all(bind=engine)
    print("[database] 数据库建表完成")


if __name__ == "__main__":
    init_database()
