import json

from typing import Optional
from sqlalchemy import text
from database.connection import SessionLocal


class ChunkRepository:
    """
    文本块数据访问对象 (DAO 模式)。

    封装所有对 guide_chunks 表的 SQL 操作，
    上层代码不直接写 SQL，通过此类访问数据。
    """

    def insert_chunk(
        self,
        chunk_id: str,
        title: str,
        text_content: str,
        source: str,
        embedding: list[float] | None = None,
    ) -> None:
        """
        插入或更新一个文本块。

        ON DUPLICATE KEY UPDATE: 如果 id 已存在则更新，
        保证重复 ingest 不会产生重复数据（幂等）。
        """
        embedding_json = json.dumps(embedding) if embedding else None
        with SessionLocal() as session:
            session.execute(
                text("""
                    INSERT INTO guide_chunks (id, title, text, source, embedding)
                    VALUES (:id, :title, :text, :source, :embedding)
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        text = VALUES(text),
                        embedding = VALUES(embedding)
                """),
                {
                    "id": chunk_id,
                    "title": title,
                    "text": text_content,
                    "source": source,
                    "embedding": embedding_json,
                },
            )
            session.commit()

    def get_all_chunks(self) -> list[dict]:
        """
        查询全部文本块（不含 embedding），用于列表展示。
        """
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT id, title, text, source FROM guide_chunks")
            )
            return [dict(row) for row in result.mappings()]

    def get_all_embeddings(self) -> list[dict]:
        """
        查询全部文本块的 id + embedding，用于重建 FAISS 索引。
        返回的 embedding 从 JSON 字符串反序列化为 list[float]。
        """
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT id, embedding FROM guide_chunks WHERE embedding IS NOT NULL")
            )
            rows = []
            for row in result.mappings():
                row_dict = dict(row)
                if row_dict["embedding"]:
                    row_dict["embedding"] = json.loads(row_dict["embedding"])
                rows.append(row_dict)
            return rows

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        根据 ID 列表批量查询文本块。
        用于 FAISS 检索后从 MySQL 取完整文本。
        """
        if not chunk_ids:
            return []
        with SessionLocal() as session:
            placeholders = ", ".join(f":id_{i}" for i in range(len(chunk_ids)))
            params = {f"id_{i}": cid for i, cid in enumerate(chunk_ids)}
            result = session.execute(
                text(
                    f"SELECT id, title, text, source FROM guide_chunks WHERE id IN ({placeholders})"
                ),
                params,
            )
            return [dict(row) for row in result.mappings()]

    def count_chunks(self) -> int:
        """返回文本块总数。"""
        with SessionLocal() as session:
            result = session.execute(text("SELECT COUNT(*) AS cnt FROM guide_chunks"))
            return result.scalar() or 0

    def delete_all(self) -> None:
        """清空所有文本块（用于重置）。"""
        with SessionLocal() as session:
            session.execute(text("DELETE FROM guide_chunks"))
            session.commit()
