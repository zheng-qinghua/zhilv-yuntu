import json
import pickle
import numpy as np
import faiss
from config import FAISS_ID_MAP_PATH, FAISS_INDEX_PATH


class VectorIndex:
    """
    FAISS 向量索引封装。

    使用 IndexFlatIP (内积索引)：
      - 先对向量做 L2 归一化
      - 归一化后内积 = 余弦相似度
      - 通过余弦相似度找到语义最相近的文本块
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: dict[int, str] = {}  # FAISS 内部序号 → chunk_id
        self.dimension: int = 0

    def build(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        构建 FAISS 索引。

        步骤:
          1. 向量转 numpy 数组
          2. L2 归一化（让内积 = 余弦相似度）
          3. 创建 IndexFlatIP 并添加向量
          4. 建立序号 → chunk_id 的映射表
        """
        if not embeddings:
            print("[vector_index] 没有向量数据，跳过构建")
            return

        vectors = np.array(embeddings, dtype=np.float32)
        self.dimension = vectors.shape[1]

        # L2 归一化: 每个向量除以它的 L2 范数
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # 避免除零
        vectors = vectors / norms

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.id_map = {i: cid for i, cid in enumerate(chunk_ids)}

        print(f"[vector_index] 索引构建完成: {len(chunk_ids)} 个向量, {self.dimension} 维")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """
        搜索最相似的 top_k 个文本块。

        返回: [(chunk_id, similarity_score), ...]
        分数范围 [-1, 1]，越接近 1 越相似。
        """
        if self.index is None:
            print("[vector_index] 索引未初始化")
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        # 查询向量也要做 L2 归一化
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        query_vec = query_vec / norm

        distances, indices = self.index.search(query_vec, top_k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))

        return results

    def save(self) -> bool:
        """
        持久化索引到磁盘。

        FAISS 索引 → pickle 二进制文件
        id_map   → JSON 文本文件
        """
        if self.index is None:
            return False

        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(FAISS_ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)

        print(f"[vector_index] 索引已保存到 {FAISS_INDEX_PATH}")
        return True

    def load(self) -> bool:
        """从磁盘加载索引。"""
        if not FAISS_INDEX_PATH.exists() or not FAISS_ID_MAP_PATH.exists():
            print("[vector_index] 索引文件不存在，需要先运行 ingest")
            return False

        with open(FAISS_INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(FAISS_ID_MAP_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            self.id_map = {int(k): v for k, v in raw_map.items()}

        self.dimension = self.index.d

        print(f"[vector_index] 索引已加载: {len(self.id_map)} 个向量")
        return True

    @property
    def is_ready(self) -> bool:
        """索引是否已就绪。"""
        return self.index is not None and len(self.id_map) > 0
