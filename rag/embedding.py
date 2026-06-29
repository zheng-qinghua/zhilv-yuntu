import httpx

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)


class EmbeddingService:
    """
    文本转向量服务 (Strategy 模式)。

    当前使用 SiliconFlow BAAI/bge-m3 模型，
    可替换为其他兼容 OpenAI API 格式的 embedding 服务。
    """

    def __init__(self):
        self.api_key = EMBEDDING_API_KEY
        self.model = EMBEDDING_MODEL
        self.base_url = (EMBEDDING_BASE_URL or "").rstrip("/")
        self.batch_size = EMBEDDING_BATCH_SIZE

    def embed_single(self, text: str) -> list[float] | None:
        """将单条文本转向量。"""
        result = self.embed_batch([text])
        if result and len(result) > 0:
            return result[0]
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """
        批量转向量。自动按 batch_size 拆分，避免超过 API 单次上限。
        """
        if not self.api_key:
            print("[embedding] 未配置 EMBEDDING_API_KEY")
            return None

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_vectors: list[list[float]] = []

        try:
            with httpx.Client(timeout=120) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    payload = {"model": self.model, "input": batch}

                    resp = client.post(url, json=payload, headers=headers)

                    if resp.status_code != 200:
                        print(f"[embedding] API 返回错误 {resp.status_code}: {resp.text[:200]}")
                        return None

                    data = resp.json()
                    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                    vectors = [item["embedding"] for item in items if "embedding" in item]

                    if len(vectors) != len(batch):
                        print(f"[embedding] 返回数量不匹配: 期望 {len(batch)}, 实际 {len(vectors)}")
                        return None

                    all_vectors.extend(vectors)
                    print(f"[embedding] 批次 {i // self.batch_size + 1}: 嵌入 {len(vectors)} 条")

            print(f"[embedding] 全部完成: 共 {len(all_vectors)} 条文本")
            return all_vectors
        except Exception as exc:
            print(f"[embedding] 调用失败: {type(exc).__name__}: {exc}")
            return None
