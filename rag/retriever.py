import logging

from database.chunk_repo import ChunkRepository
from rag.embedding import EmbeddingService
from rag.vector_index import VectorIndex

logger = logging.getLogger(__name__)

def _rule_based_score(
    query: str,
    chunk: dict[str, str],
    destination: str | None = None
) -> int:
  title = chunk.get("title", "")
  text_content = chunk.get("text", "")
  source = chunk.get("source", "")

  keywords = query.strip().split()

  score = 0
  for kw in keywords:
    if kw in title:
      score += 3
    if kw in text_content:
      score += 1

  if title == "文档开头":
    score -= 8

  if "行程" in title and "行程参考" not in title:
    score += 4

  if "行程参考" in title:
    score -= 4

  if "目的地简介" in title:
    score -= 2

  if destination:
    chunk_full_text = f"{source} {title} {text_content}".lower()
    if destination.lower() not in chunk_full_text:
      score -= 5

  return score

class Retriever:
  def __init__(
      self,
      vector_index: VectorIndex,
      embedding_service: EmbeddingService,
      chunk_repo: ChunkRepository
  ):
    self.vector_index = vector_index
    self.embedding_service = embedding_service
    self.chunk_repo = chunk_repo

  def retrieve(
      self,
      query: str,
      top_k: int = 5,
      destination: str | None = None
  ) -> list[str]:
    query_vec = self.embedding_service.embed_single(query)
    if query_vec is None:
      print("[retriever] query embedding 失败")
      return []

    candidate_k = max(top_k * 2, 6)
    search_results = self.vector_index.search(query_vec, candidate_k)

    if not search_results:
      print("[retriever] FAISS 搜索无结果")
      return []

    chunk_ids = [cid for cid, _ in search_results]
    chunks = self.chunk_repo.get_chunks_by_ids(chunk_ids)

    chunk_map = {c["id"]: c for c in chunks}

    matched_chunks: list[dict[str, str]] = []
    for cid, faiss_score in search_results:
      if cid in chunk_map:
        enriched = dict(chunk_map[cid])
        enriched["faiss_score"] = round(faiss_score, 4)
        matched_chunks.append(enriched)

    print("[retriever] 使用规则进行精排")
    scored = []
    for idx, chunk in enumerate(matched_chunks):
      s = _rule_based_score(query, chunk, destination=destination)
      scored.append((s, -idx, chunk))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = [chunk for _, _, chunk in scored[:top_k]]

    results: list[str] = []
    for chunk in selected:
      formatted = (
        f"[来源: {chunk['source']} | 标题: {chunk['title']}]\n"
        f"{chunk['text']}"
      )
      results.append(formatted)

    print(f"[retriever] 检索完成，返回 {len(results)} 条")
    return results
