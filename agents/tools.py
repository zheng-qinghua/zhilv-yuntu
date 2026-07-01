"""Agent tools — callable functions the ReAct Agent can invoke."""
import base64
from pathlib import Path
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES
from config import VISION_API_KEY, VISION_MODEL, VISION_BASE_URL, VISION_TIMEOUT_SECONDS


class ToolResult:
    def __init__(self, content: str, sources: list[str] | None = None):
        self.content = content
        self.sources = sources or []


# Global registry populated at server start
_tool_registry: dict = {}


def register_tool(name: str, fn):
    _tool_registry[name] = fn


def get_tool(name: str):
    return _tool_registry.get(name)


def list_tools() -> dict:
    return _tool_registry


# ============================================================
# Tool implementations
# ============================================================

class RAGSearchTool:
    """RAG 检索工具：搜索本地攻略库中的旅行信息。"""
    name = "rag_search"
    description = (
        "搜索本地旅行攻略数据库。当你需要了解某个目的地、景点、美食、住宿等旅行相关信息时使用。"
        "参数: query (搜索关键词), destination (目的地城市名，可选)"
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, query: str, destination: str = "") -> ToolResult:
        from agents.rag_tool import get_destination_guide_context
        try:
            contexts = get_destination_guide_context(
                destination=destination or "",
                preferences=None, pace=None, special_notes=None,
                top_k=5, retriever=self.retriever,
            )
        except Exception:
            contexts = []

        if not contexts:
            return ToolResult("未在本地攻略库中找到相关信息。", [])

        text = "\n\n---\n\n".join(contexts)
        return ToolResult(text, contexts[:5])


class WebSearchTool:
    """网络搜索工具：当本地攻略不足时搜索互联网信息。"""
    name = "web_search"
    description = (
        "搜索互联网获取最新旅行信息。当本地攻略库没有足够信息，或需要实时信息时使用。"
        "参数: query (搜索关键词，如 '故宫 门票价格 开放时间')"
    )

    def __call__(self, query: str) -> ToolResult:
        from tools.web_search import _search_bing, _search_sogou
        results = _search_bing(query, max_results=6)
        if not results:
            results = _search_sogou(query, max_results=6)

        if not results:
            return ToolResult("网络搜索未返回有效结果。", [])

        contexts = [
            f"[来源: 网页搜索 | 标题: {r['title']}]\n{r['body']}\n链接: {r['href']}"
            for r in results
        ]
        text = "\n\n---\n\n".join(contexts)
        return ToolResult(text, contexts[:5])


class VisionAnalyzeTool:
    """多模态视觉识别工具——用 qwen3-omni-flash 子Agent实际查看图片内容。"""
    name = "vision_analyze"
    description = (
        "当用户上传了一张图片时，用多模态视觉模型识别图片中的景点、建筑、场景等内容。"
        "这是识别图片的主要方式，可以准确判断图片中是什么景点或地标。"
        "参数: file_path (图片在服务器上的完整路径), query_hint (用户附带的文字描述，可选)"
    )

    def __call__(self, file_path: str, query_hint: str = "") -> ToolResult:
        import httpx
        import json as _json

        image_path = Path(file_path)
        if not image_path.exists():
            alt = Path(__file__).resolve().parent.parent / "photo" / image_path.name
            if alt.exists():
                image_path = alt
            else:
                return ToolResult(f"图片文件不存在: {file_path}", [])

        # Read and encode image
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            # Detect mime type
            ext = image_path.suffix.lower()
            mime = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp",
            }.get(ext, "image/jpeg")
        except Exception as e:
            return ToolResult(f"读取图片文件失败: {e}", [])

        # Build prompt for the vision model
        prompt = (
            "请仔细观察这张图片，识别图片中的主要内容。重点判断：\n"
            "1. 这是哪个景点/地标/建筑？请给出具体名称。\n"
            "2. 这个景点位于哪个城市？\n"
            "3. 图片中有什么标志性特征（石刻、建筑风格、自然景观等）？\n"
            "4. 简要介绍这个景点的背景（50字以内）。\n\n"
            "如果图片不是景点而是一般风景或物体，请如实描述你看到的内容。\n"
            "请用中文回答，简洁直接。"
        )
        if query_hint and query_hint.strip():
            prompt = f"用户补充说明：{query_hint.strip()}\n\n{prompt}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ]

        # Call vision model
        try:
            with httpx.Client(timeout=VISION_TIMEOUT_SECONDS) as client:
                resp = client.post(
                    f"{VISION_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {VISION_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": VISION_MODEL,
                        "messages": messages,
                        "max_tokens": 800,
                        "temperature": 0.3,
                    },
                )

            if resp.status_code != 200:
                print(f"[vision_analyze] API error {resp.status_code}: {resp.text[:300]}")
                return ToolResult(
                    f"视觉模型调用失败 (HTTP {resp.status_code})。请引导用户描述图片内容。", []
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"[vision_analyze] 识别结果: {content[:200]}")

            return ToolResult(
                f"[来源: 视觉识别(qwen3-omni-flash)]\n{content}", [content]
            )

        except httpx.TimeoutException:
            return ToolResult("视觉模型调用超时，请重试或引导用户描述图片内容。", [])
        except Exception as e:
            print(f"[vision_analyze] 异常: {e}")
            return ToolResult(
                f"视觉模型调用失败: {type(e).__name__}。请引导用户描述图片内容。", []
            )
