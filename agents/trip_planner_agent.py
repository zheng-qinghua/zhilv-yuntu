import json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from datamodels.schemas import TripRequest


class PlannerDayDraft(BaseModel):
    """LLM 生成的单日草稿，比 DayPlan 更精简"""
    day_index: int = Field(..., ge=1)
    theme: str
    spot_name: str
    spot_description: str
    meal_name: str
    meal_notes: str
    daily_note: str


class PlannerDraft(BaseModel):
    """LLM 生成的完整行程草稿"""
    summary: str
    tips: list[str] = Field(default_factory=list)
    days: list[PlannerDayDraft] = Field(default_factory=list)


def _extract_json_from_text(raw_text: str) -> str | None:
    """
    从 LLM 响应中提取 JSON 字符串。

    LLM 的输出可能嵌套在 ```json ... ``` 代码块中，
    也可能包含正文说明文字。需要从中提取出纯 JSON。
    """
    text = raw_text.strip()
    # 去掉 ``` 代码块包裹
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # 截取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _build_chat_llm():
    """创建 LangChain ChatOpenAI 实例（适配 DeepSeek API）。"""
    if not LLM_API_KEY:
        return None

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def generate_planner_draft(
    request: TripRequest,
    rag_contexts: list[str],
    day_count: int,
) -> tuple[PlannerDraft | None, dict[str, int]]:
    """
    调用 LLM 生成行程草稿。

    参数:
      request: 用户旅行请求
      rag_contexts: RAG 检索到的攻略上下文
      day_count: 行程天数

    返回:
      (PlannerDraft | None, token用量字典)
    """
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    llm = _build_chat_llm()
    if llm is None:
        print("[agent] 无法创建 LLM 实例，检查 LLM_API_KEY")
        return None, empty_usage

    guide_context = "\n\n".join(rag_contexts) if rag_contexts else "（无本地攻略上下文）"

    system_prompt = (
        "你是一个旅行规划大师。"
        "请根据以下信息生成简洁的结构化行程草稿。"
        "你需要严格根据用户提供的目的地、预算、节奏和本地攻略来规划。"
        "请只输出一个 JSON 对象，不要包 Markdown、解释文字或数组。"
        "如果用户有特别备注（如想看日落、不想太赶），请在具体的某一天安排里体现。"
        "每天的安排要符合给定的节奏，不要太赶或太松。"
        "景点和美食推荐请尽量使用攻略中出现过的实际名称。"
    )

    human_prompt = f"""
目的地：{request.destination}
出发日期：{request.start_date.isoformat()}
结束日期：{request.end_date.isoformat()}
天数：{day_count}
人数：{request.travelers}
预算：{request.budget}
偏好：{'、'.join(request.preferences) if request.preferences else '无特别偏好'}
节奏：{request.pace or '适中'}
特别备注：{request.special_notes or '无'}

本地攻略上下文：
{guide_context}

要求：
1. 设计一个全局 summary，用1-2句话概括
2. 输出 {day_count} 天的 daily draft
3. 每天只有一个主要景点、一个推荐美食和一个当天备注
4. tips 保持简洁（2-3 条）
5. day_index 从 1 到 {day_count}
6. 特别备注中的要求请在 days 里体现，不要只写在 tips 里
7. 只输出 JSON 对象，不要使用 ```json 包裹

JSON 结构示例：
{{
  "summary": "行程概述",
  "tips": ["提示1", "提示2"],
  "days": [
    {{
      "day_index": 1,
      "theme": "当天主题",
      "spot_name": "主要景点",
      "spot_description": "景点推荐理由",
      "meal_name": "推荐美食",
      "meal_notes": "美食说明",
      "daily_note": "当天备注"
    }}
  ]
}}
"""

    print(f"[agent] 正在调用 {LLM_MODEL} 生成行程...")

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
    except Exception as e:
        print(f"[agent] LLM 调用失败: {e}")
        return None, empty_usage

    metadata = getattr(response, "response_metadata", None) or {}
    token_info = metadata.get("token_usage", {})
    token_usage = {
        "prompt_tokens": token_info.get("prompt_tokens", 0),
        "completion_tokens": token_info.get("completion_tokens", 0),
    }

    raw_text = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw_text, list):
        raw_text = "".join(str(x) for x in raw_text)

    json_text = _extract_json_from_text(str(raw_text))
    if json_text is None:
        print(f"[agent] 无法从 LLM 响应中提取 JSON")
        print(f"[agent] 原始响应预览: {str(raw_text)[:300]}")
        return None, token_usage

    try:
        result = PlannerDraft.model_validate(json.loads(json_text))
    except Exception:
        # 尝试修复：移除或转义控制字符后重试
        import re
        cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', ' ', json_text)
        try:
            result = PlannerDraft.model_validate(json.loads(cleaned))
        except Exception as exc:
            print(f"[agent] JSON 数据校验失败: {exc}")
            print(f"[agent] JSON 原文: {json_text[:300]}")
            return None, token_usage

    if len(result.days) != day_count:
        print(f"[agent] 天数不匹配: 期望 {day_count} 天，LLM 返回 {len(result.days)} 天")
        return None, token_usage

    print(f"[agent] 行程生成成功！token: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}")
    return result, token_usage
