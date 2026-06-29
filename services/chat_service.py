"""Chat orchestration service: intent detection, Q&A, trip planning."""
from datetime import timedelta, datetime
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES
from agents.rag_tool import get_destination_guide_context
from database.chat_repo import ChatRepo
from rag.retriever import Retriever
from services.trip_service import TripService

# 行程规划关键词
_PLANNING_KEYWORDS = [
    "规划", "行程", "安排", "计划", "攻略", "几天",
    "旅游", "旅行", "预算", "出发", "几日",
    "帮我", "设计", "定制", "路线",
]


def _detect_intent(content: str) -> str:
    """规则级意图检测：planning 或 qa。"""
    has_planning_kw = any(kw in content for kw in _PLANNING_KEYWORDS)
    # 检查是否包含目的地名称（简单的城市名检测）
    cities = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉",
        "西安", "重庆", "长沙", "大理", "丽江", "三亚", "厦门", "昆明",
        "贵阳", "桂林", "拉萨", "哈尔滨", "青岛", "苏州", "黄山", "张家界",
        "郑州", "济南", "天津", "沈阳", "大连", "海口", "乌鲁木齐", "呼和浩特",
        "南宁", "南昌", "福州", "合肥", "太原", "兰州", "西宁", "银川",
    ]
    has_city = any(city in content for city in cities)
    if has_planning_kw and has_city:
        return "planning"
    return "qa"


def _extract_destination(content: str) -> str | None:
    """简单提取内容中提到的目的地城市。"""
    cities = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉",
        "西安", "重庆", "长沙", "大理", "丽江", "三亚", "厦门", "昆明",
        "贵阳", "桂林", "拉萨", "哈尔滨", "青岛", "苏州", "黄山", "张家界",
        "郑州", "济南", "天津", "沈阳", "大连", "海口", "乌鲁木齐", "呼和浩特",
        "南宁", "南昌", "福州", "合肥", "太原", "兰州", "西宁", "银川",
    ]
    for city in cities:
        if city in content:
            return city
    return None


class ChatService:
    def __init__(self, trip_service: TripService, retriever: Retriever, chat_repo: ChatRepo):
        self.trip_service = trip_service
        self.retriever = retriever
        self.chat_repo = chat_repo

    def handle_message(self, user_id: int, session_id: int | None, content: str) -> dict:
        # 创建或使用已有 session
        if session_id is None:
            title = content[:30] + ("..." if len(content) > 30 else "")
            sess = self.chat_repo.create_session(user_id, title=title)
            session_id = sess["id"]

        # 保存用户消息
        self.chat_repo.add_message(session_id, "user", content)

        # 意图检测 + 路由
        intent = _detect_intent(content)

        if intent == "planning":
            answer, sources = self._handle_planning(content, user_id, session_id)
        else:
            answer, sources = self._handle_qa(content)

        # 保存助手消息
        msg = self.chat_repo.add_message(session_id, "assistant", answer, rag_sources=sources)

        # 如果是新会话且是规划意图，更新标题
        if intent == "planning":
            dest = _extract_destination(content) or ""
            self.chat_repo.update_session(session_id, title=f"{dest}行程规划" if dest else content[:30])

        return {
            "session_id": session_id,
            "message_id": msg["id"],
            "role": "assistant",
            "content": answer,
            "rag_sources": sources,
        }

    def _handle_qa(self, content: str) -> tuple[str, list[str]]:
        """处理一般问答：RAG检索 → LLM生成回答。"""
        destination = _extract_destination(content) or ""

        # RAG检索
        rag_contexts = get_destination_guide_context(
            destination=destination or "旅行",
            preferences=None,
            pace=None,
            special_notes=None,
            top_k=5,
            retriever=self.retriever,
        )

        # 构建 context 文本
        context_text = "\n\n---\n\n".join(rag_contexts) if rag_contexts else "暂无相关攻略资料"

        # 调用LLM
        system_prompt = (
            "你是'智旅'，一个专业的旅行助手。请根据提供的参考攻略资料回答用户的问题。\n\n"
            "规则：\n"
            "1. 如果参考资料中有相关信息，请详细引用并回答\n"
            "2. 如果参考资料不充分，可以结合你的知识进行补充，但需说明哪些内容来自参考、哪些是常识\n"
            "3. 回答结构清晰，使用适当的段落和列表\n"
            "4. 如果用户询问具体景点，给出实用的游览建议（时间、交通、门票等）\n"
            "5. 用中文回答，语气友好专业\n\n"
            f"参考攻略资料：\n{context_text}"
        )

        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=0.7,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL or None,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
            response = llm.invoke([
                ("system", system_prompt),
                ("human", content),
            ])
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"抱歉，AI服务暂时不可用（{e}）。请稍后重试。"

        return answer, rag_contexts[:5] if rag_contexts else []

    def _handle_planning(self, content: str, user_id: int, session_id: int) -> tuple[str, list[str]]:
        """处理旅行规划：解析需求 → 生成行程 → 格式化输出。"""
        from services.nl_parser import NLToTripParser

        # 尝试用LLM解析旅行需求
        try:
            parser = NLToTripParser()
            parsed = parser.parse(content)

            from datetime import datetime, date
            # 处理未指定日期的情况：默认今天起3天
            if parsed.start_date:
                start_date = datetime.strptime(parsed.start_date, "%Y-%m-%d").date()
            else:
                start_date = date.today()
            if parsed.end_date:
                end_date = datetime.strptime(parsed.end_date, "%Y-%m-%d").date()
            else:
                end_date = start_date + timedelta(days=2)

            from datamodels.schemas import TripRequest
            req = TripRequest(
                destination=parsed.destination,
                start_date=start_date,
                end_date=end_date,
                travelers=parsed.num_travelers or 1,
                budget=float(parsed.budget),
                preferences=parsed.preferences if isinstance(parsed.preferences, list) else [],
                pace=parsed.pace or "适中",
                special_notes=parsed.special_notes or None,
            )

            itinerary = self.trip_service.generate_itinerary(req, origin_city=None)

            # 格式化为可读文本
            lines = [
                f"## {itinerary.destination} {len(itinerary.days)} 日旅行规划",
                "",
                f"**概述**: {itinerary.summary}",
                "",
            ]

            for day in itinerary.days:
                lines.append(f"### 第{day.day_index}天 — {day.date}")
                lines.append(f"**主题**: {day.theme}")
                if day.spots:
                    s = day.spots[0]
                    lines.append(f"- **景点**: {s.name}")
                    lines.append(f"  {s.description}")
                    lines.append(f"- **门票**: 约{s.estimated_cost:.0f}元")
                if day.meals:
                    m = day.meals[0]
                    lines.append(f"- **美食**: {m.name} — {m.notes}")
                if day.hotel:
                    lines.append(f"- **住宿**: {day.hotel.name}（约{day.hotel.estimated_cost:.0f}元）")
                lines.append("")

            lines.append("---")
            lines.append("## 预算明细")
            bb = itinerary.budget_breakdown
            lines.append(f"- 交通: {bb.transport}元")
            lines.append(f"- 住宿: {bb.hotel}元")
            lines.append(f"- 餐饮: {bb.meals}元")
            lines.append(f"- 门票: {bb.tickets}元")
            lines.append(f"- 其他: {bb.other}元")
            lines.append(f"- **合计: {bb.total}元**")
            lines.append("")

            if itinerary.tips:
                lines.append("## 旅行提示")
                for tip in itinerary.tips:
                    lines.append(f"- {tip}")

            answer = "\n".join(lines)
            sources = itinerary.source_notes if itinerary.source_notes else []

        except Exception as e:
            # 解析或生成失败时，退回QA模式
            fallback_answer, sources = self._handle_qa(content)
            answer = f"*未能生成完整行程规划（{e}），以下为相关旅行建议：*\n\n{fallback_answer}"

        return answer, sources

    # ---- session management delegates ----

    def list_sessions(self, user_id: int, favorite_only: bool = False) -> list[dict]:
        sessions = self.chat_repo.list_sessions_by_user(user_id, favorite_only=favorite_only)
        for s in sessions:
            if s.get("updated_at"):
                s["updated_at"] = str(s["updated_at"])
            if s.get("created_at"):
                s["created_at"] = str(s["created_at"])
        return sessions

    def get_session_detail(self, session_id: int, user_id: int) -> dict | None:
        sess = self.chat_repo.get_session(session_id)
        if sess is None or sess["user_id"] != user_id:
            return None
        messages = self.chat_repo.get_messages(session_id)
        for m in messages:
            if m.get("created_at"):
                m["created_at"] = str(m["created_at"])
        sess["messages"] = messages
        if sess.get("updated_at"):
            sess["updated_at"] = str(sess["updated_at"])
        if sess.get("created_at"):
            sess["created_at"] = str(sess["created_at"])
        return sess

    def update_session(self, session_id: int, title: str | None = None, is_favorite: bool | None = None):
        self.chat_repo.update_session(session_id, title=title, is_favorite=is_favorite)

    def delete_session(self, session_id: int):
        self.chat_repo.delete_session(session_id)
