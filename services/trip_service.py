# ============================================================
# services/trip_service.py — 行程编排服务
# ============================================================

# timedelta: 日期加减，用于从 start_date 逐天推进
from datetime import timedelta

from agents.rag_tool import get_destination_guide_context
from agents.trip_planner_agent import generate_planner_draft
from database.chunk_repo import ChunkRepository
from datamodels.schemas import (
    BudgetBreakdown,
    DayPlan,
    HotelItem,
    Itinerary,
    MealItem,
    SpotItem,
    TokenUsage,
    TransportItem,
    TripRequest,
)
from rag.embedding import EmbeddingService
from rag.retriever import Retriever
from rag.vector_index import VectorIndex
from tools.transport_cost import estimate_daily_city_transport, estimate_intercity_transport


# ============================================================
# 辅助函数
# ============================================================

def _guess_origin_city() -> str:
    """
    尝试推断用户出发城市。

    策略:
      1. 命令行参数 origin_city（由 main.py 传入）
      2. .env 中的 TRAVEL_ORIGIN_CITY
      3. 默认 "北京"
    """
    from config import TRAVEL_ORIGIN_CITY
    return TRAVEL_ORIGIN_CITY or "北京"


def _estimate_ticket_cost(spot_name: str, description: str | None = None) -> float:
    """
    根据景点名称中的关键词粗略估算门票价格。

    为什么不用真实 API 查门票：
      这是 MVP，先用规则粗略估算。后续可以接地图 API 拿真实数据。
    """
    text = f"{spot_name} {description or ''}"

    # any(kw in text): 只要 text 中包含任意一个关键词就命中
    if any(kw in text for kw in ("古城", "古镇", "公园", "廊道", "村", "湿地", "街区")):
        return 0.0         # 免费景点
    if any(kw in text for kw in ("寺", "塔", "博物馆", "遗址")):
        return 75.0        # 文化类
    if any(kw in text for kw in ("索道", "缆车", "游船", "演出", "水上", "乐园")):
        return 150.0       # 体验类
    return 50.0            # 默认


def _build_fallback_spots(destination: str, rag_contexts: list[str], count: int) -> list[str]:
    """
    LLM 故障时使用的硬编码回退景点表。

    设计考虑：
      LLM API 可能超时/限流/返回格式错误——此时用这个表兜底，
      保证用户在任何情况下都能拿到一份（虽然不如 AI 生成的那么个性化）的计划。
    """
    city_spot_map = {
        "大理": ["大理古城", "洱海生态廊道", "喜洲古镇", "双廊古镇", "崇圣寺三塔"],
        "成都": ["大熊猫繁育基地", "宽窄巷子", "锦里", "都江堰", "青城山"],
        "三亚": ["亚龙湾", "蜈支洲岛", "天涯海角", "南山文化旅游区", "亚特兰蒂斯水世界"],
        "厦门": ["鼓浪屿", "环岛路", "曾厝垵", "南普陀寺", "沙坡尾"],
        "西安": ["兵马俑", "大雁塔", "回民街", "城墙", "大唐不夜城"],
    }

    candidates = city_spot_map.get(
        destination,
        [f"{destination}推荐景点 {i+1}" for i in range(count)]
    )

    # 确保有 count 个：不够就补 "第N天" 占位
    while len(candidates) < count:
        candidates.append(f"{destination} 第{len(candidates) + 1}天")
    return candidates[:count]


# ============================================================
# 编排主函数
# ============================================================

class TripService:
    """
    行程编排服务。

    设计模式: Service Layer（服务层）
    - 不直接操作数据库
    - 不直接调用 LLM
    - 只做流程编排：RAG → Agent → 组装 → 返回

    依赖注入：
      所有底层组件通过构造函数传入，方便测试和替换。
    """

    def __init__(
        self,
        retriever: Retriever,
        chunk_repo: ChunkRepository,
    ):
        self.retriever = retriever
        self.chunk_repo = chunk_repo

    def generate_itinerary(self, request: TripRequest, origin_city: str | None = None) -> Itinerary:
        """
        根据用户请求生成完整行程。

        完整流程:
          Step 1: 计算天数
          Step 2: RAG 检索攻略上下文（本地不足自动联网搜索）
          Step 3: Agent 调用 LLM 生成草稿
          Step 4: 计算交通费用（城市内+跨城）
          Step 5: 组装 DayPlan（LLM 草稿 + 规则兜底）
          Step 6: 预算汇总
          Step 7: 生成 summary & tips（含交通提示）
          Step 8: 组装最终 Itinerary 对象
        """
        # ---- Step 1: 计算天数 ----
        # (end_date - start_date).days: timedelta 的天数属性
        # +1: 例如 6/1 到 6/3 是 3 天（1号、2号、3号）
        day_count = (request.end_date - request.start_date).days + 1
        day_count = max(day_count, 1)   # 至少 1 天

        # ---- Step 2: RAG 检索 ----
        rag_contexts = get_destination_guide_context(
            destination=request.destination,
            preferences=request.preferences,
            pace=request.pace,
            special_notes=request.special_notes,
            top_k=5,
            retriever=self.retriever,
        )

        # ---- Step 3: Agent 生成草稿 ----
        llm_draft, planner_usage = generate_planner_draft(
            request, rag_contexts, day_count
        )

        token_usage = TokenUsage(
            planner_prompt=planner_usage.get("prompt_tokens", 0),
            planner_completion=planner_usage.get("completion_tokens", 0),
        )

        # ---- Step 4: 计算交通费用 ----
        daily_transport = estimate_daily_city_transport(request.destination, day_count)
        city_transport_per_day = daily_transport / max(day_count, 1)

        if not origin_city:
            origin_city = _guess_origin_city()
        intercity_cost = 0.0
        intercity_mode = None
        intercity = {}
        if origin_city and origin_city != request.destination:
            intercity = estimate_intercity_transport(origin_city, request.destination)
            intercity_cost = intercity["cost"] * 2
            intercity_mode = intercity["mode"]
            print(f"[service] 跨城交通: {origin_city} <-> {request.destination} = {intercity_cost}元 ({intercity_mode}, 来源: {intercity['source']})")

        transport_total = daily_transport + intercity_cost

        # ---- Step 5: 组装每天的 DayPlan ----
        fallback_spots = _build_fallback_spots(
            request.destination, rag_contexts, day_count
        )

        days: list[DayPlan] = []
        ticket_total = 0.0

        for i in range(day_count):
            day_number = i + 1
            # timedelta(days=i): 从 start_date 往后推 i 天
            current_date = request.start_date + timedelta(days=i)

            # 从 LLM 草稿中找对应的那一天
            llm_day = None
            if llm_draft is not None:
                # next(..., None): 取第一个匹配的，没有返回 None
                llm_day = next(
                    (d for d in llm_draft.days if d.day_index == day_number),
                    None,
                )

            # 优先用 LLM 生成的数据，失败则用规则兜底
            spot_name = llm_day.spot_name if llm_day else fallback_spots[i]
            theme = llm_day.theme if llm_day else f"第{day_number}天：{request.destination}轻松游"
            spot_desc = (
                llm_day.spot_description
                if llm_day
                else "根据本地攻略和旅行偏好安排，适合半天慢慢游览。"
            )
            meal_name = llm_day.meal_name if llm_day else f"{request.destination}特色餐饮"
            meal_note = llm_day.meal_notes if llm_day else "根据攻略推荐。"
            daily_note = llm_day.daily_note if llm_day else "今天以轻松游览为主，灵活调整。"
            ticket_cost = _estimate_ticket_cost(spot_name, spot_desc)
            ticket_total += ticket_cost

            day_plan = DayPlan(
                day_index=day_number,
                date=current_date,
                theme=theme,
                spots=[
                    SpotItem(
                        name=spot_name,
                        description=spot_desc,
                        estimated_cost=ticket_cost,
                        location=request.destination,
                    )
                ],
                meals=[
                    MealItem(
                        name=meal_name,
                        meal_type="午餐",
                        estimated_cost=80.0,
                        notes=meal_note,
                    )
                ],
                hotel=HotelItem(
                    name=f"{request.destination}舒适型住宿",
                    level="舒适型",
                    estimated_cost=300.0,
                    location=f"{request.destination}市区",
                ),
                transport=[
                    TransportItem(
                        mode="公共交通+打车",
                        from_place="酒店",
                        to_place=spot_name,
                        estimated_cost=round(city_transport_per_day, 2),
                        duration="30分钟",
                    )
                ],
                notes=[f"节奏：{request.pace or '适中'}", daily_note],
            )
            days.append(day_plan)

        # ---- Step 6: 预算汇总 ----
        hotel_total = day_count * 300.0
        meal_total = day_count * 80.0
        # 杂费: 按总预算 5% 预留（保险、纪念品等）
        other = round(request.budget * 0.05, 2)
        actual_total = transport_total + hotel_total + meal_total + ticket_total + other

        # ---- Step 7: summary & tips ----
        if llm_draft:
            summary = llm_draft.summary
            tips = llm_draft.tips
        else:
            prefs = "、".join(request.preferences) if request.preferences else "常规旅行"
            summary = f"{request.destination} {day_count} 日{prefs}之旅。"
            tips = [
                f"建议根据{request.destination}当天天气准备合适衣物。",
                "热门景点建议错峰出发，预留拍照和用餐缓冲时间。",
            ]

        # 添加交通费用提示
        if intercity_cost > 0 and intercity_mode:
            tips.insert(0,
                f"跨城交通: {origin_city} ↔ {request.destination} ({intercity_mode}) 往返约 {intercity_cost:.0f} 元 (数据来源: {intercity.get('source', '估算')})"
            )
        tips.append(
            f"市内交通: 日均约 {city_transport_per_day:.0f} 元，{day_count} 天合计 {daily_transport:.0f} 元"
        )

        # ---- Step 8: 组装最终 Itinerary ----
        itinerary = Itinerary(
            trip_id=f"trip_{request.destination}_{request.start_date.isoformat()}",
            destination=request.destination,
            summary=summary,
            days=days,
            estimated_budget=actual_total,
            budget_breakdown=BudgetBreakdown(
                transport=round(transport_total, 2),
                hotel=round(hotel_total, 2),
                meals=round(meal_total, 2),
                tickets=round(ticket_total, 2),
                other=round(other, 2),
                total=round(actual_total, 2),
            ),
            tips=tips,
            source_notes=rag_contexts[:3],
            token_usage=token_usage,
        )

        print(f"[service] 行程生成完毕: {request.destination} {day_count} 天")
        return itinerary
