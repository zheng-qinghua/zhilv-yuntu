import logging

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


def _llm_rewrite_query(
    destination: str,
    preferences: list[str] | None = None,
    pace: str | None = None,
    special_notes: str | None = None,
) -> str | None:
    """
    用 LLM 把自然语言需求改写成检索关键词。

    为什么需要这一步：
      用户输入 "大理慢节奏看日落拍照之旅" 是自然语言，直接向量检索效果差。
      因为攻略文本里不会写"慢节奏之旅"，而是写"洱海生态廊道适合骑行，日落时分非常出片"。
      LLM 把自然语言 → 信息密集的关键词（如 "大理 洱海 双廊 日落 拍照 轻松 攻略"），
      检索命中率大幅提升。
    """
    if not LLM_API_KEY:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("langchain_openai 未安装，无法使用 LLM Query Rewrite")
        return None

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL or None,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )

    system_prompt = (
        "你是一个 RAG 检索 query 改写专家。"
        "你的任务是把用户的旅行需求改写成适合向量检索的关键词组合。"
        "规则："
        "1. 只输出关键词，用空格分隔"
        "2. 不要输出解释、标点或任何多余文字"
        "3. 关键词要具体，优先包含：景点名称、活动类型、场景特征"
        "4. 必须包含目的地城市名"
    )

    parts = [f"目的地：{destination}"]
    if preferences:
        parts.append(f"偏好：{'、'.join(preferences)}")
    if pace:
        parts.append(f"节奏：{pace}")
    if special_notes:
        parts.append(f"备注：{special_notes}")

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", "\n".join(parts)),
        ])

        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            raw = "".join(str(x) for x in raw)

        query = raw.strip()
        if query:
            logger.info(f"[rewrite] 输入: {parts} → 输出: {query}")
            return query
    except Exception as exc:
        logger.warning(f"[rewrite] LLM 调用失败: {exc}")

    return None


def _rule_rewrite_query(
    destination: str,
    preferences: list[str] | None = None,
    pace: str | None = None,
    special_notes: str | None = None,
) -> str:
    """
    规则级 Query Rewrite（LLM 不可用时的 fallback）。

    用预定义的映射表从备注中提取关键词，不需要调 LLM，零成本。
    """
    keywords: list[str] = [destination]

    if preferences:
        for p in preferences:
            if p not in keywords:
                keywords.append(p)

    if pace:
        keywords.append(pace)

    note_keyword_map = {
        "日落": ["日落", "傍晚", "双廊", "洱海"],
        "日出": ["日出", "清晨", "才村"],
        "拍照": ["拍照", "摄影", "出片"],
        "美食": ["美食", "小吃", "餐饮"],
        "轻松": ["轻松", "慢节奏", "休闲"],
        "骑行": ["骑行", "洱海生态廊道"],
        "古镇": ["古镇", "古城"],
        "熊猫": ["大熊猫", "熊猫基地"],
        "潜水": ["潜水", "水上项目"],
    }

    if special_notes:
        for trigger, values in note_keyword_map.items():
            if trigger in special_notes:
                for v in values:
                    if v not in keywords:
                        keywords.append(v)

    for term in ["景点", "行程", "攻略", "推荐"]:
        if term not in keywords:
            keywords.append(term)

    return " ".join(keywords)


def get_destination_guide_context(
    destination: str,
    preferences: list[str] | None = None,
    pace: str | None = None,
    special_notes: str | None = None,
    top_k: int = 5,
    retriever: Retriever | None = None,
) -> list[str]:
    """
    对外接口：根据旅行需求返回攻略上下文。

    流程:
      1. Query Rewrite（LLM 优先、规则 fallback）
      2. 本地 RAG 检索
      3. 如果本地结果不足（< 3 条），自动上网搜索补充
      4. 返回格式化文本列表（本地 + 网络混合）
    """
    # Step 1: Query Rewrite
    query = _llm_rewrite_query(
        destination=destination,
        preferences=preferences,
        pace=pace,
        special_notes=special_notes,
    )
    if not query:
        query = _rule_rewrite_query(
            destination=destination,
            preferences=preferences,
            pace=pace,
            special_notes=special_notes,
        )

    print(f"[rag_tool] 检索 query: {query}")

    # Step 2: 本地 RAG 检索
    local_contexts: list[str] = []
    if retriever is not None:
        local_contexts = retriever.retrieve(
            query=query,
            top_k=top_k,
            destination=destination,
        )
    print(f"[rag_tool] 本地检索到 {len(local_contexts)} 条上下文")

    # Step 3: 本地结果相关性校验 + 不足时联网搜索
    MIN_LOCAL_THRESHOLD = 3
    need_web_search = len(local_contexts) < MIN_LOCAL_THRESHOLD

    if not need_web_search and destination:
        # 检查本地结果是否真的与目的地相关
        relevant_count = sum(
            1 for c in local_contexts
            if destination in c  # 目的地名出现在上下文中
        )
        relevance_ratio = relevant_count / max(len(local_contexts), 1)
        print(f"[rag_tool] 目的地相关性: {relevant_count}/{len(local_contexts)} = {relevance_ratio:.0%}")

        if relevance_ratio < 0.4:
            print(f"[rag_tool] 本地结果与'{destination}'相关性不足，标记为需联网搜索")
            need_web_search = True
            # 扔掉不相关的结果
            local_contexts = [c for c in local_contexts if destination in c]

    if need_web_search:
        reason = "数量不足" if len(local_contexts) < MIN_LOCAL_THRESHOLD else "目的地不匹配"
        print(f"[rag_tool] {reason}，启动联网搜索...")

        try:
            from tools.web_search import search_travel_guides
            web_contexts = search_travel_guides(
                destination=destination,
                preferences=preferences,
                max_results=8,
            )
            # 本地结果 + 网络结果组合
            combined = local_contexts + web_contexts
            # 去重并限制 top_k
            seen = set()
            unique: list[str] = []
            for c in combined:
                key = c[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            local_contexts = unique[:top_k]
            print(f"[rag_tool] 组合后共 {len(local_contexts)} 条上下文（本地+网络）")
        except ImportError:
            print("[rag_tool] 联网搜索依赖未安装。pip install duckduckgo-search bs4 lxml")
        except Exception as exc:
            print(f"[rag_tool] 联网搜索异常: {exc}")

    return local_contexts
