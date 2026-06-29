"""
交通费用查询工具。

策略:
  - 通过实时 web 搜索获取城市间高铁/飞机票价
  - 用 LLM 从搜索结果中提取价格区间
  - 城市内交通用车费模型估算（打车 + 公交）

为什么不用 12306/Ctrip 官方 API:
  - 12306 需要实名认证且封闭；Ctrip 无反爬极严
  - 用搜索引擎获取公开的价格信息更稳定、不触发反爬
"""

import re
import json
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES
from tools.web_search import search_transport_info


# ---- 跨城市交通价格热数据缓存 ----
# 中国主要城市间高铁二等座参考票价（元），2026年基准
# 数据来源: 12306.cn 高铁二等座公布票价
_KNOWN_TRANSPORT_COSTS: dict[str, dict[str, float]] = {
    "北京": {
        "上海": 553, "广州": 862, "深圳": 944, "西安": 515, "成都": 778,
        "杭州": 626, "南京": 443, "武汉": 520, "重庆": 790, "长沙": 649,
        "郑州": 309, "济南": 194, "天津": 54, "沈阳": 296, "哈尔滨": 544,
        "昆明": 1147, "贵阳": 882, "厦门": 790, "福州": 719, "合肥": 436,
        "南昌": 627, "太原": 197, "兰州": 690, "呼和浩特": 197,
    },
    "上海": {
        "北京": 553, "广州": 793, "深圳": 867, "成都": 932, "西安": 669,
        "杭州": 73, "南京": 139, "苏州": 39, "武汉": 336, "重庆": 859,
        "长沙": 478, "郑州": 447, "济南": 398, "天津": 508, "厦门": 428,
        "福州": 373, "合肥": 203, "南昌": 334, "昆明": 879,
    },
    "广州": {
        "深圳": 74, "珠海": 70, "长沙": 314, "武汉": 463, "成都": 566,
        "重庆": 453, "西安": 813, "北京": 862, "上海": 793, "杭州": 720,
        "南京": 627, "厦门": 254, "福州": 339, "南宁": 169, "贵阳": 317,
        "昆明": 480, "桂林": 164, "三亚": 350, "海口": 200,
    },
    "深圳": {
        "广州": 74, "珠海": 144, "厦门": 190, "长沙": 388, "成都": 672,
        "重庆": 570, "武汉": 538, "北京": 944, "上海": 867, "杭州": 820,
        "南京": 704, "福州": 264, "昆明": 610, "贵阳": 430,
    },
    "成都": {
        "西安": 263, "重庆": 154, "贵阳": 260, "昆明": 487, "北京": 778,
        "上海": 932, "广州": 566, "深圳": 672, "武汉": 375, "长沙": 584,
        "郑州": 502, "兰州": 230, "西宁": 285, "拉萨": 800,
    },
    "西安": {
        "成都": 263, "北京": 515, "上海": 669, "郑州": 239, "兰州": 174,
        "武汉": 454, "重庆": 328, "广州": 813, "南京": 540, "杭州": 653,
        "太原": 178, "银川": 245, "西宁": 232, "乌鲁木齐": 566,
    },
    "杭州": {
        "上海": 73, "南京": 117, "苏州": 110, "黄山": 120, "北京": 626,
        "广州": 720, "深圳": 820, "武汉": 386, "厦门": 389, "福州": 268,
        "南昌": 263, "合肥": 256, "长沙": 405,
    },
    "南京": {
        "上海": 139, "杭州": 117, "苏州": 99, "北京": 443, "武汉": 210,
        "广州": 627, "深圳": 704, "西安": 540, "郑州": 317, "合肥": 67,
        "济南": 279, "天津": 406,
    },
    "重庆": {
        "成都": 154, "贵阳": 138, "武汉": 286, "西安": 328, "长沙": 424,
        "广州": 453, "深圳": 570, "北京": 790, "上海": 859, "昆明": 671,
    },
    "武汉": {
        "长沙": 164, "南京": 210, "重庆": 286, "郑州": 244, "西安": 454,
        "广州": 463, "深圳": 538, "北京": 520, "上海": 336, "成都": 375,
        "合肥": 133, "南昌": 107, "杭州": 386,
    },
    "长沙": {
        "武汉": 164, "广州": 314, "深圳": 388, "重庆": 424, "南昌": 157,
        "北京": 649, "上海": 478, "成都": 584, "杭州": 405, "贵阳": 258,
        "昆明": 498, "南宁": 289,
    },
    "郑州": {
        "北京": 309, "西安": 239, "武汉": 244, "南京": 317, "上海": 447,
        "成都": 502, "济南": 182, "太原": 172, "合肥": 307, "长沙": 393,
    },
    "昆明": {
        "大理": 145, "丽江": 220, "成都": 487, "贵阳": 212, "西双版纳": 180,
        "广州": 480, "深圳": 610, "北京": 1147, "上海": 879, "重庆": 671,
    },
    "大理": {
        "昆明": 145, "丽江": 80, "成都": 690, "重庆": 660,
    },
    "三亚": {
        "海口": 100, "广州": 350, "深圳": 350,
    },
    "厦门": {
        "深圳": 190, "福州": 93, "杭州": 389, "上海": 428, "广州": 254,
        "南京": 480, "南昌": 272,
    },
    "济南": {
        "北京": 194, "上海": 398, "南京": 279, "郑州": 182, "青岛": 121,
        "天津": 139,
    },
    "青岛": {
        "济南": 121, "北京": 314, "上海": 570, "南京": 398,
    },
    "天津": {
        "北京": 54, "济南": 139, "南京": 406, "上海": 508, "沈阳": 245,
    },
    "苏州": {"上海": 39, "南京": 99, "杭州": 110, "北京": 523},
    "合肥": {"南京": 67, "上海": 203, "武汉": 133, "北京": 436, "杭州": 256, "郑州": 307},
    "南昌": {"武汉": 107, "长沙": 157, "杭州": 263, "上海": 334, "广州": 472, "厦门": 272},
    "福州": {"厦门": 93, "杭州": 268, "上海": 373, "深圳": 264, "广州": 339},
    "贵阳": {"重庆": 138, "成都": 260, "昆明": 212, "长沙": 258, "广州": 317, "深圳": 430},
    "南宁": {"广州": 169, "长沙": 289, "桂林": 128, "昆明": 256},
    "兰州": {"西安": 174, "成都": 230, "西宁": 58, "乌鲁木齐": 496},
    "太原": {"北京": 197, "西安": 178, "郑州": 172, "石家庄": 68},
    "哈尔滨": {"北京": 544, "沈阳": 245, "长春": 109, "大连": 403},
    "沈阳": {"北京": 296, "天津": 245, "哈尔滨": 245, "大连": 173},
    "大连": {"沈阳": 173, "北京": 397, "哈尔滨": 403},
    "呼和浩特": {"北京": 197, "包头": 64, "太原": 209},
    "海口": {"三亚": 100, "广州": 480, "深圳": 420},
    "乌鲁木齐": {"兰州": 496, "西安": 566},
    "拉萨": {"成都": 800, "西宁": 224},
    "西宁": {"兰州": 58, "西安": 232, "成都": 285, "拉萨": 224},
}


def _parse_price_from_text(text: str) -> float | None:
    """
    从文本中提取价格信息（人民币元）。

    匹配模式:
      - "票价约 300 元"
      - "二等座 553 元"
      - "¥263"
      - "200-500元"
    """
    # 带单位的数字
    patterns = [
        r'约\s*(\d+)\s*元',
        r'票价[约]?\s*(\d+)\s*元',
        r'二等座\s*(\d+)\s*元',
        r'(\d+)\s*元[起/左右]?',
        r'[¥￥]\s*(\d+)',
        r'价格[：:]\s*(\d+)',
        r'(\d+)[-~]\d+\s*元',  # 取最低价
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            price = float(match.group(1))
            if 10 <= price <= 5000:  # 合理范围
                return price

    # 尝试匹配纯数字（如果前面有明显交通相关词）
    transport_patterns = [
        r'高铁[^。]+\b(\d{2,4})\b',
        r'火车[^。]+\b(\d{2,4})\b',
        r'机票[^。]+\b(\d{3,4})\b',
    ]
    for pat in transport_patterns:
        match = re.search(pat, text)
        if match:
            price = float(match.group(1))
            if 10 <= price <= 5000:
                return price

    return None


def _llm_extract_price(texts: list[str]) -> float | None:
    """
    当正则解析失败时，用 LLM 从搜索结果中提取交通价格。
    """
    if not LLM_API_KEY or not texts:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL or None,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )

    combined = "\n---\n".join(texts[:5])

    system_prompt = (
        "你是交通价格提取器。从给定文本中提取两个城市之间的交通费用。"
        "请只输出一个 JSON: {\"price\": 数字, \"transport_type\": \"高铁/飞机/火车\"}"
        "如果找不到价格，输出 {\"price\": null}。不要输出其他内容。"
    )

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", f"文本:\n{combined}"),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        data = json.loads(raw.strip())
        price = data.get("price")
        if price and isinstance(price, (int, float)) and 10 <= price <= 5000:
            return float(price)
    except Exception:
        pass

    return None


def estimate_intercity_transport(
    from_city: str,
    to_city: str,
    prefer_highspeed: bool = True,
) -> dict:
    """
    估算两个城市之间的单程交通费用。

    三级策略:
      1. 查已知价格表（最快）
      2. 正则从 web 搜索结果提取
      3. LLM 提取
      4. 距离估算兜底

    返回: {"cost": float, "mode": str, "duration_hours": float, "source": str}
    """
    # ---- Level 1: 热数据缓存 ----
    city_map = _KNOWN_TRANSPORT_COSTS.get(from_city, {})
    if to_city in city_map:
        return {
            "cost": city_map[to_city],
            "mode": "高铁" if prefer_highspeed else "火车",
            "duration_hours": _estimate_duration(from_city, to_city),
            "source": "本地交通价格库",
        }
    # 双向查找
    reverse_map = _KNOWN_TRANSPORT_COSTS.get(to_city, {})
    if from_city in reverse_map:
        return {
            "cost": reverse_map[from_city],
            "mode": "高铁" if prefer_highspeed else "火车",
            "duration_hours": _estimate_duration(from_city, to_city),
            "source": "本地交通价格库",
        }

    # ---- Level 2: Web 搜索 + 正则 ----
    # 先算距离估算值作为底线参考
    floor_price = _estimate_by_distance(from_city, to_city) * 0.2

    print(f"[transport_cost] 搜索 {from_city} → {to_city} 交通费用...")
    search_results = search_transport_info(from_city, to_city)

    for text in search_results:
        price = _parse_price_from_text(text)
        if price and price >= floor_price:
            return {
                "cost": price,
                "mode": "高铁" if prefer_highspeed else "火车",
                "duration_hours": _estimate_duration(from_city, to_city),
                "source": "网络搜索",
            }

    # ---- Level 3: LLM 提取 ----
    llm_price = _llm_extract_price(search_results)
    if llm_price and llm_price >= floor_price:
        return {
            "cost": llm_price,
            "mode": "高铁",
            "duration_hours": _estimate_duration(from_city, to_city),
            "source": "AI 估算",
        }

    # ---- Level 4: 距离估算兜底 ----
    estimated = _estimate_by_distance(from_city, to_city)
    return {
        "cost": estimated,
        "mode": "高铁",
        "duration_hours": _estimate_duration(from_city, to_city),
        "source": "距离估算",
    }


def estimate_daily_city_transport(destination: str, days: int) -> float:
    """
    估算城市内每日交通费用。

    简化模型:
      - 一线城市（北京/上海/广州/深圳）: 日均 80 元（地铁+打车混合）
      - 旅游城市（大理/三亚/丽江等）: 日均 100 元（景点分散，打车多）
      - 普通城市: 日均 60 元
    """
    tier1 = {"北京", "上海", "广州", "深圳", "杭州"}
    tourist = {"大理", "丽江", "三亚", "张家界", "桂林", "九寨沟", "黄山", "西双版纳"}

    if destination in tier1:
        daily = 80.0
    elif destination in tourist:
        daily = 100.0
    else:
        daily = 60.0

    return daily * days


def _estimate_duration(from_city: str, to_city: str) -> float:
    """根据城市名粗略估算高铁时长。"""
    duration_map = {
        ("北京", "上海"): 4.5, ("北京", "西安"): 4.5, ("北京", "成都"): 7.5,
        ("上海", "杭州"): 1.0, ("上海", "南京"): 1.5, ("上海", "苏州"): 0.5,
        ("成都", "西安"): 3.5, ("成都", "重庆"): 1.5, ("成都", "贵阳"): 3.0,
        ("西安", "郑州"): 2.0, ("西安", "兰州"): 3.0,
        ("广州", "深圳"): 0.5, ("广州", "珠海"): 1.0, ("广州", "长沙"): 2.5,
        ("武汉", "长沙"): 1.5, ("武汉", "南京"): 3.0,
        ("重庆", "贵阳"): 2.0, ("重庆", "武汉"): 4.0,
        ("昆明", "大理"): 2.0, ("昆明", "丽江"): 3.5,
        ("大理", "丽江"): 1.5,
    }

    key = (from_city, to_city)
    rev_key = (to_city, from_city)
    if key in duration_map:
        return duration_map[key]
    if rev_key in duration_map:
        return duration_map[rev_key]

    # 默认: 中等距离 3.0 小时
    return 3.0


def _estimate_by_distance(from_city: str, to_city: str) -> float:
    """
    基于城市名粗略估算高铁票价（元/km * 估计距离）。
    中国高铁二等座约 0.46 元/km。
    """
    # 城市坐标（大致经纬度近似）
    coords = {
        "北京": (39.9, 116.4), "上海": (31.2, 121.5), "成都": (30.6, 104.1),
        "西安": (34.3, 108.9), "杭州": (30.3, 120.2), "南京": (32.1, 118.8),
        "广州": (23.1, 113.3), "深圳": (22.5, 114.1), "重庆": (29.6, 106.5),
        "武汉": (30.6, 114.3), "长沙": (28.2, 113.0), "昆明": (25.0, 102.7),
        "大理": (25.6, 100.3), "丽江": (26.9, 100.2), "三亚": (18.2, 109.5),
        "厦门": (24.5, 118.1), "贵阳": (26.6, 106.7), "郑州": (34.7, 113.6),
        "兰州": (36.1, 103.8), "苏州": (31.3, 120.6), "福州": (26.1, 119.3),
        "合肥": (31.8, 117.2), "南昌": (28.7, 115.9), "济南": (36.7, 117.0),
        "太原": (37.9, 112.5), "南宁": (22.8, 108.4), "海口": (20.0, 110.3),
    }

    if from_city not in coords or to_city not in coords:
        return 300.0  # 默认

    import math
    lat1, lon1 = coords[from_city]
    lat2, lon2 = coords[to_city]

    # 用赤道长度近似计算两点间的球面距离 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    distance_km = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 高铁票价约 0.46 元/km，加 20% 浮动
    cost = distance_km * 0.46 * 1.2
    return round(cost, 2)
