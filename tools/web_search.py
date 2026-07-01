"""
网络搜索工具 —— 当本地攻略库未覆盖目的地时，从搜索引擎获取旅行攻略。

搜索引擎优先级: Bing > Google > 搜狗
数据源策略:
  - 小红书定向: "site:xiaohongshu.com [destination] 旅游攻略"
  - 通用搜索: "[destination] 旅游攻略 景点 美食 交通"
  - 搜索结果取 title + snippet 拼成上下文，不做全页抓取（避免反爬）
"""

import re
import httpx
from bs4 import BeautifulSoup

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _search_bing(query: str, max_results: int = 5) -> list[dict]:
    """Bing 网页搜索。"""
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                "https://www.bing.com/search",
                params={"q": query, "setlang": "zh-CN", "mkt": "zh-CN"},
                headers={
                    "User-Agent": _UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )

        if resp.status_code != 200:
            print(f"[bing] Bing返回 {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select("li.b_algo"):
            title_el = item.select_one("h2 a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            snippet_el = item.select_one(".b_caption p, .b_lineclamp2, .b_algoSlug")
            body = snippet_el.get_text(strip=True) if snippet_el else ""

            if title:
                results.append({"title": title, "body": body, "href": href})
            if len(results) >= max_results:
                break

        if results:
            print(f"[bing] 获取 {len(results)} 条结果")
        return results
    except Exception as exc:
        print(f"[bing] Bing搜索失败: {type(exc).__name__}: {exc}")
        return []


def _search_sogou(query: str, max_results: int = 5) -> list[dict]:
    """搜狗网页搜索（可能触发验证码，仅作备用）。"""
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                "https://www.sogou.com/web",
                params={"query": query, "ie": "utf8"},
                headers={
                    "User-Agent": _UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )

        if resp.status_code != 200:
            return []

        # 检测验证码页面
        if "captcha" in resp.text[:500].lower() or len(resp.text) < 10000:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select(".results .vrwrap, .results .rb, .vrwrap"):
            title_el = item.select_one("h3 a, .vr-title a, .vrTitle a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            snippet_el = item.select_one(
                ".star-wiki, .space-txt, .str-text, .vr-text-info, p.str_info, .fb"
            )
            body = snippet_el.get_text(strip=True) if snippet_el else ""
            if title and body:
                results.append({"title": title, "body": body, "href": href})
            if len(results) >= max_results:
                break

        if results:
            print(f"[sogou] 获取 {len(results)} 条结果")
        return results
    except Exception as exc:
        print(f"[sogou] 搜狗搜索失败: {type(exc).__name__}: {exc}")
        return []


def search_travel_guides(
    destination: str,
    preferences: list[str] | None = None,
    max_results: int = 8,
) -> list[str]:
    """
    搜索目的地旅行攻略，优先检索小红书内容，辅以通用旅游网站。

    返回格式与 RAG retriever 一致:
      "[来源: xxx | 标题: xxx]\n正文摘要..."
    """
    contexts: list[str] = []

    # ---- 第一轮: Bing搜索 ----
    general_query = f"{destination} 旅游攻略 景点 美食 交通 住宿"
    if preferences:
        general_query += " " + " ".join(preferences)

    print(f"[web_search] Bing搜索: {general_query}")
    results = _search_bing(general_query, max_results=max_results)
    if not results:
        results = _search_sogou(general_query, max_results=max_results)

    seen_bodies = set()
    for r in results:
        if r["body"] not in seen_bodies:
            seen_bodies.add(r["body"])
            contexts.append(
                f"[来源: 网页搜索 | 标题: {r['title']}]\n{r['body']}\n原文链接: {r['href']}"
            )

    print(f"[web_search] 共获取 {len(contexts)} 条网络攻略")
    return contexts


def search_transport_info(
    from_city: str,
    to_city: str,
) -> list[str]:
    """
    搜索两地之间的交通方式和大致费用。

    返回格式化文本列表，每条约 1-2 句话。
    """
    contexts: list[str] = []

    queries = [
        f"{from_city} 到 {to_city} 高铁 票价",
        f"{from_city} 到 {to_city} 火车票 价格",
    ]

    seen = set()
    for query in queries:
        results = _search_bing(query, max_results=3)
        if not results:
            results = _search_sogou(query, max_results=3)
        for r in results:
            if r["body"] not in seen:
                seen.add(r["body"])
                contexts.append(
                    f"[来源: 交通查询 | 标题: {r['title']}]\n{r['body']}"
                )

    print(f"[web_search] 交通信息: 获取 {len(contexts)} 条")
    return contexts
