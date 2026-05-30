# ============================================================
# main.py - 智旅云图 CLI 入口
# ============================================================

import sys
import io

# 修复 Windows 下 stdin/stdout 编码问题：
#   Windows Python 默认用 GBK (cp936) 读写控制台，
#   但 bash/pipe 传入的 UTF-8 字符会被错误解码产生 surrogate，
#   导致后续 HTTP 请求 JSON 序列化失败。
#   这里强制用 UTF-8 重新包装 stdin/stdout。
if sys.stdin.encoding != "utf-8":
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
from datetime import datetime


def cmd_ingest():
    """
    将 data/ 下攻略文件:
      1. 切分成文本块
      2. 调用 Embedding API 转向量
      3. 写入 MySQL
      4. 构建 FAISS 索引
      5. 持久化索引到磁盘
    """
    from database.connection import engine
    from database.init_db import init_database
    from database.chunk_repo import ChunkRepository
    from rag.chunker import load_all_chunks
    from rag.embedding import EmbeddingService
    from rag.vector_index import VectorIndex

    # 1. 确保数据库表存在
    init_database()

    # 2. 加载并切分文本块
    chunks = load_all_chunks()
    print(f"[ingest] 共切分 {len(chunks)} 个文本块")

    if not chunks:
        print("[ingest] 没有找到攻略文件，请在 data/ 下放入 .md 文件")
        return

    # 3. 向量化
    embed_service = EmbeddingService()
    texts = [f"{c['title']}\n{c['text']}" for c in chunks]

    print(f"[ingest] 正在嵌入 {len(texts)} 条文本...")
    vectors = embed_service.embed_batch(texts)

    if vectors is None:
        print("[ingest] 嵌入失败，请检查 EMBEDDING_API_KEY 配置")
        return

    # 4. 写入 MySQL
    repo = ChunkRepository()
    for chunk, vec in zip(chunks, vectors):
        repo.insert_chunk(
            chunk_id=chunk["id"],
            title=chunk["title"],
            text_content=chunk["text"],
            source=chunk["source"],
            embedding=vec,
        )

    print(f"[ingest] 已写入 MySQL {len(chunks)} 条记录")

    # 5. 构建 FAISS 索引
    all_embeddings = repo.get_all_embeddings()
    chunk_ids = [row["id"] for row in all_embeddings]
    emb_list = [row["embedding"] for row in all_embeddings]

    vec_index = VectorIndex()
    vec_index.build(chunk_ids, emb_list)
    vec_index.save()

    print(f"[ingest] FAISS 索引构建完成，共 {len(chunk_ids)} 个向量")
    print("[ingest] 全部完成！")


def cmd_generate():
    """交互式生成旅行规划。"""
    from database.chunk_repo import ChunkRepository
    from database.init_db import init_database
    from datamodels.schemas import TripRequest
    from rag.embedding import EmbeddingService
    from rag.retriever import Retriever
    from rag.vector_index import VectorIndex
    from services.trip_service import TripService

    # 确保数据库表存在
    init_database()

    # 检查数据库是否有数据
    repo = ChunkRepository()
    if repo.count_chunks() == 0:
        print("数据库为空，请先运行: python main.py --ingest")
        return

    # 加载 FAISS 索引
    vec_index = VectorIndex()
    if not vec_index.load():
        print("FAISS 索引不存在，请先运行: python main.py --ingest")
        return

    # 组装模块
    embed_service = EmbeddingService()
    retriever = Retriever(vec_index, embed_service, repo)
    trip_service = TripService(retriever, repo)

    # ---- 交互式输入 ----
    print("=" * 60)
    print("  智旅云图 · 智能旅行规划系统")
    print("=" * 60)
    print()

    destination = input("请输入目的地（如：大理、成都、西安）: ").strip()
    if not destination:
        print("目的地不能为空")
        return

    start_str = input("出发日期 (YYYY-MM-DD): ").strip()
    end_str = input("结束日期 (YYYY-MM-DD): ").strip()

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        print("日期格式错误，请使用 YYYY-MM-DD 格式")
        return

    if end_date < start_date:
        print("结束日期不能早于出发日期")
        return

    travelers_str = input("出行人数 (默认 2): ").strip()
    travelers = int(travelers_str) if travelers_str else 2

    budget_str = input("总预算/元 (默认 5000): ").strip()
    budget = float(budget_str) if budget_str else 5000.0

    prefs_str = input("旅行偏好，用逗号分隔 (如: 自然风光,美食,古镇): ").strip()
    preferences = (
        [p.strip() for p in prefs_str.split(",") if p.strip()]
        if prefs_str else []
    )

    pace = input("旅行节奏 (轻松/适中/紧凑，默认 适中): ").strip() or "适中"

    notes = input("特别备注 (如: 想看日落、不想太赶，可留空): ").strip() or None

    # 构建请求对象
    request = TripRequest(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        travelers=travelers,
        budget=budget,
        preferences=preferences,
        pace=pace,
        special_notes=notes,
    )

    print()
    print(f"正在为 {destination} 规划 {(end_date - start_date).days + 1} 天行程...")
    print("-" * 60)

    # 生成行程
    itinerary = trip_service.generate_itinerary(request)

    # ---- 打印结果 ----
    print()
    print("=" * 60)
    print(f"  {itinerary.destination} 旅行规划")
    print("=" * 60)
    print()
    print(f"概述: {itinerary.summary}")
    print()

    for day in itinerary.days:
        print(f"--- 第 {day.day_index} 天 ({day.date}) ---")
        print(f"  主题: {day.theme}")
        if day.spots:
            spot = day.spots[0]
            print(f"  景点: {spot.name}")
            print(f"  介绍: {spot.description}")
            print(f"  门票: {spot.estimated_cost}元")
        if day.meals:
            meal = day.meals[0]
            print(f"  美食: {meal.name} - {meal.notes}")
        if day.hotel:
            print(f"  住宿: {day.hotel.name} ({day.hotel.estimated_cost}元)")
        for note in day.notes:
            print(f"  备注: {note}")
        print()

    print("-" * 60)
    print("预算明细:")
    bb = itinerary.budget_breakdown
    print(f"  交通: {bb.transport}元")
    print(f"  住宿: {bb.hotel}元")
    print(f"  餐饮: {bb.meals}元")
    print(f"  门票: {bb.tickets}元")
    print(f"  其他: {bb.other}元")
    print(f"  合计: {bb.total}元")
    print()

    if itinerary.tips:
        print("旅行提示:")
        for tip in itinerary.tips:
            print(f"  - {tip}")
        print()

    if itinerary.source_notes:
        print("参考攻略来源:")
        for i, src in enumerate(itinerary.source_notes, 1):
            print(f"  {i}. {src[:80]}...")
        print()

    tu = itinerary.token_usage
    if tu:
        print(f"Token 用量: 输入={tu.planner_prompt}, 输出={tu.planner_completion}")
        print()


def main():
    parser = argparse.ArgumentParser(description="智旅云图 - 智能旅行规划系统")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="将 data/ 下的攻略文件导入数据库并构建 FAISS 索引",
    )
    args = parser.parse_args()

    if args.ingest:
        cmd_ingest()
    else:
        cmd_generate()


if __name__ == "__main__":
    main()
