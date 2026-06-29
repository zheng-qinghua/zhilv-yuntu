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
from services.nl_parser import NLToTripParser
from services.trip_service import TripService
from datamodels.schemas import TripRequest


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
    """自然语言交互式生成旅行规划。"""
    from database.chunk_repo import ChunkRepository
    from database.init_db import init_database
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

    # ---- 出发城市 ----
    from config import TRAVEL_ORIGIN_CITY
    default_origin = TRAVEL_ORIGIN_CITY or "北京"
    origin_input = input(f"您的出发城市 (默认 {default_origin}): ").strip()
    origin_city = origin_input if origin_input else default_origin
    print(f"  出发城市: {origin_city}")

    # ---- 自然语言交互式输入 ----
    print()
    print("请用一句话描述您的旅行需求，例如：")
    print('  "我想在6月4号到6月14号去成都旅游，一个人，预算5000，节奏适中"')
    print('  "6月10日去大理，两个人，预算3000，喜欢美食和古镇"')
    print()

    parser = NLToTripParser()

    while True:
        user_input = input("您的需求: ").strip()
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        try:
            parsed_request = parser.parse(user_input)
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            retry = input("是否重新输入？(y/n): ").strip().lower()
            if retry != 'y':
                return
            continue

        # 显示解析结果供确认
        print("\n📋 我理解的需求如下：")
        print(f"  目的地: {parsed_request.destination}")
        print(f"  日期: {parsed_request.start_date} 至 {parsed_request.end_date}")
        print(f"  人数: {parsed_request.num_travelers}")
        print(f"  预算: {parsed_request.budget} 元")
        print(f"  节奏: {parsed_request.pace}")
        prefs_display = parsed_request.preferences if parsed_request.preferences else "无"
        print(f"  偏好: {prefs_display}")
        print(f"  备注: {parsed_request.special_notes if parsed_request.special_notes else '无'}")

        break

    # 将解析结果转换为 TripRequest 所需格式
    from datetime import datetime

    start_date = datetime.strptime(parsed_request.start_date, "%Y-%m-%d").date()
    if parsed_request.end_date:
        end_date = datetime.strptime(parsed_request.end_date, "%Y-%m-%d").date()
    else:
        from datetime import timedelta
        end_date = start_date + timedelta(days=2)
        print(f"  (未指定结束日期，默认 {end_date.isoformat()}，共 {(end_date - start_date).days + 1} 天)")


    # 构建请求对象（字段名与原有 TripRequest 一致）
    request = TripRequest(
        destination=parsed_request.destination,
        start_date=start_date,
        end_date=end_date,
        travelers=parsed_request.num_travelers,
        budget=float(parsed_request.budget),  # budget 已经是数值
        preferences=parsed_request.preferences,  # 直接使用解析器返回的列表
        pace=parsed_request.pace,
        special_notes=parsed_request.special_notes or None,
    )

    print()
    print(f"正在为 {request.destination} 规划 {(request.end_date - request.start_date).days + 1} 天行程...")
    print("-" * 60)

    # 生成行程
    itinerary = trip_service.generate_itinerary(request, origin_city=origin_city)

    # ---- 打印结果（与原代码相同） ----
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
