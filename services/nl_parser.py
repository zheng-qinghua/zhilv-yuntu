# -*- coding: utf-8 -*-

"""
自然语言旅行需求解析器
将用户的一句话输入（如："6月4号到6月14号去成都，一个人，预算5000"）
解析为结构化的 TripRequest 对象。
"""

import json
import re
from datetime import datetime
from typing import Optional
from openai import OpenAI
from datamodels.schemas import TripRequest
from config import (
    LLM_API_KEY,
    LLM_MODEL,
)


class NLToTripParser:
    def __init__(self):
        # 使用 OpenAI SDK 调用 DeepSeek API（兼容模式）
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url="https://api.deepseek.com"   # DeepSeek 官方 API 端点
        )
        self.model = LLM_MODEL   # 例如 "deepseek-chat" 或 "deepseek-v4-pro"

    def parse(self, user_input: str):
        """将自然语言解析为 TripRequest 所需的字段对象"""
        system_prompt = "你是一个旅行需求解析器，请严格按照 JSON 格式返回结果，不要包含任何额外文字。"
        user_prompt = f"""
    请从以下用户输入中提取旅行计划所需的信息。如果某个信息未明确提及，就设为 null。

    用户输入：{user_input}

    你需要提取的字段及说明：
    - destination: 目的地城市（如成都、大理、西安）
    - start_date: 出发日期，格式 YYYY-MM-DD（如果用户说“6月4号”，就转为 2026-06-04；年份默认当前年或用户提到的年份）
    - end_date: 结束日期，格式 YYYY-MM-DD
    - num_travelers: 人数（整数）
    - budget: 总预算（整数，单位元）
    - pace: 旅行节奏，只能是“轻松”、“适中”、“紧凑”之一，如果没明确说就设为 null
    - preferences: 偏好列表，用逗号分隔，例如“自然风光,美食”。如果用户提到多个就用逗号拼起来，否则为 null
    - special_notes: 特别备注（例如“想看日落”、“不想太赶”等），没有则为 null

    请仅返回一个 JSON 对象，格式如下：
    {{
      "destination": "成都",
      "start_date": "2026-06-04",
      "end_date": "2026-06-14",
      "num_travelers": 1,
      "budget": 5000,
      "pace": "适中",
      "preferences": "自然风光,美食",
      "special_notes": null
    }}
    """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"调用 DeepSeek API 失败: {e}")

        # 解析 JSON
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("LLM 返回的内容无法解析为 JSON")

        # 标准化日期（补全年份 + 纠正 LLM 幻觉的过期年份）
        current_year = datetime.now().year
        start_date = data.get("start_date")
        if start_date:
            start_date = self._normalize_date(start_date, current_year)
        end_date = data.get("end_date")
        if end_date:
            end_date = self._normalize_date(end_date, current_year)

        # 处理 preferences：字符串 -> 列表
        preferences_raw = data.get("preferences")
        if preferences_raw is None or preferences_raw == "":
            preferences_list = []
        elif isinstance(preferences_raw, str):
            # 分割并去除空格，过滤掉空字符串
            preferences_list = [item.strip() for item in preferences_raw.split(",") if item.strip()]
        else:
            # 如果已经是列表，直接使用
            preferences_list = preferences_raw if isinstance(preferences_raw, list) else []

        # 处理 budget：确保是数值
        budget_raw = data.get("budget")
        if budget_raw is None:
            budget_value = 5000.0
        else:
            try:
                budget_value = float(budget_raw)
            except (ValueError, TypeError):
                budget_value = 5000.0

        # 构造简单的返回对象
        class ParsedRequest:
            pass

        parsed = ParsedRequest()
        parsed.destination = data.get("destination")
        parsed.start_date = start_date
        parsed.end_date = end_date
        parsed.num_travelers = data.get("num_travelers") or 2
        parsed.budget = budget_value
        parsed.preferences = preferences_list  # 已经是 list[str]
        parsed.pace = data.get("pace") or "适中"
        parsed.special_notes = data.get("special_notes") or ""

        return parsed

    @staticmethod
    def _normalize_date(date_str: str, default_year: int) -> str:
        """确保日期格式为 YYYY-MM-DD，纠正 LLM 幻觉的过期年份。"""
        # 如果已经是完整格式，检查年份是否合理
        full_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
        if full_match:
            year = int(full_match.group(1))
            # 如果 LLM 返回的年份 < 当前年份，修正为当前年份
            if year < default_year:
                return f"{default_year}-{full_match.group(2)}-{full_match.group(3)}"
            return date_str
        # 尝试匹配 MM-DD 格式
        match = re.match(r'(\d{1,2})-(\d{1,2})', date_str)
        if match:
            month, day = match.groups()
            return f"{default_year}-{int(month):02d}-{int(day):02d}"
        return date_str