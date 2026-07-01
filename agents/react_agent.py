"""ReAct Agent — Reasoning + Acting loop with tool calling.

Architecture:
  User Message → Agent Thought → Tool Call → Observation → ... → Final Answer

The Agent receives observations from tool executions and can decide:
  - To use another tool for more information
  - That current results are insufficient and re-search with better keywords
  - That it has enough information and provide the final answer
"""

import json
import re
import math
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES
from agents.tools import ToolResult, list_tools

MAX_REACT_ROUNDS = 5


def _build_system_prompt(tools_desc: str) -> str:
    return f"""你是"智旅"，一个智能旅行助手，采用ReAct框架思考与行动并行。

## 可用工具
{tools_desc}

## 响应格式
你必须严格遵循以下格式，每次只能输出一个 Thought/Action 或 Thought/Final Answer 对：

```
Thought: [你的思考过程，分析当前已知信息，判断是否需要更多信息]
Action: [工具名称]
Action Input: [JSON格式的参数，如 {{"query": "故宫 攻略", "destination": "北京"}}]
```

或者当你认为信息足够回答用户问题时：

```
Thought: [你的思考过程，确认信息足够]
Final Answer: [用户的完整回答]
```

## 规则
1. 每次响应只能包含一个 Action 或一个 Final Answer
2. Action Input 必须是合法的JSON对象
3. 如果第一次搜索的结果不够好或不够具体，可以使用更精确的关键词再次搜索
4. 对于图片识别请求，先使用 search_image_info 获取图片可能的内容，再用 rag_search 或 web_search 获取详细信息
5. 对于旅行规划请求，先用 rag_search 获取攻略，信息不足时用 web_search 补充
6. 回答使用中文，结构清晰，包含实用建议
7. 在回答末尾，简要说明使用了哪些信息来源
"""


def _build_chat_llm():
    from langchain_openai import ChatOpenAI
    if not LLM_API_KEY:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL or None,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def _parse_react_output(text: str) -> dict:
    """Parse ReAct output: extract Thought, Action/Action Input or Final Answer."""
    result = {"thought": None, "action": None, "action_input": None, "final_answer": None}

    # Extract Thought
    thought_m = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)', text, re.DOTALL)
    if thought_m:
        result["thought"] = thought_m.group(1).strip()

    # Extract Final Answer first
    fa_m = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
    if fa_m:
        result["final_answer"] = fa_m.group(1).strip()
        return result

    # Extract Action
    action_m = re.search(r'Action:\s*(\S+)', text)
    if action_m:
        result["action"] = action_m.group(1).strip()

    # Extract Action Input (JSON)
    ai_m = re.search(r'Action Input:\s*(\{.+?\})\s*$', text, re.DOTALL)
    if ai_m:
        try:
            result["action_input"] = json.loads(ai_m.group(1))
        except json.JSONDecodeError:
            # Try to find any JSON-like structure
            json_m = re.search(r'\{.+\}', text, re.DOTALL)
            if json_m:
                try:
                    result["action_input"] = json.loads(json_m.group())
                except json.JSONDecodeError:
                    pass

    return result


def _execute_tool(action: str, action_input: dict) -> ToolResult:
    """Execute a tool call and return the observation."""
    tools = list_tools()
    tool = tools.get(action)
    if tool is None:
        return ToolResult(f"错误: 未知工具 '{action}'。可用工具: {', '.join(tools.keys())}")

    try:
        return tool(**action_input)
    except TypeError as e:
        return ToolResult(f"工具参数错误: {e}。请检查 Action Input 的 JSON 格式。")
    except Exception as e:
        return ToolResult(f"工具执行异常: {e}")


class ReActAgent:
    """ReAct Agent — 思考→行动→观察→...→最终回答"""

    def __init__(self):
        self.llm = _build_chat_llm()
        self.tools = list_tools()

    def run(self, user_message: str, image_filename: str | None = None,
            image_file_path: str | None = None) -> dict:
        """
        Run the ReAct loop for a user message.

        Returns: {"answer": str, "sources": list[str], "rounds": int}
        """
        # Build tools description
        tools_desc_lines = []
        for name, tool in self.tools.items():
            tools_desc_lines.append(f"- **{name}**: {tool.description}")
        tools_desc = "\n".join(tools_desc_lines)

        system_prompt = _build_system_prompt(tools_desc)

        # Build initial message
        user_prompt = user_message
        if image_filename:
            file_path_hint = f", file_path: \"{image_file_path}\"" if image_file_path else ""
            user_prompt = (
                f"[用户上传了一张图片，文件名: {image_filename}{file_path_hint}]\n\n"
                f"你的任务流程：\n"
                f"1. 第一步：调用 vision_analyze 工具识别图片。参数: file_path=\"{image_file_path or ''}\", query_hint=\"{user_message or ''}\"\n"
                f"   — 这个工具会用多模态视觉模型真实地查看图片内容，能准确识别景点、建筑、地标。\n"
                f"2. 第二步：如果 vision_analyze 识别出了具体景点名称，"
                f"立即调用 rag_search 和/或 web_search 获取该景点的详细攻略信息。\n"
                f"3. 第三步：综合视觉识别结果和攻略信息，给出完整的回答。\n"
                f"4. 如果 vision_analyze 也无法识别，请在 Final Answer 中如实说明，并引导用户描述图片内容。\n\n"
                f"用户附带的文字描述: {user_message or '无'}\n\n"
                f"现在开始第一步，调用 vision_analyze 识别图片。"
            )

        # Conversation history
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        all_sources = []
        final_answer = None
        rounds = 0

        for round_idx in range(MAX_REACT_ROUNDS):
            rounds = round_idx + 1
            print(f"[react] 第{rounds}轮推理...")

            try:
                response = self.llm.invoke(messages)
                raw_output = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"[react] LLM调用失败: {e}")
                final_answer = f"抱歉，AI服务暂时不可用（{e}）。请稍后重试。"
                break

            print(f"[react] Agent输出: {raw_output[:200]}...")

            parsed = _parse_react_output(raw_output)

            # Check for final answer
            if parsed["final_answer"]:
                final_answer = parsed["final_answer"]
                print(f"[react] Agent给出最终回答")
                break

            # Check for action
            if not parsed["action"]:
                print(f"[react] 解析失败，未找到Action或Final Answer")
                # Try to use the raw output as answer
                if len(raw_output.strip()) > 20:
                    final_answer = raw_output.strip()
                else:
                    final_answer = "抱歉，我无法处理这个请求。请重新描述您的问题。"
                break

            # Execute the tool
            action_input = parsed["action_input"] or {}
            print(f"[react] 执行工具: {parsed['action']} {action_input}")
            result = _execute_tool(parsed["action"], action_input)

            # Collect sources
            if result.sources:
                all_sources.extend(result.sources)

            # Build observation and feed back
            observation = f"Observation: {result.content[:2000]}"
            messages.append(("assistant", raw_output))
            messages.append(("human", observation))
            print(f"[react] 观察结果: {result.content[:150]}...")

        else:
            # Max rounds reached — force final answer
            print(f"[react] 达到最大轮次({MAX_REACT_ROUNDS})，强制要求Agent回答")
            messages.append((
                "human",
                "你已经进行了足够多的搜索。请现在给出你的 Final Answer，"
                "综合所有已获取的信息回答用户的问题。"
            ))
            try:
                response = self.llm.invoke(messages)
                raw = response.content if hasattr(response, "content") else str(response)
                fa_m = re.search(r'Final Answer:\s*(.+)', raw, re.DOTALL)
                final_answer = fa_m.group(1).strip() if fa_m else raw.strip()
            except Exception as e:
                final_answer = f"抱歉，处理超时（{e}）。请简化问题后重试。"

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s[:80]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "answer": final_answer or "抱歉，无法生成回答。",
            "sources": unique_sources[:8],
            "rounds": rounds,
        }
