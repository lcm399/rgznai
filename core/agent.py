"""智能体主循环：LLM + 工具调用（Function Calling）。

流程：接收请求 → LLM 分析决策（选工具 + 定参数）→ 执行工具 → 结果回传 LLM →
生成讲解。返回结构化 trace，便于前端可视化「分析决策」过程（也是演示视频的核心素材）。
"""
from __future__ import annotations

import json

from core import llm, prompts, tools

MAX_ROUNDS = 5
HISTORY_LIMIT = 12  # 最多保留最近 12 条历史消息（约 6 轮），避免上下文无限膨胀


def run(user_msg: str, history: list[dict] | None = None, on_step=None) -> dict:
    """运行一次智能体对话。

    返回 {"trace": [{"name","args","result"}, ...], "answer": str}。
    history 为历史 user/assistant 消息（不含本次 user_msg）。
    """
    client, model = llm.get_client()
    messages = [{"role": "system", "content": prompts.SYSTEM_AGENT}]
    if history:
        messages.extend(history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": user_msg})

    trace: list[dict] = []
    for _ in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.build_tools(),
            temperature=0.3,
        )
        msg = resp.choices[0].message

        # 模型决定调用工具
        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tools.execute(name, args)
                step = {"name": name, "args": args, "result": result}
                trace.append(step)
                if on_step:
                    on_step(step)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue  # 带着工具结果再问一次模型

        # 无工具调用，返回最终回答
        return {"trace": trace, "answer": msg.content or "（未生成回答）"}

    return {"trace": trace, "answer": "（对话轮次过多，请重新描述需求）"}


def run_stream(user_msg: str, history: list[dict] | None = None, on_step=None):
    """流式运行智能体：工具调用阶段非流式（可靠判断），最终回答流式输出。

    逐次 yield 事件字典，供前端可视化：
      {"type": "step",  "step": {"name","args","result"}}   # 每次调用工具后
      {"type": "token", "text": "..."}                       # 最终回答的增量文本
      {"type": "done",  "answer": "..."}                     # 结束，answer 为完整回答
    """
    client, model = llm.get_client()
    messages = [{"role": "system", "content": prompts.SYSTEM_AGENT}]
    if history:
        messages.extend(history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": user_msg})

    trace: list[dict] = []
    for _ in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.build_tools(),
            temperature=0.3,
        )
        msg = resp.choices[0].message

        # 模型决定调用工具 → 执行后回传结果，继续下一轮
        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tools.execute(name, args)
                step = {"name": name, "args": args, "result": result,
                        "figure": tools.render(name, args)}
                trace.append(step)
                if on_step:
                    on_step(step)
                yield {"type": "step", "step": step}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # 无工具调用 → 最终回答，重新流式生成（不带工具，保证纯文本逐字输出）
        answer = msg.content or ""
        full = ""
        try:
            for delta in llm.chat_stream(messages, temperature=0.3):
                full += delta
                yield {"type": "token", "text": delta}
        except Exception:
            full = ""  # 流式失败则丢弃已输出的残片，回退到非流式结果
        if not full:
            # 流式失败或返回空时的兜底：直接使用上面已拿到的完整回答
            full = answer or "（未生成回答）"
            yield {"type": "token", "text": full}
        yield {"type": "done", "answer": full}
        return

    fallback = "（对话轮次过多，请重新描述需求）"
    yield {"type": "token", "text": fallback}
    yield {"type": "done", "answer": fallback}
