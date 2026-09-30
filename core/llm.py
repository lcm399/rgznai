"""LLM 客户端封装。

统一走 OpenAI 兼容接口，DeepSeek / Qwen / GLM / OpenAI 只改环境变量即可切换。
核心功能（电路计算、FFT 等）是确定性计算、不依赖本模块，因此没配 Key 时
demo 的工程计算部分仍可正常运行，只有"对话/答疑"会给出友好提示。
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from openai import OpenAI


def _load_config() -> tuple[str, str, str]:
    """读取 API Key / 接口地址 / 模型名：优先环境变量（本地 .env），
    Streamlit Cloud 上回退到 st.secrets。"""
    def _get(key: str, default: str) -> str:
        val = os.getenv(key)
        if val:
            return val.strip()
        try:
            import streamlit as st  # 惰性导入，本地无副作用
            return str(st.secrets.get(key, default)).strip()
        except Exception:
            return default
    api_key = _get("LLM_API_KEY", "")
    base_url = _get("LLM_BASE_URL", "https://api.deepseek.com")
    model = _get("LLM_MODEL", "deepseek-chat")
    return api_key, base_url, model


def is_configured() -> bool:
    """是否已配置可用的 API Key。"""
    api_key, _, _ = _load_config()
    return bool(api_key)


def get_client() -> tuple[OpenAI, str]:
    """返回 (OpenAI 客户端, 模型名)。未配置 Key 时抛异常，由上层处理。"""
    api_key, base_url, model = _load_config()
    if not api_key:
        raise RuntimeError(
            "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入 Key，"
            "或通过系统环境变量设置 LLM_API_KEY。"
        )
    return OpenAI(api_key=api_key, base_url=base_url), model


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """一次性对话：传入消息列表，返回完整回复文本。"""
    client, model = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return resp.choices[0].message.content or ""


def chat_stream(messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
    """流式对话：逐段 yield 增量文本，适合 Streamlit 打字机效果。"""
    client, model = get_client()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
