"""简易 RAG 知识库（无外部向量库依赖）。

把实验手册 / 教材片段放进 data/ 目录（.md / .txt），本模块按关键词
相关度做检索，把命中的片段拼进 Prompt，让 LLM"带着资料回答"，
显著降低胡编乱造的概率、提高专业准确率。

说明：这是轻量版（关键词打分），不依赖 embedding API，10 天工期够用。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 400  # 每个片段约 400 字，便于检索与拼接


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """按字符数粗切分，尽量在句号/换行处断开。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 回退到最近的句号/换行，避免切断句子
            for sep in "。\n":
                pos = text.rfind(sep, start + size // 2, end)
                if pos > 0:
                    end = pos + 1
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


@lru_cache(maxsize=1)
def load_knowledge() -> list[dict]:
    """加载 data/ 下所有文本文件，返回 [{source, text}, ...]。

    结果做进程级缓存：知识库内容在一次运行内不变，避免每次检索都重复读盘/切分。
    若运行期间新增了知识文件，重启应用即可生效。
    """
    docs = []
    if not DATA_DIR.exists():
        return docs
    for p in sorted(DATA_DIR.glob("*")):
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for chunk in _chunk_text(text):
            docs.append({"source": p.name, "text": chunk})
    return docs


def _tokenize(s: str) -> set[str]:
    """简单分词：提取中英文单词/数字。"""
    return set(re.findall(r"[A-Za-z0-9]+|[一-鿿]", s.lower()))


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """关键词重合度检索，返回 top_k 个相关片段。"""
    docs = load_knowledge()
    if not docs:
        return []
    q_tokens = _tokenize(query)
    scored = []
    for d in docs:
        d_tokens = _tokenize(d["text"])
        if not q_tokens:
            score = 0
        else:
            overlap = q_tokens & d_tokens
            score = len(overlap) / len(q_tokens)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k] if s > 0]


def build_context(query: str, top_k: int = 4) -> str:
    """把检索到的片段拼成上下文字符串，供填入 Prompt。"""
    hits = retrieve(query, top_k)
    if not hits:
        return ""
    parts = ["【知识库资料】"]
    for i, d in enumerate(hits, 1):
        parts.append(f"[{i}]（来源：{d['source']}）\n{d['text']}")
    return "\n\n".join(parts)
