"""统一的单位解析与参数校验。

解决两个问题：
  1. 用户/LLM 可能给出带单位的参数（"1kΩ"、"1μF"、"10mH"、"5V"、"1kHz"），
     本模块把它们统一转换为 SI 基本单位（Ω、F、H、V、Hz、s）；
  2. 对参数做合法性校验（缺失、非数值、NaN/Inf、非正），返回清晰中文错误。

纯数字输入（1000 / 0.001 / 10）完全兼容，不做任何改动。
"""
from __future__ import annotations

import math
import re

# 单位前缀（大小写敏感：m=毫，M=兆；u/μ=微）
_PREFIX = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "μ": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
}

# 每个物理量的基本单位符号（小写，用于大小写不敏感的末尾匹配）
_BASE_UNITS = {
    "resistance": ("ω", "ohm", "r"),          # Ω 的 lower 是 ω
    "capacitance": ("f", "farad"),
    "inductance": ("h", "henry"),
    "voltage": ("v", "volt"),
    "frequency": ("hz", "hertz"),
    "time": ("s", "sec", "second"),
}

_NUM_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(.*)$")


def parse(value, category: str) -> float:
    """把 value 解析为 SI 基本单位的 float。

    value 可以是 int/float，或带单位的字符串（如 "1kΩ"、"1μF"、"5V"）。
    category 取值：resistance / capacitance / inductance / voltage / frequency / time。
    解析失败抛 ValueError（中文说明）。
    """
    if isinstance(value, bool):  # bool 是 int 子类，先排除
        raise ValueError(f"参数类型错误：{value!r} 不是数值")

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        raise ValueError("参数为空")

    m = _NUM_RE.match(s)
    if not m:
        raise ValueError(f"无法解析数值：{value!r}")

    num = float(m.group(1))
    rest = m.group(2).strip()  # 剩余部分是「前缀 + 单位」或纯前缀

    for unit in _BASE_UNITS[category]:
        if rest.lower().endswith(unit):
            rest = rest[: -len(unit)].strip()
            break

    if rest == "":
        mult = 1.0
    elif rest in _PREFIX:
        mult = _PREFIX[rest]
    else:
        raise ValueError(f"无法识别的单位：{rest!r}（原始输入 {value!r}）")

    return num * mult


def require(value, category: str, name: str, *, positive: bool = False,
            non_negative: bool = False) -> float:
    """解析并校验参数，返回 float。非法时抛带中文说明的 ValueError。"""
    if value is None:
        raise ValueError(f"参数 {name} 缺失")
    val = parse(value, category)
    if not math.isfinite(val):
        raise ValueError(f"参数 {name}={value!r} 不是有限数值（可能为 NaN/Inf）")
    if positive and val <= 0:
        raise ValueError(f"参数 {name}={val} 必须为正数（不能为 0 或负数）")
    if non_negative and val < 0:
        raise ValueError(f"参数 {name}={val} 不能为负数")
    return val


def resistance(v, name="R", **kw):
    return require(v, "resistance", name, **kw)


def capacitance(v, name="C", **kw):
    return require(v, "capacitance", name, **kw)


def inductance(v, name="L", **kw):
    return require(v, "inductance", name, **kw)


def voltage(v, name="Vin", **kw):
    return require(v, "voltage", name, **kw)


def frequency(v, name="fs", **kw):
    return require(v, "frequency", name, **kw)
