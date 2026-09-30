"""实验数据解读 —— 信号统计分析 + FFT 频谱。

对用户粘贴的示波器/频谱仪数据做确定性分析：
统计量（均值/峰值/有效值）、FFT 找主频、异常点检测。
结果交给 LLM 解读，形成"精确计算 + 智能解释"的组合。
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（避免图上中文显示为方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


def parse_data(text: str) -> tuple[np.ndarray, np.ndarray]:
    """解析用户粘贴的数据，返回 (t, y)。

    支持三种格式：
      1) 单列：每行一个采样值（例如幅度），自动编号为时间索引；
      2) 两列：时间(或序号) 与 数值，用逗号/空格/制表符分隔；
      3) 单行多值：一整行逗号/空格分隔的采样序列，自动编号为索引。
    """
    lines = [
        ln.strip() for ln in text.strip().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return np.array([]), np.array([])

    row_tokens = [ln.replace(",", " ").split() for ln in lines]

    # 格式 3：单行多值 → 视为一段采样序列
    if len(row_tokens) == 1 and len(row_tokens[0]) > 2:
        y = np.array([float(x) for x in row_tokens[0]])
        return np.arange(len(y), dtype=float), y

    # 格式 1：每行一个值
    if all(len(r) == 1 for r in row_tokens):
        y = np.array([float(r[0]) for r in row_tokens])
        return np.arange(len(y), dtype=float), y

    # 格式 2：两列（时间 数值），多余列取前两列
    t = np.array([float(r[0]) for r in row_tokens])
    y = np.array([float(r[1]) for r in row_tokens])
    return t, y


def analyze(y: np.ndarray, fs: float | None = None) -> dict:
    """计算信号统计量与频谱特征。fs 为采样率(Hz)，缺省按时间索引。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return {"错误": "没有有效数据"}

    stats = {
        "样本数": n,
        "均值": float(np.mean(y)),
        "最大值": float(np.max(y)),
        "最小值": float(np.min(y)),
        "峰峰值": float(np.ptp(y)),
        "有效值 RMS": float(np.sqrt(np.mean(y**2))),
        "标准差": float(np.std(y)),
    }

    # FFT 频谱（去直流分量后找主频）
    y_ac = y - np.mean(y)
    yf = np.fft.rfft(y_ac)
    freq = np.fft.rfftfreq(n, d=1.0 / fs if fs else 1.0)
    mag = np.abs(yf)
    # 主频（除直流外的最大值对应频率）
    peak_idx = int(np.argmax(mag[1:])) + 1 if n > 1 else 0
    dom_freq = float(freq[peak_idx]) if n > 1 else 0.0

    stats["主频"] = f"{dom_freq:.4g} Hz（采样率按 {'fs=' + str(fs) if fs else '时间索引'}）"

    # 异常点检测（3σ 准则）
    thr = 3.0 * np.std(y)
    outliers = int(np.sum(np.abs(y_ac) > thr)) if n > 1 else 0
    stats["疑似异常点数"] = outliers

    # 波形特征：峰值因数 + 粗略波形类型 + 总谐波失真 THD（需 fs）
    rms = float(np.sqrt(np.mean(y_ac ** 2))) if n else 0.0
    peak = float(np.max(np.abs(y_ac))) if n else 0.0
    crest = peak / rms if rms > 0 else 0.0
    stats["峰值因数"] = f"{crest:.4g}"
    if crest < 1.15:
        wtype = "方波"
    elif crest < 1.65:
        wtype = "正弦波"
    else:
        wtype = "三角波/脉冲"
    stats["波形类型(粗略)"] = wtype

    if fs and n >= 8 and peak_idx > 0 and mag[peak_idx] > 0:
        fund = mag[peak_idx]
        others = float(np.sum(mag[1:] ** 2)) - float(fund ** 2)
        thd = math.sqrt(max(others, 0.0)) / fund * 100.0
        stats["总谐波失真 THD"] = f"{thd:.3g}%"
    return stats


def plot_signal(t: np.ndarray, y: np.ndarray, fs: float | None = None) -> plt.Figure:
    """绘制时域波形 + 幅度谱。"""
    n = len(y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5))

    ax1.plot(t, y, color="#2b6cb0", lw=1.2)
    ax1.set_xlabel("时间/序号")
    ax1.set_ylabel("幅值")
    ax1.set_title("时域波形")
    ax1.grid(True, alpha=0.3)

    y_ac = y - np.mean(y)
    yf = np.abs(np.fft.rfft(y_ac))
    freq = np.fft.rfftfreq(n, d=1.0 / fs if fs else 1.0)
    ax2.stem(freq, yf, linefmt="#2b6cb0", markerfmt="o", basefmt=" ")
    ax2.set_xlabel("频率 (Hz)")
    ax2.set_ylabel("幅度")
    ax2.set_title("幅度谱 (FFT，去直流)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
