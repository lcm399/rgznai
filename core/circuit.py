"""电路分析 —— 确定性计算核心。

用解析公式精确计算（不靠 LLM），保证结果可复现、不"幻觉"。
目前支持：
  - RC 低通 / 高通滤波器：截止频率 + Bode 图（幅频/相频）
  - RL 低通 / 高通滤波器：截止频率 + Bode 图
  - RLC 串联谐振：谐振频率、品质因数、带宽 + 电流幅频响应曲线
  - RC 充放电瞬态（一阶暂态）：时间常数 τ + 充/放电曲线
  - RLC 阶跃响应（二阶暂态）：阻尼比、欠/过/临界阻尼、超调量 + 曲线
  - 电阻分压 / 运放（反相、同相）放大
"""
from __future__ import annotations

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 无界面后端，供 Streamlit 渲染
import matplotlib.pyplot as plt

# 中文字体（避免图上中文显示为方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


def rc_filter_cutoff(R: float, C: float) -> dict:
    """RC 滤波器的截止频率（-3dB 点）。R 单位欧姆，C 单位法拉。"""
    fc = 1.0 / (2.0 * math.pi * R * C)
    return {
        "fc": fc,
        "fc_khz": fc / 1e3,
        "tau": R * C,  # 时间常数 τ = RC（秒）
    }


def _rc_magnitude_phase(f: np.ndarray, fc: float, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """RC 滤波器的幅频/相频特性。kind: 'lowpass' 或 'highpass'。"""
    ratio = f / np.where(fc > 0, fc, 1.0)
    if kind == "lowpass":
        mag = 1.0 / np.sqrt(1.0 + ratio**2)          # |H(f)|
        phase = -np.degrees(np.arctan(ratio))         # 相位（度）
    else:  # highpass
        mag = ratio / np.sqrt(1.0 + ratio**2)
        phase = 90.0 - np.degrees(np.arctan(ratio))
    return mag, phase


def _plot_first_order_bode(fc: float, kind: str, title: str) -> plt.Figure:
    """绘制一阶滤波器 Bode 图（RC/RL 共用）。"""
    fmin, fmax = max(fc / 100, 1.0), fc * 100
    f = np.logspace(np.log10(fmin), np.log10(fmax), 500)
    mag, phase = _rc_magnitude_phase(f, fc, kind)
    mag_db = 20 * np.log10(mag)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    ax1.semilogx(f, mag_db, color="#2b6cb0", lw=2)
    ax1.axhline(-3, color="gray", ls="--", lw=1)
    ax1.axvline(fc, color="#e53e3e", ls="--", lw=1, label=f"fc = {fc/1e3:.3g} kHz")
    ax1.set_ylabel("幅值 (dB)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(f, phase, color="#2b6cb0", lw=2)
    ax2.axvline(fc, color="#e53e3e", ls="--", lw=1)
    ax2.set_xlabel("频率 (Hz)")
    ax2.set_ylabel("相位 (deg)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_rc_bode(R: float, C: float, kind: str = "lowpass") -> plt.Figure:
    """绘制 RC 滤波器 Bode 图。"""
    fc = rc_filter_cutoff(R, C)["fc"]
    label = "低通滤波器" if kind == "lowpass" else "高通滤波器"
    return _plot_first_order_bode(fc, kind, f"RC {label} Bode 图  (R={R:.3g}Ω, C={C:.3g}F)")


def plot_rl_bode(R: float, L: float, kind: str = "lowpass") -> plt.Figure:
    """绘制 RL 滤波器 Bode 图。"""
    fc = R / (2.0 * math.pi * L)
    label = "低通滤波器" if kind == "lowpass" else "高通滤波器"
    return _plot_first_order_bode(fc, kind, f"RL {label} Bode 图  (R={R:.3g}Ω, L={L:.3g}H)")


def rlc_series_params(R: float, L: float, C: float) -> dict:
    """RLC 串联谐振关键参数。

    R: 电阻(Ω)，L: 电感(H)，C: 电容(F)。
    返回谐振频率 f0、品质因数 Q、带宽 BW、阻尼比 ζ。
    """
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
    Q = (1.0 / R) * math.sqrt(L / C)
    BW = f0 / Q if Q > 0 else float("inf")
    BW_alt = R / (2.0 * math.pi * L)  # 带宽另一等价公式：BW = R/(2πL)，用于交叉验证
    zeta = (R / 2.0) * math.sqrt(C / L)  # 阻尼比
    return {"f0": f0, "f0_khz": f0 / 1e3, "Q": Q, "BW": BW, "BW_hz": BW,
            "BW_alt": BW_alt, "zeta": zeta}


def plot_rlc_response(R: float, L: float, C: float, V: float = 1.0) -> plt.Figure:
    """绘制 RLC 串联电路电流幅频响应（谐振曲线）。"""
    p = rlc_series_params(R, L, C)
    f0 = p["f0"]
    f = np.logspace(np.log10(max(f0 / 20, 1)), np.log10(f0 * 20), 600)
    w = 2 * math.pi * f
    # 串联 RLC 电流：|I| = V / sqrt(R^2 + (wL - 1/(wC))^2)
    z = np.sqrt(R**2 + (w * L - 1.0 / (w * C)) ** 2)
    I = V / z

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f, I * 1000, color="#2b6cb0", lw=2)
    ax.axvline(f0, color="#e53e3e", ls="--", lw=1, label=f"f0 = {f0/1e3:.3g} kHz")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("电流 (mA)")
    ax.set_title(f"RLC 串联谐振曲线  (R={R:.3g}Ω, L={L:.3g}H, C={C:.3g}F)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def format_rc_result(R: float, C: float, kind: str) -> dict:
    """整合 RC 滤波器的文本结果，供前端直接展示。"""
    r = rc_filter_cutoff(R, C)
    name = "低通" if kind == "lowpass" else "高通"
    return {
        "电路类型": f"一阶 RC {name}滤波器",
        "截止频率 fc": f"{r['fc']:.4g} Hz（{r['fc_khz']:.4g} kHz）",
        "时间常数 τ": f"{r['tau']:.4g} s",
        "说明": (
            "fc 即 -3dB 截止频率，高于(低通)/低于(高通)该频率的信号被明显衰减。"
            "Bode 图见下方。"
        ),
    }


def format_rlc_result(R: float, L: float, C: float) -> dict:
    """整合 RLC 串联谐振的文本结果。"""
    p = rlc_series_params(R, L, C)
    bw1, bw2 = p["BW"], p["BW_alt"]
    # 交叉验证：BW = f0/Q 与 BW = R/(2πL) 两个独立公式应一致
    rel = abs(bw1 - bw2) / max(abs(bw1), abs(bw2), 1e-300)
    cross = "✅ 两公式一致" if rel < 1e-6 else f"⚠️ 两公式有偏差（相对误差 {rel:.2e}）"
    return {
        "电路类型": "RLC 串联谐振电路",
        "谐振频率 f0": f"{p['f0']:.4g} Hz（{p['f0_khz']:.4g} kHz）",
        "品质因数 Q": f"{p['Q']:.4g}",
        "带宽 BW": f"{p['BW']:.4g} Hz（BW = f0/Q = R/(2πL) = {bw2:.4g} Hz）",
        "阻尼比 ζ": f"{p['zeta']:.4g}",
        "结果自校验": cross,
        "说明": "Q 越大谐振峰越尖锐、选频性越好；ζ<1 为欠阻尼，ζ>1 为过阻尼。",
    }


def rl_filter_cutoff(R: float, L: float) -> dict:
    """RL 滤波器的截止频率与时间常数。"""
    fc = R / (2.0 * math.pi * L)
    return {"fc": fc, "fc_khz": fc / 1e3, "tau": L / R}


def format_rl_result(R: float, L: float, kind: str) -> dict:
    """整合 RL 滤波器的文本结果。"""
    r = rl_filter_cutoff(R, L)
    name = "低通" if kind == "lowpass" else "高通"
    return {
        "电路类型": f"一阶 RL {name}滤波器",
        "截止频率 fc": f"{r['fc']:.4g} Hz（{r['fc_khz']:.4g} kHz）",
        "时间常数 τ": f"{r['tau']:.4g} s（τ = L/R）",
        "说明": "fc = R/(2πL)，Bode 图见下方。",
    }


def format_divider_result(R1: float, R2: float, Vin: float = 1.0) -> dict:
    """整合电阻分压电路的文本结果。"""
    ratio = R2 / (R1 + R2)
    return {
        "电路类型": "电阻分压电路",
        "分压比": f"{ratio:.4g}",
        "输出电压 Vout": f"{Vin * ratio:.4g} V（Vin={Vin} V）",
        "说明": "Vout = Vin × R2/(R1+R2)，常用于电平变换与偏置电压。",
    }


def format_op_amp_result(config: str, Rf: float, Rin: float, Vin: float = 1.0) -> dict:
    """整合运算放大器（反相/同相）的文本结果。"""
    if config == "inverting":
        gain = -Rf / Rin
        name, formula = "反相放大器", "A = -Rf/Rin"
    elif config == "non_inverting":
        gain = 1.0 + Rf / Rin
        name, formula = "同相放大器", "A = 1 + Rf/Rin"
    else:
        raise ValueError(f"config 参数非法：{config!r}（应为 inverting 或 non_inverting）")
    return {
        "电路类型": name,
        "电压增益 A": f"{gain:.4g}（{formula}）",
        "输出电压 Vout": f"{gain * Vin:.4g} V（Vin={Vin} V）",
        "说明": "理想运放：反相放大输入输出反相；同相放大输入输出同相、输入阻抗高。",
    }


# ---------------------------------------------------------------- 时域暂态过程
def rc_transient(R: float, C: float, V0: float = 1.0, periods: int = 5) -> dict:
    """RC 充放电瞬态曲线数据（一阶暂态）。返回时间轴与充/放电电压。"""
    tau = R * C
    t = np.linspace(0, periods * tau, 500)
    v_charge = V0 * (1.0 - np.exp(-t / tau))
    v_discharge = V0 * np.exp(-t / tau)
    return {"tau": tau, "t": t, "v_charge": v_charge, "v_discharge": v_discharge}


def plot_rc_transient(R: float, C: float, V0: float = 1.0) -> plt.Figure:
    """绘制 RC 充/放电曲线（标注 τ 与 63.2% 点）。"""
    d = rc_transient(R, C, V0)
    tau, t = d["tau"], d["t"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t * 1e3, d["v_charge"], color="#2b6cb0", lw=2,
            label="充电 v_c(t)=V0(1−e^(−t/τ))")
    ax.plot(t * 1e3, d["v_discharge"], color="#e53e3e", lw=2, ls="--",
            label="放电 v_c(t)=V0·e^(−t/τ)")
    ax.axhline(V0 * 0.632, color="gray", ls=":", lw=1, label="63.2% V0")
    ax.axvline(tau * 1e3, color="green", ls=":", lw=1, label=f"τ = {tau:.4g} s")
    ax.set_xlabel("时间 t (ms)")
    ax.set_ylabel("电容电压 v_c (V)")
    ax.set_title(f"RC 充放电瞬态曲线  (R={R:.3g}Ω, C={C:.3g}F, V0={V0:.3g}V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def format_rc_transient_result(R: float, C: float, V0: float = 1.0) -> dict:
    """整合 RC 充放电瞬态的文本结果。"""
    tau = R * C
    return {
        "电路类型": "RC 充放电瞬态（一阶暂态过程）",
        "时间常数 τ": f"{tau:.4g} s（τ = RC）",
        "充电到 63.2% 用时": f"{tau:.4g} s（≈1τ）",
        "充电到 ~95% 用时": f"{3 * tau:.4g} s（≈3τ）",
        "基本充满用时": f"{5 * tau:.4g} s（≈5τ）",
        "说明": "τ 是暂态过程的特征时间：每经过一个 τ，剩余差量衰减到原来的 1/e≈36.8%。",
    }


def rlc_step_params(R: float, L: float, C: float, V: float = 1.0) -> dict:
    """RLC 串联电路阶跃响应（电容电压）关键参数。"""
    w0 = 1.0 / math.sqrt(L * C)
    zeta = (R / 2.0) * math.sqrt(C / L)
    if zeta < 1.0:
        case = "欠阻尼"
        wd = w0 * math.sqrt(1.0 - zeta ** 2)
        overshoot = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))
        tp = math.pi / wd
    elif zeta > 1.0:
        case = "过阻尼"
        wd = w0 * math.sqrt(zeta ** 2 - 1.0)
        overshoot, tp = 0.0, 0.0
    else:
        case = "临界阻尼"
        wd = overshoot = tp = 0.0
    return {"w0": w0, "zeta": zeta, "case": case, "wd": wd,
            "overshoot": overshoot, "tp": tp}


def _rlc_step_voltage(t: np.ndarray, V: float, p: dict) -> np.ndarray:
    """RLC 串联阶跃响应下电容电压 v_c(t)（解析解）。"""
    w0, zeta = p["w0"], p["zeta"]
    if zeta < 1.0:
        wd = p["wd"]
        decay = np.exp(-zeta * w0 * t)
        vc = V * (1.0 - decay * (np.cos(wd * t)
                                 + (zeta / math.sqrt(1.0 - zeta ** 2)) * np.sin(wd * t)))
    elif zeta > 1.0:
        a = w0 * math.sqrt(zeta ** 2 - 1.0)
        decay = np.exp(-zeta * w0 * t)
        vc = V * (1.0 - decay * (np.cosh(a * t)
                                 + (zeta / math.sqrt(zeta ** 2 - 1.0)) * np.sinh(a * t)))
    else:
        vc = V * (1.0 - np.exp(-w0 * t) * (1.0 + w0 * t))
    return vc


def plot_rlc_step(R: float, L: float, C: float, V: float = 1.0) -> plt.Figure:
    """绘制 RLC 串联阶跃响应（电容电压），体现欠/过/临界阻尼。"""
    p = rlc_step_params(R, L, C, V)
    w0, zeta = p["w0"], p["zeta"]
    t_max = (2.5 * 2.0 * math.pi) / w0 if zeta < 1.0 else (6.0 * max(zeta, 1.0)) / w0
    t = np.linspace(0, t_max, 600)
    vc = _rlc_step_voltage(t, V, p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t * 1e3, vc, color="#2b6cb0", lw=2)
    ax.axhline(V, color="gray", ls="--", lw=1, label=f"稳态值 V={V:.3g}V")
    ax.set_xlabel("时间 t (ms)")
    ax.set_ylabel("电容电压 v_c (V)")
    ax.set_title(f"RLC 串联阶跃响应（{p['case']}，ζ={zeta:.4g}）  R={R:.3g}Ω, L={L:.3g}H, C={C:.3g}F")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def format_rlc_step_result(R: float, L: float, C: float, V: float = 1.0) -> dict:
    """整合 RLC 串联阶跃响应的文本结果。"""
    p = rlc_step_params(R, L, C, V)
    res = {
        "电路类型": "RLC 串联阶跃响应（二阶暂态过程）",
        "无阻尼自然频率 ω0": f"{p['w0']:.4g} rad/s",
        "阻尼比 ζ": f"{p['zeta']:.4g}",
        "响应类型": p["case"],
    }
    if p["zeta"] < 1.0:
        res["阻尼振荡频率 ωd"] = f"{p['wd']:.4g} rad/s"
        res["峰值时间 tp"] = f"{p['tp']:.4g} s"
        res["最大超调量"] = f"{p['overshoot']:.4g}%"
    res["说明"] = "ζ<1 欠阻尼振荡；ζ=1 临界阻尼（无超调最快到达稳态）；ζ>1 过阻尼（无振荡、响应慢）。"
    return res
