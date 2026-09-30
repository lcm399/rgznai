"""仿真代码生成 —— 根据电路参数生成可运行的仿真代码。

确定性部分：按用户给的参数，直接拼出 Python / LTSpice / Verilog 模板，
保证代码一定可运行（不靠 LLM 现场编）。LLM 负责加解释、按需扩展。
"""
from __future__ import annotations


def generate_python_rc(R: float, C: float, kind: str = "lowpass") -> str:
    """生成 RC 滤波器的 Python 数值仿真（阶跃响应 + 频率响应）。

    注意：模板本身要产出含 f-string 的代码，故对字面量大括号用 {{ }} 转义。
    """
    kind_zh = "低通" if kind == "lowpass" else "高通"
    # 生成期就分支，避免在产出代码里留条件判断
    if kind == "lowpass":
        vout_expr = "1 - np.exp(-t / (R * C))"   # 低通：充电
        H_expr = "1 / np.sqrt(1 + ratio**2)"
    else:
        vout_expr = "np.exp(-t / (R * C))"       # 高通：放电
        H_expr = "ratio / np.sqrt(1 + ratio**2)"

    return f'''# RC {kind_zh}滤波器 数值仿真
# 参数：R={R} Ω, C={C} F
import numpy as np
import matplotlib.pyplot as plt

R, C = {R}, {C}
fc = 1 / (2 * np.pi * R * C)   # 截止频率

# ---- 1. 阶跃响应（时域）----
t = np.linspace(0, 5 * R * C, 500)
vout = {vout_expr}

# ---- 2. 频率响应（频域）----
f = np.logspace(0, 5, 500)
ratio = f / fc
H = {H_expr}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(t * 1e3, vout); ax1.set_title("阶跃响应")
ax1.set_xlabel("t (ms)"); ax1.set_ylabel("vout")
ax2.semilogx(f, 20 * np.log10(H)); ax2.set_title("幅频响应")
ax2.set_xlabel("f (Hz)"); ax2.set_ylabel("|H| (dB)")
ax2.axvline(fc, color="r", ls="--", label=f"fc={{fc:.3g}}Hz")
ax2.legend(); plt.tight_layout(); plt.show()
'''


def generate_ltspice_rc(R: float, C: float, kind: str = "lowpass") -> str:
    """生成 LTSpice 网表（AC 分析）。"""
    return f'''* RC {("低通" if kind == "lowpass" else "高通")}滤波器 LTSpice 网表
* R={R}Ω  C={C}F
V1 in 0 AC 1
R1 in out {R}
C1 out 0 {C}
.ac dec 100 1 100k
.end
'''


def generate_verilog_counter(bits: int = 4) -> str:
    """生成一个可综合的同步计数器（示例：硬件赛道的 Verilog 模板）。"""
    max_val = 2**bits - 1
    return f'''// {bits}位同步加法计数器（可综合）
module counter #(parameter WIDTH = {bits})(
    input  wire clk,
    input  wire rst_n,          // 低有效复位
    output reg  [WIDTH-1:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 0;
        else
            q <= q + 1'b1;      // 计数到 {max_val} 后自动回绕
    end
endmodule
'''


def generate_python_rlc(R: float, L: float, C: float) -> str:
    """生成 RLC 串联谐振电路的 Python 数值仿真（频率扫描显示谐振峰）。"""
    return f'''# RLC 串联谐振 数值仿真
# 参数：R={R} Ω, L={L} H, C={C} F
import numpy as np
import matplotlib.pyplot as plt

R, L, C = {R}, {L}, {C}
f0 = 1 / (2 * np.pi * np.sqrt(L * C))   # 谐振频率
Q = (1 / R) * np.sqrt(L / C)            # 品质因数

# 频率扫描：串联谐振时电流最大
f = np.logspace(np.log10(f0 / 20), np.log10(f0 * 20), 800)
w = 2 * np.pi * f
I = 1 / np.sqrt(R**2 + (w * L - 1 / (w * C))**2)   # |I| = V/Z，令 V=1V

plt.figure(figsize=(7, 4))
plt.semilogx(f, I * 1000)
plt.axvline(f0, color='r', ls='--', label=f'f0 = {{f0:.3g}} Hz')
plt.xlabel('频率 (Hz)'); plt.ylabel('电流 (mA)')
plt.title(f'RLC 串联谐振曲线  (Q = {{Q:.3g}})')
plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout(); plt.show()
'''


def generate_python_rlc_transient(R: float, L: float, C: float) -> str:
    """生成 RLC 串联电路阶跃响应的 Python 数值仿真（解析解，覆盖欠/过/临界阻尼）。"""
    return f'''# RLC 串联阶跃响应 数值仿真（二阶暂态）
# 参数：R={R} Ω, L={L} H, C={C} F
import numpy as np
import matplotlib.pyplot as plt

R, L, C = {R}, {L}, {C}
V = 1.0
w0 = 1 / np.sqrt(L * C)
zeta = (R / 2) * np.sqrt(C / L)         # 阻尼比
print(f"阻尼比 zeta = {{zeta:.3f}}")

t = np.linspace(0, max(15 / w0, 8 * zeta / w0), 800)
if zeta < 1:                            # 欠阻尼（衰减振荡）
    wd = w0 * np.sqrt(1 - zeta**2)
    vc = V * (1 - np.exp(-zeta*w0*t) * (np.cos(wd*t) + zeta/np.sqrt(1-zeta**2)*np.sin(wd*t)))
elif zeta > 1:                          # 过阻尼
    a = w0 * np.sqrt(zeta**2 - 1)
    vc = V * (1 - np.exp(-zeta*w0*t) * (np.cosh(a*t) + zeta/np.sqrt(zeta**2-1)*np.sinh(a*t)))
else:                                   # 临界阻尼
    vc = V * (1 - np.exp(-w0*t) * (1 + w0*t))

plt.figure(figsize=(7, 4))
plt.plot(t * 1e3, vc)
plt.axhline(V, color='gray', ls='--')
plt.xlabel('时间 (ms)'); plt.ylabel('电容电压 v_c (V)')
plt.title(f'RLC 阶跃响应  (zeta = {{zeta:.3f}})')
plt.grid(True); plt.tight_layout(); plt.show()
'''


def generate(code_type: str, **params) -> str:
    """统一入口：按 code_type 分发到具体生成器。"""
    if code_type == "python_rc":
        return generate_python_rc(params["R"], params["C"], params.get("kind", "lowpass"))
    if code_type == "python_rlc":
        return generate_python_rlc(params["R"], params["L"], params["C"])
    if code_type == "python_rlc_transient":
        return generate_python_rlc_transient(params["R"], params["L"], params["C"])
    if code_type == "ltspice_rc":
        return generate_ltspice_rc(params["R"], params["C"], params.get("kind", "lowpass"))
    if code_type == "verilog_counter":
        return generate_verilog_counter(params.get("bits", 4))
    raise ValueError(f"未知代码类型: {code_type}")
