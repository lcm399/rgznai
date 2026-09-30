"""工具注册表：把确定性计算能力包装成 LLM 可调用的工具（Function Calling）。

这是「智能体」的核心：LLM 理解用户意图后，主动决定调用哪个工具、传什么参数，
拿到精确结果后再向用户讲解——而不是代码里按页面写死。对应通知要求展示的
「接收请求 → 分析决策 → 反馈结果」闭环。
"""
from __future__ import annotations

import json

from core import circuit, rag, signal, simulation, units


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    """构造一个 OpenAI 兼容的 function 工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def build_tools() -> list[dict]:
    """返回可供 LLM 调用的工具列表。"""
    return [
        _schema(
            "analyze_rc_filter",
            "分析 RC 低通/高通滤波器，返回截止频率、时间常数等精确结果（用于电路分析）。",
            {
                "R": {"type": "number", "description": "电阻值，单位欧姆(Ω)"},
                "C": {"type": "number", "description": "电容值，单位法拉(F)"},
                "kind": {"type": "string", "enum": ["lowpass", "highpass"], "description": "低通或高通"},
            },
            ["R", "C", "kind"],
        ),
        _schema(
            "analyze_rlc_circuit",
            "分析 RLC 串联谐振电路，返回谐振频率、品质因数、带宽等精确结果。",
            {
                "R": {"type": "number", "description": "电阻(Ω)"},
                "L": {"type": "number", "description": "电感(H)"},
                "C": {"type": "number", "description": "电容(F)"},
            },
            ["R", "L", "C"],
        ),
        _schema(
            "analyze_signal",
            "分析一段简短的采样序列，返回统计量（均值/峰值/RMS）与主频、异常点。",
            {
                "data": {"type": "string", "description": "采样值，逗号或空格分隔，如 '0,0.5,0.87,1,0.87,0.5,0'"},
                "fs": {"type": "number", "description": "采样率(Hz)，未知可省略"},
            },
            ["data"],
        ),
        _schema(
            "generate_simulation_code",
            "生成可运行的仿真代码（Python 数值仿真 / LTSpice 网表 / Verilog 计数器）。",
            {
                "code_type": {"type": "string",
                              "enum": ["python_rc", "python_rlc", "python_rlc_transient",
                                       "ltspice_rc", "verilog_counter"],
                              "description": "代码类型"},
                "R": {"type": "number", "description": "电阻(Ω)，RC 电路用"},
                "L": {"type": "number", "description": "电感(H)，RLC 电路用"},
                "C": {"type": "number", "description": "电容(F)，RC/RLC 电路用"},
                "kind": {"type": "string", "enum": ["lowpass", "highpass"], "description": "滤波器类型"},
                "bits": {"type": "integer", "description": "计数器位宽，Verilog 用"},
            },
            ["code_type"],
        ),
        _schema(
            "search_knowledge",
            "从电子信息教材/实验手册知识库中检索与问题最相关的资料片段，用于专业答疑。",
            {"query": {"type": "string", "description": "检索关键词或问题"}},
            ["query"],
        ),
        _schema(
            "analyze_voltage_divider",
            "分析电阻分压电路，返回分压比与输出电压。",
            {
                "R1": {"type": "number", "description": "上端电阻(Ω)"},
                "R2": {"type": "number", "description": "下端电阻(Ω)"},
                "Vin": {"type": "number", "description": "输入电压(V)，默认 1"},
            },
            ["R1", "R2"],
        ),
        _schema(
            "analyze_rl_filter",
            "分析 RL 低通/高通滤波器，返回截止频率与时间常数。",
            {
                "R": {"type": "number", "description": "电阻(Ω)"},
                "L": {"type": "number", "description": "电感(H)"},
                "kind": {"type": "string", "enum": ["lowpass", "highpass"], "description": "低通或高通"},
            },
            ["R", "L", "kind"],
        ),
        _schema(
            "analyze_op_amp",
            "分析运算放大器（反相/同相放大），返回电压增益与输出电压。",
            {
                "config": {"type": "string", "enum": ["inverting", "non_inverting"], "description": "反相或同相"},
                "Rf": {"type": "number", "description": "反馈电阻(Ω)"},
                "Rin": {"type": "number", "description": "输入电阻(Ω)"},
                "Vin": {"type": "number", "description": "输入电压(V)，默认 1"},
            },
            ["config", "Rf", "Rin"],
        ),
        _schema(
            "simulate_rc_transient",
            "计算 RC 充放电瞬态过程（一阶暂态），返回时间常数 τ 及充/放电关键时间。",
            {
                "R": {"type": "number", "description": "电阻(Ω)"},
                "C": {"type": "number", "description": "电容(F)"},
                "V0": {"type": "number", "description": "电源/初始电压(V)，默认 5"},
            },
            ["R", "C"],
        ),
        _schema(
            "simulate_rlc_transient",
            "分析 RLC 串联电路阶跃响应（二阶暂态），返回阻尼比、响应类型、超调量等。",
            {
                "R": {"type": "number", "description": "电阻(Ω)"},
                "L": {"type": "number", "description": "电感(H)"},
                "C": {"type": "number", "description": "电容(F)"},
                "V": {"type": "number", "description": "阶跃电压幅度(V)，默认 1"},
            },
            ["R", "L", "C"],
        ),
    ]


def _kind(v) -> str:
    """解析并校验滤波器类型，返回 'lowpass' / 'highpass'。"""
    if v is None:
        return "lowpass"
    s = str(v).strip().lower()
    if s in ("lowpass", "低通", "lp"):
        return "lowpass"
    if s in ("highpass", "高通", "hp"):
        return "highpass"
    raise ValueError(f"kind 参数非法：{v!r}（应为 lowpass 或 highpass）")


def _config(v) -> str:
    """解析并校验运放拓扑，返回 'inverting' / 'non_inverting'。"""
    if v is None:
        return "inverting"
    s = str(v).strip().lower()
    if s in ("inverting", "反相", "invert", "inv"):
        return "inverting"
    if s in ("non_inverting", "non-inverting", "同相", "noninverting"):
        return "non_inverting"
    raise ValueError(f"config 参数非法：{v!r}（应为 inverting 或 non_inverting）")


def _generate_code(args: dict) -> dict:
    """按 code_type 解析并校验参数，生成仿真代码。"""
    ct = args.get("code_type", "python_rc")
    if ct in ("python_rc", "ltspice_rc"):
        params = {
            "R": units.resistance(args.get("R", 1000), "R", positive=True),
            "C": units.capacitance(args.get("C", 1e-6), "C", positive=True),
            "kind": _kind(args.get("kind")),
        }
    elif ct in ("python_rlc", "python_rlc_transient"):
        params = {
            "R": units.resistance(args.get("R", 100), "R", positive=True),
            "L": units.inductance(args.get("L", 1e-3), "L", positive=True),
            "C": units.capacitance(args.get("C", 1e-6), "C", positive=True),
        }
    elif ct == "verilog_counter":
        bits = int(float(args.get("bits", 4)))
        if not 1 <= bits <= 64:
            raise ValueError(f"参数 bits={bits} 超出范围（应为 1~64 的整数）")
        params = {"bits": bits}
    else:
        raise ValueError(f"code_type 参数非法：{ct!r}")
    return {"code_type": ct, "code": simulation.generate(ct, **params)}


def execute(name: str, args: dict) -> str:
    """执行工具，返回 JSON 字符串（作为 tool 消息内容回传给 LLM）。

    所有参数先经 units 统一解析（支持 "1kΩ" 等带单位输入）并校验，
    非法参数返回清晰的中文错误，而不是让程序崩溃。
    """
    args = args or {}
    try:
        if name == "analyze_rc_filter":
            R = units.resistance(args.get("R"), "R", positive=True)
            C = units.capacitance(args.get("C"), "C", positive=True)
            res = circuit.format_rc_result(R, C, _kind(args.get("kind")))
        elif name == "analyze_rl_filter":
            R = units.resistance(args.get("R"), "R", positive=True)
            L = units.inductance(args.get("L"), "L", positive=True)
            res = circuit.format_rl_result(R, L, _kind(args.get("kind")))
        elif name == "analyze_rlc_circuit":
            R = units.resistance(args.get("R"), "R", positive=True)
            L = units.inductance(args.get("L"), "L", positive=True)
            C = units.capacitance(args.get("C"), "C", positive=True)
            res = circuit.format_rlc_result(R, L, C)
        elif name == "analyze_voltage_divider":
            R1 = units.resistance(args.get("R1"), "R1", positive=True)
            R2 = units.resistance(args.get("R2"), "R2", positive=True)
            Vin = units.voltage(args.get("Vin", 1.0), "Vin")
            res = circuit.format_divider_result(R1, R2, Vin)
        elif name == "analyze_op_amp":
            config = _config(args.get("config"))
            Rf = units.resistance(args.get("Rf"), "Rf", non_negative=True)
            Rin = units.resistance(args.get("Rin"), "Rin", positive=True)
            Vin = units.voltage(args.get("Vin", 1.0), "Vin")
            res = circuit.format_op_amp_result(config, Rf, Rin, Vin)
        elif name == "simulate_rc_transient":
            R = units.resistance(args.get("R"), "R", positive=True)
            C = units.capacitance(args.get("C"), "C", positive=True)
            V0 = units.voltage(args.get("V0", 5.0), "V0")
            res = circuit.format_rc_transient_result(R, C, V0)
        elif name == "simulate_rlc_transient":
            R = units.resistance(args.get("R"), "R", positive=True)
            L = units.inductance(args.get("L"), "L", positive=True)
            C = units.capacitance(args.get("C"), "C", positive=True)
            V = units.voltage(args.get("V", 1.0), "V")
            res = circuit.format_rlc_step_result(R, L, C, V)
        elif name == "analyze_signal":
            data = args.get("data")
            if not data:
                raise ValueError("参数 data 缺失（请提供采样序列）")
            t, y = signal.parse_data(str(data))
            fs = args.get("fs")
            res = signal.analyze(y, fs=units.frequency(fs, "fs") if fs else None)
        elif name == "generate_simulation_code":
            res = _generate_code(args)
        elif name == "search_knowledge":
            ctx = rag.build_context(str(args.get("query", "")))
            res = {"knowledge": ctx or "未检索到相关资料"}
        else:
            raise ValueError(f"未知工具: {name}")
        return json.dumps(res, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": f"参数错误：{e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"工具执行失败: {e}"}, ensure_ascii=False)


def render(name: str, args: dict):
    """返回该工具对应的 matplotlib 图（无图的工具返回 None）。

    用于在智能体对话里直接展示图形——体现「AI 不仅会算、还会画」。
    参数解析与 execute 保持一致（同样支持带单位输入）。
    """
    args = args or {}
    try:
        if name == "analyze_rc_filter":
            return circuit.plot_rc_bode(
                units.resistance(args.get("R"), positive=True),
                units.capacitance(args.get("C"), positive=True), _kind(args.get("kind")))
        if name == "analyze_rl_filter":
            return circuit.plot_rl_bode(
                units.resistance(args.get("R"), positive=True),
                units.inductance(args.get("L"), positive=True), _kind(args.get("kind")))
        if name == "analyze_rlc_circuit":
            return circuit.plot_rlc_response(
                units.resistance(args.get("R"), positive=True),
                units.inductance(args.get("L"), positive=True),
                units.capacitance(args.get("C"), positive=True))
        if name == "simulate_rc_transient":
            return circuit.plot_rc_transient(
                units.resistance(args.get("R"), positive=True),
                units.capacitance(args.get("C"), positive=True),
                units.voltage(args.get("V0", 5.0)))
        if name == "simulate_rlc_transient":
            return circuit.plot_rlc_step(
                units.resistance(args.get("R"), positive=True),
                units.inductance(args.get("L"), positive=True),
                units.capacitance(args.get("C"), positive=True),
                units.voltage(args.get("V", 1.0)))
        if name == "analyze_signal":
            t, y = signal.parse_data(str(args.get("data")))
            fs = args.get("fs")
            return signal.plot_signal(t, y, fs=units.frequency(fs) if fs else None)
    except Exception:
        return None
    return None
