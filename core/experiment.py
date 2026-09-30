"""实验工作台编排层。

把已有确定性计算能力组织成「实验任务闭环」：
实验识别 → 参数解析 → 理论 → 确定性计算 → 图形 → 结果验证 → 结论 → 记录 → 报告。

原则（与核心模块一致）：
  - 不重新实现计算，只复用 core.tools / core.units 的既有函数；
  - 数值全部来自确定性工具，AI 仅做解释，且与计算结果严格分层；
  - 报告 / 记录只写入真实数据，缺失项明确写「未进行该项测试」，绝不伪造。
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime

from core import tools, units

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECORDS_DIR = os.path.join(ROOT_DIR, "records")


# ---------------------------------------------------------------------------
# 实验定义（元数据 + 参数 schema + 理论/思考题）
# ---------------------------------------------------------------------------
def _text(key, label, unit, default, category, positive=True, non_negative=False):
    return {"key": key, "label": label, "unit": unit, "default": default,
            "type": "text", "category": category,
            "positive": positive, "non_negative": non_negative}


def _select(key, label, options):
    return {"key": key, "label": label, "type": "select", "options": options}


EXPERIMENTS = [
    {
        "id": "rc_filter",
        "name": "RC 滤波实验",
        "goal": "测量 RC 滤波器的截止频率与时间常数，理解一阶滤波器的频率响应特性。",
        "tool": "analyze_rc_filter",
        "has_figure": True,
        "figure_desc": "Bode 图（幅频 + 相频）",
        "cross_check": False,
        "key_fields": ["截止频率 fc", "时间常数 τ"],
        "formula": ["fc = 1/(2πRC)", "τ = RC"],
        "theory": [
            "RC 一阶滤波器由电阻 R 与电容 C 组成，输出取自电容两端为低通、取电阻两端为高通。",
            "截止频率 fc 是幅值下降 -3dB 对应的频率，高于（低通）/低于（高通）该频率的信号被明显衰减。",
            "时间常数 τ 决定暂态响应的快慢。",
        ],
        "thinking": [
            "若 R 增大一倍而 C 不变，fc 如何变化？",
            "把 R 与 C 位置互换，滤波器类型会如何改变？",
        ],
        "params": [
            _text("R", "电阻 R", "Ω", "1kΩ", "resistance"),
            _text("C", "电容 C", "F", "1μF", "capacitance"),
            _select("kind", "滤波器类型", [("低通", "lowpass"), ("高通", "highpass")]),
        ],
    },
    {
        "id": "rl_filter",
        "name": "RL 滤波实验",
        "goal": "测量 RL 滤波器的截止频率，理解电感在高频/低频下的阻抗特性。",
        "tool": "analyze_rl_filter",
        "has_figure": True,
        "figure_desc": "Bode 图（幅频 + 相频）",
        "cross_check": False,
        "key_fields": ["截止频率 fc", "时间常数 τ"],
        "formula": ["fc = R/(2πL)", "τ = L/R"],
        "theory": [
            "RL 滤波器由电阻 R 与电感 L 组成，截止频率 fc = R/(2πL)。",
            "电感对高频呈现高阻抗、对低频呈现低阻抗，特性与 RC 电路对偶。",
        ],
        "thinking": ["电感与电容在滤波电路中的角色有何对偶关系？"],
        "params": [
            _text("R", "电阻 R", "Ω", "1kΩ", "resistance"),
            _text("L", "电感 L", "H", "1mH", "inductance"),
            _select("kind", "滤波器类型", [("低通", "lowpass"), ("高通", "highpass")]),
        ],
    },
    {
        "id": "rlc_resonance",
        "name": "RLC 串联谐振实验",
        "goal": "测量 RLC 串联电路的谐振频率、品质因数与带宽，并通过两条独立公式交叉验证带宽。",
        "tool": "analyze_rlc_circuit",
        "has_figure": True,
        "figure_desc": "谐振曲线（电流幅频响应）",
        "cross_check": True,
        "key_fields": ["谐振频率 f0", "品质因数 Q", "带宽 BW"],
        "formula": ["f0 = 1/(2π√(LC))", "Q = (1/R)√(L/C)", "BW = f0/Q = R/(2πL)"],
        "theory": [
            "RLC 串联谐振时感抗与容抗大小相等、相位相反而相互抵消，回路阻抗最小、电流最大。",
            "带宽 BW 用两条独立公式互查：BW = f0/Q 与 BW = R/(2πL)，一致则结果可信。",
        ],
        "thinking": ["Q 越大，谐振峰越尖锐，选频性如何变化？"],
        "params": [
            _text("R", "电阻 R", "Ω", "10Ω", "resistance"),
            _text("L", "电感 L", "H", "10mH", "inductance"),
            _text("C", "电容 C", "F", "100μF", "capacitance"),
        ],
    },
    {
        "id": "rlc_transient",
        "name": "RLC 暂态响应实验",
        "goal": "分析 RLC 串联电路的二阶阶跃响应，计算阻尼比并判别欠/临界/过阻尼，测量超调量。",
        "tool": "simulate_rlc_transient",
        "has_figure": True,
        "figure_desc": "暂态响应曲线（标注阻尼类型）",
        "cross_check": False,
        "key_fields": ["无阻尼自然频率 ω0", "阻尼比 ζ", "响应类型", "最大超调量"],
        "formula": ["ω0 = 1/√(LC)", "ζ = (R/2)√(C/L)", "ωd = ω0√(1-ζ²)", "σ = e^(-πζ/√(1-ζ²))×100%"],
        "theory": [
            "二阶 RLC 电路的阶跃响应由阻尼比 ζ 决定：ζ<1 欠阻尼（衰减振荡、有超调），"
            "ζ=1 临界阻尼（无超调、最快到达稳态），ζ>1 过阻尼（无振荡、响应慢）。",
        ],
        "thinking": ["为什么欠阻尼会出现超调与振荡？", "若想消除振荡，应如何调整 R？"],
        "params": [
            _text("R", "电阻 R", "Ω", "10Ω", "resistance"),
            _text("L", "电感 L", "H", "10mH", "inductance"),
            _text("C", "电容 C", "F", "100μF", "capacitance"),
            _text("V", "阶跃电压 V", "V", "1V", "voltage"),
        ],
    },
    {
        "id": "op_amp",
        "name": "运算放大器实验",
        "goal": "测量反相/同相放大器的电压增益与输出电压，理解虚短虚断。",
        "tool": "analyze_op_amp",
        "has_figure": False,
        "figure_desc": None,
        "cross_check": False,
        "key_fields": ["电压增益 A", "输出电压 Vout"],
        "formula": ["反相：A = -Rf/Rin", "同相：A = 1 + Rf/Rin", "Vout = A · Vin"],
        "theory": [
            "理想运放两输入端虚短（电压相等）、虚断（无电流流入），由此推导增益。",
            "反相放大输入输出反相；同相放大输入输出同相、输入阻抗高。",
        ],
        "thinking": ["若 Rf=0，反相放大器退化成什么？"],
        "params": [
            _select("config", "电路结构", [("反相放大器", "inverting"), ("同相放大器", "non_inverting")]),
            _text("Rf", "反馈电阻 Rf", "Ω", "10kΩ", "resistance", positive=False, non_negative=True),
            _text("Rin", "输入电阻 Rin", "Ω", "1kΩ", "resistance"),
            _text("Vin", "输入电压 Vin", "V", "1V", "voltage"),
        ],
    },
    {
        "id": "voltage_divider",
        "name": "电阻分压实验",
        "goal": "测量电阻分压电路的输出电压与分压比，验证基尔霍夫电压定律。",
        "tool": "analyze_voltage_divider",
        "has_figure": False,
        "figure_desc": None,
        "cross_check": False,
        "key_fields": ["输出电压 Vout"],
        "formula": ["Vout = Vin · R2/(R1+R2)"],
        "theory": ["电阻串联分压，输出电压与 R2 成正比。", "当 R1=R2 时分压比为 1/2。"],
        "thinking": ["若 R2 ≫ R1，Vout 趋近于什么？"],
        "params": [
            _text("R1", "电阻 R1", "Ω", "1kΩ", "resistance"),
            _text("R2", "电阻 R2", "Ω", "1kΩ", "resistance"),
            _text("Vin", "输入电压 Vin", "V", "5V", "voltage"),
        ],
    },
    {
        "id": "rc_transient",
        "name": "RC 充放电瞬态实验",
        "goal": "测量 RC 电路的充放电时间常数，理解指数充放电规律。",
        "tool": "simulate_rc_transient",
        "has_figure": True,
        "figure_desc": "充放电曲线（标注 τ 与 63.2% 点）",
        "cross_check": False,
        "key_fields": ["时间常数 τ"],
        "formula": ["τ = RC", "v(t) = V0(1 - e^(-t/τ))"],
        "theory": ["经过一个 τ，剩余差量衰减到 1/e≈36.8%；约 5τ 后基本达到稳态。"],
        "thinking": ["如何从实验曲线上读出时间常数 τ？"],
        "params": [
            _text("R", "电阻 R", "Ω", "1kΩ", "resistance"),
            _text("C", "电容 C", "F", "1μF", "capacitance"),
            _text("V0", "电源/初始电压 V0", "V", "5V", "voltage"),
        ],
    },
    {
        "id": "signal",
        "name": "信号分析实验",
        "goal": "对采样信号做统计、FFT 频谱、波形识别与 THD 分析，判断信号特征。",
        "tool": "analyze_signal",
        "has_figure": True,
        "figure_desc": "时域波形 + FFT 幅度谱",
        "cross_check": False,
        "key_fields": ["主频", "有效值 RMS", "波形类型(粗略)"],
        "formula": ["RMS = √(Σx²/N)", "峰值因数 = peak/RMS"],
        "theory": [
            "时域描述信号随时间的变化，频域描述信号的频率成分，FFT 是两者桥梁。",
            "峰值因数启发式：方波 <1.15 < 正弦 <1.65 < 三角/脉冲，可粗略判别波形类型。",
        ],
        "thinking": ["若采样率低于信号频率两倍，会出现什么现象？"],
        "params": [
            {"key": "data", "label": "采样数据", "type": "textarea", "default": ""},
            {"key": "fs", "label": "采样率 fs", "unit": "Hz", "default": "10k", "type": "text",
             "category": "frequency", "positive": True},
        ],
    },
]


def get_experiment(exp_id: str) -> dict:
    for e in EXPERIMENTS:
        if e["id"] == exp_id:
            return e
    raise KeyError(f"未知实验: {exp_id}")


def _parse_param(p, raw):
    """解析单个参数，返回 (si_value, error_or_none)。select/textarea 不做单位解析。"""
    typ = p.get("type", "text")
    if typ == "select":
        return raw, None
    if typ == "textarea":
        return None, None
    cat = p.get("category", "resistance")
    try:
        val = units.require(raw, cat, p.get("label", p["key"]),
                            positive=p.get("positive", True),
                            non_negative=p.get("non_negative", False))
        return val, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def run_experiment(exp_id: str, params: dict) -> dict:
    """执行一次实验，返回结构化结果（含图、验证清单、结论、交叉验证）。"""
    exp = get_experiment(exp_id)

    parsed, parse_errors, param_errors = {}, [], {}
    for p in exp["params"]:
        val, err = _parse_param(p, params.get(p["key"]))
        parsed[p["key"]] = val
        if err:
            parse_errors.append(err)
            param_errors[p["key"]] = err

    # 确定性计算（复用 tools.execute：单位解析 + 公式计算 + 参数校验）
    result = {}
    compute_ok = False
    try:
        result = json.loads(tools.execute(exp["tool"], dict(params)))
        compute_ok = "error" not in result
    except Exception:  # noqa: BLE001
        compute_ok = False

    # 图形（复用 tools.render；失败不影响数值结果）
    figure, fig_ok = None, False
    if exp.get("has_figure") and compute_ok:
        try:
            figure = tools.render(exp["tool"], dict(params))
            fig_ok = figure is not None
        except Exception:  # noqa: BLE001
            fig_ok = False

    # 交叉验证（复用 format_rlc_result 已返回的「结果自校验」）
    cross_check, cross_ok = None, None
    if exp.get("cross_check"):
        cross_check = result.get("结果自校验")
        cross_ok = bool(cross_check and "一致" in str(cross_check))

    verification = _build_verification(exp, parse_errors, compute_ok, fig_ok, cross_check, cross_ok)
    conclusion = _build_conclusion(exp, result, compute_ok, fig_ok, cross_ok)

    return {
        "exp": exp,
        "exp_id": exp_id,
        "params": dict(params),
        "parsed": parsed,
        "parse_errors": parse_errors,
        "param_errors": param_errors,
        "result": result,
        "compute_ok": compute_ok,
        "figure": figure,
        "fig_ok": fig_ok,
        "cross_check": cross_check,
        "cross_ok": cross_ok,
        "verification": verification,
        "conclusion": conclusion,
    }


def _build_verification(exp, parse_errors, compute_ok, fig_ok, cross_check, cross_ok):
    items = []
    ok = not parse_errors
    items.append(("参数有效", "ok" if ok else "fail",
                  "全部参数解析为有效数值" if ok else "；".join(parse_errors)))
    items.append(("单位解析", "ok" if ok else "fail",
                  "工程单位（kΩ/μF/mH 等）已换算为 SI 单位" if ok else "单位解析失败"))
    items.append(("公式计算", "ok" if compute_ok else "fail",
                  "确定性公式计算完成" if compute_ok else "计算未完成"))
    if exp.get("has_figure"):
        items.append(("图形生成", "ok" if fig_ok else "warn",
                      exp.get("figure_desc") if fig_ok else "图形未生成（数值结果仍有效）"))
    if exp.get("cross_check"):
        items.append(("交叉验证", "ok" if cross_ok else "warn",
                      str(cross_check) if cross_check else "未获得交叉验证结果"))
    return items


def _build_conclusion(exp, result, compute_ok, fig_ok, cross_ok):
    if not compute_ok:
        err = result.get("error", "未知错误")
        return f"本次{exp['name']}未完成：计算失败（{err}）。请检查参数后重试。"
    parts = [f"本次{exp['name']}已完成确定性计算"]
    kv = [f"{k}={result[k]}" for k in exp.get("key_fields", []) if k in result]
    if kv:
        parts.append("，关键结果：" + "，".join(kv))
    checks = ["公式计算完成"]
    if fig_ok:
        checks.append("图形生成成功")
    if exp.get("cross_check"):
        checks.append("双公式交叉验证一致" if cross_ok else "交叉验证未通过")
    parts.append("；" + "、".join(checks) + "。")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 实验报告 / 实验记录
# ---------------------------------------------------------------------------
def generate_report(record: dict) -> str:
    """从真实实验数据生成 Markdown 报告。缺失项写「未进行该项测试」。"""
    exp = get_experiment(record["exp_id"])
    L = []
    L.append(f"# {exp['name']}")
    L.append("")
    L.append(f"> 生成时间：{record.get('time', '')}｜本报告由实际实验数据自动生成，不含虚构数据。")
    L.append("")
    L.append("## 一、实验目的")
    L.append("")
    L.append(exp["goal"])
    L.append("")
    L.append("## 二、实验原理")
    L.append("")
    for f in exp["formula"]:
        L.append(f"- `{f}`")
    L.append("")
    for t in exp["theory"]:
        L.append(f"- {t}")
    L.append("")
    L.append("## 三、实验参数")
    L.append("")
    L.append("| 参数 | 输入 | 解析值(SI) |")
    L.append("|------|------|-----------|")
    for p in exp["params"]:
        raw = record["params"].get(p["key"], "")
        parsed = record["parsed"].get(p["key"])
        if p.get("type") == "select":
            txt = str(raw)
        elif p.get("type") == "textarea":
            txt = "（采样数据，见实验）"
        elif parsed is None:
            txt = "—"
        else:
            txt = f"{parsed:g}"
        L.append(f"| {p['label']} | {raw} | {txt} |")
    L.append("")
    L.append("## 四、理论计算")
    L.append("")
    L.append("（由确定性公式计算，见下方结果；关键公式已列于「实验原理」。）")
    L.append("")
    L.append("## 五、实验结果")
    L.append("")
    if record.get("compute_ok"):
        for k, v in record["result"].items():
            if k == "说明":
                continue
            L.append(f"- **{k}**：{v}")
    else:
        L.append(f"未获得计算结果（{record['result'].get('error', '计算失败')}）。")
    L.append("")
    L.append("## 六、结果图形")
    L.append("")
    if record.get("figure_file"):
        L.append(f"![{exp.get('figure_desc', '结果图形')}]({record['figure_file']})")
    elif record.get("fig_ok"):
        L.append(f"✅ 图形已生成（{exp.get('figure_desc', '')}），见实验工作台页面。")
    else:
        L.append("未进行该项测试（图形未生成或该实验无图形）。")
    L.append("")
    L.append("## 七、结果验证")
    L.append("")
    for label, status, note in record.get("verification", []):
        mark = {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(status, "—")
        L.append(f"- {mark} {label}：{note}")
    L.append("")
    L.append("## 八、结果分析")
    L.append("")
    L.append(record.get("ai_analysis") or "未进行该项测试（未生成 AI 分析）。")
    L.append("")
    L.append("## 九、实验结论")
    L.append("")
    L.append(record.get("conclusion", "未生成结论。"))
    L.append("")
    L.append("## 十、思考题")
    L.append("")
    for i, q in enumerate(exp.get("thinking", []), 1):
        L.append(f"{i}. {q}")
    L.append("")
    return "\n".join(L)


def save_record(record: dict, figure=None) -> str:
    """把实验记录保存为本地 JSON（不写数据库、不含 Key/完整 Prompt）。返回文件路径。"""
    os.makedirs(RECORDS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_rel = None
    if figure is not None:
        fig_dir = os.path.join(RECORDS_DIR, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_rel = os.path.join("figures", f"{record['exp_id']}_{ts}.png")
        try:
            figure.savefig(os.path.join(RECORDS_DIR, fig_rel), dpi=120, bbox_inches="tight")
        except Exception:  # noqa: BLE001
            fig_rel = None
    record = dict(record)
    record.pop("figure", None)  # matplotlib 对象不入 JSON
    record["figure_file"] = fig_rel
    record["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fn = os.path.join(RECORDS_DIR, f"{record['exp_id']}_{ts}.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return fn


def list_records() -> list:
    """列出已保存的实验记录（按时间倒序）。"""
    if not os.path.isdir(RECORDS_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(RECORDS_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(RECORDS_DIR, fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:  # noqa: BLE001
            continue
    return out


def system_checks() -> list:
    """对系统各子系统做真实自检，返回 [(名称, 状态, 说明)]，供首页状态总览展示。"""
    checks = []
    def add(name, fn):
        try:
            ok, note = fn()
            checks.append((name, "ok" if ok else "fail", note))
        except Exception as e:  # noqa: BLE001
            checks.append((name, "fail", f"自检异常：{e}"))

    from core import circuit, rag, agent  # noqa: F401  (局部导入避免循环)

    add("核心计算", lambda: ("截止频率 fc" in circuit.format_rc_result(1000.0, 1e-6, "lowpass"), "RC 公式计算正常"))
    add("单位解析", lambda: (units.resistance("1kΩ") == 1000.0, "1kΩ → 1000 Ω"))
    add("交叉验证", lambda: ("一致" in circuit.format_rlc_result(10.0, 1e-2, 1e-4).get("结果自校验", ""), "RLC 带宽双公式一致"))
    add("RAG", lambda: (len(rag.load_knowledge()) > 0, f"{len(rag.load_knowledge())} 个知识片段"))
    add("Agent", lambda: (callable(getattr(agent, "run_stream", None)), "Function Calling 主循环就绪"))
    add("实验编排", lambda: (callable(run_experiment) and len(EXPERIMENTS) == 8,
                            f"{len(EXPERIMENTS)} 个实验 + 运行/报告/记录就绪"))

    try:
        import tests.test_core as tc
        n = len([x for x in dir(tc) if x.startswith("test_")])
        checks.append(("测试", "ok" if n >= 24 else "warn", f"{n} 项离线测试"))
    except Exception:  # noqa: BLE001
        checks.append(("测试", "warn", "无法读取测试数量"))

    add("安全", lambda: _api_key_scan())
    return checks


def _api_key_scan():
    """扫描源码/配置，确认无真实 API Key 泄漏（.env 已 gitignore，不扫描）。"""
    key_re = re.compile(r"sk-[A-Za-z0-9]{16,}")
    for pat in ["app.py", "run_tests.py", "core/*.py", "tests/*.py", "requirements.txt"]:
        for fn in glob.glob(os.path.join(ROOT_DIR, pat)):
            try:
                with open(fn, encoding="utf-8") as f:
                    if key_re.search(f.read()):
                        return False, f"{os.path.basename(fn)} 疑似含真实 Key"
            except Exception:  # noqa: BLE001
                continue
    return True, "源码无真实 API Key（.env 已 gitignore）"
