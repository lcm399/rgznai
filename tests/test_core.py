"""确定性计算核心的回归测试（无需 LLM / 网络，可离线运行）。

运行方式：
    python run_tests.py
或  python -m pytest tests/   （若安装了 pytest）

覆盖：单位解析、工具执行（含非法参数）、电路计算与交叉验证、仿真代码生成、
信号分析、RAG 知识库。所有断言基于精确公式，不依赖任何外部服务。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import circuit, experiment, rag, signal, simulation, tools, units  # noqa: E402


# ---------------------------------------------------------------- 单位解析
def test_units():
    # 基本单位
    assert units.resistance("1kΩ", positive=True) == 1000.0
    assert units.resistance("1K", positive=True) == 1000.0
    assert units.capacitance("1μF", positive=True) == 1e-6
    assert units.capacitance("1uF", positive=True) == 1e-6
    assert units.inductance("10mH", positive=True) == 0.01
    assert units.voltage("5V") == 5.0
    assert units.frequency("1kHz") == 1000.0
    # 纯数字原样返回
    assert units.resistance(1000) == 1000.0
    assert units.capacitance(0.001) == 0.001
    # 大小写敏感：m=毫 vs M=兆
    assert units.resistance("1mΩ") == 1e-3
    assert units.resistance("1MΩ") == 1e6
    # 科学计数法
    assert units.capacitance("1e-6") == 1e-6
    # 非法输入
    for bad, cat, kw in [
        ("", "resistance", {}),
        ("abc", "resistance", {}),
        ("1xyz", "resistance", {}),
        (None, "resistance", {}),
    ]:
        try:
            units.require(bad, cat, "X", **kw)
            raise AssertionError(f"应抛错：{bad!r}")
        except ValueError:
            pass
    # 非正校验
    for bad in (0, -1, "0Ω"):
        try:
            units.resistance(bad, positive=True)
            raise AssertionError(f"应抛错：{bad!r}")
        except ValueError:
            pass


# ---------------------------------------------------------------- 工具执行
def test_tools_rc_with_units():
    r = json.loads(tools.execute("analyze_rc_filter", {"R": "1kΩ", "C": "1μF", "kind": "低通"}))
    assert "error" not in r
    assert "159.2" in r["截止频率 fc"]  # fc = 1/(2π·1000·1e-6) ≈ 159.15 Hz


def test_tools_op_amp():
    r = json.loads(tools.execute("analyze_op_amp",
                                 {"config": "inverting", "Rf": "10kΩ", "Rin": "1kΩ", "Vin": 1.0}))
    assert "error" not in r
    assert r["电压增益 A"].startswith("-10")  # A = -Rf/Rin = -10
    r2 = json.loads(tools.execute("analyze_op_amp",
                                  {"config": "non_inverting", "Rf": "10kΩ", "Rin": "1kΩ"}))
    assert r2["电压增益 A"].startswith("11")  # A = 1 + Rf/Rin = 11


def test_tools_invalid_kind():
    r = json.loads(tools.execute("analyze_rc_filter", {"R": 1000, "C": 1e-6, "kind": "bandpass"}))
    assert "error" in r and "参数错误" in r["error"]


def test_tools_invalid_config():
    r = json.loads(tools.execute("analyze_op_amp",
                                 {"config": "weird", "Rf": 1, "Rin": 1}))
    assert "error" in r and "参数错误" in r["error"]


def test_tools_missing_param():
    r = json.loads(tools.execute("analyze_rc_filter", {"R": 1000}))
    assert "error" in r and "参数错误" in r["error"]


def test_tools_code_generation():
    r = json.loads(tools.execute("generate_simulation_code",
                                 {"code_type": "python_rc", "R": "1k", "C": "1uF"}))
    assert r["code_type"] == "python_rc"
    assert "R, C = 1000.0, 1e-06" in r["code"] or "1000.0" in r["code"]
    # 非法 bits
    r2 = json.loads(tools.execute("generate_simulation_code",
                                  {"code_type": "verilog_counter", "bits": 0}))
    assert "error" in r2


def test_tools_unknown():
    r = json.loads(tools.execute("no_such_tool", {}))
    assert "error" in r


def test_tools_render_figure():
    # 图形工具应返回非空 matplotlib Figure（含带单位输入），中文字体不报错
    for name, args in [
        ("analyze_rc_filter", {"R": "1kΩ", "C": "1μF"}),
        ("analyze_rl_filter", {"R": 1000, "L": "10mH"}),
        ("analyze_rlc_circuit", {"R": 100, "L": 1e-3, "C": 1e-6}),
        ("simulate_rc_transient", {"R": 1000, "C": 1e-6, "V0": 5}),
        ("simulate_rlc_transient", {"R": 10, "L": 1e-2, "C": 1e-4}),
    ]:
        fig = tools.render(name, args)
        assert fig is not None, f"{name} 应返回图形"


# ---------------------------------------------------------------- 电路计算
def test_circuit_rc():
    r = circuit.format_rc_result(1000, 1e-6, "lowpass")
    fc = 1 / (2 * 3.141592653589793 * 1000 * 1e-6)
    assert abs(float(r["截止频率 fc"].split()[0]) - fc) / fc < 1e-3


def test_circuit_rlc_cross_check():
    # BW = f0/Q 与 BW = R/(2πL) 两公式应一致，自校验为 ✅
    r = circuit.format_rlc_result(100, 1e-3, 1e-6)
    assert "一致" in r["结果自校验"]
    p = circuit.rlc_series_params(100, 1e-3, 1e-6)
    import math
    bw_alt = 100 / (2 * math.pi * 1e-3)
    assert abs(p["BW"] - p["BW_alt"]) < 1e-9 * max(p["BW"], 1)
    assert abs(p["BW"] - bw_alt) < 1e-9 * max(bw_alt, 1)


def test_circuit_op_amp_invalid():
    try:
        circuit.format_op_amp_result("bad", 10000, 1000)
        raise AssertionError("应抛错")
    except ValueError:
        pass


def test_circuit_rlc_step_overshoot():
    # 欠阻尼应给出正超调量
    p = circuit.rlc_step_params(10, 1e-2, 1e-4)
    assert p["case"] == "欠阻尼"
    assert p["overshoot"] > 0


# ---------------------------------------------------------------- 仿真代码
def test_simulation_generate():
    code = simulation.generate("python_rlc", R=100, L=1e-3, C=1e-6)
    assert "f0 = 1 / (2 * np.pi * np.sqrt(L * C))" in code
    assert "100" in code and "0.001" in code and "1e-06" in code


# ---------------------------------------------------------------- 信号分析
def test_signal_analyze():
    import numpy as np
    t = np.arange(200) / 10000.0
    y = np.sin(2 * np.pi * 1000 * t)
    stats = signal.analyze(y, fs=10000.0)
    assert stats["样本数"] == 200
    # 主频应接近 1000 Hz
    assert "1000" in stats["主频"]
    assert stats["波形类型(粗略)"] == "正弦波"


# ---------------------------------------------------------------- RAG
def test_rag_load():
    docs = rag.load_knowledge()
    assert len(docs) > 0  # 知识库示例应存在
    # lru_cache：两次调用返回同一对象
    assert rag.load_knowledge() is docs


def test_rag_retrieve():
    hits = rag.retrieve("RC 低通滤波器 截止频率")
    assert hits, "应检索到 RC 相关片段"


# ---------------------------------------------------------------- 实验工作台编排层
def test_experiment_run_rc():
    r = experiment.run_experiment("rc_filter", {"R": "1kΩ", "C": "1μF", "kind": "lowpass"})
    assert r["compute_ok"] is True
    assert "159.2" in r["result"]["截止频率 fc"]
    assert r["fig_ok"] is True
    assert "已完成确定性计算" in r["conclusion"]


def test_experiment_cross_check():
    r = experiment.run_experiment("rlc_resonance", {"R": "10Ω", "L": "10mH", "C": "100μF"})
    assert r["cross_ok"] is True
    assert "一致" in r["cross_check"]
    # 验证清单含交叉验证项且状态 ok
    kinds = {label: status for label, status, _ in r["verification"]}
    assert kinds.get("交叉验证") == "ok"


def test_experiment_op_amp():
    r = experiment.run_experiment("op_amp",
                                  {"config": "inverting", "Rf": "10kΩ", "Rin": "1kΩ", "Vin": "1V"})
    assert r["result"]["电压增益 A"].startswith("-10")


def test_experiment_bad_param_no_crash():
    # 非法参数不应崩溃：解析失败被记录，结论诚实说明失败
    r = experiment.run_experiment("rc_filter", {"R": "abc", "C": "1μF", "kind": "lowpass"})
    assert r["compute_ok"] is False
    assert r["parse_errors"]
    assert "未完成" in r["conclusion"]
    report = experiment.generate_report({k: v for k, v in r.items() if k != "figure"})
    assert "未获得计算结果" in report  # 失败时不伪造结果


def test_experiment_report_sections():
    r = experiment.run_experiment("rlc_resonance", {"R": "10Ω", "L": "10mH", "C": "100μF"})
    rec = {k: v for k, v in r.items() if k != "figure"}
    report = experiment.generate_report(rec)
    # 报告应含 10 个章节（一~十）
    for h in ("一、实验目的", "五、实验结果", "九、实验结论", "十、思考题"):
        assert h in report


def test_experiment_save_load():
    import os
    import tempfile
    old = experiment.RECORDS_DIR
    experiment.RECORDS_DIR = tempfile.mkdtemp()
    try:
        r = experiment.run_experiment("voltage_divider",
                                      {"R1": "1kΩ", "R2": "1kΩ", "Vin": "5V"})
        rec = {k: v for k, v in r.items() if k != "figure"}
        fn = experiment.save_record(rec, None)
        assert os.path.exists(fn)
        # 记录不含 matplotlib figure / 密钥字段
        assert "figure" not in rec
        records = experiment.list_records()
        assert len(records) == 1
        assert records[0]["exp_id"] == "voltage_divider"
    finally:
        experiment.RECORDS_DIR = old


def test_experiment_system_checks():
    checks = experiment.system_checks()
    assert checks, "应有自检项"
    # 核心子系统自检必须全部通过（确定性计算/单位/交叉验证/RAG/Agent/报告/测试）
    for name, status, note in checks:
        assert status == "ok", f"{name} 自检未通过：{note}"


def run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n通过 {passed}/{passed + failed}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
