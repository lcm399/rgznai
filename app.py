"""电子信息实验仿真 AI 助手 —— Streamlit 主程序。

运行方式：
    pip install -r requirements.txt
    streamlit run app.py

设计要点：核心工程计算（电路/频谱）为确定性计算，不联网也能演示；
LLM 负责自然语言理解与解释，未配置 Key 时给出友好提示。
"""
from __future__ import annotations

import json
import math
import os
import time

import streamlit as st

# 先加载 .env（若存在），再导入依赖环境变量的模块
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from core import agent, circuit, experiment, llm, prompts, rag, signal, simulation

st.set_page_config(page_title="电子信息实验仿真 AI 助手", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------- 答辩素材（创新点 / 问答）
INNOVATIONS = [
    ("AI 理解与确定性计算分层",
     "数值全部由 numpy/scipy 解析公式算出，LLM 只做理解与讲解；冲突时以计算为准，杜绝「大模型瞎算电路」。"),
    ("多工具 Agent 协同（Function Calling）",
     "AI 自主完成「接收请求→分析决策→调用工具→反馈结果」，而非页面写死的问答。"),
    ("工程单位自然语言解析",
     "「1kΩ / 1μF / 10mH」直接解析为 SI 单位，大小写敏感（m=毫 vs M=兆），贴近真实实验。"),
    ("工程计算交叉验证",
     "关键物理量用两条独立公式互查（BW=f0/Q 与 BW=R/(2πL)），不一致自动警示。"),
    ("RAG 工程知识增强",
     "教材 / 实验手册片段检索、带来源回答，零外部依赖、可离线运行。"),
    ("自动生成实验图形",
     "Bode 图、暂态曲线、频谱图随确定性结果自动生成，中文字体适配。"),
    ("实验可记录、可复现、可信度显式",
     "一键复制参数重跑；本地 JSON/Markdown 记录与报告；✓⚠️❌ 可信度等级，拒绝「准确率 99%」式虚报。"),
    ("从「问答」升级为「实验闭环」",
     "7 阶段实验状态机把「做实验」变成可追踪、可定位、可恢复的闭环。"),
]

DEFENSE_FAQ = [
    ("为什么不让大模型直接算？",
     "大模型会「猜」数字且经常算错；我们把数值交给确定性公式，LLM 只负责理解与组织，可复现、可验证。"),
    ("没有 API Key 还能演示吗？",
     "能。电路计算、Bode 图、FFT、代码生成、交叉验证全部离线可跑；只有 AI 讲解需要 Key。"),
    ("交叉验证如何体现严谨性？",
     "RLC 带宽用 BW=f0/Q 与 BW=R/(2πL) 两条独立公式互查，相对误差超阈值自动警示。"),
    ("如何保证不虚构实验结果？",
     "报告只写入实际运行数据；未做的项明确写「未进行该项测试」；可信度用 ✓/⚠️/❌ 而非百分比。"),
    ("项目如何扩展新实验？",
     "在 experiment.EXPERIMENTS 里加一条元数据即接入工作台，复用工具 / 单位解析 / 绘图，无需重写计算。"),
]


# ---------------------------------------------------------------- 工具函数
def llm_block(messages: list[dict]) -> None:
    """流式输出 LLM 回复；未配置 Key 时提示。"""
    if not llm.is_configured():
        st.info("💡 未配置 LLM_API_KEY（复制 .env.example 为 .env 并填入），"
                "下方确定性计算结果仍可正常演示。")
        return
    box = st.empty()
    try:
        full = ""
        for delta in llm.chat_stream(messages):
            full += delta
            box.markdown(full)
    except Exception as e:  # 网络/额度等异常也要能继续演示
        box.error(f"LLM 调用失败：{e}")


# ---------------------------------------------------------------- 实验工作台辅助
def _stage_states(exp: dict, result: dict | None) -> list[tuple[str, str]]:
    """计算 7 阶段实验状态机的各阶段状态。

    状态：done=完成 / active=进行中 / fail=失败 / warn=部分通过 / na=不适用。
    """
    has = result is not None
    compute_ok = has and result.get("compute_ok")
    fig_ok = has and result.get("fig_ok")
    if has:
        ss = {s for _, s, _ in result.get("verification", [])}
        if "fail" in ss:
            verify = "fail"
        elif "warn" in ss:
            verify = "warn"
        else:
            verify = "done"
    else:
        verify = "active"

    def fig_state():
        if not has:
            return "active"
        if not exp.get("has_figure"):
            return "na"
        return "done" if fig_ok else "fail"

    return [
        ("实验准备", "done"),
        ("参数输入", "done" if has else "active"),
        ("理论分析", "done"),
        ("实验计算", "done" if compute_ok else ("fail" if has else "active")),
        ("图形生成", fig_state()),
        ("结果验证", verify),
        ("实验结论", "done" if has else "active"),
    ]


_STAGE_ICON = {"done": "✅", "active": "🔄", "fail": "❌", "warn": "⚠️", "na": "➖"}

_STAGE_DESC = {
    "实验准备": "选定实验与目标",
    "参数输入": "填写并解析参数",
    "理论分析": "展示公式原理",
    "实验计算": "确定性公式计算",
    "图形生成": "自动绘制结果图",
    "结果验证": "校验清单 + 交叉验证",
    "实验结论": "生成实验结论",
}


def _render_stepper(exp: dict, result: dict | None) -> None:
    """渲染 7 阶段实验状态机（stepper），每阶段附简短说明并给出下一步提示。"""
    states = _stage_states(exp, result)
    cols = st.columns(len(states))
    for c, (label, state) in zip(cols, states):
        with c:
            if state == "active":
                c.markdown(f"{_STAGE_ICON[state]} **{label}** 🔦")
            else:
                c.markdown(f"{_STAGE_ICON[state]} **{label}**")
            c.caption(_STAGE_DESC[label])

    # 下一步提示
    if result is None:
        st.info("👉 **下一步**：在上方填写参数，点击「▶️ 开始实验」运行实验闭环。")
    else:
        pending = next((l for l, s in states if s == "active"), None)
        if pending:
            st.caption(f"👉 当前阶段：**{pending}**（{_STAGE_DESC[pending]}）")


def _trust_level(result: dict) -> str:
    """可信度等级：✓已验证 / ⚠️部分验证 / ❌未验证（绝不写"准确率 99%"）。"""
    ss = {s for _, s, _ in result.get("verification", [])}
    if "fail" in ss:
        return "❌ 未验证"
    if "warn" in ss:
        return "⚠️ 部分验证"
    return "✓ 已验证"


def _result_sources(exp: dict) -> list[tuple[str, str]]:
    """结果来源说明（可追溯性）；不暴露 API Key / 系统 Prompt / 敏感配置。"""
    verify = "校验清单" + (" + RLC 双公式交叉验证" if exp.get("cross_check") else "")
    return [
        ("参数", "用户在工作台输入（带工程单位，大小写敏感）"),
        ("理论", "core/experiment.py · 实验定义中的解析公式"),
        ("计算", "确定性工具（numpy/scipy 公式，复用 core/tools）"),
        ("验证", verify),
        ("解释", "LLM Agent（仅解释原理，不改变数值）"),
    ]


def _render_workbench_result(exp: dict, result: dict) -> None:
    """渲染实验结果：五步流程 → 错误提示 → 可信度 → 计算 → 图形 → 验证 → AI → 结论 → 报告 → 来源。"""
    # ---- 五步流程总览（输入 → 计算 → 验证 → AI 解释 → 结论）----
    st.markdown("🔁 **实验流程**：`参数输入 → 确定性计算 → 结果验证 → AI 解释 → 实验结论`")
    st.caption("计算 / 验证 / 结论均为离线确定性步骤；AI 仅在第 4 步做解释，不改变数值。")

    # ---- 友好参数错误提示（哪里错 / 为什么错 / 怎么改）----
    errs = result.get("param_errors") or {}
    if errs or not result.get("compute_ok"):
        st.error("❌ 参数输入有误，请按下方提示修改后，重新点击「▶️ 开始实验」：")
        for p in exp["params"]:
            key = p["key"]
            if key in errs:
                raw = result["params"].get(key, "")
                unit = p.get("unit") or ""
                hint = f"（例如 `{p['default']}`）" if p.get("default") else ""
                st.markdown(f"- **{p['label']}**：输入 `{raw}` 无效 —— {errs[key]}。"
                            f"请填写有效的 {unit} 数值{hint}。")

    # ---- 可信度面板 ----
    st.subheader("🛡️ 结果可信度")
    tl = _trust_level(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("可信度等级", tl)
    c2.metric("计算类型", "确定性计算")
    c3.metric("交叉验证", "通过" if result.get("cross_ok") else ("—" if exp.get("cross_check") else "不适用"))
    with st.expander("查看计算依据 / 输入 / 验证清单", expanded=False):
        st.markdown("**计算依据（公式）**")
        for f in exp["formula"]:
            st.code(f)
        st.markdown("**输入（原始 → SI）**")
        rows = []
        for p in exp["params"]:
            raw = result["params"].get(p["key"], "")
            parsed = result["parsed"].get(p["key"])
            if p.get("type") == "select":
                parsed_txt = str(raw)
            elif p.get("type") == "textarea":
                parsed_txt = "（采样数据）"
            elif parsed is None:
                parsed_txt = "❌ 解析失败"
            else:
                parsed_txt = f"{parsed:g}"
            rows.append(f"| {p['label']} | {raw} | {parsed_txt} |")
        st.markdown("| 参数 | 输入 | SI 值 |\n|------|------|-------|\n" + "\n".join(rows))
        st.caption("验证清单见下方「✅ 结果验证」区块。")

    # ---- 确定性计算结果（与 AI 严格分层）----
    st.subheader("📊 确定性计算结果（以此为准）")
    st.caption("以下数值全部由 numpy/scipy 解析公式计算，**不经过 LLM**，可复现、可验证。")
    box = st.container(border=True)
    with box:
        if result.get("compute_ok"):
            for k, v in result["result"].items():
                if k == "说明":
                    box.caption(v)
                else:
                    box.markdown(f"**{k}**：{v}")
        else:
            box.error(f"计算未完成：{result['result'].get('error', '未知错误')}")

    # ---- 图形 ----
    if exp.get("has_figure"):
        st.subheader("📉 结果图形")
        if result.get("fig_ok") and result.get("figure") is not None:
            st.pyplot(result["figure"])
        else:
            st.warning("图形未生成（数值结果仍有效）。")

    # ---- 结果验证（独立顶层区块）----
    st.subheader("✅ 结果验证")
    for label, status, note in result["verification"]:
        mark = {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(status, "➖")
        st.markdown(f"- {mark} **{label}**：{note}")
    if result.get("cross_check"):
        st.info(f"🔗 交叉验证：{result['cross_check']}")

    # ---- AI 实验分析（可选、分层）----
    st.subheader("🤖 AI 实验分析（仅供参考）")
    st.caption("⚠️ 若 AI 解释与上方确定性计算存在差异，**请以计算结果为准**。")
    if not llm.is_configured():
        st.info("未配置 LLM_API_KEY，跳过 AI 分析；确定性结果不受影响。")
    elif st.button("✨ 生成 AI 分析", key=f"ai_{exp['id']}"):
        summary = "\n".join(f"{k}: {v}" for k, v in result["result"].items())
        with st.spinner("AI 正在解读结果……"):
            try:
                text = llm.chat([
                    {"role": "system", "content": prompts.SYSTEM_CIRCUIT},
                    {"role": "user", "content": f"请围绕以下确定性计算结果解释原理（不要更改数值）：\n{summary}"},
                ])
                st.session_state["wb_ai"] = text
            except Exception as e:  # noqa: BLE001
                st.error(f"AI 分析失败：{e}")
    if st.session_state.get("wb_ai"):
        st.markdown(st.session_state["wb_ai"])

    # ---- 自动实验结论 ----
    st.subheader("📌 实验结论")
    st.success(result["conclusion"])

    # ---- 报告 + 记录 ----
    st.subheader("📄 实验报告 / 记录")
    rec = {k: v for k, v in result.items() if k != "figure"}
    rec["ai_analysis"] = st.session_state.get("wb_ai")
    report_md = experiment.generate_report(rec)
    cr, cs, _ = st.columns([1, 1, 1])
    with cr:
        st.download_button("⬇️ 下载报告 (Markdown)", report_md,
                           file_name=f"{exp['id']}_实验报告.md", mime="text/markdown",
                           use_container_width=True)
    with cs:
        if st.button("💾 保存实验记录", use_container_width=True):
            fn = experiment.save_record(rec, result.get("figure"))
            st.success(f"已保存：{fn}")
    with st.expander("📝 报告预览", expanded=False):
        st.markdown(report_md)

    records = experiment.list_records()
    if records:
        with st.expander(f"🗂️ 历史实验记录（{len(records)} 条）", expanded=False):
            for r in records[:10]:
                st.markdown(f"- `{r.get('time','')}` · **{r.get('exp',{}).get('name','')}**"
                            f" · {(r.get('conclusion') or '')[:48]}")

    # ---- 结果来源（可追溯性；不暴露 Key / Prompt / 敏感配置）----
    st.subheader("🔎 结果来源")
    for src, note in _result_sources(exp):
        st.markdown(f"- **{src}**：{note}")


# ---------------------------------------------------------------- 首页系统状态总览
def _render_system_status() -> None:
    """首页「系统状态总览」：对子系统做真实自检（非静态文案），响应式网格布局。"""
    st.markdown("### 📡 系统状态总览")
    checks = experiment.system_checks()
    n_ok = sum(1 for _, s, _ in checks if s == "ok")
    per_row = 4
    for i in range(0, len(checks), per_row):
        row = checks[i:i + per_row]
        cols = st.columns(len(row))
        for c, (name, status, note) in zip(cols, row):
            with c:
                mark = {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(status, "➖")
                c.markdown(f"{mark} **{name}**")
                c.caption(note)
    st.caption(f"自检结果：{n_ok}/{len(checks)} 项通过（数值为真实运行结果，非静态文案）")


# ---------------------------------------------------------------- 侧边栏导航
st.sidebar.title("⚡ 电子信息实验仿真 AI 助手")
st.sidebar.caption("iCAN 2026 · 软件赛道参赛作品")

# 处理来自「一键体验 / 一键演示」按钮的跳转请求。
# 必须在本轮 radio 实例化之前写入 nav，否则 Streamlit 抛
# StreamlitWidgetAlreadyInstantiatedError（按钮里已改为写入 _pending_nav）。
if st.session_state.get("_pending_nav"):
    st.session_state.nav = st.session_state.pop("_pending_nav")

page = st.sidebar.radio(
    "功能导航",
    ["🏠 首页", "🧪 实验工作台", "🔍 电路分析", "💻 仿真代码生成",
     "📈 实验数据解读", "🎓 专业答疑", "🤖 智能体对话", "🎤 答辩模式"],
    key="nav",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"LLM 状态：{'✅ 已配置' if llm.is_configured() else '⚠️ 未配置（核心计算仍可用）'}"
)


# ---------------------------------------------------------------- 首页
if page == "🏠 首页":
    st.title("⚡ 电子信息实验仿真 AI 助手")
    st.caption("iCAN 2026 软件赛道 · 电子信息工程实验仿真智能体")

    # ---- 一句话定位（L2）----
    st.markdown(
        "> 用自然语言完成电子信息工程实验的 **理解 → 计算 → 验证 → 解释 → 记录** 完整闭环，"
        "数值由确定性公式计算、AI 只负责讲解，杜绝「大模型瞎算电路」。"
    )

    # ---- 三个主要入口（L3）----
    st.markdown("### 🚀 从这里开始")
    entries = [
        ("🧪", "开始实验", "进入实验工作台，走完 7 阶段实验闭环（计算→图形→验证→结论→报告）", "🧪 实验工作台"),
        ("💬", "智能问答", "用自然语言提问，AI 自主调用工具精确计算并讲解原理", "🤖 智能体对话"),
        ("🎤", "答辩模式", "一页式速览：定位 / 八大创新点 / 评审问答 / 一键演示", "🎤 答辩模式"),
    ]
    ecols = st.columns(len(entries))
    for c, (icon, title, desc, target) in zip(ecols, entries):
        with c:
            if c.button(f"{icon} {title}", use_container_width=True):
                st.session_state["_pending_nav"] = target
                st.rerun()
            c.caption(desc)

    st.markdown("---")

    # ---- 快速体验：点击直达演示，无需手动输入参数（L4 真实能力）----
    st.markdown("### 🎯 快速体验")
    st.markdown("点击下方任一卡片，直接看到智能体「**分析决策 → 调用工具 → 反馈结果**」全过程：")

    demos = [
        ("🔍 RC 低通滤波器", "一个 RC 低通滤波器，R=1kΩ，C=1μF，求截止频率并画 Bode 图", "🤖 智能体对话", None),
        ("⚡ RLC 暂态响应", "RLC 串联电路 R=10Ω，L=10mH，C=100μF 加阶跃电压，分析阻尼比与超调量", "🤖 智能体对话", None),
        ("🔺 反相放大器", "反相放大器 Rf=10kΩ，Rin=1kΩ，求放大倍数与输出电压", "🤖 智能体对话", None),
        ("🎓 RLC 谐振答疑", "什么是 RLC 串联谐振？品质因数 Q 的含义是什么？", "🤖 智能体对话", None),
        ("📈 1kHz 正弦信号", None, "📈 实验数据解读", "sine_1k"),
    ]
    cols = st.columns(len(demos))
    for c, (label, q, target, sample) in zip(cols, demos):
        with c:
            if c.button(label, use_container_width=True):
                if sample == "sine_1k":  # 预载 1kHz 正弦采样数据，跳到数据解读页
                    st.session_state.sig_raw = "\n".join(
                        f"{math.sin(2 * math.pi * 1000 * i / 10000):.4f}" for i in range(200))
                    st.session_state.sig_fs = 10000.0
                else:
                    st.session_state.agent_preset = q
                st.session_state["_pending_nav"] = target
                st.rerun()

    st.markdown("---")

    # ---- 核心能力与技术特点 ----
    st.markdown("### 🧩 四大核心能力")
    cap_cols = st.columns(4)
    caps = [
        ("🔍 电路分析", "输入 RC/RLC 参数 → 精确计算截止频率/谐振/暂态，画图并讲解原理"),
        ("💻 仿真代码生成", "电路描述 → 生成 Python / LTSpice / Verilog 可运行代码"),
        ("📈 实验数据解读", "粘贴示波器数据 → 统计 + FFT 频谱 + 异常检测，判断信号特征"),
        ("🎓 专业答疑", "教材/实验手册做成知识库（RAG），带资料准确回答、降低幻觉"),
    ]
    for c, (title, desc) in zip(cap_cols, caps):
        with c:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("### ⚙️ 技术特点")
    st.markdown(
        """
        - **确定性计算 + LLM 双引擎**：电路与频谱用 numpy/scipy 精确公式计算，AI 不"瞎编"；
        - **智能体工具链（Function Calling）**：AI 主动选择工具、传参数、拿结果再讲解；
        - **核心功能可离线演示**：未配 API Key 时，工程计算部分仍可完整跑通；
        - **结果交叉验证**：关键量用两个独立公式互相校验，工程结果可靠可信。
        """
    )

    st.markdown("---")
    _render_system_status()


# ---------------------------------------------------------------- 实验工作台
elif page == "🧪 实验工作台":
    st.title("🧪 实验工作台")
    st.caption("完整实验闭环：选择实验 → 参数输入 → 理论 → 确定性计算 → 图形 → 验证 → 结论 → 报告")

    exp_by_name = {e["name"]: e["id"] for e in experiment.EXPERIMENTS}
    name = st.selectbox("选择实验", list(exp_by_name.keys()))
    exp = experiment.get_experiment(exp_by_name[name])

    # ---- 实验目标 + 理论（阶段 1/3）----
    st.markdown("### 🎯 实验目标")
    st.markdown(exp["goal"])
    with st.expander("📚 理论公式与原理", expanded=True):
        for f in exp["formula"]:
            st.code(f)
        for t in exp["theory"]:
            st.markdown(f"- {t}")

    # ---- 参数输入（阶段 2）----
    st.markdown("### ⚙️ 参数输入")
    st.caption("支持带单位输入（如 `1kΩ`、`1μF`、`10mH`），系统自动换算为 SI 单位。")
    params = {}
    pcols = st.columns(3)
    for i, p in enumerate(exp["params"]):
        with pcols[i % 3]:
            key = f"wb_{exp['id']}_{p['key']}"
            if p.get("type") == "select":
                label_to_val = {label: val for label, val in p["options"]}
                label = st.selectbox(p["label"], list(label_to_val.keys()), key=key)
                params[p["key"]] = label_to_val[label]
            elif p.get("type") == "textarea":
                params[p["key"]] = st.text_area(
                    p["label"], height=120, key=key,
                    placeholder="每行一个采样值，或「时间 数值」两列")
            else:
                params[p["key"]] = st.text_input(
                    f"{p['label']}（{p['unit']}）", value=p["default"], key=key)

    c_run, c_copy = st.columns([1, 1])
    if c_run.button("▶️ 开始实验", type="primary", use_container_width=True):
        st.session_state["wb_last"] = experiment.run_experiment(exp["id"], params)
        st.session_state["wb_ai"] = None  # 换参数后旧的 AI 分析作废
        st.rerun()
    if c_copy.button("📋 复制实验参数（可复现）", use_container_width=True):
        st.session_state["wb_copy"] = json.dumps(params, ensure_ascii=False, indent=2)
    st.caption("同参数重新运行结果一致（确定性、可复现）；修改参数后请重新点击「▶️ 开始实验」。")
    if st.session_state.get("wb_copy"):
        with st.expander("📋 可复现参数（粘贴回输入框即可重跑）", expanded=True):
            st.code(st.session_state["wb_copy"], language="json")

    # ---- 状态机 stepper + 结果（阶段 4~7）----
    st.markdown("### 🧭 实验状态")
    result = st.session_state.get("wb_last")
    if result and result.get("exp_id") == exp["id"]:
        _render_stepper(exp, result)
        st.markdown("---")
        _render_workbench_result(exp, result)
    else:
        _render_stepper(exp, None)
        st.info("输入参数后点击「▶️ 开始实验」运行实验闭环。")


# ---------------------------------------------------------------- 电路分析
elif page == "🔍 电路分析":
    st.title("🔍 电路分析")
    st.caption("确定性计算 + AI 讲解：输入参数，实时得到精确结果与图形")

    ctype = st.selectbox("电路类型", [
        "RC 低通滤波器", "RC 高通滤波器", "RC 充放电瞬态",
        "RL 低通滤波器", "RL 高通滤波器",
        "RLC 串联谐振", "RLC 阶跃响应",
        "电阻分压电路", "反相放大器", "同相放大器",
    ])

    # 按电路类型显示对应输入参数
    if ctype == "RC 充放电瞬态":
        c1, c2, c3 = st.columns(3)
        R = c1.number_input("电阻 R (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        C = c2.number_input("电容 C (F)", min_value=1e-12, value=1e-6, format="%.3g")
        V0 = c3.number_input("电源/初始电压 V0 (V)", value=5.0, format="%.3g")
    elif ctype == "RLC 阶跃响应":
        c1, c2, c3, c4 = st.columns(4)
        R = c1.number_input("电阻 R (Ω)", min_value=1e-6, value=10.0, format="%.3g")
        L = c2.number_input("电感 L (H)", min_value=1e-9, value=1e-2, format="%.3g")
        C = c3.number_input("电容 C (F)", min_value=1e-12, value=1e-4, format="%.3g")
        V = c4.number_input("阶跃电压 V (V)", value=1.0, format="%.3g")
    elif ctype.startswith("RC"):
        c1, c2 = st.columns(2)
        R = c1.number_input("电阻 R (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        C = c2.number_input("电容 C (F)", min_value=1e-12, value=1e-6, format="%.3g")
    elif ctype.startswith("RL"):
        c1, c2 = st.columns(2)
        R = c1.number_input("电阻 R (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        L = c2.number_input("电感 L (H)", min_value=1e-9, value=1e-3, format="%.3g")
    elif ctype == "RLC 串联谐振":
        c1, c2, c3 = st.columns(3)
        R = c1.number_input("电阻 R (Ω)", min_value=1e-6, value=100.0, format="%.3g")
        L = c2.number_input("电感 L (H)", min_value=1e-9, value=1e-3, format="%.3g")
        C = c3.number_input("电容 C (F)", min_value=1e-12, value=1e-6, format="%.3g")
    elif ctype == "电阻分压电路":
        c1, c2, c3 = st.columns(3)
        R1 = c1.number_input("R1 (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        R2 = c2.number_input("R2 (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        Vin = c3.number_input("输入电压 Vin (V)", value=5.0, format="%.3g")
    else:  # 运算放大器
        c1, c2, c3 = st.columns(3)
        Rf = c1.number_input("反馈电阻 Rf (Ω)", min_value=1e-6, value=10000.0, format="%.3g")
        Rin = c2.number_input("输入电阻 Rin (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        Vin = c3.number_input("输入电压 Vin (V)", value=1.0, format="%.3g")

    if st.button("开始分析", type="primary"):
        fig = None
        if ctype == "RC 充放电瞬态":
            result = circuit.format_rc_transient_result(R, C, V0)
            fig = circuit.plot_rc_transient(R, C, V0)
        elif ctype == "RLC 阶跃响应":
            result = circuit.format_rlc_step_result(R, L, C, V)
            fig = circuit.plot_rlc_step(R, L, C, V)
        elif ctype.startswith("RC"):
            kind = "lowpass" if "低通" in ctype else "highpass"
            result = circuit.format_rc_result(R, C, kind)
            fig = circuit.plot_rc_bode(R, C, kind)
        elif ctype.startswith("RL"):
            kind = "lowpass" if "低通" in ctype else "highpass"
            result = circuit.format_rl_result(R, L, kind)
            fig = circuit.plot_rl_bode(R, L, kind)
        elif ctype == "RLC 串联谐振":
            result = circuit.format_rlc_result(R, L, C)
            fig = circuit.plot_rlc_response(R, L, C)
        elif ctype == "电阻分压电路":
            result = circuit.format_divider_result(R1, R2, Vin)
        else:
            config = "inverting" if "反相" in ctype else "non_inverting"
            result = circuit.format_op_amp_result(config, Rf, Rin, Vin)

        st.subheader("📊 计算结果")
        for k, v in result.items():
            if k == "说明":
                st.caption(v)
            else:
                st.markdown(f"**{k}**：{v}")
        if fig is not None:
            st.pyplot(fig)

        st.subheader("🤖 AI 原理讲解")
        summary = "\n".join(f"{k}: {v}" for k, v in result.items())
        llm_block([
            {"role": "system", "content": prompts.SYSTEM_CIRCUIT},
            {"role": "user", "content": f"请围绕以下计算结果讲解原理：\n{summary}"},
        ])


# ---------------------------------------------------------------- 仿真代码生成
elif page == "💻 仿真代码生成":
    st.title("💻 仿真代码生成")
    st.caption("按参数生成可直接运行的 Python / LTSpice / Verilog 仿真代码")

    code_type = st.selectbox(
        "代码类型",
        ["Python 数值仿真 (RC)", "Python 数值仿真 (RLC 谐振)", "Python 数值仿真 (RLC 暂态)",
         "LTSpice 网表 (RC)", "Verilog 计数器"],
    )

    is_rlc = "RLC" in code_type
    if is_rlc:
        c1, c2, c3 = st.columns(3)
        with c1:
            R = st.number_input("R (Ω)", min_value=1e-6, value=100.0, format="%.3g")
        with c2:
            L = st.number_input("L (H)", min_value=1e-9, value=1e-3, format="%.3g")
        with c3:
            C = st.number_input("C (F)", min_value=1e-12, value=1e-6, format="%.3g")
    else:
        c1, c2 = st.columns(2)
        with c1:
            R = st.number_input("R (Ω)", min_value=1e-6, value=1000.0, format="%.3g")
        with c2:
            C = st.number_input("C (F)", min_value=1e-12, value=1e-6, format="%.3g")
    kind = "lowpass"
    if code_type.startswith(("Python", "LTSpice")) and "RLC" not in code_type:
        kind = st.radio("滤波器类型", ["低通 lowpass", "高通 highpass"], horizontal=True).split()[0]

    if st.button("生成代码", type="primary"):
        if code_type == "Python 数值仿真 (RC)":
            code = simulation.generate_python_rc(R, C, kind)
        elif code_type == "Python 数值仿真 (RLC 谐振)":
            code = simulation.generate_python_rlc(R, L, C)
        elif code_type == "Python 数值仿真 (RLC 暂态)":
            code = simulation.generate_python_rlc_transient(R, L, C)
        elif code_type.startswith("LTSpice"):
            code = simulation.generate_ltspice_rc(R, C, kind)
        else:
            code = simulation.generate_verilog_counter()
        st.code(code, language="python" if "Python" in code_type else ("verilog" if "Verilog" in code_type else "text"))

        st.subheader("🤖 AI 代码解读")
        llm_block([
            {"role": "system", "content": prompts.SYSTEM_SIMULATION},
            {"role": "user", "content": f"解释以下代码的作用与关键参数含义：\n\n{code}"},
        ])


# ---------------------------------------------------------------- 实验数据解读
elif page == "📈 实验数据解读":
    st.title("📈 实验数据解读")
    st.caption("粘贴示波器/传感器数据 → 统计 + FFT 频谱 + 异常检测")

    # 一键示例数据（便于演示，覆盖正弦/方波两种典型波形）
    st.caption("👇 一键载入示例数据（采样率按 1000 Hz）：")
    _n = 100
    _sine = "\n".join(f"{math.sin(2 * math.pi * 50 * i / 1000):.4f}" for i in range(_n))
    _square = "\n".join(("1.0" if (i % 20) < 10 else "-1.0") for i in range(_n))
    c_ex1, c_ex2 = st.columns(2)
    if c_ex1.button("正弦波 50Hz", use_container_width=True):
        st.session_state.sig_raw = _sine
        st.session_state.sig_fs = 1000.0
    if c_ex2.button("方波 50Hz", use_container_width=True):
        st.session_state.sig_raw = _square
        st.session_state.sig_fs = 1000.0

    raw = st.text_area(
        "粘贴数据（每行一个值；或两列「时间 数值」，用逗号/空格分隔）",
        height=180,
        key="sig_raw",
        placeholder="0.000\n0.309\n0.588\n0.809\n0.951\n1.000\n...",
    )
    fs = st.number_input("采样率 fs (Hz)，未知可留空", min_value=0.0, value=0.0,
                         key="sig_fs", format="%.3g")

    if st.button("分析数据", type="primary") and raw.strip():
        try:
            t, y = signal.parse_data(raw)
            stats = signal.analyze(y, fs=fs or None)
        except Exception as e:
            st.error(f"数据解析失败：{e}")
            st.stop()

        if len(y) == 0:
            st.error("未解析到有效数据：请粘贴至少一行数值（或「时间 数值」两列）。")
            st.stop()

        st.subheader("📊 统计与频谱结果")
        for k, v in stats.items():
            st.markdown(f"**{k}**：{v}")
        st.pyplot(signal.plot_signal(t, y, fs=fs or None))

        st.subheader("🤖 AI 智能解读")
        summary = "\n".join(f"{k}: {v}" for k, v in stats.items())
        llm_block([
            {"role": "system", "content": prompts.SYSTEM_SIGNAL},
            {"role": "user", "content": f"请解读以下信号分析结果：\n{summary}"},
        ])


# ---------------------------------------------------------------- 专业答疑（RAG）
elif page == "🎓 专业答疑":
    st.title("🎓 专业答疑")
    st.caption("基于 data/ 知识库（教材/实验手册片段）的 RAG 问答")

    n_docs = len(rag.load_knowledge())
    st.caption(f"当前知识库已加载 **{n_docs}** 个文本片段（把 .md/.txt 放进 data/ 目录即可扩充）")

    question = st.text_input("输入你的问题", placeholder="例如：RC 低通滤波器的截止频率如何计算？")
    if st.button("提问", type="primary") and question.strip():
        context = rag.build_context(question)
        with st.expander("🔎 检索到的资料片段", expanded=False):
            st.text(context if context else "（未检索到相关资料，将依靠模型自身知识回答）")

        messages = [
            {"role": "system", "content": prompts.SYSTEM_QA},
        ]
        if context:
            messages.append({"role": "user", "content": f"{context}\n\n问题：{question}"})
        else:
            messages.append({"role": "user", "content": question})
        llm_block(messages)


# ---------------------------------------------------------------- 智能体对话
elif page == "🤖 智能体对话":
    st.title("🤖 智能体对话")
    st.caption("真正的智能体：AI 判断需求 → 主动调用工具精确计算 → 返回结果并讲解")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 示例一键体验（方便评审快速演示，覆盖各类工具）
    st.caption("👇 点击示例一键体验：")
    examples = [
        ("RC 低通", "一个 RC 低通滤波器，R=1kΩ，C=1μF，求截止频率"),
        ("运放放大", "一个反相放大器，Rf=10kΩ，Rin=1kΩ，求放大倍数"),
        ("RLC 暂态", "一个 RLC 串联电路，R=10Ω，L=10mH，C=100μF，加阶跃电压，分析暂态响应（阻尼比、超调量）"),
        ("生成代码", "生成 RLC 串联谐振电路的 Python 仿真代码，R=100Ω，L=1mH，C=1μF"),
        ("专业答疑", "什么是 RLC 串联谐振？"),
    ]
    cols = st.columns(len(examples))
    preset = None
    for c, (label, q) in zip(cols, examples):
        if c.button(label, use_container_width=True):
            preset = q

    # 首页「一键体验」跳转过来的问题：作为预设问题（用后即清，避免 rerun 重复触发）
    jump_q = st.session_state.pop("agent_preset", None)
    if jump_q:
        preset = jump_q

    if prompt := (st.chat_input("或直接输入你的问题……") or preset):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not llm.is_configured():
                st.info("未配置 LLM_API_KEY，无法运行智能体。请配置后刷新。")
                answer = ("（未配置 LLM_API_KEY，无法运行智能体；请配置后刷新。"
                          "电路计算、图形、代码生成等核心功能仍可在其它页面离线演示。）")
            else:
                # 运行智能体，可视化「接收请求 → 分析决策 → 反馈结果」闭环，
                # 最终回答流式输出（打字机效果）。
                # figures/status 在 try 前初始化：即使中途失败，已生成的结果与图形也保留展示。
                figures = []
                status = None
                full_answer = ""
                t_start = time.time()
                try:
                    status = st.status("🤔 正在分析需求……", expanded=True)
                    status.write("**① 接收请求** —— 已理解你的意图")
                    answer_box = st.empty()
                    for event in agent.run_stream(prompt, history=st.session_state.agent_history):
                        if event["type"] == "step":
                            step = event["step"]
                            status.write(f"**② 分析决策** —— 决定调用工具 `{step['name']}`")
                            status.caption(f"参数：`{step['args']}`")
                            status.write("**③ 反馈结果** —— 工具返回精确结果：")
                            status.code(step["result"], language="json")
                            if step.get("figure") is not None:
                                figures.append((step["name"], step["figure"]))
                        elif event["type"] == "token":
                            full_answer += event["text"]
                            answer_box.markdown(full_answer + " ▌")
                        elif event["type"] == "done":
                            full_answer = event["answer"]
                            answer_box.markdown(full_answer)
                    elapsed = time.time() - t_start
                    answer = full_answer
                    status.update(label=f"✅ 分析完成（耗时 {elapsed:.1f}s）", state="complete", expanded=False)
                except Exception as e:
                    answer = f"智能体运行失败：{e}"
                    if status is not None:
                        status.update(label="⚠️ 分析中断（已生成的结果仍保留）", state="error", expanded=False)
                    st.error(answer)
                # 已成功生成的工具图，即使 AI 解释失败也保留展示（不丢失确定性结果）
                for name, fig in figures:
                    st.caption(f"📊 工具 `{name}` 生成的图：")
                    st.pyplot(fig)

            st.session_state.agent_history.append({"role": "user", "content": prompt})
            st.session_state.agent_history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------- 答辩模式
elif page == "🎤 答辩模式":
    st.title("🎤 答辩模式")
    st.caption("面向现场评审的一页式速览：定位 → 痛点 → 解法 → 创新点 → 常见问答 → 一键演示")

    # ---- 一键开始答辩演示（3 分钟版主流程起点：Demo 1 RC 闭环）----
    if st.button("🚀 开始答辩演示（按 3 分钟版顺序）", type="primary", use_container_width=True):
        st.session_state.agent_preset = "一个 RC 低通滤波器，R=1kΩ，C=1μF，求截止频率并画 Bode 图"
        st.session_state["_pending_nav"] = "🤖 智能体对话"
        st.rerun()

    # ---- 一句话定位 ----
    st.markdown("### 🎯 一句话定位")
    st.success(
        "面向**电子信息工程实验教学**的仿真智能体："
        "让 AI 只负责理解与讲解，数值全部由确定性公式计算 —— 解决「大模型算不准电路」的痛点。"
    )

    # ---- 痛点 / 解法 ----
    c_pain, c_sol = st.columns(2)
    with c_pain:
        st.markdown("### 😖 痛点")
        st.markdown(
            "- 大模型直接答电路题常「一本正经地算错」；\n"
            "- 学生算完参数无法判断对错、无法追溯依据；\n"
            "- 实验结论缺验证、缺记录、难以复现。"
        )
    with c_sol:
        st.markdown("### 💡 解法")
        st.markdown(
            "- **确定性计算 + LLM 双引擎**，数值交给公式；\n"
            "- **交叉验证 + 可信度面板**，结果可查、可追溯；\n"
            "- **实验闭环 + 报告记录**，可复现、可存档。"
        )

    st.markdown("---")

    # ---- 8 大创新点 ----
    st.markdown("### ✨ 八大创新点")
    for i, (title, desc) in enumerate(INNOVATIONS, 1):
        st.markdown(f"**{i}. {title}**")
        st.markdown(f"　{desc}")

    st.markdown("---")

    # ---- 系统状态 ----
    _render_system_status()

    st.markdown("---")

    # ---- 一键演示跳转 ----
    st.markdown("### 🚀 一键演示")
    st.markdown("点击跳转到对应演示页（参数已预填，无需手动输入）：")
    demo_jumps = [
        ("🧪 实验工作台 · RC 滤波闭环", "🧪 实验工作台", None),
        ("🤖 智能体 · RC 低通", "🤖 智能体对话", "一个 RC 低通滤波器，R=1kΩ，C=1μF，求截止频率并画 Bode 图"),
        ("🤖 智能体 · RLC 暂态交叉验证", "🤖 智能体对话", "RLC 串联电路 R=10Ω，L=10mH，C=100μF 加阶跃电压，分析阻尼比与超调量"),
        ("📈 实验数据 · 1kHz 正弦", "📈 实验数据解读", None),
    ]
    dcols = st.columns(len(demo_jumps))
    for c, (label, target, q) in zip(dcols, demo_jumps):
        if c.button(label, use_container_width=True):
            if q:
                st.session_state.agent_preset = q
            elif target == "📈 实验数据解读":
                st.session_state.sig_raw = "\n".join(
                    f"{math.sin(2 * math.pi * 1000 * i / 10000):.4f}" for i in range(200))
                st.session_state.sig_fs = 10000.0
            st.session_state["_pending_nav"] = target
            st.rerun()

    st.markdown("---")

    # ---- 常见问答 ----
    st.markdown("### 💬 评审常见问答")
    for q, a in DEFENSE_FAQ:
        with st.expander(f"Q：{q}", expanded=False):
            st.markdown(a)
