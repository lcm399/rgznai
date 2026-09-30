"""一键运行所有离线回归测试（无需 LLM / 网络）。"""
from tests.test_core import run_all

if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
