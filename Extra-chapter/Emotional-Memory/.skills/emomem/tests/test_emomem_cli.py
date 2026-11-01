"""emomem（vendored 自包含版）CLI 端到端回归测试。

通过 importlib 加载 CLI 模块，所有命令都指向 tmp_path 的 --data-dir，
不触碰 ~/.emomem，也不依赖网络 / 本地 embedding 模型（无模型时自动词法降级）。
同时验证「自包含」承诺：import 到的 src.memory 必须是应用内副本。
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = APP_ROOT / "bin" / "emomem.py"


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location("emomem_cli_under_test", str(CLI_PATH))
    assert spec is not None and spec.loader is not None, "CLI 模块无法定位"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # 应用根应已在 sys.path 上（CLI 模块顶部注入）
    assert str(APP_ROOT) in sys.path, "CLI 未把应用根目录注入 sys.path"
    return module


def invoke(cli_module, data_dir: str, *argv: str) -> tuple[int, str, str]:
    """进程内运行 CLI 的 main()，返回 (exit_code, stdout, stderr)。"""
    sys_argv = ["--data-dir", data_dir, *argv]
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_module.main(sys_argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, out.getvalue(), err.getvalue()


def jload(stdout: str) -> dict:
    return json.loads(stdout.strip())


# ── 自包含 ────────────────────────────────────────────────

def test_imports_app_internal_copy(cli_module):
    """self-contained 证明：src.memory.manager 必须来自应用内副本。"""
    from src.memory import manager as vendor_manager

    resolved = Path(vendor_manager.__file__).resolve()
    assert resolved.is_relative_to(APP_ROOT), f"解析到了非应用内副本: {resolved}"


# ── 基础 ──────────────────────────────────────────────────

def test_help_works_without_creating_dirs(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "--help")
    assert code == 0
    assert "跨会话长期记忆工具" in out
    assert "add" in out
    assert not list(tmp_path.iterdir()), "--help 不应初始化数据目录"


def test_info_initializes_and_reports_backend(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "info", "--json")
    assert code == 0, err
    payload = jload(out)
    assert payload["data_dir"] == str(tmp_path)
    assert payload["permanent"] >= 1
    assert str(APP_ROOT) in payload["app_root"]
    # 无本地 embedding 模型时必须是降级后端，而不是报错
    assert "bounded_lexical_fallback" in payload["vector_backend"]


def test_unknown_command_exits_nonzero(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "definitely-not-a-command")
    assert code != 0


# ── 写入与召回 ────────────────────────────────────────────

def test_add_then_search_roundtrip(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "add", "用户喜欢喝手冲咖啡")
    assert code == 0, err
    assert "lt_" in out

    code, out, err = invoke(cli_module, str(tmp_path), "search", "咖啡", "--json")
    assert code == 0, err
    results = jload(out)["results"]
    assert any("手冲咖啡" in text for text in results)


def test_add_event_writes_layer_and_emotion(cli_module, tmp_path):
    code, out, err = invoke(
        cli_module, str(tmp_path),
        "add-event", "用户在成都出差", "--importance", "0.8", "--emotion", "joy=70",
    )
    assert code == 0, err
    assert "用户在成都出差" in out

    code, out, err = invoke(cli_module, str(tmp_path), "events", "--json")
    assert code == 0, err
    assert any("成都出差" in text for text in jload(out)["events"])


def test_remember_creates_confirmed(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "remember", "用户生日是3月2日", "--json")
    assert code == 0, err
    payload = jload(out)
    assert payload["confirmation_state"] == "confirmed"
    assert payload["memory_id"]

    code, out, err = invoke(cli_module, str(tmp_path), "confirmed", "--json")
    assert code == 0, err
    assert any("3月2日" in row.get("text", "") for row in jload(out)["confirmed"])


def test_list_layers_filter(cli_module, tmp_path):
    invoke(cli_module, str(tmp_path), "add", "短期测试事实", "--layer", "short_term")
    invoke(cli_module, str(tmp_path), "add", "永久测试事实", "--layer", "permanent")

    code, out, err = invoke(cli_module, str(tmp_path), "list", "--layer", "permanent", "--json")
    assert code == 0, err
    texts = [row.get("text", "") for row in jload(out)["items"]]
    assert any("永久测试事实" in t for t in texts)
    assert not any("短期测试事实" in t for t in texts)


# ── 治理 ──────────────────────────────────────────────────

def test_weaken_and_reactivate_flow(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "add", "要被强化的事实")
    assert code == 0, err
    memory_id = out.split("：")[-1].split()[0].strip()
    assert memory_id.startswith("lt_")

    assert invoke(cli_module, str(tmp_path), "strengthen", memory_id)[0] == 0
    assert invoke(cli_module, str(tmp_path), "weaken", memory_id)[0] == 0
    assert invoke(cli_module, str(tmp_path), "reactivate", memory_id)[0] == 0
    assert invoke(cli_module, str(tmp_path), "weaken", "no_such_id")[0] != 0


def test_correct_produces_revision(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "remember", "用户原来是程序员", "--json")
    assert code == 0, err
    old_id = jload(out)["memory_id"]

    code, out, err = invoke(
        cli_module, str(tmp_path), "correct", old_id, "用户现在是产品经理", "--json"
    )
    assert code == 0, err
    new_id = jload(out)["new_memory_id"]
    assert new_id and new_id != old_id

    code, out, err = invoke(cli_module, str(tmp_path), "search", "产品经理", "--json")
    assert code == 0, err
    assert any("产品经理" in t for t in jload(out)["results"])


def test_delete_removes_fact(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "remember", "要被删除的事实", "--json")
    assert code == 0, err
    memory_id = jload(out)["memory_id"]

    code, out, err = invoke(cli_module, str(tmp_path), "delete", memory_id, "--json")
    assert code == 0, err
    assert jload(out)["ok"] is True


def test_pin_unpin(cli_module, tmp_path):
    invoke(cli_module, str(tmp_path), "remember", "要固定的事实")
    _, out, _ = invoke(cli_module, str(tmp_path), "confirmed", "--json")
    row = next(r for r in jload(out)["confirmed"] if "要固定的事实" in r.get("text", ""))
    memory_id = row["id"]

    assert invoke(cli_module, str(tmp_path), "pin", memory_id)[0] == 0
    assert invoke(cli_module, str(tmp_path), "unpin", memory_id)[0] == 0
    assert invoke(cli_module, str(tmp_path), "pin", "no_such_id")[0] != 0


# ── 候选审核 ──────────────────────────────────────────────

def test_candidates_confirm_flow(cli_module, tmp_path):
    invoke(
        cli_module, str(tmp_path),
        "store-interaction", "我喜欢科幻小说", "好的，记住了", "--emotion-intensity", "0.7",
    )
    code, out, err = invoke(cli_module, str(tmp_path), "candidates", "list", "--json")
    assert code == 0, err
    candidates = jload(out)["candidates"]
    assert candidates, "store-interaction 应产生候选"
    cand_id = candidates[0]["id"]

    assert invoke(cli_module, str(tmp_path), "candidates", "confirm", cand_id)[0] == 0

    code, out, err = invoke(cli_module, str(tmp_path), "search", "科幻小说", "--json")
    assert code == 0, err
    assert any("科幻小说" in t for t in jload(out)["results"])


# ── 备份 ──────────────────────────────────────────────────

def test_export_import_roundtrip(cli_module, tmp_path):
    invoke(cli_module, str(tmp_path), "add", "需要备份的事实")
    backup = tmp_path / "backup.json"
    assert invoke(cli_module, str(tmp_path), "export", str(backup))[0] == 0
    assert backup.exists()

    restore = tmp_path / "restore"
    code, out, err = invoke(
        cli_module, str(restore), "import", str(backup), "--replace", "--json"
    )
    assert code == 0, err
    assert int(jload(out)["imported"]) >= 1

    code, out, err = invoke(cli_module, str(restore), "search", "需要备份", "--json")
    assert code == 0, err
    assert any("需要备份的事实" in t for t in jload(out)["results"])


# ── 显式 FeatureSettings ──────────────────────────────────

def test_self_growth_disabled_via_features(cli_module, tmp_path):
    code, out, err = invoke(cli_module, str(tmp_path), "store-interaction", "普通闲聊", "")
    assert code == 0, err
    code, out, err = invoke(cli_module, str(tmp_path), "settings", "--json")
    payload = jload(out)
    assert payload["self_growth_enabled"] is False
    assert payload["autonomous_memory_enabled"] is False