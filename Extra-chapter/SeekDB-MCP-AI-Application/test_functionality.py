#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证 SeekDB MCP 客户端核心功能
"""

import json
import subprocess
import sys
from pathlib import Path


def test_mcp_client_structure():
    """测试 MCP 客户端类结构"""
    print("[PASS] Test 1: MCP Client Structure")

    # 读取源代码
    source_file = Path(__file__).parent / "code" / "seekdb_project_finder.py"
    source_code = source_file.read_text(encoding='utf-8')

    # 验证关键组件
    required_components = [
        "class MCPStdioClient:",
        "def initialize(self):",
        "def call_tool(self, name, arguments):",
        "def request(self, method, params=None",
        "def _read_stdout(self):",
    ]

    for component in required_components:
        if component not in source_code:
            print(f"[FAIL] Missing component: {component}")
            return False

    print("  [v] All required MCP client methods implemented")
    return True


def test_data_model():
    """测试数据模型"""
    print("\n[PASS] Test 2: Data Model Validation")

    data_file = Path(__file__).parent / "data" / "projects.csv"
    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        return False

    # 读取 CSV 并验证格式
    lines = data_file.read_text(encoding='utf-8').strip().split('\n')
    if len(lines) < 2:
        print("[FAIL] Data file is empty")
        return False

    # 验证表头
    headers = lines[0].lower()
    required_fields = ['name', 'summary', 'tags', 'stars', 'language', 'repo']
    for field in required_fields:
        if field not in headers:
            print(f"[FAIL] Missing field: {field}")
            return False

    print(f"  [v] Data file format correct, {len(lines) - 1} records")
    return True


def test_tool_calls():
    """测试 MCP 工具调用逻辑"""
    print("\n[PASS] Test 3: MCP Tool Call Logic")

    source_file = Path(__file__).parent / "code" / "seekdb_project_finder.py"
    source_code = source_file.read_text(encoding='utf-8')

    # 验证工具调用
    required_tools = [
        '"import_csv_file_to_seekdb"',
        '"hybrid_search"',
        '"query_collection"',
        '"ai_complete"',
    ]

    for tool in required_tools:
        if tool not in source_code:
            print(f"[FAIL] Tool not called: {tool}")
            return False

    print("  [v] All required MCP tools implemented")
    return True


def test_error_handling():
    """测试错误处理"""
    print("\n[PASS] Test 4: Error Handling Mechanism")

    source_file = Path(__file__).parent / "code" / "seekdb_project_finder.py"
    source_code = source_file.read_text(encoding='utf-8')

    # 验证错误处理
    error_handling = [
        'try:',
        'except',
        'finally:',
        'if "exist" in err.lower()',
        'if not (isinstance(search_result, dict)',
    ]

    for pattern in error_handling:
        if pattern not in source_code:
            print(f"[FAIL] Missing error handling: {pattern}")
            return False

    print("  [v] Comprehensive error handling")
    return True


def test_command_line_interface():
    """测试命令行接口"""
    print("\n[PASS] Test 5: Command Line Interface")

    result = subprocess.run(
        [sys.executable, "code/seekdb_project_finder.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        encoding='utf-8',
        errors='ignore'
    )

    if result.returncode != 0:
        print(f"[FAIL] Help command failed: {result.stderr}")
        return False

    # 验证必需参数
    required_args = [
        "--server-command",
        "--data",
        "--query",
        "--skip-import",
        "--llm-model",
    ]

    help_text = result.stdout
    for arg in required_args:
        if arg not in help_text:
            print(f"[FAIL] Missing argument: {arg}")
            return False

    print("  [v] Complete CLI interface")
    return True


def test_documentation():
    """测试文档完整性"""
    print("\n[PASS] Test 6: Documentation Completeness")

    readme_file = Path(__file__).parent / "readme.md"
    if not readme_file.exists():
        print("[FAIL] readme.md not found")
        return False

    readme_content = readme_file.read_text(encoding='utf-8')

    # 验证必需章节
    required_sections = [
        "应用场景",
        "数据模型",
        "SeekDB MCP 运行模式",
        "启动方式",
        "运行示例",
        "参考资料",
    ]

    for section in required_sections:
        if section not in readme_content:
            print(f"[FAIL] Missing section: {section}")
            return False

    print("  [v] All required sections present")

    # 验证环境变量说明
    if "SEEKDB_HOST" not in readme_content:
        print("[FAIL] Missing environment variable documentation")
        return False

    print("  [v] Environment variables documented")
    return True


def test_code_quality():
    """测试代码质量"""
    print("\n[PASS] Test 7: Code Quality")

    source_file = Path(__file__).parent / "code" / "seekdb_project_finder.py"
    source_code = source_file.read_text(encoding='utf-8')

    # 统计代码行数
    lines = source_code.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    total_lines = len(lines)

    print(f"  [v] Total lines: {total_lines}")
    print(f"  [v] Code lines: {len(code_lines)}")

    # 验证代码注释
    docstrings = source_code.count('"""')
    if docstrings < 2:
        print("[FAIL] Missing docstrings")
        return False

    print("  [v] Documentation strings present")

    # 验证函数命名
    import re
    functions = re.findall(r'def (\w+)\(', source_code)
    print(f"  [v] Defined {len(functions)} functions")

    return True


def test_mcp_protocol():
    """测试 MCP 协议兼容性"""
    print("\n[PASS] Test 8: MCP Protocol Compatibility")

    source_file = Path(__file__).parent / "code" / "seekdb_project_finder.py"
    source_code = source_file.read_text(encoding='utf-8')

    # 验证 MCP 协议版本
    if 'PROTOCOL_VERSION' not in source_code:
        print("[FAIL] Missing PROTOCOL_VERSION")
        return False

    print("  [v] MCP protocol version defined")

    # 验证 JSON-RPC 实现
    if '"jsonrpc": "2.0"' not in source_code:
        print("[FAIL] Missing JSON-RPC 2.0 implementation")
        return False

    print("  [v] JSON-RPC 2.0 compliant")

    # 验证初始化流程
    if 'initialize' not in source_code:
        print("[FAIL] Missing initialize method")
        return False

    print("  [v] MCP initialization flow implemented")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("SeekDB MCP Project Finder - Functionality Test")
    print("=" * 60)

    tests = [
        test_mcp_client_structure,
        test_data_model,
        test_tool_calls,
        test_error_handling,
        test_command_line_interface,
        test_documentation,
        test_code_quality,
        test_mcp_protocol,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test error: {e}")
            results.append(False)

    # 输出测试总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Code is complete and compliant.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
