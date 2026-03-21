"""
LLM 多供应商支持模块的单元测试和集成测试
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 将 code 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_provider import (
    LLMProvider,
    PROVIDER_PRESETS,
    _detect_provider,
    _resolve_provider,
    create_llm,
    create_openai_client,
)


# =============================================================================
# 单元测试
# =============================================================================

class TestLLMProviderEnum(unittest.TestCase):
    """LLMProvider 枚举测试"""

    def test_provider_values(self):
        self.assertEqual(LLMProvider.MOONSHOT.value, "moonshot")
        self.assertEqual(LLMProvider.DEEPSEEK.value, "deepseek")
        self.assertEqual(LLMProvider.MINIMAX.value, "minimax")
        self.assertEqual(LLMProvider.OPENAI.value, "openai")

    def test_provider_from_string(self):
        self.assertEqual(LLMProvider("moonshot"), LLMProvider.MOONSHOT)
        self.assertEqual(LLMProvider("minimax"), LLMProvider.MINIMAX)

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError):
            LLMProvider("invalid_provider")


class TestProviderPresets(unittest.TestCase):
    """供应商预设配置测试"""

    def test_all_providers_have_presets(self):
        for provider in LLMProvider:
            self.assertIn(provider, PROVIDER_PRESETS)

    def test_preset_structure(self):
        for provider, preset in PROVIDER_PRESETS.items():
            self.assertIn("api_key_env", preset)
            self.assertIn("base_url", preset)
            self.assertIn("default_model", preset)

    def test_minimax_preset(self):
        preset = PROVIDER_PRESETS[LLMProvider.MINIMAX]
        self.assertEqual(preset["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(preset["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(preset["default_model"], "MiniMax-M2.5")

    def test_moonshot_preset(self):
        preset = PROVIDER_PRESETS[LLMProvider.MOONSHOT]
        self.assertEqual(preset["api_key_env"], "MOONSHOT_API_KEY")
        self.assertEqual(preset["base_url"], "https://api.moonshot.cn/v1")

    def test_deepseek_preset(self):
        preset = PROVIDER_PRESETS[LLMProvider.DEEPSEEK]
        self.assertEqual(preset["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(preset["base_url"], "https://api.deepseek.com")

    def test_openai_preset(self):
        preset = PROVIDER_PRESETS[LLMProvider.OPENAI]
        self.assertEqual(preset["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(preset["base_url"], "https://api.openai.com/v1")


class TestDetectProvider(unittest.TestCase):
    """自动检测供应商测试"""

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=False)
    def test_explicit_provider_env(self):
        result = _detect_provider()
        self.assertEqual(result, LLMProvider.MINIMAX)

    @patch.dict(os.environ, {"LLM_PROVIDER": "deepseek"}, clear=False)
    def test_explicit_deepseek(self):
        result = _detect_provider()
        self.assertEqual(result, LLMProvider.DEEPSEEK)

    @patch.dict(os.environ, {"LLM_PROVIDER": "INVALID"}, clear=False)
    def test_invalid_provider_falls_back(self):
        # Should fall back to auto-detection
        result = _detect_provider()
        self.assertIsInstance(result, LLMProvider)

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "",
        "MOONSHOT_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "MINIMAX_API_KEY": "test-key",
        "OPENAI_API_KEY": "",
    }, clear=False)
    def test_auto_detect_minimax(self):
        result = _detect_provider()
        self.assertEqual(result, LLMProvider.MINIMAX)

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "",
        "MOONSHOT_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "MINIMAX_API_KEY": "",
        "OPENAI_API_KEY": "",
    }, clear=False)
    def test_priority_moonshot_first(self):
        result = _detect_provider()
        self.assertEqual(result, LLMProvider.MOONSHOT)


class TestResolveProvider(unittest.TestCase):
    """供应商解析测试"""

    def test_explicit_provider(self):
        p, preset = _resolve_provider("minimax")
        self.assertEqual(p, LLMProvider.MINIMAX)
        self.assertEqual(preset["api_key_env"], "MINIMAX_API_KEY")

    def test_case_insensitive(self):
        p, _ = _resolve_provider("MiniMax")
        self.assertEqual(p, LLMProvider.MINIMAX)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _resolve_provider("not_a_provider")

    def test_none_auto_detects(self):
        p, preset = _resolve_provider(None)
        self.assertIsInstance(p, LLMProvider)
        self.assertIn("api_key_env", preset)


class TestCreateLLM(unittest.TestCase):
    """create_llm 工厂函数测试"""

    @patch("langchain_openai.ChatOpenAI")
    def test_create_with_minimax(self, mock_chat):
        mock_chat.return_value = MagicMock()
        llm = create_llm(
            provider="minimax",
            api_key="test-key",
        )
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["api_key"], "test-key")

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_model_override(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm(
            provider="minimax",
            model="MiniMax-M2.5-highspeed",
            api_key="test-key",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5-highspeed")

    @patch("langchain_openai.ChatOpenAI")
    def test_custom_base_url(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm(
            provider="minimax",
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://custom.api.com/v1")

    @patch("langchain_openai.ChatOpenAI")
    def test_temperature_clamping_minimax(self, mock_chat):
        """MiniMax 温度不能为 0，应自动调整"""
        mock_chat.return_value = MagicMock()
        create_llm(
            provider="minimax",
            temperature=0.0,
            api_key="test-key",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertGreater(call_kwargs["temperature"], 0)

    @patch("langchain_openai.ChatOpenAI")
    def test_temperature_normal_for_others(self, mock_chat):
        """其他供应商温度 0 应保持不变"""
        mock_chat.return_value = MagicMock()
        create_llm(
            provider="moonshot",
            temperature=0.0,
            api_key="test-key",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                create_llm(provider="minimax")

    @patch("langchain_openai.ChatOpenAI")
    def test_moonshot_provider(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm(provider="moonshot", api_key="test-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.moonshot.cn/v1")

    @patch("langchain_openai.ChatOpenAI")
    def test_deepseek_provider(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm(provider="deepseek", api_key="test-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.deepseek.com")

    @patch("langchain_openai.ChatOpenAI")
    def test_max_tokens_passed(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm(provider="minimax", api_key="test-key", max_tokens=4096)
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 4096)


class TestCreateOpenAIClient(unittest.TestCase):
    """create_openai_client 工厂函数测试"""

    @patch("openai.OpenAI")
    def test_create_minimax_client(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_openai_client(
            provider="minimax",
            api_key="test-key",
        )
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["api_key"], "test-key")
        self.assertEqual(model, "MiniMax-M2.5")

    @patch("openai.OpenAI")
    def test_create_moonshot_client(self, mock_openai):
        mock_openai.return_value = MagicMock()
        client, model = create_openai_client(
            provider="moonshot",
            api_key="test-key",
        )
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.moonshot.cn/v1")

    @patch("openai.OpenAI")
    def test_returns_default_model(self, mock_openai):
        mock_openai.return_value = MagicMock()
        _, model = create_openai_client(provider="deepseek", api_key="test-key")
        self.assertEqual(model, "deepseek-chat")

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                create_openai_client(provider="minimax")


class TestC8Config(unittest.TestCase):
    """C8 配置文件测试"""

    def _load_config(self):
        c8_path = os.path.join(os.path.dirname(__file__), '..', 'C8')
        sys.path.insert(0, c8_path)
        if 'config' in sys.modules:
            del sys.modules['config']
        import config
        return config.RAGConfig

    def test_config_has_provider_field(self):
        RAGConfig = self._load_config()
        config = RAGConfig()
        self.assertEqual(config.llm_provider, "")
        self.assertEqual(config.llm_model, "")

    def test_config_with_minimax(self):
        RAGConfig = self._load_config()
        config = RAGConfig(llm_provider="minimax", llm_model="MiniMax-M2.5")
        self.assertEqual(config.llm_provider, "minimax")
        self.assertEqual(config.llm_model, "MiniMax-M2.5")

    def test_config_to_dict_includes_provider(self):
        RAGConfig = self._load_config()
        config = RAGConfig(llm_provider="minimax")
        d = config.to_dict()
        self.assertIn("llm_provider", d)
        self.assertEqual(d["llm_provider"], "minimax")


class TestC9Config(unittest.TestCase):
    """C9 配置文件测试"""

    def _load_config(self):
        c9_path = os.path.join(os.path.dirname(__file__), '..', 'C9')
        sys.path.insert(0, c9_path)
        if 'config' in sys.modules:
            del sys.modules['config']
        import config
        return config.GraphRAGConfig

    def test_config_has_provider_field(self):
        GraphRAGConfig = self._load_config()
        config = GraphRAGConfig()
        self.assertEqual(config.llm_provider, "")
        self.assertEqual(config.llm_model, "")

    def test_config_with_minimax(self):
        GraphRAGConfig = self._load_config()
        config = GraphRAGConfig(llm_provider="minimax", llm_model="MiniMax-M2.5")
        self.assertEqual(config.llm_provider, "minimax")
        self.assertEqual(config.llm_model, "MiniMax-M2.5")

    def test_config_to_dict_includes_provider(self):
        GraphRAGConfig = self._load_config()
        config = GraphRAGConfig(llm_provider="deepseek")
        d = config.to_dict()
        self.assertIn("llm_provider", d)


class TestC8GenerationModuleSignature(unittest.TestCase):
    """C8 GenerationIntegrationModule 签名测试"""

    def test_c8_module_accepts_provider_param(self):
        """验证 C8 模块构造函数接受 provider 参数"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "c8_gen",
            os.path.join(os.path.dirname(__file__), '..', 'C8', 'rag_modules', 'generation_integration.py'),
        )
        source = spec.loader.get_data(spec.origin).decode()
        self.assertIn("provider", source)
        self.assertIn("create_llm", source)

    def test_c8_module_imports_llm_provider(self):
        """验证 C8 模块导入了 llm_provider"""
        path = os.path.join(os.path.dirname(__file__), '..', 'C8', 'rag_modules', 'generation_integration.py')
        with open(path) as f:
            content = f.read()
        self.assertIn("from utils.llm_provider import create_llm", content)


class TestC9GenerationModuleSignature(unittest.TestCase):
    """C9 GenerationIntegrationModule 签名测试"""

    def test_c9_module_accepts_provider_param(self):
        """验证 C9 模块构造函数接受 provider 参数"""
        path = os.path.join(os.path.dirname(__file__), '..', 'C9', 'rag_modules', 'generation_integration.py')
        with open(path) as f:
            content = f.read()
        self.assertIn("provider", content)
        self.assertIn("create_openai_client", content)

    def test_c9_module_imports_llm_provider(self):
        """验证 C9 模块导入了 llm_provider"""
        path = os.path.join(os.path.dirname(__file__), '..', 'C9', 'rag_modules', 'generation_integration.py')
        with open(path) as f:
            content = f.read()
        self.assertIn("from utils.llm_provider import create_openai_client", content)


# =============================================================================
# 集成测试（需要 MINIMAX_API_KEY 环境变量）
# =============================================================================

@unittest.skipUnless(
    os.getenv("MINIMAX_API_KEY"),
    "需要设置 MINIMAX_API_KEY 环境变量"
)
class TestMiniMaxIntegration(unittest.TestCase):
    """MiniMax API 集成测试"""

    def test_create_llm_real(self):
        """测试创建真实的 MiniMax LLM 实例"""
        llm = create_llm(provider="minimax")
        self.assertIsNotNone(llm)

    def test_create_openai_client_real(self):
        """测试创建真实的 MiniMax OpenAI 客户端"""
        client, model = create_openai_client(provider="minimax")
        self.assertIsNotNone(client)
        self.assertEqual(model, "MiniMax-M2.5")

    def test_minimax_completion(self):
        """测试 MiniMax 补全请求"""
        import re
        client, model = create_openai_client(provider="minimax")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请用一句话回答：1+1等于几？"}],
            temperature=0.1,
            max_tokens=100,
        )
        answer = response.choices[0].message.content
        self.assertIsNotNone(answer)
        # 去除 MiniMax 可能返回的思考标签
        clean = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        self.assertIn("2", clean)


if __name__ == "__main__":
    unittest.main()
