"""
LLM 多供应商支持模块

支持通过环境变量或参数切换不同的 LLM 供应商，包括：
- Moonshot (Kimi)
- DeepSeek
- MiniMax
- OpenAI
- 任意 OpenAI 兼容的供应商

使用方法：
    # 方式一：通过环境变量 LLM_PROVIDER 自动选择
    from utils.llm_provider import create_llm, create_openai_client

    llm = create_llm()  # 返回 LangChain ChatOpenAI 实例
    client = create_openai_client()  # 返回 openai.OpenAI 实例

    # 方式二：显式指定供应商
    llm = create_llm(provider="minimax", model="MiniMax-M2.5")
    client = create_openai_client(provider="minimax")
"""

import os
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """支持的 LLM 供应商"""
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    MINIMAX = "minimax"
    OPENAI = "openai"


# 各供应商的预设配置
PROVIDER_PRESETS = {
    LLMProvider.MOONSHOT: {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2-0711-preview",
    },
    LLMProvider.DEEPSEEK: {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    LLMProvider.MINIMAX: {
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.5",
    },
    LLMProvider.OPENAI: {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}


def _detect_provider() -> LLMProvider:
    """
    根据环境变量自动检测可用的 LLM 供应商。

    优先级：LLM_PROVIDER 环境变量 > 按 API Key 自动检测
    """
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider:
        try:
            return LLMProvider(env_provider)
        except ValueError:
            logger.warning(f"未知的 LLM_PROVIDER: {env_provider}，将自动检测")

    for provider in [LLMProvider.MOONSHOT, LLMProvider.DEEPSEEK,
                     LLMProvider.MINIMAX, LLMProvider.OPENAI]:
        preset = PROVIDER_PRESETS[provider]
        if os.getenv(preset["api_key_env"]):
            logger.info(f"自动检测到供应商: {provider.value}")
            return provider

    return LLMProvider.MOONSHOT


def _resolve_provider(
    provider: Optional[str] = None,
) -> tuple:
    """解析供应商配置，返回 (LLMProvider, preset_dict)。"""
    if provider:
        try:
            p = LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(
                f"不支持的供应商: {provider}，"
                f"可选值: {[e.value for e in LLMProvider]}"
            )
    else:
        p = _detect_provider()

    preset = PROVIDER_PRESETS[p]
    return p, preset


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    创建 LangChain ChatOpenAI 实例。

    所有受支持的供应商均通过 OpenAI 兼容接口接入。

    Args:
        provider: 供应商名称（moonshot / deepseek / minimax / openai）
        model: 模型名称，为空则使用供应商默认模型
        temperature: 生成温度
        max_tokens: 最大 token 数
        api_key: API Key，为空则从环境变量读取
        base_url: API 地址，为空则使用供应商默认值

    Returns:
        langchain_openai.ChatOpenAI 实例
    """
    from langchain_openai import ChatOpenAI

    p, preset = _resolve_provider(provider)

    resolved_api_key = api_key or os.getenv(preset["api_key_env"], "")
    if not resolved_api_key:
        raise ValueError(
            f"请设置 {preset['api_key_env']} 环境变量，"
            f"或通过 api_key 参数传入"
        )

    resolved_model = model or preset["default_model"]
    resolved_base_url = base_url or preset["base_url"]

    # MiniMax 温度范围为 (0, 1]，确保不传 0
    if p == LLMProvider.MINIMAX and temperature <= 0:
        temperature = 0.01

    logger.info(
        f"创建 LLM: provider={p.value}, model={resolved_model}, "
        f"base_url={resolved_base_url}"
    )

    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )


def create_openai_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    创建 openai.OpenAI 客户端实例。

    适用于需要直接使用 OpenAI SDK 的场景（如 C9 生成模块）。

    Args:
        provider: 供应商名称
        api_key: API Key
        base_url: API 地址

    Returns:
        (openai.OpenAI 客户端, 默认模型名称)
    """
    from openai import OpenAI

    p, preset = _resolve_provider(provider)

    resolved_api_key = api_key or os.getenv(preset["api_key_env"], "")
    if not resolved_api_key:
        raise ValueError(
            f"请设置 {preset['api_key_env']} 环境变量，"
            f"或通过 api_key 参数传入"
        )

    resolved_base_url = base_url or preset["base_url"]

    logger.info(
        f"创建 OpenAI 客户端: provider={p.value}, "
        f"base_url={resolved_base_url}"
    )

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )

    return client, preset["default_model"]
