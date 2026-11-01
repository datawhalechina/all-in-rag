"""emomem 裁剪版配置模型（vendored）。

本文件是从 Reverie 的 src/config/settings.py 裁剪而来，只保留记忆模块
（src.memory.manager）真正用到的部分：

- ``_ValidatedSettingsModel``：pydantic 基类（赋值时校验）。
- ``MemorySettings``：记忆存储/遗忘/治理配置（原样保留，含 validator）。
- ``FeatureSettings``：功能开关（原样保留，含 validator）。
- 精简的 ``_Settings`` / ``load_settings()``：仅为满足 manager 顶层
  ``from ..config.settings import ..., load_settings`` 的导入面。CLI 永远显式
  构造 ``MemorySettings``/``FeatureSettings``，不会真正走到加载路径。

路径默认值改为自包含：数据默认落在 ``~/.emomem``（可用 ``REVERIE_DATA_DIR``
环境变量覆盖），不触碰 Reverie 仓库的 data/ 目录，也不往 d:/src/skills 里写东西。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── 路径常量（自包含默认 ── 数据默认 ~/.emomem）────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(
    os.environ.get("REVERIE_DATA_DIR", str(Path.home() / ".emomem"))
).expanduser().resolve()
MEMORY_DIR = DATA_DIR / "memory"


# ── Settings 基类 ──────────────────────────────────────────

class _ValidatedSettingsModel(BaseModel):
    """设置项在运行时赋值时仍然校验（与上游语义一致）。"""

    model_config = ConfigDict(validate_assignment=True, extra="ignore")


# ── 记忆配置 ───────────────────────────────────────────────

class MemorySettings(_ValidatedSettingsModel):
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5")
    vector_quantization: Literal["float32", "int8"] = Field(default="int8")
    vector_partitioning_enabled: bool = Field(default=True)
    lancedb_path: str = Field(default=str(MEMORY_DIR / "vectors"))
    sqlite_path: str = Field(default=str(MEMORY_DIR / "metadata.db"))
    # Retention: user questionnaire default
    retention_days: int = Field(default=365 * 2, ge=365, le=365 * 3)
    interaction_capture_enabled: bool = Field(default=True)
    short_term_capture_probability: float = Field(default=1.0, ge=0.01, le=1.0)
    # Forgetting
    forgetting_enabled: bool = Field(default=True)
    long_term_forgetting_enabled: bool = Field(default=True)
    short_term_forgetting_enabled: bool = Field(default=True)
    long_term_forget_days: int = Field(default=90, ge=60, le=365)
    short_term_forget_days: int = Field(default=7, ge=1, le=59)
    long_term_forget_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    short_term_forget_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    # Legacy fallback kept for older config files.
    forget_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    # Retrieval decay. Defaults to a 90-day half-life before layer/salience scaling.
    decay_lambda: float = Field(default=0.0077, ge=0.0001, le=0.10)
    recall_reinforcement_alpha: float = Field(default=0.12, ge=0.0, le=0.50)
    minimum_retrieval_retention: float = Field(default=0.05, ge=0.0, le=0.95)
    # Misremembering (1%-10%, scaled by memory age inside the eligible window)
    misremembering_enabled: bool = Field(default=False)
    misremember_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    long_term_misremembering_enabled: bool = Field(default=True)
    short_term_misremembering_enabled: bool = Field(default=True)
    long_term_misremember_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    short_term_misremember_probability: float = Field(default=0.05, ge=0.01, le=0.10)
    # Lifecycle governance (P0)
    memory_lifecycle_governance_enabled: bool = Field(default=True)
    memory_long_budget_chars: int = Field(default=250_000, ge=50_000, le=2_000_000)

    @field_validator("retention_days")
    @classmethod
    def _retention_is_explicit_year_choice(cls, value: int) -> int:
        if value not in {365, 365 * 2, 365 * 3}:
            raise ValueError("retention_days must be exactly 1, 2, or 3 years")
        return value


# ── 功能开关 ───────────────────────────────────────────────

class FeatureSettings(_ValidatedSettingsModel):
    """记忆链路用到的功能开关（裁剪版，仅保留 manager 读取的字段）。"""

    autonomous_memory_enabled: bool = Field(default=True)
    autonomous_memory_llm_enabled: bool = Field(default=False)
    self_growth_enabled: bool = Field(default=True)
    self_growth_from_memory_enabled: bool = Field(default=True)
    self_growth_from_web_enabled: bool = Field(default=False)
    self_growth_interval_days: int = Field(default=90, ge=30, le=365)
    hypa_compression_enabled: bool = Field(default=False)
    neko_import_enabled: bool = Field(default=False)
    neko_import_source_dir: str = Field(default="", max_length=2048)

    @model_validator(mode="after")
    def _validate_dependent_ranges(self) -> "FeatureSettings":
        if self.self_growth_interval_days < 30 or self.self_growth_interval_days > 365:
            raise ValueError("self_growth_interval_days out of range")
        return self


# ── 极简加载面（CLI 不会真正走到这）────────────────────────

class _Settings(_ValidatedSettingsModel):
    """裁剪版顶层容器：只有记忆模块需要的两个字段。"""

    memory: MemorySettings = Field(default_factory=MemorySettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)


_settings: _Settings | None = None


def load_settings() -> _Settings:
    """返回默认配置（不做磁盘 I/O，不回退到 data/config.json）。

    上游版本的 load_settings 会读 data/config.json 并创建一堆目录；
    此处为了自包含与零副作用，直接返回内存默认值。
    """
    global _settings
    if _settings is None:
        _settings = _Settings()
    return _settings