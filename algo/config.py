"""配置加载。

所有阈值、路径、模型参数都放在 YAML 里，代码中不允许硬编码（论文未给出的参数
必须在配置里显式出现，并在报告里注明是复现假设）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """支持点号访问的字典：cfg.get("measure.alarm_low_mm")。"""

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            return self[key]
        cur: Any = self
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def clone(self) -> "Config":
        return Config(copy.deepcopy(dict(self)))


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
