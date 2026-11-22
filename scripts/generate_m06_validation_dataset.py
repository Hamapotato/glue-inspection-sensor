"""生成固定标准胶路的 M06 验证数据集。

示例：
    py -3 scripts/generate_m06_validation_dataset.py `
      --out outputs/m06-validation/glue-m06-matched-v1-100

注意：合成数据只用于 M06 流程验证，不能作为真实指标验收数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.config import load_config  # noqa: E402
from algo.m06_validation_dataset import (  # noqa: E402
    generate_m06_validation_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 M06 同轨迹验证数据集")
    parser.add_argument(
        "--out",
        default="outputs/m06-validation/glue-m06-matched-v1-100",
        help="输出目录，必须不存在或为空",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖 m06_validation.seed",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径，默认 configs/default.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = generate_m06_validation_dataset(
        args.out,
        config,
        seed=args.seed,
    )
    print(f"数据集已写入：{summary.output_dir}")
    print(f"随机种子：{summary.seed}")
    print(f"样本数：{summary.sample_count}")
    print(f"模板图像：{summary.reference_image}")
    print(f"场景：{summary.scenario_counts}")
    print("警告：该数据集来源为 synthetic，acceptance_eligible=false。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
