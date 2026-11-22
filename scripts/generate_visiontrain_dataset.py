"""生成 VisionTrain 涂胶分割训练用的合成数据集。

示例：
    py -3 scripts/generate_visiontrain_dataset.py `
      --out outputs/synthetic/visiontrain-glue-binary-v5-500 `
      --samples 500 `
      --seed 20261121

注意：合成数据只用于流程验证、冒烟和预训练，不能作为真实指标验收数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.config import load_config  # noqa: E402
from algo.synthetic_dataset import generate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 VisionTrain 涂胶合成分割数据集")
    parser.add_argument(
        "--out",
        default="outputs/synthetic/visiontrain-glue-binary-v5",
        help="输出目录，必须不存在或为空",
    )
    parser.add_argument("--samples", type=int, default=None, help="覆盖配置中的样本数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径，默认 configs/default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = generate_dataset(
        args.out,
        cfg,
        samples=args.samples,
        seed=args.seed,
    )
    print(f"数据集已写入：{summary.output_dir}")
    print(f"随机种子：{summary.seed}")
    print(f"样本数：{summary.sample_count}")
    print(f"划分：{summary.split_counts}")
    print(f"场景：{summary.scenario_counts}")
    print("警告：该数据集来源为 synthetic，acceptance_eligible=false。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
