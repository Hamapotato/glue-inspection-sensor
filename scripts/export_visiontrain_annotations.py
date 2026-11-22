"""将合成掩膜转换为 VisionTrain 可导入的图像与标注数据。

示例：
    py -3 scripts/export_visiontrain_annotations.py `
      --source outputs/synthetic/visiontrain-glue-binary-v5 `
      --out D:/VMProjects/visiontrain_glue_binary_v5_full

注意：输出是合成数据，只能用于流程验证、冒烟和预训练。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.config import load_config  # noqa: E402
from algo.visiontrain_export import export_visiontrain_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将合成胶条掩膜转换为 VisionTrain 二分类分割标注"
    )
    parser.add_argument(
        "--source",
        default="outputs/synthetic/visiontrain-glue-binary-v5",
        help="synthetic_dataset 生成的源目录",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="VisionTrain 导入目录；必须不存在或为空",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径，默认 configs/default.yaml",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=None,
        help="只导出指定 image_id；可重复使用，省略时导出全部",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = export_visiontrain_dataset(
        args.source,
        args.out,
        cfg,
        image_ids=args.image_id,
    )
    print(f"VisionTrain 数据已写入：{summary.output_dir}")
    print(f"图像数：{summary.image_count}")
    print(f"含 glue 图像数：{summary.glue_image_count}")
    print(f"background 图像数：{summary.background_image_count}")
    print(f"glue 多边形数：{summary.annotation_count}")
    print("警告：该目录来源为 synthetic，不得用于真实验收指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
