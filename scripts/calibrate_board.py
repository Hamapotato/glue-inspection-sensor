"""棋盘格标定：读一组标定图，输出内参、畸变、重投影误差与毫米/像素尺度。

用法：
    py -3 scripts/calibrate_board.py --images calib/ --cols 9 --rows 6 --square-mm 5 --out calib.json

输出对齐论文表 4-2 的统计方式（逐视图 RMS 重投影误差）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.calib import calibrate_from_images  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="棋盘格标定")
    parser.add_argument("--images", nargs="+", required=True, help="图像目录或文件")
    parser.add_argument("--cols", type=int, default=9, help="内角点列数")
    parser.add_argument("--rows", type=int, default=6, help="内角点行数")
    parser.add_argument("--square-mm", type=float, default=5.0, help="方格边长（mm）")
    parser.add_argument("--out", default=None, help="标定结果输出 JSON")
    args = parser.parse_args()

    files = collect(args.images)
    if not files:
        print("未找到图像文件", file=sys.stderr)
        return 1

    calib = calibrate_from_images(files, pattern=(args.cols, args.rows), square_mm=args.square_mm)

    print(f"参与标定的图像：{len(files)} 张")
    print(f"重投影误差 RMS：{calib.rms:.4f} px")
    print(f"内参 K：\n{calib.K}")
    print(f"畸变系数：{calib.dist.ravel() if calib.dist is not None else None}")
    print(f"毫米/像素：{calib.mm_per_px:.6f}" if calib.mm_per_px else "毫米/像素：未估计")
    if calib.per_view_errors:
        print("逐视图重投影误差：")
        for i, err in enumerate(calib.per_view_errors):
            print(f"  view {i:02d}: {err:.4f} px")

    if args.out:
        calib.save(args.out)
        print(f"已保存 -> {args.out}")
    else:
        print(json.dumps(calib.to_dict(), ensure_ascii=False, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
