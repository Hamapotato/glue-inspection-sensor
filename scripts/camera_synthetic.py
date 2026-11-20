"""合成相机：按 GLUE 帧协议向 stdout 输出灰度帧（无需任何相机，用于联调采集→检测链路）。

帧协议（小端）：
  b"GLUE" | width u32 | height u32 | len u32 | data[len]

用法：
  python3 scripts/camera_synthetic.py --width 1280 --height 960 --fps 10
  python3 scripts/camera_synthetic.py --width 640 --height 480 --fps 20 --strip-width 20
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time


def make_frame(width: int, height: int, strip_width: int, pos: int, rng: random.Random) -> bytearray:
    bg = 18
    buf = bytearray([bg]) * (width * height)
    y0 = height // 2 - strip_width // 2
    # 水平胶条，左右边界随时间移动，模拟视场内的位置变化
    x0 = 40 + pos % 60
    x1 = width - 40 - (pos * 2) % 60
    for y in range(max(0, y0), min(height, y0 + strip_width)):
        row = y * width
        for x in range(x0, x1):
            buf[row + x] = 225
    # 噪声
    for _ in range(width * height // 50):
        i = rng.randrange(width * height)
        v = buf[i] + rng.randint(-8, 8)
        buf[i] = 0 if v < 0 else (255 if v > 255 else v)
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--strip-width", type=int, default=120, help="胶条宽度（像素）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = sys.stdout.buffer
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    pos = 0
    try:
        while True:
            t0 = time.perf_counter()
            data = make_frame(args.width, args.height, args.strip_width, pos, rng)
            out.write(b"GLUE" + struct.pack("<III", args.width, args.height, len(data)) + data)
            out.flush()
            pos += 1
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    except (BrokenPipeError, OSError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
