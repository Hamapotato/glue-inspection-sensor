"""海康（HIKROBOT）工业相机采集：把 MVS SDK 的 Mono8 帧按 GLUE 协议输出到 stdout。

前置条件（板端）：
  1. 已安装 MVS Linux aarch64 SDK（默认在 /opt/MVS）；
  2. 已装 udev 规则并有 video 组权限；
  3. python3 可 import SDK 的 MvImport（脚本会自动尝试常见路径，也可用 --sdk 指定）。

用法：
  python3 scripts/camera_mvs.py --device 0 --fps 10
  # 过曝时手动曝光 + 亮度统计（推荐先用这个调）
  python3 scripts/camera_mvs.py --fps 10 --exposure-us 3000 --stats
  # 自动曝光（先连续自动，再手动锁定更稳）
  python3 scripts/camera_mvs.py --auto-exposure continuous --stats
  # 调好后用固定值跑采集
  python3 scripts/camera_mvs.py --fps 10 --exposure-us 3000

注意：
  - stdout 只输出帧协议数据；日志一律写 stderr（会被 Rust 侧继承显示）；
  - 默认关闭自动曝光/自动增益（ExposureAuto=Off, GainAuto=Off），避免亮度在帧间跳变；
  - --stats 会每 N 帧打印均值/饱和像素比例，帮助判断曝光是否合适（目标：mean 80~180，饱和 <5%）。
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

DEFAULT_SDK_PATHS = [
    os.environ.get("MVS_PYTHON_PATH", ""),
    "/opt/MVS/Samples/aarch64/Python/MvImport",
    "/opt/MVS/Samples/64/Python/MvImport",
    "/opt/MVS/Samples/32/Python/MvImport",
    "/opt/MVS/Samples/Linux/Python/MvImport",
]

AUTO_MODES = {"off": 0, "once": 1, "continuous": 2}


def load_mvimport(sdk: str | None) -> None:
    candidates = ([sdk] if sdk else []) + [p for p in DEFAULT_SDK_PATHS if p]
    for p in candidates:
        if p and Path(p).is_dir():
            sys.path.insert(0, p)
            try:
                import MvCameraControl_class  # noqa: F401

                print(f"[mvs] 使用 SDK 路径：{p}", file=sys.stderr)
                return
            except ImportError:
                sys.path.pop(0)
    raise SystemExit(
        "[mvs] 找不到 MvImport：请安装 MVS SDK 或用 --sdk 指定 MvImport 目录（含 MvCameraControl_class.py）"
    )


def print_stats(data: bytes, seq: int, width: int, height: int) -> None:
    """打印亮度统计（stderr），用于调曝光。"""
    n = width * height
    step = max(1, n // 200_000)  # 大图抽样统计，避免拖慢采集
    vals = data[::step]
    mean = sum(vals) / len(vals)
    sat = sum(1 for v in vals if v >= 250) / len(vals) * 100.0
    dark = sum(1 for v in vals if v <= 5) / len(vals) * 100.0
    print(
        f"[mvs] frame {seq:5d}  mean={mean:6.1f}  sat(>=250)={sat:5.1f}%  dark(<=5)={dark:5.1f}%",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="设备序号（枚举顺序）")
    ap.add_argument("--fps", type=float, default=10.0, help="采集帧率（0 表示全速）")
    ap.add_argument("--exposure-us", type=float, default=None, help="曝光时间（微秒）；设置后自动曝光会被关闭")
    ap.add_argument("--gain-db", type=float, default=None, help="增益（dB，可选；先曝光后增益）")
    ap.add_argument(
        "--auto-exposure",
        choices=["off", "once", "continuous"],
        default="off",
        help="自动曝光模式：off（默认，亮度稳定）/once/continuous",
    )
    ap.add_argument("--stats", action="store_true", help="每 N 帧打印亮度统计（调曝光用）")
    ap.add_argument("--stats-every", type=int, default=10, help="每 N 帧打印一次统计")
    ap.add_argument("--sdk", default=None, help="MvImport 目录")
    args = ap.parse_args()

    load_mvimport(args.sdk)
    from MvCameraControl_class import (  # type: ignore
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_PIXEL_CONVERT_PARAM,
        MV_FRAME_OUT_INFO_EX,
        MV_TRIGGER_MODE_OFF,
        MV_USB_DEVICE,
        MvCamera,
        PixelType_Gvsp_Mono8,
    )
    from ctypes import POINTER, cast, c_ubyte

    out = sys.stdout.buffer
    cam = MvCamera()
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, dev_list)
    if ret != 0:
        print(f"[mvs] 枚举失败 ret={ret}", file=sys.stderr)
        return 2
    if dev_list.nDeviceNum == 0:
        print("[mvs] 未发现 USB 相机：检查 USB3 口/线、udev 规则、是否被 MVS 客户端占用", file=sys.stderr)
        return 2
    if args.device >= dev_list.nDeviceNum:
        print(f"[mvs] --device {args.device} 越界（共 {dev_list.nDeviceNum} 台）", file=sys.stderr)
        return 2
    print(f"[mvs] 发现 {dev_list.nDeviceNum} 台相机，使用第 {args.device} 台", file=sys.stderr)

    info = cast(dev_list.pDeviceInfo[args.device], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(info)
    if ret != 0:
        print(f"[mvs] CreateHandle 失败 ret={ret}", file=sys.stderr)
        return 1
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"[mvs] OpenDevice 失败 ret={ret}（设备可能被占用）", file=sys.stderr)
        return 1
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # 曝光/增益：默认关闭自动，避免帧间亮度跳变（过曝的常见原因）
    auto = AUTO_MODES[args.auto_exposure]
    cam.MV_CC_SetEnumValue("ExposureAuto", auto)
    if args.exposure_us is not None:
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        cam.MV_CC_SetFloatValue("ExposureTime", float(args.exposure_us))
    cam.MV_CC_SetEnumValue("GainAuto", 0)
    if args.gain_db is not None:
        cam.MV_CC_SetFloatValue("Gain", float(args.gain_db))
    # 统一为 Mono8（黑白相机本来就是 Mono8）
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

    # 回读实际值，方便确认参数是否生效
    import MvCameraControl_class as mv  # type: ignore

    exp_val = mv.MVCC_FLOATVALUE()
    if cam.MV_CC_GetFloatValue("ExposureTime", exp_val) == 0:
        print(f"[mvs] ExposureTime={exp_val.fCurValue:.0f} us", file=sys.stderr)
    gain_val = mv.MVCC_FLOATVALUE()
    if cam.MV_CC_GetFloatValue("Gain", gain_val) == 0:
        print(f"[mvs] Gain={gain_val.fCurValue:.2f} dB", file=sys.stderr)
    if args.auto_exposure == "continuous":
        print("[mvs] 自动曝光=continuous（亮度会波动；调好后请用固定 --exposure-us）", file=sys.stderr)

    st_param = mv.MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", st_param)
    payload = (c_ubyte * st_param.nCurValue)()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"[mvs] StartGrabbing 失败 ret={ret}", file=sys.stderr)
        cam.MV_CC_CloseDevice()
        return 1

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    frame_info = MV_FRAME_OUT_INFO_EX()
    print("[mvs] 开始采集，按 Ctrl-C 结束", file=sys.stderr)
    seq = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret = cam.MV_CC_GetOneFrameTimeout(payload, st_param.nCurValue, frame_info, 2000)
            if ret != 0:
                print(f"[mvs] 取帧失败 ret={ret}（超时/掉线）", file=sys.stderr)
                continue
            w, h = frame_info.nWidth, frame_info.nHeight
            if frame_info.enPixelType != PixelType_Gvsp_Mono8:
                # 非 Mono8：转成 Mono8
                dst = (c_ubyte * (w * h))()
                conv = MV_CC_PIXEL_CONVERT_PARAM()
                conv.nWidth, conv.nHeight = w, h
                conv.enSrcPixelType = frame_info.enPixelType
                conv.pSrcData = cast(payload, POINTER(c_ubyte))
                conv.nSrcDataLen = frame_info.nFrameLen
                conv.enDstPixelType = PixelType_Gvsp_Mono8
                conv.pDstBuffer = dst
                conv.nDstBufferSize = w * h
                ret = cam.MV_CC_ConvertPixelType(conv)
                if ret != 0:
                    print(f"[mvs] 像素格式转换失败 ret={ret}", file=sys.stderr)
                    continue
                data = bytes(dst)
            else:
                data = bytes(payload[: frame_info.nFrameLen])
            if len(data) != w * h:
                print(f"[mvs] 帧长度异常 {len(data)} != {w}*{h}", file=sys.stderr)
                continue
            out.write(b"GLUE" + struct.pack("<III", w, h, len(data)) + data)
            out.flush()
            if args.stats and seq % max(1, args.stats_every) == 0:
                print_stats(data, seq, w, h)
            seq += 1
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    except (BrokenPipeError, OSError, KeyboardInterrupt):
        pass
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
