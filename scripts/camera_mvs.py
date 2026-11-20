"""海康（HIKROBOT）工业相机采集：把 MVS SDK 的 Mono8 帧按 GLUE 协议输出到 stdout。

前置条件（板端）：
  1. 已安装 MVS Linux aarch64 SDK（默认在 /opt/MVS）；
  2. 已装 udev 规则并有 video 组权限；
  3. python3 可 import SDK 的 MvImport（脚本会自动尝试常见路径，也可用 --sdk 指定）。

用法：
  python3 scripts/camera_mvs.py --device 0 --fps 10
  python3 scripts/camera_mvs.py --sdk /opt/MVS/Samples/aarch64/Python/MvImport --fps 5

注意：stdout 只输出帧协议数据；日志一律写 stderr（会被 Rust 侧继承显示）。
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="设备序号（枚举顺序）")
    ap.add_argument("--fps", type=float, default=10.0, help="采集帧率（0 表示全速）")
    ap.add_argument("--exposure-us", type=float, default=None, help="曝光时间（微秒，可选）")
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
    from ctypes import POINTER, byref, cast, c_ubyte, sizeof

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
    if args.exposure_us is not None:
        cam.MV_CC_SetFloatValue("ExposureTime", float(args.exposure_us))
    # 统一为 Mono8（黑白相机本来就是 Mono8）
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

    payload = (c_ubyte * 1)()
    st_param = __import__("MvCameraControl_class").MVCC_INTVALUE()
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
