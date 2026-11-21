"""AidLite 推理服务（供 Rust `aidlite` 后端通过 stdin/stdout 调用）。

为什么用子进程：AidLite 只有 C++（mangled）与 Python 接口，没有稳定的 C ABI；
Rust 侧通过常驻 Python 进程调用，避免每帧启动开销。

协议（小端）：
  请求：b"AIDL" + u32 len + f32 payload（len = 320*320*4，灰度 letterbox 后 /255）
  响应：b"RSLT" + u32 cls_len + u32 reg_len + cls(f32) + reg(f32)
  错误：b"ERR!" + u32 msg_len + utf8 文本

用法（板端，真实模型）：
  python3 scripts/aidlite_server.py --model outputs/demo/onnx/rpn-static.onnx --accel cpu
用法（无 AidLite 的机器上测试协议）：
  python3 scripts/aidlite_server.py --model x --mock

stdout 只输出协议帧，日志写 stderr。
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

SIZE = 320
CLS_N = 3600
REG_N = 3600 * 4
MAGIC_REQ = b"AIDL"
MAGIC_RESP = b"RSLT"
MAGIC_ERR = b"ERR!"


def write_err(out, msg: str) -> None:
    data = msg.encode("utf-8", "replace")
    out.write(MAGIC_ERR + struct.pack("<I", len(data)) + data)
    out.flush()


def read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class AidliteModel:
    def __init__(self, model: str, accel: str) -> None:
        import aidlite  # type: ignore

        aidlite.set_log_level(aidlite.LogLevel.WARNING)
        cfg = aidlite.Config.create_instance()
        cfg.accelerate_type = (
            aidlite.AccelerateType.TYPE_GPU if accel == "gpu" else aidlite.AccelerateType.TYPE_CPU
        )
        cfg.implement_type = aidlite.ImplementType.TYPE_LOCAL
        cfg.framework_type = aidlite.FrameworkType.TYPE_ONNX
        self._aidlite = aidlite
        self._model = aidlite.Model.create_instance(model)
        self._interp = aidlite.InterpreterBuilder.build_interpreter_from_model_and_config(
            self._model, cfg
        )
        if self._interp is None:
            raise RuntimeError("build_interpreter returned None")
        self._interp.init()
        ret = self._interp.load_model()
        if ret != 0:
            raise RuntimeError(f"load_model failed ret={ret}")
        info = self._interp.get_input_tensor_info()
        self._in_name = info[0][0].name if info and info[0] else "image"
        out_info = self._interp.get_output_tensor_info()
        names = [t.name for g in out_info for t in g]
        self._aidlite = aidlite
        self._names = names
        print(f"[aidlite-server] model loaded, inputs={[t.name for g in info for t in g]} outputs={names}", file=sys.stderr)

    def run(self, payload: bytes):
        interp = self._interp
        ret = interp.set_input_tensor(interp.input_tensor_name_to_index(self._in_name), payload)
        if ret != 0:
            raise RuntimeError(f"set_input_tensor failed ret={ret}")
        ret = interp.invoke()
        if ret != 0:
            raise RuntimeError(f"invoke failed ret={ret}")
        outs = []
        for i, name in enumerate(self._names[:2]):
            idx = interp.output_tensor_name_to_index(name)
            arr = np.asarray(interp.get_output_tensor(idx, self._aidlite.DataType.TYPE_FLOAT32), dtype=np.float32)
            outs.append(arr.reshape(-1))
        cls = outs[0][:CLS_N] if len(outs[0]) >= CLS_N else outs[0]
        reg = outs[1][:REG_N] if len(outs[1]) >= REG_N else outs[1]
        return cls, reg

    def close(self):
        try:
            self._interp.destroy()
        except Exception:  # noqa: BLE001
            pass


def mock_outputs() -> tuple[np.ndarray, np.ndarray]:
    cls = np.full(CLS_N, -20.0, dtype=np.float32)
    reg = np.zeros(REG_N, dtype=np.float32)
    cls[100] = 20.0  # 一个高分 anchor，便于协议测试
    return cls, reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--accel", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--mock", action="store_true", help="不加载模型，返回固定输出（协议测试用）")
    args = ap.parse_args()

    out = sys.stdout.buffer
    inp = sys.stdin.buffer

    model = None
    if not args.mock:
        try:
            model = AidliteModel(args.model, args.accel)
        except Exception as e:  # noqa: BLE001
            write_err(out, f"init failed: {e}")
            return 1

    while True:
        head = read_exact(inp, 8)
        if head is None:
            break
        if head[:4] != MAGIC_REQ:
            write_err(out, f"bad magic {head[:4]!r}")
            return 1
        (size,) = struct.unpack("<I", head[4:])
        if size != SIZE * SIZE * 4:
            write_err(out, f"bad payload size {size}")
            return 1
        payload = read_exact(inp, size)
        if payload is None:
            break
        try:
            if model is None:
                cls, reg = mock_outputs()
            else:
                cls, reg = model.run(payload)
        except Exception as e:  # noqa: BLE001
            write_err(out, f"infer failed: {e}")
            return 1
        out.write(MAGIC_RESP + struct.pack("<II", len(cls), len(reg)) + cls.tobytes() + reg.tobytes())
        out.flush()

    if model is not None:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
