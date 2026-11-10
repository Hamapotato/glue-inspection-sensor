"""缺陷判定规则（论文 5.3.3、1.3）。

判定顺序（与论文一致）：
1. 断胶：RoI 提取后出现多个互不相连的 RoI（论文用"数量 > 1"判断）；
2. 过宽/过窄：计算宽度与数据库标准值比较，按双报警线分级；
3. 反光复判：若首轮判为缺陷但存在反光，则改判为"反光可疑"，提示用户修正参数，
   不直接判缺陷（这是论文降低误判率的关键设计，见 3.3.4、5.3.3）。

阈值全部来自配置，见 configs/default.yaml 的 ``decide`` 与 ``measure.width``。
"""

from __future__ import annotations

from .config import Config
from .types import MeasureResult, ReflectionResult, Verdict


def decide(
    measure: MeasureResult,
    n_rois: int,
    reflection: ReflectionResult,
    cfg: Config,
) -> Verdict:
    """综合量测结果、RoI 数量与反光结论，给出判定。"""
    broken_min_rois = int(cfg.get("decide.broken_min_rois", 2))
    reflection_override = bool(cfg.get("decide.reflection_override", True))
    standard_mm = float(cfg.get("measure.width.standard_mm", 10.0))
    warn_delta = float(cfg.get("measure.width.warn_delta_mm", 0.5))
    alarm_delta = float(cfg.get("measure.width.alarm_delta_mm", 1.0))

    details: dict = {
        "n_rois": int(n_rois),
        "width_mm": round(float(measure.width_mm), 4),
        "standard_mm": standard_mm,
        "calibrated": bool(measure.width_mm > 0),
    }

    code = "OK"
    severity = "none"
    message = "合格"

    if n_rois >= broken_min_rois:
        code, severity, message = "BROKEN", "alarm", f"检测到 {n_rois} 段互不相连的胶条，判为断胶"
    elif measure.width_mm > 0:
        delta = float(measure.width_mm) - standard_mm
        details["delta_mm"] = round(delta, 4)
        if abs(delta) >= alarm_delta:
            if delta < 0:
                code, severity, message = "TOO_NARROW", "alarm", f"胶宽 {measure.width_mm:.2f} mm，低于标准值 {standard_mm:.2f} mm"
            else:
                code, severity, message = "TOO_WIDE", "alarm", f"胶宽 {measure.width_mm:.2f} mm，高于标准值 {standard_mm:.2f} mm"
        elif abs(delta) >= warn_delta:
            if delta < 0:
                code, severity, message = "WARN_NARROW", "warn", f"胶宽接近下限：{measure.width_mm:.2f} mm"
            else:
                code, severity, message = "WARN_WIDE", "warn", f"胶宽接近上限：{measure.width_mm:.2f} mm"
    else:
        details["uncalibrated"] = True
        message = "合格（未标定，宽度判定已跳过）"

    if reflection_override and code != "OK" and reflection.has_reflection:
        details["pre_override_code"] = code
        code = "REFLECTION_SUSPECT"
        severity = "warn"
        message = "存在反光干扰，请修正相机参数或机器人姿态后复检"

    return Verdict(code=code, severity=severity, message=message, details=details)
