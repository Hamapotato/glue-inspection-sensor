"""扫描矩形主线的质量判定。

判定顺序：

1. 扫描窗未形成或缺少左右边缘点，判为断胶；
2. 平均胶宽超出报警线，判为过窄或过宽；
3. 局部宽度极差或标准差超限，判为宽窄不一致；
4. 中心线偏离首尾弦线超限，判为弯曲。

所有业务阈值来自 ``configs/default.yaml``。
"""

from __future__ import annotations

from .config import Config
from .types import MeasureResult, Verdict


def decide(measure: MeasureResult, cfg: Config) -> Verdict:
    """根据扫描窗量测结果给出质量结论。"""

    standard_mm = float(cfg.get("measure.width.standard_mm", 10.0))
    warn_delta_mm = float(cfg.get("measure.width.warn_delta_mm", 0.5))
    alarm_delta_mm = float(cfg.get("measure.width.alarm_delta_mm", 1.0))
    width_range_alarm_mm = float(cfg.get("measure.scan.width_range_alarm_mm"))
    width_std_alarm_mm = float(cfg.get("measure.scan.width_std_alarm_mm"))
    curve_warn_mm = float(cfg.get("measure.scan.curve_warn_mm"))
    curve_alarm_mm = float(cfg.get("measure.scan.curve_alarm_mm"))
    curve_warn_px = float(cfg.get("measure.scan.curve_warn_px"))
    curve_alarm_px = float(cfg.get("measure.scan.curve_alarm_px"))

    details: dict[str, object] = {
        "width_mm": round(float(measure.width_mm), 4),
        "standard_mm": standard_mm,
        "calibrated": bool(measure.width_mm > 0),
        "break_windows": int(measure.break_windows),
        "missing_edge_windows": int(measure.missing_edge_windows),
        "min_width_mm": round(float(measure.min_width_mm), 4),
        "max_width_mm": round(float(measure.max_width_mm), 4),
        "width_std_mm": round(float(measure.width_std_mm), 4),
        "curve_deviation_mm": round(float(measure.curve_deviation_mm), 4),
        "curve_deviation_px": round(float(measure.curve_deviation_px), 4),
    }

    if not measure.scan_windows or measure.break_windows > 0:
        if measure.scan_windows:
            message = (
                f"扫描窗缺失 {measure.break_windows} 次胶条左右边缘，判为断胶"
            )
        else:
            message = "未形成有效扫描窗，未找到可测量的胶条边缘"
        details["break_source"] = "scan_windows"
        return Verdict(
            code="BROKEN",
            severity="alarm",
            message=message,
            details=details,
        )

    code = "OK"
    severity = "none"
    message = "合格"

    if measure.width_mm > 0:
        delta_mm = float(measure.width_mm) - standard_mm
        width_range_mm = max(
            0.0,
            float(measure.max_width_mm) - float(measure.min_width_mm),
        )
        details["delta_mm"] = round(delta_mm, 4)
        details["width_range_mm"] = round(width_range_mm, 4)

        if abs(delta_mm) >= alarm_delta_mm:
            if delta_mm < 0:
                code, severity = "TOO_NARROW", "alarm"
                message = (
                    f"平均胶宽 {measure.width_mm:.2f} mm，"
                    f"低于标准值 {standard_mm:.2f} mm"
                )
            else:
                code, severity = "TOO_WIDE", "alarm"
                message = (
                    f"平均胶宽 {measure.width_mm:.2f} mm，"
                    f"高于标准值 {standard_mm:.2f} mm"
                )
        elif (
            width_range_mm >= width_range_alarm_mm
            or float(measure.width_std_mm) >= width_std_alarm_mm
        ):
            code, severity = "WIDTH_UNSTABLE", "alarm"
            message = "沿涂胶方向的胶宽波动超过允许范围"
        elif 0 < measure.min_width_mm < standard_mm - alarm_delta_mm:
            code, severity = "TOO_NARROW", "alarm"
            message = (
                f"局部胶宽 {measure.min_width_mm:.2f} mm，"
                f"低于标准值 {standard_mm:.2f} mm"
            )
        elif measure.max_width_mm > standard_mm + alarm_delta_mm:
            code, severity = "TOO_WIDE", "alarm"
            message = (
                f"局部胶宽 {measure.max_width_mm:.2f} mm，"
                f"高于标准值 {standard_mm:.2f} mm"
            )

        if code == "OK" and measure.curve_deviation_mm >= curve_alarm_mm:
            code, severity = "BENT", "alarm"
            message = (
                f"胶条中心线弯曲 {measure.curve_deviation_mm:.2f} mm，"
                "超过允许范围"
            )
        elif code == "OK" and measure.curve_deviation_mm >= curve_warn_mm:
            code, severity = "BENT", "warn"
            message = (
                f"胶条中心线弯曲 {measure.curve_deviation_mm:.2f} mm，"
                "接近报警线"
            )
        elif code == "OK" and abs(delta_mm) >= warn_delta_mm:
            if delta_mm < 0:
                code, severity = "WARN_NARROW", "warn"
                message = f"胶宽接近下限：{measure.width_mm:.2f} mm"
            else:
                code, severity = "WARN_WIDE", "warn"
                message = f"胶宽接近上限：{measure.width_mm:.2f} mm"
    else:
        details["uncalibrated"] = True
        message = "合格（未标定，宽度判定已跳过）"
        if measure.curve_deviation_px >= curve_alarm_px:
            code, severity = "BENT", "alarm"
            message = (
                f"胶条中心线弯曲 {measure.curve_deviation_px:.1f} px，"
                "超过允许范围"
            )
        elif measure.curve_deviation_px >= curve_warn_px:
            code, severity = "BENT", "warn"
            message = (
                f"胶条中心线弯曲 {measure.curve_deviation_px:.1f} px，"
                "接近报警线"
            )

    return Verdict(code=code, severity=severity, message=message, details=details)
