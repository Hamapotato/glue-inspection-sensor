//! 领域层：配置、结果类型与判定规则（论文 5.3.3），与 Python `algo/types.py`、`algo/decide.py` 语义一致。

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub preprocess: PreprocessCfg,
    pub measure: MeasureCfg,
    pub calib: CalibCfg,
    pub decide: DecideCfg,
    pub quality: QualityCfg,
    pub locate: LocateCfg,
    pub reflect: ReflectCfg,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PreprocessCfg {
    pub segment: SegmentCfg,
    pub morph: MorphCfg,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SegmentCfg {
    pub method: String,
}

impl Default for SegmentCfg {
    fn default() -> Self {
        Self {
            method: "otsu".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MorphCfg {
    pub ksize: u32,
    pub order: String,
}

impl Default for MorphCfg {
    fn default() -> Self {
        Self {
            ksize: 5,
            order: "open-close".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MeasureCfg {
    pub bfs_connectivity: u8,
    pub min_area_px: u32,
    pub width: WidthCfg,
}

impl Default for MeasureCfg {
    fn default() -> Self {
        Self {
            bfs_connectivity: 8,
            min_area_px: 200,
            width: WidthCfg::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WidthCfg {
    pub standard_mm: f64,
    pub warn_delta_mm: f64,
    pub alarm_delta_mm: f64,
}

impl Default for WidthCfg {
    fn default() -> Self {
        Self {
            standard_mm: 10.0,
            warn_delta_mm: 0.5,
            alarm_delta_mm: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CalibCfg {
    pub enabled: bool,
    pub mm_per_px: Option<f64>,
}

/// 推理定位（RPN）配置：默认关闭，打开后由 `glue-infer` 提供 RoI。
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LocateCfg {
    pub enabled: bool,
    #[serde(rename = "type")]
    pub kind: String,
    pub weights: Option<String>,
    pub input_size: usize,
    pub score_thr: f64,
    pub pre_topk: usize,
    pub post_topk: usize,
    pub min_size: f64,
    /// K-means 先验框（w,h）；训练与推理必须一致，None 表示用默认 9 个 anchor。
    pub anchors_wh: Option<Vec<[f64; 2]>>,
}

impl Default for LocateCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: "heuristic".into(),
            weights: None,
            input_size: 320,
            score_thr: 0.5,
            pre_topk: 1000,
            post_topk: 300,
            min_size: 8.0,
            anchors_wh: None,
        }
    }
}

/// 反光检测（YOLO）配置：默认关闭。
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ReflectCfg {
    pub enabled: bool,
    #[serde(rename = "type")]
    pub kind: String,
    pub weights: Option<String>,
    pub input_size: usize,
    pub score_thr: f64,
}

impl Default for ReflectCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: "null_reflector".into(),
            weights: None,
            input_size: 416,
            score_thr: 0.5,
        }
    }
}

/// 检测器抽象：由 `glue-infer` 实现（RPN 定位 / YOLO 反光）。
/// 输入为灰度图原始字节（行优先、0..255），返回像素坐标框。
pub trait ObjectDetector: Send + Sync {
    fn detect(&self, gray: &[u8], w: usize, h: usize) -> anyhow::Result<Vec<BBox>>;
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DecideCfg {
    pub broken_min_rois: usize,
    pub reflection_override: bool,
}

impl Default for DecideCfg {
    fn default() -> Self {
        Self {
            broken_min_rois: 2,
            reflection_override: true,
        }
    }
}

/// 图像质量门控（论文未设该环节，属于工程增强；阈值走配置）。
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QualityCfg {
    pub enabled: bool,
    pub min_mean: f64,
    pub max_mean: f64,
    pub max_overexposed_ratio: f64,
    pub min_contrast: f64,
    pub min_sharpness: f64,
}

impl Default for QualityCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            min_mean: 20.0,
            max_mean: 235.0,
            max_overexposed_ratio: 0.30,
            min_contrast: 25.0,
            min_sharpness: 10.0,
        }
    }
}

impl Config {
    pub fn from_yaml(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Ok(serde_yaml::from_str(&text)?)
    }

    pub fn mm_per_px(&self) -> Option<f64> {
        self.calib.mm_per_px
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl BBox {
    pub fn width(&self) -> f64 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f64 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn to_array(self) -> [f64; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictCode {
    Ok,
    NoQuality,
    NoRoi,
    Broken,
    TooNarrow,
    TooWide,
    WarnNarrow,
    WarnWide,
    ReflectionSuspect,
}

impl VerdictCode {
    /// 与 Python 端一致的字符串编码（OK / NO_ROI / BROKEN / ...）。
    pub fn as_str(self) -> &'static str {
        match self {
            VerdictCode::Ok => "OK",
            VerdictCode::NoQuality => "NO_QUALITY",
            VerdictCode::NoRoi => "NO_ROI",
            VerdictCode::Broken => "BROKEN",
            VerdictCode::TooNarrow => "TOO_NARROW",
            VerdictCode::TooWide => "TOO_WIDE",
            VerdictCode::WarnNarrow => "WARN_NARROW",
            VerdictCode::WarnWide => "WARN_WIDE",
            VerdictCode::ReflectionSuspect => "REFLECTION_SUSPECT",
        }
    }

    pub fn is_defect(self) -> bool {
        !matches!(
            self,
            VerdictCode::Ok
                | VerdictCode::ReflectionSuspect
                | VerdictCode::NoRoi
                | VerdictCode::NoQuality
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measure {
    pub area_px: f64,
    pub length_px: f64,
    pub width_px: f64,
    pub width_mm: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub code: VerdictCode,
    pub width_mm: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timings {
    #[serde(default)]
    pub quality_ms: f64,
    pub locate_ms: f64,
    pub measure_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameResult {
    pub rois: Vec<BBox>,
    pub measure: Option<Measure>,
    pub decision: Decision,
    pub timings: Timings,
    /// 降级/推理失败等说明（空表示无异常），用于满足"不静默降级"。
    #[serde(default)]
    pub notes: Vec<String>,
}

/// 图像质量指标（由 `glue-vision::quality_metrics` 计算），供判定使用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub mean: f64,
    pub p05: f64,
    pub p95: f64,
    pub contrast: f64,
    pub overexposed_ratio: f64,
    pub sharpness: f64,
}

/// 质量门控判定：返回 Some(Decision) 表示"无法判定"，应跳过后续检测。
pub fn decide_quality(cfg: &Config, m: &QualityMetrics) -> Option<Decision> {
    let q = &cfg.quality;
    if !q.enabled {
        return None;
    }
    if m.mean < q.min_mean {
        return Some(Decision {
            code: VerdictCode::NoQuality,
            width_mm: None,
            message: format!(
                "无法判定：图像过暗（均值 {:.0} < {:.0}）",
                m.mean, q.min_mean
            ),
        });
    }
    if m.overexposed_ratio > q.max_overexposed_ratio || m.mean > q.max_mean {
        return Some(Decision {
            code: VerdictCode::NoQuality,
            width_mm: None,
            message: format!(
                "无法判定：图像过曝（饱和像素 {:.0}% > {:.0}%）",
                m.overexposed_ratio * 100.0,
                q.max_overexposed_ratio * 100.0
            ),
        });
    }
    if m.contrast < q.min_contrast {
        return Some(Decision {
            code: VerdictCode::NoQuality,
            width_mm: None,
            message: format!(
                "无法判定：对比度过低（{:.0} < {:.0}）",
                m.contrast, q.min_contrast
            ),
        });
    }
    if m.sharpness < q.min_sharpness {
        return Some(Decision {
            code: VerdictCode::NoQuality,
            width_mm: None,
            message: format!(
                "无法判定：图像模糊（清晰度 {:.1} < {:.1}）",
                m.sharpness, q.min_sharpness
            ),
        });
    }
    None
}

/// 反光复判：检测到反光框且开启 `reflection_override` 时，把可复判的缺陷改为"反光可疑"，
/// 提示调整光源/参数，而不是直接判废（论文 5.3.3）。断胶不覆盖（需人工确认）。
pub fn decide_with_reflection(cfg: &Config, mut d: Decision, reflection_boxes: usize) -> Decision {
    if reflection_boxes == 0 || !cfg.decide.reflection_override {
        return d;
    }
    if matches!(
        d.code,
        VerdictCode::TooNarrow
            | VerdictCode::TooWide
            | VerdictCode::WarnNarrow
            | VerdictCode::WarnWide
            | VerdictCode::Ok
    ) {
        let prev = d.code.as_str();
        d.code = VerdictCode::ReflectionSuspect;
        d.message = format!(
            "检测到反光干扰（{} 处），原判定 {}；建议调整光源/曝光后复判",
            reflection_boxes, prev
        );
    }
    d
}

pub fn decide(cfg: &Config, roi_count: usize, measure: Option<&Measure>) -> Decision {
    let width_mm = measure.and_then(|m| m.width_mm);

    if roi_count == 0 {
        return Decision {
            code: VerdictCode::NoRoi,
            width_mm: None,
            message: "未检测到胶条区域".into(),
        };
    }

    if roi_count >= cfg.decide.broken_min_rois {
        return Decision {
            code: VerdictCode::Broken,
            width_mm,
            message: format!("断胶：检测到 {roi_count} 个互不相连的 RoI"),
        };
    }

    let Some(mm) = width_mm else {
        return Decision {
            code: VerdictCode::Ok,
            width_mm: None,
            message: "未标定，无法换算毫米，默认合格".into(),
        };
    };

    let std = cfg.measure.width.standard_mm;
    let delta = mm - std;
    let alarm = cfg.measure.width.alarm_delta_mm;
    let warn = cfg.measure.width.warn_delta_mm;

    let (code, msg) = if delta <= -alarm {
        (
            VerdictCode::TooNarrow,
            format!("过窄：{mm:.2} mm，低于报警线 {:.1} mm", std - alarm),
        )
    } else if delta >= alarm {
        (
            VerdictCode::TooWide,
            format!("过宽：{mm:.2} mm，高于报警线 {:.1} mm", std + alarm),
        )
    } else if delta <= -warn {
        (
            VerdictCode::WarnNarrow,
            format!("预警过窄：{mm:.2} mm，预警线 {:.1} mm", std - warn),
        )
    } else if delta >= warn {
        (
            VerdictCode::WarnWide,
            format!("预警过宽：{mm:.2} mm，预警线 {:.1} mm", std + warn),
        )
    } else {
        (VerdictCode::Ok, format!("合格：{mm:.2} mm"))
    };
    Decision {
        code,
        width_mm,
        message: msg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(width: Option<f64>) -> Measure {
        Measure {
            area_px: 1.0,
            length_px: 1.0,
            width_px: 1.0,
            width_mm: width,
        }
    }

    #[test]
    fn broken_when_multiple_rois() {
        assert_eq!(
            decide(&Config::default(), 2, None).code,
            VerdictCode::Broken
        );
    }

    #[test]
    fn no_roi() {
        assert_eq!(decide(&Config::default(), 0, None).code, VerdictCode::NoRoi);
    }

    #[test]
    fn ok_within_warn() {
        assert_eq!(
            decide(&Config::default(), 1, Some(&m(Some(10.2)))).code,
            VerdictCode::Ok
        );
    }

    #[test]
    fn too_narrow_at_alarm_line() {
        assert_eq!(
            decide(&Config::default(), 1, Some(&m(Some(9.0)))).code,
            VerdictCode::TooNarrow
        );
    }

    #[test]
    fn warn_wide_between_lines() {
        assert_eq!(
            decide(&Config::default(), 1, Some(&m(Some(10.6)))).code,
            VerdictCode::WarnWide
        );
    }

    fn q(mean: f64, contrast: f64, sharpness: f64, over: f64) -> QualityMetrics {
        QualityMetrics {
            mean,
            p05: 10.0,
            p95: 10.0 + contrast,
            contrast,
            overexposed_ratio: over,
            sharpness,
        }
    }

    #[test]
    fn quality_dark_rejected() {
        let cfg = Config::default();
        let d = decide_quality(&cfg, &q(5.0, 100.0, 100.0, 0.0)).expect("dark");
        assert_eq!(d.code, VerdictCode::NoQuality);
        assert!(d.message.contains("过暗"));
    }

    #[test]
    fn quality_overexposed_rejected() {
        let cfg = Config::default();
        let d = decide_quality(&cfg, &q(200.0, 100.0, 100.0, 0.6)).expect("over");
        assert_eq!(d.code, VerdictCode::NoQuality);
    }

    #[test]
    fn quality_blur_rejected() {
        let cfg = Config::default();
        let d = decide_quality(&cfg, &q(120.0, 100.0, 2.0, 0.0)).expect("blur");
        assert!(d.message.contains("模糊"));
    }

    #[test]
    fn quality_normal_passes() {
        let cfg = Config::default();
        assert!(decide_quality(&cfg, &q(120.0, 100.0, 500.0, 0.0)).is_none());
    }

    #[test]
    fn quality_disabled_passes() {
        let mut cfg = Config::default();
        cfg.quality.enabled = false;
        assert!(decide_quality(&cfg, &q(0.0, 0.0, 0.0, 1.0)).is_none());
    }

    #[test]
    fn reflection_overrides_width_defect() {
        let cfg = Config::default();
        let base = Decision {
            code: VerdictCode::TooNarrow,
            width_mm: Some(9.0),
            message: "过窄".into(),
        };
        let d = decide_with_reflection(&cfg, base, 2);
        assert_eq!(d.code, VerdictCode::ReflectionSuspect);
        assert!(d.message.contains("反光"));
    }

    #[test]
    fn reflection_keeps_broken() {
        let cfg = Config::default();
        let base = Decision {
            code: VerdictCode::Broken,
            width_mm: None,
            message: "断胶".into(),
        };
        assert_eq!(
            decide_with_reflection(&cfg, base, 1).code,
            VerdictCode::Broken
        );
    }

    #[test]
    fn locate_and_reflect_default_off() {
        let c = Config::default();
        assert!(!c.locate.enabled && !c.reflect.enabled);
    }

    #[test]
    fn parse_repo_config_if_present() {
        let path = Path::new("../../../configs/default.yaml");
        if path.exists() {
            let c = Config::from_yaml(path).expect("parse config");
            assert!(c.measure.width.standard_mm > 0.0);
        }
    }
}
