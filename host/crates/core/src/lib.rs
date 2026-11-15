//! 领域层：配置、结果类型与判定规则（论文 5.3.3），与 Python `algo/types.py`、`algo/decide.py` 语义一致。

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub preprocess: PreprocessCfg,
    pub measure: MeasureCfg,
    pub calib: CalibCfg,
    pub decide: DecideCfg,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            preprocess: PreprocessCfg::default(),
            measure: MeasureCfg::default(),
            calib: CalibCfg::default(),
            decide: DecideCfg::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PreprocessCfg {
    pub segment: SegmentCfg,
    pub morph: MorphCfg,
}

impl Default for PreprocessCfg {
    fn default() -> Self {
        Self { segment: SegmentCfg::default(), morph: MorphCfg::default() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SegmentCfg {
    pub method: String,
}

impl Default for SegmentCfg {
    fn default() -> Self {
        Self { method: "otsu".into() }
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
        Self { ksize: 5, order: "open-close".into() }
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
        Self { bfs_connectivity: 8, min_area_px: 200, width: WidthCfg::default() }
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
        Self { standard_mm: 10.0, warn_delta_mm: 0.5, alarm_delta_mm: 1.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CalibCfg {
    pub enabled: bool,
    pub mm_per_px: Option<f64>,
}

impl Default for CalibCfg {
    fn default() -> Self {
        Self { enabled: false, mm_per_px: None }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DecideCfg {
    pub broken_min_rois: usize,
    pub reflection_override: bool,
}

impl Default for DecideCfg {
    fn default() -> Self {
        Self { broken_min_rois: 2, reflection_override: true }
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
    NoRoi,
    Broken,
    TooNarrow,
    TooWide,
    WarnNarrow,
    WarnWide,
    ReflectionSuspect,
}

impl VerdictCode {
    pub fn is_defect(self) -> bool {
        !matches!(self, VerdictCode::Ok | VerdictCode::ReflectionSuspect | VerdictCode::NoRoi)
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
        (VerdictCode::TooNarrow, format!("过窄：{mm:.2} mm，低于报警线 {:.1} mm", std - alarm))
    } else if delta >= alarm {
        (VerdictCode::TooWide, format!("过宽：{mm:.2} mm，高于报警线 {:.1} mm", std + alarm))
    } else if delta <= -warn {
        (VerdictCode::WarnNarrow, format!("预警过窄：{mm:.2} mm，预警线 {:.1} mm", std - warn))
    } else if delta >= warn {
        (VerdictCode::WarnWide, format!("预警过宽：{mm:.2} mm，预警线 {:.1} mm", std + warn))
    } else {
        (VerdictCode::Ok, format!("合格：{mm:.2} mm"))
    };
    Decision { code, width_mm, message: msg }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(width: Option<f64>) -> Measure {
        Measure { area_px: 1.0, length_px: 1.0, width_px: 1.0, width_mm: width }
    }

    #[test]
    fn broken_when_multiple_rois() {
        assert_eq!(decide(&Config::default(), 2, None).code, VerdictCode::Broken);
    }

    #[test]
    fn no_roi() {
        assert_eq!(decide(&Config::default(), 0, None).code, VerdictCode::NoRoi);
    }

    #[test]
    fn ok_within_warn() {
        assert_eq!(decide(&Config::default(), 1, Some(&m(Some(10.2)))).code, VerdictCode::Ok);
    }

    #[test]
    fn too_narrow_at_alarm_line() {
        assert_eq!(decide(&Config::default(), 1, Some(&m(Some(9.0)))).code, VerdictCode::TooNarrow);
    }

    #[test]
    fn warn_wide_between_lines() {
        assert_eq!(decide(&Config::default(), 1, Some(&m(Some(10.6)))).code, VerdictCode::WarnWide);
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
