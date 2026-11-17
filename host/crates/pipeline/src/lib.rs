//! 检测流水线：质量门控 → 定位（启发式 / RPN）→ 裁剪 → 量测 → 判定（含反光复判）。
//!
//! 推理由 `glue-core::ObjectDetector` 抽象注入（实现位于 `glue-infer`），
//! 这样 pipeline 不依赖具体推理运行时；配置关闭时走原启发式链路。

use std::time::Instant;

use glue_core::{decide, decide_quality, Config, FrameResult, ObjectDetector, Timings};
use glue_vision::{
    locate, measure_mask, morph, otsu_threshold, quality_metrics, threshold, BinaryMask, GrayImage,
};
use rayon::prelude::*;

#[derive(Default)]
pub struct Detectors {
    pub locate: Option<Box<dyn ObjectDetector>>,
    pub reflect: Option<Box<dyn ObjectDetector>>,
}

pub struct Inspector {
    cfg: Config,
    mm_per_px: Option<f64>,
    detectors: Detectors,
}

impl Inspector {
    pub fn new(cfg: Config) -> Self {
        Self::with_detectors(cfg, Detectors::default())
    }

    pub fn with_detectors(cfg: Config, detectors: Detectors) -> Self {
        let mm_per_px = cfg.mm_per_px();
        Self {
            cfg,
            mm_per_px,
            detectors,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn measure_roi(
        &self,
        img: &GrayImage,
        mask: Option<&BinaryMask>,
        bbox: &glue_core::BBox,
    ) -> glue_core::Measure {
        let pad = (self.cfg.preprocess.morph.ksize as f64 / 2.0 + 1.0).max(1.0);
        let bb = glue_core::BBox {
            x1: (bbox.x1 - pad).max(0.0),
            y1: (bbox.y1 - pad).max(0.0),
            x2: (bbox.x2 + pad).min(img.w as f64),
            y2: (bbox.y2 + pad).min(img.h as f64),
        };
        let ksize = self.cfg.preprocess.morph.ksize;
        let order = &self.cfg.preprocess.morph.order;
        let roi_mask = match mask {
            Some(m) => morph(&m.crop(&bb), ksize, order),
            None => {
                // 推理路径没有整图掩膜：在 RoI 内做 Otsu 分割后量测
                let roi = img.crop(&bb);
                let t = otsu_threshold(&roi);
                morph(&threshold(&roi, t), ksize, order)
            }
        };
        measure_mask(&roi_mask, self.mm_per_px)
    }

    pub fn inspect(&self, img: &GrayImage) -> FrameResult {
        let t0 = Instant::now();
        let mut notes: Vec<String> = Vec::new();

        // 质量门控：太暗/过曝/模糊/低对比度直接"无法判定"，不进入定位与量测。
        let quality = quality_metrics(img);
        if let Some(decision) = decide_quality(&self.cfg, &quality) {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            return FrameResult {
                rois: Vec::new(),
                measure: None,
                decision,
                timings: Timings {
                    quality_ms: ms,
                    locate_ms: 0.0,
                    measure_ms: 0.0,
                    total_ms: ms,
                },
                notes,
            };
        }
        let quality_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 定位：优先推理（若注入），失败/空结果时回退启发式并记录 note（不静默降级）。
        let (rois, mask) = if let Some(det) = &self.detectors.locate {
            match det.detect(&img.data, img.w, img.h) {
                Ok(boxes) if !boxes.is_empty() => (boxes, None),
                Ok(_) => {
                    notes.push("RPN 未检出 RoI，回退启发式定位".into());
                    let (r, m) = locate(img, &self.cfg);
                    (r, Some(m))
                }
                Err(e) => {
                    notes.push(format!("RPN 推理失败：{e}；回退启发式定位"));
                    let (r, m) = locate(img, &self.cfg);
                    (r, Some(m))
                }
            }
        } else {
            let (r, m) = locate(img, &self.cfg);
            (r, Some(m))
        };
        let locate_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let measure = rois
            .first()
            .map(|bbox| self.measure_roi(img, mask.as_ref(), bbox));
        let measure_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let mut decision = decide(&self.cfg, rois.len(), measure.as_ref());

        // 反光复判（可选）：检测到反光框时提示调光，而不是直接判废。
        if let Some(det) = &self.detectors.reflect {
            match det.detect(&img.data, img.w, img.h) {
                Ok(boxes) if !boxes.is_empty() => {
                    decision = glue_core::decide_with_reflection(&self.cfg, decision, boxes.len());
                }
                Ok(_) => {}
                Err(e) => notes.push(format!("YOLO 反光推理失败：{e}")),
            }
        }

        FrameResult {
            rois,
            measure,
            decision,
            timings: Timings {
                quality_ms,
                locate_ms,
                measure_ms,
                total_ms: t0.elapsed().as_secs_f64() * 1000.0,
            },
            notes,
        }
    }

    pub fn inspect_batch(&self, images: &[GrayImage]) -> Vec<FrameResult> {
        images.par_iter().map(|img| self.inspect(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use glue_core::VerdictCode;

    fn strip(w: usize, h: usize, strip_w: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h, 15);
        let y0 = h / 2 - strip_w / 2;
        for y in y0..y0 + strip_w {
            for x in 10..w - 10 {
                img.set(x, y, 220);
            }
        }
        img
    }

    struct Fixed(Vec<glue_core::BBox>);
    impl ObjectDetector for Fixed {
        fn detect(&self, _gray: &[u8], _w: usize, _h: usize) -> Result<Vec<glue_core::BBox>> {
            Ok(self.0.clone())
        }
    }

    struct Failing;
    impl ObjectDetector for Failing {
        fn detect(&self, _gray: &[u8], _w: usize, _h: usize) -> Result<Vec<glue_core::BBox>> {
            anyhow::bail!("模拟推理失败")
        }
    }

    #[test]
    fn normal_strip_ok() {
        let insp = Inspector::new(Config::default());
        let r = insp.inspect(&strip(320, 120, 10));
        assert_eq!(r.decision.code, VerdictCode::Ok);
        assert_eq!(r.rois.len(), 1);
    }

    #[test]
    fn detector_roi_used_for_measure() {
        let bbox = glue_core::BBox {
            x1: 10.0,
            y1: 55.0,
            x2: 310.0,
            y2: 65.0,
        };
        let insp = Inspector::with_detectors(
            Config::default(),
            Detectors {
                locate: Some(Box::new(Fixed(vec![bbox]))),
                reflect: None,
            },
        );
        let r = insp.inspect(&strip(320, 120, 10));
        let m = r.measure.expect("measure");
        assert!((m.width_px - 10.0).abs() < 1.5, "width={}", m.width_px);
        assert!(r.notes.is_empty());
    }

    #[test]
    fn failing_detector_falls_back_with_note() {
        let insp = Inspector::with_detectors(
            Config::default(),
            Detectors {
                locate: Some(Box::new(Failing)),
                reflect: None,
            },
        );
        let r = insp.inspect(&strip(320, 120, 10));
        assert_eq!(r.decision.code, VerdictCode::Ok);
        assert_eq!(r.rois.len(), 1, "fallback heuristic should find the strip");
        assert!(
            r.notes.iter().any(|n| n.contains("RPN 推理失败")),
            "notes={:?}",
            r.notes
        );
    }

    #[test]
    fn reflection_marks_suspect() {
        let bbox = glue_core::BBox {
            x1: 60.0,
            y1: 55.0,
            x2: 120.0,
            y2: 65.0,
        };
        let insp = Inspector::with_detectors(
            Config::default(),
            Detectors {
                locate: None,
                reflect: Some(Box::new(Fixed(vec![bbox]))),
            },
        );
        let r = insp.inspect(&strip(320, 120, 10));
        assert_eq!(
            r.decision.code,
            VerdictCode::ReflectionSuspect,
            "{}",
            r.decision.message
        );
    }

    #[test]
    fn broken_when_two_rois() {
        let mut img = GrayImage::new(400, 120, 15);
        for y in 50..60 {
            for x in 20..180 {
                img.set(x, y, 220);
            }
            for x in 220..380 {
                img.set(x, y, 220);
            }
        }
        let insp = Inspector::new(Config::default());
        assert_eq!(insp.inspect(&img).decision.code, VerdictCode::Broken);
    }

    #[test]
    fn empty_dark_is_no_quality() {
        let insp = Inspector::new(Config::default());
        let r = insp.inspect(&GrayImage::new(200, 100, 10));
        assert_eq!(r.decision.code, VerdictCode::NoQuality);
    }

    #[test]
    fn textured_but_no_strip_is_no_roi() {
        let (w, h) = (200usize, 100usize);
        let mut img = GrayImage::new(w, h, 60);
        for y in 0..h {
            for x in 0..w {
                if x % 16 < 6 && y % 16 < 6 {
                    img.set(x, y, 200);
                }
            }
        }
        let insp = Inspector::new(Config::default());
        let r = insp.inspect(&img);
        assert_eq!(
            r.decision.code,
            VerdictCode::NoRoi,
            "message={}",
            r.decision.message
        );
    }

    #[test]
    fn overexposed_is_no_quality() {
        let insp = Inspector::new(Config::default());
        assert_eq!(
            insp.inspect(&GrayImage::new(200, 100, 250)).decision.code,
            VerdictCode::NoQuality
        );
    }

    #[test]
    fn blurred_strip_is_no_quality() {
        let img = strip(320, 120, 10);
        let (w, h) = (img.w, img.h);
        let r = 12usize;
        let mut blurred = GrayImage::new(w, h, 0);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                        sum += img.get(dx, dy) as u32;
                        cnt += 1;
                    }
                }
                blurred.set(x, y, (sum / cnt) as u8);
            }
        }
        let insp = Inspector::new(Config::default());
        assert_eq!(insp.inspect(&blurred).decision.code, VerdictCode::NoQuality);
    }
}
