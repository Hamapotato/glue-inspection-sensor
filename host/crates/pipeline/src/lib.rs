//! 检测流水线：定位 → 裁剪 → 量测 → 判定，与 Python `algo/pipeline.py` 分段一致。

use std::time::Instant;

use glue_core::{decide, decide_quality, Config, FrameResult, Timings};
use glue_vision::{locate, measure_mask, morph, quality_metrics, GrayImage};
use rayon::prelude::*;

pub struct Inspector {
    cfg: Config,
    mm_per_px: Option<f64>,
}

impl Inspector {
    pub fn new(cfg: Config) -> Self {
        let mm_per_px = cfg.mm_per_px();
        Self { cfg, mm_per_px }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn inspect(&self, img: &GrayImage) -> FrameResult {
        let t0 = Instant::now();

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
            };
        }
        let quality_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (rois, mask) = locate(img, &self.cfg);
        let locate_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let pad = (self.cfg.preprocess.morph.ksize as f64 / 2.0 + 1.0).max(1.0);
        let measure = rois.first().map(|bbox| {
            let bb = glue_core::BBox {
                x1: (bbox.x1 - pad).max(0.0),
                y1: (bbox.y1 - pad).max(0.0),
                x2: (bbox.x2 + pad).min(img.w as f64),
                y2: (bbox.y2 + pad).min(img.h as f64),
            };
            let roi_mask = mask.crop(&bb);
            let roi_mask = morph(
                &roi_mask,
                self.cfg.preprocess.morph.ksize,
                &self.cfg.preprocess.morph.order,
            );
            measure_mask(&roi_mask, self.mm_per_px)
        });
        let measure_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let decision = decide(&self.cfg, rois.len(), measure.as_ref());
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
        }
    }

    pub fn inspect_batch(&self, images: &[GrayImage]) -> Vec<FrameResult> {
        images.par_iter().map(|img| self.inspect(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn normal_strip_ok() {
        let insp = Inspector::new(Config::default());
        let r = insp.inspect(&strip(320, 120, 10));
        assert_eq!(r.decision.code, VerdictCode::Ok);
        assert_eq!(r.rois.len(), 1);
    }

    #[test]
    fn two_strips_broken() {
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
        // 有对比度与清晰度（棋盘格），但没有胶条：每个方格面积小于 min_area_px → NO_ROI，
        // 说明质量门控放行后仍会正常走定位流程。
        let (w, h) = (200usize, 100usize);
        let mut img = GrayImage::new(w, h, 60);
        for y in 0..h {
            for x in 0..w {
                // 6×6 亮点、间隔 16 px：面积远小于 min_area_px，且不成大连通域
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
        let r = insp.inspect(&GrayImage::new(200, 100, 250));
        assert_eq!(r.decision.code, VerdictCode::NoQuality);
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
        let r = insp.inspect(&blurred);
        assert_eq!(
            r.decision.code,
            VerdictCode::NoQuality,
            "message={}",
            r.decision.message
        );
    }
}
