//! 检测流水线：定位 → 裁剪 → 量测 → 判定，与 Python `algo/pipeline.py` 分段一致。

use std::time::Instant;

use glue_core::{decide, Config, FrameResult, Timings};
use glue_vision::{locate, measure_mask, morph, GrayImage};
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
            let roi_mask = morph(&roi_mask, self.cfg.preprocess.morph.ksize,
                                 &self.cfg.preprocess.morph.order);
            measure_mask(&roi_mask, self.mm_per_px)
        });
        let measure_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let decision = decide(&self.cfg, rois.len(), measure.as_ref());
        FrameResult {
            rois,
            measure,
            decision,
            timings: Timings {
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
    fn empty_is_no_roi() {
        let insp = Inspector::new(Config::default());
        let r = insp.inspect(&GrayImage::new(200, 100, 10));
        assert_eq!(r.decision.code, VerdictCode::NoRoi);
    }
}
