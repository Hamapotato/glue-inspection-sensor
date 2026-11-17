//! YOLO5FL 后处理：按 5 个尺度解码 + NMS（对齐 `algo/net/yolo.py`）。

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// 5 个输出尺度（自粗到细）对应的 stride。
const STRIDES: [usize; 5] = [32, 16, 8, 4, 2];

/// 15 个先验框（每层 3 个，自粗到细），与 `algo/net/yolo.py::DEFAULT_ANCHORS` 一致。
const ANCHORS: [[[f32; 2]; 3]; 5] = [
    [[200.0, 60.0], [280.0, 80.0], [360.0, 100.0]],
    [[110.0, 40.0], [160.0, 56.0], [210.0, 70.0]],
    [[60.0, 26.0], [90.0, 36.0], [120.0, 48.0]],
    [[40.0, 20.0], [60.0, 28.0], [84.0, 38.0]],
    [[28.0, 16.0], [40.0, 22.0], [56.0, 30.0]],
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let uni = area_a + area_b - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

/// 解码拼接后的输出（shape `(1, N, 5+nc)`，`nc=1`），返回过滤后的检测框。
pub fn decode(out: &[f32], (img_w, img_h): (usize, usize), o_shape: &[usize], thr: f32) -> Result<Vec<Detection>> {
    if o_shape.len() != 3 {
        bail!("YOLO 输出应为 (B,N,6)，实际 {o_shape:?}");
    }
    let total = o_shape[1];
    let per = o_shape[2];
    if per < 5 {
        bail!("输出每行长度 {per} 过小");
    }
    let mut dets = Vec::new();
    let mut offset = 0usize;
    for (l, stride) in STRIDES.iter().enumerate() {
        let gh = img_h / stride;
        let gw = img_w / stride;
        let cells = gh * gw;
        let rows = cells * 3;
        if offset + rows > total {
            bail!("输出长度不足：level {l} 需要 {rows} 行，剩余 {}", total - offset);
        }
        for (a, anchor) in ANCHORS[l].iter().enumerate() {
            let (aw, ah) = (anchor[0], anchor[1]);
            for c in 0..cells {
                let base = (offset + a * cells + c) * per;
                let obj = sigmoid(out[base + 4]);
                let cls = sigmoid(out[base + 5]);
                let score = obj * cls;
                if score < thr {
                    continue;
                }
                let gx = (c % gw) as f32;
                let gy = (c / gw) as f32;
                let cx = (sigmoid(out[base]) + gx) * *stride as f32;
                let cy = (sigmoid(out[base + 1]) + gy) * *stride as f32;
                let w = out[base + 2].exp() * aw;
                let h = out[base + 3].exp() * ah;
                dets.push(Detection {
                    x1: cx - w / 2.0,
                    y1: cy - h / 2.0,
                    x2: cx + w / 2.0,
                    y2: cy + h / 2.0,
                    score,
                });
            }
        }
        offset += rows;
    }

    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i]);
        for j in i + 1..dets.len() {
            if !suppressed[j] && iou(&dets[i], &dets[j]) > 0.45 {
                suppressed[j] = true;
            }
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level_sizes(img_w: usize, img_h: usize) -> Vec<usize> {
        STRIDES.iter().map(|s| (img_w / s) * (img_h / s) * 3).collect()
    }

    #[test]
    fn decode_single_positive() {
        let (w, h) = (128usize, 128usize);
        let sizes = level_sizes(w, h);
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 16368);
        let mut out = vec![0f32; total * 6];
        // 最粗层（stride=32，4x4 网格）第 0 个 anchor 的第 0 个格子：obj=2, cls=2
        let base = 0usize;
        out[base + 4] = 2.0;
        out[base + 5] = 2.0;
        let dets = decode(&out, (w, h), &[1, total, 6], 0.5).expect("decode");
        assert_eq!(dets.len(), 1, "{dets:?}");
        let s = sigmoid(2.0) * sigmoid(2.0);
        assert!((dets[0].score - s).abs() < 1e-5);
    }

    #[test]
    fn decode_threshold_filters_low_score() {
        let (w, h) = (128usize, 128usize);
        let total: usize = level_sizes(w, h).iter().sum();
        let mut out = vec![0f32; total * 6];
        out[4] = -5.0;
        out[5] = -5.0;
        let dets = decode(&out, (w, h), &[1, total, 6], 0.5).expect("decode");
        assert!(dets.is_empty());
    }
}
