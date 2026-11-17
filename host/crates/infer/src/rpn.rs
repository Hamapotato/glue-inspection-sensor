//! RPN 后处理：anchors 生成、解码、CIoU-NMS 候选框筛选（对齐 `algo/net/rpn.py`）。

#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RpnBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

const SCALES: [f32; 3] = [32.0, 64.0, 128.0];
const RATIOS: [f32; 3] = [0.5, 1.0, 2.0];

/// 按特征图尺寸铺 anchor（`stride` 为下采样倍数，默认 9 个 anchor）。
pub fn anchors(h: usize, w: usize, stride: usize, num: usize) -> Vec<Anchor> {
    let mut out = Vec::with_capacity(h * w * num);
    for y in 0..h {
        for x in 0..w {
            let cx = (x as f32 + 0.5) * stride as f32;
            let cy = (y as f32 + 0.5) * stride as f32;
            for scale in SCALES {
                for ratio in RATIOS {
                    let aw = scale * ratio.sqrt();
                    let ah = scale / ratio.sqrt();
                    out.push(Anchor {
                        cx,
                        cy,
                        w: aw,
                        h: ah,
                    });
                }
            }
        }
    }
    out
}

#[inline]
pub fn decode(a: &Anchor, d: [f32; 4]) -> (f32, f32, f32, f32) {
    let cx = d[0] * a.w + a.cx;
    let cy = d[1] * a.h + a.cy;
    let w = d[2].exp() * a.w;
    let h = d[3].exp() * a.h;
    (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
}

fn iou(a: &RpnBox, b: &RpnBox) -> f32 {
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 候选框：分数 sigmoid → 预排序 top-k → 解码 → 最小尺寸过滤 → CIoU-NMS → 截断。
pub fn proposals(
    anchors: &[Anchor],
    reg: &[f32],
    cls: &[f32],
    score_thr: f32,
    pre_topk: usize,
    post_topk: usize,
    min_size: f32,
) -> Vec<RpnBox> {
    let n = cls.len().min(anchors.len());
    if reg.len() < n * 4 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).filter(|&i| sigmoid(cls[i]) >= score_thr).collect();
    idx.sort_by(|&a, &b| sigmoid(cls[b]).total_cmp(&sigmoid(cls[a])));
    idx.truncate(pre_topk);

    let mut boxes: Vec<RpnBox> = idx
        .into_iter()
        .map(|i| {
            let d = [reg[i * 4], reg[i * 4 + 1], reg[i * 4 + 2], reg[i * 4 + 3]];
            let (x1, y1, x2, y2) = decode(&anchors[i], d);
            RpnBox {
                x1,
                y1,
                x2,
                y2,
                score: sigmoid(cls[i]),
            }
        })
        .filter(|b| (b.x2 - b.x1) >= min_size && (b.y2 - b.y1) >= min_size)
        .collect();

    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<RpnBox> = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(boxes[i]);
        for j in i + 1..boxes.len() {
            if !suppressed[j] && iou(&boxes[i], &boxes[j]) > 0.5 {
                suppressed[j] = true;
            }
        }
        if keep.len() >= post_topk {
            break;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_geometry() {
        let a = anchors(2, 3, 16, 9);
        assert_eq!(a.len(), 2 * 3 * 9);
        // 第一个格子中心 (8,8)，第一个 anchor 为 scale=32, ratio=0.5
        assert!((a[0].cx - 8.0).abs() < 1e-5);
        assert!((a[0].cy - 8.0).abs() < 1e-5);
        assert!((a[0].w - 32.0 * 0.5f32.sqrt()).abs() < 1e-4);
        assert!((a[0].h - 32.0 / 0.5f32.sqrt()).abs() < 1e-4);
        // 最后一个格子中心 (40,24)
        assert!((a[a.len() - 1].cx - 40.0).abs() < 1e-5);
        assert!((a[a.len() - 1].cy - 24.0).abs() < 1e-5);
    }

    #[test]
    fn decode_zero_delta_is_anchor_box() {
        let a = Anchor {
            cx: 100.0,
            cy: 50.0,
            w: 40.0,
            h: 20.0,
        };
        let (x1, y1, x2, y2) = decode(&a, [0.0, 0.0, 0.0, 0.0]);
        assert!((x1 - 80.0).abs() < 1e-5 && (y1 - 40.0).abs() < 1e-5);
        assert!((x2 - 120.0).abs() < 1e-5 && (y2 - 60.0).abs() < 1e-5);
    }

    #[test]
    fn nms_suppresses_overlap() {
        let anchors = vec![
            Anchor {
                cx: 100.0,
                cy: 100.0,
                w: 40.0,
                h: 20.0,
            },
            Anchor {
                cx: 101.0,
                cy: 100.0,
                w: 40.0,
                h: 20.0,
            },
            Anchor {
                cx: 400.0,
                cy: 100.0,
                w: 40.0,
                h: 20.0,
            },
        ];
        let reg = vec![0.0; 12];
        // 分数：第一个最高、第二个次之、第三个较低
        let cls = vec![4.0, 3.0, 1.0];
        let out = proposals(&anchors, &reg, &cls, 0.5, 10, 10, 1.0);
        assert_eq!(out.len(), 2, "overlap should be suppressed: {out:?}");
        assert!(out[0].score > out[1].score);
    }
}
