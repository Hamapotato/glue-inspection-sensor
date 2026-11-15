//! 视觉算法：灰度图、Otsu、形态学、连通域、Zhang-Suen 骨架、骨架长度、宽度换算。
//! 纯 Rust 实现（不依赖 OpenCV），语义对齐 Python 端 `algo/preprocess.py`、`algo/measure.py`。

use anyhow::{bail, Result};
use glue_core::{BBox, Config, Measure};

#[derive(Debug, Clone)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(w: usize, h: usize, fill: u8) -> Self {
        Self { w, h, data: vec![fill; w * h] }
    }

    pub fn from_vec(w: usize, h: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != w * h {
            bail!("像素数量 {} 与尺寸 {}x{} 不匹配", data.len(), w, h);
        }
        Ok(Self { w, h, data })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.w + x] = v;
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let img = image::open(path.as_ref())?.to_luma8();
        let (w, h) = img.dimensions();
        Ok(Self { w: w as usize, h: h as usize, data: img.into_raw() })
    }

    pub fn crop(&self, bbox: &BBox) -> Self {
        let x1 = bbox.x1.floor().max(0.0) as usize;
        let y1 = bbox.y1.floor().max(0.0) as usize;
        let x2 = (bbox.x2.ceil() as usize).min(self.w);
        let y2 = (bbox.y2.ceil() as usize).min(self.h);
        let (w, h) = (x2.saturating_sub(x1), y2.saturating_sub(y1));
        let mut out = Vec::with_capacity(w * h);
        for y in y1..y2 {
            out.extend_from_slice(&self.data[y * self.w + x1..y * self.w + x2]);
        }
        Self { w, h, data: out }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryMask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

impl BinaryMask {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![false; w * h] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: bool) {
        self.data[y * self.w + x] = v;
    }

    pub fn count(&self) -> usize {
        self.data.iter().filter(|v| **v).count()
    }

    pub fn crop(&self, bbox: &BBox) -> Self {
        let x1 = bbox.x1.floor().max(0.0) as usize;
        let y1 = bbox.y1.floor().max(0.0) as usize;
        let x2 = (bbox.x2.ceil() as usize).min(self.w);
        let y2 = (bbox.y2.ceil() as usize).min(self.h);
        let (w, h) = (x2.saturating_sub(x1), y2.saturating_sub(y1));
        let mut data = Vec::with_capacity(w * h);
        for y in y1..y2 {
            data.extend_from_slice(&self.data[y * self.w + x1..y * self.w + x2]);
        }
        Self { w, h, data }
    }
}

/// 大津法（论文 4.2.1），分割规则 `pixel > t`。
pub fn otsu_threshold(img: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for &v in &img.data {
        hist[v as usize] += 1;
    }
    let total: u64 = img.data.len() as u64;
    let mut sum_all = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_all += i as f64 * c as f64;
    }
    let (mut w_bg, mut sum_bg) = (0u64, 0.0f64);
    let (mut best_t, mut best_var) = (0u8, -1.0f64);
    for t in 0..256usize {
        w_bg += hist[t];
        if w_bg == 0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += t as f64 * hist[t] as f64;
        let m_bg = sum_bg / w_bg as f64;
        let m_fg = (sum_all - sum_bg) / w_fg as f64;
        let var = w_bg as f64 * w_fg as f64 * (m_bg - m_fg) * (m_bg - m_fg);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

pub fn threshold(img: &GrayImage, t: u8) -> BinaryMask {
    let mut m = BinaryMask::new(img.w, img.h);
    for (i, &v) in img.data.iter().enumerate() {
        m.data[i] = v > t;
    }
    m
}

fn ellipse_offsets(ksize: u32) -> Vec<(i32, i32)> {
    let r = (ksize.max(1) / 2) as i32;
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                out.push((dx, dy));
            }
        }
    }
    out
}

fn erode(mask: &BinaryMask, ksize: u32) -> BinaryMask {
    let offs = ellipse_offsets(ksize);
    let mut out = BinaryMask::new(mask.w, mask.h);
    for y in 0..mask.h {
        for x in 0..mask.w {
            if !mask.get(x, y) {
                continue;
            }
            let mut keep = true;
            for &(dx, dy) in &offs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= mask.w as i32 || ny >= mask.h as i32
                    || !mask.get(nx as usize, ny as usize)
                {
                    keep = false;
                    break;
                }
            }
            out.set(x, y, keep);
        }
    }
    out
}

fn dilate(mask: &BinaryMask, ksize: u32) -> BinaryMask {
    let offs = ellipse_offsets(ksize);
    let mut out = BinaryMask::new(mask.w, mask.h);
    for y in 0..mask.h {
        for x in 0..mask.w {
            if !mask.get(x, y) {
                continue;
            }
            for &(dx, dy) in &offs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < mask.w as i32 && ny < mask.h as i32 {
                    out.set(nx as usize, ny as usize, true);
                }
            }
        }
    }
    out
}

pub fn morph(mask: &BinaryMask, ksize: u32, order: &str) -> BinaryMask {
    if ksize == 0 {
        return mask.clone();
    }
    let open = |m: &BinaryMask| dilate(&erode(m, ksize), ksize);
    let close = |m: &BinaryMask| erode(&dilate(m, ksize), ksize);
    match order {
        "open" => open(mask),
        "close" => close(mask),
        "close-open" => open(&close(mask)),
        _ => close(&open(mask)),
    }
}

#[derive(Debug, Clone)]
pub struct Component {
    pub area: u32,
    pub bbox: BBox,
}

pub fn connected_components(mask: &BinaryMask, connectivity: u8, min_area: u32) -> Vec<Component> {
    let (w, h) = (mask.w, mask.h);
    let mut seen = vec![false; w * h];
    let neighbors: Vec<(i32, i32)> = if connectivity >= 8 {
        vec![(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    } else {
        vec![(0, -1), (-1, 0), (1, 0), (0, 1)]
    };
    let mut out = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            let idx = sy * w + sx;
            if seen[idx] || !mask.data[idx] {
                continue;
            }
            let mut stack = vec![(sx, sy)];
            seen[idx] = true;
            let mut area = 0u32;
            let (mut x1, mut y1, mut x2, mut y2) = (usize::MAX, usize::MAX, 0usize, 0usize);
            while let Some((x, y)) = stack.pop() {
                area += 1;
                x1 = x1.min(x);
                y1 = y1.min(y);
                x2 = x2.max(x);
                y2 = y2.max(y);
                for &(dx, dy) in &neighbors {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if !seen[ni] && mask.data[ni] {
                        seen[ni] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
            if area >= min_area {
                out.push(Component {
                    area,
                    bbox: BBox {
                        x1: x1 as f64,
                        y1: y1 as f64,
                        x2: x2 as f64 + 1.0,
                        y2: y2 as f64 + 1.0,
                    },
                });
            }
        }
    }
    out.sort_by(|a, b| b.area.cmp(&a.area));
    out
}

/// Zhang-Suen 细化（论文 4.4.2）。
pub fn zhang_suen_thinning(mask: &BinaryMask) -> BinaryMask {
    let (w, h) = (mask.w, mask.h);
    let mut img = mask.data.clone();
    if w < 3 || h < 3 {
        return BinaryMask { w, h, data: img };
    }
    let idx = |x: usize, y: usize| y * w + x;
    loop {
        let mut changed = false;
        for step in 0..2 {
            let mut remove = Vec::new();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !img[idx(x, y)] {
                        continue;
                    }
                    let p2 = img[idx(x, y - 1)];
                    let p3 = img[idx(x + 1, y - 1)];
                    let p4 = img[idx(x + 1, y)];
                    let p5 = img[idx(x + 1, y + 1)];
                    let p6 = img[idx(x, y + 1)];
                    let p7 = img[idx(x - 1, y + 1)];
                    let p8 = img[idx(x - 1, y)];
                    let p9 = img[idx(x - 1, y - 1)];
                    let seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
                    let b = seq[..8].iter().filter(|v| **v).count();
                    if !(2..=6).contains(&b) {
                        continue;
                    }
                    let a = (0..8).filter(|i| !seq[*i] && seq[i + 1]).count();
                    if a != 1 {
                        continue;
                    }
                    let (c1, c2) = if step == 0 {
                        (!(p2 && p4 && p6), !(p4 && p6 && p8))
                    } else {
                        (!(p2 && p4 && p8), !(p2 && p6 && p8))
                    };
                    if c1 && c2 {
                        remove.push(idx(x, y));
                    }
                }
            }
            for i in remove {
                img[i] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    BinaryMask { w, h, data: img }
}

/// 骨架长度（像素）：取骨架点沿主方向的投影跨度（复现假设 A11）。
///
/// 早期实现用图直径（两次最短路），但 Otsu 后的二值边缘存在锯齿，骨架会产生
/// 台阶与短分支，使路径长度系统性偏长（实测 +9%），宽度随之偏小。改为 PCA
/// 主方向投影跨度，对直线胶条更稳；弯曲胶条会略低估，后续可换曲线拟合。
pub fn skeleton_length(skel: &BinaryMask) -> f64 {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for y in 0..skel.h {
        for x in 0..skel.w {
            if skel.get(x, y) {
                pts.push((x as f64, y as f64));
            }
        }
    }
    if pts.len() < 2 {
        return 0.0;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for &(x, y) in &pts {
        let (dx, dy) = (x - mx, y - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ct, st) = (theta.cos(), theta.sin());
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for &(x, y) in &pts {
        let proj = (x - mx) * ct + (y - my) * st;
        lo = lo.min(proj);
        hi = hi.max(proj);
    }
    (hi - lo).max(0.0)
}

pub fn measure_roi(roi: &GrayImage, cfg: &Config, mm_per_px: Option<f64>) -> Measure {
    let t = otsu_threshold(roi);
    let mask = threshold(roi, t);
    let mask = morph(&mask, cfg.preprocess.morph.ksize, &cfg.preprocess.morph.order);
    let area = mask.count() as f64;
    let skel = zhang_suen_thinning(&mask);
    let length = skeleton_length(&skel);
    let width_px = if length > 0.0 { area / length } else { 0.0 };
    Measure { area_px: area, length_px: length, width_px, width_mm: mm_per_px.map(|s| width_px * s) }
}

/// 在已有掩膜上量测（论文 4.4）：面积 ÷ 骨架长度，再乘标定系数。
pub fn measure_mask(mask: &BinaryMask, mm_per_px: Option<f64>) -> Measure {
    let area = mask.count() as f64;
    // 骨架的端部会被细化算法缩短约半个胶宽，直接拿骨架长度做分母会高估宽度
    // （实测：1200×120 的胶条，骨架只有 1075 长）。这里改用掩膜沿主方向的跨度作为
    // 长度：直线胶条与骨架长度一致，且没有端部缩短问题。
    let length = mask_extent(mask);
    let width_px = if length > 0.0 { area / length } else { 0.0 };
    Measure { area_px: area, length_px: length, width_px, width_mm: mm_per_px.map(|s| width_px * s) }
}

/// 掩膜沿主方向的投影跨度（对直线胶条即长度）。
pub fn mask_extent(mask: &BinaryMask) -> f64 {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for y in 0..mask.h {
        for x in 0..mask.w {
            if mask.get(x, y) {
                pts.push((x as f64, y as f64));
            }
        }
    }
    if pts.len() < 2 {
        return 0.0;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for &(x, y) in &pts {
        let (dx, dy) = (x - mx, y - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ct, st) = (theta.cos(), theta.sin());
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for &(x, y) in &pts {
        let proj = (x - mx) * ct + (y - my) * st;
        lo = lo.min(proj);
        hi = hi.max(proj);
    }
    (hi - lo).max(0.0)
}

/// 定位：返回 RoI 列表与对应掩膜（量测直接在掩膜上做，避免对紧贴裁剪区域再做 Otsu）。
pub fn locate(img: &GrayImage, cfg: &Config) -> (Vec<BBox>, BinaryMask) {
    // 低对比度（全黑/全白/均匀噪声）视为无效图像
    let (mut lo, mut hi) = (255u8, 0u8);
    for &v in &img.data {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi.saturating_sub(lo) < 40 {
        return (Vec::new(), BinaryMask::new(img.w, img.h));
    }
    let t = otsu_threshold(img);
    let mask = threshold(img, t);
    let area = mask.count();
    if area == 0 || area > (0.9 * (mask.w * mask.h) as f64) as usize {
        return (Vec::new(), mask);
    }
    let rois = connected_components(&mask, cfg.measure.bfs_connectivity, cfg.measure.min_area_px)
        .into_iter()
        .take(10)
        .map(|c| c.bbox)
        .collect();
    (rois, mask)
}

pub fn locate_rois(img: &GrayImage, cfg: &Config) -> Vec<BBox> {
    locate(img, cfg).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(w: usize, h: usize, strip_w: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h, 20);
        let y0 = h / 2 - strip_w / 2;
        for y in y0..y0 + strip_w {
            for x in 10..w - 10 {
                img.set(x, y, 220);
            }
        }
        img
    }

    #[test]
    fn otsu_between_peaks() {
        let t = otsu_threshold(&strip(100, 60, 10));
        assert!((20..220).contains(&t), "阈值 {t}");
    }

    #[test]
    fn components_count() {
        let mut img = GrayImage::new(80, 60, 0);
        for y in 10..20 {
            for x in 10..30 {
                img.set(x, y, 255);
            }
        }
        for y in 40..50 {
            for x in 50..70 {
                img.set(x, y, 255);
            }
        }
        assert_eq!(connected_components(&threshold(&img, 128), 8, 10).len(), 2);
    }

    #[test]
    fn thinning_length() {
        let mut img = GrayImage::new(120, 40, 0);
        for y in 18..22 {
            for x in 10..110 {
                img.set(x, y, 255);
            }
        }
        let skel = zhang_suen_thinning(&threshold(&img, 128));
        let len = skeleton_length(&skel);
        assert!((90.0..=105.0).contains(&len), "长度 {len}");
    }

    #[test]
    fn width_area_over_length() {
        let m = measure_roi(&strip(200, 80, 10), &Config::default(), Some(0.25));
        assert!((m.width_px - 10.0).abs() < 1.5, "宽度 {}", m.width_px);
        assert!((m.width_mm.unwrap() - 2.5).abs() < 0.4);
    }

    #[test]
    fn locate_single_strip() {
        let rois = locate_rois(&strip(320, 120, 20), &Config::default());
        assert_eq!(rois.len(), 1);
        assert!((rois[0].y2 - rois[0].y1 - 20.0).abs() < 4.0);
    }
}
