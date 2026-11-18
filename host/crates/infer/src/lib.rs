//! `glue-infer` 库：ONNX 推理（tract，纯 Rust）与 `glue-core::ObjectDetector` 实现。
//!
//! - `RpnDetector`：RPN 特征图 → anchors 解码 → NMS → 原图像素框（定位）；
//! - `YoloDetector`：YOLO5FL 输出 → 解码 → NMS → 原图像素框（反光）；
//! - 预处理与 Python `algo/data.py::letterbox` 保持一致：等比缩放 + 114 填充 + /255。
//!
//! 限制：模型输入尺寸与 anchor 必须与训练/导出时一致；当前默认 9 个 RPN anchor、
//! YOLO 默认 15 个 anchor（K-means 定制的 anchor 尚未从配置传入）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Context, Result};
use glue_core::{BBox, Config, ObjectDetector};
use tract_onnx::prelude::*;

pub mod rpn;
pub mod yolo;
#[cfg(feature = "rknn")]
pub mod rknn;

/// 运行器：输入 f32（行优先、1×1×h×w），返回各输出张量（形状, 数据）。
pub type Runner =
    Arc<dyn Fn(&[f32], usize, usize) -> Result<Vec<(Vec<usize>, Vec<f32>)>> + Send + Sync>;

#[allow(clippy::type_complexity)]
fn cache() -> &'static Mutex<HashMap<PathBuf, Runner>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Runner>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 按路径缓存加载的 ONNX 模型（进程内复用，避免每帧重复加载）。
pub fn runner_for(path: &Path) -> Result<Runner> {
    if let Some(r) = cache().lock().expect("infer cache poisoned").get(path) {
        return Ok(r.clone());
    }
    let path = path.to_path_buf();
    let model = tract_onnx::onnx()
        .model_for_path(&path)
        .with_context(|| format!("加载 ONNX 失败：{}", path.display()))?
        .into_optimized()
        .context("优化模型失败")?
        .into_runnable()
        .context("构建可运行模型失败")?;
    let runner: Runner = Arc::new(move |input: &[f32], w: usize, h: usize| {
        let tensor = Tensor::from_shape(&[1usize, 1, h, w], input)?;
        let outputs = model.run(tvec!(tensor.into()))?;
        let mut out = Vec::with_capacity(outputs.len());
        for value in outputs {
            let view = value.to_array_view::<f32>()?;
            out.push((view.shape().to_vec(), view.iter().copied().collect()));
        }
        Ok(out)
    });
    cache()
        .lock()
        .expect("infer cache poisoned")
        .insert(path, runner.clone());
    Ok(runner)
}

/// 加载模型并运行一次，返回按输出顺序的张量（形状, 数据）。
pub fn run_model(
    model_path: &Path,
    input: &[f32],
    (w, h): (usize, usize),
) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    if input.len() != w * h {
        bail!("输入像素数 {} 与形状 {}x{} 不匹配", input.len(), w, h);
    }
    let runner = runner_for(model_path)?;
    runner(input, w, h)
}

/// 等比缩放 + 居中 114 填充 + 归一化，返回 (画布 f32, 缩放比, pad_x, pad_y)。
pub fn letterbox(
    gray: &[u8],
    w: usize,
    h: usize,
    size: usize,
) -> Result<(Vec<f32>, f32, usize, usize)> {
    if gray.len() != w * h || w == 0 || h == 0 {
        bail!("灰度图尺寸不匹配：{} vs {}x{}", gray.len(), w, h);
    }
    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as usize).max(1).min(size);
    let nh = ((h as f32 * scale).round() as usize).max(1).min(size);
    let pad_x = (size - nw) / 2;
    let pad_y = (size - nh) / 2;
    let mut canvas = vec![114.0f32 / 255.0; size * size];
    for y in 0..nh {
        // 双线性采样（与 OpenCV INTER_LINEAR 近似）
        let sy = ((y as f32 + 0.5) * h as f32 / nh as f32 - 0.5).clamp(0.0, h as f32 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = sy - y0 as f32;
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) * w as f32 / nw as f32 - 0.5).clamp(0.0, w as f32 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = sx - x0 as f32;
            let p00 = gray[y0 * w + x0] as f32;
            let p01 = gray[y0 * w + x1] as f32;
            let p10 = gray[y1 * w + x0] as f32;
            let p11 = gray[y1 * w + x1] as f32;
            let v = p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy;
            canvas[(y + pad_y) * size + x + pad_x] = v / 255.0;
        }
    }
    Ok((canvas, scale, pad_x, pad_y))
}

/// 把 letterbox 坐标映射回原图并裁剪到图内。
fn unletterbox(b: BBox, scale: f32, pad: (usize, usize), w: usize, h: usize) -> BBox {
    let (px, py) = (pad.0 as f64, pad.1 as f64);
    let s = scale as f64;
    BBox {
        x1: ((b.x1 - px) / s).clamp(0.0, w as f64),
        y1: ((b.y1 - py) / s).clamp(0.0, h as f64),
        x2: ((b.x2 - px) / s).clamp(0.0, w as f64),
        y2: ((b.y2 - py) / s).clamp(0.0, h as f64),
    }
}

/// RPN 定位器：输出胶条 RoI（像素坐标，原图）。
pub struct RpnDetector {
    pub model: PathBuf,
    pub size: usize,
    pub score_thr: f32,
    pub pre_topk: usize,
    pub post_topk: usize,
    pub min_size: f32,
    /// K-means 先验框（w,h），None 用默认 9 个 anchor；必须与训练一致
    pub anchors_wh: Option<Vec<(f32, f32)>>,
}

impl ObjectDetector for RpnDetector {
    fn detect(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<BBox>> {
        let (canvas, scale, pad_x, pad_y) = letterbox(gray, w, h, self.size)?;
        let outputs = run_model(&self.model, &canvas, (self.size, self.size))?;
        if outputs.len() < 2 {
            bail!("RPN 模型应输出 cls 与 reg 两个张量");
        }
        let (cls_shape, cls) = &outputs[0];
        let (_reg_shape, reg) = &outputs[1];
        let f = self.size / 16;
        let n = f * f * 9;
        if cls.len() != n || reg.len() != n * 4 {
            bail!("RPN 输出形状与 feat {f}x{f}×9 不匹配（cls={cls_shape:?}）");
        }
        let anchors = match &self.anchors_wh {
            Some(wh) => rpn::anchors_with_wh(f, f, 16, wh),
            None => rpn::anchors(f, f, 16, 9),
        };
        let boxes = rpn::proposals(
            &anchors,
            reg,
            cls,
            self.score_thr,
            self.pre_topk,
            self.post_topk,
            self.min_size,
        );
        Ok(boxes
            .iter()
            .map(|b| {
                unletterbox(
                    BBox {
                        x1: b.x1 as f64,
                        y1: b.y1 as f64,
                        x2: b.x2 as f64,
                        y2: b.y2 as f64,
                    },
                    scale,
                    (pad_x, pad_y),
                    w,
                    h,
                )
            })
            .filter(|b| b.width() > 1.0 && b.height() > 1.0)
            .collect())
    }
}

/// YOLO 反光检测器：输出反光区域框（像素坐标，原图）。
pub struct YoloDetector {
    pub model: PathBuf,
    pub size: usize,
    pub score_thr: f32,
}

impl ObjectDetector for YoloDetector {
    fn detect(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<BBox>> {
        let (canvas, scale, pad_x, pad_y) = letterbox(gray, w, h, self.size)?;
        let outputs = run_model(&self.model, &canvas, (self.size, self.size))?;
        let (shape, out) = outputs.first().context("YOLO 模型无输出")?;
        let dets = yolo::decode(out, (self.size, self.size), shape, self.score_thr)?;
        Ok(dets
            .iter()
            .map(|d| {
                unletterbox(
                    BBox {
                        x1: d.x1 as f64,
                        y1: d.y1 as f64,
                        x2: d.x2 as f64,
                        y2: d.y2 as f64,
                    },
                    scale,
                    (pad_x, pad_y),
                    w,
                    h,
                )
            })
            .filter(|b| b.width() > 1.0 && b.height() > 1.0)
            .collect())
    }
}

/// 按配置构建检测器：权重缺失/类型不符时返回错误（由调用方决定是否降级并记录）。
pub type Detectors = (Option<Box<dyn ObjectDetector>>, Option<Box<dyn ObjectDetector>>);

/// 按配置构建检测器：权重缺失/类型不符时返回错误（由调用方决定是否降级并记录）。
pub fn detectors_from_config(cfg: &Config) -> Result<Detectors> {
    let mut locate: Option<Box<dyn ObjectDetector>> = None;
    if cfg.locate.enabled {
        let path = cfg
            .locate
            .weights
            .as_ref()
            .context("locate.enabled=true 但未配置 weights")?;
        if cfg.locate.kind == "rknn" {
            #[cfg(feature = "rknn")]
            {
                if !Path::new(path).exists() {
                    bail!("RKNN 模型不存在：{path}（先运行 scripts/export_rknn.py 转换）");
                }
                locate = Some(Box::new(rknn::RknnDetector::open(
                    PathBuf::from(path),
                    cfg.locate.input_size,
                    cfg.locate.score_thr as f32,
                    cfg.locate.pre_topk,
                    cfg.locate.post_topk,
                    cfg.locate.min_size as f32,
                    cfg.locate
                        .anchors_wh
                        .as_ref()
                        .map(|v| v.iter().map(|p| (p[0] as f32, p[1] as f32)).collect()),
                )?));
            }
            #[cfg(not(feature = "rknn"))]
            bail!("locate.type=rknn 需要以 --features rknn 构建（NPU 运行库仅在板端可用）");
        } else if cfg.locate.kind != "rpn" {
            bail!("locate.type={} 暂只支持 rpn/rknn", cfg.locate.kind);
        } else {
        if !Path::new(path).exists() {
            bail!("RPN 权重不存在：{path}");
        }
        locate = Some(Box::new(RpnDetector {
            model: PathBuf::from(path),
            size: cfg.locate.input_size,
            score_thr: cfg.locate.score_thr as f32,
            pre_topk: cfg.locate.pre_topk,
            post_topk: cfg.locate.post_topk,
            min_size: cfg.locate.min_size as f32,
            anchors_wh: cfg
                .locate
                .anchors_wh
                .as_ref()
                .map(|v| v.iter().map(|p| (p[0] as f32, p[1] as f32)).collect()),
        }));
        }
    }
    let mut reflect: Option<Box<dyn ObjectDetector>> = None;
    if cfg.reflect.enabled {
        let path = cfg
            .reflect
            .weights
            .as_ref()
            .context("reflect.enabled=true 但未配置 weights")?;
        if !Path::new(path).exists() {
            bail!("YOLO 权重不存在：{path}");
        }
        if cfg.reflect.kind != "yolo5fl" && cfg.reflect.kind != "yolo" {
            bail!("reflect.type={} 暂只支持 yolo5fl/yolo", cfg.reflect.kind);
        }
        reflect = Some(Box::new(YoloDetector {
            model: PathBuf::from(path),
            size: cfg.reflect.input_size,
            score_thr: cfg.reflect.score_thr as f32,
        }));
    }
    Ok((locate, reflect))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_aspect_and_pads() {
        let (w, h) = (40usize, 20usize);
        let gray = vec![255u8; w * h];
        let (canvas, scale, pad_x, pad_y) = letterbox(&gray, w, h, 40).expect("letterbox");
        assert_eq!(canvas.len(), 40 * 40);
        assert!((scale - 1.0).abs() < 1e-6);
        assert_eq!((pad_x, pad_y), (0, 10));
        // 上半部分应为填充值 114/255
        assert!((canvas[0] - 114.0 / 255.0).abs() < 1e-6);
        // 中间应为 1.0
        assert!((canvas[20 * 40 + 20] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unletterbox_roundtrip() {
        let b = BBox {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 40.0,
        };
        let mapped = unletterbox(b, 0.5, (5, 5), 100, 100);
        assert!((mapped.x1 - 10.0).abs() < 1e-4);
        assert!((mapped.y1 - 30.0).abs() < 1e-4);
    }

    #[test]
    fn detectors_disabled_by_default() {
        let (l, r) = detectors_from_config(&Config::default()).expect("build");
        assert!(l.is_none() && r.is_none());
    }
}
