//! RKNN NPU 推理（RK3588），可选 feature `rknn`。
//!
//! 通过 `librknnrt.so` 的 C API 动态加载（libloading），输入预处理与后处理复用
//! 与 ONNX 路径相同的 letterbox / anchors / proposals 逻辑，保证两条推理路径一致。
//!
//! 说明：
//! - 只在需要时编译：`cargo build --features rknn`；
//! - 运行环境必须有 `librknnrt.so`（RK3588 板端 SDK）；可用环境变量 `GLUE_RKNN_LIB` 指定路径；
//! - 缺少库/模型格式不符时返回错误，由上层回退启发式并写入 `notes`，不静默降级。

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use glue_core::{BBox, ObjectDetector};
use libloading::Library;

use crate::{letterbox, rpn, unletterbox};

const RKNN_TENSOR_FLOAT32: u32 = 0;
const RKNN_TENSOR_NCHW: u32 = 0;

#[repr(C)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: u32,
    fmt: u32,
}

#[repr(C)]
struct RknnOutput {
    want_float: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
    is_prealloc: u8,
    type_: u32,
    fmt: u32,
}

type FnInit = unsafe extern "C" fn(*mut u64, *const c_void, u32, u32, *mut c_void) -> i32;
type FnInputsSet = unsafe extern "C" fn(u64, u32, *mut RknnInput) -> i32;
type FnRun = unsafe extern "C" fn(u64, *mut c_void) -> i32;
type FnOutputsGet = unsafe extern "C" fn(u64, u32, *mut RknnOutput, *mut c_void) -> i32;
type FnOutputsRelease = unsafe extern "C" fn(u64, u32, *mut RknnOutput) -> i32;
type FnDestroy = unsafe extern "C" fn(u64) -> i32;

struct RknnCore {
    _lib: Library,
    ctx: u64,
    destroy: FnDestroy,
    inputs_set: FnInputsSet,
    run: FnRun,
    outputs_get: FnOutputsGet,
    outputs_release: FnOutputsRelease,
}

impl Drop for RknnCore {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.ctx);
        }
    }
}

impl RknnCore {
    fn load(model: &[u8]) -> Result<Self> {
        let lib_path = std::env::var("GLUE_RKNN_LIB").unwrap_or_else(|_| "librknnrt.so".to_string());
        let lib = unsafe { Library::new(&lib_path) }
            .with_context(|| format!("加载 RKNN 运行库失败：{lib_path}（板端需 librknnrt.so）"))?;
        // 先把符号拷成函数指针，再移动 Library
        let init: FnInit = unsafe { *lib.get(b"rknn_init\0")? };
        let inputs_set: FnInputsSet = unsafe { *lib.get(b"rknn_inputs_set\0")? };
        let run: FnRun = unsafe { *lib.get(b"rknn_run\0")? };
        let outputs_get: FnOutputsGet = unsafe { *lib.get(b"rknn_outputs_get\0")? };
        let outputs_release: FnOutputsRelease = unsafe { *lib.get(b"rknn_outputs_release\0")? };
        let destroy: FnDestroy = unsafe { *lib.get(b"rknn_destroy\0")? };

        let mut ctx: u64 = 0;
        let ret = unsafe { init(&mut ctx, model.as_ptr() as *const c_void, model.len() as u32, 0, std::ptr::null_mut()) };
        if ret != 0 {
            bail!("rknn_init 失败（ret={ret}）");
        }
        Ok(Self { _lib: lib, ctx, destroy, inputs_set, run, outputs_get, outputs_release })
    }

    /// 执行一次推理，返回 (cls, reg) 两个 f32 输出。
    fn infer(&self, canvas: &[f32], size: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = size * size;
        let mut input = canvas.to_vec();
        let mut ios = RknnInput {
            index: 0,
            buf: input.as_mut_ptr() as *mut c_void,
            size: (n * 4) as u32,
            pass_through: 0,
            type_: RKNN_TENSOR_FLOAT32,
            fmt: RKNN_TENSOR_NCHW,
        };
        let ret = unsafe { (self.inputs_set)(self.ctx, 1, &mut ios) };
        if ret != 0 {
            bail!("rknn_inputs_set 失败（ret={ret}）");
        }
        let ret = unsafe { (self.run)(self.ctx, std::ptr::null_mut()) };
        if ret != 0 {
            bail!("rknn_run 失败（ret={ret}）");
        }
        let mut outs = [
            RknnOutput { want_float: 1, index: 0, buf: std::ptr::null_mut(), size: 0, is_prealloc: 0, type_: 0, fmt: 0 },
            RknnOutput { want_float: 1, index: 1, buf: std::ptr::null_mut(), size: 0, is_prealloc: 0, type_: 0, fmt: 0 },
        ];
        let ret = unsafe { (self.outputs_get)(self.ctx, 2, outs.as_mut_ptr(), std::ptr::null_mut()) };
        if ret != 0 {
            bail!("rknn_outputs_get 失败（ret={ret}，模型应输出 cls/reg 两个张量）");
        }
        let copy = |o: &RknnOutput| -> Vec<f32> {
            let cnt = (o.size / 4) as usize;
            unsafe { std::slice::from_raw_parts(o.buf as *const f32, cnt).to_vec() }
        };
        let cls = copy(&outs[0]);
        let reg = copy(&outs[1]);
        unsafe {
            (self.outputs_release)(self.ctx, 2, outs.as_mut_ptr());
        }
        Ok((cls, reg))
    }
}

/// RKNN 版 RPN 定位器：接口与 `RpnDetector` 一致，供 pipeline 直接替换。
pub struct RknnDetector {
    inner: Mutex<RknnCore>,
    size: usize,
    score_thr: f32,
    pre_topk: usize,
    post_topk: usize,
    min_size: f32,
    anchors_wh: Option<Vec<(f32, f32)>>,
}

impl RknnDetector {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        model: PathBuf,
        size: usize,
        score_thr: f32,
        pre_topk: usize,
        post_topk: usize,
        min_size: f32,
        anchors_wh: Option<Vec<(f32, f32)>>,
    ) -> Result<Self> {
        let bytes = std::fs::read(&model).with_context(|| format!("读取 RKNN 模型失败：{}", model.display()))?;
        let core = RknnCore::load(&bytes)?;
        Ok(Self {
            inner: Mutex::new(core),
            size,
            score_thr,
            pre_topk,
            post_topk,
            min_size,
            anchors_wh,
        })
    }
}

impl ObjectDetector for RknnDetector {
    fn detect(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<BBox>> {
        let (canvas, scale, pad_x, pad_y) = letterbox(gray, w, h, self.size)?;
        let (cls, reg) = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("RKNN 上下文锁被毒化"))?
            .infer(&canvas, self.size)?;
        let f = self.size / 16;
        let n = f * f * 9;
        if cls.len() != n || reg.len() != n * 4 {
            bail!(
                "RKNN 输出形状不匹配：cls={} reg={}，期望 cls={n} reg={}（feat {f}x{f}×9）",
                cls.len(),
                reg.len(),
                n * 4
            );
        }
        let anchors = match &self.anchors_wh {
            Some(wh) => rpn::anchors_with_wh(f, f, 16, wh),
            None => rpn::anchors(f, f, 16, 9),
        };
        let boxes = rpn::proposals(&anchors, &reg, &cls, self.score_thr, self.pre_topk, self.post_topk, self.min_size);
        Ok(boxes
            .iter()
            .map(|b| {
                unletterbox(
                    BBox { x1: b.x1 as f64, y1: b.y1 as f64, x2: b.x2 as f64, y2: b.y2 as f64 },
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
