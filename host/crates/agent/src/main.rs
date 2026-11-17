//! `glue-agent`：无界面检测服务（RK3588 / Linux 板端）。
//!
//! 与 GUI 共用 `glue-core/vision/pipeline`，但不依赖 egui，便于交叉编译到 aarch64。
//! 用法：
//!   glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out results.json
//!   glue-agent --config ... --input samples --watch 5 --state state.jsonl
//!
//! 输出：stdout JSON 汇总；--state 指定时追加 JSONL（每帧一行，便于 systemd/日志采集）。

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use glue_core::Config;
use glue_pipeline::Inspector;
use glue_vision::GrayImage;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut cfg_path = "configs/default.yaml".to_string();
    let mut input = "samples".to_string();
    let mut out: Option<PathBuf> = None;
    let mut state: Option<PathBuf> = None;
    let mut mm_per_px: Option<f64> = None;
    let mut watch_secs: Option<u64> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => cfg_path = args.next().unwrap_or(cfg_path),
            "--input" => input = args.next().unwrap_or(input),
            "--out" => out = args.next().map(PathBuf::from),
            "--state" => state = args.next().map(PathBuf::from),
            "--mm-per-px" => mm_per_px = args.next().and_then(|s| s.parse().ok()),
            "--watch" => watch_secs = args.next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                println!(
                    "用法: glue-agent --config <yaml> --input <dir> [--mm-per-px F] [--out json] [--state jsonl] [--watch SEC]"
                );
                return Ok(());
            }
            other => bail!("未知参数：{other}"),
        }
    }

    let mut cfg =
        Config::from_yaml(&cfg_path).with_context(|| format!("读取配置失败：{cfg_path}"))?;
    if let Some(mm) = mm_per_px {
        cfg.calib.enabled = true;
        cfg.calib.mm_per_px = Some(mm);
    }
    let inspector = inspector(cfg);

    match watch_secs {
        Some(secs) => {
            let state_path = state.unwrap_or_else(|| PathBuf::from("glue-state.jsonl"));
            eprintln!("[agent] 监视模式：每 {secs}s 扫描 {input}，状态写 {state_path:?}");
            loop {
                run_once(&inspector, &input, None, Some(&state_path))?;
                std::thread::sleep(Duration::from_secs(secs));
            }
        }
        None => {
            run_once(&inspector, &input, out.as_deref(), state.as_deref())?;
            Ok(())
        }
    }
}

/// 按配置构建检测器（RPN 定位 / YOLO 反光）；推理配置有误时打印告警并回退启发式链路。
fn inspector(cfg: Config) -> Inspector {
    let mut detectors = glue_pipeline::Detectors::default();
    match glue_infer::detectors_from_config(&cfg) {
        Ok((locate, reflect)) => {
            detectors.locate = locate;
            detectors.reflect = reflect;
        }
        Err(e) => eprintln!("[warn] 推理模块未启用：{e}"),
    }
    Inspector::with_detectors(cfg, detectors)
}

fn run_once(
    inspector: &Inspector,
    input: &str,
    out: Option<&Path>,
    state: Option<&Path>,
) -> Result<()> {
    let files = collect_images(Path::new(input))?;
    let mut records = Vec::with_capacity(files.len());
    for path in &files {
        let img = GrayImage::from_file(path)?;
        let r = inspector.inspect(&img);
        let record = serde_json::json!({
            "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            "verdict": r.decision.code.as_str(),
            "message": r.decision.message,
            "width_mm": r.decision.width_mm,
            "rois": r.rois.iter().map(|b| b.to_array()).collect::<Vec<_>>(),
            "timings_ms": r.timings,
        });
        if let Some(sp) = state {
            append_line(sp, &record)?;
        }
        records.push(record);
    }
    let summary = serde_json::json!({
        "count": records.len(),
        "ok": records.iter().filter(|r| r["verdict"] == "OK").count(),
        "results": records,
    });
    let text = serde_json::to_string_pretty(&summary)?;
    match out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(p, text)?;
            eprintln!("[agent] 已写入 {}", p.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn append_line(path: &Path, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{value}")?;
    Ok(())
}

fn collect_images(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        bail!("输入路径不存在：{}", input.display());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .as_deref(),
                Some("png" | "jpg" | "jpeg" | "bmp")
            )
        })
        .collect();
    files.sort();
    Ok(files)
}
