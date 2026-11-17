//! `glue-cli`：检测单图/目录、生成合成样本、性能基准。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use glue_core::Config;
use glue_pipeline::Inspector;
use glue_vision::GrayImage;

#[derive(Parser)]
#[command(name = "glue-cli", version, about = "涂胶检测上位机（Rust）命令行入口")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 检测单张图片或整个目录
    Inspect {
        #[arg(long, default_value = "configs/default.yaml")]
        config: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        mm_per_px: Option<f64>,
    },
    /// 生成合成样例
    Gen {
        #[arg(long, default_value = "outputs/rust_samples")]
        out: PathBuf,
        #[arg(long, default_value_t = 20)]
        count: usize,
        #[arg(long, default_value = "1280x960")]
        size: String,
        /// 生成用标定比例（mm/px），决定胶条像素宽度
        #[arg(long, default_value_t = 0.0833)]
        mm_per_px: f64,
    },
    /// 性能基准
    Bench {
        #[arg(long, default_value = "configs/default.yaml")]
        config: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 50)]
        iters: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect {
            config,
            input,
            out,
            mm_per_px,
        } => inspect(&config, &input, out, mm_per_px),
        Cmd::Gen {
            out,
            count,
            size,
            mm_per_px,
        } => gen(&out, count, &size, mm_per_px),
        Cmd::Bench {
            config,
            input,
            iters,
        } => bench(&config, &input, iters),
    }
}

fn load_config(path: &Path, mm_per_px: Option<f64>) -> Result<Config> {
    let mut cfg =
        Config::from_yaml(path).with_context(|| format!("读取配置失败：{}", path.display()))?;
    if let Some(mm) = mm_per_px {
        cfg.calib.enabled = true;
        cfg.calib.mm_per_px = Some(mm);
    }
    Ok(cfg)
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

fn inspect(
    config: &Path,
    input: &Path,
    out: Option<PathBuf>,
    mm_per_px: Option<f64>,
) -> Result<()> {
    let cfg = load_config(config, mm_per_px)?;
    let inspector = Inspector::new(cfg);
    let files = collect_images(input)?;
    let mut records = Vec::with_capacity(files.len());
    for path in &files {
        let img = GrayImage::from_file(path)?;
        let result = inspector.inspect(&img);
        records.push(serde_json::json!({
            "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            "verdict": result.decision.code,
            "message": result.decision.message,
            "width_mm": result.decision.width_mm,
            "rois": result.rois.iter().map(|b| b.to_array()).collect::<Vec<_>>(),
            "measure": result.measure,
            "timings_ms": result.timings,
        }));
    }
    let summary = serde_json::json!({
        "count": records.len(),
        "ok": records.iter().filter(|r| r["verdict"] == "OK").count(),
        "results": records,
    });
    let text = serde_json::to_string_pretty(&summary)?;
    match out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&path, &text)?;
            println!("已写入 {}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn gen(out: &Path, count: usize, size: &str, mm_per_px: f64) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let (w, h) = parse_size(size)?;
    let kinds = [
        "normal",
        "narrow",
        "wide",
        "broken",
        "no_roi",
        "dark",
        "overexposed",
        "blurry",
    ];
    let mut truth = Vec::new();
    for i in 0..count {
        let kind = kinds[i % kinds.len()];
        // 按物理宽度换算像素宽度：标准 10 mm，过窄 5 mm，过宽 15 mm（按 1280 宽度等比缩放）
        let mm = match kind {
            "narrow" => 5.0,
            "wide" => 15.0,
            _ => 10.0,
        };
        let scale = (w as f64 / 1280.0).max(0.05);
        let strip_w = ((mm / mm_per_px) * scale).round().max(3.0) as usize;
        let base: u8 = match kind {
            "dark" => 6,
            "overexposed" => 252,
            "no_roi" => 90,
            _ => 18,
        };
        let mut img = GrayImage::new(w, h, base);
        if kind == "no_roi" {
            // 有对比度但无胶条：6×6 亮点、间隔 16 px（单个面积远小于 min_area_px）
            for y in 0..h {
                for x in 0..w {
                    let on = x % 16 < 6 && y % 16 < 6;
                    img.set(x, y, if on { 200 } else { 60 });
                }
            }
        }
        let draw_strip = matches!(kind, "normal" | "narrow" | "wide" | "broken" | "blurry");
        if draw_strip {
            let y0 = h / 2 - strip_w / 2;
            let (x0, x1) = (40usize, w.saturating_sub(40));
            if kind == "broken" {
                let mid = (x0 + x1) / 2;
                for y in y0..y0 + strip_w {
                    for x in x0..mid.saturating_sub(15) {
                        img.set(x, y, 225);
                    }
                    for x in mid + 15..x1 {
                        img.set(x, y, 225);
                    }
                }
            } else {
                for y in y0..y0 + strip_w {
                    for x in x0..x1 {
                        img.set(x, y, 225);
                    }
                }
            }
        }
        let mut state: u32 = (i as u32).wrapping_mul(7919).wrapping_add(17);
        let amp: i32 = match kind {
            "dark" | "overexposed" => 3,
            "no_roi" => 30,
            _ => 8,
        };
        for v in img.data.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((state >> 24) as i32 % (2 * amp + 1)) - amp;
            *v = (*v as i32 + n).clamp(0, 255) as u8;
        }
        if kind == "blurry" {
            img = box_blur(&img, 12);
        }
        let path = out.join(format!("{i:04}_{kind}.png"));
        image::GrayImage::from_raw(w as u32, h as u32, img.data.clone())
            .unwrap()
            .save(&path)?;
        truth.push(serde_json::json!({"file": path.file_name().unwrap(), "kind": kind, "strip_px": strip_w}));
    }
    std::fs::write(
        out.join("truth.json"),
        serde_json::to_string_pretty(&truth)?,
    )?;
    println!("已生成 {count} 张样例到 {}", out.display());
    Ok(())
}

fn bench(config: &Path, input: &Path, iters: usize) -> Result<()> {
    let cfg = load_config(config, None)?;
    let inspector = Inspector::new(cfg);
    let files = collect_images(input)?;
    let images: Vec<GrayImage> = files
        .iter()
        .map(GrayImage::from_file)
        .collect::<Result<Vec<_>>>()?;
    if images.is_empty() {
        bail!("目录中没有图片");
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters * images.len());
    let t0 = Instant::now();
    for _ in 0..iters {
        for img in &images {
            samples.push(inspector.inspect(img).timings.total_ms);
        }
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| samples[((samples.len() as f64 - 1.0) * p) as usize];
    println!(
        "帧数 {} ｜ p50 {:.2} ms ｜ p95 {:.2} ms ｜ max {:.2} ms ｜ 吞吐 {:.1} fps",
        samples.len(),
        pct(0.5),
        pct(0.95),
        samples.last().unwrap(),
        1000.0 * samples.len() as f64 / wall
    );
    Ok(())
}

fn box_blur(img: &GrayImage, r: usize) -> GrayImage {
    let (w, h) = (img.w, img.h);
    let mut out = GrayImage::new(w, h, 0);
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
            out.set(x, y, (sum / cnt) as u8);
        }
    }
    out
}

fn parse_size(s: &str) -> Result<(usize, usize)> {
    let mut parts = s.split('x');
    let w: usize = parts.next().unwrap_or("1280").parse()?;
    let h: usize = parts.next().unwrap_or("960").parse()?;
    Ok((w, h))
}
