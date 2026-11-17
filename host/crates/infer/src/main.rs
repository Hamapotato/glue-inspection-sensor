//! `glue-infer`：Rust 端 ONNX 推理（tract，纯 Rust，无 C++ 依赖）。
//!
//! 子命令：
//! - `run`：跑模型原始输出（raw f32 + JSON 元数据），用于与 Python/onnxruntime 对拍；
//! - `rpn`：RPN 原始输出 → 候选框（anchors + 解码 + 分数 + NMS）；
//! - `yolo`：YOLO 原始输出 → 检测框（解码 + NMS）。
//!
//! 本 crate 只做推理，不做判定；判定仍在 `glue-core`。

use std::path::Path;

use anyhow::{bail, Context, Result};
use glue_infer::{rpn, run_model, yolo};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        print_help();
        return Ok(());
    };
    match cmd.as_str() {
        "run" => cmd_run(&args[1..]),
        "rpn" => cmd_rpn(&args[1..]),
        "yolo" => cmd_yolo(&args[1..]),
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => bail!("未知子命令：{other}"),
    }
}

fn print_help() {
    println!(
        "用法：
  glue-infer run  --model <onnx> --input <raw f32> --in-shape WxH --out <raw f32> [--meta json]
  glue-infer rpn  --model <onnx> --input <raw f32> --in-shape WxH --out <boxes.json> [--score-thr 0.5]
  glue-infer yolo --model <onnx> --input <raw f32> --in-shape WxH --out <dets.json> [--score-thr 0.5]
说明：--in-shape 为单通道图像宽x高（如 320x320）；输入为行优先 f32。"
    );
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn parse_shape(s: &str) -> Result<(usize, usize)> {
    let mut it = s.split('x');
    let w = it.next().unwrap_or("").parse::<usize>().context("宽度")?;
    let h = it.next().unwrap_or("").parse::<usize>().context("高度")?;
    Ok((w, h))
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        bail!("输入文件长度不是 4 的倍数：{}", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_f32(path: &Path, data: &[f32]) -> Result<()> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<()> {
    let model = arg(args, "--model").context("缺少 --model")?;
    let input = arg(args, "--input").context("缺少 --input")?;
    let shape = parse_shape(arg(args, "--in-shape").context("缺少 --in-shape")?)?;
    let out = arg(args, "--out").context("缺少 --out")?;
    let meta = arg(args, "--meta");

    let data = read_f32(Path::new(input))?;
    let outputs = run_model(Path::new(model), &data, shape)?;

    let mut flat = Vec::new();
    let mut infos = Vec::new();
    for (i, (s, d)) in outputs.iter().enumerate() {
        infos.push(serde_json::json!({
            "index": i, "shape": s, "len": d.len(), "offset": flat.len()
        }));
        flat.extend_from_slice(d);
    }
    write_f32(Path::new(out), &flat)?;
    let meta_json =
        serde_json::json!({"model": model, "inputs": [1, 1, shape.1, shape.0], "outputs": infos});
    if let Some(m) = meta {
        std::fs::write(m, serde_json::to_string_pretty(&meta_json)?)?;
    }
    println!("{}", serde_json::to_string(&meta_json)?);
    Ok(())
}

fn cmd_rpn(args: &[String]) -> Result<()> {
    let model = arg(args, "--model").context("缺少 --model")?;
    let input = arg(args, "--input").context("缺少 --input")?;
    let shape = parse_shape(arg(args, "--in-shape").context("缺少 --in-shape")?)?;
    let out = arg(args, "--out").context("缺少 --out")?;
    let score_thr: f32 = arg(args, "--score-thr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    let data = read_f32(Path::new(input))?;
    let outputs = run_model(Path::new(model), &data, shape)?;
    if outputs.len() < 2 {
        bail!("RPN 模型应输出 cls 与 reg 两个张量");
    }
    let (_cls_shape, cls) = &outputs[0];
    let (_reg_shape, reg) = &outputs[1];
    let n = cls.len();
    let h = shape.1 / 16;
    let w = shape.0 / 16;
    if n != h * w * 9 {
        bail!("cls 长度 {n} 与 feat {h}x{w}×9 不匹配");
    }
    let anchors = rpn::anchors(h, w, 16, 9);
    let boxes = rpn::proposals(&anchors, reg, cls, score_thr, 1000, 300, 8.0);
    let json = serde_json::json!({
        "count": boxes.len(),
        "boxes": boxes.iter().map(|b| serde_json::json!({
            "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "score": b.score
        })).collect::<Vec<_>>()
    });
    std::fs::write(out, serde_json::to_string_pretty(&json)?)?;
    println!("{}", serde_json::to_string(&json)?);
    Ok(())
}

fn cmd_yolo(args: &[String]) -> Result<()> {
    let model = arg(args, "--model").context("缺少 --model")?;
    let input = arg(args, "--input").context("缺少 --input")?;
    let shape = parse_shape(arg(args, "--in-shape").context("缺少 --in-shape")?)?;
    let out = arg(args, "--out").context("缺少 --out")?;
    let score_thr: f32 = arg(args, "--score-thr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    let data = read_f32(Path::new(input))?;
    let outputs = run_model(Path::new(model), &data, shape)?;
    let (o_shape, o) = &outputs[0];
    let dets = yolo::decode(o, shape, o_shape, score_thr)?;
    let json = serde_json::json!({
        "count": dets.len(),
        "detections": dets.iter().map(|d| serde_json::json!({
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "score": d.score
        })).collect::<Vec<_>>()
    });
    std::fs::write(out, serde_json::to_string_pretty(&json)?)?;
    println!("{}", serde_json::to_string(&json)?);
    Ok(())
}
