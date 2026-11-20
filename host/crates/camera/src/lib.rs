//! `glue-camera`：相机采集抽象（当前实现为"命令管道"后端）。
//!
//! 设计：相机采集与检测解耦，`FrameSource` 输出统一灰帧；具体相机 SDK
//! （如海康 MVS）由外部采集脚本通过 stdout 按协议送入，Rust 侧不直接依赖 SDK，
//! 便于在 Windows/板端使用同一套软件，也避免为每家 SDK 维护 FFI。
//!
//! 帧协议（小端）：
//! ```text
//! magic: b"GLUE" (4 bytes) | width: u32 | height: u32 | len: u32 | data: [u8; len]
//! ```
//! data 为行优先 8-bit 灰度（Mono8）；len 必须等于 width*height。

use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use anyhow::{bail, Context, Result};
use glue_core::CameraCfg;

pub const MAGIC: [u8; 4] = *b"GLUE";
const MAX_SIDE: u32 = 16384;
const MAX_BYTES: usize = 64 * 1024 * 1024;

/// 一帧灰度图像（行优先，Mono8）。
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub seq: u64,
}

impl Frame {
    pub fn new(width: usize, height: usize, data: Vec<u8>, seq: u64) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("帧尺寸非法：{width}x{height}");
        }
        if data.len() != width * height {
            bail!("帧数据长度不匹配：{} != {width}*{height}", data.len());
        }
        Ok(Self { width, height, data, seq })
    }
}

/// 相机帧源。`grab` 阻塞到下一帧；关闭由 `Drop` 负责。
pub trait FrameSource {
    fn grab(&mut self) -> Result<Frame>;
}

/// 从任意 `Read` 读取一帧（协议见模块注释）；供测试与命令后端复用。
pub fn read_frame<R: Read>(reader: &mut R, seq: u64) -> Result<Frame> {
    let mut head = [0u8; 16];
    if let Err(e) = reader.read_exact(&mut head) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            bail!("采集流已结束（EOF）");
        }
        return Err(e).context("读取帧头失败");
    }
    if head[..4] != MAGIC {
        bail!("帧头 magic 错误：{:?}（采集脚本需按 GLUE 协议输出）", &head[..4]);
    }
    let width = u32::from_le_bytes(head[4..8].try_into().expect("4 bytes")) as usize;
    let height = u32::from_le_bytes(head[8..12].try_into().expect("4 bytes")) as usize;
    let len = u32::from_le_bytes(head[12..16].try_into().expect("4 bytes")) as usize;
    if width == 0 || height == 0 || width > MAX_SIDE as usize || height > MAX_SIDE as usize {
        bail!("帧尺寸非法：{width}x{height}");
    }
    if len > MAX_BYTES || len != width * height {
        bail!("帧长度非法：len={len}，期望 {}", width * height);
    }
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data).with_context(|| format!("读取帧数据失败（期望 {len} 字节）"))?;
    Frame::new(width, height, data, seq)
}

/// 命令管道帧源：spawn 一个采集进程，从其 stdout 读取帧。
pub struct CommandSource {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    seq: u64,
    command: String,
}

impl CommandSource {
    pub fn open(cfg: &CameraCfg) -> Result<Self> {
        let cmd = cfg
            .command
            .as_ref()
            .context("camera.command 未配置（例如 python3 scripts/camera_mvs.py --fps 10）")?;
        let (shell, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
        let mut child = Command::new(shell)
            .arg(flag)
            .arg(cmd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("启动采集命令失败：{cmd}"))?;
        let stdout = child.stdout.take().context("采集进程没有 stdout")?;
        Ok(Self { child, reader: BufReader::new(stdout), seq: 0, command: cmd.clone() })
    }

    pub fn command(&self) -> &str {
        &self.command
    }
}

impl FrameSource for CommandSource {
    fn grab(&mut self) -> Result<Frame> {
        let frame = read_frame(&mut self.reader, self.seq)?;
        self.seq += 1;
        Ok(frame)
    }
}

impl Drop for CommandSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 按配置打开帧源。当前仅支持 `source=command`。
pub fn open(cfg: &CameraCfg) -> Result<Box<dyn FrameSource>> {
    if !cfg.enabled {
        bail!("camera.enabled=false");
    }
    match cfg.source.as_str() {
        "command" => Ok(Box::new(CommandSource::open(cfg)?)),
        other => bail!("camera.source={other} 暂只支持 command"),
    }
}

/// 可选：命令的默认解释器路径提示（文档/报错用）。
pub fn command_hint() -> PathBuf {
    PathBuf::from(if cfg!(windows) { "cmd /C" } else { "sh -c" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame_bytes(w: u32, h: u32, fill: u8) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&(w * h).to_le_bytes());
        v.extend(std::iter::repeat_n(fill, (w * h) as usize));
        v
    }

    #[test]
    fn reads_valid_frame() {
        let raw = frame_bytes(4, 3, 7);
        let mut cur = Cursor::new(raw);
        let f = read_frame(&mut cur, 0).expect("frame");
        assert_eq!((f.width, f.height), (4, 3));
        assert_eq!(f.data, vec![7u8; 12]);
    }

    #[test]
    fn reads_consecutive_frames() {
        let mut raw = frame_bytes(2, 2, 1);
        raw.extend(frame_bytes(2, 2, 2));
        let mut cur = Cursor::new(raw);
        assert_eq!(read_frame(&mut cur, 0).unwrap().data[0], 1);
        assert_eq!(read_frame(&mut cur, 1).unwrap().data[0], 2);
        assert!(read_frame(&mut cur, 2).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = frame_bytes(2, 2, 0);
        raw[0] = b'X';
        assert!(read_frame(&mut Cursor::new(raw), 0).is_err());
    }

    #[test]
    fn rejects_size_mismatch() {
        let mut raw = frame_bytes(2, 2, 0);
        raw[12..16].copy_from_slice(&9u32.to_le_bytes());
        assert!(read_frame(&mut Cursor::new(raw), 0).is_err());
    }

    #[test]
    fn frame_validates_shape() {
        assert!(Frame::new(0, 1, vec![], 0).is_err());
        assert!(Frame::new(2, 2, vec![0; 3], 0).is_err());
        assert!(Frame::new(2, 2, vec![0; 4], 0).is_ok());
    }
}
