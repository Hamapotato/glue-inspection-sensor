//! AidLite 推理后端（feature `aidlite`，QCS8550/AidLux 板端）。
//!
//! AidLite 只提供 C++（mangled）与 Python 接口，没有稳定的 C ABI；这里通过
//! **常驻 Python 子进程**调用（`scripts/aidlite_server.py`），协议见该脚本注释。
//! 预处理/anchors/解码/NMS 与 ONNX 路径完全共用，便于直接对比耗时。

use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use glue_core::{BBox, ObjectDetector};

use crate::{letterbox, rpn, unletterbox};

const REQ_MAGIC: [u8; 4] = *b"AIDL";
const RESP_MAGIC: [u8; 4] = *b"RSLT";
const ERR_MAGIC: [u8; 4] = *b"ERR!";

pub struct AidliteDetector {
    child: Child,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    size: usize,
    score_thr: f32,
    pre_topk: usize,
    post_topk: usize,
    min_size: f32,
    anchors_wh: Option<Vec<(f32, f32)>>,
}

impl AidliteDetector {
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
        let cmd_string = if let Ok(cmd) = std::env::var("GLUE_AIDLITE_CMD") {
            cmd
        } else {
            let python = std::env::var("GLUE_AIDLITE_PYTHON").unwrap_or_else(|_| "python3".into());
            let script = std::env::var("GLUE_AIDLITE_SCRIPT")
                .unwrap_or_else(|_| "scripts/aidlite_server.py".into());
            let accel = std::env::var("GLUE_AIDLITE_ACCEL").unwrap_or_else(|_| "cpu".into());
            format!("{python} {script} --model {} --accel {accel}", model.display())
        };
        let (shell, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
        let mut child = Command::new(shell)
            .arg(flag)
            .arg(&cmd_string)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("启动 AidLite 服务失败：{cmd_string}"))?;
        let stdin = child.stdin.take().context("AidLite 子进程没有 stdin")?;
        let stdout = child.stdout.take().context("AidLite 子进程没有 stdout")?;
        Ok(Self {
            child,
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            size,
            score_thr,
            pre_topk,
            post_topk,
            min_size,
            anchors_wh,
        })
    }
}

impl ObjectDetector for AidliteDetector {
    fn detect(&self, gray: &[u8], w: usize, h: usize) -> Result<Vec<BBox>> {
        let (canvas, scale, pad_x, pad_y) = letterbox(gray, w, h, self.size)?;
        let payload = canvas.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();

        // 请求：magic + len + f32 payload
        {
            let mut stdin = self.stdin.lock().map_err(|_| anyhow::anyhow!("AidLite stdin 锁被毒化"))?;
            stdin.write_all(&REQ_MAGIC)?;
            stdin.write_all(&(payload.len() as u32).to_le_bytes())?;
            stdin.write_all(&payload)?;
            stdin.flush()?;
        }

        // 响应头
        let mut head = [0u8; 8];
        let mut stdout = self.stdout.lock().map_err(|_| anyhow::anyhow!("AidLite stdout 锁被毒化"))?;
        let stdout = &mut *stdout;
        read_exact(stdout, &mut head)?;
        if head[..4] == ERR_MAGIC {
            let mut lenb = [0u8; 4];
            read_exact(stdout, &mut lenb)?;
            let len = u32::from_le_bytes(lenb) as usize;
            let mut msg = vec![0u8; len.min(4096)];
            read_exact(stdout, &mut msg)?;
            bail!("AidLite 服务错误：{}", String::from_utf8_lossy(&msg));
        }
        if head[..4] != RESP_MAGIC {
            bail!("AidLite 响应 magic 错误：{:?}", &head[..4]);
        }
        let cls_len = u32::from_le_bytes(head[4..8].try_into().expect("4 bytes")) as usize;
        let mut lenb = [0u8; 4];
        read_exact(stdout, &mut lenb)?;
        let reg_len = u32::from_le_bytes(lenb) as usize;
        let mut cls_bytes = vec![0u8; cls_len * 4];
        read_exact(stdout, &mut cls_bytes)?;
        let mut reg_bytes = vec![0u8; reg_len * 4];
        read_exact(stdout, &mut reg_bytes)?;
        let cls: Vec<f32> = cls_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        let reg: Vec<f32> = reg_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();

        let f = self.size / 16;
        let n = f * f * 9;
        if cls.len() != n || reg.len() != n * 4 {
            bail!("AidLite 输出形状不匹配：cls={} reg={}，期望 cls={n} reg={}", cls.len(), reg.len(), n * 4);
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

fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<()> {
    reader.read_exact(buf).context("读取 AidLite 响应失败（服务退出？）")
}

impl Drop for AidliteDetector {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
