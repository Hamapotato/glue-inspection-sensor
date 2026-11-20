//! GUI 相机采集工作线程：后台 grab → inspect → mpsc 送 UI，UI 线程不做阻塞采集（AGENTS H4）。

use eframe::egui;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use glue_core::{Config, FrameResult};
use glue_pipeline::Inspector;
use glue_vision::GrayImage;

use crate::state::build_inspector;

pub enum CamMsg {
    Frame { img: GrayImage, result: FrameResult, seq: u64 },
    Error(String),
    Stopped,
}

pub struct CameraWorker {
    stop: Arc<AtomicBool>,
    rx: Receiver<CamMsg>,
    join: Option<JoinHandle<()>>,
    pub started: Instant,
}

impl CameraWorker {
    pub fn start(cfg: Config, mm_per_px: Option<f64>, ctx: egui::Context) -> Result<Self, String> {
        let mut run_cfg = cfg;
        if let Some(mm) = mm_per_px {
            run_cfg.calib.enabled = true;
            run_cfg.calib.mm_per_px = Some(mm);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let join = std::thread::Builder::new()
            .name("glue-camera".into())
            .spawn(move || {
                let inspector: Inspector = build_inspector(&run_cfg, None);
                let mut src = match glue_camera::open(&run_cfg.camera) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(CamMsg::Error(format!("打开相机失败：{e}")));
                        return;
                    }
                };
                let mut seq = 0u64;
                while !stop_thread.load(Ordering::Relaxed) {
                    match src.grab() {
                        Ok(f) => {
                            let img = GrayImage { w: f.width, h: f.height, data: f.data };
                            let result = inspector.inspect(&img);
                            if tx.send(CamMsg::Frame { img, result, seq }).is_err() {
                                break;
                            }
                            seq += 1;
                            ctx.request_repaint();
                        }
                        Err(e) => {
                            let _ = tx.send(CamMsg::Error(format!("采集失败：{e}")));
                            break;
                        }
                    }
                }
                let _ = tx.send(CamMsg::Stopped);
            })
            .map_err(|e| format!("启动采集线程失败：{e}"))?;
        Ok(Self { stop, rx, join: Some(join), started: Instant::now() })
    }

    pub fn try_recv(&self) -> Option<CamMsg> {
        match self.rx.try_recv() {
            Ok(m) => Some(m),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(CamMsg::Stopped),
        }
    }
}

impl Drop for CameraWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}
