//! 界面状态与业务动作（与 egui 解耦，便于单元测试与无头模式）。

use std::path::{Path, PathBuf};

use glue_core::{Config, FrameResult};
use glue_pipeline::Inspector;
use glue_store::{Record, Stats, Store};
use glue_vision::GrayImage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Inspect,
    Camera,
    Recipe,
    History,
    Stats,
    Settings,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Recipe {
    pub standard_mm: f64,
    pub warn_delta_mm: f64,
    pub alarm_delta_mm: f64,
    pub mm_per_px: Option<f64>,
}

pub struct AppState {
    pub cfg_path: String,
    pub cfg: Config,
    pub mm_per_px: Option<f64>,
    pub db_path: String,
    pub store: Store,
    pub dir: String,
    pub files: Vec<PathBuf>,
    pub current: usize,
    pub image: Option<GrayImage>,
    pub result: Option<FrameResult>,
    pub status: String,
    pub history: Vec<Record>,
    pub stats: Option<Stats>,
    pub recipe_path: String,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// 按配置构建 Inspector（推理器可选注入；推理配置有误时告警并回退启发式）。
pub fn build_inspector(cfg: &Config, mm_per_px: Option<f64>) -> Inspector {
    let mut cfg = cfg.clone();
    if mm_per_px.is_some() {
        cfg.calib.enabled = true;
        cfg.calib.mm_per_px = mm_per_px;
    }
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

impl AppState {
    pub fn new() -> Self {
        let cfg_path = env_or("GLUE_CONFIG", "../configs/default.yaml");
        let cfg = Config::from_yaml(&cfg_path).unwrap_or_default();
        let mm_per_px = cfg.mm_per_px();
        let db_path = env_or("GLUE_DB", "../outputs/host.db");
        let store =
            Store::open(&db_path).unwrap_or_else(|_| Store::open_memory().expect("memory store"));
        Self {
            cfg_path,
            cfg,
            mm_per_px,
            db_path,
            store,
            dir: env_or("GLUE_SAMPLES", "../outputs/rust_samples"),
            files: Vec::new(),
            current: 0,
            image: None,
            result: None,
            status: "就绪".into(),
            history: Vec::new(),
            stats: None,
            recipe_path: env_or("GLUE_RECIPE", "../outputs/recipe.json"),
        }
    }

    pub fn reload_config(&mut self) {
        match Config::from_yaml(&self.cfg_path) {
            Ok(c) => {
                self.mm_per_px = c.mm_per_px();
                self.cfg = c;
                self.status = format!("已加载配置 {}", self.cfg_path);
            }
            Err(e) => self.status = format!("配置加载失败：{e}"),
        }
    }

    pub fn inspector(&self) -> Inspector {
        build_inspector(&self.cfg, self.mm_per_px)
    }

    pub fn scan(&mut self) {
        let dir = Path::new(&self.dir);
        if !dir.is_dir() {
            self.status = format!("目录不存在：{}", self.dir);
            return;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
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
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        self.files = files;
        self.current = 0;
        self.status = format!("扫描到 {} 张图", self.files.len());
        self.load_current();
    }

    pub fn load_current(&mut self) {
        self.image = None;
        self.result = None;
        let Some(path) = self.files.get(self.current) else {
            return;
        };
        match GrayImage::from_file(path) {
            Ok(img) => {
                self.image = Some(img);
                self.status = format!("已加载 {}", path.display());
            }
            Err(e) => self.status = format!("读取失败：{e}"),
        }
    }

    pub fn detect_current(&mut self) {
        let Some(img) = &self.image else {
            self.status = "没有可检测的图像".into();
            return;
        };
        let insp = self.inspector();
        let r = insp.inspect(img);
        let file = self
            .files
            .get(self.current)
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let verdict = r.decision.code.as_str().to_string();
        let width = r.decision.width_mm;
        let total = r.timings.total_ms;
        if let Err(e) = self.store.insert(&file, &verdict, width, total) {
            self.status = format!("入库失败：{e}");
        } else {
            self.status = format!(
                "{file}: {verdict}，{:.2} mm，{:.1} ms",
                width.unwrap_or(0.0),
                total
            );
        }
        self.result = Some(r);
        self.refresh_history();
        self.refresh_stats();
    }

    pub fn detect_all(&mut self) -> usize {
        let insp = self.inspector();
        let mut count = 0;
        for (i, path) in self.files.iter().enumerate() {
            let Ok(img) = GrayImage::from_file(path) else {
                continue;
            };
            let r = insp.inspect(&img);
            let file = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let _ = self.store.insert(
                &file,
                r.decision.code.as_str(),
                r.decision.width_mm,
                r.timings.total_ms,
            );
            count += 1;
            if i == self.current {
                self.result = Some(r);
            }
        }
        self.status = format!("批量检测完成：{count} 张");
        self.refresh_history();
        self.refresh_stats();
        count
    }

    pub fn refresh_history(&mut self) {
        self.history = self.store.recent(200).unwrap_or_default();
    }

    pub fn refresh_stats(&mut self) {
        self.stats = self.store.stats().ok();
    }

    pub fn save_recipe(&mut self) {
        let recipe = Recipe {
            standard_mm: self.cfg.measure.width.standard_mm,
            warn_delta_mm: self.cfg.measure.width.warn_delta_mm,
            alarm_delta_mm: self.cfg.measure.width.alarm_delta_mm,
            mm_per_px: self.mm_per_px,
        };
        if let Some(p) = Path::new(&self.recipe_path).parent() {
            std::fs::create_dir_all(p).ok();
        }
        match serde_json::to_string_pretty(&recipe)
            .map_err(anyhow::Error::from)
            .and_then(|s| {
                std::fs::write(&self.recipe_path, s)?;
                Ok(())
            }) {
            Ok(()) => self.status = format!("配方已保存 {}", self.recipe_path),
            Err(e) => self.status = format!("配方保存失败：{e}"),
        }
    }

    pub fn load_recipe(&mut self) {
        let Ok(text) = std::fs::read_to_string(&self.recipe_path) else {
            self.status = "配方文件不存在".into();
            return;
        };
        match serde_json::from_str::<Recipe>(&text) {
            Ok(r) => {
                self.cfg.measure.width.standard_mm = r.standard_mm;
                self.cfg.measure.width.warn_delta_mm = r.warn_delta_mm;
                self.cfg.measure.width.alarm_delta_mm = r.alarm_delta_mm;
                self.mm_per_px = r.mm_per_px;
                self.status = "配方已加载".into();
            }
            Err(e) => self.status = format!("配方解析失败：{e}"),
        }
    }

    pub fn clear_history(&mut self) {
        self.store.clear().ok();
        self.refresh_history();
        self.refresh_stats();
        self.status = "历史记录已清空".into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_roundtrip() {
        std::fs::create_dir_all("target").ok();
        let path =
            std::path::Path::new("target").join(format!("glue_recipe_{}.json", std::process::id()));
        let r = Recipe {
            standard_mm: 10.5,
            warn_delta_mm: 0.4,
            alarm_delta_mm: 0.9,
            mm_per_px: Some(0.08),
        };
        std::fs::write(&path, serde_json::to_string(&r).unwrap()).unwrap();
        let back: Recipe = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!((back.standard_mm - 10.5).abs() < 1e-9);
        assert_eq!(back.mm_per_px, Some(0.08));
    }
}
