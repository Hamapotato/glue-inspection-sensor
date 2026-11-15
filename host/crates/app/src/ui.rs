//! egui 界面：检测 / 配方 / 历史 / 统计 / 设置。

use eframe::egui;
use glue_core::{FrameResult, VerdictCode};
use glue_vision::GrayImage;

use crate::state::{AppState, Page};

pub struct HostApp {
    st: AppState,
    page: Page,
    texture: Option<egui::TextureHandle>,
    tex_key: String,
}

impl HostApp {
    pub fn new() -> Self {
        let mut st = AppState::new();
        st.scan();
        Self { st, page: Page::Inspect, texture: None, tex_key: String::new() }
    }

    fn ensure_texture(&mut self, ctx: &egui::Context) {
        let Some(img) = &self.st.image else { return };
        let key = format!("{}:{}x{}", self.st.current, img.w, img.h);
        if self.tex_key == key && self.texture.is_some() {
            return;
        }
        let rgba: Vec<u8> = img.data.iter().flat_map(|&v| [v, v, v, 255]).collect();
        let color = egui::ColorImage::from_rgba_unmultiplied([img.w, img.h], &rgba);
        self.texture = Some(ctx.load_texture(key.clone(), color, egui::TextureOptions::LINEAR));
        self.tex_key = key;
    }

    fn nav(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("nav").resizable(false).default_width(150.0).show(ctx, |ui| {
            ui.add_space(8.0);
            ui.heading("涂胶检测");
            ui.separator();
            for (p, label) in [
                (Page::Inspect, "检测"),
                (Page::Recipe, "配方"),
                (Page::History, "历史"),
                (Page::Stats, "统计"),
                (Page::Settings, "设置"),
            ] {
                if ui.selectable_label(self.page == p, label).clicked() {
                    self.page = p;
                }
            }
            ui.separator();
            ui.label(format!("图像 {} 张", self.st.files.len()));
        });
    }

    fn bottom(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.st.status);
            });
        });
    }

    fn inspect_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("图像目录");
            ui.add(egui::TextEdit::singleline(&mut self.st.dir).desired_width(420.0));
            if ui.button("扫描").clicked() {
                self.st.scan();
            }
            if ui.button("上一张").clicked() {
                if self.st.current > 0 {
                    self.st.current -= 1;
                    self.st.load_current();
                }
            }
            if ui.button("下一张").clicked() {
                if self.st.current + 1 < self.st.files.len() {
                    self.st.current += 1;
                    self.st.load_current();
                }
            }
            if ui.button("检测当前").clicked() {
                self.st.detect_current();
            }
            if ui.button("批量检测").clicked() {
                self.st.detect_all();
            }
        });
        ui.separator();
        ui.horizontal_top(|ui| {
            self.image_view(ui);
            ui.separator();
            self.result_panel(ui);
        });
    }

    fn image_view(&mut self, ui: &mut egui::Ui) {
        let Some(img) = &self.st.image else {
            ui.label("请先扫描并加载图像");
            return;
        };
        let (iw, ih) = (img.w as f32, img.h as f32);
        let avail = ui.available_width().max(200.0) - 260.0;
        let scale = (avail / iw).min(600.0 / ih).max(0.05);
        let size = egui::vec2(iw * scale, ih * scale);
        let tex = self.texture.as_ref().map(|t| t.id());
        let Some(tex_id) = tex else {
            ui.label("纹理未就绪");
            return;
        };
        let resp = ui.add(egui::Image::new((tex_id, size)).fit_to_exact_size(size));
        let rect = resp.rect;
        let painter = ui.painter_at(rect);
        if let Some(r) = &self.st.result {
            draw_overlay(&painter, rect, r, scale);
        }
    }

    fn result_panel(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.set_width(240.0);
            ui.heading("检测结果");
            if let Some(r) = &self.st.result {
                let color = verdict_color(r.decision.code);
                ui.colored_label(color, format!("{:?}", r.decision.code));
                ui.label(&r.decision.message);
                if let Some(m) = &r.measure {
                    ui.label(format!("面积 {:.0} px", m.area_px));
                    ui.label(format!("长度 {:.1} px", m.length_px));
                    ui.label(format!("宽度 {:.2} px = {:.2} mm", m.width_px, m.width_mm.unwrap_or(0.0)));
                }
                ui.separator();
                ui.label(format!("定位 {:.1} ms", r.timings.locate_ms));
                ui.label(format!("量测 {:.1} ms", r.timings.measure_ms));
                ui.label(format!("总计 {:.1} ms", r.timings.total_ms));
            } else {
                ui.label("尚未检测");
            }
        });
    }

    fn recipe_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("配方");
        let w = &mut self.st.cfg.measure.width;
        egui::Grid::new("recipe").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label("标准宽度 (mm)");
            ui.add(egui::DragValue::new(&mut w.standard_mm).speed(0.05).range(0.0..=50.0));
            ui.end_row();
            ui.label("预警线 (mm)");
            ui.add(egui::DragValue::new(&mut w.warn_delta_mm).speed(0.05).range(0.0..=10.0));
            ui.end_row();
            ui.label("报警线 (mm)");
            ui.add(egui::DragValue::new(&mut w.alarm_delta_mm).speed(0.05).range(0.0..=10.0));
            ui.end_row();
            ui.label("标定 mm/px");
            let mut mm = self.st.mm_per_px.unwrap_or(0.0);
            if ui.add(egui::DragValue::new(&mut mm).speed(0.0001).range(0.0..=10.0)).changed() {
                self.st.mm_per_px = if mm > 0.0 { Some(mm) } else { None };
            }
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("保存配方").clicked() {
                self.st.save_recipe();
            }
            if ui.button("加载配方").clicked() {
                self.st.load_recipe();
            }
            if ui.button("重新加载配置").clicked() {
                self.st.reload_config();
            }
        });
        ui.label(format!("配方文件：{}", self.st.recipe_path));
    }

    fn history_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("历史记录");
            if ui.button("刷新").clicked() {
                self.st.refresh_history();
            }
            if ui.button("清空").clicked() {
                self.st.clear_history();
            }
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("history").striped(true).num_columns(5).show(ui, |ui| {
                ui.strong("时间");
                ui.strong("文件");
                ui.strong("判定");
                ui.strong("宽度 mm");
                ui.strong("耗时 ms");
                ui.end_row();
                for r in &self.st.history {
                    ui.label(&r.ts);
                    ui.label(&r.file);
                    ui.colored_label(verdict_color_by_name(&r.verdict), &r.verdict);
                    ui.label(r.width_mm.map(|w| format!("{w:.2}")).unwrap_or_default());
                    ui.label(format!("{:.1}", r.total_ms));
                    ui.end_row();
                }
            });
        });
    }

    fn stats_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("统计");
            if ui.button("刷新").clicked() {
                self.st.refresh_stats();
            }
        });
        let Some(s) = &self.st.stats else {
            ui.label("暂无数据");
            return;
        };
        ui.label(format!("总检测：{} 次", s.total));
        ui.label(format!("合格率：{:.1}%", s.ok_rate * 100.0));
        ui.label(format!(
            "宽度：均值 {} / 最小 {} / 最大 {} mm",
            fmt_opt(s.width_mean),
            fmt_opt(s.width_min),
            fmt_opt(s.width_max)
        ));
        ui.label(format!("耗时：p50 {:.1} ms ｜ p95 {:.1} ms", s.lat_p50, s.lat_p95));
        ui.add_space(8.0);
        egui::Grid::new("verdicts").striped(true).num_columns(2).show(ui, |ui| {
            for (k, n) in &s.by_verdict {
                ui.colored_label(verdict_color_by_name(k), k);
                ui.label(format!("{n}"));
                ui.end_row();
            }
        });
    }

    fn settings_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("设置");
        egui::Grid::new("settings").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label("配置文件");
            ui.add(egui::TextEdit::singleline(&mut self.st.cfg_path).desired_width(420.0));
            ui.end_row();
            ui.label("数据库");
            ui.add(egui::TextEdit::singleline(&mut self.st.db_path).desired_width(420.0));
            ui.end_row();
            ui.label("配方文件");
            ui.add(egui::TextEdit::singleline(&mut self.st.recipe_path).desired_width(420.0));
            ui.end_row();
        });
        if ui.button("重新加载配置").clicked() {
            self.st.reload_config();
        }
        ui.label("提示：数据库路径修改后需重启生效；文件回放模式不接相机。");
    }
}

impl eframe::App for HostApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_texture(ctx);
        self.nav(ctx);
        self.bottom(ctx);
        egui::CentralPanel::default().show(ctx, |ui| match self.page {
            Page::Inspect => self.inspect_page(ui),
            Page::Recipe => self.recipe_page(ui),
            Page::History => self.history_page(ui),
            Page::Stats => self.stats_page(ui),
            Page::Settings => self.settings_page(ui),
        });
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into())
}

fn verdict_color(code: VerdictCode) -> egui::Color32 {
    match code {
        VerdictCode::Ok => egui::Color32::from_rgb(60, 180, 90),
        VerdictCode::NoRoi => egui::Color32::GRAY,
        VerdictCode::Broken => egui::Color32::from_rgb(220, 60, 60),
        VerdictCode::TooNarrow | VerdictCode::TooWide => egui::Color32::from_rgb(230, 120, 40),
        VerdictCode::WarnNarrow | VerdictCode::WarnWide => egui::Color32::from_rgb(220, 190, 40),
        VerdictCode::ReflectionSuspect => egui::Color32::from_rgb(150, 90, 220),
    }
}

fn verdict_color_by_name(name: &str) -> egui::Color32 {
    match name {
        "OK" => egui::Color32::from_rgb(60, 180, 90),
        "NO_ROI" => egui::Color32::GRAY,
        "BROKEN" => egui::Color32::from_rgb(220, 60, 60),
        "TOO_NARROW" | "TOO_WIDE" => egui::Color32::from_rgb(230, 120, 40),
        _ => egui::Color32::from_rgb(220, 190, 40),
    }
}

fn draw_overlay(painter: &egui::Painter, rect: egui::Rect, result: &FrameResult, scale: f32) {
    for b in &result.rois {
        let r = egui::Rect::from_min_max(
            egui::pos2(rect.left() + b.x1 as f32 * scale, rect.top() + b.y1 as f32 * scale),
            egui::pos2(rect.left() + b.x2 as f32 * scale, rect.top() + b.y2 as f32 * scale),
        );
        painter.rect_stroke(r, 0.0, egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 200, 120)), egui::StrokeKind::Inside);
    }
    if let Some(m) = &result.measure {
        let text = format!("宽 {:.2} mm", m.width_mm.unwrap_or(0.0));
        painter.text(
            egui::pos2(rect.left() + 8.0, rect.top() + 8.0),
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::proportional(18.0),
            egui::Color32::from_rgb(255, 230, 120),
        );
    }
    let _ = GrayImage::new(1, 1, 0);
}
