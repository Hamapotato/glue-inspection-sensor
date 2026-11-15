//! egui 界面：检测 / 配方 / 历史 / 统计 / 设置（浅色工业 HMI 风格）。

use eframe::egui;
use eframe::egui::Color32;
use glue_core::{FrameResult, VerdictCode};
use glue_vision::GrayImage;

use crate::state::{AppState, Page};

const BG: Color32 = Color32::from_rgb(244, 246, 249);
const NAV_BG: Color32 = Color32::from_rgb(30, 41, 59);
const NAV_ACTIVE: Color32 = Color32::from_rgb(37, 99, 235);
const TEXT: Color32 = Color32::from_rgb(17, 24, 39);
const MUTED: Color32 = Color32::from_rgb(107, 114, 128);
const OK: Color32 = Color32::from_rgb(22, 163, 74);
const WARN: Color32 = Color32::from_rgb(217, 119, 6);
const ALARM: Color32 = Color32::from_rgb(220, 38, 38);
const NEUTRAL: Color32 = Color32::from_rgb(107, 114, 128);

pub struct HostApp {
    st: AppState,
    page: Page,
    texture: Option<egui::TextureHandle>,
    tex_key: String,
}

impl HostApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        install_style(&cc.egui_ctx);
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
        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(168.0)
            .frame(egui::Frame::default().fill(NAV_BG).inner_margin(14.0))
            .show(ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(Color32::from_rgb(226, 232, 240));
                ui.add_space(6.0);
                ui.label(egui::RichText::new("涂胶检测").size(22.0).strong().color(Color32::WHITE));
                ui.label(egui::RichText::new("Glue Inspection HMI").size(11.0).color(Color32::from_rgb(148, 163, 184)));
                ui.add_space(16.0);
                for (p, icon, label) in [
                    (Page::Inspect, "🔍", "检测"),
                    (Page::Recipe, "📋", "配方"),
                    (Page::History, "🕘", "历史"),
                    (Page::Stats, "📊", "统计"),
                    (Page::Settings, "⚙", "设置"),
                ] {
                    let text = egui::RichText::new(format!("{icon}  {label}")).size(16.0);
                    let selected = self.page == p;
                    let btn = egui::Button::new(text)
                        .fill(if selected { NAV_ACTIVE } else { Color32::TRANSPARENT })
                        .min_size(egui::vec2(140.0, 34.0));
                    if ui.add(btn).clicked() {
                        self.page = p;
                    }
                }
                ui.add_space(6.0);
                ui.separator();
                ui.label(egui::RichText::new(format!("图像 {} 张", self.st.files.len())).size(12.0));
                ui.label(egui::RichText::new(format!("λ {:.1} ms", self.st.result.as_ref().map(|r| r.timings.total_ms).unwrap_or(0.0))).size(12.0));
            });
    }

    fn bottom(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status")
            .frame(egui::Frame::default().fill(Color32::WHITE).inner_margin(8.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let (dot, color) = match self.st.result.as_ref().map(|r| r.decision.code) {
                        Some(VerdictCode::Ok) => ("●", OK),
                        Some(c) if c.is_defect() => ("●", ALARM),
                        Some(VerdictCode::NoRoi) => ("●", NEUTRAL),
                        _ => ("○", MUTED),
                    };
                    ui.colored_label(color, dot);
                    ui.label(egui::RichText::new(&self.st.status).size(13.0).color(TEXT));
                });
            });
    }

    fn inspect_page(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("图像目录").size(13.0).color(MUTED));
                ui.add(egui::TextEdit::singleline(&mut self.st.dir).desired_width(360.0));
                if ui.button("扫描").clicked() {
                    self.st.scan();
                }
                ui.separator();
                if ui.button("◀ 上一张").clicked() && self.st.current > 0 {
                    self.st.current -= 1;
                    self.st.load_current();
                }
                if ui.button("下一张 ▶").clicked() && self.st.current + 1 < self.st.files.len() {
                    self.st.current += 1;
                    self.st.load_current();
                }
                ui.separator();
                if ui
                    .add(egui::Button::new(egui::RichText::new("检测当前").strong()).fill(Color32::from_rgb(37, 99, 235)))
                    .clicked()
                {
                    self.st.detect_current();
                }
                if ui.button("批量检测").clicked() {
                    self.st.detect_all();
                }
            });
        });
        ui.add_space(8.0);

        ui.horizontal_top(|ui| {
            self.image_view(ui);
            ui.add_space(8.0);
            self.result_panel(ui);
        });
    }

    fn image_view(&mut self, ui: &mut egui::Ui) {
        let Some(img) = &self.st.image else {
            egui::Frame::default().fill(Color32::from_rgb(229, 231, 235)).inner_margin(40.0).show(ui, |ui| {
                ui.label(egui::RichText::new("请先扫描目录并加载图像").size(16.0).color(MUTED));
            });
            return;
        };
        let (iw, ih) = (img.w as f32, img.h as f32);
        let avail_w = (ui.available_width() - 300.0).max(320.0);
        let scale = (avail_w / iw).min(620.0 / ih).max(0.05);
        let size = egui::vec2(iw * scale, ih * scale);
        let Some(tex_id) = self.texture.as_ref().map(|t| t.id()) else { return };

        let response = egui::Frame::default()
            .fill(Color32::from_rgb(17, 24, 39))
            .inner_margin(10.0)
            .show(ui, |ui| {
                let resp = ui.add(egui::Image::new((tex_id, size)).fit_to_exact_size(size));
                if let Some(r) = &self.st.result {
                    let painter = ui.painter_at(resp.rect);
                    draw_overlay(&painter, resp.rect, r, scale);
                }
                resp
            })
            .inner;
        let _ = response;
    }

    fn result_panel(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.set_width(280.0);
            if let Some(r) = &self.st.result {
                let color = verdict_color(r.decision.code);
                egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
                    ui.label(egui::RichText::new("检测结果").size(13.0).color(MUTED));
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(r.decision.code.as_str())
                            .size(24.0)
                            .strong()
                            .color(Color32::WHITE)
                            .background_color(color),
                    );
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&r.decision.message).size(14.0).color(TEXT));
                    ui.add_space(6.0);
                    if let Some(m) = &r.measure {
                        ui.label(egui::RichText::new(format!("宽度  {:.2} mm", m.width_mm.unwrap_or(0.0))).size(20.0).strong());
                        ui.label(egui::RichText::new(format!("像素  {:.1} px", m.width_px)).size(12.0).color(MUTED));
                        ui.label(egui::RichText::new(format!("面积  {:.0} px ｜ 长度 {:.0} px", m.area_px, m.length_px)).size(12.0).color(MUTED));
                    }
                });
                ui.add_space(8.0);
                egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
                    ui.label(egui::RichText::new("耗时").size(13.0).color(MUTED));
                    ui.label(egui::RichText::new(format!("总计 {:.1} ms", r.timings.total_ms)).size(18.0).strong());
                    ui.label(egui::RichText::new(format!("定位 {:.1} ms ｜ 量测 {:.1} ms", r.timings.locate_ms, r.timings.measure_ms)).size(12.0).color(MUTED));
                });
            } else {
                egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
                    ui.label(egui::RichText::new("尚未检测").size(16.0).color(MUTED));
                    ui.label(egui::RichText::new("点击工具栏「检测当前」").size(12.0).color(MUTED));
                });
            }
        });
    }

    fn recipe_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("配方");
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            ui.label(egui::RichText::new("判定阈值").size(14.0).strong());
            ui.add_space(6.0);
            let w = &mut self.st.cfg.measure.width;
            egui::Grid::new("recipe").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                ui.label("标准宽度 (mm)");
                ui.add(egui::DragValue::new(&mut w.standard_mm).speed(0.05).range(0.0..=50.0));
                ui.end_row();
                ui.label("预警线 (mm)");
                ui.add(egui::DragValue::new(&mut w.warn_delta_mm).speed(0.05).range(0.0..=10.0));
                ui.end_row();
                ui.label("报警线 (mm)");
                ui.add(egui::DragValue::new(&mut w.alarm_delta_mm).speed(0.05).range(0.0..=10.0));
                ui.end_row();
            });
        });
        ui.add_space(8.0);
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            ui.label(egui::RichText::new("标定").size(14.0).strong());
            ui.add_space(6.0);
            let mut mm = self.st.mm_per_px.unwrap_or(0.0);
            ui.horizontal(|ui| {
                ui.label("mm / px");
                if ui.add(egui::DragValue::new(&mut mm).speed(0.0001).range(0.0..=10.0)).changed() {
                    self.st.mm_per_px = if mm > 0.0 { Some(mm) } else { None };
                }
                ui.label(egui::RichText::new("合成样本用 0.0833").size(12.0).color(MUTED));
            });
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
    }

    fn history_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("历史记录");
            ui.add_space(8.0);
            if ui.button("刷新").clicked() {
                self.st.refresh_history();
            }
            if ui.button("清空").clicked() {
                self.st.clear_history();
            }
        });
        ui.add_space(6.0);
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("history").striped(true).num_columns(5).spacing([18.0, 6.0]).show(ui, |ui| {
                    for h in ["时间", "文件", "判定", "宽度 mm", "耗时 ms"] {
                        ui.label(egui::RichText::new(h).strong().color(MUTED));
                    }
                    ui.end_row();
                    for r in &self.st.history {
                        ui.label(egui::RichText::new(&r.ts).size(12.0));
                        ui.label(&r.file);
                        ui.colored_label(verdict_color_by_name(&r.verdict), egui::RichText::new(&r.verdict).strong());
                        ui.label(r.width_mm.map(|w| format!("{w:.2}")).unwrap_or_default());
                        ui.label(format!("{:.1}", r.total_ms));
                        ui.end_row();
                    }
                });
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
        ui.add_space(6.0);
        let Some(s) = &self.st.stats else {
            ui.label(egui::RichText::new("暂无数据，先做几次检测").color(MUTED));
            return;
        };
        ui.horizontal(|ui| {
            stat_tile(ui, "总检测", &format!("{}", s.total), TEXT);
            stat_tile(ui, "合格率", &format!("{:.1}%", s.ok_rate * 100.0), OK);
            stat_tile(ui, "耗时 p50", &format!("{:.1} ms", s.lat_p50), TEXT);
            stat_tile(ui, "耗时 p95", &format!("{:.1} ms", s.lat_p95), TEXT);
        });
        ui.add_space(10.0);
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            ui.label(egui::RichText::new("判定分布").size(14.0).strong());
            ui.add_space(4.0);
            let total = s.total.max(1) as f32;
            for (k, n) in &s.by_verdict {
                let ratio = *n as f32 / total;
                ui.horizontal(|ui| {
                    ui.add_sized([90.0, 18.0], egui::Label::new(egui::RichText::new(k).color(verdict_color_by_name(k)).strong()));
                    let bar_w = 320.0 * ratio;
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(320.0, 16.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 3.0, Color32::from_rgb(229, 231, 235));
                    let filled = egui::Rect::from_min_size(rect.min, egui::vec2(bar_w.max(2.0), rect.height()));
                    ui.painter().rect_filled(filled, 3.0, verdict_color_by_name(k));
                    ui.label(format!("{n}  ({:.0}%)", ratio * 100.0));
                });
            }
        });
        ui.add_space(8.0);
        if let (Some(mn), Some(mx), Some(mean)) = (s.width_min, s.width_max, s.width_mean) {
            egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
                ui.label(egui::RichText::new("宽度").size(14.0).strong());
                ui.label(format!("均值 {mean:.2} mm ｜ 最小 {mn:.2} mm ｜ 最大 {mx:.2} mm"));
            });
        }
    }

    fn settings_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("设置");
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
            egui::Grid::new("settings").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
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
        });
        ui.add_space(6.0);
        if ui.button("重新加载配置").clicked() {
            self.st.reload_config();
        }
        ui.add_space(6.0);
        ui.label(egui::RichText::new("文件回放模式：不接相机、不接 PLC；数据库路径修改后重启生效。").size(12.0).color(MUTED));
    }
}

impl eframe::App for HostApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_texture(ctx);
        self.nav(ctx);
        self.bottom(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(12.0))
            .show(ctx, |ui| match self.page {
                Page::Inspect => self.inspect_page(ui),
                Page::Recipe => self.recipe_page(ui),
                Page::History => self.history_page(ui),
                Page::Stats => self.stats_page(ui),
                Page::Settings => self.settings_page(ui),
            });
    }
}

fn stat_tile(ui: &mut egui::Ui, title: &str, value: &str, color: Color32) {
    egui::Frame::group(ui.style()).fill(Color32::WHITE).show(ui, |ui| {
        ui.set_min_width(150.0);
        ui.label(egui::RichText::new(title).size(12.0).color(MUTED));
        ui.label(egui::RichText::new(value).size(26.0).strong().color(color));
    });
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for path in [
        "C:/Windows/Fonts/Deng.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert("cjk".to_owned(), egui::FontData::from_owned(bytes).into());
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(family).or_default().insert(0, "cjk".to_owned());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);
}

fn install_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    let mut v = egui::Visuals::light();
    v.panel_fill = BG;
    v.window_fill = Color32::WHITE;
    v.extreme_bg_color = Color32::from_rgb(249, 250, 251);
    v.selection.bg_fill = Color32::from_rgb(37, 99, 235);
    v.widgets.noninteractive.fg_stroke.color = TEXT;
    v.widgets.inactive.fg_stroke.color = TEXT;
    style.visuals = v;
    ctx.set_style(style);
}

fn verdict_color(code: VerdictCode) -> Color32 {
    match code {
        VerdictCode::Ok => OK,
        VerdictCode::NoRoi => NEUTRAL,
        VerdictCode::Broken => ALARM,
        VerdictCode::TooNarrow | VerdictCode::TooWide => WARN,
        VerdictCode::WarnNarrow | VerdictCode::WarnWide => Color32::from_rgb(202, 138, 4),
        VerdictCode::ReflectionSuspect => Color32::from_rgb(124, 58, 237),
    }
}

fn verdict_color_by_name(name: &str) -> Color32 {
    match name {
        "OK" => OK,
        "NO_ROI" => NEUTRAL,
        "BROKEN" => ALARM,
        "TOO_NARROW" | "TOO_WIDE" => WARN,
        _ => Color32::from_rgb(202, 138, 4),
    }
}

fn draw_overlay(painter: &egui::Painter, rect: egui::Rect, result: &FrameResult, scale: f32) {
    for b in &result.rois {
        let r = egui::Rect::from_min_max(
            egui::pos2(rect.left() + b.x1 as f32 * scale, rect.top() + b.y1 as f32 * scale),
            egui::pos2(rect.left() + b.x2 as f32 * scale, rect.top() + b.y2 as f32 * scale),
        );
        painter.rect_stroke(r, 0.0, egui::Stroke::new(2.0, Color32::from_rgb(52, 211, 153)), egui::StrokeKind::Inside);
    }
    if let Some(m) = &result.measure {
        painter.text(
            egui::pos2(rect.left() + 10.0, rect.top() + 10.0),
            egui::Align2::LEFT_TOP,
            format!("宽 {:.2} mm", m.width_mm.unwrap_or(0.0)),
            egui::FontId::proportional(18.0),
            Color32::from_rgb(253, 224, 71),
        );
    }
    let _ = GrayImage::new(1, 1, 0);
}
