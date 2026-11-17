//! egui 界面：检测 / 配方 / 历史 / 统计 / 设置。
//! 交互规范见 host/AGENTS.md 第 6 节。

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
const CHIP_BG: Color32 = Color32::from_rgb(241, 245, 249);

pub struct HostApp {
    st: AppState,
    page: Page,
    texture: Option<egui::TextureHandle>,
    tex_key: String,
    selected: Option<i64>,
}

impl HostApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        install_style(&cc.egui_ctx);
        let mut st = AppState::new();
        st.scan();
        Self {
            st,
            page: Page::Inspect,
            texture: None,
            tex_key: String::new(),
            selected: None,
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let (prev, next, detect, batch, refresh) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Space),
                i.modifiers.command && i.key_pressed(egui::Key::B),
                i.key_pressed(egui::Key::F5),
            )
        });
        if prev && self.st.current > 0 {
            self.st.current -= 1;
            self.st.load_current();
        }
        if next && self.st.current + 1 < self.st.files.len() {
            self.st.current += 1;
            self.st.load_current();
        }
        if detect {
            self.st.detect_current();
        }
        if batch {
            self.st.detect_all();
        }
        if refresh {
            self.st.refresh_history();
            self.st.refresh_stats();
        }
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
                ui.label(
                    egui::RichText::new("涂胶检测")
                        .size(22.0)
                        .strong()
                        .color(Color32::WHITE),
                );
                ui.label(
                    egui::RichText::new("Glue Inspection HMI")
                        .size(11.0)
                        .color(Color32::from_rgb(148, 163, 184)),
                );
                ui.add_space(16.0);
                for (p, label) in [
                    (Page::Inspect, "检测"),
                    (Page::Recipe, "配方"),
                    (Page::History, "历史"),
                    (Page::Stats, "统计"),
                    (Page::Settings, "设置"),
                ] {
                    let text = egui::RichText::new(label).size(16.0);
                    let selected = self.page == p;
                    let btn = egui::Button::new(text)
                        .fill(if selected {
                            NAV_ACTIVE
                        } else {
                            Color32::TRANSPARENT
                        })
                        .min_size(egui::vec2(140.0, 34.0));
                    if ui.add(btn).clicked() {
                        self.page = p;
                    }
                }
                ui.add_space(10.0);
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("图像 {} 张", self.st.files.len())).size(12.0),
                );
                ui.label(
                    egui::RichText::new("← → 切换 · 空格 检测")
                        .size(11.0)
                        .color(Color32::from_rgb(148, 163, 184)),
                );
            });
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        let title = match self.page {
            Page::Inspect => "检测",
            Page::Recipe => "配方",
            Page::History => "历史记录",
            Page::Stats => "统计",
            Page::Settings => "设置",
        };
        egui::TopBottomPanel::top("topbar")
            .frame(
                egui::Frame::default()
                    .fill(Color32::WHITE)
                    .inner_margin(10.0),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(title).size(18.0).strong().color(TEXT));
                    ui.add_space(12.0);
                    let calib = self
                        .st
                        .mm_per_px
                        .map(|v| format!("标定 {v:.4} mm/px"))
                        .unwrap_or_else(|| "未标定".into());
                    chip(
                        ui,
                        &calib,
                        if self.st.mm_per_px.is_some() {
                            OK
                        } else {
                            WARN
                        },
                    );
                    chip(
                        ui,
                        &format!("标准 {:.1} mm", self.st.cfg.measure.width.standard_mm),
                        TEXT,
                    );
                    chip(ui, &format!("记录 {} 条", self.st.history.len()), TEXT);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("快捷键").clicked() {
                            self.st.status =
                                "← → 上/下一张 ｜ 空格 检测当前 ｜ Ctrl+B 批量检测 ｜ F5 刷新统计"
                                    .into();
                        }
                    });
                });
            });
    }

    fn bottom(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status")
            .frame(
                egui::Frame::default()
                    .fill(Color32::WHITE)
                    .inner_margin(8.0),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let (dot, color) = match self.st.result.as_ref().map(|r| r.decision.code) {
                        Some(VerdictCode::Ok) => ("●", OK),
                        Some(c) if c.is_defect() => ("●", ALARM),
                        Some(VerdictCode::NoQuality) => ("●", Color32::from_rgb(202, 138, 4)),
                        Some(VerdictCode::NoRoi) => ("●", NEUTRAL),
                        _ => ("○", MUTED),
                    };
                    ui.colored_label(color, dot);
                    ui.label(egui::RichText::new(&self.st.status).size(13.0).color(TEXT));
                });
            });
    }

    fn inspect_page(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("图像目录").size(13.0).color(MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.st.dir).desired_width(320.0));
                    if ui.button("扫描").clicked() {
                        self.st.scan();
                    }
                    ui.separator();
                    let has_prev = self.st.current > 0;
                    let has_next = self.st.current + 1 < self.st.files.len();
                    if ui
                        .add_enabled(has_prev, egui::Button::new("◀ 上一张"))
                        .clicked()
                    {
                        self.st.current -= 1;
                        self.st.load_current();
                    }
                    if ui
                        .add_enabled(has_next, egui::Button::new("下一张 ▶"))
                        .clicked()
                    {
                        self.st.current += 1;
                        self.st.load_current();
                    }
                    if ui
                        .add_enabled(has_next, egui::Button::new("批量检测"))
                        .clicked()
                    {
                        self.st.detect_all();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.st.image.is_some(),
                            egui::Button::new(egui::RichText::new("检测当前（空格）").strong())
                                .fill(Color32::from_rgb(37, 99, 235)),
                        )
                        .clicked()
                    {
                        self.st.detect_current();
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
            egui::Frame::default()
                .fill(Color32::from_rgb(229, 231, 235))
                .inner_margin(40.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("请先在上方输入目录并点「扫描」")
                            .size(16.0)
                            .color(MUTED),
                    );
                    ui.label(
                        egui::RichText::new("默认目录：../outputs/rust_samples")
                            .size(12.0)
                            .color(MUTED),
                    );
                });
            return;
        };
        let (iw, ih) = (img.w as f32, img.h as f32);
        let avail_w = (ui.available_width() - 300.0).max(320.0);
        let scale = (avail_w / iw).min(620.0 / ih).max(0.05);
        let size = egui::vec2(iw * scale, ih * scale);
        let Some(tex_id) = self.texture.as_ref().map(|t| t.id()) else {
            return;
        };
        egui::Frame::default()
            .fill(Color32::from_rgb(17, 24, 39))
            .inner_margin(10.0)
            .show(ui, |ui| {
                let resp = ui.add(egui::Image::new((tex_id, size)).fit_to_exact_size(size));
                if let Some(r) = &self.st.result {
                    let painter = ui.painter_at(resp.rect);
                    draw_overlay(&painter, resp.rect, r, scale);
                }
            });
        let _ = GrayImage::new(1, 1, 0);
    }

    fn result_panel(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.set_width(280.0);
            if let Some(r) = &self.st.result {
                let color = verdict_color(r.decision.code);
                egui::Frame::group(ui.style())
                    .fill(Color32::WHITE)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("检测结果").size(13.0).color(MUTED));
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(r.decision.code.as_str())
                                .size(22.0)
                                .strong()
                                .color(Color32::WHITE)
                                .background_color(color),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(&r.decision.message)
                                .size(14.0)
                                .color(TEXT),
                        );
                        if let Some(m) = &r.measure {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "宽度  {:.2} mm",
                                    m.width_mm.unwrap_or(0.0)
                                ))
                                .size(20.0)
                                .strong(),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "面积 {:.0} px ｜ 长度 {:.0} px",
                                    m.area_px, m.length_px
                                ))
                                .size(12.0)
                                .color(MUTED),
                            );
                        }
                    });
                ui.add_space(8.0);
                egui::Frame::group(ui.style())
                    .fill(Color32::WHITE)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("耗时").size(13.0).color(MUTED));
                        ui.label(
                            egui::RichText::new(format!("总计 {:.1} ms", r.timings.total_ms))
                                .size(18.0)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "定位 {:.1} ms ｜ 量测 {:.1} ms",
                                r.timings.locate_ms, r.timings.measure_ms
                            ))
                            .size(12.0)
                            .color(MUTED),
                        );
                    });
            } else {
                egui::Frame::group(ui.style())
                    .fill(Color32::WHITE)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("尚未检测").size(16.0).color(MUTED));
                        ui.label(
                            egui::RichText::new("点「检测当前」或按空格")
                                .size(12.0)
                                .color(MUTED),
                        );
                    });
            }
            if !self.st.history.is_empty() {
                ui.add_space(8.0);
                egui::Frame::group(ui.style())
                    .fill(Color32::WHITE)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("最近记录").size(13.0).color(MUTED));
                        for r in self.st.history.iter().take(5) {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(&r.file).size(12.0));
                                ui.colored_label(
                                    verdict_color_by_name(&r.verdict),
                                    egui::RichText::new(&r.verdict).size(12.0),
                                );
                                ui.label(
                                    egui::RichText::new(
                                        r.width_mm
                                            .map(|w| format!("{w:.2} mm"))
                                            .unwrap_or_default(),
                                    )
                                    .size(12.0)
                                    .color(MUTED),
                                );
                            });
                        }
                    });
            }
        });
    }

    fn recipe_page(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .show(ui, |ui| {
                ui.label(egui::RichText::new("判定阈值").size(14.0).strong());
                ui.add_space(6.0);
                let w = &mut self.st.cfg.measure.width;
                egui::Grid::new("recipe")
                    .num_columns(2)
                    .spacing([16.0, 10.0])
                    .show(ui, |ui| {
                        ui.label("标准宽度 (mm)");
                        ui.add(
                            egui::DragValue::new(&mut w.standard_mm)
                                .speed(0.05)
                                .range(0.0..=50.0),
                        );
                        ui.end_row();
                        ui.label("预警线 (mm)");
                        ui.add(
                            egui::DragValue::new(&mut w.warn_delta_mm)
                                .speed(0.05)
                                .range(0.0..=10.0),
                        );
                        ui.end_row();
                        ui.label("报警线 (mm)");
                        ui.add(
                            egui::DragValue::new(&mut w.alarm_delta_mm)
                                .speed(0.05)
                                .range(0.0..=10.0),
                        );
                        ui.end_row();
                    });
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("改动立即生效；保存后写入配方 JSON")
                        .size(12.0)
                        .color(MUTED),
                );
            });
        ui.add_space(8.0);
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .show(ui, |ui| {
                ui.label(egui::RichText::new("标定").size(14.0).strong());
                ui.add_space(6.0);
                let mut mm = self.st.mm_per_px.unwrap_or(0.0);
                ui.horizontal(|ui| {
                    ui.label("mm / px");
                    if ui
                        .add(
                            egui::DragValue::new(&mut mm)
                                .speed(0.0001)
                                .range(0.0..=10.0),
                        )
                        .changed()
                    {
                        self.st.mm_per_px = if mm > 0.0 { Some(mm) } else { None };
                    }
                    ui.label(
                        egui::RichText::new("合成样本用 0.0833")
                            .size(12.0)
                            .color(MUTED),
                    );
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
            if ui.button("刷新（F5）").clicked() {
                self.st.refresh_history();
            }
            if ui.button("清空").clicked() {
                self.st.clear_history();
                self.selected = None;
            }
            ui.label(
                egui::RichText::new("点击行查看详情")
                    .size(12.0)
                    .color(MUTED),
            );
        });
        ui.add_space(6.0);
        let rows = self.st.history.clone();
        let mut clicked: Option<i64> = None;
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(480.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("history")
                            .striped(true)
                            .num_columns(5)
                            .spacing([18.0, 6.0])
                            .show(ui, |ui| {
                                for h in ["时间", "文件", "判定", "宽度 mm", "耗时 ms"] {
                                    ui.label(egui::RichText::new(h).strong().color(MUTED));
                                }
                                ui.end_row();
                                for r in &rows {
                                    ui.label(egui::RichText::new(&r.ts).size(12.0));
                                    if ui
                                        .selectable_label(self.selected == Some(r.id), &r.file)
                                        .clicked()
                                    {
                                        clicked = Some(r.id);
                                    }
                                    ui.colored_label(
                                        verdict_color_by_name(&r.verdict),
                                        egui::RichText::new(&r.verdict).strong(),
                                    );
                                    ui.label(
                                        r.width_mm.map(|w| format!("{w:.2}")).unwrap_or_default(),
                                    );
                                    ui.label(format!("{:.1}", r.total_ms));
                                    ui.end_row();
                                }
                            });
                    });
            });
        if clicked.is_some() {
            self.selected = clicked;
        }
        if let Some(id) = self.selected {
            if let Some(r) = self.st.history.iter().find(|r| r.id == id) {
                ui.add_space(8.0);
                egui::Frame::group(ui.style())
                    .fill(Color32::WHITE)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("记录详情").size(14.0).strong());
                        ui.label(format!("文件：{}", r.file));
                        ui.label(format!("时间：{} ｜ 判定：{}", r.ts, r.verdict));
                        ui.label(format!(
                            "宽度：{} mm ｜ 耗时：{:.1} ms",
                            r.width_mm
                                .map(|w| format!("{w:.2}"))
                                .unwrap_or_else(|| "-".into()),
                            r.total_ms
                        ));
                    });
            }
        }
    }

    fn stats_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("刷新（F5）").clicked() {
                self.st.refresh_stats();
            }
        });
        ui.add_space(4.0);
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
        ui.add_space(8.0);
        ui.horizontal_top(|ui| {
            egui::Frame::group(ui.style())
                .fill(Color32::WHITE)
                .show(ui, |ui| {
                    ui.set_min_width(360.0);
                    ui.label(egui::RichText::new("判定分布").size(14.0).strong());
                    ui.add_space(4.0);
                    let total = s.total.max(1) as f32;
                    for (k, n) in &s.by_verdict {
                        let ratio = *n as f32 / total;
                        ui.horizontal(|ui| {
                            ui.add_sized(
                                [90.0, 18.0],
                                egui::Label::new(
                                    egui::RichText::new(k)
                                        .color(verdict_color_by_name(k))
                                        .strong(),
                                ),
                            );
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(220.0, 14.0), egui::Sense::hover());
                            ui.painter()
                                .rect_filled(rect, 3.0, Color32::from_rgb(229, 231, 235));
                            let filled = egui::Rect::from_min_size(
                                rect.min,
                                egui::vec2((220.0 * ratio).max(2.0), rect.height()),
                            );
                            ui.painter()
                                .rect_filled(filled, 3.0, verdict_color_by_name(k));
                            ui.label(format!("{n}  ({:.0}%)", ratio * 100.0));
                        });
                    }
                });
            ui.add_space(8.0);
            egui::Frame::group(ui.style())
                .fill(Color32::WHITE)
                .show(ui, |ui| {
                    ui.set_min_width(360.0);
                    ui.label(egui::RichText::new("宽度分布").size(14.0).strong());
                    let widths: Vec<f64> = self
                        .st
                        .history
                        .iter()
                        .filter_map(|r| r.width_mm)
                        .filter(|w| *w > 0.0)
                        .collect();
                    draw_histogram(ui, &widths, 10);
                    if let (Some(mn), Some(mx), Some(mean)) =
                        (s.width_min, s.width_max, s.width_mean)
                    {
                        ui.label(
                            egui::RichText::new(format!(
                                "均值 {mean:.2} ｜ 最小 {mn:.2} ｜ 最大 {mx:.2} mm"
                            ))
                            .size(12.0)
                            .color(MUTED),
                        );
                    }
                });
        });
    }

    fn settings_page(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style())
            .fill(Color32::WHITE)
            .show(ui, |ui| {
                egui::Grid::new("settings")
                    .num_columns(2)
                    .spacing([16.0, 10.0])
                    .show(ui, |ui| {
                        ui.label("配置文件");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.st.cfg_path).desired_width(420.0),
                        );
                        ui.end_row();
                        ui.label("数据库");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.st.db_path).desired_width(420.0),
                        );
                        ui.end_row();
                        ui.label("配方文件");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.st.recipe_path)
                                .desired_width(420.0),
                        );
                        ui.end_row();
                    });
            });
        ui.add_space(6.0);
        if ui.button("重新加载配置").clicked() {
            self.st.reload_config();
        }
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new("文件回放模式：不接相机、不接 PLC；数据库路径修改后重启生效。")
                .size(12.0)
                .color(MUTED),
        );
    }
}

impl eframe::App for HostApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_shortcuts(ctx);
        self.ensure_texture(ctx);
        self.nav(ctx);
        self.top_bar(ctx);
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

fn chip(ui: &mut egui::Ui, text: &str, color: Color32) {
    egui::Frame::default()
        .fill(CHIP_BG)
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(12.0).color(color).strong());
        });
}

fn stat_tile(ui: &mut egui::Ui, title: &str, value: &str, color: Color32) {
    egui::Frame::group(ui.style())
        .fill(Color32::WHITE)
        .show(ui, |ui| {
            ui.set_min_width(150.0);
            ui.label(egui::RichText::new(title).size(12.0).color(MUTED));
            ui.label(egui::RichText::new(value).size(26.0).strong().color(color));
        });
}

fn draw_histogram(ui: &mut egui::Ui, widths: &[f64], bins: usize) {
    if widths.len() < 2 {
        ui.label(
            egui::RichText::new("样本不足（至少 2 条记录）")
                .size(12.0)
                .color(MUTED),
        );
        return;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for w in widths {
        lo = lo.min(*w);
        hi = hi.max(*w);
    }
    if (hi - lo).abs() < 1e-9 {
        ui.label(format!("全部为 {lo:.2} mm"));
        return;
    }
    let mut counts = vec![0usize; bins];
    for w in widths {
        let idx = (((w - lo) / (hi - lo)) * bins as f64) as usize;
        counts[idx.min(bins - 1)] += 1;
    }
    let max_c = *counts.iter().max().unwrap() as f32;
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (i, c) in counts.iter().enumerate() {
            let h = if max_c > 0.0 {
                70.0 * (*c as f32) / max_c
            } else {
                0.0
            };
            let (rect, _) = ui.allocate_exact_size(egui::vec2(26.0, 80.0), egui::Sense::hover());
            let bar = egui::Rect::from_min_max(
                egui::pos2(rect.left(), rect.bottom() - h.max(1.0)),
                rect.right_bottom(),
            );
            ui.painter()
                .rect_filled(bar, 2.0, Color32::from_rgb(37, 99, 235));
            let label = format!("{:.1}", lo + (i as f64 + 0.5) * (hi - lo) / bins as f64);
            if i % 2 == 0 {
                ui.painter().text(
                    egui::pos2(rect.center().x, rect.bottom() + 4.0),
                    egui::Align2::CENTER_TOP,
                    label,
                    egui::FontId::proportional(9.0),
                    MUTED,
                );
            }
        }
    });
    ui.add_space(14.0);
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(custom) = std::env::var("GLUE_FONT") {
        candidates.push(custom);
    }
    candidates.extend(
        [
            // Windows
            "C:/Windows/Fonts/Deng.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            // Linux（RK3588 / Debian / Ubuntu）
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            fonts
                .font_data
                .insert("cjk".to_owned(), egui::FontData::from_owned(bytes).into());
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts
                    .families
                    .entry(family)
                    .or_default()
                    .insert(0, "cjk".to_owned());
            }
            eprintln!("[ui] 字体：{path}");
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
        VerdictCode::NoQuality => Color32::from_rgb(202, 138, 4),
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
            egui::pos2(
                rect.left() + b.x1 as f32 * scale,
                rect.top() + b.y1 as f32 * scale,
            ),
            egui::pos2(
                rect.left() + b.x2 as f32 * scale,
                rect.top() + b.y2 as f32 * scale,
            ),
        );
        painter.rect_stroke(
            r,
            0.0,
            egui::Stroke::new(2.0, Color32::from_rgb(52, 211, 153)),
            egui::StrokeKind::Inside,
        );
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
}
