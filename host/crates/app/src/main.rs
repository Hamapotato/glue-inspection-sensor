mod state;
mod ui;

use crate::state::AppState;

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--headless") {
        let dir = args.get(pos + 1).cloned().unwrap_or_else(|| "../outputs/rust_samples".into());
        let mut st = AppState::new();
        st.dir = dir;
        if let Some(pos) = args.iter().position(|a| a == "--mm-per-px") {
            if let Some(v) = args.get(pos + 1).and_then(|s| s.parse::<f64>().ok()) {
                st.mm_per_px = Some(v);
            }
        }
        st.scan();
        let n = st.detect_all();
        println!("无头模式：检测 {n} 张");
        for r in st.history.iter().take(10) {
            println!("{:<24} {:<12} width={:?} {:.1}ms", r.file, r.verdict, r.width_mm, r.total_ms);
        }
        if let Some(s) = &st.stats {
            println!("合格率 {:.1}% ｜ p50 {:.1}ms ｜ p95 {:.1}ms", s.ok_rate * 100.0, s.lat_p50, s.lat_p95);
        }
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "涂胶检测上位机",
        options,
        Box::new(|_cc| Ok(Box::new(ui::HostApp::new()))),
    )
}
