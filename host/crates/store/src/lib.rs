//! 检测记录存储（SQLite）：历史查询、统计。
//!
//! 表结构见 docs/host-software-architecture.md §6.4；首版只存单帧结果，位置级结果与
//! 图片存档后续扩展。

use anyhow::Result;
use chrono::Local;
use rusqlite::{params, Connection};

#[derive(Debug, Clone)]
pub struct Record {
    pub id: i64,
    pub ts: String,
    pub file: String,
    pub verdict: String,
    pub width_mm: Option<f64>,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total: usize,
    pub by_verdict: Vec<(String, usize)>,
    pub ok_rate: f64,
    pub width_mean: Option<f64>,
    pub width_min: Option<f64>,
    pub width_max: Option<f64>,
    pub lat_p50: f64,
    pub lat_p95: f64,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS inspection (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts TEXT NOT NULL,
                 file TEXT NOT NULL,
                 verdict TEXT NOT NULL,
                 width_mm REAL,
                 total_ms REAL NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_inspection_ts ON inspection(ts);",
        )?;
        Ok(Self { conn })
    }

    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS inspection (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts TEXT NOT NULL,
                 file TEXT NOT NULL,
                 verdict TEXT NOT NULL,
                 width_mm REAL,
                 total_ms REAL NOT NULL
             );",
        )?;
        Ok(Self { conn })
    }

    pub fn insert(
        &self,
        file: &str,
        verdict: &str,
        width_mm: Option<f64>,
        total_ms: f64,
    ) -> Result<i64> {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.conn.execute(
            "INSERT INTO inspection (ts, file, verdict, width_mm, total_ms) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ts, file, verdict, width_mm, total_ms],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<Record>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, ts, file, verdict, width_mm, total_ms FROM inspection ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok(Record {
                id: r.get(0)?,
                ts: r.get(1)?,
                file: r.get(2)?,
                verdict: r.get(3)?,
                width_mm: r.get(4)?,
                total_ms: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM inspection", [])?;
        Ok(())
    }

    pub fn stats(&self) -> Result<Stats> {
        let rows = self.recent(100_000)?;
        let total = rows.len();
        let mut by_verdict: Vec<(String, usize)> = Vec::new();
        let mut widths: Vec<f64> = Vec::new();
        let mut lats: Vec<f64> = Vec::new();
        for r in &rows {
            match by_verdict.iter_mut().find(|(k, _)| *k == r.verdict) {
                Some((_, n)) => *n += 1,
                None => by_verdict.push((r.verdict.clone(), 1)),
            }
            if let Some(w) = r.width_mm {
                if w > 0.0 {
                    widths.push(w);
                }
            }
            lats.push(r.total_ms);
        }
        by_verdict.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let ok = by_verdict
            .iter()
            .find(|(k, _)| k == "OK")
            .map(|(_, n)| *n)
            .unwrap_or(0);
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| -> f64 {
            if lats.is_empty() {
                0.0
            } else {
                lats[((lats.len() as f64 - 1.0) * p) as usize]
            }
        };
        let (wmin, wmax, wmean) = if widths.is_empty() {
            (None, None, None)
        } else {
            let mn = widths.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = widths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (
                Some(mn),
                Some(mx),
                Some(widths.iter().sum::<f64>() / widths.len() as f64),
            )
        };
        Ok(Stats {
            total,
            ok_rate: if total > 0 {
                ok as f64 / total as f64
            } else {
                0.0
            },
            by_verdict,
            width_mean: wmean,
            width_min: wmin,
            width_max: wmax,
            lat_p50: pct(0.5),
            lat_p95: pct(0.95),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query() {
        let s = Store::open_memory().unwrap();
        s.insert("a.png", "OK", Some(10.0), 5.0).unwrap();
        s.insert("b.png", "TOO_NARROW", Some(5.0), 7.0).unwrap();
        let rows = s.recent(10).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].file, "b.png");
    }

    #[test]
    fn stats_aggregate() {
        let s = Store::open_memory().unwrap();
        for (v, w) in [("OK", 10.0), ("OK", 10.2), ("TOO_WIDE", 13.0)] {
            s.insert("x", v, Some(w), 8.0).unwrap();
        }
        let st = s.stats().unwrap();
        assert_eq!(st.total, 3);
        assert!((st.ok_rate - 2.0 / 3.0).abs() < 1e-6);
        assert!((st.width_mean.unwrap() - 11.0666).abs() < 1e-3);
        assert!(st.lat_p50 > 0.0);
    }

    #[test]
    fn clear_works() {
        let s = Store::open_memory().unwrap();
        s.insert("x", "OK", None, 1.0).unwrap();
        s.clear().unwrap();
        assert_eq!(s.recent(10).unwrap().len(), 0);
    }
}
