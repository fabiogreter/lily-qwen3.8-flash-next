//! `GET /v1/lily/history?range=` and `GET /v1/lily/sessions/{lineage}/history?range=`:
//! the recorder's minutes (`serve::recorder`) as the contract's grid
//! (docs/dashboard.md § 3.4, § 5).
//!
//! **The grid.** A range has a bucket width (60 s for 15m and 1h, 900 s for
//! 24h, 3 600 s for 7d) that divides the day, so bucket edges fall on the
//! same UTC times whatever the clock says. The data runs to the last
//! complete minute (the minute under way is left out). The last bucket is
//! the one that minute falls in: for 15m and 1h it is that minute, for 24h
//! and 7d it can be partial, stamped with its nominal end (later than now)
//! and with its `busy_share` over the minutes it covers. The grid is the
//! range's `span / step` buckets up to it, oldest first, each stamped with
//! its end. Every bucket is present: one the recorder has nothing for has
//! every value `null` and its counts 0.
//!
//! **The figures** (§ 5). A bucket's average rate is its tokens over its
//! busy time (the recorder's union of running intervals, decode and prefill
//! each over their own), its maximum the best minute in it. The machine
//! series are the minute means (memory pressure the worst level, swap the
//! most); a session's history has them all `null`. The sampler's series are
//! weighted by the milliseconds of each minute its windows covered
//! (`PowerMinute`), so a minute it only half covered counts for half: the
//! GPU's power, activity and frequency by the time it measured, the system
//! power by the part of it that had a system power. Queue times are the
//! requests that finished in the bucket: the median by nearest rank and the
//! longest. The summary is the same arithmetic over the whole grid: tokens
//! over busy time, the best minute, busy time over the wall time the grid
//! covers, and the mean
//! system power over the time the sampler covered, and its energy: the sum
//! over the minutes of mean power times covered time, so a gap (the sampler
//! off, failed or lily not running) adds nothing and is not filled in
//! (`null` while nothing measures power). lily's own estimated power
//! (`lily_power_w`, `lily_energy_wh`) is the same over the time that had a
//! figure of it.

use rusqlite::{Connection, params};

use super::contract::{
    History, HistoryPoint, HistorySummary, MemoryPressure, Range, Rate,
};
use crate::serve::recorder::{MINUTE_MS, Minute};

/// Tokens per second, to the hundredth; `None` without busy time or tokens.
fn per_second(tokens: f64, busy_ms: f64) -> Option<f64> {
    (tokens > 0.0 && busy_ms > 0.0).then(|| round(tokens / (busy_ms / 1e3), 100.0))
}

fn round(value: f64, scale: f64) -> f64 {
    (value * scale).round() / scale
}

/// `part / whole` to four places, `None` over nothing.
fn ratio(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| round(part as f64 / whole as f64, 1e4))
}

/// A range's buckets at `now_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// Start of the first (oldest) bucket, unix ms.
    pub start: u64,
    pub step: u64,
    pub buckets: u64,
    /// The end of the data: the last complete minute's end, unix ms.
    pub data_end: u64,
}

impl Grid {
    pub fn at(range: Range, now_ms: u64) -> Self {
        let step = range.step_ms();
        let buckets = range.span_ms() / step;
        let data_end = now_ms / MINUTE_MS * MINUTE_MS;
        // The bucket the last complete minute falls in.
        let end = data_end.saturating_sub(1) / step * step + step;
        Self { start: end.saturating_sub(buckets * step), step, buckets, data_end }
    }

    /// The nominal end of the last bucket, unix ms (after now when the
    /// bucket is partial).
    pub fn end(&self) -> u64 {
        self.start + self.buckets * self.step
    }

    /// The bucket the time `t` (unix ms) falls in, if it is in the data.
    fn bucket(&self, t: u64) -> Option<usize> {
        (t >= self.start && t < self.data_end)
            .then(|| ((t - self.start) / self.step) as usize)
    }

    /// Milliseconds of bucket `b` the data covers: the step but for a
    /// partial last bucket.
    pub fn covered(&self, b: usize) -> u64 {
        let from = self.start + b as u64 * self.step;
        (from + self.step).min(self.data_end).saturating_sub(from)
    }

    /// The minutes the data covers, `[first, last)`.
    pub fn minutes(&self) -> (u64, u64) {
        (self.start / MINUTE_MS, self.data_end / MINUTE_MS)
    }
}

/// One minute of machine figures as the history shows them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MachineRow {
    pub minute: u64,
    pub cpu_load: Option<f64>,
    /// `kern.memorystatus_vm_pressure_level`, the worst in the minute.
    pub pressure_level: Option<u32>,
    pub swap_bytes: Option<u64>,
    pub gpu_active: Option<f64>,
    pub power_w: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub gpu_freq_mhz: Option<f64>,
    /// Milliseconds of the minute the sampler's windows covered: the weight
    /// of its GPU figures above (`None`: it measured nothing).
    pub power_covered_ms: Option<u64>,
    /// Of those, the milliseconds that had a system power: the weight of
    /// `power_w` and the time the energy is integrated over.
    pub sys_covered_ms: Option<u64>,
    /// lily's own estimated power (CPU and GPU), its mean over
    /// `lily_covered_ms`.
    pub lily_power_w: Option<f64>,
    /// Of `power_covered_ms`, the milliseconds that had a figure of lily's
    /// own power: the weight of `lily_power_w` and the time lily's energy is
    /// integrated over.
    pub lily_covered_ms: Option<u64>,
}

/// The mean of the values present.
fn mean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let (n, sum) = values.flatten().fold((0u32, 0.0), |(n, s), v| (n + 1, s + v));
    (n > 0).then(|| sum / n as f64)
}

fn gpu_weight(m: &MachineRow) -> Option<u64> {
    m.power_covered_ms
}

fn sys_weight(m: &MachineRow) -> Option<u64> {
    m.sys_covered_ms
}

fn lily_weight(m: &MachineRow) -> Option<u64> {
    m.lily_covered_ms
}

/// The mean of a sampler series over the time its windows covered: each
/// minute's figure weighted by its covered milliseconds, minutes without
/// either left out.
fn weighted_mean(
    minutes: &[&MachineRow],
    series: impl Fn(&MachineRow) -> Option<f64>,
    weight: impl Fn(&MachineRow) -> Option<u64>,
) -> Option<f64> {
    let (ms, sum) = minutes.iter().fold((0.0, 0.0), |(ms, sum), m| {
        match (series(m), weight(m).filter(|&c| c > 0)) {
            (Some(v), Some(c)) => (ms + c as f64, sum + v * c as f64),
            _ => (ms, sum),
        }
    });
    (ms > 0.0).then(|| sum / ms)
}

/// Sums of a run of minutes.
#[derive(Debug, Clone, Copy, Default)]
struct Totals {
    any: bool,
    requests: u64,
    decode_tokens: f64,
    prefill_tokens: f64,
    decode_busy_ms: f64,
    prefill_busy_ms: f64,
    busy_ms: f64,
    prompt_tokens: u64,
    cached_tokens: u64,
    drafted_tokens: u64,
    accepted_tokens: u64,
    sessions_max: u64,
    decode_max: Option<f64>,
    prefill_max: Option<f64>,
}

impl Totals {
    fn add(&mut self, m: &Minute) {
        self.any = true;
        self.requests += m.requests;
        self.decode_tokens += m.decode_tokens;
        self.prefill_tokens += m.prefill_tokens;
        self.decode_busy_ms += m.decode_busy_ms;
        self.prefill_busy_ms += m.prefill_busy_ms;
        self.busy_ms += m.busy_ms;
        self.prompt_tokens += m.prompt_tokens;
        self.cached_tokens += m.cached_tokens;
        self.drafted_tokens += m.drafted_tokens;
        self.accepted_tokens += m.accepted_tokens;
        self.sessions_max = self.sessions_max.max(m.sessions_max);
        let best = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.decode_max =
            best(self.decode_max, per_second(m.decode_tokens, m.decode_busy_ms));
        self.prefill_max =
            best(self.prefill_max, per_second(m.prefill_tokens, m.prefill_busy_ms));
    }
}

/// The median by nearest rank and the longest of `queue` (milliseconds),
/// to the tenth.
fn queue_figures(queue: &mut [f64]) -> (Option<f64>, Option<f64>) {
    if queue.is_empty() {
        return (None, None);
    }
    queue.sort_by(f64::total_cmp);
    let median = queue[queue.len().div_ceil(2) - 1];
    (Some(round(median, 10.0)), queue.last().map(|&q| round(q, 10.0)))
}

/// The history of `range` at `now_ms` from the recorder's `minutes` (the
/// grid's, any order), the `machine` minutes (`None` for a session's
/// history) and the finish times and queue milliseconds of the requests in
/// the grid (`queue`).
pub fn build(
    range: Range,
    now_ms: u64,
    minutes: &[(u64, Minute)],
    machine: Option<&[MachineRow]>,
    queue: &[(u64, f64)],
) -> History {
    let grid = Grid::at(range, now_ms);
    let n = grid.buckets as usize;
    let mut totals = vec![Totals::default(); n];
    let mut all = Totals::default();
    for (minute, m) in minutes {
        if let Some(b) = grid.bucket(minute * MINUTE_MS) {
            totals[b].add(m);
            all.add(m);
        }
    }
    let mut queues: Vec<Vec<f64>> = vec![Vec::new(); n];
    for &(t, q) in queue {
        if let Some(b) = grid.bucket(t) {
            queues[b].push(q);
        }
    }
    let mut machines: Vec<Vec<&MachineRow>> = vec![Vec::new(); n];
    for m in machine.unwrap_or_default() {
        if let Some(b) = grid.bucket(m.minute * MINUTE_MS) {
            machines[b].push(m);
        }
    }
    let points: Vec<HistoryPoint> = (0..n)
        .map(|b| {
            let t = &totals[b];
            let ms = &machines[b];
            let (queue_ms_p50, queue_ms_max) = queue_figures(&mut queues[b]);
            HistoryPoint {
                t: grid.start + (b as u64 + 1) * grid.step,
                busy_share: match grid.covered(b) {
                    0 => 0.0,
                    covered => round((t.busy_ms / covered as f64).min(1.0), 1e4),
                },
                requests: t.requests,
                decode_tps_avg: per_second(t.decode_tokens, t.decode_busy_ms),
                decode_tps_max: t.decode_max,
                prefill_tps_avg: per_second(t.prefill_tokens, t.prefill_busy_ms),
                prefill_tps_max: t.prefill_max,
                decode_tokens: t.decode_tokens.round() as u64,
                prefill_tokens: t.prefill_tokens.round() as u64,
                acceptance: ratio(t.accepted_tokens, t.drafted_tokens),
                cache_hit: ratio(t.cached_tokens, t.prompt_tokens),
                queue_ms_p50,
                queue_ms_max,
                sessions_max: t.sessions_max,
                cpu_load: mean(ms.iter().map(|m| m.cpu_load)).map(|v| round(v, 1e3)),
                gpu_active: weighted_mean(ms, |m| m.gpu_active, gpu_weight)
                    .map(|v| round(v, 1e3)),
                memory_pressure: ms
                    .iter()
                    .filter_map(|m| m.pressure_level)
                    .max()
                    .and_then(MemoryPressure::of_level),
                swap_bytes: ms.iter().filter_map(|m| m.swap_bytes).max(),
                power_w: weighted_mean(ms, |m| m.power_w, sys_weight)
                    .map(|v| round(v, 100.0)),
                gpu_power_w: weighted_mean(ms, |m| m.gpu_power_w, gpu_weight)
                    .map(|v| round(v, 100.0)),
                gpu_freq_mhz: weighted_mean(ms, |m| m.gpu_freq_mhz, gpu_weight)
                    .map(|v| round(v, 10.0)),
                lily_power_w: weighted_mean(ms, |m| m.lily_power_w, lily_weight)
                    .map(|v| round(v, 100.0)),
            }
        })
        .collect();

    // The energy is each minute's mean power over the time the sampler
    // covered in it (watt-milliseconds), and the average power is the same
    // sum over the covered time: minutes it did not cover count for nothing.
    let integrate = |series: fn(&MachineRow) -> Option<f64>,
                     weight: fn(&MachineRow) -> Option<u64>| {
        machines
            .iter()
            .flatten()
            .filter_map(|m| series(m).zip(weight(m).filter(|&c| c > 0)))
            .fold((0.0, 0.0), |(ms, sum), (w, c)| (ms + c as f64, sum + w * c as f64))
    };
    let (covered_ms, watt_ms) = integrate(|m| m.power_w, sys_weight);
    // lily's own the same way, over the time that had a figure of it.
    let (lily_ms, lily_watt_ms) = integrate(|m| m.lily_power_w, lily_weight);
    let wall_ms = grid.data_end.saturating_sub(grid.start).max(1) as f64;
    let summary = HistorySummary {
        decode: Rate {
            avg: per_second(all.decode_tokens, all.decode_busy_ms),
            max: all.decode_max,
        },
        prefill: Rate {
            avg: per_second(all.prefill_tokens, all.prefill_busy_ms),
            max: all.prefill_max,
        },
        tokens_generated: points.iter().map(|p| p.decode_tokens).sum(),
        tokens_prefilled: points.iter().map(|p| p.prefill_tokens).sum(),
        requests: all.requests,
        busy_share: round((all.busy_ms / wall_ms).min(1.0), 1e4),
        cache_hit: ratio(all.cached_tokens, all.prompt_tokens),
        acceptance: ratio(all.accepted_tokens, all.drafted_tokens),
        power_avg_w: (covered_ms > 0.0).then(|| round(watt_ms / covered_ms, 100.0)),
        energy_wh: (covered_ms > 0.0).then(|| round(watt_ms / 3_600_000.0, 1e3)),
        lily_energy_wh: (lily_ms > 0.0).then(|| round(lily_watt_ms / 3_600_000.0, 1e3)),
    };
    History { range, step_s: grid.step / 1000, points, summary }
}

// --- reading the recorder's tables ----------------------------------------------

fn minute_of(
    r: &rusqlite::Row<'_>,
    sessions_max: Option<u64>,
) -> rusqlite::Result<(u64, Minute)> {
    Ok((
        r.get(0)?,
        Minute {
            requests: r.get(1)?,
            decode_tokens: r.get(2)?,
            prefill_tokens: r.get(3)?,
            decode_busy_ms: r.get(4)?,
            prefill_busy_ms: r.get(5)?,
            busy_ms: r.get(6)?,
            prompt_tokens: r.get(7)?,
            cached_tokens: r.get(8)?,
            drafted_tokens: r.get(9)?,
            accepted_tokens: r.get(10)?,
            sessions_max: match sessions_max {
                Some(n) => n,
                None => r.get(11)?,
            },
        },
    ))
}

const MINUTE_COLUMNS: &str = "minute, requests, decode_tokens, prefill_tokens, decode_busy_ms,
    prefill_busy_ms, busy_ms, prompt_tokens, cached_tokens, drafted_tokens, accepted_tokens";

/// The rollups of minutes `[from, to)`, every request's or `lineage`'s (one
/// session runs once at a time: `sessions_max` 1).
pub fn minutes(
    conn: &Connection,
    (from, to): (u64, u64),
    lineage: Option<u64>,
) -> rusqlite::Result<Vec<(u64, Minute)>> {
    match lineage {
        None => {
            let mut select = conn.prepare_cached(&format!(
                "SELECT {MINUTE_COLUMNS}, sessions_max FROM minute
                 WHERE minute >= ?1 AND minute < ?2"
            ))?;
            select.query_map(params![from, to], |r| minute_of(r, None))?.collect()
        }
        Some(lineage) => {
            let mut select = conn.prepare_cached(&format!(
                "SELECT {MINUTE_COLUMNS} FROM minute_session
                 WHERE lineage = ?1 AND minute >= ?2 AND minute < ?3"
            ))?;
            select
                .query_map(params![lineage, from, to], |r| minute_of(r, Some(1)))?
                .collect()
        }
    }
}

/// The machine minutes `[from, to)`.
pub fn machine(
    conn: &Connection,
    (from, to): (u64, u64),
) -> rusqlite::Result<Vec<MachineRow>> {
    let mut select = conn.prepare_cached(
        "SELECT minute, cpu_load_mean, memory_pressure, swap_bytes_max, gpu_active_mean,
                power_w_mean, gpu_power_w_mean, gpu_freq_mhz_mean, power_covered_ms,
                sys_covered_ms, lily_w_mean, lily_covered_ms
         FROM machine WHERE minute >= ?1 AND minute < ?2",
    )?;
    select
        .query_map(params![from, to], |r| {
            Ok(MachineRow {
                minute: r.get(0)?,
                cpu_load: r.get(1)?,
                pressure_level: r.get(2)?,
                swap_bytes: r.get(3)?,
                gpu_active: r.get(4)?,
                power_w: r.get(5)?,
                gpu_power_w: r.get(6)?,
                gpu_freq_mhz: r.get(7)?,
                power_covered_ms: r.get(8)?,
                sys_covered_ms: r.get(9)?,
                lily_power_w: r.get(10)?,
                lily_covered_ms: r.get(11)?,
            })
        })?
        .collect()
}

/// Finish time and queue milliseconds of the requests that finished in
/// `[from_ms, to_ms)`, every request's or `lineage`'s; failed requests have
/// no queue time. One statement each, so the lineage's goes through its
/// index. The global one over 7 days reads every stored request's two
/// columns (tens of thousands of rows for a busy week: milliseconds), which
/// the exact median needs; the minute rollups cannot give it.
pub fn queue(
    conn: &Connection,
    (from_ms, to_ms): (u64, u64),
    lineage: Option<u64>,
) -> rusqlite::Result<Vec<(u64, f64)>> {
    let row = |r: &rusqlite::Row<'_>| Ok((r.get(0)?, r.get(1)?));
    match lineage {
        None => conn
            .prepare_cached(
                "SELECT t, queue_ms FROM requests
                 WHERE t >= ?1 AND t < ?2 AND queue_ms IS NOT NULL",
            )?
            .query_map(params![from_ms, to_ms], row)?
            .collect(),
        Some(lineage) => conn
            .prepare_cached(
                "SELECT t, queue_ms FROM requests
                 WHERE lineage = ?1 AND t >= ?2 AND t < ?3 AND queue_ms IS NOT NULL",
            )?
            .query_map(params![lineage, from_ms, to_ms], row)?
            .collect(),
    }
}

/// Whether the database holds any request of `lineage`.
pub fn recorded(conn: &Connection, lineage: u64) -> rusqlite::Result<bool> {
    conn.prepare_cached("SELECT EXISTS (SELECT 1 FROM requests WHERE lineage = ?1)")?
        .query_row([lineage], |r| r.get(0))
}

/// The history of `range` at `now_ms` from the database: every request's,
/// or `lineage`'s. A lineage the database holds nothing of is `None`
/// unless `listed` (a cache entry now), whose history is the empty grid.
pub fn read(
    conn: &Connection,
    range: Range,
    now_ms: u64,
    lineage: Option<u64>,
    listed: bool,
) -> rusqlite::Result<Option<History>> {
    if let Some(id) = lineage
        && !listed
        && !recorded(conn, id)?
    {
        return Ok(None);
    }
    let grid = Grid::at(range, now_ms);
    let span = grid.minutes();
    let rows = minutes(conn, span, lineage)?;
    let machine = match lineage {
        None => Some(machine(conn, span)?),
        Some(_) => None,
    };
    let queue = queue(conn, (grid.start, grid.data_end), lineage)?;
    Ok(Some(build(range, now_ms, &rows, machine.as_deref(), &queue)))
}

/// One lineage's figures over everything the database holds of it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SessionTotals {
    pub requests: u64,
    pub decode_tps_avg: Option<f64>,
    pub decode_tps_max: Option<f64>,
    pub prefill_tps_avg: Option<f64>,
    pub cache_hit: Option<f64>,
    pub acceptance: Option<f64>,
}

/// `lineage`'s figures over the 7 days kept (§ 5: averages over busy time,
/// the maximum the best minute).
pub fn session_totals(
    conn: &Connection,
    lineage: u64,
) -> rusqlite::Result<SessionTotals> {
    let mut select = conn.prepare_cached(
        "SELECT coalesce(sum(requests), 0), coalesce(sum(decode_tokens), 0),
                coalesce(sum(decode_busy_ms), 0),
                max(CASE WHEN decode_busy_ms > 0 AND decode_tokens > 0
                         THEN decode_tokens * 1000.0 / decode_busy_ms END),
                coalesce(sum(prefill_tokens), 0), coalesce(sum(prefill_busy_ms), 0),
                coalesce(sum(cached_tokens), 0), coalesce(sum(prompt_tokens), 0),
                coalesce(sum(accepted_tokens), 0), coalesce(sum(drafted_tokens), 0)
         FROM minute_session WHERE lineage = ?1",
    )?;
    select.query_row([lineage], |r| {
        let max: Option<f64> = r.get(3)?;
        Ok(SessionTotals {
            requests: r.get(0)?,
            decode_tps_avg: per_second(r.get(1)?, r.get(2)?),
            decode_tps_max: max.map(|m| round(m, 100.0)),
            prefill_tps_avg: per_second(r.get(4)?, r.get(5)?),
            cache_hit: ratio(r.get(6)?, r.get(7)?),
            acceptance: ratio(r.get(8)?, r.get(9)?),
        })
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/serve/dashboard/history.rs"]
mod tests;
