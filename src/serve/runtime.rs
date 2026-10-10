//! Runtime settings (docs/dashboard.md § 7): the settings the dashboard can
//! change without a restart, one shared [`RuntimeSettings`] that every
//! consumer reads when it needs a value instead of keeping the copy it took
//! at startup.
//!
//! **Who reads what, when.** Request preparation (on the HTTP thread) takes
//! one [`Snapshot`] per request ([`RuntimeSettings::snapshot`], an `Arc`
//! clone under a read lock held for nothing else) and prepares the whole
//! request from it, so a change never reaches a request already prepared:
//! the sampling defaults, the default reasoning effort and the thinking
//! controls' defaults. The engine thread never takes that lock: it reads
//! atomics. The idle timer reads `idle_unload` whenever it computes its
//! deadline, and a change nudges the waiting engine so the deadline is
//! computed again (the timer is re-armed: the new delay counts from the last
//! request, as the old one did; the dashboard never moves the timer itself).
//! The weights' pin reads `pin_hold` at each use and is re-armed the same
//! way (a hold shorter than the time since the last request releases at
//! once), the disk tier `disk_cache_ttl` at each expiry pass (none before
//! the values are [settled](RuntimeSettings::settle), at most
//! [`SETTLE_BOUND`] after the start), and admission `durable_min_tokens` once
//! per request.
//!
//! **Validation** is the command line's: the same parsers and checks
//! (`parse_duration_secs`, `ThinkingBudgets::parse`, `SamplingParams::validate`,
//! the tool-turn factor's and the grace's checks, the requests' reasoning
//! effort mapping), so a value the flag would refuse is refused with the
//! same message. A JSON value stands for the flag's text: a string as it is
//! (not trimmed), a number or a boolean as JSON writes it.
//!
//! **Persistence** is the recorder's (`serve::recorder`): it owns the
//! database, so it is the one writer of this state after startup. A change
//! is planned here ([`RuntimeSettings::plan`]), written there in one
//! transaction (the `settings` rows and the `settings_log` entries) and only
//! then published here ([`RuntimeSettings::publish`]), so what the server
//! does and what the database says never disagree. At startup the stored
//! overrides are applied over the command line once the database opens
//! ([`RuntimeSettings::apply_stored`]): database over flag over default.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::api::{
    Defaults, ThinkingBudgets, check_grace, check_tool_turn_factor, template_effort,
};
use super::dashboard::contract::{Setting, SettingValue};
use super::dashboard::settings;
use super::parse_duration_secs;

const NEXT_REQUEST: &str = "next request";

/// How long the disk tier waits at most for the stored overrides before its
/// first expiry pass (see [`RuntimeSettings::expiry_gate`]).
pub const SETTLE_BOUND: Duration = Duration::from_secs(30);

/// The settings that change without a restart and when a change takes
/// effect, in the settings view's order. `--queue` and
/// `--decode-checkpoint-tokens` are not among them (see docs/server.md,
/// "Runtime settings").
pub const RUNTIME: &[(&str, &str)] = &[
    ("temperature", NEXT_REQUEST),
    ("top_p", NEXT_REQUEST),
    ("top_k", NEXT_REQUEST),
    ("min_p", NEXT_REQUEST),
    ("presence_penalty", NEXT_REQUEST),
    ("frequency_penalty", NEXT_REQUEST),
    ("repetition_penalty", NEXT_REQUEST),
    ("reasoning_effort", NEXT_REQUEST),
    ("thinking_budget", NEXT_REQUEST),
    ("thinking_budget_tool_turn_factor", NEXT_REQUEST),
    ("thinking_budget_grace", NEXT_REQUEST),
    ("thinking_nudges", NEXT_REQUEST),
    ("tool_call_ends_thinking", NEXT_REQUEST),
    ("idle_unload", "timer re-armed on change"),
    ("pin_hold", "hold re-armed on change"),
    ("disk_cache_ttl", "next expiry pass"),
    ("durable_min_tokens", NEXT_REQUEST),
];

/// When a change of `name` takes effect; `None` for a setting that needs a
/// restart (or one lily does not know).
pub fn applies(name: &str) -> Option<&'static str> {
    RUNTIME.iter().find(|(n, _)| *n == name).map(|(_, applies)| *applies)
}

/// The runtime values, as one request (or one look) sees them.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// What request preparation fills in where a request says nothing.
    pub defaults: Defaults,
    /// 0: never unloads.
    pub idle_unload_secs: u64,
    /// 0: held until memory pressure or the unload.
    pub pin_hold_secs: u64,
    /// 0: entries stay until the budget evicts them.
    pub disk_cache_ttl_secs: u64,
    /// 0: no durable prefix entries.
    pub durable_min_tokens: usize,
}

/// The flag of setting `name`, as its errors name it.
fn flag(name: &str) -> String {
    format!("--{}", name.replace('_', "-"))
}

/// The command-line text a JSON value stands for: a string as it is (not
/// trimmed, as clap does not trim), a number or a boolean as JSON writes it.
fn flag_text(name: &str, value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        _ => Err(format!(
            "{} takes a number, a string or a boolean, as its flag's text",
            flag(name)
        )),
    }
}

/// `text` as the flag's type, refused as clap words it.
fn parse<T: std::str::FromStr>(name: &str, text: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    text.parse::<T>()
        .map_err(|e| format!("invalid value '{text}' for '{}': {e}", flag(name)))
}

/// A switch's text: clap's boolean parser takes `true` or `false` only.
fn parse_bool(name: &str, text: &str) -> Result<bool, String> {
    match text {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!(
            "invalid value '{text}' for '{}': use true or false",
            flag(name)
        )),
    }
}

impl Snapshot {
    /// Sets `name` to the flag text `value` stands for, checked as the
    /// command line checks it. Returns the value as the settings view shows
    /// it (the canonical text read back as the setting's JSON kind), or the
    /// refusal. `None` for a setting that is not a runtime one.
    pub fn set(
        &mut self,
        name: &str,
        value: &Value,
    ) -> Option<Result<SettingValue, String>> {
        applies(name)?;
        Some(self.set_runtime(name, value))
    }

    fn set_runtime(
        &mut self,
        name: &str,
        value: &Value,
    ) -> Result<SettingValue, String> {
        let text = flag_text(name, value)?;
        let sampling = &mut self.defaults.sampling;
        let thinking = &mut self.defaults.thinking_controls;
        let canonical = match name {
            "temperature" | "top_p" | "min_p" | "presence_penalty"
            | "frequency_penalty" | "repetition_penalty" => {
                let v: f32 = parse(name, &text)?;
                let mut next = *sampling;
                *match name {
                    "temperature" => &mut next.temperature,
                    "top_p" => &mut next.top_p,
                    "min_p" => &mut next.min_p,
                    "presence_penalty" => &mut next.presence_penalty,
                    "frequency_penalty" => &mut next.frequency_penalty,
                    _ => &mut next.repetition_penalty,
                } = v;
                // The checks the start runs on the merged defaults.
                next.validate().map_err(|e| format!("{e:#}"))?;
                *sampling = next;
                v.to_string()
            }
            "top_k" => {
                let v: usize = parse(name, &text)?;
                sampling.top_k = v;
                v.to_string()
            }
            "reasoning_effort" => {
                // The requests' mapping, stored as the template's name.
                let Some(effort) = template_effort(&text) else {
                    return Err(format!(
                        "invalid value '{text}' for '{}': use low, medium or xhigh (high and max are xhigh)",
                        flag(name)
                    ));
                };
                self.defaults.reasoning_effort = Some(effort.to_owned());
                effort.to_owned()
            }
            "thinking_budget" => {
                thinking.budgets =
                    ThinkingBudgets::parse(&text).map_err(|e| format!("{e:#}"))?;
                text
            }
            "thinking_budget_tool_turn_factor" => {
                let f: f64 = parse(name, &text)?;
                // As the start checks the flag.
                check_tool_turn_factor(f).map_err(|e| format!("{e:#}"))?;
                thinking.tool_turn_factor = f;
                f.to_string()
            }
            "thinking_budget_grace" => {
                let v: usize = parse(name, &text)?;
                check_grace(v).map_err(|e| format!("{e:#}"))?;
                thinking.grace = v;
                v.to_string()
            }
            "thinking_nudges" => {
                thinking.nudges = parse_bool(name, &text)?;
                text
            }
            "tool_call_ends_thinking" => {
                thinking.tool_call_ends_thinking = parse_bool(name, &text)?;
                text
            }
            "idle_unload" | "pin_hold" | "disk_cache_ttl" => {
                let secs = parse_duration_secs(&text).map_err(|e| format!("{e:#}"))?;
                *match name {
                    "idle_unload" => &mut self.idle_unload_secs,
                    "pin_hold" => &mut self.pin_hold_secs,
                    _ => &mut self.disk_cache_ttl_secs,
                } = secs;
                text
            }
            "durable_min_tokens" => {
                let v: usize = parse(name, &text)?;
                self.durable_min_tokens = v;
                v.to_string()
            }
            other => unreachable!("{other} is listed in RUNTIME but not handled"),
        };
        Ok(settings::value_for(name, &canonical))
    }
}

/// Why a write was refused, as the route answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// 404 unknown setting, 409 restart-only, 400 invalid value.
    pub status: u16,
    pub message: String,
    pub setting: Option<String>,
}

impl Refusal {
    fn new(status: u16, message: impl Into<String>, setting: &str) -> Self {
        Self { status, message: message.into(), setting: Some(setting.to_owned()) }
    }
}

/// Whether `name` may be written: 404 for a name the settings view does not
/// list, 409 for one that needs a restart.
pub fn writable(name: &str) -> Result<(), Refusal> {
    if applies(name).is_some() {
        Ok(())
    } else if settings::names().any(|n| n == name) {
        Err(Refusal::new(
            409,
            format!("{name} needs a restart and cannot be changed here"),
            name,
        ))
    } else {
        Err(Refusal::new(404, "unknown setting", name))
    }
}

/// A validated change: overrides to set (each value as the settings view
/// shows it) and overrides to delete.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Change {
    pub set: Vec<(String, SettingValue)>,
    pub reset: Vec<String>,
}

/// One setting's change, as the log stores it: the effective value before
/// and after.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    pub old: Option<SettingValue>,
    pub new: Option<SettingValue>,
}

/// A change worked out against the current state: what the overrides and
/// the snapshot become, and the entries for the log (one per setting whose
/// value in effect changed). Setting a value already in effect (the
/// override's, or the flag's or default without one) changes nothing;
/// setting the flag's value while an override is in effect keeps an
/// override with that value (it pins it against a later change of the
/// flag). A reset of a setting without an override changes nothing in
/// effect, but its row is still deleted (a stale one the start ignored).
#[derive(Debug, Clone)]
pub struct Plan {
    overrides: BTreeMap<String, SettingValue>,
    snapshot: Snapshot,
    pub entries: Vec<Entry>,
    reset_rows: Vec<String>,
}

impl Plan {
    /// The override of `name` after the change (`None`: deleted).
    pub fn override_of(&self, name: &str) -> Option<&SettingValue> {
        self.overrides.get(name)
    }

    /// The rows a reset deletes, by name, whether or not they were in effect.
    pub fn reset_rows(&self) -> &[String] {
        &self.reset_rows
    }

    /// Whether the database has anything to write.
    pub fn writes_anything(&self) -> bool {
        !self.entries.is_empty() || !self.reset_rows.is_empty()
    }
}

/// Whether two values are the same setting value: an integer and a number
/// compare as numbers (the listing writes `1`, a JSON body may say `1.0`).
pub fn same_value(a: &Option<SettingValue>, b: &Option<SettingValue>) -> bool {
    match (a, b) {
        (Some(SettingValue::Int(x)), Some(SettingValue::Number(y)))
        | (Some(SettingValue::Number(y)), Some(SettingValue::Int(x))) => {
            *x as f64 == *y
        }
        _ => a == b,
    }
}

/// A PATCH body: a JSON object's members in order, each name once.
pub fn parse_patch(body: &[u8]) -> Result<Vec<(String, Value)>, String> {
    use serde::de::{Deserializer, MapAccess, Visitor};

    /// Every member, duplicates included (`serde_json::Map` keeps the last).
    struct Members(Vec<(String, Value)>);

    impl<'de> serde::Deserialize<'de> for Members {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct Each;
            impl<'de> Visitor<'de> for Each {
                type Value = Members;
                fn expecting(
                    &self,
                    f: &mut std::fmt::Formatter<'_>,
                ) -> std::fmt::Result {
                    f.write_str("a JSON object")
                }
                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<Members, A::Error> {
                    let mut members = Vec::new();
                    while let Some(name) = map.next_key::<String>()? {
                        members.push((name, map.next_value::<Value>()?));
                    }
                    Ok(Members(members))
                }
            }
            d.deserialize_map(Each)
        }
    }

    let Members(members) = serde_json::from_slice(body).map_err(|_| {
        "the body must be a JSON object of setting names and values".to_owned()
    })?;
    for (i, (name, _)) in members.iter().enumerate() {
        if members[..i].iter().any(|(earlier, _)| earlier == name) {
            return Err(format!("the body names {name} more than once"));
        }
    }
    Ok(members)
}

/// What applying the stored overrides at startup did.
#[derive(Debug, Clone, Default)]
pub struct Startup {
    /// The overrides now in effect, from the flag's or default value.
    pub applied: Vec<Entry>,
    /// Rows left alone (kept in the database): the name and why.
    pub ignored: Vec<(String, String)>,
}

#[derive(Debug)]
struct State {
    snapshot: Arc<Snapshot>,
    /// The database's overrides in effect, by name.
    overrides: BTreeMap<String, SettingValue>,
}

/// What the engine thread reads, as atomics (never a lock).
#[derive(Debug, Default)]
struct Atomics {
    idle_unload_secs: Arc<AtomicU64>,
    pin_hold_secs: Arc<AtomicU64>,
    disk_cache_ttl_secs: Arc<AtomicU64>,
    durable_min_tokens: Arc<AtomicUsize>,
}

/// The shared runtime settings: the command line's values, the database's
/// overrides over them, and the snapshot and atomics the consumers read.
pub struct RuntimeSettings {
    /// What the command line (over the defaults) said.
    base: Snapshot,
    /// The settings view's value for each runtime setting without an
    /// override (the flag's or the default; `None` where unset).
    base_values: BTreeMap<String, Option<SettingValue>>,
    state: RwLock<State>,
    atomics: Atomics,
    /// Called after the idle unload or the pin hold changed, so the engine
    /// waits for the new deadline.
    nudge: OnceLock<Box<dyn Fn() + Send + Sync>>,
    /// The recorder's first open attempt is over: the stored overrides are
    /// in effect, or there are none to wait for.
    settled: Arc<AtomicBool>,
    created: Instant,
}

impl std::fmt::Debug for RuntimeSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSettings")
            .field("base", &self.base)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl RuntimeSettings {
    /// The settings as the command line gave them: `base` the values in
    /// effect, `listing` the settings view built from the same flags (its
    /// values are what a reset returns to).
    pub fn new(base: Snapshot, listing: &[Setting]) -> Self {
        let base_values = RUNTIME
            .iter()
            .map(|(name, _)| {
                let value = listing
                    .iter()
                    .find(|s| s.name == *name)
                    .and_then(|s| s.value.clone());
                ((*name).to_owned(), value)
            })
            .collect();
        let settings = Self {
            base: base.clone(),
            base_values,
            state: RwLock::new(State {
                snapshot: Arc::new(base.clone()),
                overrides: BTreeMap::new(),
            }),
            atomics: Atomics::default(),
            nudge: OnceLock::new(),
            settled: Arc::new(AtomicBool::new(false)),
            created: Instant::now(),
        };
        settings.store_atomics(&base);
        settings
    }

    fn store_atomics(&self, s: &Snapshot) {
        let a = &self.atomics;
        a.idle_unload_secs.store(s.idle_unload_secs, Ordering::Release);
        a.pin_hold_secs.store(s.pin_hold_secs, Ordering::Release);
        a.disk_cache_ttl_secs.store(s.disk_cache_ttl_secs, Ordering::Release);
        a.durable_min_tokens.store(s.durable_min_tokens, Ordering::Release);
    }

    /// Marks the values settled: the recorder's first attempt to open its
    /// database is over (with the stored overrides applied, if it opened),
    /// or there is no database.
    pub fn settle(&self) {
        self.settled.store(true, Ordering::Release);
    }

    pub fn settled(&self) -> bool {
        self.settled.load(Ordering::Acquire)
    }

    /// What a startup log line adds to a value it states: nothing once
    /// settled, before that a note that the database may still override it.
    pub fn provisional(&self) -> &'static str {
        if self.settled() {
            ""
        } else {
            " (the flag's value; database overrides may apply)"
        }
    }

    /// The disk tier's gate: no expiry pass until settled, or
    /// [`SETTLE_BOUND`] after these settings were made.
    pub fn expiry_gate(&self) -> super::disk::ExpiryGate {
        super::disk::ExpiryGate::new(self.settled.clone(), self.created + SETTLE_BOUND)
    }

    /// Calls `nudge` after every change of the idle unload or the pin hold;
    /// a second one is refused (`false`).
    pub fn on_timers_changed(&self, nudge: impl Fn() + Send + Sync + 'static) -> bool {
        self.nudge.set(Box::new(nudge)).is_ok()
    }

    /// The values in effect now; a request prepared from it keeps them.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.state.read().unwrap_or_else(|p| p.into_inner()).snapshot.clone()
    }

    /// The database's overrides in effect.
    pub fn overrides(&self) -> BTreeMap<String, SettingValue> {
        self.state.read().unwrap_or_else(|p| p.into_inner()).overrides.clone()
    }

    pub fn idle_unload_secs(&self) -> u64 {
        self.atomics.idle_unload_secs.load(Ordering::Acquire)
    }

    pub fn pin_hold_secs(&self) -> u64 {
        self.atomics.pin_hold_secs.load(Ordering::Acquire)
    }

    pub fn disk_cache_ttl_secs(&self) -> u64 {
        self.atomics.disk_cache_ttl_secs.load(Ordering::Acquire)
    }

    pub fn durable_min_tokens(&self) -> usize {
        self.atomics.durable_min_tokens.load(Ordering::Acquire)
    }

    /// The handle the idle timer reads.
    pub fn idle_unload_handle(&self) -> Arc<AtomicU64> {
        self.atomics.idle_unload_secs.clone()
    }

    /// The handle the weights' pin reads.
    pub fn pin_hold_handle(&self) -> Arc<AtomicU64> {
        self.atomics.pin_hold_secs.clone()
    }

    /// The handle the disk tier reads.
    pub fn disk_cache_ttl_handle(&self) -> Arc<AtomicU64> {
        self.atomics.disk_cache_ttl_secs.clone()
    }

    /// The handle admission reads.
    pub fn durable_min_tokens_handle(&self) -> Arc<AtomicUsize> {
        self.atomics.durable_min_tokens.clone()
    }

    /// The effective value of `name` as the settings view shows it, in
    /// `overrides` (or the flag's or default value without one).
    fn value_in(
        &self,
        overrides: &BTreeMap<String, SettingValue>,
        name: &str,
    ) -> Option<SettingValue> {
        overrides
            .get(name)
            .cloned()
            .or_else(|| self.base_values.get(name).cloned().flatten())
    }

    /// Validates the PATCH body `patch` (name to JSON value) against the
    /// command line's checks: every name first (404, 409), then every value
    /// (400), so nothing is applied unless all of it can be.
    pub fn validate(&self, patch: &[(String, Value)]) -> Result<Change, Refusal> {
        for (name, _) in patch {
            writable(name)?;
        }
        // Each value against the values in effect; the checks are per
        // setting, so their order does not matter.
        let mut scratch = (*self.snapshot()).clone();
        let mut change = Change::default();
        for (name, value) in patch {
            match scratch.set(name, value) {
                Some(Ok(value)) => change.set.push((name.clone(), value)),
                Some(Err(message)) => return Err(Refusal::new(400, message, name)),
                None => unreachable!("writable() let {name} through"),
            }
        }
        Ok(change)
    }

    /// `snapshot` rebuilt from the base with `overrides` on top.
    fn rebuild(
        &self,
        overrides: &BTreeMap<String, SettingValue>,
    ) -> Result<Snapshot, String> {
        let mut snapshot = self.base.clone();
        for (name, value) in overrides {
            let json = serde_json::to_value(value).map_err(|e| e.to_string())?;
            match snapshot.set(name, &json) {
                Some(Ok(_)) => {}
                Some(Err(message)) => return Err(format!("{name}: {message}")),
                None => return Err(format!("{name} is not a runtime setting")),
            }
        }
        Ok(snapshot)
    }

    /// Works `change` out against the state now. Nothing changes until the
    /// plan is [published](Self::publish).
    pub fn plan(&self, change: &Change) -> Result<Plan, String> {
        self.plan_with(change, false)
    }

    /// [`Self::plan`]; `stored` keeps every override set, also one equal to
    /// the value in effect (the database's rows at startup are overrides
    /// whatever they equal).
    fn plan_with(&self, change: &Change, stored: bool) -> Result<Plan, String> {
        let current = self.overrides();
        let mut overrides = current.clone();
        let mut touched: Vec<&str> = Vec::new();
        for (name, value) in &change.set {
            // The value in effect already: nothing to change.
            if !stored
                && same_value(&self.value_in(&overrides, name), &Some(value.clone()))
            {
                continue;
            }
            overrides.insert(name.clone(), value.clone());
            if !touched.contains(&name.as_str()) {
                touched.push(name);
            }
        }
        for name in &change.reset {
            if overrides.remove(name).is_some() && !touched.contains(&name.as_str()) {
                touched.push(name);
            }
        }
        let snapshot = self.rebuild(&overrides)?;
        let entries = touched
            .into_iter()
            .map(|name| Entry {
                name: name.to_owned(),
                old: self.value_in(&current, name),
                new: self.value_in(&overrides, name),
            })
            .collect();
        Ok(Plan { overrides, snapshot, entries, reset_rows: change.reset.clone() })
    }

    /// Puts `plan` into effect: the next request's snapshot, the engine's
    /// atomics, and a nudge for the engine when a timer changed.
    pub fn publish(&self, plan: Plan) {
        let timers = plan
            .entries
            .iter()
            .any(|e| matches!(e.name.as_str(), "idle_unload" | "pin_hold"));
        self.store_atomics(&plan.snapshot);
        {
            let mut state = self.state.write().unwrap_or_else(|p| p.into_inner());
            state.snapshot = Arc::new(plan.snapshot);
            state.overrides = plan.overrides;
        }
        if timers && let Some(nudge) = self.nudge.get() {
            nudge();
        }
    }

    /// Applies the overrides stored in the database (`rows`: name and JSON
    /// text) over the command line. A row for a setting this lily cannot
    /// change at runtime, or whose value no longer passes the flag's
    /// checks, is left out (and left in the database).
    pub fn apply_stored(&self, rows: &[(String, String)]) -> Startup {
        let mut startup = Startup::default();
        let mut change = Change::default();
        let mut scratch = (*self.snapshot()).clone();
        for (name, text) in rows {
            if let Err(refusal) = writable(name) {
                let why = if refusal.status == 404 {
                    "not a setting this lily knows".to_owned()
                } else {
                    "not a runtime setting in this lily".to_owned()
                };
                startup.ignored.push((name.clone(), why));
                continue;
            }
            let value = match serde_json::from_str::<Value>(text) {
                Ok(value) => value,
                Err(e) => {
                    startup.ignored.push((
                        name.clone(),
                        format!("unreadable value {text:?}: {e}"),
                    ));
                    continue;
                }
            };
            match scratch.set(name, &value) {
                Some(Ok(value)) => change.set.push((name.clone(), value)),
                Some(Err(message)) => startup.ignored.push((name.clone(), message)),
                None => unreachable!("writable() let {name} through"),
            }
        }
        match self.plan_with(&change, true) {
            Ok(plan) => {
                startup.applied = plan.entries.clone();
                self.publish(plan);
            }
            // Every value passed alone; refusing them together cannot happen,
            // but if it did the command line stays in effect.
            Err(why) => startup
                .ignored
                .extend(change.set.into_iter().map(|(n, _)| (n, why.clone()))),
        }
        startup
    }

    /// The settings view's `listing` with the overrides in effect: their
    /// value, seconds and `database` as the source.
    pub fn overlay(&self, listing: &[Setting]) -> Vec<Setting> {
        let overrides = self.overrides();
        listing
            .iter()
            .map(|s| match overrides.get(&s.name) {
                Some(value) => settings::with_override(s, value),
                None => s.clone(),
            })
            .collect()
    }
}

/// A command line and the runtime settings it makes, for the tests of the
/// modules that use them.
#[cfg(test)]
pub mod testing {
    use super::*;
    use crate::kernels::sample::SamplingParams;
    use crate::serve::api::ThinkingDefaults;
    use crate::serve::dashboard::contract::Setting;
    use crate::serve::dashboard::settings::{self, FlagValue};

    /// The thinking budget's flag default.
    pub const BUDGET: &str = "low=8000,medium=8000,xhigh=16000";

    fn flag(name: &str, given: Option<&str>, default: Option<&str>) -> FlagValue {
        FlagValue {
            name: name.to_owned(),
            given: given.map(str::to_owned),
            default: default.map(str::to_owned),
        }
    }

    /// `--temperature 0.6 --idle-unload 30m` over clap's defaults and a
    /// checkpoint whose sampling defaults are temperature 1, top_k 20,
    /// top_p 0.95: the settings view's listing.
    pub fn listing() -> Vec<Setting> {
        settings::build(
            &[
                flag("temperature", Some("0.6"), None),
                flag("idle_unload", Some("30m"), Some("0")),
                flag("pin_hold", None, Some("1m")),
                flag("disk_cache_ttl", None, Some("3d")),
                flag("durable_min_tokens", None, Some("1024")),
                flag("thinking_budget", None, Some(BUDGET)),
                flag("thinking_budget_tool_turn_factor", None, Some("1")),
                flag("thinking_budget_grace", None, Some("128")),
                flag("thinking_nudges", None, Some("true")),
                flag("tool_call_ends_thinking", None, Some("true")),
                flag("max_seq", None, Some("131072")),
            ],
            &[
                ("temperature", SettingValue::Number(1.0)),
                ("top_k", SettingValue::Int(20)),
                ("top_p", SettingValue::Number(0.95)),
                ("min_p", SettingValue::Int(0)),
                ("presence_penalty", SettingValue::Int(0)),
                ("frequency_penalty", SettingValue::Int(0)),
                ("repetition_penalty", SettingValue::Int(1)),
            ],
        )
    }

    /// The values the server starts with from [`listing`]'s command line.
    pub fn snapshot() -> Snapshot {
        Snapshot {
            defaults: Defaults {
                sampling: SamplingParams {
                    temperature: 0.6,
                    top_k: 20,
                    top_p: 0.95,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repetition_penalty: 1.0,
                    seed: 0,
                },
                thinking: true,
                reasoning_effort: None,
                thinking_controls: ThinkingDefaults {
                    budgets: ThinkingBudgets::parse(BUDGET).expect("budget"),
                    tool_turn_factor: 1.0,
                    nudges: true,
                    tool_call_ends_thinking: true,
                    grace: crate::thinking::DEFAULT_GRACE,
                },
            },
            idle_unload_secs: 1800,
            pin_hold_secs: 60,
            disk_cache_ttl_secs: 3 * 86_400,
            durable_min_tokens: 1024,
        }
    }

    pub fn runtime() -> Arc<RuntimeSettings> {
        Arc::new(RuntimeSettings::new(snapshot(), &listing()))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/runtime.rs"]
mod tests;
