//! `GET /v1/lily/settings`: every command-line setting with its value, its
//! default and where the value came from (docs/dashboard.md § 7).
//!
//! The values are what the command line said, as the flag's text: lily
//! itself (`src/main.rs`) hands over each argument's text and its default
//! ([`FlagValue`], from clap), so a duration shows as `30m` and its
//! `value_s` is what `parse_duration_secs` makes of it. A few defaults are
//! not clap's but computed (the checkpoint's sampling defaults, the disk
//! tier's directory); the server passes those in. Nothing is editable yet:
//! runtime settings are phase 4, and until then every setting is
//! `editable: false` with no `applies`.
//!
//! The groups and names follow the dashboard's mock (`dashboard/src/mock/
//! data.ts`), so the Settings view renders the same sections.

use super::contract::{Setting, SettingSource, SettingType, SettingValue};
use crate::serve::parse_duration_secs;

/// One command-line argument: clap's id (the `Cli` field name, which is
/// also the setting's name), the text given on the command line (`None`
/// when it was not given) and the text of clap's default (`None` when the
/// flag has none).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlagValue {
    pub name: String,
    pub given: Option<String>,
    pub default: Option<String>,
}

/// How a setting's text reads as a JSON value.
#[derive(Clone, Copy)]
enum Kind {
    Number,
    Int,
    Enum(&'static [&'static str]),
    Text,
    Bool,
    Duration,
}

/// A setting the dashboard lists: the flag `--<name with dashes>`, its
/// section, its kind, the bounds the flag's own validation enforces (none
/// where it enforces none) and a one-line definition.
struct Descriptor {
    name: &'static str,
    group: &'static str,
    kind: Kind,
    min: Option<f64>,
    max: Option<f64>,
    help: Option<&'static str>,
}

const fn d(name: &'static str, group: &'static str, kind: Kind) -> Descriptor {
    Descriptor { name, group, kind, min: None, max: None, help: None }
}

impl Descriptor {
    const fn bounds(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    const fn help(mut self, help: &'static str) -> Self {
        self.help = Some(help);
        self
    }
}

const SAMPLING: &str = "Sampling";
const THINKING: &str = "Thinking";
const LIFECYCLE: &str = "Lifecycle";
const CACHES: &str = "Caches";
const ENGINE: &str = "Engine";

/// Every setting shown, in display order. The bounds are those of
/// `SamplingParams::validate` and the flags' own checks; the hidden flags
/// (a retired one, a fault injection for tests) are not settings.
const DESCRIPTORS: &[Descriptor] = &[
    d("temperature", SAMPLING, Kind::Number)
        .bounds(Some(0.0), None)
        .help("Default when a request sets none; 0 is greedy."),
    d("top_p", SAMPLING, Kind::Number).bounds(Some(0.0), Some(1.0)),
    d("top_k", SAMPLING, Kind::Int)
        .bounds(Some(0.0), None)
        .help("0 keeps every token."),
    d("min_p", SAMPLING, Kind::Number).bounds(Some(0.0), Some(1.0)),
    d("presence_penalty", SAMPLING, Kind::Number),
    d("frequency_penalty", SAMPLING, Kind::Number),
    d("repetition_penalty", SAMPLING, Kind::Number).bounds(Some(0.0), None),
    d("thinking", THINKING, Kind::Bool)
        .help("Chat prompts open a reasoning block unless the request says otherwise."),
    // The flag's text goes to the chat template as it is (unlike a request's
    // `reasoning_effort`, which `api::map_reasoning_effort` maps from none,
    // low, medium, high, xhigh or max): these are the efforts the template
    // and the thinking budgets know.
    d("reasoning_effort", THINKING, Kind::Enum(&["low", "medium", "xhigh"]))
        .help("Unset: the template's default (xhigh)."),
    d("thinking_budget", THINKING, Kind::Text)
        .help("Tokens by reasoning effort, one number for all, or off."),
    d("thinking_budget_tool_turn_factor", THINKING, Kind::Number)
        .bounds(Some(0.0), None),
    d("thinking_budget_grace", THINKING, Kind::Int).bounds(Some(0.0), None),
    d("thinking_nudges", THINKING, Kind::Bool),
    d("tool_call_ends_thinking", THINKING, Kind::Bool),
    d("thinking_texts", THINKING, Kind::Text),
    d("idle_unload", LIFECYCLE, Kind::Duration).help("0 never unloads."),
    d("pin_weights", LIFECYCLE, Kind::Enum(&["auto", "always", "off"])),
    d("pin_hold", LIFECYCLE, Kind::Duration)
        .help("0 holds the pin until memory pressure or the unload."),
    d("disk_cache_ttl", CACHES, Kind::Duration)
        .help("0 keeps entries until the budget evicts them."),
    d("cache_bytes", CACHES, Kind::Text).help("Unset: derived at load (see the log)."),
    d("max_sessions", CACHES, Kind::Int),
    d("disk_cache_dir", CACHES, Kind::Text),
    d("disk_cache_bytes", CACHES, Kind::Text).help("0 turns the disk tier off."),
    d("durable_min_tokens", CACHES, Kind::Int)
        .help("0 turns durable prefix entries off."),
    d("decode_checkpoint_tokens", CACHES, Kind::Int)
        .help("0 turns decode checkpoints off."),
    d("ngram_table", CACHES, Kind::Enum(&["paged", "resident"])),
    d("ngram_preload", CACHES, Kind::Bool),
    d("ngram_lock", CACHES, Kind::Bool),
    d("model", ENGINE, Kind::Text),
    d("bind", ENGINE, Kind::Text),
    d("max_seq", ENGINE, Kind::Int),
    d("kv_cache", ENGINE, Kind::Enum(&["q8", "bf16"])),
    d("max_batch", ENGINE, Kind::Int),
    d("mtp_drafts", ENGINE, Kind::Int).help("0 turns speculative decoding off."),
    d("queue", ENGINE, Kind::Int),
    d("memory_limit_gb", ENGINE, Kind::Number)
        .help("Unset: planned from the machine's memory."),
    d("vision", ENGINE, Kind::Enum(&["auto", "off"])),
    d("image_max_pixels", ENGINE, Kind::Int),
    d("image_min_pixels", ENGINE, Kind::Int),
];

/// A setting's text as the JSON value its kind reads as; a text that does
/// not parse (which the flag's own parser would have refused at startup)
/// stays a string rather than disappearing.
fn value_of(kind: Kind, text: &str) -> SettingValue {
    let text = text.trim();
    match kind {
        Kind::Int => text
            .parse()
            .map_or_else(|_| SettingValue::Text(text.into()), SettingValue::Int),
        Kind::Number => match text.parse::<u64>() {
            Ok(n) => SettingValue::Int(n),
            Err(_) => text
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map_or_else(|| SettingValue::Text(text.into()), SettingValue::Number),
        },
        Kind::Bool => match text {
            "true" => SettingValue::Bool(true),
            "false" => SettingValue::Bool(false),
            other => SettingValue::Text(other.into()),
        },
        Kind::Enum(_) | Kind::Text | Kind::Duration => SettingValue::Text(text.into()),
    }
}

fn kind_name(kind: Kind) -> SettingType {
    match kind {
        Kind::Number => SettingType::Number,
        Kind::Int => SettingType::Int,
        Kind::Enum(_) => SettingType::Enum,
        Kind::Text => SettingType::String,
        Kind::Bool => SettingType::Bool,
        Kind::Duration => SettingType::Duration,
    }
}

/// Every setting from the command line's `flags` (a setting whose flag is
/// missing there shows as unset), with `computed` defaults for the flags
/// whose default is not clap's (the checkpoint's sampling defaults, the
/// derived disk directory, the K/V format and batch size the server falls
/// back to). The value is the flag's text when given, else the default;
/// the source says which.
pub fn build(flags: &[FlagValue], computed: &[(&str, SettingValue)]) -> Vec<Setting> {
    DESCRIPTORS
        .iter()
        .map(|desc| {
            let flag = flags.iter().find(|f| f.name == desc.name);
            let given = flag.and_then(|f| f.given.as_deref());
            let default = flag
                .and_then(|f| f.default.as_deref())
                .map(|text| value_of(desc.kind, text))
                .or_else(|| {
                    computed
                        .iter()
                        .find(|(n, _)| *n == desc.name)
                        .map(|(_, v)| v.clone())
                });
            let flag_value = given.map(|text| value_of(desc.kind, text));
            let value = flag_value.clone().or_else(|| default.clone());
            let value_s = match (desc.kind, &value) {
                (Kind::Duration, Some(SettingValue::Text(text))) => {
                    parse_duration_secs(text).ok()
                }
                _ => None,
            };
            let numeric = matches!(desc.kind, Kind::Number | Kind::Int);
            Setting {
                name: desc.name.to_owned(),
                group: desc.group.to_owned(),
                kind: kind_name(desc.kind),
                editable: false,
                flag: format!("--{}", desc.name.replace('_', "-")),
                default,
                flag_value,
                value,
                value_s,
                source: if given.is_some() {
                    SettingSource::Flag
                } else {
                    SettingSource::Default
                },
                min: desc.min.filter(|_| numeric),
                max: desc.max.filter(|_| numeric),
                step: None,
                options: match desc.kind {
                    Kind::Enum(options) => {
                        Some(options.iter().map(|o| (*o).to_owned()).collect())
                    }
                    _ => None,
                },
                help: desc.help.map(str::to_owned),
                applies: None,
            }
        })
        .collect()
}

/// The names the dashboard lists, for tests that hold the table against
/// the command line.
pub fn names() -> impl Iterator<Item = &'static str> {
    DESCRIPTORS.iter().map(|d| d.name)
}
