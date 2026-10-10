use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use serde_json::json;

use super::testing::{BUDGET, listing, runtime, snapshot};
use super::*;
use crate::serve::dashboard::contract::SettingSource;

/// `name` set to `value` on the starting snapshot.
fn set(name: &str, value: Value) -> Result<SettingValue, String> {
    snapshot().set(name, &value).expect("a runtime setting")
}

fn refused(name: &str, value: Value) -> String {
    set(name, value).expect_err("refused")
}

fn patch(value: Value) -> Vec<(String, Value)> {
    parse_patch(&serde_json::to_vec(&value).unwrap()).expect("an object")
}

#[test]
fn every_runtime_setting_is_listed_and_editable_and_says_when_it_applies() {
    let all = listing();
    for (name, applies) in RUNTIME {
        let s = all.iter().find(|s| s.name == *name).expect(name);
        assert!(s.editable, "{name}");
        assert_eq!(s.applies.as_deref(), Some(*applies), "{name}");
    }
    // The three the spec left open: one runtime, two restart-only.
    assert_eq!(super::applies("durable_min_tokens"), Some("next request"));
    for name in
        ["queue", "decode_checkpoint_tokens", "max_seq", "thinking", "thinking_texts"]
    {
        let s = all.iter().find(|s| s.name == name).expect(name);
        assert!(!s.editable && s.applies.is_none(), "{name}");
    }
    assert_eq!(super::applies("idle_unload"), Some("timer re-armed on change"));
    assert_eq!(super::applies("pin_hold"), Some("hold re-armed on change"));
    assert_eq!(super::applies("disk_cache_ttl"), Some("next expiry pass"));
}

#[test]
fn sampling_values_are_checked_as_the_start_checks_the_defaults() {
    // Accepted: the value as the listing shows it (the f32's shortest text).
    assert_eq!(set("temperature", json!(0.7)), Ok(SettingValue::Number(0.7)));
    assert_eq!(set("temperature", json!("0.7")), Ok(SettingValue::Number(0.7)));
    assert_eq!(set("temperature", json!(0)), Ok(SettingValue::Int(0)));
    assert_eq!(set("top_k", json!(40)), Ok(SettingValue::Int(40)));
    assert_eq!(set("top_k", json!(0)), Ok(SettingValue::Int(0)));
    assert_eq!(set("presence_penalty", json!(-1.5)), Ok(SettingValue::Number(-1.5)));
    assert_eq!(set("top_p", json!(1)), Ok(SettingValue::Int(1)));
    // Refused with `SamplingParams::validate`'s messages, which a bad flag
    // fails the start with.
    assert_eq!(
        refused("temperature", json!(-1)),
        "temperature must be a finite number >= 0"
    );
    assert_eq!(
        refused("temperature", json!("inf")),
        "temperature must be a finite number >= 0"
    );
    assert_eq!(refused("top_p", json!(0)), "top_p must be in (0, 1]");
    assert_eq!(refused("top_p", json!(1.5)), "top_p must be in (0, 1]");
    assert_eq!(refused("min_p", json!(2)), "min_p must be in [0, 1]");
    assert_eq!(
        refused("repetition_penalty", json!(0)),
        "repetition_penalty must be > 0"
    );
    assert_eq!(refused("frequency_penalty", json!("NaN")), "penalties must be finite");
    // Text the flag's parser refuses, as clap words it.
    assert_eq!(
        refused("temperature", json!("warm")),
        "invalid value 'warm' for '--temperature': invalid float literal"
    );
    // Not trimmed, as clap does not trim.
    assert_eq!(
        refused("temperature", json!(" 0.5")),
        "invalid value ' 0.5' for '--temperature': invalid float literal"
    );
    assert_eq!(
        refused("top_k", json!(-1)),
        "invalid value '-1' for '--top-k': invalid digit found in string"
    );
    assert!(
        refused("top_k", json!(2.5)).starts_with("invalid value '2.5' for '--top-k'")
    );
    // What is no flag text at all.
    for value in [json!(null), json!([1]), json!({"a": 1})] {
        assert_eq!(
            refused("temperature", value),
            "--temperature takes a number, a string or a boolean, as its flag's text"
        );
    }
}

#[test]
fn thinking_values_are_checked_by_the_flags_own_parsers() {
    assert_eq!(
        set("reasoning_effort", json!("low")),
        Ok(SettingValue::Text("low".into()))
    );
    // The requests' mapping: high and max are xhigh, stored as xhigh.
    for alias in ["high", "max", "xhigh"] {
        assert_eq!(
            set("reasoning_effort", json!(alias)),
            Ok(SettingValue::Text("xhigh".into()))
        );
    }
    for bad in ["turbo", "none", "Low", " low"] {
        assert_eq!(
            refused("reasoning_effort", json!(bad)),
            format!(
                "invalid value '{bad}' for '--reasoning-effort': use low, medium or xhigh (high and max are xhigh)"
            )
        );
    }
    // `ThinkingBudgets::parse`, its messages with their context.
    assert_eq!(
        set("thinking_budget", json!("off")),
        Ok(SettingValue::Text("off".into()))
    );
    assert_eq!(
        set("thinking_budget", json!(4000)),
        Ok(SettingValue::Text("4000".into()))
    );
    for text in ["0", "low=abc", "turbo=5", "low=0", "8000x"] {
        let expected = format!("{:#}", ThinkingBudgets::parse(text).expect_err(text));
        assert_eq!(refused("thinking_budget", json!(text)), expected, "{text}");
    }
    let mut s = snapshot();
    s.set("thinking_budget", &json!("low=100, xhigh=900")).unwrap().unwrap();
    let b = s.defaults.thinking_controls.budgets;
    assert_eq!((b.low, b.medium, b.xhigh), (Some(100), None, Some(900)));
    // The tool-turn factor's check is `src/main.rs`'s.
    assert_eq!(
        set("thinking_budget_tool_turn_factor", json!(0.5)),
        Ok(SettingValue::Number(0.5))
    );
    for value in
        [json!(0), json!(-1), json!("nan"), json!("inf"), json!(100.5), json!(1e300)]
    {
        let expected = format!(
            "{:#}",
            crate::serve::api::check_tool_turn_factor(
                value.as_f64().unwrap_or(f64::NAN)
            )
            .unwrap_err()
        );
        assert_eq!(refused("thinking_budget_tool_turn_factor", value), expected);
        assert_eq!(
            expected,
            "--thinking-budget-tool-turn-factor must be a positive number of at most 100"
        );
    }
    assert!(set("thinking_budget_tool_turn_factor", json!(100)).is_ok());
    // Caps far above any context, so nothing they feed overflows.
    let cap = crate::serve::api::MAX_THINKING_TOKENS;
    assert!(set("thinking_budget_grace", json!(cap)).is_ok());
    assert_eq!(
        refused("thinking_budget_grace", json!(cap + 1)),
        format!("--thinking-budget-grace must be at most {cap} tokens")
    );
    assert!(set("thinking_budget", json!(cap)).is_ok());
    assert_eq!(
        refused("thinking_budget", json!(cap + 1)),
        format!("a thinking budget must be at most {cap} tokens")
    );
    assert!(
        refused("thinking_budget", json!(format!("low={}", cap + 1)))
            .contains("at most")
    );
    assert_eq!(set("thinking_budget_grace", json!(64)), Ok(SettingValue::Int(64)));
    assert!(
        refused("thinking_budget_grace", json!("a lot"))
            .contains("--thinking-budget-grace")
    );
    // Switches: true or false, as JSON or as the flag's text.
    assert_eq!(set("thinking_nudges", json!(false)), Ok(SettingValue::Bool(false)));
    assert_eq!(
        set("tool_call_ends_thinking", json!("false")),
        Ok(SettingValue::Bool(false))
    );
    assert_eq!(
        refused("thinking_nudges", json!("yes")),
        "invalid value 'yes' for '--thinking-nudges': use true or false"
    );
    assert_eq!(
        refused("thinking_nudges", json!(1)),
        refused("thinking_nudges", json!("1"))
    );
}

#[test]
fn durations_are_parsed_exactly_as_the_flag_parses_them() {
    for text in [
        "30m", "90", "1.5h", "2d", "45s", "0", " 10m ", "10M", "1h ", "0.5", "3d",
        "abc", "10x", "-5", "", "m", "1e3", "5 m", "..", "1.2.3h",
    ] {
        for name in ["idle_unload", "pin_hold", "disk_cache_ttl"] {
            let mut s = snapshot();
            let got = s.set(name, &json!(text)).expect("runtime");
            match parse_duration_secs(text) {
                Ok(secs) => {
                    assert_eq!(
                        got,
                        Ok(SettingValue::Text(text.trim().into())),
                        "{name} {text:?}"
                    );
                    let stored = match name {
                        "idle_unload" => s.idle_unload_secs,
                        "pin_hold" => s.pin_hold_secs,
                        _ => s.disk_cache_ttl_secs,
                    };
                    assert_eq!(stored, secs, "{name} {text:?}");
                }
                Err(error) => {
                    assert_eq!(got, Err(format!("{error:#}")), "{name} {text:?}")
                }
            }
        }
    }
    // Past the year-long cap, as the flag refuses it.
    for text in ["366d", "8761h", "31536001", "1e400", "99999999999999999999999d"] {
        let mut s = snapshot();
        let got = s.set("idle_unload", &json!(text)).unwrap();
        assert_eq!(
            got,
            Err(format!("{:#}", parse_duration_secs(text).unwrap_err())),
            "{text}"
        );
    }
    assert_eq!(
        refused("pin_hold", json!("366d")),
        "duration \"366d\" is longer than the 365-day cap"
    );
    assert!(set("disk_cache_ttl", json!("365d")).is_ok());
    // A JSON number is the flag's bare seconds.
    let mut s = snapshot();
    assert_eq!(
        s.set("pin_hold", &json!(90)).unwrap(),
        Ok(SettingValue::Text("90".into()))
    );
    assert_eq!(s.pin_hold_secs, 90);
    assert!(s.set("idle_unload", &json!(1.5)).unwrap().is_ok(), "1.5 s, rounded down");
    assert_eq!(s.idle_unload_secs, 1);
    assert_eq!(set("durable_min_tokens", json!(0)), Ok(SettingValue::Int(0)));
}

#[test]
fn a_patch_names_its_settings_first_then_its_values_and_applies_nothing_on_a_refusal() {
    let settings = runtime();
    let refusal = |p: Value| settings.validate(&patch(p)).expect_err("refused");
    // Unknown before restart-only before an invalid value, wherever they are.
    let r = refusal(json!({"temperature": "warm", "nope": 1}));
    assert_eq!(
        (r.status, r.setting.as_deref(), r.message.as_str()),
        (404, Some("nope"), "unknown setting")
    );
    let r = refusal(json!({"temperature": "warm", "max_seq": 1}));
    assert_eq!((r.status, r.setting.as_deref()), (409, Some("max_seq")));
    assert_eq!(r.message, "max_seq needs a restart and cannot be changed here");
    let r = refusal(json!({"temperature": 0.5, "top_p": 5}));
    assert_eq!((r.status, r.setting.as_deref()), (400, Some("top_p")));
    assert_eq!(r.message, "top_p must be in (0, 1]");
    assert!(settings.overrides().is_empty(), "nothing was applied");
    assert_eq!(settings.snapshot().defaults.sampling.temperature, 0.6);
    // All good: the values as the listing shows them, nothing applied yet.
    let change = settings
        .validate(&patch(json!({"temperature": 0.5, "idle_unload": " 5m"})))
        .expect("valid");
    // In the body's order.
    assert_eq!(
        change.set,
        vec![
            ("temperature".to_owned(), SettingValue::Number(0.5)),
            ("idle_unload".to_owned(), SettingValue::Text("5m".into())),
        ]
    );
    assert!(settings.overrides().is_empty());
}

#[test]
fn a_published_change_reaches_the_next_snapshot_and_the_atomics_never_an_older_snapshot()
 {
    let settings = runtime();
    // A request prepared before the change holds this.
    let before = settings.snapshot();
    let handle = settings.idle_unload_handle();
    let change = settings
        .validate(&patch(json!({
            "temperature": 0.2,
            "idle_unload": "5m",
            "pin_hold": "0",
            "disk_cache_ttl": "1d",
            "durable_min_tokens": 4096,
            "reasoning_effort": "medium",
        })))
        .expect("valid");
    let plan = settings.plan(&change).expect("plan");
    // Planning changes nothing.
    assert_eq!(settings.snapshot().defaults.sampling.temperature, 0.6);
    let entries: Vec<_> = plan
        .entries
        .iter()
        .map(|e| (e.name.as_str(), e.old.clone(), e.new.clone()))
        .collect();
    // One entry per setting, in the body's order: the value before (the
    // flag's or the default, null for an unset one) and after.
    assert_eq!(
        entries,
        vec![
            (
                "temperature",
                Some(SettingValue::Number(0.6)),
                Some(SettingValue::Number(0.2))
            ),
            (
                "idle_unload",
                Some(SettingValue::Text("30m".into())),
                Some(SettingValue::Text("5m".into()))
            ),
            (
                "pin_hold",
                Some(SettingValue::Text("1m".into())),
                Some(SettingValue::Text("0".into()))
            ),
            (
                "disk_cache_ttl",
                Some(SettingValue::Text("3d".into())),
                Some(SettingValue::Text("1d".into()))
            ),
            (
                "durable_min_tokens",
                Some(SettingValue::Int(1024)),
                Some(SettingValue::Int(4096))
            ),
            ("reasoning_effort", None, Some(SettingValue::Text("medium".into()))),
        ]
    );
    settings.publish(plan);
    let after = settings.snapshot();
    assert_eq!(after.defaults.sampling.temperature, 0.2);
    assert_eq!(after.defaults.reasoning_effort.as_deref(), Some("medium"));
    // The request that took the old snapshot keeps it.
    assert_eq!(before.defaults.sampling.temperature, 0.6);
    assert_eq!(before.defaults.reasoning_effort, None);
    assert_eq!(before.idle_unload_secs, 1800);
    // The engine's atomics, the handles it holds included.
    assert_eq!(handle.load(Ordering::Acquire), 300);
    assert_eq!(
        (
            settings.idle_unload_secs(),
            settings.pin_hold_secs(),
            settings.disk_cache_ttl_secs(),
            settings.durable_min_tokens()
        ),
        (300, 0, 86_400, 4096)
    );
    // Setting an override to its own value again changes nothing.
    let again = settings.validate(&patch(json!({"temperature": "0.2"}))).unwrap();
    assert!(settings.plan(&again).unwrap().entries.is_empty());
}

#[test]
fn a_prepared_request_keeps_the_snapshot_it_was_prepared_from() {
    use crate::serve::api::{SamplingFields, resolve_sampling};
    let resolve = |d: &Defaults| {
        resolve_sampling(&SamplingFields::default(), &d.sampling).unwrap()
    };
    let settings = runtime();
    // Preparation reads one snapshot and resolves the request from it.
    let snapshot = settings.snapshot();
    let change =
        settings.validate(&patch(json!({"temperature": 1.3, "top_k": 5}))).unwrap();
    settings.publish(settings.plan(&change).unwrap());
    let prepared = resolve(&snapshot.defaults);
    assert_eq!((prepared.temperature, prepared.top_k), (0.6, 20), "the old values");
    let next = resolve(&settings.snapshot().defaults);
    assert_eq!((next.temperature, next.top_k), (1.3, 5), "the next request's");
}

#[test]
fn a_reset_returns_to_the_flag_or_else_the_default() {
    let settings = runtime();
    let change = settings
        .validate(&patch(
            json!({"idle_unload": "5m", "pin_hold": "10s", "temperature": 1.1}),
        ))
        .unwrap();
    settings.publish(settings.plan(&change).unwrap());
    let reset = Change {
        set: Vec::new(),
        reset: vec![
            "idle_unload".into(),
            "pin_hold".into(),
            "temperature".into(),
            "top_p".into(),
        ],
    };
    let plan = settings.plan(&reset).unwrap();
    // top_p had no override: resetting it changes nothing.
    let entries: Vec<_> =
        plan.entries.iter().map(|e| (e.name.as_str(), e.new.clone())).collect();
    assert_eq!(
        entries,
        vec![
            ("idle_unload", Some(SettingValue::Text("30m".into()))),
            ("pin_hold", Some(SettingValue::Text("1m".into()))),
            ("temperature", Some(SettingValue::Number(0.6))),
        ]
    );
    settings.publish(plan);
    assert!(settings.overrides().is_empty());
    // idle_unload back at its flag (30m), pin_hold at its default (1m), the
    // temperature at its flag over the checkpoint's 1.0.
    assert_eq!((settings.idle_unload_secs(), settings.pin_hold_secs()), (1800, 60));
    assert_eq!(settings.snapshot().defaults.sampling.temperature, 0.6);
    let shown = settings.overlay(&listing());
    let get = |name: &str| shown.iter().find(|s| s.name == name).unwrap().clone();
    assert_eq!(get("idle_unload").source, SettingSource::Flag);
    assert_eq!(get("pin_hold").source, SettingSource::Default);
}

#[test]
fn the_listing_shows_an_override_with_its_seconds_and_the_database_as_source() {
    let settings = runtime();
    let change =
        settings.validate(&patch(json!({"pin_hold": "2m", "top_k": 7}))).unwrap();
    settings.publish(settings.plan(&change).unwrap());
    let shown = settings.overlay(&listing());
    let get = |name: &str| shown.iter().find(|s| s.name == name).unwrap().clone();
    let hold = get("pin_hold");
    assert_eq!(
        (hold.value, hold.value_s, hold.source),
        (Some(SettingValue::Text("2m".into())), Some(120), SettingSource::Database)
    );
    assert_eq!(
        hold.default,
        Some(SettingValue::Text("1m".into())),
        "the default stays"
    );
    let top_k = get("top_k");
    assert_eq!(
        (top_k.value, top_k.source),
        (Some(SettingValue::Int(7)), SettingSource::Database)
    );
    assert_eq!(get("temperature").source, SettingSource::Flag);
}

#[test]
fn stored_overrides_come_over_the_flags_and_bad_rows_are_left_out() {
    let settings = runtime();
    let rows: Vec<(String, String)> = [
        ("temperature", "0.7"),
        ("idle_unload", "\"5m\""),
        ("max_seq", "1"),
        ("from_a_newer_lily", "1"),
        ("top_p", "5"),
        ("pin_hold", "not json"),
        ("thinking_budget", format!("\"{BUDGET}\"").as_str()),
    ]
    .iter()
    .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
    .collect();
    let startup = settings.apply_stored(&rows);
    let applied: Vec<_> = startup
        .applied
        .iter()
        .map(|e| (e.name.as_str(), e.old.clone(), e.new.clone()))
        .collect();
    assert_eq!(
        applied,
        vec![
            (
                "temperature",
                Some(SettingValue::Number(0.6)),
                Some(SettingValue::Number(0.7))
            ),
            (
                "idle_unload",
                Some(SettingValue::Text("30m".into())),
                Some(SettingValue::Text("5m".into()))
            ),
            // An override equal to the flag is still an override.
            (
                "thinking_budget",
                Some(SettingValue::Text(BUDGET.into())),
                Some(SettingValue::Text(BUDGET.into()))
            ),
        ]
    );
    let ignored: Vec<_> =
        startup.ignored.iter().map(|(n, why)| (n.as_str(), why.as_str())).collect();
    assert_eq!(ignored[0], ("max_seq", "not a runtime setting in this lily"));
    assert_eq!(ignored[1], ("from_a_newer_lily", "not a setting this lily knows"));
    assert_eq!(ignored[2], ("top_p", "top_p must be in (0, 1]"));
    assert_eq!(ignored[3].0, "pin_hold");
    assert!(
        ignored[3].1.starts_with("unreadable value \"not json\""),
        "{}",
        ignored[3].1
    );
    // Database over flag (0.6) over the checkpoint's default (1.0).
    assert_eq!(settings.snapshot().defaults.sampling.temperature, 0.7);
    assert_eq!(settings.idle_unload_secs(), 300);
    assert_eq!(settings.pin_hold_secs(), 60, "the bad row left the default");
    assert_eq!(settings.snapshot().defaults.sampling.top_p, 0.95);
}

#[test]
fn a_change_of_a_timer_nudges_the_engine_and_nothing_else_does() {
    let settings = runtime();
    let nudges = Arc::new(AtomicUsize::new(0));
    {
        let nudges = nudges.clone();
        assert!(settings.on_timers_changed(move || {
            nudges.fetch_add(1, Ordering::Relaxed);
        }));
    }
    assert!(!settings.on_timers_changed(|| {}), "one nudge only");
    let publish = |p: Value| {
        let change = settings.validate(&patch(p)).unwrap();
        settings.publish(settings.plan(&change).unwrap());
    };
    publish(json!({"temperature": 0.9, "disk_cache_ttl": "1h"}));
    assert_eq!(nudges.load(Ordering::Relaxed), 0);
    publish(json!({"idle_unload": "10m"}));
    publish(json!({"pin_hold": "5s"}));
    assert_eq!(nudges.load(Ordering::Relaxed), 2);
    // Nothing changed, nothing to nudge.
    publish(json!({"pin_hold": "5s"}));
    assert_eq!(nudges.load(Ordering::Relaxed), 2);
}

#[test]
fn readers_on_other_threads_see_whole_snapshots() {
    // A reader never sees a half-applied change: temperature and top_k move
    // together.
    let settings = runtime();
    let seen = Arc::new(Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let reader = {
            let (settings, seen) = (settings.clone(), seen.clone());
            scope.spawn(move || {
                for _ in 0..2000 {
                    let s = settings.snapshot();
                    seen.lock().unwrap().push((
                        s.defaults.sampling.temperature,
                        s.defaults.sampling.top_k,
                    ));
                }
            })
        };
        for i in 0..200u32 {
            let k = (i % 2) as usize;
            let change = settings
                .validate(&patch(json!({"temperature": if k == 0 { 0.1 } else { 0.9 }, "top_k": k + 1})))
                .unwrap();
            settings.publish(settings.plan(&change).unwrap());
        }
        reader.join().unwrap();
    });
    for (t, k) in seen.lock().unwrap().iter() {
        assert!(matches!((t, k), (0.6, 20) | (0.1, 1) | (0.9, 2)), "{t} {k}");
    }
}

#[test]
fn a_patch_body_is_an_object_naming_each_setting_once_in_its_order() {
    assert_eq!(
        parse_patch(br#"{"top_k": 5, "temperature": 0.5}"#),
        Ok(vec![("top_k".into(), json!(5)), ("temperature".into(), json!(0.5))])
    );
    assert_eq!(
        parse_patch(br#"{"top_k": 5, "temperature": 0.5, "top_k": 6}"#),
        Err("the body names top_k more than once".into())
    );
    for bad in [&b"[1]"[..], b"", b"{", b"\"x\"", b"null", b"{} {}"] {
        assert_eq!(
            parse_patch(bad),
            Err("the body must be a JSON object of setting names and values".into()),
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
}

#[test]
fn the_value_in_effect_is_a_no_op_and_the_flags_value_over_an_override_pins_it() {
    let settings = runtime();
    let plan =
        |p: Value| settings.plan(&settings.validate(&patch(p)).unwrap()).unwrap();
    // The flag's 0.6 with no override, the checkpoint's repetition penalty
    // 1 written as 1.0, the default hold: nothing changes, no override.
    let p =
        plan(json!({"temperature": 0.6, "repetition_penalty": 1.0, "pin_hold": "1m"}));
    assert!(p.entries.is_empty() && !p.writes_anything());
    settings.publish(p);
    assert!(settings.overrides().is_empty());
    // An override, then the flag's value: an override with that value.
    settings.publish(plan(json!({"temperature": 0.5})));
    let p = plan(json!({"temperature": "0.6"}));
    let entries: Vec<_> =
        p.entries.iter().map(|e| (e.old.clone(), e.new.clone())).collect();
    assert_eq!(
        entries,
        vec![(Some(SettingValue::Number(0.5)), Some(SettingValue::Number(0.6)))]
    );
    settings.publish(p);
    assert_eq!(
        settings.overrides().get("temperature"),
        Some(&SettingValue::Number(0.6))
    );
    // Integers and numbers compare as numbers.
    assert!(same_value(&Some(SettingValue::Int(1)), &Some(SettingValue::Number(1.0))));
    assert!(!same_value(&Some(SettingValue::Int(1)), &Some(SettingValue::Number(1.5))));
    assert!(!same_value(
        &Some(SettingValue::Text("1".into())),
        &Some(SettingValue::Int(1))
    ));
}

#[test]
fn a_reset_deletes_the_row_even_without_an_override_in_effect() {
    let settings = runtime();
    let reset = Change { set: Vec::new(), reset: vec!["top_p".into()] };
    let plan = settings.plan(&reset).unwrap();
    assert!(plan.entries.is_empty(), "nothing in effect changes");
    assert_eq!(plan.reset_rows(), ["top_p".to_owned()]);
    assert!(plan.writes_anything());
}

#[test]
fn the_settings_settle_once_and_gate_the_disk_tiers_expiry() {
    let settings = runtime();
    assert!(!settings.settled());
    assert_eq!(
        settings.provisional(),
        " (the flag's value; database overrides may apply)"
    );
    let gate = settings.expiry_gate();
    assert!(!gate.is_open(), "30 s have not passed");
    settings.settle();
    assert!(settings.settled() && gate.is_open());
    assert_eq!(settings.provisional(), "");
    // Without a settle, the bound opens it: a gate made at a deadline past.
    let late = crate::serve::disk::ExpiryGate::new(
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        std::time::Instant::now(),
    );
    assert!(late.is_open());
}
