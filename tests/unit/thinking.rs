use super::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const THINK_END: u32 = 1;
const TOOL_CALL: u32 = 2;
const TOOL_CALL_END: u32 = 3;
/// Ids from here are one character each (`CHAR + c`): the toy encoding.
const CHAR: u32 = 10_000;

fn encode(text: &str) -> Result<Vec<u32>> {
    Ok(text.chars().map(|c| CHAR + c as u32).collect())
}

/// The toy decoding of inserted ids, the markers spelled out.
fn text(ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| match id {
            THINK_END => "</think>".to_owned(),
            TOOL_CALL => "<tool_call>".to_owned(),
            TOOL_CALL_END => "</tool_call>".to_owned(),
            id => char::from_u32(id - CHAR).expect("a toy char id").to_string(),
        })
        .collect()
}

fn texts() -> ThinkingTexts {
    ThinkingTexts {
        close: vec!["Act now.".into(), "Go.".into()],
        nudges: vec![
            NudgeLevel { at: 0.5, texts: vec!["Half.".into(), "Halfway.".into()] },
            NudgeLevel { at: 0.75, texts: vec!["Three quarters.".into()] },
        ],
    }
}

fn tokens(texts: &ThinkingTexts) -> Arc<ThinkingTokens> {
    Arc::new(
        ThinkingTokens::new(THINK_END, TOOL_CALL, TOOL_CALL_END, texts, encode)
            .expect("toy tokens"),
    )
}

fn control(settings: ThinkingSettings) -> ThinkingControl {
    ThinkingControl::new(tokens(&texts()), settings, true)
}

/// An ordinary model token with this text (ids below CHAR stand for words).
fn word(c: &mut ThinkingControl, text: &str) -> Action {
    c.decide(100, text)
}

fn budget(n: usize, grace: usize) -> ThinkingSettings {
    ThinkingSettings { budget: Some(n), grace, ..ThinkingSettings::default() }
}

#[test]
fn nothing_happens_without_an_open_block_or_after_it_closed() {
    let settings = ThinkingSettings { tool_call_ends_thinking: true, ..budget(2, 0) };
    let mut closed = ThinkingControl::new(tokens(&texts()), settings, false);
    assert!(!closed.may_act_next());
    for _ in 0..10 {
        assert_eq!(word(&mut closed, "x\n"), Action::Keep);
    }
    assert_eq!(closed.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(closed.closed(), None);

    let mut c = control(settings);
    assert_eq!(c.decide(THINK_END, "</think>"), Action::Keep, "the model closed it");
    assert!(!c.is_open() && !c.may_act_next());
    for _ in 0..10 {
        assert_eq!(word(&mut c, "x\n"), Action::Keep);
    }
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(c.closed(), None);
}

#[test]
fn a_tool_call_at_a_line_start_ends_thinking_with_only_the_close_tag() {
    let on = ThinkingSettings { tool_call_ends_thinking: true, ..Default::default() };
    // Right after the generation prompt's `<think>\n`.
    let mut c = control(on);
    assert!(c.may_act_next());
    let Action::Before(ids) = c.decide(TOOL_CALL, "<tool_call>") else {
        panic!("no close before the tool call");
    };
    assert_eq!(text(&ids), "</think>\n\n", "no preface in front of a tool call");
    assert!(!c.is_open());
    assert_eq!(c.closed(), Some(Closed { by: ClosedBy::ToolCall, thinking_tokens: 0 }));

    // After some reasoning that ends a line.
    let mut c = control(on);
    assert_eq!(word(&mut c, "Let me look"), Action::Keep);
    assert!(!c.may_act_next(), "mid-line: nothing to foresee");
    assert_eq!(word(&mut c, ".\n"), Action::Keep);
    assert!(c.may_act_next());
    assert!(matches!(c.decide(TOOL_CALL, "<tool_call>"), Action::Before(_)));
    assert_eq!(c.closed().map(|c| c.thinking_tokens), Some(2));

    // Mid-line it is reasoning text: kept, the block stays open.
    let mut c = control(on);
    assert_eq!(word(&mut c, "I would call "), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert!(c.is_open());

    // With the rule off it is kept too.
    let mut c = control(ThinkingSettings::default());
    assert!(!c.may_act_next());
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert!(c.is_open());
}

#[test]
fn the_budget_closes_at_the_first_line_end_from_the_budget_on() {
    let mut c = control(budget(5, 4));
    for t in ["a", "b", "c"] {
        assert_eq!(word(&mut c, t), Action::Keep);
        assert!(!c.may_act_next() || c.thinking_tokens() + 1 >= 5);
    }
    assert_eq!(word(&mut c, "d"), Action::Keep);
    assert!(c.may_act_next(), "the next token reaches the budget");
    assert_eq!(word(&mut c, "e"), Action::Keep, "the budget, mid-line");
    assert_eq!(word(&mut c, " f"), Action::Keep);
    let Action::After(ids) = word(&mut c, "g\n") else {
        panic!("no close at the line end")
    };
    assert_eq!(text(&ids), "\nAct now.\n</think>\n\n");
    assert_eq!(c.closed(), Some(Closed { by: ClosedBy::Budget, thinking_tokens: 7 }));
    assert!(!c.is_open() && !c.may_act_next());
}

#[test]
fn the_close_puts_its_preface_on_a_paragraph_of_its_own() {
    let close = |last: &str| {
        let mut c = control(budget(1, 0));
        let Action::After(ids) = word(&mut c, last) else { panic!("no close") };
        text(&ids)
    };
    assert_eq!(close("x\n\n"), "Act now.\n</think>\n\n");
    assert_eq!(close("x\n"), "\nAct now.\n</think>\n\n");
    assert_eq!(close("x"), "\n\nAct now.\n</think>\n\n");
}

#[test]
fn grace_waits_for_a_line_then_a_sentence_then_forces() {
    // Grace 0: right at the budget, wherever it is.
    let mut c = control(budget(2, 0));
    assert_eq!(word(&mut c, "a"), Action::Keep);
    assert!(matches!(word(&mut c, "b"), Action::After(_)));

    // A sentence end does not do in the first window...
    let mut c = control(budget(2, 3));
    assert_eq!(word(&mut c, "a"), Action::Keep);
    assert_eq!(word(&mut c, "b."), Action::Keep, "over 0");
    assert_eq!(word(&mut c, " c."), Action::Keep, "over 1");
    assert_eq!(word(&mut c, " d."), Action::Keep, "over 2");
    // ...but does in the second.
    let Action::After(ids) = word(&mut c, " e.") else { panic!("no close at over 3") };
    assert!(text(&ids).starts_with("\n\n"), "mid-line: a paragraph break first");

    // Neither: forced once both windows are over.
    let mut c = control(budget(1, 2));
    for over in 0..4 {
        assert_eq!(word(&mut c, "w"), Action::Keep, "over {over}");
    }
    assert!(matches!(word(&mut c, "w"), Action::After(_)), "over 4 = 2 * grace");
}

#[test]
fn nothing_lands_inside_a_code_fence_or_a_tool_call() {
    // The budget is reached inside a fence: its lines do not count.
    let mut c = control(budget(3, 10));
    assert_eq!(word(&mut c, "```rust\n"), Action::Keep);
    for _ in 0..5 {
        assert_eq!(word(&mut c, "let x = 1;\n"), Action::Keep);
    }
    assert_eq!(word(&mut c, "```"), Action::Keep, "the fence closes mid-line");
    assert!(matches!(word(&mut c, "\n"), Action::After(_)), "its line ends");

    // An indented fence opened by a token split across the backticks.
    let mut c = control(budget(1, 10));
    assert_eq!(word(&mut c, "  ``"), Action::Keep);
    assert_eq!(word(&mut c, "`\n"), Action::Keep, "inside the fence now");
    assert_eq!(word(&mut c, "x\n"), Action::Keep);

    // A fence that outlasts both windows: the close lands there anyway.
    let mut c = control(budget(1, 2));
    assert_eq!(word(&mut c, "```\n"), Action::Keep);
    for _ in 0..3 {
        assert_eq!(word(&mut c, "code\n"), Action::Keep);
    }
    assert!(matches!(word(&mut c, "code\n"), Action::After(_)));

    // A tool call kept in the block (the rule off) holds the close back
    // until it ends, however long; past both windows it comes right after.
    let mut c = control(budget(2, 1));
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    for _ in 0..20 {
        assert_eq!(word(&mut c, "<parameter=x>\n"), Action::Keep);
    }
    assert!(matches!(c.decide(TOOL_CALL_END, "</tool_call>"), Action::After(_)));
    // Within the windows it waits for the line end after the call.
    let mut c = control(budget(2, 10));
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(word(&mut c, "<function=f>\n"), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL_END, "</tool_call>"), Action::Keep);
    assert!(matches!(word(&mut c, "\n"), Action::After(_)));
}

#[test]
fn nudges_come_at_their_fractions_and_the_block_stays_open() {
    let settings = ThinkingSettings { nudges: true, ..budget(20, 4) };
    let mut c = control(settings);
    let mut inserted = Vec::new();
    for i in 1..=19 {
        // Lines of two tokens: every even count ends a line.
        let t = if i % 2 == 0 { "b\n" } else { "a" };
        match word(&mut c, t) {
            Action::Keep => {}
            Action::After(ids) => inserted.push((c.thinking_tokens(), text(&ids))),
            Action::Before(_) => panic!("a nudge goes after the token"),
        }
    }
    assert_eq!(
        inserted,
        vec![(10, "\nHalf.\n\n".to_owned()), (16, "\nThree quarters.\n\n".to_owned())],
        "at the first line end from 50 % (10) and 75 % (15) on"
    );
    assert!(c.is_open(), "a nudge does not close the block");
    assert_eq!(c.nudged(), 2);
    assert_eq!(c.closed(), None);
    // The budget's close still comes.
    assert!(matches!(word(&mut c, "end\n"), Action::After(_)));
    assert_eq!(c.closed().map(|c| c.by), Some(ClosedBy::Budget));

    // Without `nudges`, or without a budget, none.
    let mut c = control(budget(20, 4));
    for _ in 0..19 {
        assert_eq!(word(&mut c, "x\n"), Action::Keep);
    }
    let mut c = control(ThinkingSettings { nudges: true, ..Default::default() });
    for _ in 0..100 {
        assert_eq!(word(&mut c, "x\n"), Action::Keep);
    }
}

#[test]
fn nudge_variants_rotate_with_the_seed() {
    let first_nudge = |seed: u64| {
        let mut c = control(ThinkingSettings { nudges: true, seed, ..budget(4, 0) });
        assert_eq!(word(&mut c, "x"), Action::Keep);
        let Action::After(ids) = word(&mut c, "y") else { panic!("no nudge at 2") };
        text(&ids)
    };
    assert_eq!(first_nudge(0), "\n\nHalf.\n\n");
    assert_eq!(first_nudge(1), "\n\nHalfway.\n\n");
    assert_eq!(first_nudge(2), "\n\nHalf.\n\n");
    let close = |seed: u64| {
        let mut c = control(ThinkingSettings { seed, ..budget(1, 0) });
        let Action::After(ids) = word(&mut c, "x\n\n") else { panic!("no close") };
        text(&ids)
    };
    assert_eq!(close(0), "Act now.\n</think>\n\n");
    assert_eq!(close(1), "Go.\n</think>\n\n");
}

#[test]
fn a_nudge_that_cannot_leave_a_fence_is_dropped() {
    let mut c = control(ThinkingSettings { nudges: true, ..budget(40, 2) });
    assert_eq!(word(&mut c, "```\n"), Action::Keep);
    // Levels at 20 and 30, each waiting 2 * 2 tokens before it is dropped.
    for _ in 0..36 {
        assert_eq!(word(&mut c, "code\n"), Action::Keep);
    }
    assert_eq!(c.nudged(), 0, "both levels came and went inside the fence");
    assert_eq!(
        word(&mut c, "```\n"),
        Action::Keep,
        "out of the fence: nothing pending"
    );
    assert!(!c.may_act_next());
}

/// The loops rely on it: whenever the control acts on a token, it said
/// beforehand that it might, so no step was committed ahead of that token.
#[test]
fn the_control_never_acts_unannounced() {
    let pieces =
        ["a", "b.", " c", "\n", "\n\n", "x\n", "```", "``", "`\n", "  ", "end.", "?"];
    let mut rng = SmallRng::seed_from_u64(7);
    for trial in 0..400 {
        let settings = ThinkingSettings {
            budget: rng.gen_bool(0.8).then(|| rng.gen_range(1..60)),
            grace: rng.gen_range(0..6),
            nudges: rng.gen_bool(0.5),
            tool_call_ends_thinking: rng.gen_bool(0.5),
            seed: rng.r#gen(),
        };
        let mut c = control(settings);
        for step in 0..120 {
            let announced = c.may_act_next();
            let r = rng.gen_range(0..100);
            let action = if r < 4 {
                c.decide(TOOL_CALL, "<tool_call>")
            } else if r < 8 {
                c.decide(TOOL_CALL_END, "</tool_call>")
            } else if r < 9 {
                c.decide(THINK_END, "</think>")
            } else {
                word(&mut c, pieces[rng.gen_range(0..pieces.len())])
            };
            assert!(
                action == Action::Keep || announced,
                "trial {trial} step {step}: {action:?} unannounced under {settings:?}"
            );
        }
    }
}

#[test]
fn texts_validate_and_read_from_json() {
    let ok = ThinkingTokens::new(1, 2, 3, &ThinkingTexts::default(), encode);
    assert!(ok.is_ok(), "the built-in texts");
    let bad_order = ThinkingTexts {
        close: vec![],
        nudges: vec![
            NudgeLevel { at: 0.75, texts: vec!["a".into()] },
            NudgeLevel { at: 0.5, texts: vec!["b".into()] },
        ],
    };
    assert!(ThinkingTokens::new(1, 2, 3, &bad_order, encode).is_err());
    let at_one = ThinkingTexts {
        close: vec![],
        nudges: vec![NudgeLevel { at: 1.0, texts: vec!["a".into()] }],
    };
    assert!(ThinkingTokens::new(1, 2, 3, &at_one, encode).is_err());
    // A text that spells the close itself is refused.
    let spelled = ThinkingTexts { close: vec!["</think>".into()], nudges: vec![] };
    let marker_encode = |t: &str| -> Result<Vec<u32>> {
        Ok(if t.contains("</think>") { vec![THINK_END] } else { encode(t)? })
    };
    assert!(ThinkingTokens::new(1, 2, 3, &spelled, marker_encode).is_err());

    // No close text: just the line end the template writes before the tag.
    let bare = Arc::new(
        ThinkingTokens::new(
            1,
            2,
            3,
            &ThinkingTexts { close: vec![], nudges: vec![] },
            encode,
        )
        .expect("bare"),
    );
    let mut c = ThinkingControl::new(bare, budget(1, 0), true);
    let Action::After(ids) = word(&mut c, "x") else { panic!("no close") };
    assert_eq!(text(&ids), "\n</think>\n\n");

    let parsed: ThinkingTexts = serde_json::from_str(
        r#"{"close": ["Stop."], "nudges": [{"at": 0.6, "texts": ["Soon."]}]}"#,
    )
    .expect("json");
    assert_eq!(parsed.close, vec!["Stop.".to_owned()]);
    assert_eq!(parsed.nudges[0].at, 0.6);
    assert!(serde_json::from_str::<ThinkingTexts>(r#"{"closes": []}"#).is_err());
}

#[test]
fn only_a_prompt_ending_in_an_open_block_opens_thinking() {
    assert!(opens_thinking("<|im_start|>assistant\n<think>\n"));
    assert!(!opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    assert!(!opens_thinking("<|im_start|>assistant\n"));
}
