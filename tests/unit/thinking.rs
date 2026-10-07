use super::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const THINK_END: u32 = 1;
const TOOL_CALL: u32 = 2;
const TOOL_CALL_END: u32 = 3;
/// The toy end of turn: an added token, like the markers.
const IM_END: u32 = 4;
/// Ids from here are one character each (`CHAR + c`): the toy encoding.
const CHAR: u32 = 10_000;

/// The toy vocabulary's added tokens.
fn is_added(id: u32) -> bool {
    id <= IM_END
}

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
            IM_END => "<|im_end|>".to_owned(),
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
        ThinkingTokens::new(
            THINK_END,
            TOOL_CALL,
            TOOL_CALL_END,
            texts,
            encode,
            is_added,
        )
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
    assert_eq!(closed.decide_stop(), Action::Keep);
    assert_eq!(closed.closed(), None);

    let mut c = control(settings);
    assert_eq!(c.decide(THINK_END, "</think>"), Action::Keep, "the model closed it");
    assert!(!c.is_open() && !c.may_act_next());
    for _ in 0..10 {
        assert_eq!(word(&mut c, "x\n"), Action::Keep);
    }
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(c.decide_stop(), Action::Keep, "the model's own end of turn after it");
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
        // A stop token may come at any draw: the loops always rest.
        assert!(c.may_act_next());
    }
    assert_eq!(word(&mut c, "d"), Action::Keep);
    assert_eq!(word(&mut c, "e"), Action::Keep, "the budget, mid-line");
    assert_eq!(word(&mut c, " f"), Action::Keep);
    let Action::After(ids) = word(&mut c, "g\n") else {
        panic!("no close at the line end")
    };
    assert_eq!(text(&ids), "Act now.\n</think>\n\n", "on the next line");
    assert_eq!(c.closed(), Some(Closed { by: ClosedBy::Budget, thinking_tokens: 7 }));
    assert!(!c.is_open() && !c.may_act_next());
}

/// Each seam (see the module docs): at a line start the preface follows
/// directly, mid-line after a word a paragraph break comes first, after
/// punctuation a space and the break, and right after a space nothing
/// lands until the next token.
#[test]
fn the_close_puts_its_preface_on_a_line_of_its_own_at_a_clean_seam() {
    let close = |last: &str| {
        let mut c = control(budget(1, 0));
        let Action::After(ids) = word(&mut c, last) else { panic!("no close") };
        text(&ids)
    };
    assert_eq!(close("x\n\n"), "Act now.\n</think>\n\n");
    assert_eq!(close("x\n"), "Act now.\n</think>\n\n");
    assert_eq!(close("x"), "\n\nAct now.\n</think>\n\n");
    assert_eq!(close("7"), "\n\nAct now.\n</think>\n\n");
    assert_eq!(close("é"), "\n\nAct now.\n</think>\n\n");
    assert_eq!(close("x."), " \n\nAct now.\n</think>\n\n");
    assert_eq!(close("**"), " \n\nAct now.\n</think>\n\n");

    // Right after whitespace mid-line: the close waits for the next token,
    // even past both windows (grace 0).
    let mut c = control(budget(1, 0));
    assert_eq!(word(&mut c, "x "), Action::Keep);
    assert_eq!(word(&mut c, "\t"), Action::Keep);
    assert!(c.is_open() && c.may_act_next());
    let Action::After(ids) = word(&mut c, "y") else { panic!("no close after y") };
    assert_eq!(text(&ids), "\n\nAct now.\n</think>\n\n");
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
    assert!(text(&ids).starts_with(" \n\n"), "mid-line: a paragraph break first");

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

    // A fence that outlasts both windows: the close still waits for it to
    // end, then lands at once (both windows are over).
    let mut c = control(budget(1, 2));
    assert_eq!(word(&mut c, "```\n"), Action::Keep);
    for _ in 0..50 {
        assert_eq!(word(&mut c, "code\n"), Action::Keep);
    }
    assert!(c.is_open() && c.may_act_next());
    assert!(matches!(word(&mut c, "```"), Action::After(_)), "right after the fence");
    // Grace 0 does not make it land in a fence either.
    let mut c = control(budget(2, 0));
    assert_eq!(word(&mut c, "```\n"), Action::Keep);
    assert_eq!(word(&mut c, "x"), Action::Keep);
    assert_eq!(word(&mut c, "\n"), Action::Keep);

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
            Action::Before(_) | Action::Replace(_) => {
                panic!("a nudge goes after the token")
            }
        }
    }
    assert_eq!(
        inserted,
        vec![(10, "Half.\n\n".to_owned()), (16, "Three quarters.\n\n".to_owned())],
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
        let Action::After(ids) = word(&mut c, "y\n") else { panic!("no nudge at 2") };
        text(&ids)
    };
    assert_eq!(first_nudge(0), "Half.\n\n");
    assert_eq!(first_nudge(1), "Halfway.\n\n");
    assert_eq!(first_nudge(2), "Half.\n\n");
    // A request's `seed: -1` is u64::MAX: the pick wraps instead of
    // overflowing.
    assert_eq!(first_nudge(u64::MAX), "Halfway.\n\n");
    let close = |seed: u64| {
        let mut c = control(ThinkingSettings { seed, ..budget(1, 0) });
        let Action::After(ids) = word(&mut c, "x\n\n") else { panic!("no close") };
        text(&ids)
    };
    assert_eq!(close(0), "Act now.\n</think>\n\n");
    assert_eq!(close(1), "Go.\n</think>\n\n");
    assert_eq!(close(u64::MAX), "Go.\n</think>\n\n");
    // The second level's pick is seed + 1: u64::MAX wraps to 0.
    let mut c =
        control(ThinkingSettings { nudges: true, seed: u64::MAX, ..budget(4, 0) });
    assert_eq!(word(&mut c, "x"), Action::Keep);
    assert!(matches!(word(&mut c, "y\n"), Action::After(_)));
    let Action::After(ids) = word(&mut c, "z\n") else { panic!("no nudge at 3") };
    assert_eq!(text(&ids), "Three quarters.\n\n");
}

#[test]
fn a_nudge_lands_only_at_a_line_end_within_its_window() {
    // Levels at 10 and 15 of 20, grace 2 (a window of tokens 10..=12 for
    // the first). The first finds no line end there (sentence ends do not
    // count for a nudge) and is dropped; the second lands at the line end
    // on token 16.
    let mut c = control(ThinkingSettings { nudges: true, ..budget(20, 2) });
    for _ in 0..9 {
        assert_eq!(word(&mut c, "w"), Action::Keep);
    }
    for _ in 0..3 {
        assert_eq!(word(&mut c, " end."), Action::Keep);
    }
    assert_eq!(word(&mut c, "\n"), Action::Keep, "token 13: too late for level 1");
    assert_eq!(c.nudged(), 0);
    assert_eq!(word(&mut c, "a"), Action::Keep);
    assert_eq!(word(&mut c, "b"), Action::Keep, "15: level 2 due, mid-line");
    let Action::After(ids) = word(&mut c, "c\n") else { panic!("no nudge at 16") };
    assert_eq!(text(&ids), "Three quarters.\n\n");
    assert_eq!(c.nudged(), 1);
}

#[test]
fn a_tool_call_in_a_fence_or_inside_a_kept_call_does_not_end_thinking() {
    let on = ThinkingSettings { tool_call_ends_thinking: true, ..Default::default() };
    // An example the reasoning quotes in a fence.
    let mut c = control(on);
    assert_eq!(word(&mut c, "```xml\n"), Action::Keep);
    assert!(!c.may_act_next(), "a line start inside a fence");
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL_END, "</tool_call>"), Action::Keep);
    assert_eq!(word(&mut c, "\n```\n"), Action::Keep);
    assert!(c.is_open() && c.may_act_next(), "out of the fence");
    assert!(matches!(c.decide(TOOL_CALL, "<tool_call>"), Action::Before(_)));

    // A mention mid-line is text: it opens no call, so the next line's
    // real call ends the block.
    let mut c = control(on);
    assert_eq!(word(&mut c, "Like "), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert!(c.is_open());
    assert_eq!(word(&mut c, " does.\n"), Action::Keep);
    assert!(c.may_act_next(), "no call was opened");
    assert!(matches!(c.decide(TOOL_CALL, "<tool_call>"), Action::Before(_)));
}

/// The fences of `fenced` (the lines before a `<tool_call>` line, which
/// sits inside them), then `closing`: the call inside is text, the one
/// after the fence ends the block.
fn quoted_call_then_real_one(fenced: &[&str], closing: &str) {
    let on = ThinkingSettings { tool_call_ends_thinking: true, ..Default::default() };
    let mut c = control(on);
    for line in fenced {
        assert_eq!(word(&mut c, line), Action::Keep);
        assert!(!c.may_act_next(), "{fenced:?}: after {line:?}, inside the fence");
    }
    // Unmatched: no `</tool_call>` before the fence closes.
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep, "{fenced:?}");
    assert_eq!(word(&mut c, "\n"), Action::Keep);
    assert_eq!(word(&mut c, closing), Action::Keep);
    assert!(c.may_act_next(), "{fenced:?}: out of the fence after {closing:?}");
    assert!(
        matches!(c.decide(TOOL_CALL, "<tool_call>"), Action::Before(_)),
        "{fenced:?}: the real call"
    );
}

#[test]
fn fences_open_and_close_as_commonmark_has_them() {
    quoted_call_then_real_one(&["```\n"], "```\n");
    // Four backticks: a three-backtick line inside does not close it.
    quoted_call_then_real_one(&["````md\n", "```\n", "text\n"], "````\n");
    // Longer closes too, with trailing whitespace and up to three spaces.
    quoted_call_then_real_one(&["```\n"], "   `````  \n");
    // Tildes: backticks inside do not close them, nor fewer tildes.
    quoted_call_then_real_one(&["~~~ text\n", "```\n", "~~\n"], "~~~~\n");
    // A run with an info string after it does not close, nor one with a
    // no-break space after it (only spaces and tabs may follow a close).
    quoted_call_then_real_one(&["```\n", "```rust\n"], "```\n");
    quoted_call_then_real_one(&["~~~\n", "~~~\u{a0}\n"], "~~~\t\r\n");

    let on = ThinkingSettings { tool_call_ends_thinking: true, ..Default::default() };
    // Not fences: four spaces of indentation, two backticks, a backtick
    // fence whose info string has a backtick (inline code).
    for line in ["    ```\n", "``\n", "``` a`b\n", "\t```\n"] {
        let mut c = control(on);
        assert_eq!(word(&mut c, line), Action::Keep);
        assert!(
            matches!(c.decide(TOOL_CALL, "<tool_call>"), Action::Before(_)),
            "{line:?} opened a fence"
        );
    }

    // The budget's close lands right after a closing run, mid-line: what
    // it inserts ends the line, which closes the fence.
    let mut c = control(budget(1, 0));
    assert_eq!(word(&mut c, "~~~~\n"), Action::Keep);
    assert_eq!(word(&mut c, "~~~\n"), Action::Keep, "too short to close");
    assert_eq!(
        word(&mut c, "~~~~"),
        Action::After(text_ids(" \n\nAct now.\n</think>\n\n"))
    );
    // ...but not after a run that would open one.
    let mut c = control(budget(1, 0));
    assert_eq!(word(&mut c, "```"), Action::Keep);
    assert_eq!(word(&mut c, "py"), Action::Keep);
    assert_eq!(word(&mut c, "\n"), Action::Keep, "inside the fence now");
}

/// The ids of a toy text (the markers by their spelling).
fn text_ids(t: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut rest = t;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("</think>") {
            ids.push(THINK_END);
            rest = r;
        } else {
            let c = rest.chars().next().expect("non-empty");
            ids.push(CHAR + c as u32);
            rest = &rest[c.len_utf8()..];
        }
    }
    ids
}

/// A `<tool_call>` that starts a line outside a fence opens a call the
/// block kept (the rule off): nothing lands inside it until its end. One
/// mid-line, or in a fence, opens nothing, and never leaves the block
/// latched.
#[test]
fn only_a_call_at_a_line_start_outside_a_fence_holds_the_close_back() {
    // Mid-line: the close comes at the next line end as if it were not there.
    let mut c = control(budget(1, 4));
    assert_eq!(word(&mut c, "Use "), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert!(matches!(word(&mut c, " here.\n"), Action::After(_)));

    // In a fence, unmatched: the close comes once the fence ends.
    let mut c = control(budget(1, 4));
    assert_eq!(word(&mut c, "```\n"), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(word(&mut c, "\n"), Action::Keep);
    assert!(matches!(word(&mut c, "```\n"), Action::After(_)));

    // At a line start: held back until `</tool_call>`, fences inside the
    // arguments not tracked.
    let mut c = control(budget(1, 4));
    assert_eq!(c.decide(TOOL_CALL, "<tool_call>"), Action::Keep);
    assert_eq!(word(&mut c, "\n<parameter=x>\n```\n"), Action::Keep);
    assert_eq!(c.decide(TOOL_CALL_END, "</tool_call>"), Action::Keep);
    assert!(matches!(word(&mut c, "\n"), Action::After(_)), "after the call");
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
    assert_eq!(c.nudged(), 0);
}

/// The loops rely on it: whenever the control acts on a token, it said
/// beforehand that it might, so no step was committed ahead of that token.
#[test]
fn the_control_never_acts_unannounced() {
    let pieces = [
        "a", "b.", " c", "\n", "\n\n", "x\n", "```", "``", "`\n", "  ", "end.", "?",
        "~~~", "````\n", " ",
    ];
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
            let action = if r < 2 {
                c.decide_stop()
            } else if r < 4 {
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
    let ok = ThinkingTokens::new(1, 2, 3, &ThinkingTexts::default(), encode, is_added);
    assert!(ok.is_ok(), "the built-in texts");
    let bad_order = ThinkingTexts {
        close: vec![],
        nudges: vec![
            NudgeLevel { at: 0.75, texts: vec!["a".into()] },
            NudgeLevel { at: 0.5, texts: vec!["b".into()] },
        ],
    };
    assert!(ThinkingTokens::new(1, 2, 3, &bad_order, encode, is_added).is_err());
    let at_one = ThinkingTexts {
        close: vec![],
        nudges: vec![NudgeLevel { at: 1.0, texts: vec!["a".into()] }],
    };
    assert!(ThinkingTokens::new(1, 2, 3, &at_one, encode, is_added).is_err());
    // A text that spells the close itself is refused.
    let spelled = ThinkingTexts { close: vec!["</think>".into()], nudges: vec![] };
    let marker_encode = |t: &str| -> Result<Vec<u32>> {
        Ok(if t.contains("</think>") { vec![THINK_END] } else { encode(t)? })
    };
    assert!(ThinkingTokens::new(1, 2, 3, &spelled, marker_encode, is_added).is_err());
    // So is one that spells any other added or special token: `<|im_end|>`
    // would end the turn inside the inserted group.
    let im_end = |t: &str| -> Result<Vec<u32>> {
        Ok(if t.contains("<|im_end|>") { vec![IM_END] } else { encode(t)? })
    };
    let ending = ThinkingTexts {
        close: vec![],
        nudges: vec![NudgeLevel { at: 0.5, texts: vec!["Done<|im_end|>".into()] }],
    };
    let err = ThinkingTokens::new(1, 2, 3, &ending, im_end, is_added).unwrap_err();
    assert!(err.to_string().contains("added token"), "{err}");

    // No close text: just the line end the template writes before the tag.
    let bare = Arc::new(
        ThinkingTokens::new(
            1,
            2,
            3,
            &ThinkingTexts { close: vec![], nudges: vec![] },
            encode,
            is_added,
        )
        .expect("bare"),
    );
    let mut c = ThinkingControl::new(bare.clone(), budget(1, 0), true);
    let Action::After(ids) = word(&mut c, "x") else { panic!("no close") };
    assert_eq!(text(&ids), "\n</think>\n\n");
    let mut c = ThinkingControl::new(bare.clone(), budget(1, 0), true);
    let Action::After(ids) = word(&mut c, "x\n") else { panic!("no close") };
    assert_eq!(text(&ids), "</think>\n\n", "the line end is the model's");

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

/// A stop token inside the block with a budget: replaced by the close
/// alone, placed by the seam rules, once; without a budget, or after the
/// block closed, it ends the generation.
#[test]
fn an_end_of_turn_inside_the_block_is_replaced_once_with_a_budget() {
    let replaced = |last: &str| {
        let mut c = control(budget(1000, 4));
        assert_eq!(word(&mut c, last), Action::Keep);
        assert!(c.may_act_next(), "a stop token may come at any draw");
        let Action::Replace(ids) = c.decide_stop() else { panic!("not replaced") };
        assert!(!c.is_open());
        assert_eq!(
            c.closed(),
            Some(Closed { by: ClosedBy::EndOfTurn, thinking_tokens: 1 })
        );
        assert_eq!(c.decide_stop(), Action::Keep, "only once");
        assert!(!c.may_act_next());
        text(&ids)
    };
    assert_eq!(replaced("Let me write the code now.\n"), "</think>\n\n");
    assert_eq!(replaced("Let me write the code now."), " \n</think>\n\n");
    assert_eq!(replaced("Let me write the code"), "\n</think>\n\n");
    // No clean seam after a space: the close cannot wait, so it takes the
    // break.
    assert_eq!(replaced("Let me write "), "\n</think>\n\n");
    // Inside a fence or a kept call it comes all the same.
    assert_eq!(replaced("```\n"), "</think>\n\n");

    // Right at the start of the block.
    let mut c = control(budget(10, 0));
    assert_eq!(c.decide_stop(), Action::Replace(text_ids("</think>\n\n")));
    assert_eq!(c.closed().map(|c| c.thinking_tokens), Some(0));

    // Without a budget (only the tool call rule) it is kept.
    let on = ThinkingSettings { tool_call_ends_thinking: true, ..Default::default() };
    let mut c = control(on);
    assert_eq!(word(&mut c, "done"), Action::Keep);
    assert_eq!(c.decide_stop(), Action::Keep);
    assert!(!c.may_act_next(), "mid-line, no budget: nothing to foresee");

    // After the budget's close it ends the turn as usual.
    let mut c = control(budget(1, 0));
    assert!(matches!(word(&mut c, "x\n"), Action::After(_)));
    assert_eq!(c.decide_stop(), Action::Keep);
    assert_eq!(c.closed().map(|c| c.by), Some(ClosedBy::Budget));
}

/// The model's own `</think>` closes the block wherever it comes: at the
/// budget's own token, while the close waits for a line end, or while a
/// nudge waits; nothing is inserted after it.
#[test]
fn the_models_own_close_at_a_threshold_or_during_a_wait_wins() {
    // Exactly at the budget.
    let mut c = control(budget(3, 4));
    assert_eq!(word(&mut c, "a"), Action::Keep);
    assert_eq!(word(&mut c, "b"), Action::Keep);
    assert!(c.may_act_next());
    assert_eq!(c.decide(THINK_END, "</think>"), Action::Keep);
    assert!(!c.is_open());
    assert_eq!((c.closed(), c.thinking_tokens()), (None, 2));
    assert_eq!(word(&mut c, "\n"), Action::Keep);

    // While the close waits for a line end.
    let mut c = control(budget(1, 4));
    assert_eq!(word(&mut c, "a"), Action::Keep, "at the budget, mid-line");
    assert_eq!(c.decide(THINK_END, "</think>"), Action::Keep);
    assert_eq!(word(&mut c, "\n\n"), Action::Keep);
    assert_eq!(c.closed(), None);

    // At a nudge's threshold, and while it waits.
    for waited in [0, 2] {
        let mut c = control(ThinkingSettings { nudges: true, ..budget(20, 4) });
        for _ in 0..9 + waited {
            assert_eq!(word(&mut c, "w"), Action::Keep);
        }
        assert_eq!(c.decide(THINK_END, "</think>"), Action::Keep);
        assert_eq!(word(&mut c, "\n"), Action::Keep, "no nudge after the block");
        assert_eq!((c.nudged(), c.closed()), (0, None));
    }
}

#[test]
fn a_close_cut_short_is_not_reported_as_emitted() {
    let mut c = control(budget(1, 0));
    assert!(!c.close_emitted(&[]), "no close asked for");
    let Action::After(ids) = word(&mut c, "x\n") else { panic!("no close") };
    let mut generated = vec![100];
    generated.extend(&ids);
    assert!(c.close_emitted(&generated));
    // `max_tokens` cut the group before its `</think>`.
    let cut = ids.iter().position(|&id| id == THINK_END).expect("a </think>");
    generated.truncate(1 + cut);
    assert!(!c.close_emitted(&generated));
    assert!(!c.close_emitted(&[100]), "no room for any of it");
}
