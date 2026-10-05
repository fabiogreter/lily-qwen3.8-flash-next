use super::{BatchStats, Next, Slots, admits, next_action};

#[test]
fn slots_hand_out_the_lowest_free_one_and_take_them_back() {
    let mut slots = Slots::new(3);
    assert_eq!((slots.take(), slots.take()), (Some(0), Some(1)));
    slots.give(0);
    assert_eq!(slots.in_use(), 1);
    // The freed slot comes back first, then the untouched one.
    assert_eq!((slots.take(), slots.take()), (Some(0), Some(2)));
    assert_eq!(slots.take(), None, "three slots, three rows");
    assert_eq!(slots.in_use(), 3);
    // Out of range is ignored rather than growing the table.
    slots.give(7);
    assert_eq!(slots.in_use(), 3);
}

#[test]
fn a_request_starts_alone_whatever_its_size() {
    // Nothing running: admitted even when it alone exceeds the budget, as
    // without batching (the store then evicts everything else).
    assert!(admits(0, 2, 0, 50, 10));
}

#[test]
fn a_second_request_needs_a_free_row_and_room_beside_the_one_in_flight() {
    assert!(admits(1, 2, 4, 6, 10), "4 + 6 fits a budget of 10");
    assert!(!admits(1, 2, 4, 7, 10), "4 + 7 does not");
    assert!(!admits(2, 2, 0, 0, 10), "both rows taken");
    assert!(admits(2, 3, 3, 3, 10));
    // No overflow on absurd estimates.
    assert!(!admits(1, 2, usize::MAX, 1, usize::MAX - 1));
}

#[test]
fn one_row_decodes_alone_and_is_preempted_only_for_a_job_that_could_start() {
    assert_eq!(next_action(0, false), Next::Done);
    assert_eq!(next_action(1, false), Next::Solo { preempt: true });
    // A job held back for the budget cannot start until this row is done:
    // stopping the row for it would only spin.
    assert_eq!(next_action(1, true), Next::Solo { preempt: false });
    assert_eq!(next_action(2, false), Next::Step);
    assert_eq!(next_action(4, true), Next::Step);
}

#[test]
fn sharing_statistics_count_steps_rows_and_distinct_peers() {
    let mut stats = BatchStats::default();
    assert_eq!(stats.mean_rows(), None, "no batched step yet");
    stats.record_step(1, &[1, 2]);
    stats.record_step(1, &[1, 2, 3]);
    stats.record_step(1, &[2, 1]);
    assert_eq!(stats.batched_tokens, 3);
    assert_eq!(stats.rows_sum, 7);
    assert_eq!(stats.max_rows, 3);
    assert_eq!(stats.peers, vec![2, 3], "each peer once, never itself");
    assert_eq!(stats.mean_rows(), Some(7.0 / 3.0));
}

#[test]
fn the_log_group_says_what_was_shared_and_nothing_for_a_lone_request() {
    assert_eq!(BatchStats::default().describe(40), "");
    let mut stats =
        BatchStats { solo_tokens: 10, preemptions: 1, ..Default::default() };
    stats.record_step(1, &[1, 2]);
    stats.record_step(1, &[1, 2]);
    stats.interleaved_steps = 8;
    stats.interleaved_secs = 0.5;
    assert_eq!(
        stats.describe(12),
        ", batched: 2/12 tokens in steps of up to 2 rows (mean 2.00) shared with 1 other request, \
         lone decode stopped 1 time for a new request, prefill shared with 8 decode steps (0.50s)"
    );
}
