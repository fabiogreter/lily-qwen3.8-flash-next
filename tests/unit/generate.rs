use super::*;

#[test]
fn stop_token_at_the_generation_limit_still_counts_as_stopped() {
    assert!(ends_with_stop_token(&[10, 99], &[99, 100]));
    assert!(!ends_with_stop_token(&[10, 98], &[99, 100]));
    assert!(!ends_with_stop_token(&[], &[99, 100]));
}

/// What `on_token` saw, and a callback that asks to stop at `stop_at`.
fn recorder(
    stop_at: Option<u32>,
) -> (std::rc::Rc<RefCell<Vec<u32>>>, impl FnMut(u32) -> Result<bool>) {
    let seen = std::rc::Rc::new(RefCell::new(Vec::new()));
    let log = seen.clone();
    (seen, move |t| {
        log.borrow_mut().push(t);
        Ok(Some(t) != stop_at)
    })
}

#[test]
fn an_inserted_group_is_emitted_whole_even_when_the_callback_stops() {
    // A close in front of a draw: the close, then the draw.
    let (seen, mut cb) = recorder(None);
    let mut tokens = vec![1, 2];
    let end =
        emit_acted(Action::Before(vec![7, 8]), 9, &mut tokens, 100, &mut cb).unwrap();
    assert_eq!(end, None);
    assert_eq!(tokens, [1, 2, 7, 8, 9]);
    assert_eq!(*seen.borrow(), [7, 8, 9]);

    // A close after a draw, the callback stopping at the draw (a stop
    // string, a departed client, a preemption): the close still follows,
    // and the generation ends after it.
    let (seen, mut cb) = recorder(Some(9));
    let mut tokens = vec![1];
    let end =
        emit_acted(Action::After(vec![7, 8]), 9, &mut tokens, 100, &mut cb).unwrap();
    assert_eq!(end, Some(FinishReason::Callback));
    assert_eq!(tokens, [1, 9, 7, 8]);
    assert_eq!(*seen.borrow(), [9, 7, 8]);

    // Stopped inside the group: still whole.
    let (_, mut cb) = recorder(Some(7));
    let mut tokens = vec![];
    let end = push_group(&mut tokens, &[7, 8, 9], 100, &mut cb).unwrap();
    assert_eq!(
        (end, tokens.as_slice()),
        (Some(FinishReason::Callback), &[7, 8, 9][..])
    );

    assert!(emit_acted(Action::Keep, 9, &mut vec![], 100, &mut cb).is_err());
}

#[test]
fn only_the_length_cuts_an_inserted_group_short() {
    let (seen, mut cb) = recorder(None);
    let mut tokens = vec![1, 2];
    let end =
        emit_acted(Action::After(vec![7, 8, 9]), 5, &mut tokens, 4, &mut cb).unwrap();
    assert_eq!(end, Some(FinishReason::Length));
    assert_eq!(tokens, [1, 2, 5, 7]);
    assert_eq!(*seen.borrow(), [5, 7]);
    // Exactly at the length: the group fits and the generation ends.
    let (_, mut cb) = recorder(None);
    let mut tokens = vec![1];
    let end = push_group(&mut tokens, &[7, 8], 3, &mut cb).unwrap();
    assert_eq!((end, tokens.as_slice()), (Some(FinishReason::Length), &[1, 7, 8][..]));
    // A close in front of a draw that does not fit leaves the draw out.
    let (_, mut cb) = recorder(None);
    let mut tokens = vec![1];
    let end =
        emit_acted(Action::Before(vec![7, 8]), 9, &mut tokens, 3, &mut cb).unwrap();
    assert_eq!((end, tokens.as_slice()), (Some(FinishReason::Length), &[1, 7, 8][..]));
}
