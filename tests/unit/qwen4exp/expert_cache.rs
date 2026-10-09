//! Host-side tests of the expert cache's request queue: cancellation of the
//! requests a never-committed pass enqueued (`QueuedRequests`); and of the
//! read pool's dispatch (`run_claimed`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Condvar, Mutex};

use super::{QueuedRequests, Request, Service};

fn service() -> Arc<Service> {
    Arc::new(Service {
        requests: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
        cancelled: Mutex::new(Vec::new()),
        stop: AtomicBool::new(false),
        lookups: AtomicU64::new(0),
        misses: AtomicU64::new(0),
        decode_lookups: AtomicU64::new(0),
        decode_misses: AtomicU64::new(0),
        promotions: AtomicU64::new(0),
        stale: AtomicU64::new(0),
        skipped: AtomicU64::new(0),
    })
}

fn enqueue(service: &Service, seqs: impl IntoIterator<Item = u64>) {
    let mut queue = service.requests.lock().unwrap();
    for seq in seqs {
        queue.push_back(Request { seq, layer: 0, ids: 0, count: 10 });
    }
}

fn queued(service: &Service) -> Vec<u64> {
    service.requests.lock().unwrap().iter().map(|r| r.seq).collect()
}

#[test]
fn cancel_drops_queued_requests_and_marks_the_range() {
    let service = service();
    // Three variants of 4 layers each (seqs 1..=12), the second one committed.
    enqueue(&service, 1..=12);
    service.cancel(1, 4);
    service.cancel(9, 12);
    assert_eq!(queued(&service), vec![5, 6, 7, 8]);
    // The thread may already hold a request of a cancelled range.
    assert!(service.is_cancelled(1));
    assert!(service.is_cancelled(4));
    assert!(!service.is_cancelled(5));
    assert!(service.is_cancelled(12));
    assert!(!service.is_cancelled(13));
}

#[test]
fn ranges_below_the_served_sequence_are_pruned() {
    let service = service();
    service.cancel(1, 4);
    service.cancel(9, 12);
    assert_eq!(service.cancelled.lock().unwrap().len(), 2);
    // Serving seq 5 prunes the range below it, not the one ahead.
    assert!(!service.is_cancelled(5));
    assert_eq!(*service.cancelled.lock().unwrap(), vec![(9, 12)]);
    assert!(service.is_cancelled(10));
    // Served past it: pruned too.
    assert!(!service.is_cancelled(13));
    assert!(service.cancelled.lock().unwrap().is_empty());
}

#[test]
fn dropping_the_handle_cancels_and_keeping_it_does_not() {
    let service = service();
    enqueue(&service, 1..=8);
    let dropped =
        QueuedRequests { service: service.clone(), from: 1, to: 4, armed: true };
    let kept = QueuedRequests { service: service.clone(), from: 5, to: 8, armed: true };
    drop(dropped);
    kept.keep();
    assert_eq!(queued(&service), vec![5, 6, 7, 8]);
    assert!(service.is_cancelled(2));
    assert!(!service.is_cancelled(6));
}

#[test]
fn an_empty_range_is_never_cancelled() {
    let service = service();
    enqueue(&service, 1..=3);
    // A pass without cached layers enqueues nothing: `queued_since` would
    // return `None`; a handle with `armed == false` cancels nothing either.
    let handle =
        QueuedRequests { service: service.clone(), from: 4, to: 3, armed: false };
    drop(handle);
    assert_eq!(queued(&service), vec![1, 2, 3]);
    assert!(service.cancelled.lock().unwrap().is_empty());
}

/// The read pool's dispatch runs every task once across the caller and the
/// pool, batch after batch, and fails a batch on an error or a panic without
/// losing the other tasks or the pool.
#[test]
fn run_claimed_runs_every_task_once_and_reports_failures() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().expect("pool");
    for tasks in [0, 1, 2, 9, 37, 144] {
        let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
        let threads = Mutex::new(std::collections::HashSet::new());
        super::run_claimed(&pool, tasks, |k| {
            hits[k].fetch_add(1, Ordering::Relaxed);
            threads.lock().expect("set").insert(std::thread::current().id());
            std::thread::sleep(std::time::Duration::from_micros(200));
            Ok(())
        })
        .expect("batch");
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{tasks} tasks");
        if tasks >= 37 {
            assert!(
                threads.lock().expect("set").len() > 1,
                "{tasks} tasks on one thread"
            );
        }
    }
    let err = super::run_claimed(&pool, 20, |k| {
        if k == 7 { Err(anyhow::anyhow!("region {k} unreadable")) } else { Ok(()) }
    })
    .expect_err("task 7 fails");
    assert!(err.to_string().contains("region 7"), "{err}");
    let err =
        super::run_claimed(
            &pool,
            20,
            |k| if k == 3 { panic!("bad read") } else { Ok(()) },
        )
        .expect_err("task 3 panics");
    assert!(err.to_string().contains("panicked"), "{err}");
    let after = AtomicUsize::new(0);
    super::run_claimed(&pool, 10, |_| {
        after.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })
    .expect("the pool still runs batches");
    assert_eq!(after.load(Ordering::Relaxed), 10);
}
