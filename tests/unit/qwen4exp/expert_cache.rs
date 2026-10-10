//! Host-side tests of the expert cache's request queue: cancellation of the
//! requests a never-committed pass enqueued (`QueuedRequests`); of the
//! read pool's dispatch (`run_claimed`); and of the GPU warmer (`Warmer`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::{
    QueuedRequests, REGIONS, Request, Service, WARM_OFF, WARM_ON, WARM_STARTING, Warmer,
};

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
        released: AtomicU64::new(0),
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

/// `read_experts` lays each expert's nine regions out back to back, the
/// bytes `ExpertStore::read_into` reads, with and without `F_NOCACHE`, on the
/// four-layer checkpoint (`LILY_MODEL_DIR_FLASH`).
#[test]
fn read_experts_match_the_store() {
    use crate::qwen4exp::config::Qwen4ExpConfig;
    use crate::qwen4exp::expert_store::ExpertStore;
    use crate::safetensors::Checkpoint;

    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else {
        eprintln!("LILY_MODEL_DIR_FLASH unset; skipping read_experts_match_the_store");
        return;
    };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let config = Qwen4ExpConfig::from_model_dir(&dir).expect("config");
    let store = ExpertStore::open(&ckpt, &config).expect("store");
    let pool = super::read_pool().expect("pool");
    let experts = [(1usize, 7usize), (0, config.num_experts - 1), (2, 300)];
    let bytes = store.slice_bytes();
    for nocache in [true, false] {
        store.set_nocache(nocache).expect("F_NOCACHE");
        let mut dst = vec![vec![0u8; bytes]; experts.len() + 1];
        super::read_experts(&pool, &store, &experts, &mut dst).expect("read_experts");
        for (&(layer, expert), got) in experts.iter().zip(&dst) {
            let mut regions: Vec<Vec<u8>> =
                store.region_lens().iter().map(|&n| vec![0u8; n]).collect();
            let want: [&mut [u8]; super::REGIONS] = regions
                .iter_mut()
                .map(Vec::as_mut_slice)
                .collect::<Vec<_>>()
                .try_into()
                .expect("nine buffers");
            store.read_into(layer, expert, want).expect("read_into");
            assert!(*got == regions.concat(), "layer {layer} expert {expert}");
        }
        assert!(dst[experts.len()].iter().all(|&b| b == 0), "spare buffer written");
    }
    let mut short = vec![vec![0u8; bytes - 1]];
    assert!(super::read_experts(&pool, &store, &experts[..1], &mut short).is_err());
}

/// Every (expert, region) pair of a resolution is claimed exactly once; up
/// to `BIG_FIRST_MAX` experts, the code regions of all of them before any
/// scales or biases, and beyond it each expert's regions together.
#[test]
fn claim_order_covers_every_read_once_large_regions_first() {
    for n in [9usize, 20] {
        let order: Vec<(usize, usize)> =
            (0..n * REGIONS).map(|k| super::claimed_read(k, n)).collect();
        let expected: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..REGIONS).map(move |r| (i, r))).collect();
        assert_eq!(order, expected, "n {n}");
    }
    for n in [1usize, 2, 5, 8] {
        let order: Vec<(usize, usize)> =
            (0..n * REGIONS).map(|k| super::claimed_read(k, n)).collect();
        let mut seen = order.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n * REGIONS, "n {n}");
        assert!(seen.iter().all(|&(i, r)| i < n && r < REGIONS));
        let codes = [0usize, 3, 6];
        assert!(
            order[..3 * n].iter().all(|(_, r)| codes.contains(r)),
            "n {n}: {order:?}"
        );
    }
}

/// Waits up to 10 s for the warm thread to leave `from`; returns the state.
fn warm_state_after(warmer: &Warmer, from: u8) -> u8 {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let state = warmer.shared.state.load(Ordering::Acquire);
        if state != from || std::time::Instant::now() > until {
            return state;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// A warm pass that fails, or panics, turns warming off; the thread stays
/// parked, and the service thread's releases and the drop still write the
/// flag the Warmer owns.
#[test]
fn a_failed_warm_pass_turns_warming_off_and_releases_stay_safe() {
    fn fails() -> anyhow::Result<()> {
        anyhow::bail!("injected for the test")
    }
    fn panics() -> anyhow::Result<()> {
        panic!("injected for the test")
    }
    for inject in [fails as fn() -> anyhow::Result<()>, panics] {
        let mut warmer = Warmer::spawn_with(Some(inject)).expect("spawn");
        assert_eq!(warm_state_after(&warmer, WARM_STARTING), WARM_ON);
        warmer.start();
        assert_eq!(warmer.generation, 1);
        assert_eq!(warm_state_after(&warmer, WARM_ON), WARM_OFF);
        warmer.release();
        assert_eq!(warmer.shared.flag.flag().load(Ordering::Acquire), 1);
        // Off: no more passes are asked for.
        warmer.start();
        assert_eq!(warmer.generation, 1);
        warmer.release();
        assert!(warmer.handle.as_ref().is_some_and(|h| !h.is_finished()));
        drop(warmer);
    }
}

/// Warm passes run while asked for and end at the release; the thread
/// stops at the drop.
#[test]
fn warm_passes_end_at_their_release() {
    let mut warmer = Warmer::spawn().expect("spawn");
    // Asked for before the thread is ready: nothing is.
    warmer.start();
    assert!(warmer.generation <= 1);
    assert_eq!(warm_state_after(&warmer, WARM_STARTING), WARM_ON);
    for _ in 0..20 {
        warmer.start();
        std::thread::sleep(std::time::Duration::from_micros(300));
        warmer.release();
    }
    assert_eq!(warmer.shared.state.load(Ordering::Acquire), WARM_ON);
    drop(warmer);
}

/// A warm pass nobody releases ends at `Warmer::CAP` polls; prints how long
/// that takes on the GPU.
#[test]
fn a_warm_pass_nobody_releases_ends_at_its_cap() {
    use crate::metal::{Grid, MetalContext, MslVersion};
    use crate::tensor::{DType, Tensor};
    let ctx = MetalContext::new_with_profile(false).expect("metal context");
    let kernel = ctx
        .pipeline("expert_cache_warm", super::WARM_SRC, MslVersion::V3_1)
        .expect("kernel");
    let flag = Tensor::zeros(&ctx, &[1], DType::U32).expect("flag");
    let mut times = Vec::new();
    for _ in 0..10 {
        let pass = ctx.begin().expect("pass");
        pass.dispatch_at(
            &kernel,
            &[flag.binding()],
            &[&1u32.to_ne_bytes(), &Warmer::CAP.to_ne_bytes()],
            Grid::Threads { grid: (32, 1, 1), threadgroup: (32, 1, 1) },
        )
        .expect("dispatch");
        let t = pass.commit().expect("commit").wait_retain().expect("wait");
        let t = t.timing().expect("timing");
        times.push((t.gpu_end_secs - t.gpu_start_secs) * 1e3);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "a warm pass of {} polls: {:.3} to {:.3} ms on the GPU (median {:.3})",
        Warmer::CAP,
        times[0],
        times[times.len() - 1],
        times[times.len() / 2]
    );
    assert!(times[times.len() / 2] < 20.0, "{times:?}");
}
