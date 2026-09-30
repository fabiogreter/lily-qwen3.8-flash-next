use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, Command, Stdio};

use super::*;

/// Set in the environment of the child process the lock tests spawn: the
/// lock file it takes.
const CHILD_ENV: &str = "LILY_TEST_INSTANCE_LOCK_CHILD";

fn temp_lock(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("lily-instance-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("nested").join("instance.lock")
}

/// Not a test of its own: the body of the child process. It takes the lock
/// named by [`CHILD_ENV`], says so on stdout and holds it until its stdin
/// says `exit` (or it is killed).
#[test]
#[ignore = "run as the child process of the lock tests"]
fn lock_holder_child() {
    let Some(path) = std::env::var_os(CHILD_ENV) else { return };
    let lock =
        InstanceLock::acquire_at(Path::new(&path)).expect("child takes the lock");
    println!("child holds the lock");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    drop(lock);
}

/// Spawns this test binary as [`lock_holder_child`] on `path` and waits
/// until it holds the lock.
fn spawn_holder(path: &Path) -> Child {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "instance::tests::lock_holder_child",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD_ENV, path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn the lock holder");
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    loop {
        match lines.next() {
            Some(Ok(line)) if line.contains("child holds the lock") => break,
            Some(Ok(_)) => {}
            other => panic!("the lock holder ended before taking the lock: {other:?}"),
        }
    }
    // The harness in the child reports its result on stdout when it ends; a
    // closed pipe would fail that report.
    std::thread::spawn(move || lines.for_each(drop));
    child
}

fn refusal(path: &Path) -> AlreadyRunning {
    let error = InstanceLock::acquire_at(path).expect_err("the lock is held");
    assert_eq!(exit_status(&error), EXIT_ALREADY_RUNNING);
    error.downcast::<AlreadyRunning>().expect("refused as AlreadyRunning")
}

#[test]
fn a_second_process_is_refused_with_the_holders_pid_until_the_holder_is_killed() {
    let path = temp_lock("killed");
    let mut child = spawn_holder(&path);
    let refused = refusal(&path);
    let holder = refused.holder.clone().expect("the holder recorded itself");
    assert_eq!(holder.pid, child.id());
    assert!(!holder.binary.is_empty());
    let message = refused.to_string();
    assert!(message.contains(&format!("pid {}", child.id())), "{message}");
    assert!(message.contains("tools/service/lily-service.sh stop"), "{message}");
    assert!(message.contains(&path.display().to_string()), "{message}");

    // SIGKILL: no destructor runs, the kernel still drops the lock.
    child.kill().unwrap();
    child.wait().unwrap();
    let lock =
        InstanceLock::acquire_at(&path).expect("free after the holder was killed");
    assert_eq!(lock.path(), path);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(Holder::parse(&text).unwrap().pid, std::process::id());
}

#[test]
fn the_lock_is_free_again_after_the_holder_exits() {
    let path = temp_lock("exited");
    let mut child = spawn_holder(&path);
    assert_eq!(refusal(&path).holder.map(|h| h.pid), Some(child.id()));
    child.stdin.take().unwrap().write_all(b"exit\n").unwrap();
    assert!(child.wait().unwrap().success());
    InstanceLock::acquire_at(&path).expect("free after the holder exited");
}

#[test]
fn a_second_handle_in_the_same_process_is_refused_and_dropping_frees_it() {
    let path = temp_lock("handles");
    let first = InstanceLock::acquire_at(&path).expect("first");
    let refused = refusal(&path);
    assert_eq!(refused.holder.map(|h| h.pid), Some(std::process::id()));
    drop(first);
    InstanceLock::acquire_at(&path).expect("free after the first handle dropped");
}

#[test]
fn a_refusal_behind_context_still_exits_75_and_other_errors_exit_1() {
    let refused =
        anyhow::Error::from(AlreadyRunning { path: "/x".into(), holder: None })
            .context("loading the model");
    assert_eq!(exit_status(&refused), 75);
    assert_eq!(exit_status(&anyhow::anyhow!("no checkpoint")), 1);
    // An empty or foreign file names no holder rather than failing.
    assert_eq!(Holder::parse(""), None);
    assert_eq!(Holder::parse("garbage"), None);
    assert_eq!(
        Holder::parse("pid 42\n/usr/local/bin/lily\n"),
        Some(Holder { pid: 42, binary: "/usr/local/bin/lily".into() })
    );
}
