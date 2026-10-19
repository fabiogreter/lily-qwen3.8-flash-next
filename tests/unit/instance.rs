use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, Command, Stdio};

use super::*;

/// Set in the environment of the child process the lock tests spawn: the
/// lock file it takes.
const CHILD_ENV: &str = "LILY_TEST_INSTANCE_LOCK_CHILD";

/// Held by every test that takes a lock. A child spawned by one test can
/// inherit another test's lock descriptor between its fork and its exec
/// (close-on-exec only closes it at the exec), which holds that lock for a
/// moment and failed "dropping frees it" now and then under a parallel run.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
    let _serial = serial();
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
    let _serial = serial();
    let path = temp_lock("exited");
    let mut child = spawn_holder(&path);
    assert_eq!(refusal(&path).holder.map(|h| h.pid), Some(child.id()));
    child.stdin.take().unwrap().write_all(b"exit\n").unwrap();
    assert!(child.wait().unwrap().success());
    InstanceLock::acquire_at(&path).expect("free after the holder exited");
}

#[test]
fn a_second_handle_in_the_same_process_is_refused_and_dropping_frees_it() {
    let _serial = serial();
    let path = temp_lock("handles");
    let first = InstanceLock::acquire_at(&path).expect("first");
    let refused = refusal(&path);
    assert_eq!(refused.holder.map(|h| h.pid), Some(std::process::id()));
    drop(first);
    InstanceLock::acquire_at(&path).expect("free after the first handle dropped");
}

/// Set in the environment of [`lock_path_child`]: print the default lock
/// path and exit.
const PATH_CHILD_ENV: &str = "LILY_TEST_INSTANCE_PATH_CHILD";

/// Not a test of its own: the body of a child process that prints
/// [`default_path`] as it resolves in the child's environment.
#[test]
#[ignore = "run as the child process of the lock path test"]
fn lock_path_child() {
    if std::env::var_os(PATH_CHILD_ENV).is_none() {
        return;
    }
    let path = default_path().expect("the child resolves a lock path");
    println!("lock path: {}", path.display());
}

/// The lock path a child process of this test binary resolves with `home`
/// as its `HOME` (`None`: unset), in another working directory.
fn child_lock_path(home: Option<&str>) -> String {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "instance::tests::lock_path_child",
            "--ignored",
            "--nocapture",
        ])
        .env(PATH_CHILD_ENV, "1")
        .current_dir(std::env::temp_dir())
        .stderr(Stdio::inherit());
    match home {
        Some(home) => command.env("HOME", home),
        None => command.env_remove("HOME"),
    };
    let output = command.output().expect("run the path child");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("lock path: ").map(str::to_owned))
        .expect("the child printed its lock path")
}

/// Every process of the user locks the same absolute file whatever its
/// environment: without `HOME`, with a relative one, or with one pointing
/// elsewhere, as a benchmark started with a cleared environment once did.
#[test]
fn the_lock_path_does_not_depend_on_home() {
    let here = default_path().expect("lock path");
    assert!(here.is_absolute(), "{}", here.display());
    assert!(here.ends_with("Library/Caches/lily/instance.lock"), "{}", here.display());
    let here = here.display().to_string();
    assert_eq!(child_lock_path(None), here, "HOME unset");
    assert_eq!(child_lock_path(Some(".")), here, "HOME relative");
    assert_eq!(child_lock_path(Some("/nonexistent/elsewhere")), here, "HOME elsewhere");
}

#[test]
fn the_home_falls_back_to_an_absolute_home_and_never_to_a_relative_one() {
    let account = Some(PathBuf::from("/Users/account"));
    assert_eq!(
        resolve_home(account.clone(), Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/account")
    );
    assert_eq!(
        resolve_home(None, Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/env")
    );
    assert_eq!(
        resolve_home(Some("relative".into()), Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/env")
    );
    for env in [None, Some(OsString::from(".")), Some(OsString::from(""))] {
        let error = resolve_home(None, env.clone()).expect_err("no absolute home");
        assert!(format!("{error}").contains("no home directory"), "{env:?}: {error}");
    }
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
