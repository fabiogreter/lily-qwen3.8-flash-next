# tools/service: lily as a launchd agent

Runs the server in the background for the logged-in user, starts it at login,
restarts it after a crash or a failed load, and lets `--idle-unload` return the
model's memory to the system while nobody is talking to it.

```sh
cargo build --release --locked
LILY_MODEL=~/models/Qwen3.8-Flash-Next-lily-q4 \
    tools/service/lily-service.sh install
tools/service/lily-service.sh status
tools/service/lily-service.sh logs
```

`install` renders `com.lily.server.plist.template` into
`~/Library/LaunchAgents/com.lily.server.plist` (absolute binary and model
paths, `--bind 127.0.0.1:8000 --max-seq 131072 --idle-unload 30m`, any
`LILY_EXTRA_ARGS`), then `launchctl bootstrap`s it into the user's `gui`
domain; running it again re-renders and re-bootstraps, so changed flags take
effect. `uninstall` boots the agent out and deletes the plist. `stop` sends
SIGTERM (the running request gets 10 s, resident sessions are spilled to
`~/Library/Caches/lily/sessions`, exit 0), `start` kickstarts it, `restart`
does both. `status` shows launchd's state, the pid and last exit code, the
process's memory and `/health`. Logs go to `~/Library/Logs/lily/server.log`.

Environment for `install`: `LILY_MODEL` (checkpoint directory), `LILY_BIN`,
`LILY_BIND`, `LILY_MAX_SEQ`, `LILY_IDLE_UNLOAD` (`0` keeps the model loaded),
`LILY_EXTRA_ARGS` (e.g. `"--mtp-drafts 0 --reasoning-effort low"`),
`LILY_LABEL`.

## What the plist says, and why

| key | value | reason |
|-----|-------|--------|
| `RunAtLoad` | true | up after login without a request |
| `KeepAlive` | `{SuccessfulExit: false}` | restart after a crash or a failed load (exit 1), but a clean stop (SIGTERM, exit 0) stays down until `start` or the next login |
| `ThrottleInterval` | 30 s | launchd never respawns faster than this (default 10 s); a load that keeps failing then costs one attempt per half minute instead of spinning |
| `ExitTimeOut` | 90 s | time between launchd's SIGTERM and its SIGKILL; the server needs up to 10 s for the running request plus the session spill |
| `ProcessType` | `Interactive` | `launchd.plist(5)`: jobs without a type get "light resource limits", throttling CPU and I/O; Interactive runs like an app |
| `WorkingDirectory` | the repository | the server itself does not use the working directory |
| `StandardOutPath`/`StandardErrorPath` | `~/Library/Logs/lily/server.log` | the server logs to stderr only |

Not set: `Nice` (the ProcessType is the recommended way to express this),
`EnvironmentVariables` (the server reads none; `HOME` is set by launchd and
locates the disk tier), `LimitLoadToSessionType` (defaults to `Aqua`, the
logged-in session).

## Checking it by hand

```sh
launchctl print gui/$UID/com.lily.server | grep -E 'state|pid|last exit'
curl -s http://127.0.0.1:8000/health
# {"idle_unload_secs":1800,"model":"Qwen3.8-Flash-Next","state":"ready","status":"ok"}
ps -o rss=,vsz= -p $(launchctl print gui/$UID/com.lily.server | awk '/pid = /{print $3}')
```

`state` is `loading` (503, first load), `ready`, `idle` (unloaded, the next
request reloads it and waits), `reloading`, or `stopping` (503). `status`
stays `ok` whenever a request would be served, so a client that only reads
the status code sees no difference between `ready` and `idle`.
