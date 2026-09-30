#!/bin/bash
# Installs and manages lily as a per-user launchd agent (macOS).
#
#   tools/service/lily-service.sh install    render the plist, bootstrap (or re-bootstrap) the agent
#   tools/service/lily-service.sh uninstall  bootout the agent and delete the plist
#   tools/service/lily-service.sh start      run it now (kickstart)
#   tools/service/lily-service.sh stop       SIGTERM it: the running request gets 10 s, sessions are
#                                            spilled to disk, exit 0; stays down until `start` or login
#   tools/service/lily-service.sh restart    stop, wait for the old process to release the
#                                            model's memory, then start
#   tools/service/lily-service.sh status     launchd state, pid, exit code, /health, memory
#   tools/service/lily-service.sh logs [N]   the last N (default 40) log lines, then follow
#
# Only one lily instance runs at a time (~/Library/Caches/lily/instance.lock):
# while another lily process (a development server, lily-bench, lily-probe)
# holds the lock the service's start exits 75, and launchd retries every 30 s
# until that process has exited. `install`, `start`, `restart` and `status`
# say when that is the case.
#
# Environment for `install` (all optional except the model):
#   LILY_MODEL        checkpoint directory (default: ~/models/Qwen3.8-Flash-Next-lily-q4)
#   LILY_BIN          server binary (default: <repo>/target/release/lily)
#   LILY_BIND         listen address (default: 127.0.0.1:8000)
#   LILY_MAX_SEQ      --max-seq (default: 131072)
#   LILY_IDLE_UNLOAD  --idle-unload (default: 30m; 0 never unloads)
#   LILY_EXTRA_ARGS   more server flags, whitespace separated (default: none)
#   LILY_LABEL        launchd label (default: com.lily.server)
set -euo pipefail

here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
repo=$(realpath "$here/../..")
label=${LILY_LABEL:-com.lily.server}
uid=$(id -u)
target="gui/$uid/$label"
agents="$HOME/Library/LaunchAgents"
plist="$agents/$label.plist"
log_dir="$HOME/Library/Logs/lily"
lock_file="$HOME/Library/Caches/lily/instance.lock"

die() { echo "lily-service: $*" >&2; exit 1; }

# Escapes a value for use as a sed replacement with `|` as the delimiter.
sed_escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }

# Escapes a value for XML text.
xml_escape() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

loaded() { launchctl print "$target" >/dev/null 2>&1; }

render() {
    local bin=${LILY_BIN:-$repo/target/release/lily}
    local model=${LILY_MODEL:-$HOME/models/Qwen3.8-Flash-Next-lily-q4}
    local bind=${LILY_BIND:-127.0.0.1:8000}
    local max_seq=${LILY_MAX_SEQ:-131072}
    local idle=${LILY_IDLE_UNLOAD:-30m}
    [ -x "$bin" ] || die "no server binary at $bin (cargo build --release --locked, or set LILY_BIN)"
    [ -f "$model/config.json" ] || die "no checkpoint at $model (set LILY_MODEL)"
    local extra=""
    local arg
    for arg in ${LILY_EXTRA_ARGS:-}; do
        extra+="        <string>$(xml_escape "$arg")</string>"$'\n'
    done
    mkdir -p "$agents" "$log_dir"
    local extra_file
    extra_file=$(mktemp "${TMPDIR:-/tmp}/lily-service.XXXXXX")
    printf '%s' "$extra" > "$extra_file"
    sed -e "s|@LABEL@|$(sed_escape "$(xml_escape "$label")")|g" \
        -e "s|@LILY_BIN@|$(sed_escape "$(xml_escape "$bin")")|g" \
        -e "s|@MODEL_DIR@|$(sed_escape "$(xml_escape "$model")")|g" \
        -e "s|@BIND@|$(sed_escape "$(xml_escape "$bind")")|g" \
        -e "s|@MAX_SEQ@|$(sed_escape "$(xml_escape "$max_seq")")|g" \
        -e "s|@IDLE_UNLOAD@|$(sed_escape "$(xml_escape "$idle")")|g" \
        -e "s|@WORKDIR@|$(sed_escape "$(xml_escape "$repo")")|g" \
        -e "s|@LOG_DIR@|$(sed_escape "$(xml_escape "$log_dir")")|g" \
        "$here/com.lily.server.plist.template" \
        | awk -v extra="$extra_file" '
            /@EXTRA_ARGS@/ {
                while ((getline line < extra) > 0) print line
                sub(/@EXTRA_ARGS@/, "")
            }
            { print }' > "$plist.tmp"
    rm -f "$extra_file"
    plutil -lint "$plist.tmp" >/dev/null || { rm -f "$plist.tmp"; die "rendered plist is invalid"; }
    mv "$plist.tmp" "$plist"
    echo "rendered $plist"
    echo "  $bin --model $model --bind $bind --max-seq $max_seq --idle-unload $idle ${LILY_EXTRA_ARGS:-}"
}

cmd_install() {
    render
    if loaded; then
        echo "re-bootstrapping $target (it was loaded)"
        local old
        old=$(service_pid)
        launchctl bootout "$target"
        # bootout is asynchronous; wait for the old instance to leave.
        local i
        for i in $(seq 1 100); do loaded || break; sleep 0.1; done
        # And for its process, which spills the sessions before it exits
        # (launchd kills it after the plist's 90 s ExitTimeOut): a new
        # instance started before that is refused by the instance lock and
        # only comes up on launchd's next retry.
        if [ -n "$old" ]; then
            for i in $(seq 1 60); do kill -0 "$old" 2>/dev/null || break; sleep 2; done
            if kill -0 "$old" 2>/dev/null; then
                echo "pid $old is still exiting; the new instance starts once it is gone"
            fi
        fi
    fi
    warn_other_instance
    launchctl bootstrap "gui/$uid" "$plist"
    echo "bootstrapped $target; log: $log_dir/server.log"
}

cmd_uninstall() {
    if loaded; then
        launchctl bootout "$target"
        echo "booted out $target"
    else
        echo "$target is not loaded"
    fi
    if [ -f "$plist" ]; then
        rm -f "$plist"
        echo "removed $plist"
    fi
}

cmd_start() {
    loaded || die "$target is not loaded; run install first"
    warn_other_instance
    launchctl kickstart "$target"
    echo "kickstarted $target"
}

cmd_stop() {
    loaded || die "$target is not loaded"
    if launchctl kill SIGTERM "$target" 2>/dev/null; then
        echo "sent SIGTERM to $target (a running request has 10 s; sessions are spilled to disk)"
    else
        echo "$target is not running"
        # Between two refused starts (exit 75) or failed loads there is no
        # process to signal, and launchd starts the next attempt within 30 s.
        local code
        code=$(launchctl print "$target" 2>/dev/null | awk -F' = ' '/^\tlast exit code = /{print $2; exit}')
        if [ -n "$code" ] && [ "$code" != 0 ] && [ "$code" != "(never exited)" ]; then
            echo "  its last exit was $code, so launchd starts it again within 30 s; uninstall keeps it down"
        fi
    fi
}

# The pid launchd currently tracks for the job, empty when it is not running.
service_pid() {
    launchctl print "$target" 2>/dev/null | awk -F' = ' '/^\tpid = /{print $2; exit}'
}

# Waits for the running instance to be gone. The outgoing process holds the
# model's GPU memory (73 GB) until it has finished spilling its sessions, so a
# replacement started before it exits asks a 128 GB machine for two full
# models at once. That happened on 2026-09-12 (102.8 GB + 68.6 GB resident,
# 62 MB free) and the machine panicked.
wait_for_exit() {
    local timeout=${1:-180} waited=0
    while [ -n "$(service_pid)" ]; do
        if [ "$waited" -ge "$timeout" ]; then
            die "still running ${timeout}s after SIGTERM; refusing to start a second instance"
        fi
        sleep 2
        waited=$((waited + 2))
    done
    [ "$waited" -gt 0 ] && echo "previous instance exited after ${waited}s"
    return 0
}

# The live process the instance lock file names, as "pid command", or
# nothing. The lock itself is an flock its holder keeps until it exits; the
# pid and binary written into the file are informational, so a pid that is
# gone or now runs something else is a stale record, not a holder.
lock_holder() {
    [ -f "$lock_file" ] || return 0
    local pid binary command
    pid=$(awk '/^pid /{print $2; exit}' "$lock_file")
    binary=$(sed -n 2p "$lock_file")
    { [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; } || return 0
    command=$(ps -o command= -p "$pid" 2>/dev/null) || return 0
    case "$command" in
        *"${binary##*/}"*) printf '%s %s\n' "$pid" "$command" ;;
    esac
    return 0
}

# Says so when a lily process other than the service holds the instance
# lock: the service would exit 75 and launchd retry every 30 s until that
# process is gone.
warn_other_instance() {
    local holder
    holder=$(lock_holder)
    [ -n "$holder" ] || return 0
    [ "${holder%% *}" = "$(service_pid)" ] && return 0
    echo "lily-service: another lily instance holds $lock_file: pid $holder" >&2
    echo "lily-service: the service refuses to start (exit 75); launchd retries every 30 s until that process exits" >&2
}

cmd_restart() {
    loaded || die "$target is not loaded; run install first"
    cmd_stop
    # The lock is released when the old process exits, which is before
    # launchd stops reporting its pid, so the kickstart below never races
    # into a refusal by its own predecessor.
    wait_for_exit
    warn_other_instance
    launchctl kickstart "$target"
    echo "restarted $target"
}

cmd_status() {
    if ! loaded; then
        echo "$target: not loaded"
        [ -f "$plist" ] && echo "plist present at $plist (run install)"
        return 1
    fi
    local info
    info=$(launchctl print "$target")
    local state pid code bind
    # The service's own lines are indented one tab; the deeper ones belong
    # to its endpoints and domain.
    state=$(printf '%s\n' "$info" | awk -F' = ' '/^\tstate = /{print $2}' | head -1)
    pid=$(printf '%s\n' "$info" | awk -F' = ' '/^\tpid = /{print $2}' | head -1)
    code=$(printf '%s\n' "$info" | awk -F' = ' '/^\tlast exit code = /{print $2}' | head -1)
    # The argument after --bind, read with plutil alone: a python3 here
    # resolves through the repo's .tool-versions and fails when that
    # interpreter is not installed.
    bind=
    local i=0 arg
    while arg=$(plutil -extract "ProgramArguments.$i" raw -o - "$plist" 2>/dev/null); do
        if [ "$arg" = --bind ]; then
            bind=$(plutil -extract "ProgramArguments.$((i + 1))" raw -o - "$plist" 2>/dev/null)
            break
        fi
        i=$((i + 1))
    done
    echo "$target: state ${state:-?}, pid ${pid:-none}, last exit code ${code:-none}"
    if [ "${code:-}" = 75 ] && [ -z "${pid:-}" ]; then
        echo "  exit 75: refused to start while another lily instance held the lock; launchd retries every 30 s"
    fi
    local holder
    holder=$(lock_holder)
    if [ -n "$holder" ] && [ "${holder%% *}" != "${pid:-}" ]; then
        echo "lock  held by another lily instance: pid $holder"
    fi
    echo "plist $plist"
    echo "log   $log_dir/server.log"
    if [ -n "${pid:-}" ]; then
        echo "memory: $(ps -o rss=,vsz= -p "$pid" | awk '{printf "rss %.1f GB, vsz %.1f GB", $1/1048576, $2/1048576}')"
    fi
    local health
    if health=$(curl -s -m 2 "http://$bind/health"); then
        echo "health: $health"
    else
        echo "health: http://$bind/health not answering"
    fi
}

cmd_logs() {
    local n=${1:-40}
    [ -f "$log_dir/server.log" ] || die "no log at $log_dir/server.log"
    tail -n "$n" -f "$log_dir/server.log"
}

case "${1:-}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      cmd_logs "${2:-40}" ;;
    *) sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
