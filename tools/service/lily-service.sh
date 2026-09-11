#!/bin/bash
# Installs and manages lily as a per-user launchd agent (macOS).
#
#   tools/service/lily-service.sh install    render the plist, bootstrap (or re-bootstrap) the agent
#   tools/service/lily-service.sh uninstall  bootout the agent and delete the plist
#   tools/service/lily-service.sh start      run it now (kickstart)
#   tools/service/lily-service.sh stop       SIGTERM it: the running request gets 10 s, sessions are
#                                            spilled to disk, exit 0; stays down until `start` or login
#   tools/service/lily-service.sh restart    kickstart -k (stop, then start)
#   tools/service/lily-service.sh status     launchd state, pid, exit code, /health, memory
#   tools/service/lily-service.sh logs [N]   the last N (default 40) log lines, then follow
#
# Environment for `install` (all optional except the model):
#   LILY_MODEL        checkpoint directory (default: ~/projects/personal/local-llms/models/Qwen3.8-Flash-Next-lily-q4)
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

die() { echo "lily-service: $*" >&2; exit 1; }

# Escapes a value for use as a sed replacement with `|` as the delimiter.
sed_escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }

# Escapes a value for XML text.
xml_escape() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

loaded() { launchctl print "$target" >/dev/null 2>&1; }

render() {
    local bin=${LILY_BIN:-$repo/target/release/lily}
    local model=${LILY_MODEL:-$HOME/projects/personal/local-llms/models/Qwen3.8-Flash-Next-lily-q4}
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
        launchctl bootout "$target"
        # bootout is asynchronous; wait for the old instance to leave.
        local i
        for i in $(seq 1 100); do loaded || break; sleep 0.1; done
    fi
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
    launchctl kickstart "$target"
    echo "kickstarted $target"
}

cmd_stop() {
    loaded || die "$target is not loaded"
    if launchctl kill SIGTERM "$target" 2>/dev/null; then
        echo "sent SIGTERM to $target (a running request has 10 s; sessions are spilled to disk)"
    else
        echo "$target is not running"
    fi
}

cmd_restart() {
    loaded || die "$target is not loaded; run install first"
    launchctl kickstart -k "$target"
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
    bind=$(plutil -extract ProgramArguments json -o - "$plist" | python3 -c 'import json,sys; a=json.load(sys.stdin); print(a[a.index("--bind")+1])')
    echo "$target: state ${state:-?}, pid ${pid:-none}, last exit code ${code:-none}"
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
    *) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
