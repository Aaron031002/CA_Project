#!/bin/bash
# Start the reference environment: the Docker image `ca_fp_env`, with this
# repository mounted at /workspace and the container running as your own user
# (so every file it writes stays yours). See USER_GUIDE.md §3.3.
#
# Usage (WSL / Linux / macOS terminal, from any folder):
#   bash start_env.sh                 # interactive shell at /workspace
#   bash start_env.sh <command...>    # run one command at /workspace and exit
#   bash start_env.sh --fix-owner     # give root-owned files in the repository back to you

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=ca_fp_env

if ! command -v docker >/dev/null 2>&1; then
    echo "start_env.sh: the 'docker' command isn't available in this terminal (USER_GUIDE.md §3.1)." >&2
    echo "  Windows (WSL): start Docker Desktop. If it is already running, open its Settings ->" >&2
    echo "    Resources -> WSL integration, turn on your distribution (e.g. Ubuntu), click" >&2
    echo "    Apply & restart, then open a new WSL terminal." >&2
    echo "  macOS / Linux: install Docker first." >&2
    exit 1
fi
if ! DOCKER_INFO=$(docker info 2>&1); then
    if printf '%s\n' "$DOCKER_INFO" | grep -qi 'permission denied'; then
        echo "start_env.sh: Docker is running, but your user can't use it. On Linux, add yourself" >&2
        echo "  to the docker group once, then log out and back in (USER_GUIDE.md §4):" >&2
        echo "    sudo usermod -aG docker \"\$USER\"" >&2
    else
        echo "start_env.sh: can't reach Docker. Start Docker Desktop (or the Docker engine) first;" >&2
        echo "  see USER_GUIDE.md §3.1." >&2
    fi
    # The error lines follow the "Server:" heading; fall back to the last lines.
    ERR=$(printf '%s\n' "$DOCKER_INFO" | sed -n '/^Server:/,$p' | grep -v '^Server:' | grep -v '^ *$' | head -3)
    [ -n "$ERR" ] || ERR=$(printf '%s\n' "$DOCKER_INFO" | grep -v '^ *$' | tail -3)
    echo "The error from Docker was:" >&2
    printf '%s\n' "$ERR" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "start_env.sh: the image '$IMAGE' isn't loaded yet. Load it first (USER_GUIDE.md §3.2):" >&2
    echo "  docker load -i <your image path>     (the downloaded ca_fp_env_v1.tar.gz)" >&2
    exit 1
fi

# The image is x86_64 only; on an ARM host (Apple Silicon) run it under emulation.
PLATFORM=()
case "$(uname -m)" in arm64|aarch64) PLATFORM=(--platform linux/amd64) ;; esac

if [ "${1:-}" = "--fix-owner" ]; then
    exec docker run --rm "${PLATFORM[@]}" -v "$REPO_ROOT:/workspace" "$IMAGE" \
        chown -R "$(id -u):$(id -g)" /workspace
fi

# -t only when attached to a terminal, so `bash start_env.sh <command>` also
# works from scripts.
TTY=(-i)
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

# --init: without it, a command started as `bash start_env.sh <command>` runs as
# the container's first process, which ignores Ctrl-C.
# HOME: your user ID has no home folder in the image, but conda, pip and the VS Code
# server need one to write their settings. This gives an empty one that any user ID
# can write to; it lives in memory and goes away with the container. (Not /tmp: VS
# Code keeps its sockets there, which is what an attached VS Code would then show.)
HOME_DIR=/home/ca_fp1_2026f
exec docker run "${TTY[@]}" --rm --init "${PLATFORM[@]}" \
    --user "$(id -u):$(id -g)" \
    --tmpfs "$HOME_DIR:exec,mode=1777" -e HOME="$HOME_DIR" \
    -v "$REPO_ROOT:/workspace" "$IMAGE" "$@"
