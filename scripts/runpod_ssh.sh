#!/usr/bin/env bash
# Helper: run commands on RunPod via expect (works around PTY requirement).
#
# Usage:
#   bash scripts/runpod_ssh.sh <ssh_target> <command> [timeout_seconds]
#
# Example:
#   bash scripts/runpod_ssh.sh user@ssh.runpod.io "nvidia-smi -L" 30
#
# RunPod's SSH gateway requires a PTY, which non-interactive shells lack.
# This script uses expect + RequestTTY=force to work around it.

set -euo pipefail

SSH_TARGET="${1:?Usage: $0 <ssh_target> <command> [timeout_sec]}"
CMD="${2:?Usage: $0 <ssh_target> <command> [timeout_sec]}"
TIMEOUT="${3:-300}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

expect -c "
set timeout $TIMEOUT
log_user 1
spawn ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o RequestTTY=force -i $SSH_KEY $SSH_TARGET

expect {
    -re {[#\$] } {}
    timeout { puts \"SSH_TIMEOUT\"; exit 1 }
    eof { puts \"SSH_EOF\"; exit 1 }
}

send \"$CMD\r\"

expect {
    -re {[#\$] } {}
    timeout { puts \"CMD_TIMEOUT\"; exit 1 }
}

send \"exit\r\"
expect eof
"
