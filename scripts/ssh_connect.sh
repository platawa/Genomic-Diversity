#!/usr/bin/env bash
# ssh_connect.sh — Helper for connecting to ORCD Engaging cluster
#
# Usage:
#   source scripts/ssh_connect.sh
#   remote_cmd "ls -la /orcd/data/zhang_f/001/platawa/jan31_files/"
#   remote_cmd "conda activate evo2_sep28 && python score_chromosome.py --help"
#
# Requires an active SSH ControlMaster session (one Duo push).
# The ControlPersist=300s in SSH config keeps the socket alive 5 min
# after the parent session closes.

REMOTE_USER="platawa"
REMOTE_HOST="orcd-login001.mit.edu"
REMOTE_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
SSH_SOCKET="$HOME/.ssh/${REMOTE_USER}@${REMOTE_HOST}:22"

check_socket() {
    # Returns 0 if ControlMaster socket is alive
    ssh -o "ControlPath=$SSH_SOCKET" -o ControlMaster=no \
        -O check "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null
}

remote_cmd() {
    # Run a command on the remote cluster through the ControlMaster socket
    local cmd="$1"
    if ! check_socket; then
        echo "ERROR: No active SSH ControlMaster socket found." >&2
        echo "Open an SSH session in another terminal first:" >&2
        echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}" >&2
        echo "(This triggers one Duo push. The socket persists for 5 min.)" >&2
        return 1
    fi
    ssh -o "ControlPath=$SSH_SOCKET" -o ControlMaster=no \
        "${REMOTE_USER}@${REMOTE_HOST}" "$cmd"
}

remote_sync() {
    # Sync a local file to the remote jan31_files directory
    local local_path="$1"
    local remote_subpath="${2:-$1}"
    if ! check_socket; then
        echo "ERROR: No active SSH ControlMaster socket." >&2
        return 1
    fi
    rsync -avz --exclude '__pycache__' \
        -e "ssh -o ControlPath=$SSH_SOCKET -o ControlMaster=no" \
        "$local_path" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${remote_subpath}"
}

# Auto-check on source
if check_socket; then
    echo "SSH ControlMaster socket is ACTIVE."
else
    echo "WARNING: No active SSH socket. Open a session in another terminal:"
    echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
fi
