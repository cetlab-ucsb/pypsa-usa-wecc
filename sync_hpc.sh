#!/usr/bin/env bash
set -euo pipefail

# Set this once, or provide it when running the script:
#   HPC_LOGIN="username@login.example.edu" ./sync_hpc.sh check
HPC_LOGIN="${HPC_LOGIN:-gabehoku@braid2.cnsi.ucsb.edu}"

# Absolute path to the workflow directory on the HPC.
HPC_TARGET="${HPC_TARGET:-/home/gabehoku/pypsa-usa-wecc/workflow}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"

usage() {
    echo "Usage:"
    echo "  ./sync_hpc.sh check   # preview changes; modify nothing"
    echo "  ./sync_hpc.sh apply   # update the HPC folders"
}

case "$MODE" in
    check)
        DO_DRY_RUN=1
        echo "Previewing changes. Nothing will be modified on the HPC."
        ;;
    apply)
        DO_DRY_RUN=0
        echo "Updating scripts, rules, and config on the HPC."
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 1
        ;;
esac

if [[ "$HPC_LOGIN" == "YOUR_USERNAME@YOUR_HPC_HOST" ]]; then
    echo "ERROR: Set HPC_LOGIN near the top of this file first."
    echo "Example: HPC_LOGIN=\"netid@login.cluster.edu\""
    exit 1
fi

for folder in scripts rules config; do
    if [[ ! -d "$PROJECT_ROOT/workflow/$folder" ]]; then
        echo "ERROR: Local folder not found: $PROJECT_ROOT/workflow/$folder"
        exit 1
    fi
done

# Reuse one SSH connection so password/MFA authentication is normally needed
# only once during this script run. Keep the socket path short enough for the
# macOS Unix-socket path-length limit. %d expands to the local home directory.
CONTROL_PATH="%d/.ssh/sync-hpc-%C"
SSH_OPTIONS=(
    -o ControlMaster=auto
    -o ControlPersist=60
    -o "ControlPath=$CONTROL_PATH"
)
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=$CONTROL_PATH"

close_ssh_connection() {
    ssh "${SSH_OPTIONS[@]}" -O exit "$HPC_LOGIN" >/dev/null 2>&1 || true
}
trap close_ssh_connection EXIT

echo "Checking the remote destination..."

set +e
ssh "${SSH_OPTIONS[@]}" "$HPC_LOGIN" \
    "test -d '$HPC_TARGET/scripts' &&
     test -d '$HPC_TARGET/rules' &&
     test -d '$HPC_TARGET/config'"
SSH_STATUS=$?
set -e

if [[ "$SSH_STATUS" -eq 255 ]]; then
    echo "ERROR: Could not connect to the HPC."
    echo "Check your network/VPN, login address, password, and MFA response."
    echo "No files were synchronized."
    exit 1
elif [[ "$SSH_STATUS" -ne 0 ]]; then
    echo "ERROR: One or more expected HPC folders do not exist:"
    echo "  $HPC_TARGET/scripts"
    echo "  $HPC_TARGET/rules"
    echo "  $HPC_TARGET/config"
    echo "No files were synchronized."
    exit 1
fi

# Run rsync separately for each folder. This keeps --delete safely scoped to
# that folder and prevents it from touching data, results, or other HPC files.
RSYNC_OPTIONS=(
    -avz
    --delete
    --itemize-changes
    --exclude=".DS_Store"
    --exclude="__pycache__/"
    --exclude="*.pyc"
)

if [[ "$DO_DRY_RUN" -eq 1 ]]; then
    RSYNC_OPTIONS+=(--dry-run)
fi

for folder in scripts rules config; do
    echo
    echo "Synchronizing workflow/$folder/"

    rsync \
        "${RSYNC_OPTIONS[@]}" \
        -e "$RSYNC_SSH" \
        "$PROJECT_ROOT/workflow/$folder/" \
        "$HPC_LOGIN:$HPC_TARGET/$folder/"
done

echo

if [[ "$MODE" == "check" ]]; then
    echo "Preview complete. Review the changes above."
    echo "Run './sync_hpc.sh apply' when you are ready to update the HPC."
else
    echo "HPC synchronization complete."
fi
