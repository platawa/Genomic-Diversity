#!/usr/bin/env bash
# monitor_and_submit_shards.sh
#
# Watches for the 6 in-progress SAE runs to write their COMPLETED sentinels,
# then immediately submits 4 full-coverage shards for that chromosome via
# run_sae_fast.py.  Runs as a background process (via nohup).
#
# Usage (on the cluster):
#   nohup bash monitor_and_submit_shards.sh > logs/monitor_shards.log 2>&1 &
# ---------------------------------------------------------------------------

set -uo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
N_SHARDS=4
POLL_SECS=60

# Only chromosomes that currently have running old-style jobs.
# All other chromosomes (chr4-16, chr19, chr21-22, chrX, chrY) are submitted
# immediately via the companion submit block below.
WATCH_CHROMS=(chr1 chr2 chr3 chr17 chr18 chr20)
declare -A SUBMITTED

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Monitor started. Watching: ${WATCH_CHROMS[*]}"
log "Will submit ${N_SHARDS} shards per chromosome when COMPLETED sentinel appears."

while true; do
    ALL_DONE=true

    for CHROM in "${WATCH_CHROMS[@]}"; do
        # Already submitted for this chrom
        if [[ -n "${SUBMITTED[$CHROM]+x}" ]]; then
            continue
        fi

        ALL_DONE=false

        # Find the most recent sae run dir for this chrom (any run, not just max5000)
        SAE_ROOT="${PROJECT}/results/${CHROM}/sae"
        LATEST_RUN=$(ls -td "${SAE_ROOT}"/*/COMPLETED 2>/dev/null | head -1)

        if [[ -n "${LATEST_RUN}" ]]; then
            log "${CHROM}: COMPLETED sentinel found at ${LATEST_RUN}"
            log "${CHROM}: Submitting ${N_SHARDS} full-coverage shards..."
            cd "${PROJECT}"
            bash submit_sae_fast_shards.sh "${CHROM}" "${N_SHARDS}" 2>&1 | while IFS= read -r line; do
                log "  [submit] ${line}"
            done
            SUBMITTED[$CHROM]=1
            log "${CHROM}: Done submitting."
        fi
    done

    if $ALL_DONE; then
        log "All chromosomes submitted. Monitor exiting."
        break
    fi

    sleep "${POLL_SECS}"
done
