#!/usr/bin/env bash
# Phase 2: (a) re-run recurrent's deterministic held-out eval now that
# held_out_eval.py correctly calls policy.reset(dones) (the LSTM hidden-state
# leak bug found after the first pass), and (b) run a stochastic-action pass
# across all 15 checkpoints to test whether training-time exploration noise,
# not architecture, explains the failure-rate inversion seen in phase 1.
# Reuses existing checkpoints -- no retraining.

set -uo pipefail

cd /home/ubuntu/IsaacLabv2
source /home/ubuntu/env_isaaclab/bin/activate

OUTPUT_DIR="multiseed_runs"
SEEDS=(1 2 3 4 5)
LOG="$OUTPUT_DIR/driver_phase2.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=== PHASE 2 START ==="

log "--- Re-running recurrent deterministic eval (hidden-state reset fix) ---"
for seed in "${SEEDS[@]}"; do
    log "recurrent (deterministic, corrected) seed=$seed start"
    if python pipeline.py evaluate --condition recurrent --seeds "$seed" --output_dir "$OUTPUT_DIR" --num_episodes 100 --eval_seed 999; then
        log "OK   recurrent-deterministic-corrected seed=$seed"
    else
        log "FAIL recurrent-deterministic-corrected seed=$seed"
    fi
done

log "--- Stochastic-action pass: all three conditions ---"
for condition in hierarchical direct_thrust recurrent; do
    for seed in "${SEEDS[@]}"; do
        log "stochastic $condition seed=$seed start"
        if python pipeline.py evaluate --condition "$condition" --seeds "$seed" --output_dir "$OUTPUT_DIR" --num_episodes 100 --eval_seed 999 --stochastic; then
            log "OK   stochastic $condition seed=$seed"
        else
            log "FAIL stochastic $condition seed=$seed"
        fi
    done
done

log "--- Analyze: corrected deterministic report ---"
python pipeline.py analyze --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" --csv_name held_out_eval.csv --report_name statistical_report.json 2>&1 | tee -a "$LOG"

log "--- Analyze: stochastic report ---"
python pipeline.py analyze --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" --csv_name held_out_eval_stochastic.csv --report_name statistical_report_stochastic.json 2>&1 | tee -a "$LOG"

log "=== PHASE 2 COMPLETE ==="
