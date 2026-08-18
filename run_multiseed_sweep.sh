#!/usr/bin/env bash
# Sequential driver for the multi-seed evaluation sweep: 3 conditions x 5 seeds
# = 15 training runs, then 15 held-out evaluations, then one statistical
# analysis pass. Runs on a single GPU, so everything is strictly sequential
# (no concurrent training jobs).
#
# Logs, three layers, all distinguishable by condition/seed:
#   multiseed_runs/driver.log                              <- this script's own timeline
#   multiseed_runs/<condition>/seed_<N>/train_stdout.log    <- one training run's full stdout
#   multiseed_runs/<condition>/seed_<N>/eval_stdout.log     <- one held-out eval's full stdout
#   multiseed_runs/<condition>/seed_<N>/run_manifest.json   <- condition/seed/task/checkpoint pointer
#   multiseed_runs/<condition>/seed_<N>/held_out_eval.csv   <- 100 held-out episodes for that seed
#   logs/rsl_rl/multiseed_<condition>/<timestamp>_seed<N>/  <- Isaac Lab's own TensorBoard logs
#
# Final output: multiseed_runs/statistical_report.json

set -uo pipefail  # deliberately NOT -e: one failed run should not kill the other 14

cd /home/ubuntu/IsaacLabv2
source /home/ubuntu/env_isaaclab/bin/activate

OUTPUT_DIR="multiseed_runs"
SEEDS=(1 2 3 4 5)
CONDITIONS=(hierarchical direct_thrust recurrent)
DRIVER_LOG="$OUTPUT_DIR/driver.log"
mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"
}

FAILED_RUNS=()

log "=== Multi-seed sweep started: ${#CONDITIONS[@]} conditions x ${#SEEDS[@]} seeds ==="
log "Conditions: ${CONDITIONS[*]} | Seeds: ${SEEDS[*]}"
log "hierarchical/direct_thrust train to 200 iterations, recurrent to 400 iterations (each task's own registered rsl_rl cfg -- not overridden here)."

log "=== PHASE 1/3: TRAINING (15 runs) ==="
for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log "--- TRAIN start: condition=$condition seed=$seed ---"
        if python pipeline.py train --condition "$condition" --seeds "$seed" --output_dir "$OUTPUT_DIR"; then
            log "TRAIN OK:   condition=$condition seed=$seed"
        else
            log "TRAIN FAIL: condition=$condition seed=$seed -- see $OUTPUT_DIR/$condition/seed_$seed/train_stdout.log"
            FAILED_RUNS+=("train:$condition:seed_$seed")
        fi
    done
done

log "=== PHASE 2/3: HELD-OUT EVALUATION (15 runs, 100 episodes each) ==="
for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ ! -f "$OUTPUT_DIR/$condition/seed_$seed/run_manifest.json" ]; then
            log "EVAL SKIP:  condition=$condition seed=$seed -- no run_manifest.json (training did not complete for this seed)"
            FAILED_RUNS+=("evaluate:$condition:seed_$seed:skipped_no_checkpoint")
            continue
        fi
        log "--- EVAL start: condition=$condition seed=$seed ---"
        if python pipeline.py evaluate --condition "$condition" --seeds "$seed" --output_dir "$OUTPUT_DIR" --num_episodes 100 --eval_seed 999; then
            log "EVAL OK:    condition=$condition seed=$seed"
        else
            log "EVAL FAIL:  condition=$condition seed=$seed -- see $OUTPUT_DIR/$condition/seed_$seed/eval_stdout.log"
            FAILED_RUNS+=("evaluate:$condition:seed_$seed")
        fi
    done
done

log "=== PHASE 3/3: STATISTICAL ANALYSIS ==="
python pipeline.py analyze --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$DRIVER_LOG"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log "=== SWEEP FINISHED WITH ${#FAILED_RUNS[@]} FAILURE(S): ${FAILED_RUNS[*]} ==="
else
    log "=== SWEEP FINISHED: all 15 train + 15 eval runs succeeded. ==="
fi
log "Final report (if produced): $OUTPUT_DIR/statistical_report.json"
