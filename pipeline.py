"""
pipeline.py -- multi-seed evaluation pipeline, wired to the real Isaac Lab
quadcopter environment (Isaac-Quadcopter-Direct-v0 / -Flat-v0 / -Recurrent-v0).

Lives at the IsaacLabv2 repo root so `./isaaclab.sh` and
`scripts/reinforcement_learning/rsl_rl/...` resolve as relative paths exactly
as they would from an interactive shell.

THREE COMMANDS, IN ORDER
--------------------------
    python pipeline.py train    --seeds 1 2 3 4 5
    python pipeline.py evaluate --seeds 1 2 3 4 5 --num_episodes 100
    python pipeline.py analyze  --seeds 1 2 3 4 5

Each also accepts --dry_run (fake data, no GPU/Isaac Sim involved) to sanity
check the CLI/CSV/statistics code path in isolation.

HOW THE THREE CONDITIONS MAP TO REAL ISAAC LAB TASKS
------------------------------------------------------
    hierarchical  -> Isaac-Quadcopter-Direct-v0            (200 iterations)
    direct_thrust -> Isaac-Quadcopter-Direct-Flat-v0        (200 iterations)
    recurrent     -> Isaac-Quadcopter-Direct-Recurrent-v0   (400 iterations,
                      per QuadcopterPPORunnerRecurrentCfg -- recurrent gets
                      2x the budget by design; not overridden here)
Iteration counts are NOT passed on the CLI -- each task's registered
rsl_rl runner cfg already sets the right value, and overriding it here would
silently desync from whatever the cfg says if one changes later.

TRAINING (cmd_train / real_train_one_run)
--------------------------------------------
Runs `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py` as a
subprocess per (condition, seed), with --experiment_name/--run_name set so
each run lands in a distinguishable, non-colliding log directory:
    logs/rsl_rl/multiseed_<condition>/<timestamp>_seed<N>/
Parses the exact timestamp train.py prints to stdout (rather than globbing,
which would race against concurrent runs) to locate that directory
afterward, finds the last model_*.pt checkpoint in it, and writes a small
run_manifest.json into this pipeline's own --output_dir tree pointing at
both -- that manifest is what `evaluate` reads.

EVALUATION (cmd_evaluate / real_evaluate_one_run)
------------------------------------------------------
Isaac Sim's app-launch cost (tens of seconds) makes calling a per-episode
Python function 100 times far too expensive -- the original three-script
design's `real_run_episode()` stub doesn't fit Isaac Lab's execution model.
Instead this calls a dedicated script, held_out_eval.py, ONCE per
(condition, seed): it boots Isaac Sim once, loads the frozen checkpoint
deterministically, runs `--num_episodes` envs in parallel for one
episode-length window (after a warm-up window -- see held_out_eval.py's
docstring for why that's needed), and writes held_out_eval.csv directly.
Eval seeds are offset by +100,000 from any training seed so held-out
episodes never reuse a training seed's randomness.

analyze (cmd_analyze) is unchanged from the original design: it only reads
the CSVs the steps above produce.
"""

import argparse
import csv
import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

CONDITIONS = ["hierarchical", "direct_thrust", "recurrent"]

TASK_ID_BY_CONDITION = {
    "hierarchical": "Isaac-Quadcopter-Direct-v0",
    "direct_thrust": "Isaac-Quadcopter-Direct-Flat-v0",
    "recurrent": "Isaac-Quadcopter-Direct-Recurrent-v0",
}
EXPERIMENT_NAME_BY_CONDITION = {
    "hierarchical": "multiseed_hierarchical",
    "direct_thrust": "multiseed_direct_thrust",
    "recurrent": "multiseed_recurrent",
}

ISAACLAB_ROOT = Path(__file__).resolve().parent
ISAACLAB_SH = str(ISAACLAB_ROOT / "isaaclab.sh")
EVAL_SEED_OFFSET = 100_000
# Real eval work (100 envs, warm-up + collection window) takes low single-digit
# minutes; this only needs to be generous enough to never truncate real work,
# not to match how long the (unrelated) Kit shutdown hang can run.
EVAL_SUBPROCESS_TIMEOUT_S = 900

REQUIRED_EPISODE_FIELDS = {
    "success": bool,
    "termination_cause": str,
    "final_distance_m": (int, float),
    "mean_tilt_penalty": (int, float),
    "mean_angvel_penalty": (int, float),
    "episode_return": (int, float),
}
VALID_TERMINATION_CAUSES = {"collision", "timeout", "tipped", "floor", "ceiling"}


def validate_episode_outcome(outcome: dict, condition: str, seed: int, episode_index: int):
    """
    Fail loudly, immediately, and with a specific error message if an episode
    outcome is malformed. Used on the dry-run path (the real path is
    schema-enforced by held_out_eval.py itself before it ever writes a CSV
    row, since that script owns the field names directly).
    """
    where = f"{condition}/seed_{seed}, episode {episode_index}"
    if not isinstance(outcome, dict):
        raise TypeError(f"[{where}] episode outcome must be a dict, got {type(outcome)}")

    missing = REQUIRED_EPISODE_FIELDS.keys() - outcome.keys()
    if missing:
        raise ValueError(f"[{where}] episode outcome is missing required field(s): {sorted(missing)}")

    for field, expected_type in REQUIRED_EPISODE_FIELDS.items():
        value = outcome[field]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"[{where}] field '{field}' has type {type(value).__name__}, expected {expected_type}."
            )

    if outcome["termination_cause"] not in VALID_TERMINATION_CAUSES:
        raise ValueError(
            f"[{where}] termination_cause='{outcome['termination_cause']}' is not one of "
            f"{sorted(VALID_TERMINATION_CAUSES)}."
        )

    if outcome["success"] and outcome["termination_cause"] == "collision":
        raise ValueError(f"[{where}] inconsistent outcome: success=True but termination_cause='collision'.")


FIXED_HYPERPARAMS = {
    "num_envs": 4096, "learning_rate": 5e-4, "clip": 0.2,
    "entropy_coef": 0.01, "discount": 0.99, "gae_lambda": 0.95,
    "epochs": 5, "minibatches": 16, "target_kl": 0.01,
    "steps_per_env_per_update": 128,
    # iterations are NOT listed here -- see module docstring; each task's own
    # registered rsl_rl runner cfg governs this (200 for hierarchical/direct_thrust,
    # 400 for recurrent) and is intentionally not overridden on the CLI.
}

# Manuscript's single-seed reference values, used ONLY to make the dry-run's
# fake data land in a realistic ballpark. Never used for anything except --dry_run.
_DRY_RUN_BASE_FAILURE_RATE = {"hierarchical": 0.18, "direct_thrust": 0.42, "recurrent": 0.188}
_DRY_RUN_BASE_DISTANCE = {"hierarchical": 0.042, "direct_thrust": 0.050, "recurrent": 0.041}


# =============================================================================
# STEP 1: TRAINING
# =============================================================================

def real_train_one_run(condition: str, seed: int, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    task = TASK_ID_BY_CONDITION[condition]
    experiment_name = EXPERIMENT_NAME_BY_CONDITION[condition]
    run_name = f"seed{seed}"

    cmd = [
        ISAACLAB_SH, "-p", "scripts/reinforcement_learning/rsl_rl/train.py",
        "--task", task,
        "--num_envs", str(FIXED_HYPERPARAMS["num_envs"]),
        "--seed", str(seed),
        "--headless",
        "--experiment_name", experiment_name,
        "--run_name", run_name,
    ]
    log_path = run_dir / "train_stdout.log"
    print(f"[train] {condition}/seed_{seed}: {' '.join(cmd)}")
    print(f"[train] logging to {log_path}")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=ISAACLAB_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {condition}/seed_{seed} (exit {result.returncode}); see {log_path}"
        )

    stdout_text = log_path.read_text()
    match = re.search(r"Exact experiment name requested from command line:\s*(\S+)", stdout_text)
    if not match:
        raise RuntimeError(
            f"Could not find the run timestamp in train.py's output for {condition}/seed_{seed}; see {log_path}"
        )
    timestamp = match.group(1)
    isaac_log_dir = ISAACLAB_ROOT / "logs" / "rsl_rl" / experiment_name / f"{timestamp}_{run_name}"
    if not isaac_log_dir.exists():
        raise RuntimeError(f"Expected log directory {isaac_log_dir} does not exist after training completed.")

    checkpoints = sorted(isaac_log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise RuntimeError(f"No model_*.pt checkpoints found in {isaac_log_dir}.")
    final_checkpoint = checkpoints[-1]

    manifest = {
        "condition": condition,
        "seed": seed,
        "task": task,
        "isaac_log_dir": str(isaac_log_dir),
        "checkpoint": str(final_checkpoint),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[train] {condition}/seed_{seed}: done. checkpoint={final_checkpoint}")


def _dry_run_train_one_run(condition: str, seed: int, run_dir: Path):
    """Fake training: just writes a manifest, no actual computation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"condition": condition, "seed": seed, "dry_run": True,
                "hyperparams": FIXED_HYPERPARAMS}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def cmd_train(args):
    for condition in ([args.condition] if args.condition else CONDITIONS):
        for seed in args.seeds:
            run_dir = args.output_dir / condition / f"seed_{seed}"
            print(f"[train] {condition} / seed={seed} {'(dry run)' if args.dry_run else ''}")
            if args.dry_run:
                _dry_run_train_one_run(condition, seed, run_dir)
            else:
                real_train_one_run(condition, seed, run_dir)
    print(f"\nDone. {'(Dry run -- no real training occurred.)' if args.dry_run else ''}")


# =============================================================================
# STEP 2: HELD-OUT EVALUATION
# =============================================================================

def real_evaluate_one_run(
    condition: str, seed: int, run_dir: Path, num_episodes: int, eval_seed: int, stochastic: bool = False
):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"No run_manifest.json in {run_dir}; run 'train' for {condition}/seed_{seed} first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dry_run"):
        raise RuntimeError(
            f"{run_dir} was produced by a --dry_run train step; re-run 'train' for real before evaluating."
        )

    episode_seed = EVAL_SEED_OFFSET + eval_seed
    csv_name = "held_out_eval_stochastic.csv" if stochastic else "held_out_eval.csv"
    out_csv = run_dir / csv_name
    cmd = [
        ISAACLAB_SH, "-p", "scripts/reinforcement_learning/rsl_rl/held_out_eval.py",
        "--task", manifest["task"],
        "--checkpoint", manifest["checkpoint"],
        "--num_envs", str(num_episodes),
        "--seed", str(episode_seed),
        "--out_csv", str(out_csv),
        "--headless",
    ]
    if stochastic:
        cmd.append("--stochastic")
    log_path = run_dir / ("eval_stochastic_stdout.log" if stochastic else "eval_stdout.log")
    print(f"[evaluate] {condition}/seed_{seed}: {' '.join(cmd)}")
    print(f"[evaluate] logging to {log_path}")

    # NOTE: Isaac Sim/Kit has been observed to hang in simulation_app.close() well
    # after held_out_eval.py has already written its CSV and printed "done" --
    # i.e. all real work completes (empirically ~1-2 minutes for 100 envs), only
    # process teardown hangs indefinitely (confirmed live: hierarchical/seed_1 sat
    # for ~51 minutes post-completion before being killed). Without a bound this
    # wedges the whole sequential sweep forever on one run.
    #
    # Rather than wait out a long fixed timeout every time (which was correct but
    # wasteful -- confirmed it costs the full EVAL_SUBPROCESS_TIMEOUT_S on every
    # run since the hang is the normal case, not the exception), poll for the CSV
    # becoming complete and send SIGTERM as soon as it is: empirically this lets
    # the process exit cleanly (returncode 0) within seconds, unlike SIGKILL.
    # EVAL_SUBPROCESS_TIMEOUT_S is kept as an outer safety bound for the case
    # where the CSV never completes (a genuine failure, not a shutdown hang).
    def _safe_killpg(sig):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass  # already exited between our poll() and this call

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=ISAACLAB_ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        returncode = None
        poll_start = time.monotonic()
        sigterm_sent_at = None
        sigkill_sent = False
        while True:
            live_returncode = proc.poll()
            if live_returncode is not None:
                returncode = live_returncode
                break
            elapsed = time.monotonic() - poll_start
            csv_complete = False
            if out_csv.exists():
                with open(out_csv) as f:
                    csv_complete = sum(1 for _ in f) - 1 >= num_episodes
            if csv_complete and sigterm_sent_at is None:
                print(f"[evaluate] {condition}/seed_{seed}: CSV complete after {elapsed:.0f}s, "
                      f"sending SIGTERM to end the (known) shutdown hang.")
                _safe_killpg(signal.SIGTERM)
                sigterm_sent_at = elapsed
            elif sigterm_sent_at is not None and not sigkill_sent and elapsed - sigterm_sent_at > 20:
                print(f"[evaluate] {condition}/seed_{seed}: still alive 20s after SIGTERM, sending SIGKILL.")
                _safe_killpg(signal.SIGKILL)
                sigkill_sent = True
            elif sigterm_sent_at is None and elapsed > EVAL_SUBPROCESS_TIMEOUT_S:
                print(f"[evaluate] {condition}/seed_{seed}: exceeded {EVAL_SUBPROCESS_TIMEOUT_S}s "
                      f"with no complete CSV -- force-killing process group (likely a genuine failure).")
                _safe_killpg(signal.SIGKILL)
                sigkill_sent = True
            time.sleep(3)
        if returncode is None:
            returncode = proc.wait()

    if not out_csv.exists():
        raise RuntimeError(
            f"Held-out evaluation for {condition}/seed_{seed} produced no {out_csv} "
            f"(exit={returncode}); see {log_path}"
        )
    with open(out_csv) as f:
        row_count = sum(1 for _ in f) - 1  # minus header
    if row_count < num_episodes:
        raise RuntimeError(
            f"{out_csv} has only {row_count}/{num_episodes} episodes (exit={returncode}); see {log_path}"
        )
    if returncode != 0:
        print(
            f"[evaluate] {condition}/seed_{seed}: nonzero/killed exit ({returncode}) but {out_csv} "
            f"has all {row_count} episodes -- treating as success (shutdown hang, not a data problem)."
        )
    print(f"[evaluate] {condition}/seed_{seed}: done. wrote {out_csv} ({row_count} episodes)")


def _dry_run_episode(condition: str, training_seed: int, episode_seed: int) -> dict:
    """Fake episode outcome, sampled around the manuscript's single-seed values.
    Both the training seed and episode index perturb the outcome, so that
    different training seeds produce genuinely different per-seed summary
    statistics -- otherwise the dry run can't meaningfully exercise the SD/CI
    code path."""
    rng = random.Random(hash((training_seed, episode_seed)) & 0xFFFFFFFF)
    seed_level_shift = random.Random(training_seed).uniform(-0.05, 0.05)
    base_fail = _DRY_RUN_BASE_FAILURE_RATE[condition] + seed_level_shift
    fail = rng.random() < max(0.0, min(1.0, base_fail + rng.uniform(-0.04, 0.04)))
    base_dist = _DRY_RUN_BASE_DISTANCE[condition]
    return {
        "success": not fail,
        "termination_cause": "collision" if fail else "timeout",
        "final_distance_m": round(base_dist + seed_level_shift * 0.05 + rng.uniform(-0.005, 0.005), 5),
        "mean_tilt_penalty": round(rng.uniform(-0.03, -0.0005), 5),
        "mean_angvel_penalty": round(rng.uniform(-0.2, -0.001), 5),
        "episode_return": round(rng.uniform(100, 170), 2),
    }


def cmd_evaluate(args):
    for condition in ([args.condition] if args.condition else CONDITIONS):
        for seed in args.seeds:
            run_dir = args.output_dir / condition / f"seed_{seed}"
            print(f"[evaluate] {condition} / seed={seed} "
                  f"({args.num_episodes} episodes) {'(dry run)' if args.dry_run else ''}")

            if not args.dry_run:
                real_evaluate_one_run(
                    condition, seed, run_dir, args.num_episodes, args.eval_seed, stochastic=args.stochastic
                )
                continue

            rows = []
            for i in range(args.num_episodes):
                ep_seed = EVAL_SEED_OFFSET + args.eval_seed + i
                outcome = _dry_run_episode(condition, seed, ep_seed)
                validate_episode_outcome(outcome, condition, seed, i)
                outcome["episode_index"] = i
                rows.append(outcome)

            run_dir.mkdir(parents=True, exist_ok=True)
            out_csv = run_dir / "held_out_eval.csv"
            with open(out_csv, "w", newline="") as f:
                fieldnames = ["episode_index", "success", "termination_cause",
                              "final_distance_m", "mean_tilt_penalty",
                              "mean_angvel_penalty", "episode_return"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    print(f"\nDone. {'(Dry run -- fake data, do not use in the paper.)' if args.dry_run else ''}")


# =============================================================================
# STEP 3: STATISTICAL ANALYSIS
# =============================================================================

def _load_seed_summary(run_dir: Path, csv_name: str = "held_out_eval.csv"):
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    n = len(rows)
    failures = sum(1 for r in rows if r["success"] == "False")
    collisions = sum(1 for r in rows if r["termination_cause"] == "collision")
    return {
        "n_episodes": n,
        "failure_rate": failures / n,
        "collision_rate": collisions / n,
        "mean_final_distance_m": mean(float(r["final_distance_m"]) for r in rows),
        "mean_tilt_penalty": mean(float(r["mean_tilt_penalty"]) for r in rows),
        "mean_angvel_penalty": mean(float(r["mean_angvel_penalty"]) for r in rows),
        "mean_episode_return": mean(float(r["episode_return"]) for r in rows),
    }


def _summarize(per_seed: dict, metric: str) -> dict:
    values = [s[metric] for s in per_seed.values()]
    n = len(values)
    if n < 3:
        return {"n_seeds": n, "mean": mean(values) if values else None, "sd": None, "ci95": None,
                "note": "Fewer than 3 seeds: SD/CI not computed."}
    m, sd = mean(values), stdev(values)
    ci = None
    if HAVE_SCIPY:
        se = sd / (n ** 0.5)
        ci = list(stats.t.interval(0.95, df=n - 1, loc=m, scale=se)) if se > 0 else [m, m]
    return {"n_seeds": n, "mean": m, "sd": sd, "ci95": ci}


def _paired_test(a: dict, b: dict, metric: str) -> dict:
    common = sorted(set(a) & set(b))
    if len(common) < 3 or not HAVE_SCIPY:
        return {"n_pairs": len(common), "p_value": None,
                "note": "Need >=3 paired seeds and scipy installed."}
    try:
        method = "exact" if len(common) <= 25 else "auto"
        _, p = stats.wilcoxon([a[s][metric] for s in common], [b[s][metric] for s in common],
                               method=method)
        return {"n_pairs": len(common), "p_value": p}
    except ValueError as e:
        return {"n_pairs": len(common), "p_value": None, "note": str(e)}


def cmd_analyze(args):
    per_condition = {}
    for condition in CONDITIONS:
        per_seed = {}
        for seed in args.seeds:
            run_dir = args.output_dir / condition / f"seed_{seed}"
            summary = _load_seed_summary(run_dir, csv_name=args.csv_name)
            if summary:
                per_seed[seed] = summary
            else:
                print(f"  [WARN] no {args.csv_name} for {condition}/seed_{seed} "
                      f"-- run 'evaluate' for this seed first.")
        per_condition[condition] = per_seed

    metrics = ["failure_rate", "collision_rate", "mean_final_distance_m",
               "mean_tilt_penalty", "mean_angvel_penalty", "mean_episode_return"]

    report = {"metric_summaries": {}, "paired_tests": {}}
    for cond, per_seed in per_condition.items():
        if per_seed:
            report["metric_summaries"][cond] = {m: _summarize(per_seed, m) for m in metrics}

    if per_condition.get("hierarchical") and per_condition.get("direct_thrust"):
        report["paired_tests"]["hierarchical_vs_direct_thrust"] = {
            m: _paired_test(per_condition["hierarchical"], per_condition["direct_thrust"], m)
            for m in metrics
        }
    if per_condition.get("hierarchical") and per_condition.get("recurrent"):
        report["paired_tests"]["hierarchical_vs_recurrent"] = {
            m: _paired_test(per_condition["hierarchical"], per_condition["recurrent"], m)
            for m in metrics
        }

    out_path = args.output_dir / args.report_name
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    if args.dry_run:
        print("(Dry run -- these numbers are FAKE, generated only to test the pipeline. "
              "Do not paste them into the paper.)")

    print("\n--- Summary ---")
    for cond, m in report["metric_summaries"].items():
        fr = m["failure_rate"]
        if fr["sd"] is not None:
            print(f"{cond:15s} failure_rate = {fr['mean']:.3f} +/- {fr['sd']:.3f} (n_seeds={fr['n_seeds']})")
        else:
            print(f"{cond:15s} failure_rate = {fr['mean']} (n_seeds={fr['n_seeds']}, SD n/a)")
    for comp, metrics_d in report["paired_tests"].items():
        p = metrics_d["failure_rate"].get("p_value")
        print(f"{comp}: failure_rate p-value = {p}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in [("train", cmd_train), ("evaluate", cmd_evaluate), ("analyze", cmd_analyze)]:
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="e.g. --seeds 1 2 3 4 5 (minimum 3 for statistics)")
        p.add_argument("--condition", choices=CONDITIONS, default=None,
                        help="omit to run all three conditions")
        p.add_argument("--output_dir", type=Path, default=Path("./multiseed_runs"))
        p.add_argument("--dry_run", action="store_true",
                        help="use fake data to test the pipeline before wiring real code")
        if name == "evaluate":
            p.add_argument("--num_episodes", type=int, default=100)
            p.add_argument("--eval_seed", type=int, default=999)
            p.add_argument("--stochastic", action="store_true", default=False,
                            help="sample actions instead of taking the deterministic mean action; "
                                 "writes held_out_eval_stochastic.csv instead of held_out_eval.csv")
        if name == "analyze":
            p.add_argument("--csv_name", type=str, default="held_out_eval.csv",
                            help="which per-seed CSV to read (e.g. held_out_eval_stochastic.csv)")
            p.add_argument("--report_name", type=str, default="statistical_report.json",
                            help="output report filename")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
