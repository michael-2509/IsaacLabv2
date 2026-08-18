# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, deterministic, batched held-out evaluation for a single trained checkpoint.

Unlike play.py (which is built for an interactive, indefinitely-running viewer
loop), this script runs exactly `--num_envs` fresh held-out episodes in
parallel -- one per environment, first-termination-only -- for exactly one
`max_episode_length` window, then exits and writes a CSV. It exists because
launching the Isaac Sim app per single episode (as a naive port of the
original multi-seed-eval design would do) is far too expensive; Isaac Lab's
whole execution model is built around evaluating many envs in one process,
so held-out evaluation should be too.

Relies on the `episode_outcomes` extras key added to `_reset_idx` in
quadcopterEnv_current.py: a list of per-env dicts (env_id, success,
termination_cause, final_distance_m, mean_tilt_penalty, mean_angvel_penalty,
episode_return) populated exactly on the step where that env's episode ends.
Only the *first* such outcome per env is kept, discarding it if a fast env
completes a second episode within the same window.

Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/held_out_eval.py \\
        --task Isaac-Quadcopter-Direct-v0 \\
        --checkpoint logs/rsl_rl/multiseed_hierarchical/2026-07-26_10-00-00_seed1/model_199.pt \\
        --num_envs 100 --seed 100001 \\
        --out_csv multiseed_runs/hierarchical/seed_1/held_out_eval.csv \\
        --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Headless batched held-out evaluation with RSL-RL.")
parser.add_argument("--num_envs", type=int, required=True, help="Number of held-out episodes (one per env).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, required=True, help="Eval seed (must not overlap any training seed).")
parser.add_argument("--out_csv", type=str, required=True, help="Path to write the per-episode CSV.")
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions from the policy's distribution instead of taking the deterministic mean action. "
         "Used to test whether training-time exploration noise (not architecture) drives failure-rate results.",
)
# NOTE: --checkpoint is provided by cli_args.add_rsl_rl_args below (used as required here).
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    raise SystemExit("--checkpoint is required")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import importlib.metadata as metadata
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_checkpoint,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

REQUIRED_FIELDS = [
    "success",
    "termination_cause",
    "final_distance_m",
    "mean_tilt_penalty",
    "mean_angvel_penalty",
    "episode_return",
]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Same two calls play.py makes before touching the cfg: update_rsl_rl_cfg applies
    # CLI overrides (experiment_name/run_name/checkpoint/etc.), and
    # handle_deprecated_rsl_rl_cfg resolves the cfg structure (e.g. actor/critic
    # class_name) for the installed rsl_rl version. Skipping the latter causes a
    # KeyError: 'class_name' inside OnPolicyRunner.__init__ -- found by smoke test.
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading checkpoint for held-out eval from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, INSTALLED_RSL_RL_VERSION)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    num_envs = args_cli.num_envs
    max_steps = int(env.unwrapped.max_episode_length)
    collected: dict[int, dict] = {}

    obs = env.get_observations()
    with torch.inference_mode():
        # Warm-up window (discarded): `enable_staggered_resets` randomizes every
        # env's initial episode_length_buf on the construction-time full-batch
        # reset so training terminations don't spike in sync. That means each
        # env's *first* episode after boot is an artificially truncated stub
        # (may time out after only a few steps), not a representative held-out
        # episode. Running one full max_episode_length window lets every env
        # finish that stub and start a naturally phase-shifted episode before
        # we start recording anything.
        for _ in range(max_steps):
            actions = policy(obs, stochastic_output=args_cli.stochastic)
            obs, _, dones, _ = env.step(actions)
            # No-op for non-recurrent (MLP) models; for the LSTM ablation this
            # clears hidden state for envs that just finished an episode, so
            # a new episode doesn't inherit stale hidden state from the last
            # one in that env slot -- matches play.py's loop exactly.
            policy.reset(dones)

        # Collection window: after warm-up, envs are desynchronized, so each
        # env is at most max_steps away from completing its in-flight episode.
        # Keep only the first outcome per env (a clean, full-length episode).
        for _ in range(max_steps):
            actions = policy(obs, stochastic_output=args_cli.stochastic)
            obs, _, dones, extras = env.step(actions)
            policy.reset(dones)
            for outcome in extras.get("episode_outcomes", []):
                env_id = outcome["env_id"]
                if env_id not in collected:
                    collected[env_id] = outcome
            if len(collected) >= num_envs:
                break

    missing = num_envs - len(collected)
    if missing > 0:
        print(
            f"[WARN] Only {len(collected)}/{num_envs} envs produced an outcome within "
            f"{max_steps} steps; {missing} episode(s) will be missing from the CSV. "
            "This should not happen since max_steps == max_episode_length (every env "
            "times out by then if it hasn't already died)."
        )

    os.makedirs(os.path.dirname(args_cli.out_csv), exist_ok=True)
    with open(args_cli.out_csv, "w", newline="") as f:
        fieldnames = ["episode_index"] + REQUIRED_FIELDS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for episode_index, env_id in enumerate(sorted(collected.keys())):
            row = {"episode_index": episode_index}
            row.update({k: collected[env_id][k] for k in REQUIRED_FIELDS})
            writer.writerow(row)

    print(f"[INFO]: Wrote {len(collected)} held-out episodes to {args_cli.out_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
