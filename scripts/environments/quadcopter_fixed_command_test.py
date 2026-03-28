#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run scripted fixed-command checks for the quadcopter low-level controller."""

import argparse
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scripted fixed-command checks for the quadcopter controller.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--profile",
    type=str,
    default="all",
    choices=["all", "hover", "forward", "up", "yaw", "steps", "torque_roll", "torque_pitch", "torque_yaw"],
    help="Which scripted command profile to run.",
)
parser.add_argument("--forward_speed", type=float, default=0.4, help="Forward body-frame velocity target in m/s.")
parser.add_argument("--upward_speed", type=float, default=0.25, help="Upward velocity target in m/s.")
parser.add_argument("--yaw_rate", type=float, default=0.6, help="Yaw-rate target in rad/s.")
parser.add_argument("--hold_time", type=float, default=3.0, help="Hold time for simple command segments in seconds.")
parser.add_argument("--settle_time", type=float, default=1.5, help="Initial hover/settle duration in seconds.")
parser.add_argument("--torque_pulse", type=float, default=0.3, help="Normalized direct-wrench torque pulse amplitude.")
parser.add_argument("--moment_sign_x", type=float, default=1.0, help="Applied roll moment sign.")
parser.add_argument("--moment_sign_y", type=float, default=1.0, help="Applied pitch moment sign.")
parser.add_argument("--moment_sign_z", type=float, default=1.0, help="Applied yaw moment sign.")
parser.add_argument(
    "--keep_action_filter",
    action="store_true",
    default=False,
    help="Keep the env action rate-limit and smoothing enabled.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@dataclass
class CommandPhase:
    name: str
    action: tuple[float, float, float, float]
    duration_s: float


class PhaseMetrics:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_target = torch.zeros(4, device=self.device)
        self.sum_actual = torch.zeros(4, device=self.device)
        self.sum_sq_err = torch.zeros(4, device=self.device)
        self.max_abs_err = torch.zeros(4, device=self.device)

    def update(self, target: torch.Tensor, actual: torch.Tensor):
        # Aggregate mean metrics across vectorized environments for each time step.
        target_mean = target.mean(dim=0)
        actual_mean = actual.mean(dim=0)
        err = actual_mean - target_mean
        self.count += 1
        self.sum_target += target_mean
        self.sum_actual += actual_mean
        self.sum_sq_err += err.square()
        self.max_abs_err = torch.maximum(self.max_abs_err, err.abs())

    def summary_lines(self) -> list[str]:
        if self.count == 0:
            return ["no samples collected"]
        mean_target = self.sum_target / self.count
        mean_actual = self.sum_actual / self.count
        rmse = torch.sqrt(self.sum_sq_err / self.count)
        labels = ("vx_b", "vy_b", "vz_b", "yaw_rate")
        lines = []
        for i, label in enumerate(labels):
            lines.append(
                f"  {label}: target={mean_target[i]: .3f}, actual={mean_actual[i]: .3f}, "
                f"rmse={rmse[i]: .3f}, max_err={self.max_abs_err[i]: .3f}"
            )
        return lines


class DirectWrenchMetrics:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_lin = torch.zeros(3, device=self.device)
        self.sum_ang = torch.zeros(3, device=self.device)
        self.max_abs_ang = torch.zeros(3, device=self.device)

    def update(self, lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor):
        lin_mean = lin_vel_b.mean(dim=0)
        ang_mean = ang_vel_b.mean(dim=0)
        self.count += 1
        self.sum_lin += lin_mean
        self.sum_ang += ang_mean
        self.max_abs_ang = torch.maximum(self.max_abs_ang, ang_mean.abs())

    def summary_lines(self) -> list[str]:
        if self.count == 0:
            return ["no samples collected"]
        mean_lin = self.sum_lin / self.count
        mean_ang = self.sum_ang / self.count
        return [
            f"  lin_vel_b_mean={mean_lin.tolist()}",
            f"  ang_vel_b_mean={mean_ang.tolist()}",
            f"  ang_vel_b_peak_abs={self.max_abs_ang.tolist()}",
        ]


def build_profile() -> list[CommandPhase]:
    hover = CommandPhase("hover_settle", (0.0, 0.0, 0.0, 0.0), args_cli.settle_time)
    if args_cli.profile == "hover":
        return [hover, CommandPhase("hover_hold", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time)]
    if args_cli.profile == "torque_roll":
        return [
            hover,
            CommandPhase("roll_pulse_pos", (0.0, args_cli.torque_pulse, 0.0, 0.0), args_cli.hold_time),
            CommandPhase("hover_after_roll", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time),
        ]
    if args_cli.profile == "torque_pitch":
        return [
            hover,
            CommandPhase("pitch_pulse_pos", (0.0, 0.0, args_cli.torque_pulse, 0.0), args_cli.hold_time),
            CommandPhase("hover_after_pitch", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time),
        ]
    if args_cli.profile == "torque_yaw":
        return [
            hover,
            CommandPhase("yaw_pulse_pos", (0.0, 0.0, 0.0, args_cli.torque_pulse), args_cli.hold_time),
            CommandPhase("hover_after_yaw", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time),
        ]
    if args_cli.profile == "forward":
        return [hover, CommandPhase("forward", (args_cli.forward_speed, 0.0, 0.0, 0.0), args_cli.hold_time)]
    if args_cli.profile == "up":
        return [hover, CommandPhase("upward", (0.0, 0.0, args_cli.upward_speed, 0.0), args_cli.hold_time)]
    if args_cli.profile == "yaw":
        return [hover, CommandPhase("yaw_rate", (0.0, 0.0, 0.0, args_cli.yaw_rate), args_cli.hold_time)]
    if args_cli.profile == "steps":
        recovery_time = 1.0
        return [
            hover,
            CommandPhase("step_forward_pos", (args_cli.forward_speed, 0.0, 0.0, 0.0), args_cli.hold_time),
            CommandPhase("recover_after_forward_pos", (0.0, 0.0, 0.0, 0.0), recovery_time),
            CommandPhase("step_forward_neg", (-args_cli.forward_speed, 0.0, 0.0, 0.0), args_cli.hold_time),
            CommandPhase("recover_after_forward_neg", (0.0, 0.0, 0.0, 0.0), recovery_time),
            CommandPhase("step_up_pos", (0.0, 0.0, args_cli.upward_speed, 0.0), args_cli.hold_time),
            CommandPhase("recover_after_up_pos", (0.0, 0.0, 0.0, 0.0), recovery_time),
            CommandPhase("step_up_zero", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time),
            CommandPhase("recover_after_up_zero", (0.0, 0.0, 0.0, 0.0), recovery_time),
            CommandPhase("step_yaw_pos", (0.0, 0.0, 0.0, args_cli.yaw_rate), args_cli.hold_time),
            CommandPhase("recover_after_yaw_pos", (0.0, 0.0, 0.0, 0.0), recovery_time),
            CommandPhase("step_yaw_neg", (0.0, 0.0, 0.0, -args_cli.yaw_rate), args_cli.hold_time),
        ]
    return [
        hover,
        CommandPhase("forward", (args_cli.forward_speed, 0.0, 0.0, 0.0), args_cli.hold_time),
        CommandPhase("hover_after_forward", (0.0, 0.0, 0.0, 0.0), args_cli.settle_time),
        CommandPhase("upward", (0.0, 0.0, args_cli.upward_speed, 0.0), args_cli.hold_time),
        CommandPhase("hover_after_upward", (0.0, 0.0, 0.0, 0.0), args_cli.settle_time),
        CommandPhase("yaw_rate", (0.0, 0.0, 0.0, args_cli.yaw_rate), args_cli.hold_time),
        CommandPhase("hover_after_yaw", (0.0, 0.0, 0.0, 0.0), args_cli.settle_time),
        CommandPhase("step_forward_pos", (args_cli.forward_speed, 0.0, 0.0, 0.0), args_cli.hold_time),
        CommandPhase("step_forward_zero", (0.0, 0.0, 0.0, 0.0), args_cli.hold_time),
        CommandPhase("step_yaw_neg", (0.0, 0.0, 0.0, -args_cli.yaw_rate), args_cli.hold_time),
    ]


def to_normalized_action(phase: CommandPhase, env_cfg) -> torch.Tensor:
    action = torch.tensor(phase.action, dtype=torch.float32)
    if env_cfg.use_direct_wrench_mode:
        return action.clamp(-1.0, 1.0)
    action[0] /= env_cfg.max_target_speed_xy
    action[1] /= env_cfg.max_target_speed_xy
    action[2] /= env_cfg.max_target_speed_z
    action[3] /= env_cfg.max_target_yaw_rate
    return action.clamp(-1.0, 1.0)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.num_obstacles = 0
    env_cfg.gps_denied_enabled = False
    env_cfg.debug_vis = True
    env_cfg.enable_staggered_resets = False
    env_cfg.moment_sign_x = args_cli.moment_sign_x
    env_cfg.moment_sign_y = args_cli.moment_sign_y
    env_cfg.moment_sign_z = args_cli.moment_sign_z
    env_cfg.use_direct_wrench_mode = args_cli.profile in {"torque_roll", "torque_pitch", "torque_yaw"}
    if not args_cli.keep_action_filter:
        env_cfg.enable_action_rate_limit = False
        env_cfg.enable_action_smoothing = False

    phases = build_profile()
    total_duration_s = sum(phase.duration_s for phase in phases)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, total_duration_s + 5.0)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    step_dt = env.step_dt
    phase_metrics = DirectWrenchMetrics(env.device) if env_cfg.use_direct_wrench_mode else PhaseMetrics(env.device)
    current_phase_index = -1
    current_phase_end_step = 0
    current_action = None

    print("[INFO] Starting controller check")
    print(f"[INFO] Profile: {args_cli.profile}")
    print(f"[INFO] Action filter enabled: {args_cli.keep_action_filter}")
    print(f"[INFO] Obstacles disabled: {env_cfg.num_obstacles == 0}")
    print(f"[INFO] VIO noise disabled: {not env_cfg.gps_denied_enabled}")
    print(
        "[INFO] Moment signs: "
        f"x={env_cfg.moment_sign_x:.1f}, y={env_cfg.moment_sign_y:.1f}, z={env_cfg.moment_sign_z:.1f}"
    )
    print(f"[INFO] Direct wrench mode: {env_cfg.use_direct_wrench_mode}")
    print(f"[INFO] Total duration: {total_duration_s:.2f} s")

    total_steps = int(round(total_duration_s / step_dt))
    phase_start_step = 0

    while simulation_app.is_running() and phase_start_step < total_steps:
        phase_index = 0
        elapsed_s = phase_start_step * step_dt
        running_sum = 0.0
        for idx, phase in enumerate(phases):
            running_sum += phase.duration_s
            if elapsed_s < running_sum:
                phase_index = idx
                break

        if phase_index != current_phase_index:
            if current_phase_index >= 0:
                print(f"[RESULT] {phases[current_phase_index].name}")
                for line in phase_metrics.summary_lines():
                    print(line)
                phase_metrics.reset()
            current_phase_index = phase_index
            current_phase_end_step = int(round(running_sum / step_dt))
            current_action = to_normalized_action(phases[phase_index], env_cfg).to(env.device)
            print(
                f"[PHASE] {phases[phase_index].name}: "
                f"cmd=[{phases[phase_index].action[0]:.2f}, {phases[phase_index].action[1]:.2f}, "
                f"{phases[phase_index].action[2]:.2f}, {phases[phase_index].action[3]:.2f}] "
                f"for {phases[phase_index].duration_s:.2f}s"
            )

        actions = current_action.repeat(env.num_envs, 1)

        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
            if env_cfg.use_direct_wrench_mode:
                phase_metrics.update(env._robot.data.root_lin_vel_b, env._robot.data.root_ang_vel_b)
            else:
                target = torch.zeros(env.num_envs, 4, device=env.device)
                target[:, 0] = current_action[0] * env_cfg.max_target_speed_xy
                target[:, 1] = current_action[1] * env_cfg.max_target_speed_xy
                target[:, 2] = current_action[2] * env_cfg.max_target_speed_z
                target[:, 3] = current_action[3] * env_cfg.max_target_yaw_rate

                actual = torch.zeros_like(target)
                actual[:, :3] = env._robot.data.root_lin_vel_b
                actual[:, 3] = env._robot.data.root_ang_vel_b[:, 2]
                phase_metrics.update(target, actual)

        if bool(torch.any(terminated)) or bool(torch.any(truncated)):
            print("[WARN] Environment terminated before the scripted check finished.")
            print(f"[WARN] Done reason: {env.get_last_done_reason(0)}")
            print(f"[WARN] Pre-reset robot position: {env._last_done_pos_w[0].tolist()}")
            print(f"[WARN] Pre-reset body velocity: {env._last_done_lin_vel_b[0].tolist()}")
            print(f"[WARN] Pre-reset body angular velocity: {env._last_done_ang_vel_b[0].tolist()}")
            print(f"[WARN] Pre-reset upright projection: {env._last_done_up_proj[0].item():.4f}")
            controller_debug = env.get_last_controller_debug(0)
            if controller_debug is not None:
                print(f"[CTRL] action: {controller_debug['action']}")
                print(f"[CTRL] target_vel_b: {controller_debug['target_vel_b']}")
                print(f"[CTRL] target_vel_w: {controller_debug['target_vel_w']}")
                print(f"[CTRL] curr_vel_w: {controller_debug['curr_vel_w']}")
                print(f"[CTRL] curr_ang_vel_b: {controller_debug['curr_ang_vel_b']}")
                print(f"[CTRL] acc_cmd_w: {controller_debug['acc_cmd_w']}")
                print(f"[CTRL] b3_curr_w: {controller_debug['b3_curr_w']}")
                print(f"[CTRL] b3_des_w: {controller_debug['b3_des_w']}")
                print(f"[CTRL] tilt_err_b: {controller_debug['tilt_err_b']}")
                print(f"[CTRL] target_ang_vel_b: {controller_debug['target_ang_vel_b']}")
                print(f"[CTRL] m_tilt: {controller_debug['m_tilt']}")
                print(f"[CTRL] m_rate: {controller_debug['m_rate']}")
                print(f"[CTRL] moments: {controller_debug['moments']}")
                print(f"[CTRL] thrust_mag: {controller_debug['thrust_mag']:.6f}")
                print(f"[CTRL] target_yaw_rate: {controller_debug['target_yaw_rate']:.6f}")
                trace = env.get_last_done_trace(0)
                if trace:
                    print("[TRACE] Last controller samples before failure:")
                    for idx, entry in enumerate(trace):
                        print(
                            "[TRACE] "
                            f"{idx:02d} "
                            f"ang_b={entry['curr_ang_vel_b']} "
                            f"tilt_err_b={entry['tilt_err_b']} "
                            f"m_tilt={entry['m_tilt']} "
                            f"m_rate={entry['m_rate']} "
                            f"mom={entry['moments']} "
                            f"thrust={entry['thrust_mag']:.6f}"
                        )
            break

        phase_start_step += 1
        if phase_start_step >= current_phase_end_step:
            continue

    if current_phase_index >= 0:
        print(f"[RESULT] {phases[current_phase_index].name}")
        for line in phase_metrics.summary_lines():
            print(line)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
