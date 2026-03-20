# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class VelocityGeometricController:
    """Low-level controller: tracks target world velocity + yaw-rate with thrust/moments."""

    def __init__(self, cfg: QuadcopterEnvCfg, device: torch.device, robot_mass: float, robot_weight: float):
        self.cfg = cfg
        self.device = device
        self.robot_mass = robot_mass
        self.robot_weight = robot_weight
        self.g_vec = torch.tensor([0.0, 0.0, 9.81], device=device)

    def compute(
        self,
        curr_vel_w: torch.Tensor,
        curr_quat_w: torch.Tensor,
        curr_ang_vel_b: torch.Tensor,
        target_vel_w: torch.Tensor,
        target_yaw_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Velocity tracking in world frame.
        kv = torch.tensor(
            [self.cfg.ll_kv_xy, self.cfg.ll_kv_xy, self.cfg.ll_kv_z],
            device=self.device,
        )
        acc_cmd_w = kv * (target_vel_w - curr_vel_w) + self.g_vec

        # Enforce tilt-compatible horizontal acceleration.
        max_horiz_acc = math.tan(self.cfg.max_tilt_angle) * 9.81
        horiz_acc = acc_cmd_w[:, :2]
        horiz_norm = torch.linalg.norm(horiz_acc, dim=1, keepdim=True).clamp(min=1.0e-6)
        scale = torch.clamp(max_horiz_acc / horiz_norm, max=1.0)
        acc_cmd_w[:, :2] = horiz_acc * scale

        force_w = self.robot_mass * acc_cmd_w

        # Current rotation matrix from quaternion.
        w, x, y, z = curr_quat_w.unbind(-1)
        r_mat = torch.stack(
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y,
            ],
            dim=-1,
        ).reshape(-1, 3, 3)

        b3_curr_w = r_mat[:, :, 2]
        force_norm = torch.linalg.norm(force_w, dim=1, keepdim=True).clamp(min=1.0e-6)
        b3_des_w = force_w / force_norm

        # Use the force projection on the current body-z axis directly.
        # Full 1 / cos(tilt) compensation over-amplifies thrust while tilted and
        # can turn a recoverable lean into a runaway lateral acceleration.
        thrust_mag = torch.sum(force_w * b3_curr_w, dim=1)

        min_thrust = self.cfg.min_thrust_to_weight * self.robot_weight
        max_thrust = self.cfg.max_thrust_to_weight * self.robot_weight
        thrust_mag = torch.clamp(thrust_mag, min=min_thrust, max=max_thrust)

        # Tilt correction error in body frame.
        tilt_err_w = torch.linalg.cross(b3_curr_w, b3_des_w)
        tilt_err_b = torch.matmul(r_mat.transpose(-2, -1), tilt_err_w.unsqueeze(-1)).squeeze(-1)
        target_ang_vel_b = torch.zeros_like(curr_ang_vel_b)
        target_ang_vel_b[:, 2] = target_yaw_rate.squeeze(-1)
        ang_vel_err = curr_ang_vel_b - target_ang_vel_b

        kR = torch.tensor([self.cfg.ll_kR_xy, self.cfg.ll_kR_xy, self.cfg.ll_kR_z], device=self.device)
        kw = torch.tensor([self.cfg.ll_kw_xy, self.cfg.ll_kw_xy, self.cfg.ll_kw_z], device=self.device)
        moments = -kR * tilt_err_b - kw * ang_vel_err

        moment_limits = torch.tensor(
            [self.cfg.moment_limit_xy, self.cfg.moment_limit_xy, self.cfg.moment_limit_z],
            device=self.device,
        )
        moments = torch.maximum(torch.minimum(moments, moment_limits), -moment_limits)

        thrust = torch.zeros(curr_vel_w.shape[0], 1, 3, device=self.device)
        thrust[:, 0, 2] = thrust_mag
        return thrust, moments.unsqueeze(1)


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    lidar_num_rays = 12
    # lin_vel(3) + ang_vel(3) + gravity(3) + rel_goal(3) + goal_dist(1) + lidar(num_rays)
    observation_space = 13 + lidar_num_rays
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    max_target_speed_xy = 0.70
    max_target_speed_z = 0.40
    max_target_yaw_rate = 1.00
    action_rate_limit = 0.06
    action_smoothing = 0.20
    ll_kv_xy = 2.4
    ll_kv_z = 3.2
    ll_kR_xy = 0.040
    ll_kR_z = 0.020
    ll_kw_xy = 0.010
    ll_kw_z = 0.004
    min_thrust_to_weight = 0.7
    max_thrust_to_weight = 1.90
    moment_limit_xy = 0.006
    moment_limit_z = 0.003
    max_tilt_angle = 0.90

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    progress_to_goal_reward_scale = 8.0
    upright_reward_scale = 2.0
    tilt_penalty_reward_scale = -2.0
    collision_penalty_reward_scale = -12.0
    obstacle_proximity_penalty_reward_scale = -2.5

    # dynamic obstacle settings
    num_obstacles = 3
    obstacle_radius = 0.06
    obstacle_collision_radius = 0.12
    obstacle_xy_bound = 1.7
    obstacle_height_min = 0.5
    obstacle_height_max = 1.3
    obstacle_speed_min = 0.08
    obstacle_speed_max = 0.20

    # lidar settings
    lidar_range_max = 2.5
    lidar_noise_std = 0.01
    lidar_dropout_prob = 0.01

    # lightweight GPS-denied/VIO realism
    gps_denied_enabled = True
    vio_pos_noise_std = 0.01
    vio_vel_noise_std = 0.02
    vio_drift_walk_std = 0.0005


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_goal_distance = torch.zeros(self.num_envs, device=self.device)
        # Dynamic obstacles tracked in tensors
        self._obstacle_pos_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._obstacle_vel_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._obstacle_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._obstacle_quat_w[:, 0] = 1.0
        self._in_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lidar_dirs_b = self._build_lidar_directions_body()
        self._vio_pos_drift_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Backend mission control
        self._backend_target_override = None  # Will be set from backend

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "progress_to_goal",
                "upright",
                "tilt_penalty",
                "collision_penalty",
                "obstacle_proximity_penalty",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._ll_controller = VelocityGeometricController(
            self.cfg,
            self.device,
            self._robot_mass.item(),
            self._robot_weight,
        )

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._obstacles: list[RigidObject] = []

        for obs_idx in range(self.cfg.num_obstacles):
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{obs_idx}",
                spawn=sim_utils.SphereCfg(
                    radius=self.cfg.obstacle_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0)),
            )
            obstacle = RigidObject(obstacle_cfg)
            self.scene.rigid_objects[f"obstacle_{obs_idx}"] = obstacle
            self._obstacles.append(obstacle)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        raw_actions = actions.clone().clamp(-1.0, 1.0)
        action_delta = torch.clamp(raw_actions - self._prev_actions, -self.cfg.action_rate_limit, self.cfg.action_rate_limit)
        self._actions = (1.0 - self.cfg.action_smoothing) * self._prev_actions + self.cfg.action_smoothing * (
            self._prev_actions + action_delta
        )
        self._prev_actions = self._actions.clone()
        target_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        target_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_target_speed_xy
        target_vel_b[:, 1] = self._actions[:, 1] * self.cfg.max_target_speed_xy
        target_vel_b[:, 2] = self._actions[:, 2] * self.cfg.max_target_speed_z
        target_vel_w = quat_apply(self._robot.data.root_quat_w, target_vel_b)
        target_yaw_rate = self._actions[:, 3:4] * self.cfg.max_target_yaw_rate

        self._thrust, self._moment = self._ll_controller.compute(
            curr_vel_w=self._robot.data.root_lin_vel_w,
            curr_quat_w=self._robot.data.root_quat_w,
            curr_ang_vel_b=self._robot.data.root_ang_vel_b,
            target_vel_w=target_vel_w,
            target_yaw_rate=target_yaw_rate,
        )
        self._update_obstacles()

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        est_pos_w, est_lin_vel_b = self._get_navigation_estimate()
        desired_pos_b, _ = subtract_frame_transforms(
            est_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - est_pos_w, dim=1, keepdim=True)
        goal_distance_norm = torch.clamp(distance_to_goal / 5.0, 0.0, 1.0)
        lidar_obs = self._compute_lidar_observations()
        obs = torch.cat(
            [
                est_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                goal_distance_norm,
                lidar_obs,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        progress_to_goal = self._prev_goal_distance - distance_to_goal
        self._prev_goal_distance = distance_to_goal.detach()
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        upright = -self._robot.data.projected_gravity_b[:, 2]
        upright_reward = torch.clamp(upright, 0.0, 1.0)
        tilt_penalty = 1.0 - upright_reward
        collision_penalty = self._compute_collision_penalty()
        obstacle_proximity_penalty = self._compute_obstacle_proximity_penalty()
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "progress_to_goal": progress_to_goal * self.cfg.progress_to_goal_reward_scale,
            "upright": upright_reward * self.cfg.upright_reward_scale * self.step_dt,
            "tilt_penalty": tilt_penalty * self.cfg.tilt_penalty_reward_scale * self.step_dt,
            "collision_penalty": collision_penalty * self.cfg.collision_penalty_reward_scale * self.step_dt,
            "obstacle_proximity_penalty": obstacle_proximity_penalty
            * self.cfg.obstacle_proximity_penalty_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        collision = self._compute_collision_penalty() > 0.0
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        max_tilt_cos = math.cos(self.cfg.max_tilt_angle)
        tipped = (-self._robot.data.projected_gravity_b[:, 2]) < max_tilt_cos
        died = torch.logical_or(died, tipped)
        died = torch.logical_or(died, collision)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid synchronized spikes
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._in_collision[env_ids] = False
        self._vio_pos_drift_w[env_ids] = 0.0
        
        # Sample new goals (or use backend override)
        if self._backend_target_override is not None:
            # Use backend target for all envs
            self._desired_pos_w[env_ids, 0] = self._backend_target_override[0]
            self._desired_pos_w[env_ids, 1] = self._backend_target_override[1]
            self._desired_pos_w[env_ids, 2] = self._backend_target_override[2]
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        else:
            # Random goals (original behavior)
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Sample dynamic obstacles
        self._sample_obstacles(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + 0.6
        default_root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._prev_goal_distance[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _sample_obstacles(self, env_ids: torch.Tensor):
        if self.cfg.num_obstacles == 0:
            return
        num_ids = len(env_ids)
        for obs_idx in range(self.cfg.num_obstacles):
            xy_local = torch.empty(num_ids, 2, device=self.device).uniform_(
                -self.cfg.obstacle_xy_bound, self.cfg.obstacle_xy_bound
            )
            local_norm = torch.linalg.norm(xy_local, dim=1, keepdim=True).clamp(min=1.0e-3)
            too_close = (local_norm < 0.5).squeeze(-1)
            if torch.any(too_close):
                xy_local[too_close] = xy_local[too_close] * (0.5 / local_norm[too_close])
            self._obstacle_pos_w[env_ids, obs_idx, :2] = xy_local + self._terrain.env_origins[env_ids, :2]
            self._obstacle_pos_w[env_ids, obs_idx, 2] = torch.empty(num_ids, device=self.device).uniform_(
                self.cfg.obstacle_height_min, self.cfg.obstacle_height_max
            )

            speed = torch.empty(num_ids, device=self.device).uniform_(self.cfg.obstacle_speed_min, self.cfg.obstacle_speed_max)
            angle = torch.empty(num_ids, device=self.device).uniform_(0.0, 2.0 * torch.pi)
            self._obstacle_vel_w[env_ids, obs_idx, 0] = speed * torch.cos(angle)
            self._obstacle_vel_w[env_ids, obs_idx, 1] = speed * torch.sin(angle)
            self._obstacle_vel_w[env_ids, obs_idx, 2] = 0.0
        self._write_obstacles_to_sim(env_ids)

    def _update_obstacles(self):
        if self.cfg.num_obstacles == 0:
            return
        self._obstacle_pos_w += self._obstacle_vel_w * self.sim.cfg.dt

        local_xy = self._obstacle_pos_w[:, :, :2] - self._terrain.env_origins[:, None, :2]
        hit_upper = local_xy > self.cfg.obstacle_xy_bound
        hit_lower = local_xy < -self.cfg.obstacle_xy_bound
        bounce_mask = torch.logical_or(hit_upper, hit_lower)
        self._obstacle_vel_w[:, :, :2] = torch.where(bounce_mask, -self._obstacle_vel_w[:, :, :2], self._obstacle_vel_w[:, :, :2])
        local_xy = torch.clamp(local_xy, -self.cfg.obstacle_xy_bound, self.cfg.obstacle_xy_bound)
        self._obstacle_pos_w[:, :, :2] = local_xy + self._terrain.env_origins[:, None, :2]
        self._write_obstacles_to_sim()

    def _compute_collision_penalty(self) -> torch.Tensor:
        if self.cfg.num_obstacles == 0:
            self._in_collision[:] = False
            return torch.zeros(self.num_envs, device=self.device)

        rel = self._obstacle_pos_w - self._robot.data.root_pos_w[:, None, :]
        dists = torch.linalg.norm(rel, dim=-1)
        min_dist = torch.min(dists, dim=1).values
        self._in_collision = min_dist < self.cfg.obstacle_collision_radius
        return self._in_collision.float()

    def _compute_obstacle_proximity_penalty(self) -> torch.Tensor:
        if self.cfg.num_obstacles == 0:
            return torch.zeros(self.num_envs, device=self.device)
        rel = self._obstacle_pos_w - self._robot.data.root_pos_w[:, None, :]
        min_dist = torch.min(torch.linalg.norm(rel, dim=-1), dim=1).values
        safety_distance = self.cfg.obstacle_collision_radius * 2.0
        return torch.clamp((safety_distance - min_dist) / safety_distance, 0.0, 1.0)

    def _compute_lidar_observations(self) -> torch.Tensor:
        # Ray-sphere intersections against dynamic rigid obstacles in body frame.
        max_range = self.cfg.lidar_range_max
        if self.cfg.num_obstacles == 0:
            return torch.ones(self.num_envs, self.cfg.lidar_num_rays, device=self.device)

        rel_b_all = torch.empty(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        for obs_idx in range(self.cfg.num_obstacles):
            rel_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._obstacle_pos_w[:, obs_idx, :],
            )
            rel_b_all[:, obs_idx, :] = rel_b

        # ray: p(t)=t*d, sphere center o, radius r => t^2 - 2*(d.o)*t + (o.o-r^2)=0
        d = self._lidar_dirs_b[None, :, None, :]  # (1, R, 1, 3)
        o = rel_b_all[:, None, :, :]  # (N, 1, O, 3)
        b = torch.sum(d * o, dim=-1)  # (N, R, O)
        c = torch.sum(o * o, dim=-1) - (self.cfg.obstacle_radius**2)  # (N, 1, O)
        disc = b * b - c
        valid = disc >= 0.0
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t1 = b - sqrt_disc
        t2 = b + sqrt_disc
        inf = torch.full_like(t1, float("inf"))
        t_hit = torch.where(valid & (t1 > 0.0), t1, torch.where(valid & (t2 > 0.0), t2, inf))

        distances = torch.min(t_hit, dim=-1).values
        distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, max_range))
        distances = torch.clamp(distances, 0.0, max_range)
        lidar_obs = distances / max_range
        if self.cfg.lidar_noise_std > 0.0:
            lidar_obs = lidar_obs + torch.randn_like(lidar_obs) * self.cfg.lidar_noise_std
        if self.cfg.lidar_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(lidar_obs) < self.cfg.lidar_dropout_prob
            lidar_obs = torch.where(dropout_mask, torch.ones_like(lidar_obs), lidar_obs)
        return torch.clamp(lidar_obs, 0.0, 1.0)

    def _build_lidar_directions_body(self) -> torch.Tensor:
        # 12-ray pattern: 8 horizontal + 2 vertical + 2 diagonal.
        dirs = []
        for yaw_deg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            yaw = math.radians(yaw_deg)
            dirs.append(torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=self.device))
        dirs.append(torch.tensor([0.0, 0.0, 1.0], device=self.device))
        dirs.append(torch.tensor([0.0, 0.0, -1.0], device=self.device))
        dirs.append(torch.tensor([0.70710677, 0.0, 0.70710677], device=self.device))
        dirs.append(torch.tensor([0.70710677, 0.0, -0.70710677], device=self.device))
        lidar_dirs = torch.stack(dirs, dim=0)
        return lidar_dirs[: self.cfg.lidar_num_rays]

    def _write_obstacles_to_sim(self, env_ids: torch.Tensor | None = None):
        if self.cfg.num_obstacles == 0:
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        for obs_idx, obstacle in enumerate(self._obstacles):
            pose = torch.cat(
                [self._obstacle_pos_w[env_ids, obs_idx, :], self._obstacle_quat_w[env_ids, :]],
                dim=-1,
            )
            root_vel = torch.zeros(len(env_ids), 6, device=self.device)
            root_vel[:, :3] = self._obstacle_vel_w[env_ids, obs_idx, :]
            obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
            obstacle.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    def _get_navigation_estimate(self) -> tuple[torch.Tensor, torch.Tensor]:
        # GPS-denied estimate: pos/vel use light VIO noise + drift.
        if not self.cfg.gps_denied_enabled:
            return self._robot.data.root_pos_w, self._robot.data.root_lin_vel_b

        self._vio_pos_drift_w += (
            torch.randn_like(self._vio_pos_drift_w) * self.cfg.vio_drift_walk_std * math.sqrt(self.step_dt)
        )
        self._vio_pos_drift_w = torch.clamp(self._vio_pos_drift_w, min=-0.25, max=0.25)

        pos_noise = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.vio_pos_noise_std
        vel_noise = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.vio_vel_noise_std

        est_pos_w = self._robot.data.root_pos_w + self._vio_pos_drift_w + pos_noise
        est_lin_vel_b = self._robot.data.root_lin_vel_b + vel_noise
        return est_pos_w, est_lin_vel_b

    def set_target_position(self, x: float, y: float, z: float):
        """Set target position from external source (e.g., backend API).
        
        Args:
            x: Target X coordinate (local to env origin)
            y: Target Y coordinate (local to env origin)
            z: Target Z coordinate (absolute height)
        """
        self._backend_target_override = (x, y, z)
        # Update all current environments immediately
        self._desired_pos_w[:, 0] = x
        self._desired_pos_w[:, 1] = y
        self._desired_pos_w[:, 2] = z
        # Add terrain origins for proper world positioning
        self._desired_pos_w[:, :2] += self._terrain.env_origins[:, :2]
        print(f"[ENV] Target updated from backend: ({x:.2f}, {y:.2f}, {z:.2f})")
