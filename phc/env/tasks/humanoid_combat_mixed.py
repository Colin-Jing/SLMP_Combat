import os
from typing import List, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from rl_games.algos_torch import torch_ext

from phc.env.tasks.humanoid import Humanoid, compute_humanoid_observations_max
from phc.env.tasks.humanoid_combat import compute_humanoid_reset_in_reward
from phc.learning.network_loader import load_combat_prior
from phc.utils import torch_utils
from phc.utils.torch_utils import project_to_norm
from phc.utils.flags import flags


G1_HAND_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]
G1_TARGET_NAMES = ["pelvis", "torso_link", "left_hip_roll_link", "right_hip_roll_link"]
G1_FOOT_NAMES = [
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]

PM01_HAND_NAMES = [
    "link_elbow_yaw_l",
    "link_elbow_yaw_r",
    "link_knee_pitch_l",
    "link_ankle_roll_l",
    "link_knee_pitch_r",
    "link_ankle_roll_r",
]
PM01_TARGET_NAMES = [
    "link_base",
    "link_torso_yaw",
    "link_head_yaw",
    "link_hip_yaw_l",
    "link_hip_yaw_r",
    
]
PM01_FOOT_NAMES = [
    "link_ankle_pitch_l",
    "link_ankle_roll_l",
    "link_ankle_pitch_r",
    "link_ankle_roll_r",
]


def compute_combat_observation_g1(
    self_root_state,
    self_body_pos,
    oppo_root_state,
    oppo_body_pos,
    oppo_body_rot,
    oppo_dof_pos,
    oppo_dof_vel,
    self_contact_norm,
    oppo_contact_norm,
    hand_ids,
    target_ids,
):
    self_root_pos = self_root_state[:, 0:3]
    self_root_rot = self_root_state[:, 3:7]
    oppo_root_pos = oppo_root_state[:, 0:3]
    oppo_root_rot = oppo_root_state[:, 3:7]
    oppo_root_vel = oppo_root_state[:, 7:10]
    oppo_root_ang_vel = oppo_root_state[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(self_root_rot)
    root_pos_diff = oppo_root_pos - self_root_pos
    root_pos_diff[..., -1] = oppo_root_pos[..., -1]
    local_root_pos_diff = torch_utils.quat_rotate(heading_rot, root_pos_diff)

    local_tar_rot = torch_utils.quat_mul(heading_rot, oppo_root_rot)
    local_root_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    local_oppo_vel = torch_utils.quat_rotate(heading_rot, oppo_root_vel)
    local_oppo_ang_vel = torch_utils.quat_rotate(heading_rot, oppo_root_ang_vel)
    oppo_dof_obs = oppo_dof_pos

    oppo_body_pos_diff = oppo_body_pos - self_root_pos[:, None]
    flat_oppo_body_pos_diff = oppo_body_pos_diff.view(
        oppo_body_pos_diff.shape[0] * oppo_body_pos_diff.shape[1], oppo_body_pos_diff.shape[2]
    )
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, oppo_body_pos_diff.shape[1], 1))
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2]
    )
    local_flat_oppo_body_pos_diff = torch_utils.quat_rotate(flat_heading_rot, flat_oppo_body_pos_diff)
    local_flat_oppo_body_pos_diff = local_flat_oppo_body_pos_diff.view(
        oppo_body_pos_diff.shape[0], oppo_body_pos_diff.shape[1] * oppo_body_pos_diff.shape[2]
    )

    flat_oppo_body_rot = oppo_body_rot.view(oppo_body_rot.shape[0] * oppo_body_rot.shape[1], oppo_body_rot.shape[2])
    local_flat_oppo_body_rot = torch_utils.quat_mul(flat_heading_rot, flat_oppo_body_rot)
    local_flat_oppo_body_rot = torch_utils.quat_to_tan_norm(local_flat_oppo_body_rot)
    local_flat_oppo_body_rot = local_flat_oppo_body_rot.view(oppo_body_rot.shape[0], oppo_body_rot.shape[1] * 6)

    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot2 = heading_rot.unsqueeze(-2).repeat(
        (1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1)
    )
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot2.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot2.shape[0], -1)

    return torch.cat(
        [
            local_root_pos_diff,
            local_root_rot_obs,
            local_oppo_vel,
            local_oppo_ang_vel,
            oppo_dof_obs,
            oppo_dof_vel,
            local_flat_oppo_body_pos_diff,
            local_flat_oppo_body_rot,
            local_target_hand_pos_diff,
            self_contact_norm,
            oppo_contact_norm,
        ],
        dim=-1,
    )


def compute_combat_observation_pm01(
    self_root_state,
    self_body_pos,
    oppo_root_state,
    oppo_body_pos,
    oppo_body_rot,
    oppo_dof_pos,
    oppo_dof_vel,
    self_contact_norm,
    oppo_contact_norm,
    hand_ids,
    target_ids,
):
    return compute_combat_observation_g1(
        self_root_state,
        self_body_pos,
        oppo_root_state,
        oppo_body_pos,
        oppo_body_rot,
        oppo_dof_pos,
        oppo_dof_vel,
        self_contact_norm,
        oppo_contact_norm,
        hand_ids,
        target_ids,
    )


def compute_combat_reward_mixed_ids(
    self_terminated: torch.Tensor,
    oppo_terminated: torch.Tensor,
    self_body_pos: torch.Tensor,
    oppo_body_pos: torch.Tensor,
    self_force_norm: torch.Tensor,
    oppo_force_norm: torch.Tensor,
    self_hand_ids: torch.Tensor,
    self_target_ids: torch.Tensor,
    oppo_hand_ids: torch.Tensor,
    oppo_target_ids: torch.Tensor,
    reward_weights: dict,
):
    reward_s = reward_weights["reward_s"]
    reward_t = reward_weights["reward_t"]
    hit_dist_thresh = reward_weights["hit_dist_thresh"]

    terminate_reward = oppo_terminated - 1.0 * self_terminated

    self_target_weights = torch.ones(len(self_target_ids), device=self_body_pos.device)
    if len(self_target_ids) >= 2:
        self_target_weights[-2:] = 1.0

    oppo_target_weights = torch.ones(len(oppo_target_ids), device=self_body_pos.device)
    if len(oppo_target_ids) >= 2:
        oppo_target_weights[-2:] = 1.0

    self_hand_pos = self_body_pos[:, self_hand_ids, :]
    oppo_target_pos = oppo_body_pos[:, oppo_target_ids, :]
    self_to_oppo_dist = torch.linalg.norm(self_hand_pos.unsqueeze(2) - oppo_target_pos.unsqueeze(1), dim=-1)
    oppo_target_close = (self_to_oppo_dist < hit_dist_thresh).any(dim=1).float()
    strike_reward = (oppo_force_norm[:, oppo_target_ids] * oppo_target_close * oppo_target_weights).sum(dim=-1)

    oppo_hand_pos = oppo_body_pos[:, oppo_hand_ids, :]
    self_target_pos = self_body_pos[:, self_target_ids, :]
    oppo_to_self_dist = torch.linalg.norm(oppo_hand_pos.unsqueeze(2) - self_target_pos.unsqueeze(1), dim=-1)
    self_target_close = (oppo_to_self_dist < hit_dist_thresh).any(dim=1).float()
    got_hit_reward = (self_force_norm[:, self_target_ids] * self_target_close * self_target_weights).sum(dim=-1)

    strike_reward = strike_reward - 1.0 * got_hit_reward
    reward = strike_reward * reward_s + terminate_reward * reward_t
    reward_raw = torch.stack([strike_reward, terminate_reward], dim=-1)
    return reward, reward_raw


class HumanoidCombatMixedPlay(Humanoid):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

    def _resolve_cfg_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base_dir = os.environ.get("HYDRA_ORIG_CWD", os.getcwd())
        return os.path.join(base_dir, path)

    def _load_robot_cfg_file(self, path: str) -> EasyDict:
        cfg_path = self._resolve_cfg_path(path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Robot config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return EasyDict(data)

    def _get_hand_target_names(self, humanoid_type: str) -> Tuple[List[str], List[str]]:
        if humanoid_type == "g1":
            return G1_HAND_NAMES, G1_TARGET_NAMES
        if humanoid_type == "pm01":
            return PM01_HAND_NAMES, PM01_TARGET_NAMES
        raise ValueError(f"Unsupported humanoid_type: {humanoid_type}")

    @staticmethod
    def _compute_task_obs_size(
        self_num_bodies: int, oppo_num_bodies: int, oppo_num_dofs: int, self_num_hand: int, oppo_num_target: int
    ) -> int:
        return 15 + 2 * oppo_num_dofs + 10 * oppo_num_bodies + self_num_bodies + 3 * self_num_hand * oppo_num_target

    def _reset_default_state(self, env_ids):
        for i in range(self.num_agents):
            self._humanoid_root_states_list[i][env_ids] = self._initial_humanoid_root_states_list[i][env_ids]
            self._dof_pos_list[i][env_ids] = self._initial_dof_pos_list[i][env_ids]
            self._dof_vel_list[i][env_ids] = self._initial_dof_vel_list[i][env_ids]

    def _get_spawn_xy(self, agent_idx: int) -> Tuple[float, float]:
        cfg_pos = self.cfg["env"]["agent_spawn_positions"]
        return float(cfg_pos[agent_idx][0]), float(cfg_pos[agent_idx][1])

    def _get_spawn_quat_zw(self, agent_idx: int) -> Tuple[float, float]:
        cfg_heading = self.cfg["env"]["agent_spawn_headings"]
        yaw = float(cfg_heading[agent_idx])
        half = 0.5 * yaw
        return float(np.sin(half)), float(np.cos(half))

    def load_humanoid_configs(self, cfg):
        robot_cfg_paths = list(cfg["env"]["agent_robot_cfgs"])
        if len(robot_cfg_paths) != self.num_agents:
            raise ValueError(
                f"env.agent_robot_cfgs must have {self.num_agents} entries, got {len(robot_cfg_paths)}"
            )

        self.agent_robot_cfgs = [self._load_robot_cfg_file(p) for p in robot_cfg_paths]
        self.agent_types = [rc.humanoid_type for rc in self.agent_robot_cfgs]

        cfg_primary = EasyDict({"env": cfg["env"], "robot": self.agent_robot_cfgs[0]})
        self.humanoid_type = self.agent_types[0]
        self.load_common_humanoid_configs(cfg_primary)

        self._has_upright_start = self.agent_robot_cfgs[0].get("has_upright_start", True)
        self._real_weight = True
        self._body_names_orig = list(self.agent_robot_cfgs[0].get("body_names", []))
        self._body_names = self._body_names_orig
        self._dof_names = list(self.agent_robot_cfgs[0].get("dof_names", []))
        self.limb_weight_group = []
        self.dof_subset = torch.tensor([]).long()

        self.agent_body_names = [list(rc.get("body_names", [])) for rc in self.agent_robot_cfgs]
        self.agent_dof_names = [list(rc.get("dof_names", [])) for rc in self.agent_robot_cfgs]
        self.agent_hand_names = []
        self.agent_target_names = []
        self.agent_self_obs_sizes = []
        for i in range(self.num_agents):
            hand_names, target_names = self._get_hand_target_names(self.agent_types[i])
            self.agent_hand_names.append(hand_names)
            self.agent_target_names.append(target_names)

            num_bodies = len(self.agent_body_names[i])
            self_obs_size = 1 + num_bodies * (3 + 6 + 3 + 3) - 3
            self.agent_self_obs_sizes.append(self_obs_size)

        self.agent_task_obs_sizes = []
        self.agent_total_obs_sizes = []
        for i in range(self.num_agents):
            source_idx = (i + 1) % self.num_agents
            task_obs_size = self._compute_task_obs_size(
                self_num_bodies=len(self.agent_body_names[i]),
                oppo_num_bodies=len(self.agent_body_names[source_idx]),
                oppo_num_dofs=len(self.agent_dof_names[source_idx]),
                self_num_hand=len(self.agent_hand_names[i]),
                oppo_num_target=len(self.agent_target_names[source_idx]),
            )
            self.agent_task_obs_sizes.append(task_obs_size)
            self.agent_total_obs_sizes.append(self.agent_self_obs_sizes[i] + task_obs_size)

        self._num_self_obs = max(self.agent_self_obs_sizes)
        self._dof_size = sum(len(d) for d in self.agent_dof_names)
        self._num_actions = self._dof_size

    def _setup_character_props(self, key_bodies):
        self._dof_body_ids = np.arange(1, len(self._body_names))
        self._dof_offsets = list(range(len(self._dof_names) + 1))
        self._dof_obs_size = len(self._dof_names)
        if not self._root_height_obs:
            self._num_self_obs -= 1
        return

    def get_obs_size(self):
        return max(self.agent_total_obs_sizes)

    def get_task_obs_size(self):
        return max(self.agent_task_obs_sizes)

    def get_task_obs_size_detail(self):
        return {}

    def get_agent_obs_sizes(self):
        return list(self.agent_total_obs_sizes)

    def get_self_obs_size(self):
        return max(self.agent_self_obs_sizes)

    def _build_g1_pd_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = to_torch(
            [
                100.0, 100.0, 100.0, 200.0, 20.0, 20.0,
                100.0, 100.0, 100.0, 200.0, 20.0, 20.0,
                400.0, 400.0, 400.0,
                90.0, 60.0, 20.0, 60.0, 4.0, 4.0, 4.0,
                90.0, 60.0, 20.0, 60.0, 4.0, 4.0, 4.0,
            ],
            device=self.device,
        )
        d = to_torch(
            [
                2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
                2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
                5.0, 5.0, 5.0,
                2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,
                2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,
            ],
            device=self.device,
        )
        t = to_torch(
            [
                88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
            ],
            device=self.device,
        )
        default_pos = torch.tensor(
            [
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            device=self.device,
            dtype=torch.float,
        )
        return p, d, t, default_pos

    def _build_pm01_pd_params(self, dof_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dofs = len(dof_names)
        p = torch.zeros(num_dofs, device=self.device)
        d = torch.zeros(num_dofs, device=self.device)
        default_pos = torch.zeros(num_dofs, device=self.device)

        torque_limits = to_torch(
            [
                164.0, 164.0, 52.0, 164.0, 52.0, 52.0,
                164.0, 164.0, 52.0, 164.0, 52.0, 52.0,
                52.0, 52.0, 52.0, 52.0, 52.0, 52.0,
                52.0, 52.0, 52.0, 52.0, 52.0, 52.0,
            ],
            device=self.device,
        )

        for dof_idx, name in enumerate(dof_names):
            if "hip_pitch" in name:
                p[dof_idx], d[dof_idx], default_pos[dof_idx] = 100.0, 2.5, -0.24
            elif "hip_roll" in name:
                p[dof_idx], d[dof_idx] = 100.0, 2.5
            elif "hip_yaw" in name:
                p[dof_idx], d[dof_idx] = 100.0, 2.5
            elif "knee_pitch" in name:
                p[dof_idx], d[dof_idx], default_pos[dof_idx] = 200.0, 5.0, 0.48
            elif "ankle_pitch" in name:
                p[dof_idx], d[dof_idx], default_pos[dof_idx] = 35.0, 0.6, -0.24
            elif "ankle_roll" in name:
                p[dof_idx], d[dof_idx] = 35.0, 0.6
            elif "waist" in name:
                p[dof_idx], d[dof_idx] = 400.0, 5.0
            elif "shoulder_pitch" in name:
                p[dof_idx], d[dof_idx] = 90.0, 2.0
            elif "shoulder_roll" in name:
                p[dof_idx], d[dof_idx] = 60.0, 1.0
            elif "shoulder_yaw" in name:
                p[dof_idx], d[dof_idx] = 30.0, 0.8
            elif "elbow_pitch" in name:
                p[dof_idx], d[dof_idx] = 60.0, 1.2
            elif "elbow_yaw" in name:
                p[dof_idx], d[dof_idx] = 50.0, 1.0
            elif "shoulder" in name:
                p[dof_idx], d[dof_idx] = 40.0, 1.0
            elif "elbow" in name:
                p[dof_idx], d[dof_idx] = 40.0, 1.0
            else:
                p[dof_idx], d[dof_idx] = 20.0, 0.5

        return p, d, torque_limits, default_pos

    def _load_combat_ring_asset(self):
        env_cfg = self.cfg["env"]
        ring_asset_cfg = env_cfg["ring_asset"]
        asset_root = ring_asset_cfg["asset_root"]
        asset_file = ring_asset_cfg["asset_file"]
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = float(ring_asset_cfg["angular_damping"])
        asset_options.linear_damping = float(ring_asset_cfg["linear_damping"])
        asset_options.max_angular_velocity = float(ring_asset_cfg["max_angular_velocity"])
        asset_options.density = float(ring_asset_cfg["density"])
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._combat_ring_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _build_combat_ring(self, env_id, env_ptr):
        env_cfg = self.cfg["env"]
        default_pose = gymapi.Transform()
        default_pose.p.z = float(env_cfg["ring_pose_z"])
        handle = self.gym.create_actor(env_ptr, self._combat_ring_asset, default_pose, "combat_ring", env_id, -1, 0)
        ring_color = env_cfg["ring_color"]
        self.gym.set_rigid_body_color(
            env_ptr,
            handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(float(ring_color[0]), float(ring_color[1]), float(ring_color[2])),
        )
        self._combat_ring_handles.append(handle)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.humanoid_assets_list = []
        self.agent_assets = []
        self.agent_num_bodies = []
        self.agent_num_dofs = []
        self.agent_dof_limits_lower = []
        self.agent_dof_limits_upper = []
        self.agent_p_gains = []
        self.agent_d_gains = []
        self.agent_torque_limits = []
        self.agent_default_dof_pos = []
        self.agent_motor_efforts = []
        self.agent_torso_indices = []
        self.agent_skeleton_trees = []
        self.humanoid_masses = []
        self.humanoid_limb_and_weights = []

        for i, robot_cfg in enumerate(self.agent_robot_cfgs):
            asset_root = robot_cfg.asset["assetRoot"]
            asset_file = robot_cfg.asset["assetFileName"]
            robot_file = os.path.join(asset_root, robot_cfg.asset.get("urdfFileName", asset_file))
            xml_asset_path = os.path.join(asset_root, asset_file)

            self.agent_skeleton_trees.append(SkeletonTree.from_mjcf(xml_asset_path))

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.replace_cylinder_with_capsule = True
            asset_options.collapse_fixed_joints = True

            humanoid_asset = self.gym.load_asset(
                self.sim, os.path.dirname(robot_file), os.path.basename(robot_file), asset_options
            )
            self.agent_assets.append(humanoid_asset)
            self.agent_num_bodies.append(self.gym.get_asset_rigid_body_count(humanoid_asset))
            self.agent_num_dofs.append(self.gym.get_asset_dof_count(humanoid_asset))

            right_foot_name = robot_cfg.get("right_foot_name", "")
            left_foot_name = robot_cfg.get("left_foot_name", "")
            sensor_pose = gymapi.Transform()
            for foot_name in [right_foot_name, left_foot_name]:
                foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, foot_name)
                if foot_idx >= 0:
                    self.gym.create_asset_force_sensor(humanoid_asset, foot_idx, sensor_pose)

            if self.agent_types[i] == "g1":
                p_gains, d_gains, torque_limits, default_dof_pos = self._build_g1_pd_params()
                motor_efforts = [360.0] * len(self.agent_dof_names[i])
            else:
                p_gains, d_gains, torque_limits, default_dof_pos = self._build_pm01_pd_params(
                    self.agent_dof_names[i]
                )
                actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                motor_efforts = [prop.motor_effort for prop in actuator_props]

            self.agent_p_gains.append(p_gains)
            self.agent_d_gains.append(d_gains)
            self.agent_torque_limits.append(torque_limits)
            self.agent_default_dof_pos.append(default_dof_pos)
            self.agent_motor_efforts.append(motor_efforts)

            torso_name = "torso_link" if self.agent_types[i] == "g1" else "link_torso_yaw"
            self.agent_torso_indices.append(self.agent_body_names[i].index(torso_name) if torso_name in self.agent_body_names[i] else 0)

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            lim_low = []
            lim_up = []
            for j in range(len(dof_prop)):
                l = dof_prop["lower"][j]
                u = dof_prop["upper"][j]
                if l > u:
                    l, u = u, l
                lim_low.append(l)
                lim_up.append(u)
            self.agent_dof_limits_lower.append(to_torch(lim_low, device=self.device))
            self.agent_dof_limits_upper.append(to_torch(lim_up, device=self.device))

        self.agent_dof_offsets = [0]
        self.agent_body_offsets = [0]
        for n in self.agent_num_dofs:
            self.agent_dof_offsets.append(self.agent_dof_offsets[-1] + n)
        for n in self.agent_num_bodies:
            self.agent_body_offsets.append(self.agent_body_offsets[-1] + n)

        self.total_num_dof = self.agent_dof_offsets[-1]
        self.total_num_bodies = self.agent_body_offsets[-1]

        self.default_dof_pos_total = torch.cat(self.agent_default_dof_pos, dim=0).unsqueeze(0)
        self.kp_gains_total = torch.cat(self.agent_p_gains, dim=0)
        self.kd_gains_total = torch.cat(self.agent_d_gains, dim=0)
        self.torque_limits_total = torch.cat(self.agent_torque_limits, dim=0)
        self.dof_limits_lower = torch.cat(self.agent_dof_limits_lower, dim=0)
        self.dof_limits_upper = torch.cat(self.agent_dof_limits_upper, dim=0)
        self.motor_efforts = to_torch(
            [v for efforts in self.agent_motor_efforts for v in efforts], device=self.device
        )
        self.max_motor_effort = float(max([max(v) for v in self.agent_motor_efforts]))

        self.num_bodies = self.total_num_bodies
        self.num_dof = self.total_num_dof
        self.num_asset_joints = self.total_num_dof
        self.kp_gains = self.kp_gains_total
        self.kd_gains = self.kd_gains_total
        self.default_dof_pos = self.default_dof_pos_total
        self.torque_limits = self.torque_limits_total
        self._pd_action_offset = torch.zeros(self.total_num_dof, device=self.device)
        self._pd_action_scale = torch.ones(self.total_num_dof, device=self.device)

        self.humanoid_shapes = torch.zeros((num_envs, 10), device=self.device, dtype=torch.float)
        self.humanoid_assets_list = [self.agent_assets for _ in range(num_envs)]
        self.humanoid_handles_list = []
        self.envs = []

        self._combat_ring_handles = []
        self._load_combat_ring_asset()

        for env_id in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(env_id, env_ptr, self.humanoid_assets_list[env_id])
            self._build_combat_ring(env_id, env_ptr)
            self.envs.append(env_ptr)

        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)

    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        col_group = env_id
        char_h = float(self.cfg["env"]["spawn_height"])

        humanoid_handles = []
        total_mass = 0.0
        for i in range(self.num_agents):
            start_pose = gymapi.Transform()
            spawn_x, spawn_y = self._get_spawn_xy(i)
            spawn_qz, spawn_qw = self._get_spawn_quat_zw(i)
            start_pose.p = gymapi.Vec3(spawn_x, spawn_y, char_h)
            start_pose.r = gymapi.Quat(0.0, 0.0, spawn_qz, spawn_qw)

            actor_col_filter = 1 << i if not self._has_self_collision else 0
            humanoid_handle = self.gym.create_actor(
                env_ptr, humanoid_asset_list[i], start_pose, "humanoid", col_group, actor_col_filter, 0
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

            color = gymapi.Vec3(0.54, 0.85, 0.2) if i % 2 == 0 else gymapi.Vec3(0.97, 0.38, 0.06)
            torso_idx = self.agent_torso_indices[i]
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, torso_idx, gymapi.MESH_VISUAL, color)

            if i % 2 != 0:
                head_idx = int(self.cfg["env"]["odd_agent_head_body_index"])
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, head_idx, gymapi.MESH_VISUAL, color)

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset_list[i])
            dof_prop["driveMode"][:] = gymapi.DOF_MODE_EFFORT
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

            mass_ind = [p.mass for p in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
            total_mass += float(np.sum(mass_ind))
            humanoid_handles.append(humanoid_handle)

        self.humanoid_handles_list.append(humanoid_handles)
        self.humanoid_masses.append(total_mass)
        self.humanoid_limb_and_weights.append(torch.tensor([1.0, total_mass], device=self.device))

    def _build_termination_heights(self):
        termination_height = float(self.cfg["env"]["terminationHeight"])
        self._termination_heights_list = [
            torch.full((nb,), termination_height, device=self.device) for nb in self.agent_num_bodies
        ]
        self._termination_heights = self._termination_heights_list[0]
        self._termination_root_heights = torch.full(
            (self.num_agents,), termination_height, dtype=torch.float, device=self.device
        )

    def _setup_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.total_num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)

        num_actors = self.get_num_actors_per_env()
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        dof_state_reshaped = self._dof_state.view(self.num_envs, dofs_per_env, 2)
        contact_force_reshaped = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)

        self._humanoid_root_states_list = []
        self._initial_humanoid_root_states_list = []
        self._humanoid_actor_ids_list = []
        self._dof_pos_list = []
        self._dof_vel_list = []
        self._initial_dof_pos_list = []
        self._initial_dof_vel_list = []
        self._rigid_body_pos_list = []
        self._rigid_body_rot_list = []
        self._rigid_body_vel_list = []
        self._rigid_body_ang_vel_list = []
        self._contact_forces_list = []

        self._rigid_body_pos_hist_list = []
        self._rigid_body_rot_hist_list = []
        self._rigid_body_vel_hist_list = []
        self._rigid_body_ang_vel_hist_list = []

        self._reset_rb_pos_list = [torch.zeros(0, device=self.device)] * self.num_agents
        self._reset_rb_rot_list = [torch.zeros(0, device=self.device)] * self.num_agents
        self._reset_rb_vel_list = [torch.zeros(0, device=self.device)] * self.num_agents
        self._reset_rb_ang_vel_list = [torch.zeros(0, device=self.device)] * self.num_agents

        for i in range(self.num_agents):
            root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., i, :]
            self._humanoid_root_states_list.append(root_states)

            initial_root = root_states.clone()
            initial_root[:, 7:13] = 0
            spawn_x, spawn_y = self._get_spawn_xy(i)
            spawn_qz, spawn_qw = self._get_spawn_quat_zw(i)
            initial_root[..., 0] = spawn_x
            initial_root[..., 1] = spawn_y
            initial_root[..., 3] = 0.0
            initial_root[..., 4] = 0.0
            initial_root[..., 5] = spawn_qz
            initial_root[..., 6] = spawn_qw
            self._initial_humanoid_root_states_list.append(initial_root)
            self._humanoid_actor_ids_list.append(
                num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + i
            )

            d0, d1 = self.agent_dof_offsets[i], self.agent_dof_offsets[i + 1]
            dof_pos = dof_state_reshaped[..., d0:d1, 0]
            dof_vel = dof_state_reshaped[..., d0:d1, 1]
            self._dof_pos_list.append(dof_pos)
            self._dof_vel_list.append(dof_vel)

            init_dof_pos = self.agent_default_dof_pos[i].unsqueeze(0).repeat(self.num_envs, 1).clone()
            init_dof_vel = torch.zeros_like(dof_vel, device=self.device, dtype=torch.float)
            self._initial_dof_pos_list.append(init_dof_pos)
            self._initial_dof_vel_list.append(init_dof_vel)

            b0, b1 = self.agent_body_offsets[i], self.agent_body_offsets[i + 1]
            self._rigid_body_pos_list.append(rigid_body_state_reshaped[..., b0:b1, 0:3])
            self._rigid_body_rot_list.append(rigid_body_state_reshaped[..., b0:b1, 3:7])
            self._rigid_body_vel_list.append(rigid_body_state_reshaped[..., b0:b1, 7:10])
            self._rigid_body_ang_vel_list.append(rigid_body_state_reshaped[..., b0:b1, 10:13])
            self._contact_forces_list.append(contact_force_reshaped[..., b0:b1, :])

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._build_termination_heights()
        self.allocate_buffers()

        if self.viewer is not None or flags.server_mode:
            self._init_camera()

        self.agent_contact_body_ids = []
        agent_contact_cfg = self.cfg["env"]["agent_contact_bodies"]
        for i in range(self.num_agents):
            names = list(agent_contact_cfg[i])
            body_names = self.agent_body_names[i]
            ids = [body_names.index(n) for n in names if n in body_names]
            self.agent_contact_body_ids.append(torch.tensor(ids, device=self.device, dtype=torch.long))

        self.agent_hand_ids = []
        self.agent_target_ids = []
        for i in range(self.num_agents):
            body_names = self.agent_body_names[i]
            hand_ids = [body_names.index(n) for n in self.agent_hand_names[i] if n in body_names]
            target_ids = [body_names.index(n) for n in self.agent_target_names[i] if n in body_names]
            if len(hand_ids) == 0 or len(target_ids) == 0:
                raise ValueError(
                    f"Invalid hand/target mapping for agent {i} ({self.agent_types[i]}): "
                    f"hand_ids={len(hand_ids)}, target_ids={len(target_ids)}"
                )
            self.agent_hand_ids.append(torch.tensor(hand_ids, device=self.device, dtype=torch.long))
            self.agent_target_ids.append(torch.tensor(target_ids, device=self.device, dtype=torch.long))

        self._latest_self_obs_list = [
            torch.zeros((self.num_envs, self.agent_self_obs_sizes[i]), device=self.device, dtype=torch.float)
            for i in range(self.num_agents)
        ]
        self._latest_task_obs_list = [
            torch.zeros((self.num_envs, self.agent_task_obs_sizes[i]), device=self.device, dtype=torch.float)
            for i in range(self.num_agents)
        ]
        self._latest_full_obs_list = [
            torch.zeros((self.num_envs, self.agent_total_obs_sizes[i]), device=self.device, dtype=torch.float)
            for i in range(self.num_agents)
        ]

    def _compute_agent_self_obs(self, agent_idx: int, env_ids: torch.Tensor) -> torch.Tensor:
        return compute_humanoid_observations_max(
            self._rigid_body_pos_list[agent_idx][env_ids],
            self._rigid_body_rot_list[agent_idx][env_ids],
            self._rigid_body_vel_list[agent_idx][env_ids],
            self._rigid_body_ang_vel_list[agent_idx][env_ids],
            self._local_root_obs,
            self._root_height_obs,
        )

    def _compute_agent_task_obs(self, agent_idx: int, env_ids: torch.Tensor) -> torch.Tensor:
        source_idx = (agent_idx + 1) % self.num_agents
        root_states = self._humanoid_root_states_list[agent_idx][env_ids]
        body_pos = self._rigid_body_pos_list[agent_idx][env_ids]
        self_contact_norm = torch.linalg.norm(self._contact_forces_list[agent_idx][env_ids], dim=-1)
        opponent_root_states = self._humanoid_root_states_list[source_idx][env_ids]
        opponent_body_pos = self._rigid_body_pos_list[source_idx][env_ids]
        opponent_body_rot = self._rigid_body_rot_list[source_idx][env_ids]
        opponent_dof_pos = self._dof_pos_list[source_idx][env_ids]
        opponent_dof_vel = self._dof_vel_list[source_idx][env_ids]
        opponent_contact_norm = torch.linalg.norm(self._contact_forces_list[source_idx][env_ids], dim=-1)

        obs_fn = compute_combat_observation_pm01 if self.agent_types[source_idx] == "pm01" else compute_combat_observation_g1
        return obs_fn(
            root_states,
            body_pos,
            opponent_root_states,
            opponent_body_pos,
            opponent_body_rot,
            opponent_dof_pos,
            opponent_dof_vel,
            self_contact_norm,
            opponent_contact_norm,
            self.agent_hand_ids[agent_idx],
            self.agent_target_ids[source_idx],
        )

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        for i in range(self.num_agents):
            self_obs = self._compute_agent_self_obs(i, env_ids)
            task_obs = self._compute_agent_task_obs(i, env_ids)
            full_obs = torch.cat([self_obs, task_obs], dim=-1)

            self._latest_self_obs_list[i][env_ids] = self_obs
            self._latest_task_obs_list[i][env_ids] = task_obs
            self._latest_full_obs_list[i][env_ids] = full_obs

            row_idx = env_ids + i * self.num_envs
            self.obs_buf[row_idx].zero_()
            width = min(self.num_obs, full_obs.shape[-1])
            self.obs_buf[row_idx, :width] = full_obs[:, :width]

        return

    def get_agent_self_obs(self, agent_idx: int) -> torch.Tensor:
        return self._latest_self_obs_list[agent_idx]

    def get_agent_policy_obs(self, agent_idx: int) -> torch.Tensor:
        return self._latest_full_obs_list[agent_idx]

    def _compute_reward(self, actions):
        self.rew_buf.zero_()
        self.reward_raw.zero_()
        return

    def _compute_reset(self):
        terminated = torch.zeros_like(self.reset_buf)
        if self._enable_early_termination:
            for i in range(self.num_agents):
                root_height = self._humanoid_root_states_list[i][:, 2]
                terminated = torch.logical_or(terminated > 0, root_height < self._termination_root_heights[i]).long()

        chest_idx_0 = int(self.agent_torso_indices[0])
        chest_idx_1 = int(self.agent_torso_indices[1])
        chest_xyz = self._rigid_body_pos_list[0][:, chest_idx_0, :]
        opponent_chest_xyz = self._rigid_body_pos_list[1][:, chest_idx_1, :]
        dist_xyz = torch.linalg.norm(chest_xyz - opponent_chest_xyz, dim=-1)
        too_far = dist_xyz > float(self.cfg["env"]["too_far_dist"])
        terminated = torch.logical_or(terminated > 0, too_far).long()

        reset = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            terminated,
        )
        self.reset_buf[:] = reset
        self._terminate_buf[:] = terminated
        return

    def _reset_actors(self, env_ids):
        self._reset_default_state(env_ids)
        return

    def pre_physics_step(self, actions):
        action_list = []
        for i in range(self.num_agents):
            a = actions[i].to(self.device)
            if a.ndim == 1:
                a = a.unsqueeze(0)
            if a.shape[-1] != self.agent_num_dofs[i]:
                raise ValueError(
                    f"Agent {i} action dim mismatch: got {a.shape[-1]}, expected {self.agent_num_dofs[i]}"
                )
            action_list.append(torch.clamp(a, -10.0, 10.0))
        self.pd_tar_total = torch.cat(action_list, dim=-1)
        self.actions = self.pd_tar_total
        return

    def _compute_torques(self, actions):
        dof_pos = torch.cat(self._dof_pos_list, dim=-1)
        dof_vel = torch.cat(self._dof_vel_list, dim=-1)
        torques = self.kp_gains_total * (
            self.pd_tar_total + self.default_dof_pos_total - dof_pos
        ) - self.kd_gains_total * dof_vel
        return torch.clip(torques, -self.torque_limits_total, self.torque_limits_total)

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.control_i = i
            self.render()
            if not self.paused and self.enable_viewer_sync:
                self.torques = self._compute_torques(self.pd_tar_total)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
        return


class HumanoidCombatMixedZ(HumanoidCombatMixedPlay):
    def load_humanoid_configs(self, cfg):
        super().load_humanoid_configs(cfg)
        self._num_actions = int(cfg["env"]["z_size"])

    @staticmethod
    def _get_foot_names(humanoid_type: str) -> List[str]:
        if humanoid_type == "g1":
            return G1_FOOT_NAMES
        if humanoid_type == "pm01":
            return PM01_FOOT_NAMES
        raise ValueError(f"Unsupported humanoid_type for foot mapping: {humanoid_type}")

    @staticmethod
    def _infer_prior_input_dim(prior_net: torch.nn.Module) -> int:
        for module in prior_net.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        raise ValueError("Failed to infer prior input dim from prior network.")

    @staticmethod
    def _normalize_with_rms(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor) -> torch.Tensor:
        dim = min(x.shape[-1], running_mean.shape[0])
        x_proc = x[:, :dim]
        x_proc = (x_proc - running_mean[:dim]) / torch.sqrt(running_var[:dim] + 1e-5)
        x_proc = torch.clamp(x_proc, min=-5.0, max=5.0)
        if x.shape[-1] == dim:
            return x_proc
        return torch.cat([x_proc, x[:, dim:]], dim=-1)

    def _resolve_prior_path(self, path: str) -> str:
        return self._resolve_cfg_path(path)

    def _get_agent_prior_paths(self) -> List[str]:
        env_cfg = self.cfg["env"]
        g1_path = env_cfg["g1_prior_path"]
        pm01_path = env_cfg["pm01_prior_path"]
        type_to_path = {"g1": g1_path, "pm01": pm01_path}
        resolved = []
        for agent_type in self.agent_types:
            path = type_to_path.get(agent_type, None)
            if path is None:
                raise ValueError(f"Missing prior path for agent type: {agent_type}")
            resolved.append(self._resolve_prior_path(path))
        return resolved

    def _init_agent_priors(self):
        prior_paths = self._get_agent_prior_paths()
        prior_activation = str(self.cfg["env"]["prior_activation"])
        self.agent_prior_nets = []
        self.agent_prior_running_mean = []
        self.agent_prior_running_var = []
        self.agent_prior_input_dims = []

        for idx, ckpt_path in enumerate(prior_paths):
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Prior checkpoint not found for agent {idx}: {ckpt_path}")
            ckpt = torch_ext.load_checkpoint(ckpt_path)
            if "running_mean_std" not in ckpt:
                raise ValueError(f"Checkpoint missing running_mean_std: {ckpt_path}")
            prior_net = load_combat_prior(ckpt, device=self.device, activation=prior_activation)
            self.agent_prior_nets.append(prior_net)
            self.agent_prior_running_mean.append(ckpt["running_mean_std"]["running_mean"].float().to(self.device))
            self.agent_prior_running_var.append(ckpt["running_mean_std"]["running_var"].float().to(self.device))
            self.agent_prior_input_dims.append(self._infer_prior_input_dim(prior_net))

    def _build_agent_foot_ids(self):
        self.agent_foot_ids = []
        for i in range(self.num_agents):
            foot_names = self._get_foot_names(self.agent_types[i])
            body_names = self.agent_body_names[i]
            ids = [body_names.index(n) for n in foot_names if n in body_names]
            if len(ids) == 0:
                raise ValueError(f"Invalid foot mapping for agent {i} ({self.agent_types[i]}).")
            self.agent_foot_ids.append(torch.tensor(ids, device=self.device, dtype=torch.long))

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.z_size = int(self.cfg["env"]["z_size"])
        self._init_agent_priors()
        self._build_agent_foot_ids()

        self.reward_weights = dict(self.cfg["env"]["reward_weights"])
        self.too_close_dist = float(self.cfg["env"]["too_close_dist"])
        self.too_close_frames = int(self.cfg["env"]["too_close_frames"])
        self._too_close_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.hand_target_close_dist = float(self.cfg["env"]["hand_target_close_dist"])
        self.hand_target_close_frames = int(self.cfg["env"]["hand_target_close_frames"])
        self._hand_target_close_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._prev_root_pos_list = [
            torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float) for _ in range(self.num_agents)
        ]

    def _decode_agent_actions(self, agent_idx: int, self_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        prior_mean = self.agent_prior_running_mean[agent_idx]
        prior_var = self.agent_prior_running_var[agent_idx]
        prior_input_dim = self.agent_prior_input_dims[agent_idx]
        expected_self_obs_dim = prior_input_dim - z.shape[-1]
        if expected_self_obs_dim <= 0:
            raise ValueError(
                f"Invalid prior input split for agent {agent_idx}: "
                f"prior_input_dim={prior_input_dim}, z_dim={z.shape[-1]}"
            )

        if self_obs.shape[-1] != expected_self_obs_dim:
            raise ValueError(
                f"Self-obs dim mismatch for agent {agent_idx}: "
                f"got {self_obs.shape[-1]}, expected {expected_self_obs_dim}"
            )
        self_obs = self._normalize_with_rms(self_obs, prior_mean, prior_var)
        prior_input = torch.cat([self_obs, z], dim=-1)
        if prior_input.shape[-1] != prior_input_dim:
            raise ValueError(
                f"Prior input dim mismatch for agent {agent_idx}: "
                f"got {prior_input.shape[-1]}, expected {prior_input_dim}"
            )
        return self.agent_prior_nets[agent_idx](prior_input)

    def step(self, actions_z):
        return self.step_z(actions_z)

    def step_z(self, actions_z):
        with torch.no_grad():
            actions_z = actions_z.to(self.device)
            if actions_z.ndim == 1:
                actions_z = actions_z.unsqueeze(0)
            if actions_z.shape[-1] != self.z_size:
                raise ValueError(f"Mixed-z action dim mismatch: got {actions_z.shape[-1]}, expected {self.z_size}")
            if actions_z.shape[0] != self.num_envs * self.num_agents:
                raise ValueError(
                    f"Mixed-z batch mismatch: got {actions_z.shape[0]}, expected {self.num_envs * self.num_agents}"
                )

            actions_z = project_to_norm(actions_z, 1.0, "sphere")
            z_chunks = torch.chunk(actions_z, self.num_agents, dim=0)

            action_list = []
            for i in range(self.num_agents):
                self_obs = self.get_agent_self_obs(i)
                action_list.append(self._decode_agent_actions(i, self_obs, z_chunks[i]))

        self.pre_physics_step(action_list)
        self._physics_step()
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        self._too_close_count[env_ids] = 0
        self._hand_target_close_count[env_ids] = 0
        return

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            source_idx = (i + 1) % self.num_agents
            body_pos = self._rigid_body_pos_list[i]
            self_contact_norm = torch.linalg.norm(self._contact_forces_list[i], dim=-1)
            opponent_body_pos = self._rigid_body_pos_list[source_idx]
            opponent_contact_norm = torch.linalg.norm(self._contact_forces_list[source_idx], dim=-1)

            self_fallen = compute_humanoid_reset_in_reward(
                self.reset_buf,
                self.progress_buf,
                self.agent_foot_ids[i],
                body_pos,
                self._enable_early_termination,
                self._termination_heights_list[i],
            )
            opponent_fallen = compute_humanoid_reset_in_reward(
                self.reset_buf,
                self.progress_buf,
                self.agent_foot_ids[source_idx],
                opponent_body_pos,
                self._enable_early_termination,
                self._termination_heights_list[source_idx],
            )

            reward, _ = compute_combat_reward_mixed_ids(
                self_fallen,
                opponent_fallen,
                body_pos,
                opponent_body_pos,
                self_contact_norm,
                opponent_contact_norm,
                self.agent_hand_ids[i],
                self.agent_target_ids[i],
                self.agent_hand_ids[source_idx],
                self.agent_target_ids[source_idx],
                self.reward_weights,
            )
            self.rew_buf[i * self.num_envs : (i + 1) * self.num_envs] = reward
        return

    def _compute_reset(self):
        terminated = torch.zeros_like(self.reset_buf)
        if self._enable_early_termination:
            for i in range(self.num_agents):
                body_height = self._rigid_body_pos_list[i][..., 2]
                fall_height = body_height < self._termination_heights_list[i]
                contact_ids = self.agent_contact_body_ids[i]
                if contact_ids.numel() > 0:
                    fall_height[:, contact_ids] = False
                has_fallen = torch.any(fall_height, dim=-1) & (self.progress_buf > 1)
                terminated = torch.logical_or(terminated > 0, has_fallen).long()

            chest_idx_0 = int(self.agent_torso_indices[0])
            chest_idx_1 = int(self.agent_torso_indices[1])
            chest_xyz = self._rigid_body_pos_list[0][:, chest_idx_0, :]
            opponent_chest_xyz = self._rigid_body_pos_list[1][:, chest_idx_1, :]
            dist_xyz = torch.linalg.norm(chest_xyz - opponent_chest_xyz, dim=-1)
            too_far = dist_xyz > float(self.cfg["env"]["too_far_dist"])
            terminated = torch.logical_or(terminated > 0, too_far).long()
            too_close = (dist_xyz < self.too_close_dist) & (self.progress_buf > 1)
            self._too_close_count = torch.where(
                too_close,
                self._too_close_count + 1,
                torch.zeros_like(self._too_close_count),
            )
            too_close_term = self._too_close_count >= int(self.too_close_frames)

            contact_force_threshold = float(self.cfg["env"]["contact_force_threshold"])
            target_force_0 = torch.linalg.norm(self._contact_forces_list[0][:, self.agent_target_ids[0], :], dim=-1)  # (N, T)
            target_force_1 = torch.linalg.norm(self._contact_forces_list[1][:, self.agent_target_ids[1], :], dim=-1)  # (N, T)
            hand_close_any = (
                (target_force_0 > contact_force_threshold).any(dim=-1) |
                (target_force_1 > contact_force_threshold).any(dim=-1)
            ) & (self.progress_buf > 1)
            self._hand_target_close_count = torch.where(
                hand_close_any,
                self._hand_target_close_count + 1,
                torch.zeros_like(self._hand_target_close_count),
            )
            hand_close_term = self._hand_target_close_count >= int(self.hand_target_close_frames)
            terminated = torch.logical_or(terminated > 0, too_close_term | hand_close_term).long()

        reset = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            terminated,
        )
        self._too_close_count = torch.where(reset > 0, torch.zeros_like(self._too_close_count), self._too_close_count)
        self._hand_target_close_count = torch.where(
            reset > 0, torch.zeros_like(self._hand_target_close_count), self._hand_target_close_count
        )
        self.reset_buf[:] = reset
        self._terminate_buf[:] = terminated
        return
