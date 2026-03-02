import torch
import warnings
warnings.filterwarnings("ignore")

import phc.env.tasks.humanoid_combat as humanoid_combat

from utils import torch_utils
from isaacgym.torch_utils import *
from phc.utils.flags import flags
from phc.env.tasks.humanoid_combat import compute_humanoid_reset, compute_humanoid_reset_z


# G1 body name constants (29-DOF Unitree G1 robot - real URDF bodies only)
G1_STRIKE_BODY_NAMES = [
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
]
G1_FOOT_NAMES = [
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
]
G1_HAND_NAMES = [
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_knee_link", "left_ankle_roll_link",
    "right_knee_link", "right_ankle_roll_link",
]
G1_TARGET_NAMES = [
    "pelvis", "torso_link", "left_hip_roll_link", "right_hip_roll_link"
]


class HumanoidCombatG1(humanoid_combat.HumanoidCombat):
    """Combat task for G1 robot — overrides body-name setup from HumanoidCombat."""

    def _get_combat_body_names(self):
        """Return G1-specific body names for combat task."""
        return G1_STRIKE_BODY_NAMES, G1_FOOT_NAMES, G1_HAND_NAMES, G1_TARGET_NAMES

    def get_task_obs_size(self):
        """Task obs size for G1 combat:
        15 (root) + 58 (29 dof_obs + 29 dof_vel) + 270 (30*3 body_pos + 30*6 body_rot)
        + 60 (30 + 30 contact) + 6*8*3 (hand-target diffs) = 547
        """
        if self._enable_task_obs:
            num_bodies = len(self._body_names)  # 30 for g1
            num_dofs = len(self._dof_names)     # 29 for g1
            num_hand = len(G1_HAND_NAMES)       # 6
            num_target = len(G1_TARGET_NAMES)   # 8
            obs_size = (3 + 6 + 3 + 3)                   # root: pos_diff + rot_obs + vel + ang_vel
            obs_size += num_dofs + num_dofs              # oppo_dof_obs + oppo_dof_vel
            obs_size += num_bodies * 3 + num_bodies * 6  # body_pos_diff + body_rot
            obs_size += num_bodies + num_bodies          # self + oppo contact norm
            obs_size += num_hand * num_target * 3        # hand-target diffs
            return obs_size
        return 0

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

    def _compute_task_obs(self, env_ids=None):
        """Compute task observations using G1-specific observation function."""
        obs_list = []
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        for i in range(self.num_agents):
            opp_idx = (i + 1) % self.num_agents
            root_states = self._humanoid_root_states_list[i][env_ids]
            body_pos = self._rigid_body_pos_list[i][env_ids]
            contact_force = self._contact_forces_list[i][env_ids]
            contact_force_norm = torch.linalg.norm(contact_force, dim=-1)

            opponent_root_states = self._humanoid_root_states_list[opp_idx][env_ids]
            opponent_body_pos = self._rigid_body_pos_list[opp_idx][env_ids]
            opponent_body_rot = self._rigid_body_rot_list[opp_idx][env_ids]
            opponent_dof_pos = self._dof_pos_list[opp_idx][env_ids]
            opponent_dof_vel = self._dof_vel_list[opp_idx][env_ids]
            opponent_contact_force_norm = torch.linalg.norm(self._contact_forces_list[opp_idx][env_ids], dim=-1)
            
            obs = compute_combat_observation_g1(
                root_states, body_pos,
                opponent_root_states, opponent_body_pos, opponent_body_rot,
                opponent_dof_pos, opponent_dof_vel,
                contact_force_norm, opponent_contact_force_norm,
                self._hand_ids, self._target_ids,
            )
            obs_list.append(obs)

        return obs_list


class HumanoidCombatG1Z(HumanoidCombatG1):
    """G1 combat task with latent-space prior (combat prior from PULSE checkpoint).

    The high-level RL policy outputs z (64-dim sphere codes); the low-level
    combat prior translates (self_obs + z) → joint targets.  Any valid z on
    the unit sphere should produce valid combat motions (random z = random style).
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        self.initialize_z_models()

    def step(self, actions):
        super().step_z(actions)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()

    def _compute_reset(self):
        if flags.test:
            self.reset_buf[:], self._terminate_buf[:], self._too_close_count[:], self._hand_target_close_count[:]  = compute_humanoid_reset_z(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list, 
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents,
                                                           self._too_close_count, self.too_close_frames, self.too_close_dist, 
                                                           self._hand_target_close_count, self.hand_target_close_frames, self.hand_target_close_dist,
                                                           self._hand_ids, self._target_ids,
                                                           self.contact_force_threshold, self.too_far_dist, self.chest_body_index,
                                                             )
        else:
            self.reset_buf[:], self._terminate_buf[:], self._too_close_count[:], self._hand_target_close_count[:]  = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents,
                                                           self._too_close_count, self.too_close_frames, self.too_close_dist,
                                                           self._hand_target_close_count, self.hand_target_close_frames, self.hand_target_close_dist,
                                                           self._hand_ids, self._target_ids,
                                                           self.contact_force_threshold, self.too_far_dist, self.chest_body_index,
                                                         )
        # self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_combat_observation_g1(
    self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_body_rot,
    oppo_dof_pos, oppo_dof_vel, self_contact_norm, oppo_contact_norm,
    hand_ids, target_ids,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Combat task observation for G1 robot.

    Same as compute_combat_observation for SMPL, but the opponent DOF observation
    uses raw scalar angles (G1 single-axis joints) instead of dof_to_obs_smpl.
    """
    # root info
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

    # G1: raw scalar DOF angles (no tan_norm conversion)
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

    flat_oppo_body_rot = oppo_body_rot.view(
        oppo_body_rot.shape[0] * oppo_body_rot.shape[1], oppo_body_rot.shape[2]
    )
    local_flat_oppo_body_rot = torch_utils.quat_mul(flat_heading_rot, flat_oppo_body_rot)
    local_flat_oppo_body_rot = torch_utils.quat_to_tan_norm(local_flat_oppo_body_rot)
    local_flat_oppo_body_rot = local_flat_oppo_body_rot.view(oppo_body_rot.shape[0], oppo_body_rot.shape[1] * 6)

    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot2 = heading_rot.unsqueeze(-2).repeat(
        (1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1)
    )
    local_target_hand_pos_diff = torch_utils.quat_rotate(
        flat_heading_rot2.view(-1, 4), global_target_hand_pos_diff
    )
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot2.shape[0], -1)

    obs = torch.cat(
        [
            local_root_pos_diff, local_root_rot_obs, local_oppo_vel, local_oppo_ang_vel,
            oppo_dof_obs, oppo_dof_vel,
            local_flat_oppo_body_pos_diff, local_flat_oppo_body_rot,
            local_target_hand_pos_diff, self_contact_norm, oppo_contact_norm,
        ],
        dim=-1,
    )
    return obs
