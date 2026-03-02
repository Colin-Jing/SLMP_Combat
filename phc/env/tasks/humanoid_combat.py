

import math
import torch
import warnings
warnings.filterwarnings("ignore")

import env.tasks.humanoid as humanoid
import phc.env.tasks.humanoid_task as humanoid_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
from enum import Enum
from env.tasks.humanoid import dof_to_obs_smpl

TAR_ACTOR_ID = 1

class GameState(Enum):
    out_bound = 0
    green_win = 1
    red_win = 2
    idle = 3

class HumanoidCombat(humanoid_task.HumanoidTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg_init = cfg["env"]
        self.agent_spawn_positions = list(env_cfg_init.get("agent_spawn_positions", [[0.4, 0.1], [-0.4, -0.1]]))
        self.agent_spawn_headings = list(env_cfg_init.get("agent_spawn_headings", [3.1415926, 0.0]))
        spawn_inside_margin = env_cfg_init.get("spawn_inside_margin", [1.0, 1.0])
        self.spawn_inside_margin = (float(spawn_inside_margin[0]), float(spawn_inside_margin[1]))
        self.contact_force_threshold = float(env_cfg_init.get("contact_force_threshold", 50.0))
        self.too_far_dist = float(env_cfg_init.get("too_far_dist", 1.2))
        self.chest_body_index = int(env_cfg_init.get("chest_body_index", 11))
        self.contact_force_clip_min = float(env_cfg_init.get("contact_force_clip_min", 100.0))
        self.contact_force_clip_max = float(env_cfg_init.get("contact_force_clip_max", 3000.0))
        ring_asset_cfg = env_cfg_init.get("ring_asset", {})
        self._ring_asset_root = ring_asset_cfg.get("asset_root", "phc/data/assets/urdf/")
        self._ring_asset_file = ring_asset_cfg.get("asset_file", "combat_ring.urdf")
        self._ring_angular_damping = float(ring_asset_cfg.get("angular_damping", 0.01))
        self._ring_linear_damping = float(ring_asset_cfg.get("linear_damping", 0.01))
        self._ring_max_angular_velocity = float(ring_asset_cfg.get("max_angular_velocity", 100.0))
        self._ring_density = float(ring_asset_cfg.get("density", 1.0))
        self._ring_pose_z = float(env_cfg_init.get("ring_pose_z", -1.02))
        ring_color = env_cfg_init.get("ring_color", [1.0, 1.0, 1.0])
        self._ring_color = gymapi.Vec3(float(ring_color[0]), float(ring_color[1]), float(ring_color[2]))

        super().__init__(cfg=cfg,
                            sim_params=sim_params,
                            physics_engine=physics_engine,
                            device_type=device_type,
                            device_id=device_id,
                            headless=headless)

        self._tar_dist_min = 0.5
        self._tar_dist_max = 1.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        self.first_in = True
        env_cfg = self.cfg.env
        
        
        self._prev_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_pos_list = []
        for i in range(self.num_agents):
            self._prev_root_pos_list.append(torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float))
        
        # strike_body_names = cfg["env"]["strikeBodyNames"]
        strikeBodyNames, footNames, handNames, targetNames = self._get_combat_body_names()
        self._strike_body_ids = self._build_key_body_ids_tensor(strikeBodyNames)
        self._foot_ids = self._build_key_body_ids_tensor(footNames)
        self._hand_ids = self._build_key_body_ids_tensor(handNames)
        self._target_ids = self._build_key_body_ids_tensor(targetNames)


        ########## building for combat area ##########
        combat_area = env_cfg["combat_area"]
        self.bounding_box = torch.tensor(combat_area, device=self.device, dtype=torch.float)
        self.bounding_box_points = torch.tensor(
            [[[combat_area[0], combat_area[2], 0.0], [combat_area[1], combat_area[3], 0.0]]],
            device=self.device,
            dtype=torch.float,
        ).repeat(self.num_envs, 1, 1)
        self.reward_weights = dict(env_cfg["reward_weights"])
        self.agent_spawn_positions = list(env_cfg["agent_spawn_positions"])
        self.agent_spawn_headings = list(env_cfg["agent_spawn_headings"])
        spawn_inside_margin = env_cfg["spawn_inside_margin"]
        self.spawn_inside_margin = (float(spawn_inside_margin[0]), float(spawn_inside_margin[1]))
        self.contact_force_threshold = float(env_cfg["contact_force_threshold"])
        self.too_far_dist = float(env_cfg["too_far_dist"])
        self.chest_body_index = int(env_cfg["chest_body_index"])
        self.contact_force_clip_min = float(env_cfg["contact_force_clip_min"])
        self.contact_force_clip_max = float(env_cfg["contact_force_clip_max"])

        ring_asset_cfg = env_cfg["ring_asset"]
        self._ring_asset_root = ring_asset_cfg["asset_root"]
        self._ring_asset_file = ring_asset_cfg["asset_file"]
        self._ring_angular_damping = float(ring_asset_cfg["angular_damping"])
        self._ring_linear_damping = float(ring_asset_cfg["linear_damping"])
        self._ring_max_angular_velocity = float(ring_asset_cfg["max_angular_velocity"])
        self._ring_density = float(ring_asset_cfg["density"])
        self._ring_pose_z = float(env_cfg["ring_pose_z"])
        ring_color = env_cfg["ring_color"]
        self._ring_color = gymapi.Vec3(float(ring_color[0]), float(ring_color[1]), float(ring_color[2]))

        self.too_close_dist = float(self.cfg.env["too_close_dist"])
        self.too_close_frames = int(self.cfg.env["too_close_frames"]) # 60))
        self._too_close_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self.hand_target_close_dist = float(self.cfg.env["hand_target_close_dist"])
        self.hand_target_close_frames = int(self.cfg.env["hand_target_close_frames"]) # 30)) 
        self._hand_target_close_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        

        self.env_ids_all = torch.arange(self.num_envs).to(self.device)
        self.warmup_time = int(250/self.dt) # 10 minutes wall time
        self.step_counter = 0

    def sample_position_on_field(self, n):
        inside_x, inside_y = self.spawn_inside_margin
        x = torch.FloatTensor(n).uniform_(self.bounding_box[0] + inside_x, self.bounding_box[1] - inside_x).to(self.device)
        y = torch.FloatTensor(n).uniform_(self.bounding_box[2] + inside_y, self.bounding_box[3] - inside_y).to(self.device)
        return torch.stack([x, y], dim=-1)

    def _get_combat_body_names(self):
        """Return (strikeBodyNames, footNames, handNames, targetNames) for the current robot type.
        Override in subclasses to support non-SMPL robots."""
        strikeBodyNames = ["L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe"]
        footNames = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
        handNames = ["L_Hand", "R_Hand", "L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe"]
        targetNames = ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head", "L_Hip", "R_Hip", "L_Knee", "R_Knee"]
        return strikeBodyNames, footNames, handNames, targetNames
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 486 + 8*10*3 # 522 + 8*10*3
        return obs_size

    def post_physics_step(self):
        # self.out_bound, self.red_win, self.green_win = self.check_game_state()
        self.step_counter += 1
        super().post_physics_step()

        return
    
    def _load_combat_ring_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = self._ring_angular_damping
        asset_options.linear_damping = self._ring_linear_damping
        asset_options.max_angular_velocity = self._ring_max_angular_velocity
        asset_options.density = self._ring_density
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._combat_ring_asset = self.gym.load_asset(
            self.sim, self._ring_asset_root, self._ring_asset_file, asset_options
        )


    def _build_combat_ring(self, env_id, env_ptr):
        col_group = env_id
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.z = self._ring_pose_z
        combat_ring_handle = self.gym.create_actor(env_ptr, self._combat_ring_asset, default_pose, "combat_ring",
                                                   col_group, -1, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, combat_ring_handle, 0, gymapi.MESH_VISUAL, self._ring_color)
        self._combat_ring_handles.append(combat_ring_handle)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._combat_ring_handles = []
        self._load_combat_ring_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        super()._build_env(env_id, env_ptr, humanoid_asset_list)
        self._build_combat_ring(env_id, env_ptr)
        return
    
    def _reset_default(self, env_ids):
        super()._reset_default(env_ids)
        for i in range(self.num_agents):
            root_states = self._humanoid_root_states_list[i]
            root_states[env_ids, 7:13] = 0
            spawn_pos = self.agent_spawn_positions[i % len(self.agent_spawn_positions)]
            spawn_yaw = float(self.agent_spawn_headings[i % len(self.agent_spawn_headings)])
            yaw_half = 0.5 * spawn_yaw
            root_states[env_ids, 0] = float(spawn_pos[0])
            root_states[env_ids, 1] = float(spawn_pos[1])
            root_states[env_ids, 3] = 0.0
            root_states[env_ids, 4] = 0.0
            root_states[env_ids, 5] = math.sin(yaw_half)
            root_states[env_ids, 6] = math.cos(yaw_half)
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        self._too_close_count[env_ids] = 0
        self._hand_target_close_count[env_ids] = 0
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        
        for i in range(self.num_agents):
            self._prev_root_pos_list[i] = self._humanoid_root_states_list[i][..., 0:3].clone()

        return
    
    # def check_game_state(self):
    #     tar_pos = self._target_states[..., 0:3]
    #     fuzzy = 0.1
    #     out_bound = torch.logical_or(torch.logical_or(tar_pos[..., 0] < self.bounding_box[0] - fuzzy,   tar_pos[..., 0] > self.bounding_box[1] + fuzzy),   torch.logical_or(tar_pos[..., 1] < self.bounding_box[2] - fuzzy,  tar_pos[..., 1] > self.bounding_box[3] + fuzzy))
        
        
    #     red_win = torch.logical_and(torch.logical_and(torch.logical_and(tar_pos[..., 0] >= self.goal_bound_green[0], tar_pos[..., 0] <= self.goal_bound_green[1]), torch.logical_and(tar_pos[..., 1] >= self.goal_bound_green[2], tar_pos[..., 1] <= self.goal_bound_green[3])),  torch.logical_and(tar_pos[..., 2] >= self.goal_bound_green[4], tar_pos[..., 2] <= self.goal_bound_green[5]))
    #     green_win = torch.logical_and(torch.logical_and(torch.logical_and(tar_pos[..., 0] >= self.goal_bound_red[0], tar_pos[..., 0] <= self.goal_bound_red[1]), torch.logical_and(tar_pos[..., 1] >= self.goal_bound_red[2], tar_pos[..., 1] <= self.goal_bound_red[3])),  torch.logical_and(tar_pos[..., 2] >= self.goal_bound_red[4], tar_pos[..., 2] <= self.goal_bound_red[5]))
        
    #     return out_bound, red_win, green_win
    
    def _compute_task_obs(self, env_ids=None):
        
        obs_list = []
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        num_envs = env_ids.shape[0]
        for i in range(self.num_agents):
            root_states = self._humanoid_root_states_list[i][env_ids]
            body_pos = self._rigid_body_pos_list[i][env_ids]
            contact_force = self._contact_forces_list[i][env_ids]
            contact_force_norm = torch.linalg.norm(contact_force, dim=-1)
            
            if i%2 == 0:
                opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 13)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_body_rot = torch.stack([self._rigid_body_rot_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 4)).to(self.device)
                opponent_dof_pos = torch.stack([self._dof_pos_list[j][env_ids]for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_dof_vel = torch.stack([self._dof_vel_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j][env_ids] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)

            else:
                opponent_root_states = torch.stack([self._humanoid_root_states_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 13)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_body_rot = torch.stack([self._rigid_body_rot_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 4)).to(self.device)
                opponent_dof_pos = torch.stack([self._dof_pos_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_dof_vel = torch.stack([self._dof_vel_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((num_envs, 1, 69)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j][env_ids] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((num_envs, 1, 24, 3)).to(self.device)
                opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)

        
            obs = compute_combat_observation(root_states, body_pos, opponent_root_states[:, 0], opponent_body_pos[:, 0], opponent_body_rot[:, 0], 
                                             opponent_dof_pos[:, 0], opponent_dof_vel[:, 0], contact_force_norm, opponent_contact_force_norm[:, 0],
                                             self._hand_ids, self._target_ids)                        
            obs_list.append(obs)
        
        return obs_list

    def _compute_reward(self, actions):
        for i in range(self.num_agents):
            root_state = self._humanoid_root_states_list[i]
            prev_root_state = self._prev_root_pos_list[i]
            contact_force = self._contact_forces_list[i]
            body_pos = self._rigid_body_pos_list[i]


            if i % 2 == 0:
                opponent_root_state = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j] for j in range(self.num_agents) if j % 2 != 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
            else:
                opponent_root_state = torch.stack([self._humanoid_root_states_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -2) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 13)).to(self.device)
                opponent_contact_force = torch.stack([self._contact_forces_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)
                opponent_body_pos = torch.stack([self._rigid_body_pos_list[j] for j in range(self.num_agents) if j % 2 == 0], dim = -3) if self.num_agents > 1 else torch.zeros((self.num_envs, 1, 24, 3)).to(self.device)

            contact_force_norm = torch.linalg.norm(contact_force, dim=-1)
            # If contact force on the first two hand IDs is greater than 0, print its magnitude.
            opponent_contact_force_norm = torch.linalg.norm(opponent_contact_force, dim=-1)
            lower_clipped_force = self.contact_force_clip_min
            upper_clipped_force = self.contact_force_clip_max
            #clipped_contact_force = torch.clamp(contact_force_norm.clone(), lower_clipped_force, upper_clipped_force)
            #clipped_opponent_contact_force = torch.clamp(opponent_contact_force_norm.clone()[:, 0], lower_clipped_force, upper_clipped_force)
            contact_force_norm[contact_force_norm<lower_clipped_force] = 0
            contact_force_norm[contact_force_norm>upper_clipped_force] = upper_clipped_force
            opponent_contact_force_norm[:, 0][opponent_contact_force_norm[:, 0]<lower_clipped_force] = 0
            opponent_contact_force_norm[:, 0][opponent_contact_force_norm[:, 0]>upper_clipped_force] = upper_clipped_force




            self_fallen = compute_humanoid_reset_in_reward(self.reset_buf, self.progress_buf, self._foot_ids, body_pos, self._enable_early_termination, self._termination_heights)
            opponent_fallen = compute_humanoid_reset_in_reward(self.reset_buf, self.progress_buf, self._foot_ids, opponent_body_pos[:, 0], self._enable_early_termination, self._termination_heights)
        


            reward, reward_raw = compute_combat_reward(root_state, prev_root_state, body_pos, self_fallen, opponent_fallen,
                                                        contact_force_norm, opponent_contact_force_norm[:,0], 
                                                        opponent_root_state[:, 0],  opponent_body_pos[:, 0], self._hand_ids, self._target_ids, self.dt, self.reward_weights)

            self.rew_buf[i*self.num_envs:(i+1)*self.num_envs] = reward
                            
        return

    def _compute_reset(self):
        #game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
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
        #self.reset_buf[:] = torch.logical_or(self.reset_buf, game_done)
        self.reset_episode = self.reset_buf.clone()
        return
    


class HumanoidCombatZ(HumanoidCombat):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        super().step_z(actions)
        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        return
    
    def _compute_reset(self):
        
        # game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
        if flags.test: # self.step_counter > self.warmup_time or 
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

# borrow from NCP https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/env/tasks/humanoid_combat.py
@torch.jit.script
def compute_combat_observation(self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_body_rot,
                                  oppo_dof_pos, oppo_dof_vel, self_contact_norm, oppo_contact_norm,
                                  hand_ids, target_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
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

    oppo_dof_obs = dof_to_obs_smpl(oppo_dof_pos)
    oppo_body_pos_diff = oppo_body_pos - self_root_pos[:, None]
    flat_oppo_body_pos_diff = oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0] * oppo_body_pos_diff.shape[1],
                                              oppo_body_pos_diff.shape[2])
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, oppo_body_pos_diff.shape[1], 1))

    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_flat_oppo_body_pos_diff =  torch_utils.quat_rotate(flat_heading_rot, flat_oppo_body_pos_diff)
    local_flat_oppo_body_pos_diff = local_flat_oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0], oppo_body_pos_diff.shape[1] * oppo_body_pos_diff.shape[2])

    flat_oppo_body_rot = oppo_body_rot.view(oppo_body_rot.shape[0] * oppo_body_rot.shape[1], oppo_body_rot.shape[2])
    local_flat_oppo_body_rot = torch_utils.quat_mul(flat_heading_rot, flat_oppo_body_rot)
    local_flat_oppo_body_rot = torch_utils.quat_to_tan_norm(local_flat_oppo_body_rot)
    local_flat_oppo_body_rot = local_flat_oppo_body_rot.view(oppo_body_rot.shape[0], oppo_body_rot.shape[1] * 6)


    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot = heading_rot.unsqueeze(-2).\
        repeat((1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1))
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot.shape[0], -1)

    obs = torch.cat([local_root_pos_diff, local_root_rot_obs, local_oppo_vel, local_oppo_ang_vel,
                     oppo_dof_obs, oppo_dof_vel, local_flat_oppo_body_pos_diff, local_flat_oppo_body_rot,
                     local_target_hand_pos_diff, self_contact_norm, oppo_contact_norm], dim=-1)
    return obs


# borrow from NCP https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/env/tasks/humanoid_combat.py
#@torch.jit.script
def compute_combat_reward(root_state, prev_root_pos, body_pos, self_terminated, oppo_terminated, self_force_norm, oppo_force_norm, tar_pos, tar_body_pos, hand_ids, target_ids, dt, reward_weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, float]) -> Tuple[Tensor, Tensor]
    reward_f, reward_v, reward_s, reward_t, reward_h = reward_weights['reward_f'], reward_weights['reward_v'], reward_weights['reward_s'], reward_weights['reward_t'], reward_weights['reward_h']

    terminate_reward = oppo_terminated - 1.0 * self_terminated
    
    hit_dist_thresh = reward_weights['hit_dist_thresh']

    # Hit reward: our hand is close to opponent targets and opponent targets receive force.
    self_hand_pos = body_pos[:, hand_ids, :]                # (B, L, 3)
    oppo_target_pos = tar_body_pos[:, target_ids, :]        # (B, T, 3)
    self_to_oppo_dist = torch.linalg.norm(
        self_hand_pos.unsqueeze(2) - oppo_target_pos.unsqueeze(1), dim=-1
    )                                                       # (B, L, T)
    oppo_target_close = (self_to_oppo_dist < hit_dist_thresh).any(dim=1).float()  # (B, T)
    strike_reward = (oppo_force_norm[:, target_ids] * oppo_target_close ).sum(dim=-1)


    # Hit penalty: opponent hand is close to our targets and our targets receive force.
    oppo_hand_pos = tar_body_pos[:, hand_ids, :]            # (B, L, 3)
    self_target_pos = body_pos[:, target_ids, :]            # (B, T, 3)
    oppo_to_self_dist = torch.linalg.norm(
        oppo_hand_pos.unsqueeze(2) - self_target_pos.unsqueeze(1), dim=-1
    )                                                       # (B, L, T)
    self_target_close = (oppo_to_self_dist < hit_dist_thresh).any(dim=1).float()  # (B, T)

    got_hit_reward = (self_force_norm[:, target_ids] * self_target_close).sum(dim=-1)

    strike_reward = strike_reward - 1.0 * got_hit_reward

    reward = strike_reward * reward_s + terminate_reward * reward_t
    reward_raw = torch.stack([strike_reward, terminate_reward], dim=-1)

    return reward, reward_raw


#@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                        strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents,
                           too_close_count, too_close_frames: int, too_close_dist: float, 
                           hand_target_close_count, hand_target_close_frames: int, hand_target_close_dist: float,
                           hand_ids, target_ids,
                           contact_force_threshold: float, too_far_dist: float, chest_body_index: int):
    # type: (Tensor, Tensor, list, list, Tensor, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    new_too_close_count = too_close_count
    new_hand_close_count = hand_target_close_count

    if (enable_early_termination):
        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = fall_height
        for i in range(1, num_agents):
            body_height = rigid_body_pos_list[i][..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)
            has_fallen_temp = fall_height
            has_fallen = torch.logical_or(has_fallen, has_fallen_temp)
        
        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)


        Pelvis_id = target_ids[0]
        Pelvis_xyz = rigid_body_pos_list[0][:, Pelvis_id, :]           # (num_envs, 3)
        opponent_Pelvis_xyz = rigid_body_pos_list[1][:, Pelvis_id, :]  # (num_envs, 3)
        dist_xyz = torch.linalg.norm(Pelvis_xyz - opponent_Pelvis_xyz, dim=-1)  # (num_envs,)
        too_far = dist_xyz > too_far_dist
        terminated = torch.where(too_far, torch.ones_like(reset_buf), terminated)

        if num_agents > 1:
            # A) chest too_close for consecutive frames
            chest_xyz = rigid_body_pos_list[0][:, chest_body_index, :]
            opponent_chest_xyz = rigid_body_pos_list[1][:, chest_body_index, :]
            dist_xyz = torch.linalg.norm(chest_xyz - opponent_chest_xyz, dim=-1)
            too_close = (dist_xyz < too_close_dist) & (progress_buf > 1)

            new_too_close_count = torch.where(
                too_close,
                too_close_count + 1,
                torch.zeros_like(too_close_count),
            )
            too_close_term = new_too_close_count >= int(too_close_frames)

            # B) reset when any target_ids contact force > 50N for consecutive frames
            target_force_0 = torch.linalg.norm(contact_buf_list[0][:, target_ids, :], dim=-1)  # (N, T)
            target_force_1 = torch.linalg.norm(contact_buf_list[1][:, target_ids, :], dim=-1)  # (N, T)
            hand_close_any = (
                (target_force_0 > contact_force_threshold).any(dim=-1) |
                (target_force_1 > contact_force_threshold).any(dim=-1)
            ) & (progress_buf > 1)

            new_hand_close_count = torch.where(
                hand_close_any,
                hand_target_close_count + 1,
                torch.zeros_like(hand_target_close_count),
            )
            hand_close_term = new_hand_close_count >= int(hand_target_close_frames)

            term_any = has_failed | too_close_term | hand_close_term
            terminated = torch.where(term_any, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # Clear counters after reset to avoid cross-episode leakage
    new_too_close_count = torch.where(reset > 0, torch.zeros_like(new_too_close_count), new_too_close_count)
    new_hand_close_count = torch.where(reset > 0, torch.zeros_like(new_hand_close_count), new_hand_close_count)

    return reset, terminated, new_too_close_count, new_hand_close_count



#@torch.jit.script
def compute_humanoid_reset_z(reset_buf, progress_buf, contact_buf_list, contact_body_ids, rigid_body_pos_list,
                           strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, num_agents,
                           too_close_count, too_close_frames: int, too_close_dist: float, 
                           hand_target_close_count, hand_target_close_frames: int, hand_target_close_dist: float,
                           hand_ids, target_ids,
                           contact_force_threshold: float, too_far_dist: float, chest_body_index: int):
    # type: (Tensor, Tensor, list, Tensor, list, Tensor, float, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    new_too_close_count = too_close_count
    new_hand_close_count = hand_target_close_count

    if (enable_early_termination):
        body_height = rigid_body_pos_list[0][..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = fall_height
        for i in range(1, num_agents):
            body_height = rigid_body_pos_list[i][..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False
            fall_height = torch.any(fall_height, dim=-1)
            has_fallen_temp = fall_height
            has_fallen = torch.logical_or(has_fallen, has_fallen_temp)
        
        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

        Pelvis_id = target_ids[0]
        Pelvis_xyz = rigid_body_pos_list[0][:, Pelvis_id, :]           # (num_envs, 3)
        opponent_Pelvis_xyz = rigid_body_pos_list[1][:, Pelvis_id, :]  # (num_envs, 3)
        dist_xyz = torch.linalg.norm(Pelvis_xyz - opponent_Pelvis_xyz, dim=-1)  # (num_envs,)
        too_far = dist_xyz > too_far_dist
        terminated = torch.where(too_far, torch.ones_like(reset_buf), terminated)

        if num_agents > 1:
            # A) chest too_close for consecutive frames
            chest_xyz = rigid_body_pos_list[0][:, chest_body_index, :]
            opponent_chest_xyz = rigid_body_pos_list[1][:, chest_body_index, :]
            dist_xyz = torch.linalg.norm(chest_xyz - opponent_chest_xyz, dim=-1)
            too_close = (dist_xyz < too_close_dist) & (progress_buf > 1)

            new_too_close_count = torch.where(
                too_close,
                too_close_count + 1,
                torch.zeros_like(too_close_count),
            )
            too_close_term = new_too_close_count >= int(too_close_frames)

            # B) reset when any target_ids contact force > 50N for consecutive frames
            target_force_0 = torch.linalg.norm(contact_buf_list[0][:, target_ids, :], dim=-1)  # (N, T)
            target_force_1 = torch.linalg.norm(contact_buf_list[1][:, target_ids, :], dim=-1)  # (N, T)
            hand_close_any = (
                (target_force_0 > contact_force_threshold).any(dim=-1) |
                (target_force_1 > contact_force_threshold).any(dim=-1)
            ) & (progress_buf > 1)

            new_hand_close_count = torch.where(
                hand_close_any,
                hand_target_close_count + 1,
                torch.zeros_like(hand_target_close_count),
            )
            hand_close_term = new_hand_close_count >= int(hand_target_close_frames)

            term_any = has_failed | too_close_term | hand_close_term
            terminated = torch.where(term_any, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # Clear counters after reset to avoid cross-episode leakage
    new_too_close_count = torch.where(reset > 0, torch.zeros_like(new_too_close_count), new_too_close_count)
    new_hand_close_count = torch.where(reset > 0, torch.zeros_like(new_hand_close_count), new_hand_close_count)

    return reset, terminated, new_too_close_count, new_hand_close_count


# borrow from NCP https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/env/tasks/humanoid_combat.py
@torch.jit.script
def compute_humanoid_reset_in_reward(reset_buf, progress_buf, foot_ids, rigid_body_pos,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor) -> Tensor    
    

    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, foot_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = fall_height
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    return terminated
