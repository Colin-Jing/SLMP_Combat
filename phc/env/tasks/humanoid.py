# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from uuid import uuid4
import numpy as np
import os

import torch
import multiprocessing

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import joblib
from phc.utils import torch_utils

from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

from phc.utils.flags import flags
from phc.env.tasks.base_task import BaseTask
from tqdm import tqdm
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from collections import defaultdict
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import gc
import torch.multiprocessing as mp
from phc.utils.draw_utils import agt_color, get_color_gradient
from typing import OrderedDict


ENABLE_MAX_COORD_OBS = True
# PERTURB_OBJS = [
#     ["small", 60],
#     ["small", 7],
#     ["small", 10],
#     ["small", 35],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["large", 60],
#     ["small", 300],
# ]
PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]


class Humanoid(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.has_task = False
        self.num_agents = self.cfg["num_agents"]
        self.load_humanoid_configs(cfg)

        self.control_mode = self.cfg["env"]["control_mode"]
        if self.control_mode in ['isaac_pd']:
            self._pd_control = True
        else:
            self._pd_control = False
        self.power_scale = self.cfg["env"]["power_scale"]

        self.debug_viz = self.cfg["env"]["enable_debug_vis"]
        
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episode_length"]
        self._local_root_obs = self.cfg["env"]["local_root_obs"]
        self._root_height_obs = self.cfg["env"].get("root_height_obs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.temp_running_mean = self.cfg["env"].get("temp_running_mean", True)
        self.partial_running_mean = self.cfg["env"].get("partial_running_mean", False)
        self.self_obs_v = self.cfg["env"].get("self_obs_v", 1)

        self.key_bodies = self.cfg["env"]["key_bodies"]
        
        self._setup_character_props(self.key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        

        super().__init__(cfg=self.cfg)

        self.dt = self.control_freq_inv * sim_params.dt
        self._setup_tensors()
        self.self_obs_buf = torch.zeros((self.num_envs, self.get_self_obs_size()), device=self.device, dtype=torch.float)
        self.reward_raw = torch.zeros((self.num_envs * self.num_agents, 1)).to(self.device)
        if self.humanoid_type in ['g1', 'pm01']:
            self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs * self.num_agents, 1))
            if self.humanoid_type == "pm01":
                self.base_link_id = self._build_key_body_ids_tensor([self.cfg.robot.base_link]).squeeze()
            else:
                self.base_link_id = self._body_names.index("pelvis") if "pelvis" in self._body_names else 0
        # Initialize global frame index and recording storage
        self.frame_index = 0
        self.max_recording_frames = 1800 
        self.recording_dict = {
            "rootpos_agent1": [], "rootrot_agent1": [], "dofpos_agent1": [], "dofrot_agent1": [],
            "rootpos_agent2": [], "rootrot_agent2": [], "dofpos_agent2": [], "dofrot_agent2": []
        }
        self._nan_diag_enabled = os.environ.get("PHC_NAN_DIAG", "0") == "1"
        self._nan_diag_raise = os.environ.get("PHC_NAN_DIAG_RAISE", "0") == "1"
        self._nan_diag_reward_reported = False
        self._nan_diag_obs_reported = False
        return

    def _load_proj_asset(self):
        asset_root = "phc/data/assets/urdf/"

        small_asset_file = "block_projectile.urdf"
        # small_asset_file = "ball_medium.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 10000000.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        large_asset_file = "block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 10000000.0
        # large_asset_options.fix_base_link = True
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(self.sim, asset_root, large_asset_file, large_asset_options)
        return

    def _build_proj(self, env_id, env_ptr):
        pos = [
            [-0.01, 0.3, 0.4],
            # [ 0.0890016, -0.40830246, 0.25]
        ]
        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = pos[i][0]
            default_pose.p.y = pos[i][1]
            default_pose.p.z = pos[i][2]
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), env_id, 2)
            self._proj_handles.append(proj_handle)

        return


    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # ZL: needs to put this back
        if self.self_obs_v == 3:
            sensors_per_env = len(self.force_sensor_joints)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs * self.num_agents, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)

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
        
        self._reset_rb_pos_list = [torch.zeros(0)]  * self.num_agents
        self._reset_rb_rot_list = [torch.zeros(0)]  * self.num_agents
        self._reset_rb_vel_list = [torch.zeros(0)]  * self.num_agents
        self._reset_rb_ang_vel_list = [torch.zeros(0)]  * self.num_agents
        


        for i in range(self.num_agents):

            self._humanoid_root_states_list.append(self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., i, :])
            
            initial_humanoid_root_states = self._humanoid_root_states_list[i].clone()

            initial_humanoid_root_states[:, 7:13] = 0

            initial_humanoid_root_states[..., 0] = -3
            initial_humanoid_root_states[..., 1] = i*2 - self.num_agents/2
            
            initial_humanoid_root_states[..., 3] = 0
            initial_humanoid_root_states[..., 4] = 0
            initial_humanoid_root_states[..., 5] = 0
            initial_humanoid_root_states[..., 6] = 1

            

            if i%2==1:
                initial_humanoid_root_states[..., 0] = 3
                initial_humanoid_root_states[..., 1] = -1 * ((i-1)*2 - self.num_agents/2)

                initial_humanoid_root_states[..., 3] = 0
                initial_humanoid_root_states[..., 4] = 0
                initial_humanoid_root_states[..., 5] = 1
                initial_humanoid_root_states[..., 6] = 0


            self._initial_humanoid_root_states_list.append(initial_humanoid_root_states)

            self._humanoid_actor_ids_list.append(num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)+i)
        
        
            dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., i*self.num_dof:(i+1)*self.num_dof, 0]
            dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., i*self.num_dof:(i+1)*self.num_dof, 1]
        
            self._dof_pos_list.append(dof_pos)
            self._dof_vel_list.append(dof_vel)
        

            initial_dof_pos = torch.zeros_like(dof_pos, device=self.device, dtype=torch.float)
            initial_dof_vel = torch.zeros_like(dof_vel, device=self.device, dtype=torch.float)

            self._initial_dof_pos_list.append(initial_dof_pos)
            self._initial_dof_vel_list.append(initial_dof_vel)        

            self._rigid_body_pos_list.append(rigid_body_state_reshaped[..., i*self.num_bodies:(i+1)*self.num_bodies, 0:3])
            self._rigid_body_rot_list.append(rigid_body_state_reshaped[..., i*self.num_bodies:(i+1)*self.num_bodies, 3:7])

            self._rigid_body_vel_list.append(rigid_body_state_reshaped[..., i*self.num_bodies:(i+1)*self.num_bodies, 7:10])
            self._rigid_body_ang_vel_list.append(rigid_body_state_reshaped[..., i*self.num_bodies:(i+1)*self.num_bodies, 10:13])
       
            self._contact_forces_list.append(contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., i*self.num_bodies:(i+1)*self.num_bodies, :])

            if self.self_obs_v == 2:
                hist_len = self.past_track_steps
                self._rigid_body_pos_hist_list.append(torch.zeros(self.num_envs, hist_len, self.num_bodies, 3, device=self.device))
                self._rigid_body_rot_hist_list.append(torch.zeros(self.num_envs, hist_len, self.num_bodies, 4, device=self.device))
                self._rigid_body_vel_hist_list.append(torch.zeros(self.num_envs, hist_len, self.num_bodies, 3, device=self.device))
                self._rigid_body_ang_vel_hist_list.append(torch.zeros(self.num_envs, hist_len, self.num_bodies, 3, device=self.device))
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contact_bodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        self.allocate_buffers()

        if self.viewer != None or flags.server_mode:
            self._init_camera()

        self.count = 0

    def allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_agents * self.num_envs, self.num_obs), device=self.device,
                                   dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_agents * self.num_envs, device=self.device, dtype=torch.float)

    def load_humanoid_configs(self, cfg):
        self.humanoid_type = cfg.robot.humanoid_type
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            self.load_smpl_configs(cfg)
        elif self.humanoid_type in ['g1', 'pm01']:
            self.load_robot_configs(cfg)
        else:
            raise NotImplementedError
            
            
    def load_common_humanoid_configs(self, cfg):
        self._divide_group = cfg["env"].get("divide_group", False)
        self._group_obs = cfg["env"].get("group_obs", False)
        self._disable_group_obs = cfg["env"].get("disable_group_obs", False)
        if self._divide_group:
            self._group_num_people = group_num_people = min(cfg['env'].get("num_env_group", 128), cfg['env']['num_envs'])
            self._group_ids = torch.tensor(np.arange(cfg["env"]["num_envs"] / group_num_people).repeat(group_num_people).astype(int))

        self.force_sensor_joints = cfg["env"].get("force_sensor_joints", ["L_Ankle", "R_Ankle"]) # force tensor joints
        
        ##### Robot Configs #####
        self._has_shape_obs = cfg.robot.get("has_shape_obs", False)
        self._has_shape_obs_disc = cfg.robot.get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = cfg.robot.get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = cfg.robot.get("has_weight_obs_disc", False)
        self.has_shape_variation = cfg.robot.get("has_shape_variation", False)
        self._bias_offset = cfg.robot.get("bias_offset", False)
        self._has_self_collision = cfg.robot.get("has_self_collision", False)
        self._has_mesh = cfg.robot.get("has_mesh", True)
        self._replace_feet = cfg.robot.get("replace_feet", True)  # replace feet or not
        self._has_jt_limit = cfg.robot.get("has_jt_limit", True)
        self._has_dof_subset = cfg.robot.get("has_dof_subset", False)
        self._has_smpl_pd_offset = cfg.robot.get("has_smpl_pd_offset", False)
        self._masterfoot = cfg.robot.get("masterfoot", False)
        self._freeze_toe = cfg.robot.get("freeze_toe", True)
        ##### Robot Configs #####
        
        
        # self.shape_resampling_interval = cfg["env"].get("shape_resampling_interval", 100)
        self.getup_schedule = cfg["env"].get("getup_schedule", False)
        self._kp_scale = cfg["env"].get("kp_scale", 1.0)
        self._kd_scale = cfg["env"].get("kd_scale", self._kp_scale)
        
        self.hard_negative = cfg["env"].get("hard_negative", False)  # hard negative sampling for im
        self.cycle_motion = cfg["env"].get("cycle_motion", False)  # Cycle motion to reach 300
        self.power_reward = cfg["env"].get("power_reward", False)
        self.obs_v = cfg["env"].get("obs_v", 1)
        self.amp_obs_v = cfg["env"].get("amp_obs_v", 1)
        
        
        ## Kin stuff
        self.save_kin_info = cfg["env"].get("save_kin_info", False)
        self.kin_loss = cfg["env"].get("kin_loss", False)
        self.kin_policy = cfg["env"].get("kin_policy", False)
        self.kin_lr = cfg["env"].get("kin_lr", 5e-4)
        self.z_readout = cfg["env"].get("z_readout", False)
        self.z_read = cfg["env"].get("z_read", False)
        self.z_uniform = cfg["env"].get("z_uniform", False)
        self.z_model = cfg["env"].get("z_model", False)
        self.distill = cfg["env"].get("distill", False)
        
        self.remove_disc_rot = cfg["env"].get("remove_disc_rot", False)
        
         ## ZL Devs
        #################### Devs ####################
        self.fitting = cfg["env"].get("fitting", False)
        self.zero_out_far = cfg["env"].get("zero_out_far", False)
        self.zero_out_far_train = cfg["env"].get("zero_out_far_train", True)
        self.max_len = cfg["env"].get("max_len", -1)
        self.cycle_motion_xp = cfg["env"].get("cycle_motion_xp", False)  # Cycle motion, but cycle farrrrr.
        self.models_path = cfg["env"].get("models", ['output/dgx/smpl_im_fit_3_1/Humanoid_00185000.pth', 'output/dgx/smpl_im_fit_3_2/Humanoid_00198750.pth'])
        
        self.eval_full = cfg["env"].get("eval_full", False)
        self.auto_pmcp = cfg["env"].get("auto_pmcp", False)
        self.auto_pmcp_soft = cfg["env"].get("auto_pmcp_soft", False)
        self.strict_eval = cfg["env"].get("strict_eval", False)
        self.add_obs_noise = cfg["env"].get("add_obs_noise", False)

        self._occl_training = cfg["env"].get("occl_training", False)  # Cycle motion, but cycle farrrrr.
        self._occl_training_prob = cfg["env"].get("occl_training_prob", 0.1)  # Cycle motion, but cycle farrrrr.
        self._sim_occlu = False
        self._res_action = cfg["env"].get("res_action", False)
        self.close_distance = cfg["env"].get("close_distance", 0.25)
        self.far_distance = cfg["env"].get("far_distance", 3)
        self._zero_out_far_steps = cfg["env"].get("zero_out_far_steps", 90)
        self.past_track_steps = cfg["env"].get("past_track_steps", 5)
        #################### Devs ####################
        
    def load_smpl_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)
        
        ##### Robot Configs #####
        self._has_upright_start = cfg.robot.get("has_upright_start", True)
        self.remove_toe = cfg.robot.get("remove_toe", False)
        self.big_ankle = cfg.robot.get("big_ankle", False)
        self._real_weight_porpotion_capsules = cfg.robot.get("real_weight_porpotion_capsules", False)
        self._real_weight_porpotion_boxes = cfg.robot.get("real_weight_porpotion_boxes", False)
        self._real_weight = cfg.robot.get("real_weight", False) 
        self._master_range = cfg.robot.get("master_range", 30)
        self._freeze_toe = cfg.robot.get("freeze_toe", True)
        self._freeze_hand = cfg.robot.get("freeze_hand", True)
        self._box_body = cfg.robot.get("box_body", False)
        self.reduce_action = cfg.robot.get("reduce_action", False)
        
        
        if self._masterfoot:
            self.action_idx = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 60, 61, 62, 65, 66, 67, 68, 75, 76, 77, 80, 81, 82, 83]
        else:
            self.action_idx = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 42, 43, 44, 47, 48, 49, 50, 57, 58, 59, 62, 63, 64, 65]

        disc_idxes = []
        if self.humanoid_type == "smpl":
            self._body_names_orig = SMPL_MUJOCO_NAMES
        elif self.humanoid_type in ["smplh", "smplx"]:
            self._body_names_orig = SMPLH_MUJOCO_NAMES
            
        self._full_track_bodies = self._body_names_orig.copy()

        _body_names_orig_copy = self._body_names_orig.copy()
        _body_names_orig_copy.remove('L_Toe')  # Following UHC as hand and toes does not have realiable data.
        _body_names_orig_copy.remove('R_Toe')
        if self.humanoid_type == "smpl":
            _body_names_orig_copy.remove('L_Hand')
            _body_names_orig_copy.remove('R_Hand')
            
        self._eval_bodies = _body_names_orig_copy # default eval bodies

        self._body_names = self._body_names_orig
        self._masterfoot_config = None

        self._dof_names = self._body_names[1:]
        
        
        if self.humanoid_type == "smpl":
            remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]
            self.limb_weight_group = [
            ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'], \
                ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'], \
                    ['Pelvis',  'Torso', 'Spine', 'Chest', 'Neck', 'Head'], \
                        [ 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'], \
                            ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']]
        elif self.humanoid_type in ["smplh", "smplx"]:
            remove_names = ["L_Toe", "R_Toe"]
            self.limb_weight_group = [
                ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'], \
                    ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'], \
                        ['Pelvis',  'Torso', 'Spine', 'Chest', 'Neck', 'Head'], \
                            [ 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3'], \
                                ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']]
        
        if self.remove_disc_rot:
            remove_names = self._body_names_orig # NO AMP Rotation 
        self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]

        for idx, name in enumerate(self._dof_names):
            if not name in remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = torch.from_numpy(np.concatenate(disc_idxes)) if len(disc_idxes) > 0 else torch.tensor([]).long()
        self.left_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("L")]
        self.right_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("R")]
        
        self.left_lower_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]]
        self.right_lower_indexes = [idx for idx , name in enumerate(self._dof_names) if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]]
        
        self._load_amass_gender_betas()

    def load_robot_configs(self, cfg):
        """Load config for articulated robots (e.g. G1) - ported from PULSE."""
        self.load_common_humanoid_configs(cfg)
        self._has_upright_start = cfg["robot"].get("has_upright_start", True)
        self._real_weight = True
        self._body_names_orig = cfg["robot"].get("body_names", [])

        _body_names_orig_copy = self._body_names_orig.copy()
        self._full_track_bodies = _body_names_orig_copy

        _body_names_orig_copy = self._body_names_orig.copy()
        self._eval_bodies = _body_names_orig_copy
        self._body_names = self._body_names_orig
        self._masterfoot_config = None
        self.dof_subset = torch.tensor([]).long()

        self._dof_names = cfg["robot"].get("dof_names", [])
        self.limb_weight_group = cfg["robot"].get("limb_weight_group", [])
        self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]

    def _clear_recorded_states(self):
        del self.state_record
        self.state_record = defaultdict(list)

    def _record_states(self):
        pass
        # self.state_record['dof_pos'].append(self._dof_pos.cpu().clone())
        # self.state_record['root_states'].append(self._humanoid_root_states.cpu().clone())
        # self.state_record['progress'].append(self.progress_buf.cpu().clone())
        
    def _write_states_to_file(self, file_name):
        self.state_record['skeleton_trees'] = self.skeleton_trees
        self.state_record['humanoid_betas'] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record['progress'], dim=1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        dof_pos_all = torch.stack(self.state_record['dof_pos'])
        root_states_all = torch.stack(self.state_record['root_states'])
        fps = 60
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in range(len(split_idx)):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = dof_pos_all[start:end, humanoid_index]
            B, H = dof_pos_seg.shape

            root_states_seg = root_states_all[start:end, humanoid_index]
            body_quat = torch.cat([root_states_seg[:, None, 3:7], torch_utils.exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3))], dim=1)

            motion_dump = {
                "skeleton_tree": self.state_record['skeleton_trees'][humanoid_index].to_dict(),
                "body_quat": body_quat,
                "trans": root_states_seg[:, :3],
                "root_states_seg": root_states_seg,
                "dof_pos": dof_pos_seg,
            }
            motion_dump['fps'] = fps
            motion_dump['betas'] = self.humanoid_shapes[humanoid_index].detach().cpu().numpy()
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1

        joblib.dump(motion_dict_dump, file_name)
        self.state_record = defaultdict(list)

    def get_obs_size(self):
        return self.get_self_obs_size()

    def get_running_mean_size(self):
        return (self.get_obs_size(), )

    def get_self_obs_size(self):
        if self.self_obs_v == 1:
            return self._num_self_obs 
        elif self.self_obs_v == 2:
            return self._num_self_obs * (self.past_track_steps + 1)
        elif self.self_obs_v == 3:
            return self._num_self_obs 

    def get_action_size(self):
        return self._num_actions

    def get_dof_action_size(self):
        return self._dof_size

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['env_spacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        safe_reset = (env_ids is None) or len(env_ids) == self.num_envs
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        
        self._reset_envs(env_ids)

        if safe_reset:
            # import ipdb; ipdb.set_trace()
            # print("3resetting here!!!!", self._humanoid_root_states[0, :3] - self._rigid_body_pos[0, 0])
            # ZL: This way it will simuate one step, then get reset again, squashing any remaining wiredness. Temporary fix
            self.gym.simulate(self.sim)
            self._reset_envs(env_ids)
            torch.cuda.empty_cache()

        return
    
    def change_char_color(self):
        colors = []
        offset = np.random.randint(0, 10)
        for env_id in range(self.num_envs): 
            rand_cols = agt_color(env_id + offset)
            colors.append(rand_cols)
            
        self.sample_char_color(torch.tensor(colors), torch.arange(self.num_envs))
        

    def sample_char_color(self, cols, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(cols[env_id, 0], cols[env_id, 1], cols[env_id, 2]))
        return
    
    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return


    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids) # this funciton calle _set_env_state, and should set all state vectors 
            self._reset_env_tensors(env_ids)

            self._refresh_sim_tensors() 
            if self.self_obs_v == 2:
                self._init_tensor_history(env_ids)
            # # Debug
            # import ipdb; ipdb.set_trace()
            self._compute_observations(env_ids)
        
        
        return

    def _reset_env_tensors(self, env_ids):
        # torch.cat()
        # env_ids_int32 = torch.cat([self._humanoid_actor_ids[env_ids],self._humanoid_actor_ids_op[env_ids]])
        env_ids_int32 = torch.stack(self._humanoid_actor_ids_list)[:,env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        # print("#################### refreshing ####################")
        # print("rb", (self._rigid_body_state_reshaped[None, :] - self._rigid_body_state_reshaped[:, None]).abs().sum())
        # print("contact", (self._contact_forces[None, :] - self._contact_forces[:, None]).abs().sum())
        # print('dof_pos', (self._dof_pos[None, :] - self._dof_pos[:, None]).abs().sum())
        # print("dof_vel", (self._dof_vel[None, :] - self._dof_vel[:, None]).abs().sum())
        # print("root_states", (self._humanoid_root_states[None, :] - self._humanoid_root_states[:, None]).abs().sum())
        # print("#################### refreshing ####################")

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        # self._contact_forces[env_ids] = 0
        for i in range(self.num_agents):
            self._contact_forces_list[i][env_ids] = 0

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction

        # plane_params.static_friction = 50
        # plane_params.dynamic_friction = 50

        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        
        asset_file = self.cfg.robot.asset.assetFileName
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            ### ZL: changes
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28

            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_self_obs = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        elif self.humanoid_type in ["smpl", "smplh", "smplx"]:
            # import ipdb; ipdb.set_trace()
            self._dof_body_ids = np.arange(1, len(self._body_names))
            self._dof_offsets = np.linspace(0, len(self._dof_names) * 3, len(self._body_names)).astype(int)
            self._dof_obs_size = len(self._dof_names) * 6
            self._dof_size = len(self._dof_names) * 3
            if self.reduce_action:
                self._num_actions = len(self.action_idx)
            else:
                self._num_actions = len(self._dof_names) * 3

            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3  # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
            else:
                raise NotImplementedError()

            if self._has_shape_obs:
                self._num_self_obs += 11
            # if self._has_limb_weight_obs: self._num_self_obs += 23 + 24 if not self._masterfoot else  29 + 30 # 23 + 24 (length + weight)
            if self._has_limb_weight_obs:
                self._num_self_obs += 10

            if not self._root_height_obs:
                self._num_self_obs -= 1
            
            if self.self_obs_v == 3:
                self._num_self_obs += 6 * len(self.force_sensor_joints)

        elif self.humanoid_type in ['g1', 'pm01']:
            num_bodies = len(self._body_names)
            num_dofs = len(self._dof_names)
            self._dof_body_ids = np.arange(1, num_bodies)
            self._dof_offsets = list(range(num_dofs + 1))
            self._dof_obs_size = num_dofs  # raw scalar angles (no tan_norm)
            self._dof_size = num_dofs
            self._num_actions = num_dofs
            # max-coord obs: height(1) + num_bodies*(pos3+rot6+vel3+angvel3) - root_pos(3)
            self._num_self_obs = 1 + num_bodies * (3 + 6 + 3 + 3) - 3

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)

        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        if self.humanoid_type in ['g1']:
            head_body_name = "head_link"
        elif self.humanoid_type in ['pm01']:
            head_body_name = "link_head_yaw"
        else:
            head_body_name = "head"
        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles_list[0][0], head_body_name)
        if head_id >= 0:
            self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg.robot.asset["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            left_arm_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles_list[0][0], "left_lower_arm")
            self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])

        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _create_smpl_humanoid_xml(self, num_humanoids, smpl_robot, queue, pid):
        np.random.seed(np.random.randint(5002) * (pid + 1))
        res = {}
        for idx in num_humanoids:
            if self.has_shape_variation:
                gender_beta = self._amass_gender_betas[idx % self._amass_gender_betas.shape[0]]
            else:
                gender_beta = np.zeros(17)

            if flags.im_eval:
                gender_beta = np.zeros(17)
                    
            asset_id = uuid4()
            
            if not smpl_robot is None:
                asset_id = uuid4()
                asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"
                smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
                smpl_robot.write_xml(asset_file_real)
            else:
                asset_file_real = f"phc/data/assets/mjcf/smpl_{int(gender_beta[0])}_humanoid.xml"

            res[idx] = (gender_beta, asset_file_real)

        if not queue is None:
            queue.put(res)
        else:
            return res

    def _load_amass_gender_betas(self):
        if self._has_mesh:
            gender_betas_data = joblib.load("sample_data/amass_isaac_gender_betas.pkl")
            self._amass_gender_betas = np.array(list(gender_betas_data.values()))
        else:
            gender_betas_data = joblib.load("sample_data/amass_isaac_gender_betas_unique.pkl")
            self._amass_gender_betas = np.array(gender_betas_data)
            
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg.robot.asset["assetRoot"]
        asset_file = self.cfg.robot.asset["assetFileName"]
        self.humanoid_masses = []

        if (self.humanoid_type in ["smpl", "smplh", "smplx"]):
            self.humanoid_shapes = []
            self.humanoid_assets = []
            self.humanoid_limb_and_weights = []
            self.skeleton_trees = []
            robot_cfg = {
                "mesh": self._has_mesh,
                "replace_feet": self._replace_feet,
                "rel_joint_lm": self._has_jt_limit,
                "upright_start": self._has_upright_start,
                "remove_toe": self.remove_toe,
                "freeze_hand": self._freeze_hand, 
                "real_weight_porpotion_capsules": self._real_weight_porpotion_capsules,
                "real_weight_porpotion_boxes": self._real_weight_porpotion_boxes,
                "real_weight": self._real_weight,
                "masterfoot": self._masterfoot,
                "master_range": self._master_range,
                "big_ankle": self.big_ankle,
                "box_body": self._box_body,
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
                "model": self.humanoid_type,
                "sim": "isaacgym"
            }
            if os.path.exists("data/smpl"):
                robot = SMPL_Robot(
                    robot_cfg,
                    data_dir="data/smpl",
                )
            else:
                print("!!!!!!! SMPL files not found, loading pre-computed humanoid assets, only for demo purposes !!!!!!!")
                print("!!!!!!! SMPL files not found, loading pre-computed humanoid assets, only for demo purposes !!!!!!!")
                print("!!!!!!! SMPL files not found, loading pre-computed humanoid assets, only for demo purposes !!!!!!!")
                asset_root = "./"
                robot = None
                


            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = 100
            asset_options.vhacd_params.max_num_vertices_per_ch = 100

            if self.has_shape_variation:
                queue = mp.Queue()
                num_jobs = min(mp.cpu_count(), 64)
                
                if num_jobs <= 8:
                    num_jobs = 1
                if flags.debug:
                    num_jobs = 1
                res_acc = {}
                jobs = np.arange(num_envs)
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
                job_args = [jobs[i] for i in range(len(jobs))]

                for i in range(1, len(jobs)):
                    worker_args = (job_args[i], robot, queue, i)
                    worker = multiprocessing.Process(target=self._create_smpl_humanoid_xml, args=worker_args)
                    worker.start()
                res_acc.update(self._create_smpl_humanoid_xml(jobs[0], robot, None, 0))
                for i in tqdm(range(len(jobs) - 1)):
                    res = queue.get()
                    res_acc.update(res)

                for idx in np.arange(num_envs):
                    gender_beta, asset_file_real = res_acc[idx]
                    humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)
                    actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                    motor_efforts = [prop.motor_effort for prop in actuator_props]
                    
                    sk_tree = SkeletonTree.from_mjcf(asset_file_real)

                    # create force sensors at the feet
                    if self.self_obs_v == 3:
                        self.create_humanoid_force_sensors(humanoid_asset, self.force_sensor_joints)
                    
                    self.humanoid_shapes.append(torch.from_numpy(gender_beta).float())
                    self.humanoid_assets.append(humanoid_asset)
                    self.skeleton_trees.append(sk_tree)

                if not robot is None:
                    robot.remove_geoms()  # Clean up the geoms

                self.humanoid_shapes = torch.vstack(self.humanoid_shapes).to(self.device)
            else:
                gender_beta, asset_file_real = self._create_smpl_humanoid_xml([0], robot, None, 0)[0]
                if 'load_from_xml' in self.cfg.robot.asset and self.cfg.robot.asset.load_from_xml == True:
                    asset_root = "./"
                    asset_file_real = asset_file

                sk_tree = SkeletonTree.from_mjcf(asset_file_real)

                humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)
                actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                motor_efforts = [prop.motor_effort for prop in actuator_props]

                
                
                self.humanoid_shapes = torch.tensor(np.array([gender_beta] * num_envs)).float().to(self.device)
                self.humanoid_assets_list = [[humanoid_asset] * self.num_agents] * num_envs
                
                self.skeleton_trees = [sk_tree] * num_envs

        elif self.humanoid_type in ['g1', 'pm01']:
            self.humanoid_limb_and_weights = []
            xml_asset_path = os.path.join(asset_root, asset_file)
            robot_file = os.path.join(asset_root, self.cfg.robot.asset.urdfFileName)
            sk_tree = SkeletonTree.from_mjcf(xml_asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.replace_cylinder_with_capsule = True
            asset_options.collapse_fixed_joints = True

            robot_asset_root = os.path.dirname(robot_file)
            robot_asset_file = os.path.basename(robot_file)
            humanoid_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)

            if self.humanoid_type in ['pm01']:
                actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                motor_efforts = [prop.motor_effort for prop in actuator_props]
            else:
                motor_efforts = [360] * len(self._dof_names)

            right_foot_name = self.cfg.robot.get("right_foot_name", "right_ankle_roll_link")
            left_foot_name = self.cfg.robot.get("left_foot_name", "left_ankle_roll_link")
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, right_foot_name)
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, left_foot_name)
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
            self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

            self.humanoid_shapes = torch.tensor(np.zeros((num_envs, 10))).float().to(self.device)
            self.humanoid_assets_list = [[humanoid_asset] * self.num_agents] * num_envs
            self.skeleton_trees = [sk_tree] * num_envs

            if self.humanoid_type in ['g1']:
                # Initialize g1 PD gains (used in _build_env)
                self.p_gains = to_torch([
                    100.0, 100.0, 100.0,  # left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint
                    200.0,                # left_knee_joint
                    20.0, 20.0,          # left_ankle_pitch_joint, left_ankle_roll_joint
                    100.0, 100.0, 100.0,  # right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint
                    200.0,                # right_knee_joint
                    20.0, 20.0,          # right_ankle_pitch_joint, right_ankle_roll_joint
                    400.0, 400.0, 400.0,  # waist_yaw_joint, waist_roll_joint, waist_pitch_joint
                    90.0, 60.0, 20.0,     # left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint
                    60.0,                 # left_elbow_joint
                    4.0, 4.0, 4.0,        # left_wrist_roll_joint, left_wrist_pitch_joint, left_wrist_yaw_joint
                    90.0, 60.0, 20.0,     # right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint
                    60.0,                 # right_elbow_joint
                    4.0, 4.0, 4.0         # right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint
                ], device=self.device)

                # d_gains (damping per DOF)
                self.d_gains = to_torch([
                    2.5, 2.5, 2.5,       # left hip
                    5.0,                 # left knee
                    0.2, 0.1,            # left ankle
                    2.5, 2.5, 2.5,       # right hip
                    5.0,                 # right knee
                    0.2, 0.1,            # right ankle
                    5.0, 5.0, 5.0,       # waist
                    2.0, 1.0, 0.4,       # left shoulder
                    1.0,                 # left elbow
                    0.2, 0.2, 0.2,       # left wrist
                    2.0, 1.0, 0.4,       # right shoulder
                    1.0,                 # right elbow
                    0.2, 0.2, 0.2        # right wrist
                ], device=self.device)
                self.torque_limits_hard_coded = to_torch([
                    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                    88.0, 50.0, 50.0,
                    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                ], device=self.device)
                self.default_dof_pos = torch.tensor([[
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]], device=self.device)
            elif self.humanoid_type in ['pm01']:
                num_dofs = len(self._dof_names)
                self.p_gains = torch.zeros(num_dofs, device=self.device)
                self.d_gains = torch.zeros(num_dofs, device=self.device)
                # IsaacGym returns max-float effort for this MJCF, so use PM01 actuatorfrcrange limits.
                self.torque_limits_hard_coded = to_torch([
                    164.0, 164.0, 52.0, 164.0, 52.0, 52.0,
                    164.0, 164.0, 52.0, 164.0, 52.0, 52.0,
                    52.0, 52.0, 52.0, 52.0, 52.0, 52.0,
                    52.0, 52.0, 52.0, 52.0, 52.0, 52.0
                ], device=self.device)
                default_dof_pos = torch.zeros(num_dofs, device=self.device)
                for i, name in enumerate(self._dof_names):
                    if "hip_pitch" in name:
                        self.p_gains[i], self.d_gains[i], default_dof_pos[i] = 100.0, 2.5, -0.24 # 70.0, 7.0, -0.24
                    elif "hip_roll" in name:
                        self.p_gains[i], self.d_gains[i] = 100.0, 2.5  # 50.0, 5.0
                    elif "hip_yaw" in name:
                        self.p_gains[i], self.d_gains[i] = 100.0, 2.5 # 50.0, 5.0
                    elif "knee_pitch" in name:
                        self.p_gains[i], self.d_gains[i], default_dof_pos[i] = 200.0, 5.0, 0.48 # 70.0, 7.0, 0.48
                    elif "ankle_pitch" in name:
                        self.p_gains[i], self.d_gains[i], default_dof_pos[i] = 35.0, 0.6, -0.24
                    elif "ankle_roll" in name:
                        self.p_gains[i], self.d_gains[i] = 35.0, 0.6
                    elif "waist" in name:
                        self.p_gains[i], self.d_gains[i] = 400.0, 5.0 # 120.0, 3.0
                    elif "shoulder_pitch" in name:
                        self.p_gains[i], self.d_gains[i] = 90.0, 2.0
                    elif "shoulder_roll" in name:
                        self.p_gains[i], self.d_gains[i] = 60.0, 1.0
                    elif "shoulder_yaw" in name:
                        self.p_gains[i], self.d_gains[i] = 30.0, 0.8
                    elif "elbow_pitch" in name:
                        self.p_gains[i], self.d_gains[i] = 60.0, 1.2
                    elif "elbow_yaw" in name:
                        self.p_gains[i], self.d_gains[i] = 50.0, 1.0
                    else:
                        self.p_gains[i], self.d_gains[i] = 20.0, 0.5
                self.default_dof_pos = default_dof_pos.unsqueeze(0)

        else:
            asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            #asset_options.fix_base_link = True
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
            motor_efforts = [prop.motor_effort for prop in actuator_props]

            # create force sensors at the feet
            self.create_humanoid_force_sensors(humanoid_asset, ["right_foot", "left_foot"])
            self.humanoid_assets = [humanoid_asset] * num_envs

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_asset_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles_list = [] #num_env, num_agents,
        
        
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets_list[i])
            self.envs.append(env_ptr)
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)
        print("Humanoid Weights", self.humanoid_masses[:10])

        # max_agg_bodies, max_agg_shapes = 220, 220
        # for i in range(self.num_envs):
        #     # create env instance
        #     env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
        #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
        #     self._build_env(i, env_ptr, self.humanoid_assets_list[i])
        #     self.gym.end_aggregate(env_ptr)
            
        #     self.envs.append(env_ptr)
        # self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)
        # print("Humanoid Weights", self.humanoid_masses[:10])

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles_list[0][0])

        ######################################## Joint frictino
        # dof_prop['friction'][:] = 10
        # self.gym.set_actor_dof_properties(self.envs[0], self.humanoid_handles[0], dof_prop)

        #check
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        
        if self.control_mode == "pd":
            if self.humanoid_type in ['g1', 'pm01']:
                self.torque_limits = self.torque_limits_hard_coded.clone()
            else:
                self.torque_limits = torch.ones_like(self.dof_limits_upper) * 1000 # ZL: hacking 
        
        if self.humanoid_type in ['g1', 'pm01']:
            self._process_dof_props(dof_prop)

        if self.control_mode in ["pd", "isaac_pd"]:
            self._build_pd_action_offset_scale()
        return

    def create_humanoid_force_sensors(self, humanoid_asset, sensor_joint_names):
        for jt in sensor_joint_names:
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, jt)
            sensor_pose = gymapi.Transform()
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_constraint_solver_forces = True # for example contacts 
            sensor_options.use_world_frame = False # Local frame so we can directly send it to computation. 
            # These are the default values. 
            
            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose, sensor_options)
            
        return
    
    def _process_dof_props(self, props):
        """Store DOF limits and torque limits from URDF properties (ported from PULSE for G1 support)."""
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(props)):
            self.dof_pos_limits[i, 0] = props["lower"][i].item()
            self.dof_pos_limits[i, 1] = props["upper"][i].item()
            if self.dof_pos_limits[i, 0] == 0 and self.dof_pos_limits[i, 1] == 0:
                self.dof_pos_limits[i, 0] = -np.pi
                self.dof_pos_limits[i, 1] = np.pi
            self.dof_vel_limits[i] = props["velocity"][i].item()
            if hasattr(self, 'torque_limits_hard_coded'):
                self.torque_limits[i] = self.torque_limits_hard_coded[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r
            self.dof_pos_limits[i, 1] = m + 0.5 * r
        return props

    def _build_env(self, env_id, env_ptr, humanoid_asset_list):
        if self._divide_group or flags.divide_group:
            col_group = self._group_ids[env_id]
        else:
            col_group = env_id  # no inter-environment collision

        col_filter = 0
        if (self.humanoid_type in ["smpl", "smplh", "smplx", "g1", "pm01"]) and (not self._has_self_collision):
            col_filter = 1
        

        asset_file = self.cfg.robot.asset["assetFileName"]
        if (asset_file == "mjcf/ov_humanoid.xml" or asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            char_h = 0.927
        else:
            char_h = 0.89

        pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)  # ZL: segfault if we do not randomize the position

        humanoid_handles = []
        for i in range(self.num_agents):
            start_pose = gymapi.Transform()

            start_pose.p = gymapi.Vec3(*pos)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            if i%2==1:
                start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

            actor_col_filter = col_filter
            if self.humanoid_type in ['g1', 'pm01'] and (not self._has_self_collision):
                actor_col_filter = 1 << i

            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset_list[i], start_pose, "humanoid", col_group, actor_col_filter, 0)
        
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

            if self.humanoid_type in ['g1']:
                # pass
                # For G1: only recolor torso_link to distinguish the two agents; all other
                # bodies keep the colors defined in the URDF/XML asset.
                agent_colors = [gymapi.Vec3(0.54, 0.85, 0.2), gymapi.Vec3(0.97, 0.38, 0.06)]
                # # black and white
                # agent_colors = [gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0)]
                torso_idx = 15 # self.body_names.index("torso_link")
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, torso_idx, gymapi.MESH_VISUAL, agent_colors[i % 2])
            elif self.humanoid_type in ['pm01']:
                agent_colors = [gymapi.Vec3(0.54, 0.85, 0.2), gymapi.Vec3(0.97, 0.38, 0.06)]
                torso_idx = 13 # self.body_names.index("torso_link")
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, torso_idx, gymapi.MESH_VISUAL, agent_colors[i % 2])
                head_idx = 24
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, head_idx, gymapi.MESH_VISUAL, agent_colors[i % 2])
            else:
                for j in range(self.num_bodies):
                    # self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, color_vec)
                    if i%2==0:
                        self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset_list[i])

            if self.has_shape_variation:
                pd_scale = humanoid_mass / self.cfg['env'].get('default_humanoid_mass', 77.0 if self._real_weight else 35.0)
                self._kp_scale = pd_scale * self._kp_scale
                self._kd_scale = pd_scale * self._kd_scale
            
            if (self.control_mode == "isaac_pd"):
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_POS
                dof_prop['stiffness'] *= self._kp_scale
                dof_prop['damping'] *= self._kd_scale
                if self.humanoid_type in ['g1', 'pm01']:
                    dof_prop['stiffness'] = self.p_gains.cpu().numpy()
                    dof_prop['damping'] = self.d_gains.cpu().numpy()

            else:
                if self.control_mode == "pd":
                    if self.humanoid_type in ['g1', 'pm01']:
                        # G1/PM01: match PULSE behavior - only set drive mode to EFFORT.
                        # Do NOT override friction/damping/velocity - PULSE preserves MJCF defaults.
                        # The combat_prior was trained with MJCF friction=0.1; setting friction=1
                        # here would add 10x more dry friction, destabilizing PD control.
                        self.kp_gains = self.p_gains
                        self.kd_gains = self.d_gains
                    else:
                        self.kp_gains = to_torch(self._kp_scale * dof_prop['stiffness']/4, device=self.device)
                        self.kd_gains = to_torch(self._kd_scale * dof_prop['damping']/4, device=self.device)
                        dof_prop['velocity'][:] = 100
                        dof_prop['stiffness'][:] = 0
                        dof_prop['friction'][:] = 1
                        dof_prop['damping'][:] = 0.001
                elif self.control_mode == "force":
                    dof_prop['velocity'][:] = 100
                    dof_prop['stiffness'][:] = 0
                    dof_prop['friction'][:] = 1
                    dof_prop['damping'][:] = 0.001
                    
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_EFFORT

            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        

            if self.humanoid_type in ["smpl", "smplh", "smplx"] and self._has_self_collision:
                # compliance_vals = [0.1] * 24
                # thickness_vals = [1.0] * 24
                if self._has_mesh:
                    filter_ints = [0, 1, 224, 512, 384, 1, 1792, 64, 1056, 4096, 6, 6168, 0, 2048, 0, 20, 0, 0, 0, 0, 10, 0, 0, 0]
                else:
                    if self.humanoid_type == "smpl":
                        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif self.humanoid_type in ["smplh", "smplx"]:
                        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        
                props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
                

                # assert (len(filter_ints) == len(props))
                if len(props) > len(filter_ints):
                    filter_ints += [0] * (len(props) - len(filter_ints))

                for p_idx in range(len(props)):
                    props[p_idx].filter = filter_ints[p_idx]
                    

                self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

            elif self.humanoid_type in ['g1'] and self._has_self_collision:
                # filter_ints = [
                #     27, 7, 6, 4, 4, 4, 4, 4, 4, 4,
                #     56, 48, 32, 32, 32, 32, 32, 32, 32,
                #     603, 603, 603, 603,
                #     192, 384, 256,
                #     1536, 3072, 2048,
                # ]
                filter_ints = [
                            27,                             # pelvis:  bits 0,1,3,4 (1+2+8+16)
                            7,                              # l_thigh: bits 0,1,2   (1+2+4)
                            6,                              # l_shin:  bits 1,2     (2+4)
                            4, 4, 4, 4, 4, 4, 4,           # l_foot×7: bit 2
                            56,                             # r_thigh: bits 3,4,5   (8+16+32)
                            48,                             # r_shin:  bits 4,5     (16+32)
                            32, 32, 32, 32, 32, 32, 32,    # r_foot×7: bits 4,5 (=32 dominant)
                            603, 603, 603, 603,             # torso+head: bits 0,1,3,4,6,9 (1+2+8+16+64+512)
                            192,                            # l_sh_yaw: bits 6,7   (64+128)
                            384,                            # l_elbow:  bits 7,8   (128+256)
                            256,                            # l_hand:   bit 8
                            1536,                           # r_sh_yaw: bits 9,10  (512+1024)
                            3072,                           # r_elbow:  bits 10,11 (1024+2048)
                            2048,                           # r_hand:   bit 11
                        ]
                props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
                if len(props) > len(filter_ints):
                    filter_ints += [0] * (len(props) - len(filter_ints))
                filter_width = max(filter_ints).bit_length()
                filter_shift = filter_width * i if self.num_agents == 2 else 0
                for p_idx in range(len(props)):
                    props[p_idx].filter = filter_ints[p_idx] << filter_shift
                self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)
                

            humanoid_handles.append(humanoid_handle)
        self.humanoid_handles_list.append(humanoid_handles)
        


        mass_ind = [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        humanoid_mass = np.sum(mass_ind)
        self.humanoid_masses.append(humanoid_mass)

        curr_skeleton_tree = self.skeleton_trees[env_id]
        limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
        masses = torch.tensor(mass_ind)

        # humanoid_limb_weight = torch.cat([limb_lengths[1:], masses])

        if len(self.limb_weight_group) > 0:
            limb_lengths = [limb_lengths[group].sum() for group in self.limb_weight_group]
            masses = [masses[group].sum() for group in self.limb_weight_group]
            humanoid_limb_weight = torch.tensor(limb_lengths + masses)
        else:
            humanoid_limb_weight = torch.tensor([limb_lengths.sum().item(), masses.sum().item()])
        self.humanoid_limb_and_weights.append(humanoid_limb_weight)  # ZL: attach limb lengths and full body weight.

        # if self.humanoid_type in ["smpl", "smplh", "smplx"]:
        #     gender = self.humanoid_shapes[env_id, 0].long()
        #     percentage = 1 - np.clip((humanoid_mass - 70) / 70, 0, 1)
        #     if gender == 0:
        #         gender = 1
        #         color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Greens"))
        #     elif gender == 1:
        #         gender = 2
        #         color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Blues"))
        #     elif gender == 2:
        #         gender = 0
        #         color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Reds"))

        #     # color = torch.zeros(3)
        #     # color[gender] = 1 - np.clip((humanoid_mass - 70) / 70, 0, 1)
        #     if flags.test:
        #         color_vec = gymapi.Vec3(*agt_color(env_id + 0))
        #     # if env_id == 0:
        #     #     color_vec = gymapi.Vec3(0.23192618223760095, 0.5456516724336793, 0.7626143790849673)
        #     # elif env_id == 1:
        #     #     color_vec = gymapi.Vec3(0.907912341407151, 0.20284505959246443, 0.16032295271049596)

        # else:
        #     color_vec = gymapi.Vec3(0.54, 0.85, 0.2)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if not self._bias_offset:
                if (dof_size == 3):
                    curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                    curr_low = np.max(np.abs(curr_low))
                    curr_high = np.max(np.abs(curr_high))
                    curr_scale = max([curr_low, curr_high])
                    curr_scale = 1.2 * curr_scale
                    curr_scale = min([curr_scale, np.pi])

                    lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                    lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

                    #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                    #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

                elif (dof_size == 1):
                    curr_low = lim_low[dof_offset]
                    curr_high = lim_high[dof_offset]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.7 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                    lim_low[dof_offset] = curr_low
                    lim_high[dof_offset] = curr_high
            else:
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset:(dof_offset + dof_size)] = curr_low
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            self._L_knee_dof_idx = self._dof_names.index("L_Knee") * 3 + 1
            self._R_knee_dof_idx = self._dof_names.index("R_Knee") * 3 + 1

            # ZL: Modified SMPL to give stronger knee
            self._pd_action_scale[self._L_knee_dof_idx] = 5
            self._pd_action_scale[self._R_knee_dof_idx] = 5

            if self._has_smpl_pd_offset:
                if self._has_upright_start:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 2
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = np.pi / 2
                else:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 6
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3 + 2] = -np.pi / 2
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = -np.pi / 3
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3 + 2] = np.pi / 2

        elif self.humanoid_type in ['g1', 'pm01']:
            # G1/PM01 joints are single-axis; use zero offset so action=0 maps to default_dof_pos
            self._pd_action_offset[:] = 0

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, self._rigid_body_pos, self.max_episode_length, self._enable_early_termination, self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        return

    def _compute_observations(self, env_ids=None):
        obs_list = self._compute_humanoid_obs(env_ids)

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (ENABLE_MAX_COORD_OBS):
            obs_list = []
            for i in range(self.num_agents):
                if (env_ids is None):
                    body_pos = self._rigid_body_pos_list[i]
                    body_rot = self._rigid_body_rot_list[i]
                    body_vel = self._rigid_body_vel_list[i]
                    body_ang_vel = self._rigid_body_ang_vel_list[i]

                    if self.self_obs_v == 2:
                        # body_pos = torch.cat([self._rigid_body_pos_hist, body_pos.unsqueeze(1)], dim=1)
                        # body_rot = torch.cat([self._rigid_body_rot_hist, body_rot.unsqueeze(1)], dim=1)
                        # body_vel = torch.cat([self._rigid_body_vel_hist, body_vel.unsqueeze(1)], dim=1)
                        # body_ang_vel = torch.cat([self._rigid_body_ang_vel_hist, body_ang_vel.unsqueeze(1)], dim=1)
                        body_pos = torch.cat([self._rigid_body_pos_hist_list[i], body_pos.unsqueeze(1)], dim=1)
                        body_rot = torch.cat([self._rigid_body_rot_hist_list[i], body_rot.unsqueeze(1)], dim=1)
                        body_vel = torch.cat([self._rigid_body_vel_hist_list[i], body_vel.unsqueeze(1)], dim=1)
                        body_ang_vel = torch.cat([self._rigid_body_ang_vel_hist_list[i], body_ang_vel.unsqueeze(1)], dim=1)
                        # rigid_body_pos_hist = body_pos.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_rot_hist = body_rot.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_vel_hist = body_vel.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_ang_vel_hist = body_ang_vel.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # body_pos = torch.cat([rigid_body_pos_hist, body_pos.unsqueeze(1)], dim=1)
                        # body_rot = torch.cat([rigid_body_rot_hist, body_rot.unsqueeze(1)], dim=1)
                        # body_vel = torch.cat([rigid_body_vel_hist, body_vel.unsqueeze(1)], dim=1)
                        # body_ang_vel = torch.cat([rigid_body_ang_vel_hist, body_ang_vel.unsqueeze(1)], dim=1)
                    # if self.self_obs_v == 3:
                    #     force_sensor_readings = self.vec_sensor_tensor
                        
                        
                else:
                    body_pos = self._rigid_body_pos_list[i][env_ids]
                    body_rot = self._rigid_body_rot_list[i][env_ids]
                    body_vel = self._rigid_body_vel_list[i][env_ids]
                    body_ang_vel = self._rigid_body_ang_vel_list[i][env_ids]

                    if self.self_obs_v == 2:
                        # body_pos = torch.cat([self._rigid_body_pos_hist[env_ids], body_pos.unsqueeze(1)], dim=1)
                        # body_rot = torch.cat([self._rigid_body_rot_hist[env_ids], body_rot.unsqueeze(1)], dim=1)
                        # body_vel = torch.cat([self._rigid_body_vel_hist[env_ids], body_vel.unsqueeze(1)], dim=1)
                        # body_ang_vel = torch.cat([self._rigid_body_ang_vel_hist[env_ids], body_ang_vel.unsqueeze(1)], dim=1)
                        body_pos = torch.cat([self._rigid_body_pos_hist_list[i][env_ids], body_pos.unsqueeze(1)], dim=1)
                        body_rot = torch.cat([self._rigid_body_rot_hist_list[i][env_ids], body_rot.unsqueeze(1)], dim=1)
                        body_vel = torch.cat([self._rigid_body_vel_hist_list[i][env_ids], body_vel.unsqueeze(1)], dim=1)
                        body_ang_vel = torch.cat([self._rigid_body_ang_vel_hist_list[i][env_ids], body_ang_vel.unsqueeze(1)], dim=1)
                        # rigid_body_pos_hist = body_pos.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_rot_hist = body_rot.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_vel_hist = body_vel.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # rigid_body_ang_vel_hist = body_ang_vel.unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
                        # body_pos = torch.cat([rigid_body_pos_hist[env_ids], body_pos.unsqueeze(1)], dim=1)
                        # body_rot = torch.cat([rigid_body_rot_hist[env_ids], body_rot.unsqueeze(1)], dim=1)
                        # body_vel = torch.cat([rigid_body_vel_hist[env_ids], body_vel.unsqueeze(1)], dim=1)
                        # body_ang_vel = torch.cat([rigid_body_ang_vel_hist[env_ids], body_ang_vel.unsqueeze(1)], dim=1)
                    # if self.self_obs_v == 3:
                    #     force_sensor_readings = self.vec_sensor_tensor[env_ids]
                


                if self.humanoid_type in ["g1", "pm01", "smpl", "smplh", "smplx"] :
                    if (env_ids is None):
                        body_shape_params = self.humanoid_shapes[:, :-6] if self.humanoid_type in ["smpl", "smplh", "smplx"] else self.humanoid_shapes
                        limb_weights = self.humanoid_limb_and_weights
                    else:
                        body_shape_params = self.humanoid_shapes[env_ids, :-6] if self.humanoid_type in ["smpl", "smplh", "smplx"] else self.humanoid_shapes[env_ids]
                        limb_weights = self.humanoid_limb_and_weights[env_ids]
                        
                    if self.self_obs_v == 1:
                        obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, body_shape_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)
                    elif self.self_obs_v == 2:
                        obs = compute_humanoid_observations_smpl_max_v2(body_pos, body_rot, body_vel, body_ang_vel, body_shape_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs, self.past_track_steps + 1)
                    obs_list.append(obs)

                # elif self.humanoid_type in ['g1']:
                #     # Must match PULSE training: use smpl_max variant which has correct
                #     # local_root_obs semantics (local_root_obs=True → keep heading-relative rotation).
                #     # compute_humanoid_observations_max has INVERTED local_root_obs logic and would
                #     # replace root rotation with global rotation, causing wrong obs for the prior.
                #     if (env_ids is None):
                #         body_shape_params = self.humanoid_shapes
                #         limb_weights = self.humanoid_limb_and_weights
                #     else:
                #         body_shape_params = self.humanoid_shapes[env_ids]
                #         limb_weights = self.humanoid_limb_and_weights[env_ids]
                #     obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, body_shape_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)
                #     obs_list.append(obs)
            
        # else:
        #     if (env_ids is None):
        #         root_pos = self._rigid_body_pos[:, 0, :]
        #         root_rot = self._rigid_body_rot[:, 0, :]
        #         root_vel = self._rigid_body_vel[:, 0, :]
        #         root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
        #         dof_pos = self._dof_pos
        #         dof_vel = self._dof_vel
        #         key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        #     else:
        #         root_pos = self._rigid_body_pos[env_ids][:, 0, :]
        #         root_rot = self._rigid_body_rot[env_ids][:, 0, :]
        #         root_vel = self._rigid_body_vel[env_ids][:, 0, :]
        #         root_ang_vel = self._rigid_body_ang_vel[env_ids][:, 0, :]
        #         dof_pos = self._dof_pos[env_ids]
        #         dof_vel = self._dof_vel[env_ids]
        #         key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

        #     if (self.humanoid_type in ["smpl", "smplh", "smplx"] ) and self.self.has_shape_obs:
        #         if (env_ids is None):
        #             body_shape_params = self.humanoid_shapes
        #         else:
        #             body_shape_params = self.humanoid_shapes[env_ids]
        #         obs = compute_humanoid_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._dof_obs_size, self._dof_offsets, body_shape_params, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs)
        #     else:
        #         obs = compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets)
            # To access the data from two running agents, you can access these attributes:
        # import pdb;pdb.set_trace()
        # # For root position (rootpos) - shape [num_envs, 3] for each agent
        # rootpos_agent_0 = self._humanoid_root_states_list[0][:, 0:3]  # [num_envs, 3]
        # rootpos_agent_1 = self._humanoid_root_states_list[1][:, 0:3]  # [num_envs, 3]

        # # For root rotation (rootrot) - shape [num_envs, 4] for each agent  
        # rootrot_agent_0 = self._humanoid_root_states_list[0][:, 3:7]  # [num_envs, 4] - quaternion
        # rootrot_agent_1 = self._humanoid_root_states_list[1][:, 3:7]  # [num_envs, 4] - quaternion

        # # For DOF positions (dofpos) - shape [num_envs, num_dof] for each agent
        # dofpos_agent_0 = self._dof_pos_list[0]  # [num_envs, num_dof]
        # dofpos_agent_1 = self._dof_pos_list[1]  # [num_envs, num_dof]


        return obs_list

    def _reset_actors(self, env_ids):

        for i in range(self.num_agents):
            self._humanoid_root_states_list[i][env_ids] = self._initial_humanoid_root_states_list[i][env_ids]
            self._dof_pos_list[i][env_ids] = self._initial_dof_pos_list[i][env_ids]
            self._dof_vel_list[i][env_ids] = self._initial_dof_vel_list[i][env_ids]

        
        return

    def pre_physics_step(self, actions):
        # if flags.debug:
            # actions *= 0
        self.actions = actions.to(self.device).clone()
        if len(self.actions.shape) == 1:
            self.actions = self.actions[None, ]
            
        if (self._pd_control):
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                if self.reduce_action:
                    actions_full = torch.zeros([actions.shape[0], self._dof_size]).to(self.device)
                    actions_full[:, self.action_idx] = self.actions
                    pd_tar = self._action_to_pd_targets(actions_full)

                else:
                    pd_tar = self._action_to_pd_targets(self.actions)
                    if self._freeze_hand:
                        pd_tar[:, self._dof_names.index("L_Hand") * 3:(self._dof_names.index("L_Hand") * 3 + 3)] = 0
                        pd_tar[:, self._dof_names.index("R_Hand") * 3:(self._dof_names.index("R_Hand") * 3 + 3)] = 0
                    if self._freeze_toe:
                        pd_tar[:, self._dof_names.index("L_Toe") * 3:(self._dof_names.index("L_Toe") * 3 + 3)] = 0
                        pd_tar[:, self._dof_names.index("R_Toe") * 3:(self._dof_names.index("R_Toe") * 3 + 3)] = 0
            elif self.humanoid_type in ['g1', 'pm01']:
                pd_tar = self._action_to_pd_targets(self.actions)
            self.actions = pd_tar
            chunks = torch.chunk(pd_tar, self.num_agents, dim=0)
            pd_tar= torch.cat(chunks, dim=-1)
            # pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            # for i in range
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        
        else:
            if self.control_mode == "force":
                actions_full = self.actions
                forces = actions_full * self.motor_efforts.unsqueeze(0) * self.power_scale
                force_tensor = gymtorch.unwrap_tensor(forces)
                self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
            elif self.control_mode == "pd":
                if self.humanoid_type in ['g1', 'pm01']:
                    # Match PULSE: raw network output is passed directly to _compute_torques,
                    # only clipped. _action_to_pd_targets would multiply by _pd_action_scale
                    # (~3-4x for single-axis joints) which was NOT done during combat prior training.
                    clip_actions = 10
                    self.pd_tar = torch.clip(self.actions, -clip_actions, clip_actions)
                else:
                    self.pd_tar = self._action_to_pd_targets(self.actions)
        return
    
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions  [num_envs * num_agents, num_dof]
        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.humanoid_type in ['g1', 'pm01']:
            # G1/PM01 multi-agent: concatenate per-agent DOF state lists
            # actions is the PD target delta; add default_dof_pos to get absolute target
            dof_pos = torch.cat(self._dof_pos_list, dim=0)  # [num_envs * num_agents, num_dof]
            dof_vel = torch.cat(self._dof_vel_list, dim=0)
            torques = self.kp_gains * (actions + self.default_dof_pos - dof_pos) - self.kd_gains * dof_vel
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            # Isaac Gym DOF tensor is env-major; re-pack from agent-major to env-major.
            if self.num_agents > 1:
                torques = torch.cat(torch.chunk(torques, self.num_agents, dim=0), dim=-1)
            return torques

        control_type = "P" # self.cfg.control.control_type
        if control_type=="P": # default 
            torques = self.kp_gains*(actions - self._dof_pos) - self.kd_gains*self._dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # if self.cfg.domain_rand.randomize_torque_rfi:
            # torques = torques + (torch.rand_like(torques)*2.-1.) * self.cfg.domain_rand.rfi_lim * self.torque_limits
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    
    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.control_i = i
            self.render()
            if not self.paused and self.enable_viewer_sync:
                if self.control_mode == "pd": #### Using simple pd controller. 
                    self.torques = self._compute_torques(self.pd_tar)
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                else:
                    self.gym.simulate(self.sim)
                    
        return
    
    
    
    
    def _init_tensor_history(self, env_ids):
        # self._rigid_body_pos_hist[env_ids] = self._rigid_body_pos[env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # self._rigid_body_rot_hist[env_ids] = self._rigid_body_rot[env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # self._rigid_body_vel_hist[env_ids] = self._rigid_body_vel[env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # self._rigid_body_ang_vel_hist[env_ids] = self._rigid_body_ang_vel[env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # self.count = 0

        for i in range(self.num_agents):
            self._rigid_body_pos_hist_list[i][env_ids] = self._rigid_body_pos_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_rot_hist_list[i][env_ids] = self._rigid_body_rot_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_vel_hist_list[i][env_ids] = self._rigid_body_vel_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_ang_vel_hist_list[i][env_ids] = self._rigid_body_ang_vel_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # return
    
    def _update_tensor_history(self):
            # self._rigid_body_pos_hist = torch.cat([self._rigid_body_pos_hist[:, 1:], self._rigid_body_pos.unsqueeze(1)], dim=1)
            # self._rigid_body_rot_hist = torch.cat([self._rigid_body_rot_hist[:, 1:], self._rigid_body_rot.unsqueeze(1)], dim=1)
            # self._rigid_body_vel_hist = torch.cat([self._rigid_body_vel_hist[:, 1:], self._rigid_body_vel.unsqueeze(1)], dim=1)
            # self._rigid_body_ang_vel_hist = torch.cat([self._rigid_body_ang_vel_hist[:, 1:], self._rigid_body_ang_vel.unsqueeze(1)], dim=1)
        # print("self.count:", self.count)
        # if self.count == 0:
        #     for i in range(self.num_agents):
        #         self._rigid_body_pos_hist_list[i] = self._rigid_body_pos_list[i].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        #         self._rigid_body_rot_hist_list[i] = self._rigid_body_rot_list[i].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        #         self._rigid_body_vel_hist_list[i] = self._rigid_body_vel_list[i].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        #         self._rigid_body_ang_vel_hist_list[i] = self._rigid_body_ang_vel_list[i].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # else:
        # For envs where self.progress_buf is 0, reset history to repeats of _rigid_body_pos_list.
        env_ids = (self.progress_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        for i in range(self.num_agents):
            self._rigid_body_pos_hist_list[i] = torch.cat([self._rigid_body_pos_hist_list[i][:, 1:], self._rigid_body_pos_list[i].unsqueeze(1)], dim=1)
            self._rigid_body_rot_hist_list[i] = torch.cat([self._rigid_body_rot_hist_list[i][:, 1:], self._rigid_body_rot_list[i].unsqueeze(1)], dim=1)
            self._rigid_body_vel_hist_list[i] = torch.cat([self._rigid_body_vel_hist_list[i][:, 1:], self._rigid_body_vel_list[i].unsqueeze(1)], dim=1)
            self._rigid_body_ang_vel_hist_list[i] = torch.cat([self._rigid_body_ang_vel_hist_list[i][:, 1:], self._rigid_body_ang_vel_list[i].unsqueeze(1)], dim=1)
            
            self._rigid_body_pos_hist_list[i][env_ids] = self._rigid_body_pos_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_rot_hist_list[i][env_ids] = self._rigid_body_rot_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_vel_hist_list[i][env_ids] = self._rigid_body_vel_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
            self._rigid_body_ang_vel_hist_list[i][env_ids] = self._rigid_body_ang_vel_list[i][env_ids].unsqueeze(1).repeat(1, self.past_track_steps, 1, 1)
        # self.count +=1
        # Optional debug: when flags.debug is set, print max absolute difference between
        # the history's last entry and the current rigid body positions to confirm sync.

    def save_recorded_pt(self):
        # Stack the lists into single tensors
        final_data = {
            "rootpos_agent1": torch.stack(self.recording_dict["rootpos_agent1"]), # [frames, 3]
            "rootrot_agent1": torch.stack(self.recording_dict["rootrot_agent1"]), # [frames, 4]
            "dofpos_agent1":  torch.stack(self.recording_dict["dofpos_agent1"]),  # [frames, 23, 3]
            "dofrot_agent1":  torch.stack(self.recording_dict["dofrot_agent1"]),  # [frames, 23, 4]
            "rootpos_agent2": torch.stack(self.recording_dict["rootpos_agent2"]), # [frames, 3]
            "rootrot_agent2": torch.stack(self.recording_dict["rootrot_agent2"]), # [frames, 4]
            "dofpos_agent2":  torch.stack(self.recording_dict["dofpos_agent2"]),  # [frames, 23, 3]
            "dofrot_agent2":  torch.stack(self.recording_dict["dofrot_agent2"]),  # [frames, 23, 4]
        }

        save_path = "multi_agent_trajectories_1800frames_6.pt"
        torch.save(final_data, save_path)
        print(f"!!! Success: Recorded {self.max_recording_frames} frames and saved to {save_path} !!!")

    def post_physics_step(self):
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        
        if self.self_obs_v == 2:
            self._update_tensor_history()
        
        if not self.paused:
            self.progress_buf += 1
            
        self._refresh_sim_tensors()
        self._compute_reward(self.actions)  # ZL swapped order of reward & objecation computes. should be fine.
        self._compute_reset() 
        
        self._compute_observations()  # observation for the next step.

        #repeat for op
        self.extras["terminate"] = self._terminate_buf.repeat(self.num_agents)
        self.extras["reward_raw"] = self.reward_raw.detach()
    
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        # if not self.paused and self.frame_index < self.max_recording_frames:
        #     # We record data from the first environment (env_id = 0)
        #     env_idx = 0 
        #     # import pdb;pdb.set_trace()
        #     # Agent 1 (Index 0 in lists)
        #     self.recording_dict["rootpos_agent1"].append(self._humanoid_root_states_list[0][env_idx, 0:3].clone().cpu())
        #     self.recording_dict["rootrot_agent1"].append(self._humanoid_root_states_list[0][env_idx, 3:7].clone().cpu())
        #     # Reshape DOF from [69] to [23, 3]
        #     # self.recording_dict["dofpos_agent1"].append(self._dof_pos_list[0][env_idx].reshape(23, 3).clone().cpu())
        #     self.recording_dict["dofpos_agent1"].append(self._rigid_body_pos_list[0][0].clone().cpu())
        #     self.recording_dict["dofrot_agent1"].append(self._rigid_body_rot_list[0][0].clone().cpu())
        #     # Agent 2 (Index 1 in lists)
        #     self.recording_dict["rootpos_agent2"].append(self._humanoid_root_states_list[1][env_idx, 0:3].clone().cpu())
        #     self.recording_dict["rootrot_agent2"].append(self._humanoid_root_states_list[1][env_idx, 3:7].clone().cpu())
        #     self.recording_dict["dofpos_agent2"].append(self._rigid_body_pos_list[1][0].clone().cpu())
        #     self.recording_dict["dofrot_agent2"].append(self._rigid_body_rot_list[1][0].clone().cpu())
        #     # Reshape DOF from [69] to [23, 3]
        #     # self.recording_dict["dofpos_agent2"].append(self._dof_pos_list[1][env_idx].reshape(23, 3).clone().cpu())

        #     self.frame_index += 1

        #     # Save once we hit the target frames
        #     if self.frame_index == self.max_recording_frames:
        #         self.save_recorded_pt()
        
        # Debugging 
        # if flags.debug:
        #     body_vel = self._rigid_body_vel.clone()
        #     speeds = body_vel.norm(dim = -1).mean(dim = -1)
        #     sorted_speed, sorted_idx = speeds.sort()
        #     print(sorted_speed.numpy()[::-1][:20], sorted_idx.numpy()[::-1][:20].tolist())
        #     # import ipdb; ipdb.set_trace()

        return

    def render(self, sync_frame_time=False):
        if self.viewer or flags.server_mode:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        if self.humanoid_type in ["smpl", "smplh", "smplx", "g1", "pm01"]:
            body_ids = [self._body_names.index(name) for name in key_body_names]
            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        else:
            env_ptr = self.envs[0]
            actor_handle = self.humanoid_handles[0]
            body_ids = []

            for body_name in key_body_names:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
                assert (body_id != -1)
                body_ids.append(body_id)

            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        return body_ids

    def _build_key_body_ids_orig_tensor(self, key_body_names):
        body_ids = [self._body_names_orig.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles_list[0][0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        if self.num_agents > 1 and len(self._humanoid_root_states_list) > 1:
            char0 = self._humanoid_root_states_list[0][0, 0:3]
            char1 = self._humanoid_root_states_list[1][0, 0:3]
            self._cam_prev_char_pos = (0.5 * (char0 + char1)).cpu().numpy()
            cam_dist = 4.0
            cam_height = 1.5
        else:
            self._cam_prev_char_pos = self._humanoid_root_states_list[0][0, 0:3].cpu().numpy()
            cam_dist = 3.0
            cam_height = 1.0

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - cam_dist, cam_height)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        if self.num_agents > 1 and len(self._humanoid_root_states_list) > 1:
            char0 = self._humanoid_root_states_list[0][0, 0:3]
            char1 = self._humanoid_root_states_list[1][0, 0:3]
            char_root_pos = (0.5 * (char0 + char1)).cpu().numpy()
        else:
            char_root_pos = self._humanoid_root_states_list[0][0, 0:3].cpu().numpy()

        if self.viewer:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            if not flags.follow:
                self._cam_prev_char_pos[:] = char_root_pos
                return
        else:
            cam_pos = np.array([char_root_pos[0], char_root_pos[1] - 3.0, 1.0])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[0], new_cam_pos, new_cam_target)

        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = torch_utils.quat_to_tan_norm(torch_utils.exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert ((num_joints * joint_obs_size) == joint_dof_obs.shape[1])

    return joint_dof_obs


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # ZL this can be sped up for SMPL
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert (False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert ((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # global body rotation
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # if fall_contact.any():
        # print(masked_contact_buf[0, :, 0].nonzero())
        #     import ipdb
        #     ipdb.set_trace()

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        ############################## Debug ##############################
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']);  print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    # import ipdb
    # ipdb.set_trace()

    return reset, terminated


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat)) #SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


@torch.jit.script
def compute_humanoid_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, dof_obs_size, dof_offsets, smpl_params, local_root_obs, root_height_obs, upright, has_smpl_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, bool, bool,bool, bool) -> Tensor
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    if has_smpl_params:
        obs_list.append(smpl_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel] #69+144+72+72
    
    if has_smpl_params:
        obs_list.append(smpl_params)
        
    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs


# @torch.jit.script
# def compute_humanoid_observations_smpl_max_v2(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params, time_steps):
#     ### V2 has time steps. 
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, int) -> Tensor
#     root_pos = body_pos[:, -1, 0, :]
#     root_rot = body_rot[:, -1, 0, :]
#     B, T, J, C = body_pos.shape

#     if not upright:
#         root_rot = remove_base_rot(root_rot)

#     root_h_obs = root_pos[:, 2:3]
#     heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     heading_rot = torch_utils.calc_heading_quat(root_rot)
#     # heading_rot_inv_expand = heading_inv_rot.unsqueeze(-2)
#     # heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
#     # flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])
    
#     heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0).view(-1, 4)
#     heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    
#     root_pos_expand = root_pos.unsqueeze(-2).unsqueeze(-2)
#     local_body_pos = body_pos - root_pos_expand
#     flat_local_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand, local_body_pos.view(-1, 3))
#     local_body_pos = flat_local_body_pos.reshape(B, time_steps, J * C)
#     local_body_pos = local_body_pos[..., 3:]  # remove root pos

#     flat_local_body_rot = quat_mul(heading_inv_rot_expand, body_rot.view(-1, 4))
#     local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot).view(B, time_steps, J * 6)

#     if not (local_root_obs):
#         root_rot_obs = torch_utils.quat_to_tan_norm(body_rot[:, :, 0].view(-1, 4)) # If not local root obs, you override it. 
#         local_body_rot_obs[..., 0:6] = root_rot_obs

#     local_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand, body_vel.view(-1, 3)).view(B, time_steps, J * 3)

#     local_body_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand, body_ang_vel.view(-1, 3)).view(B, time_steps, J * 3)
    
#     ##################### Compute_history #####################
#     body_obs = torch.cat([local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim = -1)

#     obs_list = []
#     if root_height_obs:
#         body_obs = torch.cat([body_pos[:, :, 0, 2:3], body_obs], dim = -1)
        
    
#     obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    
#     if has_smpl_params:
#         raise NotImplementedError
        
#     if has_limb_weight_params:
#         raise NotImplementedError

#     obs = body_obs.view(B, -1)
#     return obs

@torch.jit.script
def compute_humanoid_observations_smpl_max_v2(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params, time_steps):
    ### V2 has time steps. 
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, int) -> Tensor
    B, T, J, C = body_pos.shape

    root_pos = body_pos[:, :, 0, :]
    root_rot = body_rot[:, :, 0, :]

    if not upright:
        root_rot = remove_base_rot(root_rot.view(-1, 4)).view(B, T, 4)

    root_h_obs = root_pos[:, :, 2:3]
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot.view(-1, 4))
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).repeat((1, J, 1)).view(-1, 4)

    root_pos_expand = root_pos.unsqueeze(2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand, local_body_pos.view(-1, 3))
    local_body_pos = flat_local_body_pos.reshape(B, T, J * C)
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_local_body_rot = quat_mul(heading_inv_rot_expand, body_rot.view(-1, 4))
    local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot).view(B, T, J * 6)

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot.view(-1, 4)).view(B, T, 6) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    local_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand, body_vel.view(-1, 3)).view(B, T, J * 3)

    local_body_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand, body_ang_vel.view(-1, 3)).view(B, T, J * 3)
    
    ##################### Compute_history #####################
    body_obs = torch.cat([local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim = -1)

    obs_list = []
    if root_height_obs:
        body_obs = torch.cat([root_h_obs, body_obs], dim = -1)
        
    
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    
    if has_smpl_params:
        raise NotImplementedError
        
    if has_limb_weight_params:
        raise NotImplementedError

    obs = body_obs.view(B, -1)
    return obs



@torch.jit.script
def compute_humanoid_observations_smpl_max_v3(body_pos, body_rot, body_vel, body_ang_vel, force_sensor_readings, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, force_sensor_readings]
    
    if has_smpl_params:
        obs_list.append(smpl_params)
        
    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs
