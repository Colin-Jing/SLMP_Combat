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
from ast import Try
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
from enum import Enum
from matplotlib.pyplot import flag
import numpy as np
import torch

from typing import Dict, Optional

from isaacgym import gymapi
from isaacgym import gymtorch

from phc.env.tasks.humanoid_z import HumanoidZ
from phc.env.util import gym_util

from isaacgym.torch_utils import *
from phc.utils import torch_utils

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from phc.utils.flags import flags
from collections import OrderedDict


class HumanoidMoCap(HumanoidZ):

    class StateInit(Enum):
        Default = 0

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._state_init = HumanoidMoCap.StateInit.Default

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        data_dir = "data/smpl"
        if self.humanoid_type in ["smpl", ]:
            self.smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral").to(self.device)
            self.smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male").to(self.device)
            self.smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female").to(self.device)
        elif self.humanoid_type in ["smplx"]:
            self.smpl_parser_n = SMPLX_Parser(model_path=data_dir, gender="neutral", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20).to(self.device)
            self.smpl_parser_m = SMPLX_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20).to(self.device)
            self.smpl_parser_f = SMPLX_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20).to(self.device)

        self.start = True  # camera flag
        return
    
    ## Disabled.
    # def get_self_obs_size(self):
    #     if self.obs_v == 2:
    #         return self._num_self_obs * self.past_track_steps
    #     else:
    #         return self._num_self_obs
        
    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        humanoid_obs_list = self._compute_humanoid_obs(env_ids)

        if (self._enable_task_obs):
            task_obs_list = self._compute_task_obs(env_ids)
            
            
        for i in range(self.num_agents):
            self.obs_buf[env_ids+i*self.num_envs] = torch.cat([humanoid_obs_list[i], task_obs_list[i]],dim=-1)
        
        # if self.obs_v == 2:
        #     # Double sub will return a copy.
        #     B, N = obs.shape
        #     sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
        #     zeros = sums == 0
        #     nonzero = ~zeros
        #     obs_slice = self.obs_buf[env_ids]
        #     obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
        #     obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
        #     self.obs_buf[env_ids] = obs_slice
        # else:
        #     self.obs_buf[env_ids] = obs

        return

    def resample_motions(self):
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        return
    
    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()


        return task_obs_detail

    def post_physics_step(self):
        super().post_physics_step()

        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True

        super()._reset_envs(env_ids)

        return

    def _reset_actors(self, env_ids):
        self._reset_default(env_ids)
        return

    def _reset_default(self, env_ids):
        # self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        # self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        super()._reset_actors(env_ids)
        self._reset_default_env_ids = env_ids
        return

    def _compute_humanoid_obs(self, env_ids=None):
        obs_list = super()._compute_humanoid_obs(env_ids)
        return obs_list

    def _set_env_state(
        self,
        env_ids,
        root_pos_list,
        root_rot_list,
        dof_pos_list,
        root_vel_list,
        root_ang_vel_list,
        dof_vel_list,
        rigid_body_pos_list=None,
        rigid_body_rot_list=None,
        rigid_body_vel_list=None,
        rigid_body_ang_vel_list=None,
    ):
        for i in range(self.num_agents):
            self._humanoid_root_states_list[i][env_ids, 0:3] = root_pos_list[i]
            self._humanoid_root_states_list[i][env_ids, 3:7] = root_rot_list[i]
            self._humanoid_root_states_list[i][env_ids, 7:10] = root_vel_list[i]
            self._humanoid_root_states_list[i][env_ids, 10:13] = root_ang_vel_list[i]
            self._dof_pos_list[i][env_ids] = dof_pos_list[i]
            self._dof_vel_list[i][env_ids] = dof_vel_list[i]

            if (not rigid_body_pos_list is None) and (not rigid_body_rot_list is None):
                self._rigid_body_pos_list[i][env_ids] = rigid_body_pos_list[i]
                self._rigid_body_rot_list[i][env_ids] = rigid_body_rot_list[i]
                self._rigid_body_vel_list[i][env_ids] = rigid_body_vel_list[i]
                self._rigid_body_ang_vel_list[i][env_ids] = rigid_body_ang_vel_list[i]
                self._reset_rb_pos_list[i] = self._rigid_body_pos_list[i][env_ids].clone()
                self._reset_rb_rot_list[i] = self._rigid_body_rot_list[i][env_ids].clone()
                self._reset_rb_vel_list[i] = self._rigid_body_vel_list[i][env_ids].clone()
                self._reset_rb_ang_vel_list[i] = self._rigid_body_ang_vel_list[i][env_ids].clone()
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self._state_reset_happened and "_reset_rb_pos_list" in self.__dict__:
            # ZL: Hack to get rigidbody pos and rot to be the correct values. Needs to be called after _set_env_state
            # Also needs to be after refresh_rigid_body_state_tensor
            env_ids = self._reset_ref_env_ids
            if len(env_ids) > 0:
                for i in range(self.num_agents):
                    self._rigid_body_pos_list[i][env_ids] = self._reset_rb_pos_list[i]
                    self._rigid_body_rot_list[i][env_ids] = self._reset_rb_rot_list[i]
                    self._rigid_body_vel_list[i][env_ids] = self._reset_rb_vel_list[i]
                    self._rigid_body_ang_vel_list[i][env_ids] = self._reset_rb_ang_vel_list[i]
                self._state_reset_happened = False
                

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # if self.self_obs_v == 2:
        #     # print("self._reset_ref_env_ids:", self._reset_ref_env_ids)
        #     env_ids = self._reset_ref_env_ids
        #     if len(env_ids) > 0:
        #         self._init_tensor_history(env_ids)
        
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states_list[0][self.viewing_env_idx, 0:3].cpu().numpy()

        if self.viewer:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        else:
            cam_pos = np.array([char_root_pos[0] + 2.5, char_root_pos[1] + 2.5, char_root_pos[2]])

        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        # if np.abs(cam_pos[2] - char_root_pos[2]) > 5:
        cam_pos[2] = char_root_pos[2] + 0.5
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[self.viewing_env_idx], new_cam_pos, new_cam_target)

        if flags.follow:
            self.start = True
        else:
            self.start = False

        if self.start:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return
