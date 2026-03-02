import time
import torch
import phc.env.tasks.humanoid as humanoid
from phc.utils import torch_utils
from typing import OrderedDict

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from collections import deque
from phc.utils.torch_utils import project_to_norm

from phc.learning.network_loader import load_combat_prior

from easydict import EasyDict

class HumanoidZ(humanoid.Humanoid):

    def initialize_z_models(self):
        check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
        self.combat_prior = load_combat_prior(check_points[0], device=self.device, activation="silu")

        self.running_mean, self.running_var = check_points[0]['running_mean_std']['running_mean'], check_points[0]['running_mean_std']['running_var']

    def _setup_character_props_z(self):
        self._num_actions = self.cfg['env'].get("z_size", 64)
        return


    def step_z(self, actions_z):

        with torch.no_grad():
            # Apply trained Model.
            
            ################ GT-Z ################
            self_obs_size = self.get_self_obs_size()
            self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)     
            actions_z = project_to_norm(actions_z, 1, "sphere")
            self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
            prior_obs = torch.cat([self_obs, actions_z], dim = -1)
            actions = self.combat_prior(prior_obs)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
