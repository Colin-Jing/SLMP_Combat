import torch
import time
import phc.env.tasks.humanoid_mocap as humanoid_mocap
from phc.utils.flags import flags
class HumanoidTask(humanoid_mocap.HumanoidMoCap):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.has_task = True
        return


    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
       
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer or flags.server_mode:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        humanoid_obs_list = self._compute_humanoid_obs(env_ids)

        if (self._enable_task_obs):
            task_obs_list = self._compute_task_obs(env_ids)
            
            
        for i in range(self.num_agents):
            self.obs_buf[env_ids+i*self.num_envs] = torch.cat([humanoid_obs_list[i], task_obs_list[i]],dim=-1)

        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return
