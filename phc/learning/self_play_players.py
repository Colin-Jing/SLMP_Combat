import torch

from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import learning.common_player as common_player

from rl_games.common.tr_helpers import unsqueeze_obs
from phc.learning.self_play_agent import construct_op_ck_name


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class SelfPlayPlayerContinuous(common_player.CommonPlayer):
    @staticmethod
    def _obs_shape_from_dim(obs_dim: int):
        return (int(obs_dim),)

    @staticmethod
    def _align_obs_dim(obs_batch: torch.Tensor, target_dim: int) -> torch.Tensor:
        if obs_batch.shape[-1] == target_dim:
            return obs_batch
        if obs_batch.shape[-1] > target_dim:
            return obs_batch[:, :target_dim]
        raise ValueError(
            f"Obs dim smaller than expected in mixed branch: got {obs_batch.shape[-1]}, expected {target_dim}"
        )

    def __init__(self, config):
        self._normalize_input = config["normalize_input"]
        self.opp_agent = 1

        super().__init__(config)

        self.llc_steps = max(1, int(config.get("llc_steps", 1)))
        return

    def get_action(self, obs, is_determenistic=False):
        obs = obs["obs"]
        env_num_agents = self.env.task.num_agents
        num_actors = obs.shape[0] // env_num_agents

        if self.opp_agent == 1:
            self_obs = obs[:num_actors]
            oppoent_obs = obs[num_actors:]
            self_obs = self._preproc_obs(self_obs)
            oppoent_obs = self._oppoent_preproc_obs(oppoent_obs)

            if self.has_batch_dimension is False:
                self_obs = unsqueeze_obs(self_obs)
                oppoent_obs = unsqueeze_obs(oppoent_obs)

            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self_obs,
                "rnn_states": self.states[:num_actors] if self.states is not None else self.states,
            }

            oppoent_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": oppoent_obs,
                "rnn_states": self.states[num_actors:] if self.states is not None else self.states,
            }
        else:
            self_obs = obs[num_actors:]
            oppoent_obs = obs[:num_actors]
            self_obs = self._preproc_obs(self_obs)
            oppoent_obs = self._oppoent_preproc_obs(oppoent_obs)

            if self.has_batch_dimension is False:
                self_obs = unsqueeze_obs(self_obs)
                oppoent_obs = unsqueeze_obs(oppoent_obs)

            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self_obs,
                "rnn_states": self.states[num_actors:] if self.states is not None else self.states,
            }

            oppoent_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": oppoent_obs,
                "rnn_states": self.states[:num_actors] if self.states is not None else self.states,
            }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            oppoent_res_dict = self.oppoent_model(oppoent_input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]

        oppoent_mu = oppoent_res_dict["mus"]
        oppoent_action = oppoent_res_dict["actions"]

        self.states = (
            res_dict["rnn_states"]
            if res_dict["rnn_states"] is None
            else torch.cat([res_dict["rnn_states"], oppoent_res_dict["rnn_states"]], dim=0)
        )
        if is_determenistic:
            current_action = mu
            current_oppoent_action = oppoent_mu
        else:
            current_action = action
            current_oppoent_action = oppoent_action
        if self.has_batch_dimension is False:
            current_action = torch.squeeze(current_action.detach())
            current_oppoent_action = torch.squeeze(current_oppoent_action.detach())

        if self.opp_agent == 1:
            current_action = torch.cat([current_action, current_oppoent_action], dim=0)
        else:
            current_action = torch.cat([current_oppoent_action, current_action], dim=0)
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def env_step(self, env, actions):
        if self.llc_steps == 1:
            return super().env_step(env, actions)

        rewards_acc = None
        done_any = None
        steps_taken = None
        obs_ret = None
        infos_ret = None

        def _masked_assign(dst, src, mask):
            if isinstance(dst, dict) and isinstance(src, dict):
                out = {}
                for k in dst.keys():
                    dv, sv = dst[k], src[k]
                    if torch.is_tensor(dv) and torch.is_tensor(sv) and dv.shape[0] == mask.shape[0]:
                        dv2 = dv.clone()
                        dv2[mask] = sv[mask]
                        out[k] = dv2
                    else:
                        out[k] = sv
                return out
            if torch.is_tensor(dst) and torch.is_tensor(src) and dst.shape[0] == mask.shape[0]:
                dst2 = dst.clone()
                dst2[mask] = src[mask]
                return dst2
            return src

        for _ in range(self.llc_steps):
            obs, rewards, dones, infos = super().env_step(env, actions)

            dones_b = dones.bool()
            if rewards_acc is None:
                rewards_acc = torch.zeros_like(rewards)
                done_any = torch.zeros_like(dones_b)
                steps_taken = torch.zeros_like(dones, dtype=torch.float32)
                obs_ret = obs
                infos_ret = infos

            alive = ~done_any
            alive_f = alive.float()
            if rewards.dim() > alive_f.dim():
                alive_f = alive_f.unsqueeze(-1)

            rewards_acc = rewards_acc + rewards * alive_f
            steps_taken = steps_taken + alive.float()

            new_done = (~done_any) & dones_b
            if new_done.any():
                obs_ret = _masked_assign(obs_ret, obs, new_done)
                infos_ret = infos

            done_any = done_any | dones_b
            if bool(done_any.all()):
                break

        return obs_ret, rewards_acc, done_any.to(dones.dtype), infos_ret

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        fn_op = construct_op_ck_name(fn)
        checkpoint_op = torch_ext.load_checkpoint(fn_op)
        # checkpoint_op = torch_ext.load_checkpoint(fn)
        self.oppoent_model.load_state_dict(checkpoint_op["model"], strict=False)
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            self.oppoent_running_mean_std.load_state_dict(checkpoint_op["running_mean_std"])

        return

    def _build_net(self, config):
        base_obs_dim = int(torch_ext.shape_whc_to_cwh(self.obs_shape)[0])
        self._mixed_obs_branch = False
        agent_obs_sizes = [base_obs_dim, base_obs_dim]
        if hasattr(self.env.task, "get_agent_obs_sizes"):
            task_obs_sizes = list(self.env.task.get_agent_obs_sizes())
            if len(task_obs_sizes) >= 2:
                s0 = int(task_obs_sizes[0])
                s1 = int(task_obs_sizes[1])
                if s0 != s1:
                    self._mixed_obs_branch = True
                    agent_obs_sizes = [s0, s1]
        self.self_obs_dim = int(agent_obs_sizes[0])
        self.oppoent_obs_dim = int(agent_obs_sizes[1])

        if self.normalize_input:
            if self._mixed_obs_branch:
                self.running_mean_std = RunningMeanStd(self._obs_shape_from_dim(self.self_obs_dim)).to(self.device)
                self.running_mean_std.eval()
                self.oppoent_running_mean_std = RunningMeanStd(self._obs_shape_from_dim(self.oppoent_obs_dim)).to(
                    self.device
                )
                self.oppoent_running_mean_std.eval()
            else:
                if "vec_env" in self.__dict__:
                    obs_shape = torch_ext.shape_whc_to_cwh(self.vec_env.env.task.get_running_mean_size())
                else:
                    obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
                self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
                self.running_mean_std.eval()
                self.oppoent_running_mean_std = RunningMeanStd(obs_shape).to(self.device)
                self.oppoent_running_mean_std.eval()
        else:
            self.running_mean_std = None
            self.oppoent_running_mean_std = None

        if self._mixed_obs_branch:
            self_config = dict(config)
            self_config["input_shape"] = self._obs_shape_from_dim(self.self_obs_dim)
            self_config["mean_std"] = self.running_mean_std
            self.model = self.network.build(self_config)
            self.model.to(self.device)
            self.model.eval()

            opp_config = dict(config)
            opp_config["input_shape"] = self._obs_shape_from_dim(self.oppoent_obs_dim)
            opp_config["mean_std"] = self.oppoent_running_mean_std
            self.oppoent_model = self.network.build(opp_config)
            self.oppoent_model.to(self.device)
            self.oppoent_model.eval()
        else:
            config["mean_std"] = self.running_mean_std
            self.model = self.network.build(config)
            self.model.to(self.device)
            self.model.eval()
            self.oppoent_model = self.network.build(config)
            self.oppoent_model.to(self.device)
            self.oppoent_model.eval()
        self.is_rnn = self.model.is_rnn()
        return

    def _eval_critic(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_critic(input)

    def _eval_task_value(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_task_value(input)

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, "env"):
            config["task_obs_size_detail"] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config["self_obs_size"] = self.env.task.get_self_obs_size()
                config["task_obs_size"] = self.env.task.get_task_obs_size()
        else:
            pass
        return config

    def _eval_actor(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input and self.running_mean_std is not None:
            obs_batch_proc = obs_batch[:, : self.running_mean_std.mean_size]
            obs_batch_out = self.running_mean_std(obs_batch_proc)
            obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size :]], dim=-1)
        if self._mixed_obs_branch:
            obs_batch = self._align_obs_dim(obs_batch, self.self_obs_dim)
        return obs_batch

    def _oppoent_preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._oppoent_preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input and self.oppoent_running_mean_std is not None:
            obs_batch_proc = obs_batch[:, : self.oppoent_running_mean_std.mean_size]
            obs_batch_out = self.oppoent_running_mean_std(obs_batch_proc)
            obs_batch = torch.cat(
                [obs_batch_out, obs_batch[:, self.oppoent_running_mean_std.mean_size :]], dim=-1
            )
        if self._mixed_obs_branch:
            obs_batch = self._align_obs_dim(obs_batch, self.oppoent_obs_dim)
        return obs_batch


class SelfPlayPlayerDiscrete(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        return

    def get_action(self, obs, is_determenistic=False):
        raise NotImplementedError()

    def _build_net(self, config):
        self.model = self.network.build(config)
        self.model.to(self.device)
        return
