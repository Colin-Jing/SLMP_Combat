import os.path as osp
import time

import torch
from torch import optim

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common.experience import ExperienceBuffer
from phc.utils.running_mean_std import RunningMeanStd

import learning.common_agent as common_agent


def construct_op_ck_name(fn):
    if ".pth" in fn:
        return fn.replace(".pth", "_op.pth")
    return fn + "_op"


class SelfPlayAgent(common_agent.CommonAgent):
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

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        self.switch_frequency = config.get("switch_frequency", 1)
        self.update_agent = config.get("update_agent", 0)

        self.env_num_agents = self.vec_env.env.task.num_agents
        base_obs_dim = int(torch_ext.shape_whc_to_cwh(self.obs_shape)[0])
        self._mixed_obs_branch = False
        agent_obs_sizes = [base_obs_dim, base_obs_dim]
        if hasattr(self.vec_env.env.task, "get_agent_obs_sizes"):
            task_obs_sizes = list(self.vec_env.env.task.get_agent_obs_sizes())
            if len(task_obs_sizes) >= 2:
                s0 = int(task_obs_sizes[0])
                s1 = int(task_obs_sizes[1])
                if s0 != s1:
                    self._mixed_obs_branch = True
                    agent_obs_sizes = [s0, s1]
        self.agent_0_obs_dim = int(agent_obs_sizes[0])
        self.agent_1_obs_dim = int(agent_obs_sizes[1])
        self.current_obs_dim = self.agent_0_obs_dim

        self.batch_size_envs = self.num_actors
        self.batch_size = self.batch_size_envs * self.horizon_length
        self.dataset = datasets.PPODataset(
            self.batch_size,
            self.minibatch_size,
            self.is_discrete,
            self.is_rnn,
            self.ppo_device,
            self.seq_len,
        )

        if self.normalize_input:
            if self._mixed_obs_branch:
                self.agent_1_running_mean_std = RunningMeanStd(
                    self._obs_shape_from_dim(self.agent_1_obs_dim)
                ).to(self.ppo_device)
            else:
                if "vec_env" in self.__dict__:
                    obs_shape = torch_ext.shape_whc_to_cwh(self.vec_env.env.task.get_running_mean_size())
                else:
                    obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
                self.agent_1_running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
        else:
            self.agent_1_running_mean_std = None

        if self.normalize_value and not hasattr(self, "value_mean_std"):
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)

        self.agent_1_value_mean_std = RunningMeanStd((1,)).to(self.ppo_device) if self.normalize_value else None

        net_config = self._build_net_config()
        if self._mixed_obs_branch:
            net_config["input_shape"] = self._obs_shape_from_dim(self.agent_1_obs_dim)
            net_config["mean_std"] = self.agent_1_running_mean_std
        self.agent_1_model = self.network.build(net_config)
        self.agent_1_model.to(self.ppo_device)
        self.agent_1_optimizer = optim.Adam(
            self.agent_1_model.parameters(),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )

        if self._mixed_obs_branch and self.agent_0_obs_dim != base_obs_dim:
            raise ValueError(
                f"Agent-0 obs dim mismatch: env base dim={base_obs_dim}, agent-0 dim={self.agent_0_obs_dim}."
            )

        self.agent_0_model = self.model
        self.agent_0_optimizer = self.optimizer
        self.agent_0_running_mean_std = self.running_mean_std if self.normalize_input else None
        self.agent_0_value_mean_std = self.value_mean_std if self.normalize_value else None

        if self.update_agent == 1:
            self.update_agent_1_model()
        else:
            self.update_agent_0_model()

        return

    def restore(self, fn):
        super().restore(fn)
        op_fn = construct_op_ck_name(fn)
        if osp.exists(op_fn):
            op_weights = torch_ext.load_checkpoint(op_fn)
            self.agent_1_model.load_state_dict(op_weights["model"])

            if self.normalize_input and self.agent_1_running_mean_std is not None:
                self.agent_1_running_mean_std.load_state_dict(op_weights["running_mean_std"])

            if self.normalize_value and self.agent_1_value_mean_std is not None:
                self.agent_1_value_mean_std.load_state_dict(op_weights["reward_mean_std"])

            self.agent_1_optimizer.load_state_dict(op_weights["optimizer"])

        if self.update_agent == 1:
            self.update_agent_1_model()
        else:
            self.update_agent_0_model()
        return

    def get_full_state_weights_agent_1(self):
        state = {
            "model": self.agent_1_model.state_dict(),
            "optimizer": self.agent_1_optimizer.state_dict(),
            "epoch": self.epoch_num,
            "last_mean_rewards": self.last_mean_rewards,
            "frame": self.frame,
        }
        if self.normalize_input and self.agent_1_running_mean_std is not None:
            state["running_mean_std"] = self.agent_1_running_mean_std.state_dict()
        if self.normalize_value and self.agent_1_value_mean_std is not None:
            state["reward_mean_std"] = self.agent_1_value_mean_std.state_dict()
        env_state = self.vec_env.get_env_state()
        state["env_state"] = env_state
        return state

    def get_full_state_weights_agent_0(self):
        state = {
            "model": self.agent_0_model.state_dict(),
            "optimizer": self.agent_0_optimizer.state_dict(),
            "epoch": self.epoch_num,
            "last_mean_rewards": self.last_mean_rewards,
            "frame": self.frame,
        }
        if self.normalize_input and self.agent_0_running_mean_std is not None:
            state["running_mean_std"] = self.agent_0_running_mean_std.state_dict()
        if self.normalize_value and self.agent_0_value_mean_std is not None:
            state["reward_mean_std"] = self.agent_0_value_mean_std.state_dict()
        env_state = self.vec_env.get_env_state()
        state["env_state"] = env_state
        return state

    def save(self, fn):
        state_agent0 = self.get_full_state_weights_agent_0()
        torch_ext.save_checkpoint(fn, state_agent0)

        op_fn = construct_op_ck_name(fn)
        state_agent1 = self.get_full_state_weights_agent_1()
        torch_ext.save_checkpoint(op_fn, state_agent1)
        return

    def update_agent_1_model(self):
        self.model = self.agent_1_model
        self.optimizer = self.agent_1_optimizer
        if self._mixed_obs_branch:
            self.current_obs_dim = self.agent_1_obs_dim
        if self.normalize_input and self.agent_1_running_mean_std is not None:
            self.running_mean_std = self.agent_1_running_mean_std
            self.agent_1_running_mean_std.unfreeze()
            if self.agent_0_running_mean_std is not None:
                self.agent_0_running_mean_std.freeze()
        if self.normalize_value and self.agent_1_value_mean_std is not None:
            self.value_mean_std = self.agent_1_value_mean_std
        return

    def update_agent_0_model(self):
        self.model = self.agent_0_model
        self.optimizer = self.agent_0_optimizer
        if self._mixed_obs_branch:
            self.current_obs_dim = self.agent_0_obs_dim
        if self.normalize_input and self.agent_0_running_mean_std is not None:
            self.running_mean_std = self.agent_0_running_mean_std
            self.agent_0_running_mean_std.unfreeze()
            if self.agent_1_running_mean_std is not None:
                self.agent_1_running_mean_std.freeze()
        if self.normalize_value and self.agent_0_value_mean_std is not None:
            self.value_mean_std = self.agent_0_value_mean_std
        return

    def init_tensors(self):
        super().init_tensors()

        batch_size = self.env_num_agents * self.num_actors

        algo_info = {
            "num_actors": self.num_actors * self.env_num_agents,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": self.use_action_masks,
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        del self.experience_buffer.tensor_dict["actions"]
        del self.experience_buffer.tensor_dict["mus"]
        del self.experience_buffer.tensor_dict["sigmas"]

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["actions"] = torch.zeros(
            batch_shape + (self.actions_num,), dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["mus"] = torch.zeros(
            batch_shape + (self.actions_num,), dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["sigmas"] = torch.zeros(
            batch_shape + (self.actions_num,), dtype=torch.float32, device=self.ppo_device
        )

        self.experience_buffer.tensor_dict["next_obses"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["obses"]
        )
        self.experience_buffer.tensor_dict["next_values"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["values"]
        )

        if "next_obses" not in self.tensor_list:
            self.tensor_list += ["next_obses"]
        return

    def set_eval(self):
        super().set_eval()
        self.agent_0_model.eval()
        self.agent_1_model.eval()

        if self.normalize_input:
            self.agent_0_running_mean_std.eval()
            self.agent_1_running_mean_std.eval()

        if self.normalize_value:
            self.agent_0_value_mean_std.eval()
            self.agent_1_value_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        self.agent_0_model.train()
        self.agent_1_model.train()

        if self.normalize_input:
            self.agent_1_running_mean_std.train()
            self.agent_0_running_mean_std.train()

        if self.normalize_value:
            self.agent_0_value_mean_std.train()
            self.agent_1_value_mean_std.train()
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict["obs"] = self._preproc_obs(obs_dict["obs"])

        if hasattr(self.model, "a2c_network"):
            net = self.model.a2c_network
            if hasattr(net, "eval_critic"):
                if hasattr(self.model, "is_rnn") and self.model.is_rnn():
                    value, _ = net.eval_critic(obs_dict)
                else:
                    value = net.eval_critic(obs_dict)
            else:
                output = net(obs_dict)
                if isinstance(output, dict):
                    value = output.get("values") or output.get("value")
                elif isinstance(output, (tuple, list)):
                    if len(output) >= 3:
                        value = output[-2]
                    elif len(output) == 2:
                        value = output[1]
                    else:
                        value = output[0]
                else:
                    raise AttributeError("Model has no eval_critic")
        elif hasattr(self.model, "eval_critic"):
            if hasattr(self.model, "is_rnn") and self.model.is_rnn():
                value, _ = self.model.eval_critic(obs_dict)
            else:
                value = self.model.eval_critic(obs_dict)
        else:
            raise AttributeError("Model has no eval_critic")

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def get_action_values(self, obs):
        obs_orig = obs["obs"]

        if self.update_agent == 0:
            agent_0_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self._preproc_obs(obs_orig[: self.num_actors]),
                "obs_orig": obs_orig[: self.num_actors],
                "rnn_states": self.rnn_states[: self.num_actors] if self.rnn_states is not None else self.rnn_states,
            }

            agent_1_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self._preproc_agent_1_obs(obs_orig[self.num_actors :]),
                "obs_orig": obs_orig[self.num_actors :],
                "rnn_states": self.rnn_states[self.num_actors :] if self.rnn_states is not None else self.rnn_states,
            }
        else:
            agent_0_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self._preproc_agent_0_obs(obs_orig[: self.num_actors]),
                "obs_orig": obs_orig[: self.num_actors],
                "rnn_states": self.rnn_states[: self.num_actors] if self.rnn_states is not None else self.rnn_states,
            }

            agent_1_input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": self._preproc_obs(obs_orig[self.num_actors :]),
                "obs_orig": obs_orig[self.num_actors :],
                "rnn_states": self.rnn_states[self.num_actors :] if self.rnn_states is not None else self.rnn_states,
            }

        with torch.no_grad():
            if self.update_agent == 0:
                agent_0_res_dict = self.model(agent_0_input_dict)
                agent_1_res_dict = self.agent_1_model(agent_1_input_dict)
            else:
                agent_0_res_dict = self.agent_0_model(agent_0_input_dict)
                agent_1_res_dict = self.model(agent_1_input_dict)

            if self.has_central_value:
                states = obs["states"][: self.num_actors]
                agent_0_input_dict = {
                    "is_train": False,
                    "states": states,
                }
                agent_0_value = self.get_central_value(agent_0_input_dict)
                agent_0_res_dict["values"] = agent_0_value

                agent_1_states = obs["states"][self.num_actors :]
                agent_1_input_dict = {
                    "is_train": False,
                    "states": agent_1_states,
                }
                agent_1_value = self.get_central_value(agent_1_input_dict)
                agent_1_res_dict["values"] = agent_1_value

        if self.update_agent == 0:
            if self.normalize_value:
                agent_0_res_dict["values"] = self.value_mean_std(agent_0_res_dict["values"], True)
                agent_1_res_dict["values"] = self.agent_1_value_mean_std(agent_1_res_dict["values"], True)
        else:
            if self.normalize_value:
                agent_0_res_dict["values"] = self.agent_0_value_mean_std(agent_0_res_dict["values"], True)
                agent_1_res_dict["values"] = self.value_mean_std(agent_1_res_dict["values"], True)

        for k in agent_0_res_dict.keys():
            if agent_0_res_dict[k] is not None:
                agent_0_res_dict[k] = torch.cat([agent_0_res_dict[k], agent_1_res_dict[k]], dim=0)

        return agent_0_res_dict

    def play_steps(self):
        self.set_eval()

        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones.repeat(self.env_num_agents).clone())

            terminated = infos["terminate"].float().unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.env_num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = (1.0 - self.dones.float()).repeat(self.env_num_agents)
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = {}
        if self.update_agent == 0:
            for each in self.tensor_list:
                if each == "states":
                    continue
                batch_dict[each] = a2c_common.swap_and_flatten01(
                    self.experience_buffer.tensor_dict[each][:, : self.num_actors]
                )
            if self.has_central_value:
                batch_dict["states"] = a2c_common.swap_and_flatten01(
                    self.experience_buffer.tensor_dict["states"][:, : self.num_actors]
                )
            batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns[:, : self.num_actors])
            batch_dict["played_frames"] = self.batch_size
            batch_dict["mb_rewards"] = a2c_common.swap_and_flatten01(mb_rewards[:, : self.num_actors])
        else:
            for each in self.tensor_list:
                if each == "states":
                    continue
                batch_dict[each] = a2c_common.swap_and_flatten01(
                    self.experience_buffer.tensor_dict[each][:, self.num_actors :]
                )
            if self.has_central_value:
                batch_dict["states"] = a2c_common.swap_and_flatten01(
                    self.experience_buffer.tensor_dict["states"][:, self.num_actors :]
                )
            batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns[:, self.num_actors :])
            batch_dict["played_frames"] = self.batch_size
            batch_dict["mb_rewards"] = a2c_common.swap_and_flatten01(mb_rewards[:, self.num_actors :])

        return batch_dict

    def train_epoch(self):
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get("rnn_masks", None)

        self.set_train()

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == "legacy":
                    if self.multi_gpu:
                        curr_train_info["kl"] = self.hvd.average_value(curr_train_info["kl"], "ep_kls")
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr,
                        self.entropy_coef,
                        self.epoch_num,
                        0,
                        curr_train_info["kl"].item(),
                    )
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = {k: [v] for k, v in curr_train_info.items()}
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info["kl"])

            if self.schedule_type == "standard":
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, "ep_kls")
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

        if self.schedule_type == "standard_epoch":
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(ep_kls), "ep_kls")
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
            )
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        train_info["mb_rewards"] = batch_dict["mb_rewards"]
        self._record_train_batch_info(batch_dict, train_info)

        if (self.epoch_num + 1) % self.switch_frequency == 0:
            self.update_agent = 1 - self.update_agent
            if self.update_agent == 0:
                self.update_agent_0_model()
            else:
                self.update_agent_1_model()

        return train_info

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
            obs_batch = self._align_obs_dim(obs_batch, self.current_obs_dim)
        return obs_batch

    def _preproc_agent_1_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_agent_1_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input and self.agent_1_running_mean_std is not None:
            obs_batch_proc = obs_batch[:, : self.agent_1_running_mean_std.mean_size]
            obs_batch_out = self.agent_1_running_mean_std(obs_batch_proc)
            obs_batch = torch.cat(
                [obs_batch_out, obs_batch[:, self.agent_1_running_mean_std.mean_size :]], dim=-1
            )
        if self._mixed_obs_branch:
            obs_batch = self._align_obs_dim(obs_batch, self.agent_1_obs_dim)
        return obs_batch

    def _preproc_agent_0_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_agent_0_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input and self.agent_0_running_mean_std is not None:
            obs_batch_proc = obs_batch[:, : self.agent_0_running_mean_std.mean_size]
            obs_batch_out = self.agent_0_running_mean_std(obs_batch_proc)
            obs_batch = torch.cat(
                [obs_batch_out, obs_batch[:, self.agent_0_running_mean_std.mean_size :]], dim=-1
            )
        if self._mixed_obs_branch:
            obs_batch = self._align_obs_dim(obs_batch, self.agent_0_obs_dim)
        return obs_batch
