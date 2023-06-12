import torch
import wandb
import numpy as np


class EyeOfGodBase:
    def __init__(self, agent_class, mediator_class, agent_nn, mediator_nn, n_inputs, cfg):
        agent_actor_inp, agent_critic_inp, \
        mediator_actor_inp, mediator_critic_inp = n_inputs

        # self.state_size = cfg.env.state_size
        # self.action_size = cfg.env.action_size
        self.n_agents = cfg.env.n_agents
        self.batch_size = cfg.env.batch_size
        self.device = cfg.device
        self.dtype = torch.float32
        self.cfg = cfg

        self.agents = [agent_class(input_sizes=[agent_actor_inp, agent_critic_inp], agent_nn=agent_nn, cfg=cfg,
                                   cfg_agent=cfg.agent, agent_i=i)
                       for i in range(self.n_agents)]

        self.mediator = mediator_class(input_sizes=[mediator_actor_inp, mediator_critic_inp],
                                       agent_nn=mediator_nn,
                                       cfg=cfg,
                                       cfg_agent=cfg.mediator, agent_i=-1)

    def _tensorify(self, inp):
        out = []

        for arr in inp:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, device=self.device, dtype=self.dtype)

            if arr.dim() == 0:
                arr = arr.reshape(-1, 1)
            elif arr.dim() == 1:
                arr = arr.unsqueeze(0)

            out.append(arr)
        out = torch.cat(out, dim=-1)

        return out

    def _get_batch(self, trajectories):
        raise NotImplementedError

    def update(self, trajectories):
        _ = self._get_batch(trajectories)

        raise NotImplementedError

    def train(self, env, log=False):
        eval_episodes = self.cfg.env.eval_episodes

        # get batch of trajectories
        for i in range(self.cfg.env.iterations):
            trajectories = []
            steps_ctn = 0
            while len(trajectories) < self.cfg.env.min_episodes_per_update \
                    or steps_ctn < self.cfg.env.min_transitions_per_update:
                traj, traj_a = self.sample_episode(env)
                steps_ctn += len(traj)
                trajectories.append(traj)

            # update
            self.update(trajectories)

            if log:
                if i % 100 == 0:
                    print(f'ITERATION {i}:')
                    wandb.log(data={}, step=i, commit=False)
                    self.evaluate_policy(eval_episodes, env)

    def evaluate_policy(self, n_episodes, env):
        raise NotImplementedError

    def sample_episode(self, env, test):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError
