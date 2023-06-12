import torch
import numpy as np
from agent.nn.actor import Actor
from agent.nn.critic import Critic
from utils import loss


class Agent:
    def __init__(self, cfg, cfg_agent, agent_idx=-1):
        """
        Class for building agents or mediator

        :param input_size: iterable(n_actor, n_critic), number of input features to NNs
        :param agent_nn: iterable, Actor and Critic NN
        :param cfg_agent: config file for agents/mediator
        :param cfg_env: config file for environment
        :param agent_i: int, "-1" if mediator, positive integers otherwise
        """
        optimizer = torch.optim.Adam
        self.lr_a = cfg_agent.lr_a
        self.lr_c = cfg_agent.lr_c
        self.gamma = cfg_agent.gamma
        self.entropy_coef = cfg_agent.entropy_coef
        self.hidden_state = cfg_agent.n_hidden
        self.state_size = cfg.env.state_size
        self.action_size = cfg_agent.action_size
        self.idx = agent_idx
        self.dtype = torch.float32
        self.cfg = cfg
        self.entropy_min = cfg_agent.entropy_min

        self.entropy_coef_decrease = 1 / ((self.entropy_coef / self.entropy_min) ** (1 / cfg_agent.entropy_steps))

        self.actor = Actor(self.state_size, self.action_size, cfg_agent.n_hidden)
        self.critic = Critic(self.state_size, self.action_size, cfg_agent.n_hidden).to(self.cfg.train_device)

        self.opt_actor = optimizer(self.actor.parameters(), lr=cfg_agent.lr_a)
        self.opt_critic = optimizer(self.critic.parameters(), lr=cfg_agent.lr_c)

    def update_entropy(self):
        self.entropy_coef = np.maximum(self.entropy_min, self.entropy_coef * self.entropy_coef_decrease)

    def select_device(self, device):
        self.actor.to(device)

    def get_policy(self, obs, off_policy):
        _, pi_dist = self.actor(obs, off_policy)

        return pi_dist

    def step(self, obs):
        action, _ = self.actor(obs, under_med=True)

        return action.squeeze(0).cpu().detach().numpy()

    def compute_ppo_stats(self, trajectory):
        rgb = trajectory['obs']
        next_rgb = trajectory['next_obs']
        action = trajectory['action'].squeeze(-1)
        done = trajectory['done']
        # coalition = trajectory['coalition']
        rewards = trajectory['reward'].unsqueeze(-1)
        k = trajectory['k']

        under_mediator = trajectory['under_mediator']

        assert len(under_mediator[under_mediator == 1]) == 0, f'There are ones in under_mediator! Check if it is ' \
                                                              f'intentional and if so comment me'
        with torch.no_grad():
            self.rets = rewards + (1 - done) * (self.gamma ** k) * self.critic(next_rgb)
            self.adv = self.rets - self.critic(rgb)

            policy = self.get_policy([rgb, under_mediator], off_policy=True)
            self.log_old = policy.log_prob(action).unsqueeze(-1)

    def update_ppo(self, trajectory, idx):
        """
        rets, adv, log_old -- отсоединены от графа вычислений!
        """
        rgb = trajectory['obs'][idx]
        # rets, adv_old, rho = self.ppo_stats(policy)
        rets, adv, log_old = self.rets[idx].detach(), self.adv[idx].detach(), self.log_old[idx].detach()
        action = trajectory['action'][idx].squeeze(-1)
        # coalition = trajectory['coalition'][idx]
        under_mediator = trajectory['under_mediator'][idx]
        adv = loss.advantage(adv)

        # _ = [0] * 16  # hi again

        # agent_critic_loss = mse(self.critic([m_obs, other_obs]), rets)
        agent_critic_loss = loss.valueLoss(self.critic(rgb), rets)
        policy = self.get_policy([rgb, under_mediator], off_policy=True)

        log_prob = policy.log_prob(action).unsqueeze(-1)
        rho = torch.exp(log_prob - log_old)
        entropy = policy.entropy().mean()

        # agent_actor_loss = adv * rho - self.entropy_coef * entropy
        agent_actor_loss = loss.ppo_loss(adv, rho) - self.entropy_coef * entropy

        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        agent_actor_loss.backward()
        agent_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)

        self.opt_actor.step()
        self.opt_critic.step()
