import numpy as np

from agent.agent import Agent
from mediator.nn.actor import Actor
from mediator.nn.critic import Critic
import torch
import numpy as np
from utils import loss
from utils.utils import SchedulerBC, AgentTraj

import torch.nn.functional as F
from sklearn.metrics import r2_score


class Mediator(Agent):
    def __init__(self, cfg, cfg_mediator, agent_idx=-1):
        super(Mediator, self).__init__(cfg, cfg_mediator, agent_idx=-1)
        optimizer = torch.optim.Adam

        self.n_agents = self.cfg.env.n_agents
        self.actor = Actor(self.state_size, self.action_size, cfg_mediator.n_hidden, self.n_agents)
        self.critic = Critic(self.state_size, self.action_size,
                             cfg_mediator.n_hidden, self.n_agents).to(self.cfg.train_device)

        self.opt_actor = optimizer(self.actor.parameters(), lr=cfg_mediator.lr_a)
        self.opt_critic = optimizer(self.critic.parameters(), lr=cfg_mediator.lr_c)
        self.lr_lambda = cfg_mediator.lr_lambda

        self.coalition_steps = np.zeros((self.n_agents, 1))
        # self.reg_coef = cfg_mediator.reg_coef
        # self.reg_coef = torch.full((self.n_agents, 1), cfg_mediator.lambda_start).to(self.cfg.train_device).log()
        self.reg_coef_log_ic = torch.tensor([[cfg_mediator.lambda_start]] * 16).to(self.cfg.train_device).log()
        self.reg_coef_log_p = torch.tensor([[cfg_mediator.lambda_start]] * 16).to(self.cfg.train_device).log()
        self.labmda_coef_data = AgentTraj(self.n_agents)
        self.bc_coef = 0
        self.bc_scheduler = SchedulerBC()
        self.criterion_bc = torch.nn.NLLLoss()

    # @torch.no_grad()
    # def update_lambda_ic(self, trajectory):
    #     for agent_idx in range(self.n_agents):
    #         mask_agent = trajectory['agent'].long() == agent_idx
    #         target = self.reg_term_ic[mask_agent]
    #         target[target > 0] *= self.cfg.mediator.lambda_pos_coef
    #         target[target < 0] *= self.cfg.mediator.lambda_neg_coef
    #
    #         if len(target[target != 0]) > 0:  # else we may get nans in .mean()
    #             self.reg_coef_log_ic[agent_idx] = self.reg_coef_log_ic[agent_idx] - self.lr_lambda * target[
    #                 target != 0].mean()
    #             self.reg_coef_log_ic[agent_idx] = torch.clip(self.reg_coef_log_ic[agent_idx], -4, 4)

    @torch.no_grad()
    def update_lambda(self, p):
        which_lambda = 'target_p' if p is True else 'target_ic'
        trajectory = self.labmda_coef_data.get_trajectory()
        # self.labmda_coef_data.clear()
        for agent_idx in range(self.n_agents):
            target = np.array(trajectory[f'agent_{agent_idx}'][which_lambda])
            target[target > 0] *= self.cfg.mediator.lambda_pos_coef
            target[target < 0] *= self.cfg.mediator.lambda_neg_coef

            if p is False:
                target[target < 0] *= 0.1

            if len(target) == 0:
                return

            if p is True:
                self.reg_coef_log_p[agent_idx] = self.reg_coef_log_p[agent_idx] - self.lr_lambda * target.mean()
                self.reg_coef_log_p[agent_idx] = torch.clip(self.reg_coef_log_p[agent_idx], -4, 4)
            else:
                self.reg_coef_log_ic[agent_idx] = self.reg_coef_log_ic[agent_idx] - self.lr_lambda * target.mean()
                self.reg_coef_log_ic[agent_idx] = torch.clip(self.reg_coef_log_ic[agent_idx], -4, 4)

    @torch.no_grad()
    def calculate_ic(self, trajectory, mask):
        # \lambda_i * ( r + v_i_in(state') - v_i_out(state) )
        mask_not_alone = trajectory['coalition_actor'][mask].sum(-1) != 1
        mask = mask[mask_not_alone]

        agent_idx = trajectory['agent'][mask].long()
        coal_neg = trajectory['coalition_actor'][mask].clone()
        coal_neg = coal_neg.scatter(dim=1, index=agent_idx, value=0)

        rewards = trajectory['rewards_actor'][mask].gather(dim=1, index=agent_idx)
        done = trajectory['done'][mask]

        v_i_in_prime = self.critic([trajectory['next_state_global_actor'][mask],
                                    trajectory['next_coalition_actor'][mask]]).gather(dim=1, index=agent_idx)

        v_i_out = self.critic([trajectory['state_global_actor'][mask],
                               coal_neg]).gather(dim=1, index=agent_idx)

        reg_term = (rewards + (1 - done) * self.gamma * v_i_in_prime) - v_i_out

        for i, term in enumerate(reg_term):
            self.labmda_coef_data.add(agent_id=agent_idx[i].item(), target_ic=term.item())

        self.reg_term_ic[mask] = self.reg_coef_log_ic[agent_idx.squeeze(-1)].exp() * reg_term

    @torch.no_grad()
    def calculate_p(self, trajectory, mask):
        # 1 / |C_{out}| * \sum_{j \notin C} \lambda_j * ( v_j_counter_coal(s) - r + v_j_out(s') )
        agent_idx = trajectory['agent'][mask].long()
        rewards = trajectory['rewards_actor'][mask]
        done = trajectory['done'][mask]

        v_prime = self.critic([trajectory['next_state_global_actor'][mask],
                               trajectory['next_coalition_actor'][mask]])
        td_target = rewards + (1 - done) * self.gamma * v_prime

        # rgb_global = []
        # rest_global = []
        # obs_rgb_critic = []
        coal_counterfactual = []
        who_not_in_coal = []
        final_state_idx = []  # indexes that correspond to real state indexes
        idx = []

        repeat = []
        count = 0

        # we iterate through each state to find all counterfactual coalitions
        for i in range(len(agent_idx)):
            coal = trajectory['coalition_actor'][mask][i]
            if coal.sum() < 2 or coal.sum() == self.n_agents:
                repeat.append(0)  # if the coalition is < 2 or full, discard this state in big tensors, i.e. P=0
                continue

            # почему первый агент всегда в коалиции?
            who_not = torch.where(coal == 0)[0].detach().cpu().numpy()  # find which of the agents not in coalition
            idx_cycle = []  # array for indexes that correspond to a state i in resulting big tensors

            for j in who_not:
                coal_counterfactual_tmp = coal.clone()
                coal_counterfactual_tmp[j] = 1  # make coalition counterfactual

                # rgb_global.append(trajectory['state_global_actor'][mask][i])
                # rest_global.append(trajectory['rest_global_actor'][mask][i])
                # obs_rgb_critic.append(trajectory['obs_rgb_critic'][mask][i])
                coal_counterfactual.append(coal_counterfactual_tmp)
                idx_cycle.append(count)  # append index
                count += 1  # increase index

            # after we collected all counterfactual coalitions,
            # append how many times we shall repeat an observation in big tensors
            repeat.append(len(who_not))
            idx.append(idx_cycle)
            who_not_in_coal.extend(who_not)  # keep track of all defectors to gather their values in critic heads
            final_state_idx.append(i)  # append an index of state if it was not discarded

        if len(coal_counterfactual) == 0:
            return

        coal_counterfactual = torch.stack(coal_counterfactual, dim=0).to(self.cfg.train_device)
        # rgb_global = torch.stack(rgb_global, dim=0).to(self.cfg.train_device)
        # rest_global = torch.stack(rest_global, dim=0).to(self.cfg.train_device)
        # obs_rgb_critic = torch.stack(obs_rgb_critic, dim=0).to(self.cfg.train_device)
        who_not_in_coal = torch.tensor(who_not_in_coal).unsqueeze(-1).to(self.cfg.train_device)
        repeat = torch.tensor(repeat).to(self.cfg.train_device)

        # repeat states as many as we have agents not in coalition
        rgb_global = torch.repeat_interleave(trajectory['state_global_actor'][mask], repeat, dim=0)
        # obs_critic = torch.repeat_interleave(trajectory['obs_rgb_critic'][mask], repeat, dim=0)
        td_target = torch.repeat_interleave(td_target, repeat, dim=0)
        # v_j_counter = self.critic([rgb_global, rest_global, obs_rgb_critic, coal_counterfactual])

        v_j_counter = torch.zeros((len(rgb_global), self.n_agents), device=self.cfg.train_device)
        # batched processing
        for k in range(0, len(rgb_global), len(mask)):
            mask_batch = np.arange(k, np.minimum(k + len(mask), len(rgb_global)))
            v_j_counter[mask_batch] = self.critic([rgb_global[mask_batch],
                                                   coal_counterfactual[mask_batch]])

        # get only those idx which correspond to agent j not in coalition
        constraint = td_target.gather(dim=1, index=who_not_in_coal) - v_j_counter.gather(dim=1, index=who_not_in_coal)

        # collect difference for lambda learning
        for num, agent_id in enumerate(who_not_in_coal):  # id агента не совпадает с номером эл. в массиве
            self.labmda_coef_data.add(agent_id=agent_id.item(), target_p=constraint[num].item())

        # get means for agents not in coalition for each state, that's a tricky spot
        for l, idx_mean in enumerate(idx):
            idx_who = who_not_in_coal[idx_mean].squeeze(-1)
            constraint_sum = (self.reg_coef_log_p[idx_who].exp() * constraint[idx_mean]).sum()
            self.reg_term_p[mask[final_state_idx[l]]] = constraint_sum

    @torch.no_grad()
    def compute_ppo_stats(self, trajectory, traj_critic):
        # state_global_actor, obs_local_actor, obs_local_critic,
        # action, rewards, next_state_global, next_obs, coalition, done
        # agent_idx = trajectory['agent']
        action = trajectory['actions'].squeeze(-1)
        rewards = trajectory['rewards_actor']
        done = trajectory['done']
        coalition = trajectory['coalition_actor']
        agent_idx = F.one_hot(trajectory['agent'].squeeze(-1).long(), self.cfg.env.n_agents).float()

        rewards_critic = traj_critic['rewards_critic']
        done_critic = traj_critic['done']
        batch_size = self.cfg.env.batch_size
        self.labmda_coef_data.clear()

        self.rets = torch.zeros((done.shape[0], self.n_agents), device=self.cfg.train_device, dtype=torch.float32)
        self.rets_critic = torch.zeros((done_critic.shape[0], self.n_agents), device=self.cfg.train_device, dtype=torch.float32)
        self.adv = torch.zeros((done.shape[0], 1), device=self.cfg.train_device, dtype=torch.float32)
        self.reg_term_ic = torch.zeros((done.shape[0], 1), device=self.cfg.train_device, dtype=torch.float32)
        self.reg_term_p = torch.zeros((done.shape[0], 1), device=self.cfg.train_device, dtype=torch.float32)
        self.log_old = torch.zeros((done.shape[0], 1), device=self.cfg.train_device, dtype=torch.float32)

        for i in range(0, len(done), batch_size):
            mask = np.arange(i, np.minimum(i + batch_size, len(done)))

            self.rets[mask] = rewards[mask] + (1 - done[mask]) * self.gamma * self.critic(
                [trajectory['next_state_global_actor'][mask],
                 trajectory['next_coalition_actor'][mask]]
            )

            self.adv[mask] = (coalition[mask] * (self.rets[mask] - self.critic(
                [trajectory['state_global_actor'][mask],
                 trajectory['coalition_actor'][mask]]))
                              ).sum(1, keepdim=True) / coalition[mask].sum(1, keepdim=True)

            if self.cfg.mediator.ic is True:
                # need to make it consistent inplace (to make it otw is a pain in the arse)
                self.calculate_ic(trajectory, mask)

            if self.cfg.mediator.p is True:
                self.calculate_p(trajectory, mask)

            policy = self.get_policy([trajectory['obs_actor'][mask],
                                      trajectory['coalition_actor'][mask],
                                      agent_idx[mask]
                                      ], off_policy=False)

            self.log_old[mask] = policy.log_prob(action[mask]).unsqueeze(-1)

        for i in range(0, len(done_critic), batch_size):
            mask = np.arange(i, np.minimum(i + batch_size, len(done_critic)))

            self.rets_critic[mask] = rewards_critic[mask] + (1 - done_critic[mask]) * self.gamma * self.critic(
                [traj_critic['next_state_global_critic'][mask],
                 traj_critic['next_coalition_critic'][mask]]
            )

    def update_ppo_actor(self, trajectory, idx):
        """
        rets, adv, log_old -- отсоединены от графа вычислений!
        """
        agent_idx = trajectory['agent'][idx].squeeze(-1).long()
        agent_idx_ohe = F.one_hot(trajectory['agent'][idx].squeeze(-1).long(), self.cfg.env.n_agents)
        action = trajectory['actions'][idx].squeeze(-1)

        rets, adv, log_old, reg_term_ic = self.rets[idx].detach(), self.adv[idx].detach(), \
                                          self.log_old[idx].detach(), self.reg_term_ic[idx].detach()
        reg_term_p = self.reg_term_p[idx].detach()

        adv = adv + reg_term_ic - reg_term_p
        adv = loss.advantage(adv)  # advantage normalization
        # coalition is inside obs_local
        policy = self.get_policy([trajectory['obs_actor'][idx],
                                  trajectory['coalition_actor'][idx],
                                  agent_idx_ohe.float()
                                  ], off_policy=False)

        log_prob = policy.log_prob(action).unsqueeze(-1)
        rho = torch.exp(log_prob - log_old)
        entropy = policy.entropy().mean()

        # agent_actor_loss = adv * rho - self.entropy_coef * entropy
        agent_actor_loss = loss.ppo_loss(adv, rho) - self.entropy_coef * entropy

        self.opt_actor.zero_grad()
        agent_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.opt_actor.step()

        return agent_actor_loss.item(), reg_term_ic.cpu().numpy(), reg_term_p.cpu().numpy()

    def update_ppo_critic(self, trajectory, idx):
        critic_pred = self.critic([trajectory['state_global_critic'][idx],
                                   trajectory['coalition_critic'][idx]])

        agent_critic_loss = loss.valueLoss(critic_pred, self.rets_critic[idx])
        r2 = r2_score(self.rets_critic[idx].detach().cpu(), critic_pred.detach().cpu())

        self.opt_critic.zero_grad()
        agent_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.opt_critic.step()

        return agent_critic_loss.item(), r2

    def update_coef_bc(self):
        self.bc_coef = np.maximum(0, self.bc_coef - self.cfg.mediator.bc_coef_decrease)

    def get_coalition_steps(self):
        return self.coalition_steps

    def restart_coalition(self):
        self.coalition_steps = np.zeros((self.cfg.env.n_agents, 1))

    def get_coalition(self):
        coalition = np.zeros_like(self.coalition_steps)
        steps = self.get_coalition_steps()
        coalition[steps > 0] = 1
        coalition[steps < 0] = -1

        return coalition

    def update_coalition(self, agent_idx_in, agent_idx_out):
        idx_reset_in = np.where(self.coalition_steps[agent_idx_in] == 0)[0]
        agent_idx_in = agent_idx_in[idx_reset_in]
        self.coalition_steps[agent_idx_in] = self.cfg.env.k_mediator

        idx_reset_out = np.where(self.coalition_steps[agent_idx_out] == 0)[0]
        agent_idx_out = agent_idx_out[idx_reset_out]
        self.coalition_steps[agent_idx_out] = -1 * self.cfg.env.k_mediator

    def decrement_coalition(self, agent_idx):
        self.coalition_steps[agent_idx] = self.coalition_steps[agent_idx] - 1

    def increment_coalition(self, agent_idx):
        self.coalition_steps[agent_idx] = self.coalition_steps[agent_idx] + 1
