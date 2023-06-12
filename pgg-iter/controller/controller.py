import torch
import numpy as np
from agent.agent import Agent
from mediator.mediator import Mediator
import wandb

np.set_printoptions(precision=4, floatmode='fixed')

import torch.nn.functional as F
import time
from utils.utils import MediatorTraj, gini, process_webm, AgentTraj, check_consistency, SchedulerBC
from collections import defaultdict


class HarvestController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_agents = self.cfg.env.n_agents
        self.agents = [Agent(self.cfg, self.cfg.agent, i) for i in range(self.n_agents)]
        self.mediator = Mediator(self.cfg, self.cfg.mediator)
        self.gamma = self.cfg.agent.gamma

        # self.device = cfg.device

        self.train_device = self.cfg.train_device
        self.inference_device = self.cfg.inference_device
        self.dtype = torch.float32
        self.batch_size = self.cfg.env.batch_size
        self.ppo_epochs = self.cfg.env.ppo_epochs
        self.ppo_epochs_mediator = self.cfg.env.ppo_epochs_mediator

        self.bc_scheduler = SchedulerBC()

    def _make_dict(self, inp):
        return {f'player_{i}': inp[i] for i in range(self.n_agents)}

    def _tensorify_mediator(self, inp):
        state_global, local_obs = inp
        local_obs = local_obs.unsqueeze(0).to(self.inference_device)

        return [state_global, local_obs]

    def _tensorify(self, obs_dict):
        timestep = np.array(obs_dict['ts'] / 10).reshape(1, -1)

        # for i, el in enumerate(inp):
        obs = obs_dict['obs'] / 57.66  # 5904
        obs = np.concatenate([obs.reshape(1, -1), timestep], axis=-1)

        obs = torch.tensor(obs, device=self.inference_device, dtype=self.dtype)

        return [obs]

    def _prepare_globals(self, obs):
        obs[0, :-1] = obs[0, :-1] / 57.66  # 5904
        obs[0, -1] = obs[0, -1] / 10

        obs = torch.tensor(obs, device=self.inference_device, dtype=self.dtype)

        return obs

    @torch.no_grad()
    def sample_episode(self, env, test=False, print_trajectory=False):
        [agent.select_device(self.inference_device) for agent in self.agents]
        self.mediator.select_device(self.inference_device)

        self.mediator.restart_coalition()
        trajectory_agents = AgentTraj(self.n_agents)
        trajectory_mediator = MediatorTraj()
        trajectory_mediator_critic = MediatorTraj()
        bc_traj = MediatorTraj(self.n_agents)
        mediator_action = self.cfg.mediator.action_size

        # For evaluation only
        pick_mediator = []
        all_rewards = []
        next_obs_rgb_glob = []
        mediator_perc = []
        final_balance = []
        med_c = []

        if print_trajectory or test:
            num_iter = 1
        else:
            num_iter = 10

        for _ in range(num_iter):
            self.mediator.restart_coalition()
            state_global, obs, done = env.reset()
            state_global = self._prepare_globals(state_global)

            timestep = 0
            while not done:
                # Agents' moves
                actions_agents = []
                obs_agents = []

                state_global_batch = []
                local_obs_med = []
                coalition_batch = []
                idx_batch = []
                agents_in = set()

                balance_old = env.balance.copy()
                under_mediator = torch.tensor(self.mediator.get_coalition(), dtype=self.dtype)
                # coalition codes – 0: can choose to enter coalition; 1: in coalition; -1: restricted from coalition
                zeros_old = len(torch.where(under_mediator == 0)[0])
                for i, agent in enumerate(self.agents):
                    obs_ag = self._tensorify(obs[f'player_{i}'])
                    obs_ag.append(under_mediator[i])

                    with torch.no_grad():
                        act = agent.step(obs_ag)

                    if under_mediator[i] == 0 or under_mediator[i] == -1:
                        trajectory_agents.add(agent_id=agent.idx,
                                              obs=obs_ag[0].squeeze(0),
                                              action=[act],
                                              under_mediator=[under_mediator[i].item()])

                    obs_agents.append(obs_ag[0].cpu().detach())
                    actions_agents.append(act.item())

                obs_agents = torch.cat(obs_agents, dim=0)

                actions_agents = np.array(actions_agents)
                actions_to_env = actions_agents.copy()

                who = np.where(actions_agents == mediator_action)[0]  # indices of agents which chose mediator
                who_not = np.where(actions_agents != mediator_action)[0]
                coalition = F.one_hot(torch.tensor(who), self.n_agents).sum(dim=0) \
                    .to(self.inference_device).reshape(-1, self.n_agents).float()
                zeros_new = len(np.where(coalition == 0)[0])

                if timestep > 0:
                    # add next_coalition to critic med
                    trajectory_mediator_critic.add(
                        agent_id=999,
                        next_coalition_critic=coalition,
                    )
                    # add next_coalition to actor med
                    for agent_idx in who_old:
                        trajectory_mediator.add(
                            agent_id=agent_idx,
                            next_coalition_actor=coalition,
                        )

                if zeros_old > 0:
                    mediator_p = np.maximum(0, (zeros_old - zeros_new) / zeros_old)
                    mediator_perc.append(mediator_p)

                self.mediator.update_coalition(who, who_not)
                actions_mediator = np.full(self.n_agents, -1)

                # Mediator's move
                for i in range(self.n_agents):
                    if actions_agents[i] == mediator_action:
                        agent_i = F.one_hot(torch.tensor(self.agents[i].idx), self.n_agents) \
                            .reshape(-1, self.n_agents).to(self.inference_device)
                        # global, local, coalition, idx
                        obs_med = self._tensorify_mediator([state_global, obs_agents[i]])

                        state_global_batch.append(obs_med[0])
                        local_obs_med.append(obs_med[1])
                        coalition_batch.append(coalition)
                        idx_batch.append(agent_i.float())

                        # decrement coalition
                        self.mediator.decrement_coalition(self.agents[i].idx)

                        # state_global, rest_global, local_rgb, local_rest
                        # add trajectory for actor update
                        trajectory_mediator.add(
                            agent_id=self.agents[i].idx,
                            state_global_actor=obs_med[0].cpu().detach(),
                            obs_actor=obs_med[0].cpu().detach(),
                            obs_critic=obs_agents.unsqueeze(0),
                            coalition_actor=coalition,
                            agent=[self.agents[i].idx]
                        )
                    else:
                        self.mediator.increment_coalition(self.agents[i].idx)

                # state_global_critic, _, _, _ = self._tensorify_mediator([state_global, obs_agents[0]])
                # add trajectory for critic mediator update
                trajectory_mediator_critic.add(
                    agent_id=999,  # for critic there are no separate agents
                    state_global_critic=state_global,
                    coalition_critic=coalition,
                    obs_critic=obs_agents.unsqueeze(0),
                )

                if len(who) > 0:
                    state_global_batch = torch.cat(state_global_batch, dim=0)
                    # rest_global_batch = torch.cat(rest_global_batch, dim=0)
                    # local_obs_med = torch.cat(local_obs_med, dim=0)
                    coalition_batch = torch.cat(coalition_batch, dim=0)
                    idx_batch = torch.cat(idx_batch, dim=0)

                    obs_med_batch = [state_global_batch, coalition_batch, idx_batch]

                    with torch.no_grad():
                        actions_mediator[who] = self.mediator.step(obs_med_batch)

                    cooperate_mediator = len(actions_mediator[actions_mediator == 1]) \
                                         / len(actions_mediator[actions_mediator != -1])
                    med_c.append(cooperate_mediator)

                    actions_to_env[who] = actions_mediator[who]

                # step
                next_state_global, next_obs, rewards, done, _ = env.step(self._make_dict(actions_to_env))

                next_state_global = self._prepare_globals(next_state_global)
                next_obs_agents = []

                # идея такая:
                # если next_under_mediator == 1 (агент вступил/сидит в коалиции) и это не последний шаг в среде:
                #   добавляем награду в список н-степ
                # если агент был на предыдущем шаге в коалиции, но на этом вышел (k-шагов прошли):
                #   добавляем последнюю награду в список, затем считаем н-степ и добавляем все в буфер
                # если агент не находится в коалиции:
                #   просто добавляем следующий стейт и награду в буфер
                next_under_mediator = self.mediator.get_coalition()
                for i, agent in enumerate(self.agents):
                    next_obs_ag = self._tensorify(next_obs[f'player_{i}'])
                    next_obs_agents.append(next_obs_ag[0].cpu().detach())

                    if (next_under_mediator[i] == 1) and (done is False):
                        trajectory_agents.add(agent_id=agent.idx,
                                              reward_under=rewards[agent.idx])
                    elif ((next_under_mediator[i] == 0) and (under_mediator[i] == 1)) \
                            or ((done is True) and (next_under_mediator[i] == 1)):
                        trajectory_agents.add(agent_id=agent.idx,
                                              reward_under=rewards[agent.idx])
                        len_k = len(trajectory_agents.get_reward_under(agent_id=agent.idx))
                        rew_nstep = trajectory_agents.calculate_nstep(agent_id=agent.idx)
                        trajectory_agents.add(agent_id=agent.idx,
                                              next_obs=next_obs_ag[0].cpu().detach().squeeze(0),
                                              done=[done],
                                              reward=rew_nstep,
                                              k=[len_k])
                    elif next_under_mediator[i] == 0 or next_under_mediator[i] == -1:
                        trajectory_agents.add(agent_id=agent.idx,
                                              next_obs=next_obs_ag[0].cpu().detach().squeeze(0),
                                              done=[done],
                                              reward=rewards[agent.idx],
                                              k=[1])

                next_obs_agents = torch.cat(next_obs_agents, dim=0)

                for i, agent in enumerate(self.agents):
                    if actions_agents[i] == mediator_action:
                        # agent_i = F.one_hot(torch.tensor(agent.idx), self.n_agents)
                        # global, local, coalition, idx
                        next_obs_med = self._tensorify_mediator([next_state_global, next_obs_agents[i]])

                        # state_global, rest_global, local_rgb, local_rest
                        trajectory_mediator.add(
                            agent_id=agent.idx,
                            next_state_global_actor=next_obs_med[0].cpu().detach(),
                            next_obs_critic=next_obs_agents.unsqueeze(0),
                            next_obs_actor=next_obs_med[1].cpu().detach(),
                            actions=[actions_mediator[agent.idx]],
                            rewards_actor=rewards.reshape(1, self.n_agents),
                            next_agent=[agent.idx],
                            done=[done]
                        )
                        agents_in.add(agent.idx)

                # [state_global, rest_global, local_rgb, local_rest]
                # next_state_global_critic, _, _, _ = self._tensorify_mediator([next_state_global, next_obs[0]])

                # add trajectory for critic mediator update
                trajectory_mediator_critic.add(
                    agent_id=999,
                    next_state_global_critic=next_state_global,
                    next_obs_critic=next_obs_agents.unsqueeze(0),
                    rewards_critic=rewards.reshape(1, self.n_agents),
                    done=[done]
                )

                all_rewards.append(rewards)
                pick_mediator.append(torch.sum(coalition) / self.n_agents)

                obs, state_global = next_obs, next_state_global
                who_old = who.copy()

                if print_trajectory is True:
                    print('-----------------------')
                    print(f'TIMESTEP: {timestep}')
                    print(f'BALANCE: {balance_old[0]}')
                    print()

                    for i, agent in enumerate(self.agents):
                        print(f'AGENT {i}', end=' ')
                        policy = agent.get_policy([obs_agents[i].unsqueeze(0), under_mediator[i]], True)
                        print(policy.probs.numpy()[0])

                    print(f'ACTIONS AGENTS: {actions_agents}')
                    print()
                    print(f'COALITION: {coalition.long().numpy()[0]}')

                    if len(who) > 0:
                        print(f'MEDIATOR POLICY', end='\n')
                        policy = self.mediator.get_policy(obs_med_batch, False)
                        print(policy.probs.numpy())

                    print(f'ACTIONS MEDIATOR: {actions_mediator}')
                    print()
                    print(f'REWARDS: {rewards}')
                    print()

                timestep += 1

            agents_in = list(agents_in)

            trajectory_mediator_critic.add(
                agent_id=999,
                next_coalition_critic=torch.zeros_like(coalition),
            )
            for agent_id in agents_in:
                trajectory_mediator.add(agent_id)
                trajectory_mediator.add(
                    agent_id=agent_id,
                    next_coalition_actor=torch.zeros_like(coalition),
                )

            final_balance.append(env.balance)

        gini_coef = gini(np.array(all_rewards).sum(0))
        info = [np.concatenate(final_balance), mediator_perc, pick_mediator, gini_coef, np.mean(med_c)]

        if test:
            return info, next_obs_rgb_glob

        trajectory_agents = trajectory_agents.get_trajectory()
        trajectory_mediator = trajectory_mediator.get_trajectory()
        trajectory_mediator_critic = trajectory_mediator_critic.get_trajectory()
        # bc_traj = bc_traj.get_trajectory()

        check_consistency(trajectory_agents)
        check_consistency(trajectory_mediator)
        check_consistency(trajectory_mediator_critic)

        return trajectory_agents, trajectory_mediator, trajectory_mediator_critic, bc_traj, info

    def _get_batch(self, trajectories):
        f = lambda x: torch.stack(x, dim=0).to(self.train_device).float() \
            if torch.is_tensor(x[0]) else \
            torch.tensor(np.stack(x, axis=0), device=self.train_device, dtype=self.dtype)

        for agent_i in range(self.n_agents):
            for k, v in trajectories[f'agent_{agent_i}'].items():
                trajectories[f'agent_{agent_i}'][k] = f(v)

        return trajectories

    def _get_batch_mediator(self, trajectories, critic=False, bc=False):
        f = lambda x: torch.cat(x, dim=0).to(self.train_device).float() \
            if torch.is_tensor(x[0]) else \
            torch.tensor(np.concatenate(x, axis=0), device=self.train_device, dtype=self.dtype)

        if critic is False:
            trajectories.pop('agent_999')

            traj = defaultdict(list)

            active_agents = 0
            for k in trajectories.keys():
                if len(trajectories[k]['obs_actor']):
                    active_agents += 1

            for agent in trajectories.keys():
                if len(trajectories[agent]['obs_actor']) == 0:
                    continue

                len_agent_obs = len(trajectories[agent]['obs_actor'])
                idx = np.random.randint(0, len_agent_obs, 200 // active_agents)
                for k_in, v_in in trajectories[agent].items():
                    traj[k_in].extend(v_in[i] for i in idx)
        else:
            traj = trajectories['agent_999']

        for k, v in traj.items():
            v = f(v)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            traj[k] = v

        return traj

    def update_mediator(self, trajectories, trajectories_critic, bc):
        ts = time.time()
        self.mediator.select_device(self.train_device)
        trajectories_actor = self._get_batch_mediator(trajectories)
        trajectories_critic = self._get_batch_mediator(trajectories_critic, critic=True)
        # trajectories_bc = self._get_batch_mediator(bc, critic=False, bc=True)
        info = defaultdict(list)

        if len(trajectories_actor['done']) == 0:
            return info

        # print(f'trajectory LEN for mediator ACTOR: {len(trajectories_actor["done"])}')
        # print(f'trajectory LEN for mediator CRITIC: {len(trajectories_critic["done"])}')

        self.mediator.compute_ppo_stats(trajectories_actor, trajectories_critic)
        num_iterations = np.ceil(len(trajectories_actor["done"]) / self.batch_size).astype(int)

        for _ in range(self.cfg.env.ppo_epochs):
            for __ in range(num_iterations):
                idx_actor = np.random.randint(0, len(trajectories_actor['done']), self.batch_size)
                idx_critic = np.random.randint(0, len(trajectories_critic['done']), self.batch_size)
                loss_actor, reg_term_ic, reg_term_p = self.mediator.update_ppo_actor(trajectories_actor, idx_actor)
                loss_critic, r2 = self.mediator.update_ppo_critic(trajectories_critic, idx_critic)
                info['loss_actor'].append(loss_actor)
                info['reg_term_ic'].append(reg_term_ic)
                info['reg_term_p'].append(reg_term_p)
                info['loss_critic'].append(loss_critic)
                info['r2_score'].append(r2)

        # if self.mediator.bc_coef > 0:
        #     idx_bc = np.random.randint(0, len(trajectories_bc['obs_rgb_actor']), self.batch_size)
        #     self.mediator.update_bc(trajectories_bc, idx_bc)

        if self.cfg.mediator.ic is True:
            self.mediator.update_lambda(p=False)
        if self.cfg.mediator.p is True:
            self.mediator.update_lambda(p=True)
        info['reg_coef_p'] = self.mediator.reg_coef_log_p.exp().cpu().numpy()
        info['reg_coef_ic'] = self.mediator.reg_coef_log_ic.exp().cpu().numpy()

        self.mediator.update_entropy()
        # self.mediator.bc_coef = self.bc_scheduler.step()

        for k, v in info.items():
            info[k] = np.mean(v)

        # print(f'MEDIATOR update is done in {time.time() - ts:.2f} sec')

        return info

    def update_agents(self, trajectories):
        ts = time.time()

        trajectory = self._get_batch(trajectories)

        # obs_rgb, obs_rest, state_global, actions_agents, rewards, next_state_global, next_obs_rgb, \
        # next_obs_rest, coalition, under_mediator, done = batch
        # print('got batch')
        for agent in self.agents:
            agent.select_device(self.train_device)
            agent.compute_ppo_stats(trajectory[f'agent_{agent.idx}'])

        for agent in self.agents:
            traj = trajectory[f'agent_{agent.idx}']
            num_iterations = np.ceil(len(traj["done"]) / self.batch_size).astype(int)
            for _ in range(self.cfg.env.ppo_epochs):
                for __ in range(num_iterations):
                    idx = np.random.randint(0, len(traj['done']), self.batch_size)
                    agent.update_ppo(traj, idx)

        [agent.update_entropy() for agent in self.agents]

        # print(f'AGENT update is done in {time.time() - ts:.2f} sec\n')

    def train(self, env):
        eval_episodes = self.cfg.env.eval_episodes

        # get batch of trajectories
        for i in range(self.cfg.env.iterations):
            rewards = []
            steps_ctn = 0

            ts = time.time()
            # info = [all_rewards, mediator_perc, pick_mediator]

            traj, trajectory_mediator, trajectory_mediator_critic, bc, info = self.sample_episode(env)
            rew, mediator_choose, mediator_choose_old, gini_coef, _ = info

            # print(f'trajectories collected in {time.time() - ts:.2f}')
            # update
            if self.cfg.mediator_on is True:
                info = self.update_mediator(trajectory_mediator, trajectory_mediator_critic, bc)
                wandb.log(data={
                    'mediator/loss_actor': info['loss_actor'],
                    'mediator/loss_critic': info['loss_critic'],
                    'mediator/reg_term_ic': info['reg_term_ic'],
                    'mediator/reg_term_p': info['reg_term_p'],
                    'mediator/r2_critic': info['r2_score'],
                    'mediator/coef_p': info['reg_coef_p'],  # mean is taken beforehand,
                    'mediator/coef_ic': info['reg_coef_ic']
                }, step=i)
            self.update_agents(traj)

            if i % 100 == 0:
                print(f'ITERATION {i}:')
                # rewards, mediator_perc, mediator_perc_old, gini_coef, _ = self.evaluate_policy(eval_episodes, env)
                print(f"\nMEAN BALANCE: {np.mean(rew):.2f}\tSTD GLOBAL REWARD: {np.std(rew):.2f}"
                      f"\t MEDIATOR ENTROPY: {self.mediator.entropy_coef:.4f}\t "
                      f"AGENT ENTROPY: {self.agents[0].entropy_coef:.4f}\t"
                      f"% MEDIATOR: {np.mean(mediator_choose_old):.4f}\n")
                #
                # wandb.log(data={
                #     'general_reward_mean': np.mean(rew),
                #     'general_reward_std': np.std(rew),
                #     'min_reward': np.min(rew),
                #     'max_reward': np.max(rew),
                #     'mediator_%': np.mean(mediator_choose),
                #     'mediator_%_old': np.mean(mediator_choose_old),
                #     'gini': gini_coef,
                #     'mediator_entropy': self.mediator.entropy_coef,
                #     'actor_entropy': self.agents[0].entropy_coef,
                # }, step=i)

        # self.print_trajectory(env)
        rewards, _, mediator_perc_old, _, med_c_arr = self.evaluate_policy(episodes=self.cfg.env.eval_episodes, env=env)

        return rewards, mediator_perc_old, med_c_arr

    def evaluate_policy(self, episodes, env):
        rewards = []
        mediator_perc = []
        mediator_perc_old = []
        gini_coef_arr = []
        med_c_arr = []

        for i in range(episodes):
            info, _ = self.sample_episode(env, test=True)
            # info = [all_rewards, mediator_perc, pick_mediator]
            all_rewards, mediator_choose, mediator_choose_old, gini_coef, med_c = info
            rewards.extend(all_rewards)
            mediator_perc.extend(mediator_choose)
            mediator_perc_old.extend(mediator_choose_old)
            gini_coef_arr.append(gini_coef)
            med_c_arr.append(med_c)

        gini_coef = np.array(gini_coef_arr).mean()
        rewards = np.mean(rewards)
        med_c_arr = np.mean(med_c_arr)
        mediator_perc_old = np.mean(mediator_perc_old)

        # self.print_log()

        return rewards, mediator_perc, mediator_perc_old, gini_coef, med_c_arr

    def print_trajectory(self, env, n_times=5):
        for _ in range(n_times):
            self.sample_episode(env, print_trajectory=True)
            print('##########################')
