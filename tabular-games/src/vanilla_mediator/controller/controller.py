import numpy as np
import torch
import torch.nn.functional as F

from src.base.controller.controller_base import EyeOfGodBase
from src.vanilla_mediator.agent.actor_critic_agent import ActorAgent, CriticAgent
from src.vanilla_mediator.agent.agent import Agent as AgentVanilla
from src.vanilla_mediator.mediator.actor_critic_mediator import ActorMediator, CriticMediator
from src.vanilla_mediator.mediator.mediator import Mediator as MediatorVanilla


class EyeOfGodVanilla(EyeOfGodBase):
    def __init__(self, cfg, Agent=AgentVanilla, Mediator=MediatorVanilla,
                 nn_agent=(ActorAgent, CriticAgent), nn_mediator=(ActorMediator, CriticMediator)):
        self.cfg = cfg

        n_inputs = [
            cfg.env.state_size,  # agent_actor_n_inputs
            cfg.env.state_size,  # agent_critic_n_inputs
            cfg.env.state_size + 2 * cfg.env.n_agents,  # mediator_actor_n_inputs: state + coalition + agent_i
            cfg.env.state_size + 1 * cfg.env.n_agents  # mediator_critic_n_inputs: state + coalition
        ]

        super(EyeOfGodVanilla, self).__init__(Agent, Mediator, nn_agent, nn_mediator, n_inputs, cfg)

    def _get_batch(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        d_st, st, act_agents, act_mediator, coal, rew, n_st, n_coal, d = map(np.array, zip(*transitions))

        idx = np.random.randint(0, len(transitions), self.batch_size)  # Choose random batch

        state = torch.tensor(st[idx], device=self.device, dtype=self.dtype)
        dummy_state = torch.tensor(d_st[idx], device=self.device, dtype=self.dtype)
        actions_agents = torch.tensor(act_agents[idx], device=self.device, dtype=self.dtype)
        actions_mediator = torch.tensor(act_mediator[idx], device=self.device, dtype=self.dtype)
        coalition = torch.tensor(coal[idx], device=self.device, dtype=self.dtype)
        rewards = torch.tensor(rew[idx], device=self.device, dtype=self.dtype)
        next_state = torch.tensor(n_st[idx], device=self.device, dtype=self.dtype)
        next_coalition = torch.tensor(n_coal[idx], device=self.device, dtype=self.dtype)
        done = torch.tensor(d[idx], device=self.device, dtype=self.dtype).unsqueeze(-1)

        assert state.dim() == 2, f'{state.shape=}'
        assert dummy_state.dim() == 2, f'{dummy_state.shape=}'
        assert actions_agents.dim() == 2, f'{actions_agents.shape=}'
        assert actions_mediator.dim() == 2, f'{actions_mediator.shape=}'
        assert coalition.dim() == 2, f'{coalition.shape=}'
        assert rewards.dim() == 2, f'{rewards.shape=}'
        assert next_state.dim() == 2, f'{next_state.shape=}'
        assert done.dim() == 2, f'{done.shape=}'

        batch = [state, dummy_state, actions_agents, actions_mediator,
                 coalition, rewards, next_state, next_coalition, done]

        return batch

    def update(self, trajectories):
        state, dummy_state, actions_agents, actions_mediator, \
        coalition, rewards, next_state, next_coalition, done = self._get_batch(trajectories)

        # reward_coalition = torch.sum(rewards * coalition, dim=1).unsqueeze(-1)

        # Update agents
        for i, agent in enumerate(self.agents):
            agent.update_agent(state,  # obs
                               actions_agents[:, i],
                               rewards[:, i].unsqueeze(-1),
                               next_state,
                               done)  # auxiliary state for actor loss computation

        # Update mediator
        if self.cfg.mediator.enabled is True:
            obs_mediator = torch.cat([state, coalition], dim=-1)
            next_obs_mediator = torch.cat([next_state, next_coalition], dim=-1)
            self.mediator.update_mediator(obs_mediator,  # obs
                                          actions_mediator,
                                          rewards,
                                          next_obs_mediator,  # next_obs
                                          coalition,
                                          done)

    def sample_episode(self, env, test=False):
        state, done = env.reset(), 0
        trajectory = []
        mediatior_action = self.cfg.agent.action_size - 1

        # For evaluation only
        pick_mediator = []
        all_rewards = []
        ts = 0

        while not done:
            # Agents' moves
            actions_agents = []
            # dummy_state = np.zeros(self.cfg.env.state_size)  # [0, 0]
            for i, agent in enumerate(self.agents):
                obs_ag = self._tensorify(state)
                with torch.no_grad():
                    act = agent.step(obs_ag)
                actions_agents.append(act)

            actions_agents = np.array(actions_agents)
            actions_to_env = actions_agents.copy()

            who = np.where(actions_agents == mediatior_action)[0]  # indices of agents which chose mediator
            coalition = F.one_hot(torch.tensor(who), self.n_agents).sum(dim=0).numpy()

            if ts > 0:
                trajectory.append((state_prev, state_prev, actions_agents_prev, actions_mediator, coalition_prev,
                                   rewards, next_state, coalition, done))

            # action "-1" means the mediator was not chosen
            actions_mediator = np.full(self.n_agents, -1)

            # Mediator's move
            for i in range(self.n_agents):
                if actions_to_env[i] == mediatior_action:
                    agent_i = F.one_hot(torch.tensor(self.agents[i].agent_i), self.n_agents)
                    obs_med = self._tensorify([state, coalition, agent_i])
                    with torch.no_grad():
                        actions_mediator[i] = self.mediator.step(obs_med)
                    actions_to_env[i] = actions_mediator[i]

            next_state, rewards, done = env.step(*actions_to_env)
            # trajectory.append((state, state, actions_agents, actions_mediator, coalition.copy(),
            #                    rewards, next_state, coalition.copy(), done))

            state_prev = np.array(state).copy()
            state = next_state

            all_rewards.append(sum(rewards))
            pick_mediator.append(np.sum(coalition) / self.n_agents)
            ts += 1

            actions_agents_prev = actions_agents
            coalition_prev = coalition.copy()

        trajectory.append((state_prev, state_prev, actions_agents, actions_mediator, coalition.copy(),
                           rewards, next_state, coalition, done))

        if test:
            return all_rewards, pick_mediator, actions_to_env

        return trajectory

    def train(self, env, log=False):
        eval_episodes = self.cfg.env.eval_episodes

        # get batch of trajectories
        for i in range(self.cfg.env.iterations):
            trajectories = []
            steps_ctn = 0
            while len(trajectories) < self.cfg.env.min_episodes_per_update \
                    or steps_ctn < self.cfg.env.min_transitions_per_update:
                traj = self.sample_episode(env)
                steps_ctn += len(traj)
                trajectories.append(traj)

            # update
            self.update(trajectories)

            if log:
                if i % 100 == 0:
                    print(f'ITERATION {i}:')
                    self.evaluate_policy(eval_episodes, env)

    def evaluate_policy(self, n_episodes, env):
        all_rewards = []
        pick_mediator = []
        total_info = []

        for e in range(n_episodes):
            rewards, mediator, info = self.sample_episode(env, test=True)
            all_rewards.append(rewards)
            pick_mediator.append(mediator)
            total_info.extend(info)

        info = env.log(self, all_rewards, pick_mediator, total_info)

        return info
