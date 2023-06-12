import numpy as np
import torch
import torch.nn.functional as F

from src.base.controller.controller_base import EyeOfGodBase
from src.n_step_threat.agent.actor_critic_agent import ActorAgent, CriticAgent
from src.n_step_threat.agent.agent import Agent
from src.n_step_threat.mediator.actor_critic_mediator import ActorMediator, CriticMediator
from src.n_step_threat.mediator.mediator import Mediator
from utils.traj import ActorTraj


class EyeOfGodNStep(EyeOfGodBase):
    def __init__(self, cfg):
        self.cfg = cfg

        n_inputs = [
            cfg.env.state_size + 1,  # agent_actor_n_inputs
            cfg.env.state_size + 1,  # agent_critic_n_inputs
            cfg.env.state_size + 2 * cfg.env.n_agents + 1,  # mediator_actor_n_inputs
            cfg.env.state_size + cfg.env.n_agents + 1  # mediator_critic_n_inputs
        ]

        super(EyeOfGodNStep, self).__init__(Agent, Mediator, [ActorAgent, CriticAgent],
                                            [ActorMediator, CriticMediator],
                                            n_inputs, cfg)

        self.steps_commit = cfg.env.steps_commit

    def _get_batch(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        st, act_agents, act_mediator, coal_b, coal, rew, n_st, n_coal, d = map(np.array, zip(*transitions))
        # trajectory.append((state_prev, actions_agents, actions_mediator, coalition.copy(),
        #                    rewards, next_state, coalition, done))
        idx = np.random.randint(0, len(transitions), self.batch_size)  # Choose random batch

        state = torch.tensor(st[idx], device=self.device, dtype=self.dtype)
        actions_agents = torch.tensor(act_agents[idx], device=self.device, dtype=self.dtype)
        actions_mediator = torch.tensor(act_mediator[idx], device=self.device, dtype=self.dtype)
        coalition_agent = torch.tensor(coal_b[idx], device=self.device, dtype=self.dtype)
        coalition_mediator = torch.tensor(coal[idx], device=self.device, dtype=self.dtype)
        rewards = torch.tensor(rew[idx], device=self.device, dtype=self.dtype)
        next_state = torch.tensor(n_st[idx], device=self.device, dtype=self.dtype)
        next_coalition = torch.tensor(n_coal[idx], device=self.device, dtype=self.dtype)
        done = torch.tensor(d[idx], device=self.device, dtype=self.dtype).unsqueeze(-1)

        assert state.dim() == 2, f'{state.shape=}'
        assert actions_agents.dim() == 2, f'{actions_agents.shape=}'
        assert actions_mediator.dim() == 2, f'{actions_mediator.shape=}'
        assert coalition_mediator.dim() == 2, f'{coalition_mediator.shape=}'
        assert coalition_agent.dim() == 2, f'{coalition_mediator.shape=}'
        assert rewards.dim() == 2, f'{rewards.shape=}'
        assert next_state.dim() == 2, f'{next_state.shape=}'
        assert done.dim() == 2, f'{done.shape=}'

        batch = [state, actions_agents, actions_mediator, coalition_agent,
                 coalition_mediator, rewards, next_state, next_coalition, done]

        return batch

    def _get_batch_actor(self, trajectories):
        f = lambda x: torch.stack(x, dim=0).to(self.device).float() \
            if torch.is_tensor(x[0]) else \
            torch.tensor(np.stack(x, axis=0), device=self.device, dtype=self.dtype)

        for agent_i in range(self.n_agents):
            for k, v in trajectories[f'agent_{agent_i}'].items():
                trajectories[f'agent_{agent_i}'][k] = f(v)

        return trajectories

    def update(self, trajectories, trajectories_actor):
        state, actions_agents, actions_mediator, coalition_agent, \
        coalition_mediator, rewards, next_state, next_coalition, done = self._get_batch(trajectories)

        traj_act = self._get_batch_actor(trajectories_actor)

        # reward_coalition = torch.sum(rewards * coalition_mediator, dim=1).unsqueeze(-1)

        # Update agents
        for i, agent in enumerate(self.agents):
            traj_i = traj_act[f'agent_{i}']
            # obs_agent = torch.cat([traj_i['state'], traj_i['coalition']], dim=-1)
            # next_obs_agent = torch.cat([traj_i['next_state'], traj_i['coalition_next']], dim=-1)

            agent.update_agent(traj_i['state'],  # obs
                               traj_i['action'],
                               traj_i['reward'],
                               traj_i['next_state'],
                               traj_i['done'],
                               state_only=traj_i['state_only'])  # auxiliary state for actor loss computation

        # Update mediator
        obs_mediator = torch.cat([state, coalition_mediator], dim=-1)
        next_obs_mediator = torch.cat([next_state, next_coalition], dim=-1)
        self.mediator.update_mediator(obs_mediator,  # obs
                                      actions_mediator,
                                      rewards,
                                      next_obs_mediator,  # next_obs
                                      coalition_mediator,
                                      done)

    def sample_episode(self, env, test=False):
        state, done = env.reset(), 0
        trajectory = []
        coalition = [2, 2]  # 0: out of coalition; 2: can choose to enter coalition; 1: in coalition, mediator decides
        counter = [0, 0]

        # For evaluation only
        pick_mediator = []
        all_rewards = []
        trajectory_actors = ActorTraj(2)

        ts = 0
        while not done:
            # Agents' moves
            actions_agents = []
            for i, agent in enumerate(self.agents):
                obs_ag = self._tensorify([state, coalition[i]])
                with torch.no_grad():
                    act = agent.step(obs_ag)
                actions_agents.append(act)
                if coalition[i] != 1:
                    trajectory_actors.add(agent_id=i,
                                          state=obs_ag.squeeze(0),
                                          coalition=[coalition[i]],
                                          action=act,
                                          state_only=state)

            actions_agents = np.array(actions_agents)
            actions_to_env = actions_agents.copy()

            # Update coalition and counter
            for i, action in enumerate(actions_to_env):
                if action == 2:
                    coalition[i] = 1
                    counter[i] += 1
                    pick_mediator.append(1)
                else:
                    coalition[i] = 0
                    counter[i] = 0
                    pick_mediator.append(0)
            coalition = np.array(coalition)
            if ts > 0:
                trajectory.append((state_prev, actions_agents_prev, actions_mediator, coalition_prev, coalition_prev,
                                   rewards, next_state, coalition, done))

            actions_mediator = np.full(self.mediator.action_size, -1)

            # Mediator's move
            for i in range(self.n_agents):
                if actions_to_env[i] == 2:
                    agent_i = F.one_hot(torch.tensor(self.agents[i].agent_i), self.mediator.n_agents)
                    obs_med = self._tensorify([state, coalition, agent_i])
                    with torch.no_grad():
                        actions_mediator[i] = self.mediator.step(obs_med)
                    actions_to_env[i] = actions_mediator[i]

            next_state, rewards, done = env.step(*actions_to_env)
            ts += 1
            # trajectory.append((state, actions_agents, actions_mediator, coalition_agent, coalition.copy(),
            #                    rewards, next_state, done))
            coalition_prev = coalition.copy()

            for i, a in enumerate(self.agents):
                next_obs_ag = self._tensorify([next_state, coalition[i]])
                if 0 < counter[i] < self.steps_commit and done == 0:
                    trajectory_actors.add(agent_id=i,
                                          reward_under=rewards[i])
                elif counter[i] == self.steps_commit or (done == 1 and counter[i] > 0):
                    trajectory_actors.add(agent_id=i,
                                          reward_under=rewards[i])
                    rew_nstep = trajectory_actors.calculate_nstep(agent_id=i)
                    trajectory_actors.add(agent_id=i,
                                          next_state=next_obs_ag.squeeze(0),
                                          done=[done],
                                          reward=[rew_nstep],
                                          coalition_next=[coalition[i]])
                else:
                    trajectory_actors.add(agent_id=i,
                                          next_state=next_obs_ag.squeeze(0),
                                          done=[done],
                                          reward=[rewards[i]],
                                          coalition_next=[coalition[i]])

            for i, cnt in enumerate(counter):
                if cnt == self.steps_commit:  # steps_commit
                    coalition[i] = 2
                    counter[i] = 0

            state_prev = np.array(state).copy()
            state = next_state

            all_rewards.append(sum(rewards))
            # pick_mediator = np.sum(pick_mediator) / len(pick_mediator)


            actions_agents_prev = actions_agents

        trajectory.append((state_prev, actions_agents, actions_mediator, coalition_prev, coalition_prev,
                           rewards, next_state, coalition, done))

        if test:
            return all_rewards, pick_mediator

        return trajectory, trajectory_actors.get_trajectory()

    def evaluate_policy(self, n_episodes, env):
        all_rewards = []
        pick_mediator = []
        total_info = []

        for e in range(n_episodes):
            rewards, mediator = self.sample_episode(env, test=True)
            all_rewards.append(rewards)
            pick_mediator.append(mediator)

        info = env.log(self, all_rewards, pick_mediator, total_info)

        return info

    def train(self, env, log=False):
        eval_episodes = self.cfg.env.eval_episodes

        # get batch of trajectories
        for i in range(self.cfg.env.iterations):
            trajectories = []
            steps_ctn = 0
            traj_actor = ActorTraj(2).get_trajectory()
            while len(trajectories) < self.cfg.env.min_episodes_per_update \
                    or steps_ctn < self.cfg.env.min_transitions_per_update:
                traj, traj_a = self.sample_episode(env)
                steps_ctn += len(traj)
                trajectories.append(traj)

                for k, dct in traj_a.items():
                    for k_in, v_in in dct.items():
                        traj_actor[k][k_in].extend(v_in)

            # update
            self.update(trajectories, traj_actor)

            if log:
                if i % 100 == 0:
                    print(f'ITERATION {i}:')
                    # wandb.log(data={}, step=i, commit=False)
                    self.evaluate_policy(eval_episodes, env)
