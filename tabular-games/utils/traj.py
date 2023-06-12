from collections import defaultdict


class ActorTraj:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.trajectory = {f'agent_{i}': defaultdict(list) for i in range(n_agents)}

    def add(self, agent_id, **kwargs):
        for k, v in kwargs.items():
            self.trajectory[f'agent_{agent_id}'][k].append(v)

    def clear(self):
        self.__init__(self.n_agents)

    def get_trajectory(self):
        [self.trajectory[k].pop('reward_under', None) for k in self.trajectory.keys()]
        return self.trajectory

    def get_len_nstep(self, agent_id):
        return len(self.trajectory[f'agent_{agent_id}']['reward_under'])

    def calculate_nstep(self, agent_id, gamma=0.99):
        rewards = self.trajectory[f'agent_{agent_id}']['reward_under']
        rew_nstep = rewards[-1]
        for r in reversed(rewards[:-1]):
            rew_nstep = r + gamma * rew_nstep

        self.trajectory[f'agent_{agent_id}']['reward_under'] = []

        return rew_nstep
