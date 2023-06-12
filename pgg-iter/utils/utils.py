import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


class AgentTraj:
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

    def get_reward_under(self, agent_id):
        return self.trajectory[f'agent_{agent_id}']['reward_under']

    def calculate_nstep(self, agent_id, gamma=0.99):
        rewards = self.get_reward_under(agent_id)
        rew_nstep = rewards[-1]
        for r in reversed(rewards[:-1]):
            rew_nstep = r + gamma * rew_nstep

        self.trajectory[f'agent_{agent_id}']['reward_under'] = []

        return rew_nstep


class MediatorTraj(AgentTraj):
    def __init__(self, n_agents=16):
        super(MediatorTraj, self).__init__(n_agents)
        self.trajectory['agent_999'] = defaultdict(list)

    def calculate_nstep(self, agent_id, gamma=0.99):
        rewards = self.get_reward_under(agent_id)
        rew_nstep = rewards[-1]
        rew_history = [rew_nstep]
        for r in reversed(rewards[:-1]):
            rew_nstep = r + gamma * rew_nstep
            rew_history.append(rew_nstep)

        self.trajectory[f'agent_{agent_id}']['reward_under'] = []

        return rew_history[::-1]  # reverse list

    def add_many(self, agent_id, **kwargs):
        for k, v in kwargs.items():
            self.trajectory[f'agent_{agent_id}'][k].extend(v)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-7
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]

    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def process_webm(state, who_where):
    for i, w in enumerate(who_where):
        l = (w[1]) * 8
        d = (w[0]) * 8

        state[l-1:l + 10, d-1:d+1, :] = [255, 255, 0]
        state[l - 1:l + 10, d + 10 - 1:d + 10 + 1, :] = [255, 255, 0]
        state[l - 1:l + 1, d - 2:d + 9, :] = [255, 255, 0]
        state[l + 9 - 1:l + 9 + 1, d - 2:d + 9, :] = [255, 255, 0]

    # plt.imshow(state)
    # plt.show()
    return state


def check_consistency(dct):
    for agent_key in dct.keys():
        for i, (k, v) in enumerate(dct[agent_key].items()):
            if i == 0:
                ref_key = k
                ref_len = len(v)
            assert len(v) == ref_len, f'len of {k} ({len(v)}) does not equal {ref_key} ({ref_len}) ' \
                                      f'for {agent_key}'


class SchedulerBC:
    def __init__(self, const_coef=0.1, off_after=1500, decrease_after=300, decrease_speed=0.001):
        self.t = 0
        self.exp_f = lambda x: 1 - np.exp(-0.02 * x) * 1.5

        self.coef = 0
        self.const_coef = const_coef
        self.off_after = off_after
        self.decrease_after = decrease_after
        self.decrease_speed = decrease_speed

    def step(self):
        if self.t > self.off_after:
            return 0
        if self.t > self.decrease_after:
            self.coef = np.maximum(self.const_coef, self.coef - self.decrease_speed)
            return self.coef

        self.coef = np.maximum(0, self.exp_f(self.t))
        self.t += 1
        return self.coef
