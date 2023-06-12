import numpy as np


class IterativePGG:
    def __init__(self, cfg):
        self.n_agents = cfg.n_agents
        self.endowment = cfg.endowment
        self.max_steps = cfg.max_steps
        self.multiplier = cfg.multiplier
        self.contrib_proportion = cfg.contrib_proportion

        self.balance = np.array([self.endowment] * self.n_agents).reshape(1, -1).astype(np.float64)
        # self.balance = np.random.uniform(0, self.endowment, size=self.n_agents).reshape(1, -1)
        self.timestep = 0
        self.done = False

    def reset(self):
        self.balance = np.array([self.endowment] * self.n_agents).reshape(1, -1).astype(np.float64)
        # self.balance = np.random.uniform(0, self.endowment, size=self.n_agents).reshape(1, -1)
        self.timestep = 0
        self.done = False

        global_state = np.concatenate([self.balance, [[self.timestep]]], axis=-1)

        obs = {
            f'player_{i}': {'obs': self.balance[:, i], 'ts': self.timestep}
            for i in range(self.n_agents)
        }

        return global_state, obs, self.done

    def step(self, action_dict):
        actions = np.array([action_dict[f'player_{i}'] for i in range(self.n_agents)])

        if np.any(actions > 2):
            raise ValueError('There is no such an action!')

        who_contrib = np.where(actions == 1, self.contrib_proportion, 0)
        income = self.multiplier * np.sum(self.balance * who_contrib) / self.n_agents
        new_balance = self.balance - (self.balance * who_contrib) + income
        reward = (new_balance - self.balance).squeeze(0)
        self.balance = new_balance
        self.timestep += 1

        if self.timestep == self.max_steps:
            self.done = True

        obs = {
            f'player_{i}': {'obs': self.balance[:, i], 'ts': self.timestep}
            for i in range(self.n_agents)
        }

        global_state = np.concatenate([self.balance, [[self.timestep]]], axis=-1)

        return global_state, obs, reward, self.done, None


# balance = [1, 1], act=[0, 1], surp=sum([0, 0.5])/2=0.25, balance_=[1.25, 0.75], rew=[0.25, -0.25]
# balance = [0.5, 1, 2], act=[1, 1, 0], surp = sum([0.25, 0.5, 0]) / 3 = 0.25, balance_=[0.5, 0.75, 2.25], rew=[0, -0.25, 0.25]
# balance = [0.5, 1, 2], act=[1, 1, 0], surp = sum([0.25, 0.5, 0]) / 3 * 2 = 0.5, balance_=[0.75, 1, 2.5], rew=[0.25, 0, .5]
