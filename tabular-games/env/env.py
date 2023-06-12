import numpy as np
import torch
from torch.nn.functional import one_hot

from env.log import rpd_log, iter_log, pgg_log, pd_log


class Dilemma:
    def __init__(self):
        self.states = [
            [[(0, 0), (7, -5)], [(-5, 7), (2, 2)]]
        ]
        self.done = 0
        self.state = 0
        self.dummy_state = [1., 0.]
        
    def reset(self):
        self.done = 0
        return self.dummy_state
    
    def get_payoffs(self, action_1, action_2):
        return self.states[self.state][action_1][action_2]
        
    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.done = 1
        return self.dummy_state, reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        info = pd_log(controller, rewards, pick_mediator)
        # info = iter_log(controller, rewards, pick_mediator)
        return info


class RandomizedPrisonersDilemma(Dilemma):
    def __init__(self):
        super().__init__()
        self.states = [
            [[(0, 0), (7, -5)], [(-5, 7), (2, 2)]],
            [[(0, 0), (1, 1)], [(1, 1), (2, 2)]]
        ]

        self.state_onehot = [[1, 0], [0, 1]]

        self.probs = [.5, .5]
        self.done = 0

    def reset(self):
        self.done = 0
        return self.change_state()

    def change_state(self):
        self.state = np.random.choice(len(self.states), p=self.probs)
        return self.state_onehot[self.state]

    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.done = 1
        return self.state_onehot[self.state], reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        rpd_log(controller, rewards, pick_mediator)


class IterativePrisonersDilemma(RandomizedPrisonersDilemma):
    def __init__(self):
        super().__init__()
        self.states = [
            [[(0, 0), (7, -5)], [(-5, 7), (-1, 4)]],
            [[(0, 0), (7, -5)], [(-5, 7), (2, 2)]]
        ]

        self.state_onehot = [[1, 0, 0], [0, 1, 1]]
        self.state_onehot.append([0, 0, 2])

        self.probs = [1., 0.]
        self.n_rounds = 0
        self.max_rounds = 0

    def reset(self):
        self.done = 0
        self.n_rounds = 0
        self.max_rounds = 2
        self.state = 0

        return self.state_onehot[0]
        # return self.state

    def step(self, action_1, action_2):
        reward = self.get_payoffs(action_1, action_2)
        self.state = 1

        self.n_rounds += 1
        if self.n_rounds >= self.max_rounds:
            self.done = 1
        next_state = self.state_onehot[self.n_rounds]

        return next_state, reward, self.done

    def log(self, controller, rewards, pick_mediator, *args):
        info = iter_log(controller, rewards, pick_mediator)
        # info = pd_log(controller, rewards, pick_mediator)
        return info