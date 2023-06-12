import torch.nn as nn


class ActorBase(nn.Module):
    def __init__(self, input_size, state_size, action_size, hidden_state):
        super().__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.act = nn.Sequential(
            nn.Linear(input_size, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, self.action_size)
        )

    def forward(self, obs, off_policy=False):
        raise NotImplementedError


class CriticBase(nn.Module):
    def __init__(self, input_size, state_size, action_size, hidden_state, mediator_n=None):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size

        if mediator_n is not None:
            out_neurons = mediator_n
        else:
            out_neurons = 1

        self.approx = nn.Sequential(
            nn.Linear(input_size, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, out_neurons)
        )

    def forward(self, obs):  # давать номер агента и действие оппонента
        raise NotImplementedError
