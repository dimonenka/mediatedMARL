import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_state, mediator_n=None):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size

        self.approx = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, obs):
        # rest = [rts, pos, orient, timestep]
        v = self.approx(obs)

        return v
