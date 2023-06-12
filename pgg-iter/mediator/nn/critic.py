import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_state, n_agents=16):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size

        # self.coal_emb = nn.Sequential(
        #     nn.Linear(n_agents, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        # )

        self.approx = nn.Sequential(
            nn.Linear(n_agents + n_agents + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_agents),
        )

    def forward(self, obs):
        # local_rest = torch.hstack([local_rest, coalition, idx, timestep]).to(self.inference_device)
        obs, coalition = obs
        # coalition = self.coal_emb(coalition)

        out = torch.cat([obs, coalition], dim=-1)
        v = self.approx(out)

        return v
