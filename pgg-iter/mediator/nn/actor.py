import torch.nn as nn
import torch


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_state=64, n_agents=16):
        super().__init__()

        self.action_size = action_size
        self.state_size = state_size

        # coalition
        # self.coal_emb = nn.Sequential(
        #     nn.Linear(n_agents, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        # )

        # self.agent_emb = nn.Sequential(
        #     nn.Linear(n_agents, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 8),
        # )

        self.act = nn.Sequential(
            # nn.Linear(n_agents + 1 + 16 + 8, 16),
            nn.Linear(4 + n_agents + n_agents, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size),
        )

    def pick_actions(self, logits, under_med=False):
        pi_dist = torch.distributions.Categorical(logits=logits)
        action = pi_dist.sample()

        return action, pi_dist

    def forward(self, obs, under_med=False):
        obs, coalition, agent_idx = obs
        # coalition = self.coal_emb(coalition)
        # agent_idx = self.agent_emb(agent_idx)
        obs = torch.cat([obs, coalition, agent_idx], dim=-1)

        logits = self.act(obs)

        action, pi_dist = self.pick_actions(logits)

        return action, pi_dist
