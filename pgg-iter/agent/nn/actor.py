import torch.nn as nn
import torch


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_state=64):
        super().__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.act = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size),
        )

    def pick_actions(self, logits, in_coalition, under_med):
        if under_med is True:
            idx_one = torch.where(in_coalition == 1.)[0]
            idx_minus_one = torch.where(in_coalition == -1.)[0]
            logits[idx_one, :-1] = float('-inf')
            logits[idx_minus_one, -1] = float('-inf')

        pi_dist = torch.distributions.Categorical(logits=logits)
        action = pi_dist.sample()

        return action, pi_dist

    # def pick_actions(self, logits, in_coalition, under_med):
    #     # idx_one = torch.where(in_coalition == 1.)[0]
    #     # idx_minus_one = torch.where(in_coalition == -1.)[0]
    #
    #     if in_coalition.dim() != 1:
    #         in_coalition = in_coalition.squeeze(-1)
    #
    #     mask = torch.zeros((len(logits), 3))  # n_actions
    #     mask[in_coalition == 1., :-1] = 1
    #     mask[in_coalition == -1., -1] = -1
    #
    #     logits = logits.masked_fill(mask == 1., -1e9)
    #     logits = logits.masked_fill(mask == -1., -1e9)
    #
    #     pi_dist = torch.distributions.Categorical(logits=logits)
    #     action = pi_dist.sample()
    #
    #     return action, pi_dist

    def forward(self, obs, under_med):
        obs, in_coalition = obs

        logits = self.act(obs)
        action, pi_dist = self.pick_actions(logits, in_coalition, under_med)

        return action, pi_dist
