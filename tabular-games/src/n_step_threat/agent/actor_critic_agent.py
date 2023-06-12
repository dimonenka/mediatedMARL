import torch

from src.base.nn.actor_critic_base import ActorBase, CriticBase


class ActorAgent(ActorBase):
    def __init__(self, input_size, state_size, action_size, hidden_state):
        super(ActorAgent, self).__init__(input_size, state_size, action_size, hidden_state)

    def forward(self, obs, off_policy=True):
        obs, in_coalition = obs[:, :-1], obs[:, -1]
        logits = self.act(obs)

        if off_policy:
            idx_zero = torch.where(in_coalition == 0.)[0]
            idx_one = torch.where(in_coalition == 1.)[0]

            logits[idx_zero, -1] = float('-inf')
            logits[idx_one, :-1] = float('-inf')

        prob = torch.softmax(logits, -1)
        pi_dist = torch.distributions.Categorical(probs=prob)

        action = pi_dist.sample()

        return action, pi_dist


class CriticAgent(CriticBase):
    def __init__(self, input_size, state_size, action_size, hidden_state, n_agents=None):
        super(CriticAgent, self).__init__(input_size, state_size, action_size, hidden_state)

    def forward(self, obs):
        obs, in_coalition = obs[:, :-1], obs[:, -1]
        v = self.approx(obs)

        return v

