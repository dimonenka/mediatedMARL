import torch

from src.base.nn.actor_critic_base import ActorBase, CriticBase


class ActorAgent(ActorBase):
    def __init__(self, input_size, state_size, action_size, hidden_state):
        super(ActorAgent, self).__init__(input_size, state_size, action_size, hidden_state)

    def forward(self, obs, off_policy=False):
        logits = self.act(obs)

        prob = torch.softmax(logits, -1)
        pi_dist = torch.distributions.Categorical(probs=prob)

        action = pi_dist.sample()

        return action, pi_dist


class CriticAgent(CriticBase):
    def __init__(self, input_size, state_size, action_size, hidden_state, n_agents=None):
        super(CriticAgent, self).__init__(input_size, state_size, action_size, hidden_state)

    def forward(self, obs):
        v = self.approx(obs)

        return v

