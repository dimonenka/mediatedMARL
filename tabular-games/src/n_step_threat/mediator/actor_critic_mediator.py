import torch

from src.base.nn.actor_critic_base import ActorBase, CriticBase


class ActorMediator(ActorBase):
    def __init__(self, input_size, state_size, action_size, hidden_state):
        super(ActorMediator, self).__init__(input_size, state_size, action_size, hidden_state)

    def forward(self, obs):
        logits = self.act(obs)
        probs = torch.softmax(logits, -1)

        pi_dist = torch.distributions.Categorical(probs=probs)
        action = pi_dist.sample()

        return action, pi_dist


class CriticMediator(CriticBase):
    def __init__(self, input_size, state_size, action_size, hidden_state, n_agents=None):
        super(CriticMediator, self).__init__(input_size, state_size, action_size, hidden_state, mediator_n=n_agents)

    def forward(self, obs):
        v = self.approx(obs)

        return v
