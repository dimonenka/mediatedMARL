from abc import ABC

import numpy as np

from src.base.meta_agent.meta_agent import MetaAgent


class Agent(MetaAgent, ABC):
    def __init__(self, input_sizes, agent_nn, cfg_agent, cfg, agent_i):
        super(Agent, self).__init__(input_sizes, agent_nn, cfg_agent=cfg_agent, cfg=cfg, agent_i=agent_i)

    def update_agent(self, state, action, reward, next_state, done):
        agent_critic_loss, advantage = self.critic_loss(state, next_state,
                                                        reward,
                                                        done)

        policy = self.get_policy(state)
        log_prob = policy.log_prob(action).unsqueeze(-1)
        entropy = policy.entropy().mean()
        agent_actor_loss = self.actor_loss(log_prob, advantage) - self.entropy_coef * entropy

        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        agent_critic_loss.backward()
        agent_actor_loss.backward()
        self.opt_actor.step()
        self.opt_critic.step()

        self.entropy_coef = np.maximum(0.1, self.entropy_coef - self.entropy_coef_decrease)
