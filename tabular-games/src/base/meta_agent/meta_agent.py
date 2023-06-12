import torch


class MetaAgent:
    def __init__(self, input_sizes, agent_nn, cfg_agent, cfg, agent_i=-1):
        """
        Class for building agents or mediator

        :param input_size: iterable(n_actor, n_critic), number of input features to NNs
        :param agent_nn: iterable, Actor and Critic NN
        :param cfg_agent: config file for agents/mediator
        :param cfg_env: config file for environment
        :param agent_i: int, "-1" if mediator, positive integers otherwise
        """
        Actor, Critic = agent_nn
        input_size_actor, input_size_critic = input_sizes
        optimizer = torch.optim.Adam

        self.lr_a = cfg_agent.lr_a
        self.lr_c = cfg_agent.lr_c
        self.gamma = cfg_agent.gamma
        self.entropy_coef = cfg_agent.entropy_coef
        self.hidden_state = cfg_agent.n_hidden
        self.state_size = cfg.env.state_size
        self.action_size = cfg_agent.action_size
        self.agent_i = agent_i
        self.device = cfg.device
        self.dtype = torch.float32
        self.cfg = cfg_agent

        self.mediator_n = None if agent_i != -1 else cfg.env.n_agents
        self.entropy_coef_decrease = cfg_agent.entropy_decrease

        self.actor = Actor(input_size_actor, self.state_size, self.action_size, cfg_agent.n_hidden)
        self.critic = Critic(input_size_critic, self.state_size, self.action_size, cfg_agent.n_hidden, n_agents=self.mediator_n)

        self.opt_actor = optimizer(self.actor.parameters(), lr=cfg_agent.lr_a)
        self.opt_critic = optimizer(self.critic.parameters(), lr=cfg_agent.lr_c)

    def _tensorify(self, list_of_non_tensors):
        tensors = []
        for item in list_of_non_tensors:
            converted = torch.tensor(item, device=self.device, dtype=self.dtype)

            if converted.dim() == 0:
                converted = converted.reshape(-1, 1)
            elif converted.dim() == 1:
                converted = converted.unsqueeze(0)

            tensors.append(converted)

        return tensors

    def _calc_advantage(self, obs, next_obs, reward, done):
        """
        Calculating advantage

        :param obs: iterable, of everything connected to observation (i.e. in_coalition)
        :param next_obs: iterable, same as previous converning next_state
        :param reward: real value
        :param done: Boolean
        :return: advantage
        """
        advantage = reward + (1 - done) * self.gamma * self.critic(next_obs) - self.critic(obs)
        assert advantage.dim() == 2, f'{advantage.shape=}'

        return advantage

    def get_policy(self, obs, off_policy=True):
        _, pi_dist = self.actor(obs, off_policy)

        return pi_dist

    def step(self, obs):
        action, _ = self.actor(obs)

        return action.squeeze(0).numpy()

    def actor_loss(self, log_probs, baseline):
        actor_loss = -1 * (log_probs * baseline.detach())
        assert actor_loss.shape == (baseline.shape[0], 1), f'{actor_loss.shape=}'

        if self.agent_i == -1:
            return actor_loss

        return actor_loss.mean()

    def critic_loss(self, state, next_state, reward, done):
        baseline = self._calc_advantage(state, next_state, reward, done)

        if self.agent_i == -1:
            assert baseline.shape == (reward.shape[0], self.mediator_n)
            critic_loss = baseline.pow(2).sum(-1).mean()
        else:
            assert baseline.shape == (reward.shape[0], 1)
            critic_loss = baseline.pow(2).mean()

        return critic_loss, baseline

    def update_agent(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def update_mediator(self, state, action, reward, next_state, coalition, done):
        raise NotImplementedError
