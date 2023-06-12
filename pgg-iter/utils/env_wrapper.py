"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

import dm_env
import dmlab2d
from gym import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
import tree

PLAYER_STR_FORMAT = 'player_{index}'

def env_creator(env_config):
    env = substrate.build(config_dict.ConfigDict(env_config))
    env = env_wrapper.MeltingPotEnv(env)
    return env


def _timestep_to_observations(timestep: dm_env.TimeStep):
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if 'WORLD' not in key
        }
    return gym_observations


def _get_global_state(timestep: dm_env.TimeStep):
    return timestep.observation[0]['WORLD.RGB']


def _remove_world_observations_from_space(
        observation: spaces.Dict) -> spaces.Dict:
    return spaces.Dict({
        key: observation[key] for key in observation if 'WORLD' not in key})


def _spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """Converts a dm_env nested structure of specs to a Gym Space.
    BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
    Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.
    Args:
      spec: The nested structure of specs
    Returns:
      The Gym space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([_spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        return spaces.Dict({key: _spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError('Unexpected spec: {}'.format(spec))


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
    """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

    def __init__(self, env: dmlab2d.Environment):
        self._env = env
        self._num_players = len(self._env.observation_spec())

    def reset(self):
        """See base class."""
        timestep = self._env.reset()
        return _get_global_state(timestep), _timestep_to_observations(timestep)

    def step(self, action):
        """See base class."""
        actions = [
            action[PLAYER_STR_FORMAT.format(index=index)]
            for index in range(self._num_players)
        ]
        timestep = self._env.step(actions)
        # rewards = {
        #     PLAYER_STR_FORMAT.format(index=index): timestep.reward[index]
        #     for index in range(self._num_players)
        # }

        rewards = np.vstack(timestep.reward)
        done = True if timestep.last() else False
        info = {}

        observations = _timestep_to_observations(timestep)
        state_global = _get_global_state(timestep)

        return state_global, observations, rewards, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def single_player_observation_space(self) -> spaces.Space:
        """The observation space for a single player in this environment."""
        return _remove_world_observations_from_space(
            _spec_to_space(self._env.observation_spec()[0]))

    def single_player_action_space(self):
        """The action space for a single player in this environment."""
        return _spec_to_space(self._env.action_spec()[0])
