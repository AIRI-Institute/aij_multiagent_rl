import abc
from typing import Dict, Optional

import numpy as np


class BaseAgent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, ckpt_dir: str) -> None:
        """Agent Loading

        Loading an agent from the directory with artifacts.

        Args:
            ckpt_dir: path to an individual directory with artifacts
                and the `agent_config.yaml` file for this agent
        """
        pass

    @abc.abstractmethod
    def get_action(self, observation: Dict[str, np.ndarray]) -> int:
        """Getting action

        Getting action from the agent based on visual observation

        Args:
            observation: dictionary with the following keys and values
                "image": numpy array with an image of a local field of
                    view with dimensions (60, 60, 3) and np.uint8 data type
                "proprio": numpy array with proprioceptive information
                    about the position of the agent on the general map
                    and the condition of its inventory with dimensions (7,)
                    and np.float32 data type
        Returns:
            int: index of the selected action
                {0, 1, 2, 3, 4, 5, 6, 7, 8}
        """
        pass

    @abc.abstractmethod
    def reset_state(self) -> None:
        """Resetting the internal state

        In case of accumulation of internal context for decision-making
        during the episode, this method is called before each new
        simulation to clear the internal state before a new episode.
        If the internal context is not used, the method can be left in
        its current form.
        """
        pass


class RandomAgent(BaseAgent):
    """Random agent

    Random agent for AIJ Multi-agent AI Contest

    Attributes:
        action_dim: discrete action dimension, for this contest
            is always 9
        rng: numpy random number generator for reproducibility
    """
    def __init__(
        self,
        action_dim: int = 9,
        seed: Optional[int] = None
    ):
        """Initialise random agent

        Args:
            action_dim: discrete action dimension, for this contest
                is always 9
            seed: random number generator seed
        """
        self.action_dim = action_dim
        if seed is None:
            seed = np.random.randint(0, int(1e6), 1).item()
        self.rng = np.random.default_rng(seed)

    def load(self, ckpt_dir: str) -> None:
        """Agent Loading

        Loading an agent from the directory with artifacts.

        Args:
            ckpt_dir: path to an individual directory with artifacts
                and the `agent_config.yaml` file for this agent
        """
        pass

    def get_action(self, observation: Dict[str, np.ndarray]) -> int:
        """Getting action

        Getting action from random agent

        Args:
            observation: dictionary with the following keys and values
                "image": numpy array with an image of a local field of
                    view with dimensions (60, 60, 3) and np.uint8 data type
                "proprio": numpy array with proprioceptive information
                    about the position of the agent on the general map
                    and the condition of its inventory with dimensions (7,)
                    and np.float32 data type
        Returns:
            int: index of the selected action
                {0, 1, 2, 3, 4, 5, 6, 7, 8}
        """
        return self.rng.integers(0, self.action_dim, 1).item()

    def reset_state(self) -> None:
        """Resetting the internal state

        In case of accumulation of internal context for decision-making
        during the episode, this method is called before each new
        simulation to clear the internal state before a new episode.
        If the internal context is not used, the method can be left in
        its current form.
        """
        pass
