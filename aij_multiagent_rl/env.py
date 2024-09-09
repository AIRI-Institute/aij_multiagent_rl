import functools
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from skimage.transform import resize

from aij_multiagent_rl.engine import GameEngine

TimestampType = Tuple[
    Dict[str, Dict[str, np.ndarray]], Dict[str, float],
    Dict[str, bool], Dict[str, bool], Dict[str, dict]
]

MOVES = ["FORWARD", "LEFT", "RIGHT", "BACKWARD",
         "PICKUP_RESOURCE", "PICKUP_TRASH",
         "DROP_RESOURCE", "DROP_TRASH", "NOOP"]


class AijMultiagentEnv(ParallelEnv):
    """Parallel multi-agent environment for AIJ
    Multi-Agent RL contest

    Environment at the testing system will have the same
    hyperparameter setting as below, so it is not recommended
    to change it

    Attributes:
        grid_size: 2D square game field size
        obs_dim: local visual observation size
        move_step: movement step size in pixels
        resource_price: reward given for resource processing
        recycle_cost: cost of recycling trash
        border_distort_range: noise range for borders distortion
        max_edge_dev: maximum segment border shift
        max_tries: maximum attempts for border generation
        machine_size: machine icon size in pixels
        machine_reach: size of machine interaction region
        agent_size: agent icon size in pixels
        agent_reach: agent reach when picking up items
        resource_size: resource icon size in pixels
        trash_size: trash icon size in pixels
        resource_prob: probability to spawn resource at a given point
        border_display_width: segments borders thickness
        ecology_penalty: decrease in agent's ecology score caused by 1 trash item
        neighbour_ecology_weight: neighbour ecology effect at the resource
            respawn rate
        global_ecology_weight: global ecology effect at the resource
            respawn rate
        init_respawn_prob: initial probability to spawn resource at a given point
        blocked_vanish_alpha: blocked segment fogging degree
        max_dead_segments: max number of blocked segments before global
            termination
    """
    # Hardcoded hyperparameters
    grid_size: int = 210  # should be div by resource_size and trash_size
    obs_dim: int = 60  # should be divisible by 5
    move_step: int = 7
    resource_price: int = 10
    recycle_cost: int = 4
    border_distort_range: Tuple[int, int] = (-1, 2)
    max_edge_dev: float = 0.1
    max_tries: int = 25
    machine_size: int = 9
    machine_reach: int = 9
    agent_size: int = 9
    agent_reach: int = 9
    resource_size: int = 7
    trash_size: int = 5
    resource_prob: float = 0.075
    border_display_width: int = 2
    ecology_penalty: int = 20
    neighbour_ecology_weight: float = 0.2
    global_ecology_weight: float = 0.3
    init_respawn_prob: float = 0.015
    blocked_vanish_alpha: float = 0.25
    max_dead_segments: int = 4

    # To enable rendering
    metadata = {"render_modes": ['rgb_array'],
                "name": "aij_multiagent_env"}

    def __init__(
        self,
        max_cycles: Optional[int] = 1000,
        state_size: Optional[int] = 110,
        render_mode: Optional[str] = 'rgb_array',
    ):
        """Multi-agent RL Environment

        Multi-agent RL Environment for AIJ Contest 2024

        Args:
            max_cycles: maximum simulation length in time steps
            state_size: display state size for rendering
            render_mode: render mode

        Valid Action Space:
            0: move forward by `move_step` pixels if possible
            1: move left by `move_step` pixels if possible
            2: move right by `move_step` pixels if possible
            3: move backward by `move_step` pixels if possible
            4: pickup resource (if closer than `agent_reach` pixels)
            5: pickup trash (if closer than `agent_reach` pixels)
            6: throw resource (put into machine if closer than `machine_reach`)
            7: throw trash (put into recycler if closer than `machine_reach`)
            8: noop
        """
        self.engine = None
        self.rng = None
        self.ecology_scores = None
        self.num_moves = None
        self.current_state = None
        self.seed = None
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.state_size = state_size
        self.action_meanings = MOVES
        self.possible_agents = [f'agent_{i}' for i in range(8)]
        self.neighbours_mapping = {
            'agent_0': ['agent_1', 'agent_3'],
            'agent_1': ['agent_0', 'agent_2'],
            'agent_2': ['agent_1', 'agent_4'],
            'agent_3': ['agent_0', 'agent_5'],
            'agent_4': ['agent_2', 'agent_7'],
            'agent_5': ['agent_3', 'agent_6'],
            'agent_6': ['agent_5', 'agent_7'],
            'agent_7': ['agent_4', 'agent_6'],
        }
        self.agents = self.possible_agents.copy()
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    @classmethod
    def _get_engine(
        cls, seed: Optional[int] = None
    ) -> Tuple[GameEngine, np.random.Generator, int]:
        """Get engine

        Get game engine for simulation round

        Args:
            seed: random seed for simulation

        Returns:
            Tuple[GameEngine, np.random.Generator, int]: tuple of:
                - GameEngine instance
                - numpy random number generator
                - seed for logging
        """
        if seed is None:
            seed = np.random.randint(0, int(1e6), 1).item()
        engine = GameEngine(
            seed=seed,
            grid_size=cls.grid_size,
            obs_dim=cls.obs_dim,
            move_step=cls.move_step,
            resource_price=cls.resource_price,
            recycle_cost=cls.recycle_cost,
            border_distort_range=cls.border_distort_range,
            max_edge_dev=cls.max_edge_dev,
            max_tries=cls.max_tries,
            machine_size=cls.machine_size,
            machine_reach=cls.machine_reach,
            agent_size=cls.agent_size,
            agent_reach=cls.agent_reach,
            resource_size=cls.resource_size,
            trash_size=cls.trash_size,
            resource_prob=cls.resource_prob,
            border_display_width=cls.border_display_width,
            blocked_vanish_alpha=cls.blocked_vanish_alpha
        )
        rng = np.random.default_rng(seed + 1)
        return engine, rng, seed

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> DictSpace:
        """Get observation space

        Get environment observation space

        Args:
            agent: agent to return observation space for

        Returns:
            DictSpace: specification of composite observation
        """
        d = self.engine.obs_dim
        gs, diag = self.engine.grid_size, self.engine.diag
        image_space = Box(0, 255, (d, d, 3), np.uint8, seed=self.rng)
        low = np.array([0., 0., 0., 0., -1., -1., 0.])
        high = np.array([np.inf, 1., 1., diag / gs, 1., 1., 1.])
        proprio_space = Box(
            low=low, high=high, shape=(7,), dtype=np.float32, seed=self.rng)
        return DictSpace({'image': image_space, 'proprio': proprio_space})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        """Get action space

        Get environment action space

        Args:
            agent: agent to return action space for

        Returns:
            Discrete: specification of discrete action space
        """
        return Discrete(len(self.action_meanings), seed=self.rng)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Any] = None
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict]]:
        """Reset environment

        Reset environment to its initial state

        Args:
            seed: numpy random seed
            options: any other additional options

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict]]:
                tuple which contains:
                    - composite observations for each agent
                    - infos for each agent
        """
        self.num_moves = 0
        self.agents = self.possible_agents.copy()
        self.engine, self.rng, self.seed = self._get_engine(seed=seed)
        self.ecology_scores = {a: 100 for a in self.agents}
        # Render next observations
        state = self.engine.get_state()
        self.current_state = state
        # Local perspective state
        local_agents_map = self.engine.agents_map(local_mode=True)
        local_state = state * np.logical_not(
            local_agents_map > 0)
        local_state = local_state + local_agents_map
        # Render local observations
        state_pad = self.engine.pad_state(local_state)
        obs = {a: self.engine.local_obs(
            state_pad=state_pad, agent_id=a) for a in self.agents}
        # Log information
        infos = {a: {
            'ecology_score': self.ecology_scores[a],
            'num_trash': self.engine.trash_by_segment(agent_id=a),
            'num_resource': self.engine.resource_by_segment(agent_id=a),
            'dead_ecology': self.engine.blocked[a]
        } for a in self.agents}
        return obs, infos

    def _make_action(self, agent_id: str, action_id: int) -> None:
        """Make action

        Execute action in the environment for a given agent

        Args:
             agent_id: agent ID in form `agent_{i}`
             action_id: action integer id from valid actions set

        Returns:
            None

        Raises:
            ValueError: if action ID is outside valid action set
        """
        if 4 > action_id >= 0:
            self.engine.move(agent_id=agent_id, action_id=action_id)
        elif action_id == 4:
            self.engine.pickup(agent_id=agent_id, type='resource')
        elif action_id == 5:
            self.engine.pickup(agent_id=agent_id, type='trash')
        elif action_id == 6:
            self.engine.drop_resource(agent_id=agent_id)
        elif action_id == 7:
            self.engine.drop_trash(agent_id=agent_id)
        elif action_id == 8:
            pass
        else:
            raise ValueError(
                f'Invalid action: {action_id} for agent: {agent_id}')

    def _update_ecology_scores(self) -> Dict[str, int]:
        """Update ecology scores

        Update ecology scores given the most recent trash
        distribution information

        Returns:
            Dict[str, int]: current number of trash per segment
        """
        new_ec_scores = {}
        trash_by_agent = {}
        for a, s in self.ecology_scores.items():
            n_trash = self.engine.trash_by_segment(agent_id=a)
            trash_by_agent[a] = n_trash
            ec_score = max(0, 100 - n_trash * self.ecology_penalty)
            new_ec_scores[a] = ec_score * int(not self.engine.blocked[a])
        self.ecology_scores = new_ec_scores
        return trash_by_agent

    def _get_resource_respawn_probs(self) -> Dict[str, float]:
        """Get resource respawn probs

        Get resource respawn probability given current trash
        distribution

        Returns:
            Dict[str, int]: respawn probs by agent ID
        """
        resp_probs = {}
        gs = np.mean(list(self.ecology_scores.values())).item()
        for a, s in self.ecology_scores.items():
            n1, n2 = self.neighbours_mapping[a]
            ns1, ns2 = self.ecology_scores[n1], self.ecology_scores[n2]
            mean_ns = (ns1 + ns2) / 2
            nw, p = self.neighbour_ecology_weight, self.init_respawn_prob
            gw = self.global_ecology_weight
            lw = max(0., 1 - nw - gw)
            r = ((lw * s + gw * gs + nw * mean_ns) / 100)
            resp_probs[a] = (r ** 2.15) * p * int(not self.engine.blocked[a])
        return resp_probs

    def _get_terminations(self) -> Dict[str, bool]:
        """Get terminations

        Get global termination and impose blocks according to
        local ecology scores

        Returns:
            Dict[str, bool]: global termination by agent ID
                (all True or all False)
        """
        global_termination = False
        for a, s in self.ecology_scores.items():
            terminated = s == 0
            if terminated:
                self.engine.add_block(agent_id=a)
        n_blocked = sum(list(self.engine.blocked.values()))
        if n_blocked > self.max_dead_segments:
            global_termination = True
        return {a: global_termination for a in self.possible_agents}

    def state(self) -> Dict[str, np.ndarray]:
        """Get global state

        Get global state for CTDE multi-agent RL paradigm.
        Note! That method won't be called at testing system
        and may be used only for training agents.

        Returns:
            Dict[str, np.ndarray]: global state with following key-value
                pairs:
                    - 'image': global visual state, shape:
                        (self.state_size, self.state_size, 3)
                    - 'wealth': array with agents wealth, shape: (8,)
                    - 'has_resource': array with binary flag, indicating that
                        resource is in inventory, shape: (8,)
                    - 'has_resource': array with binary flag, indicating that
                        trash is in inventory, shape: (8,)
        """
        state = {}
        image = self.current_state.copy()
        image = resize(image, (self.state_size, self.state_size))
        state['image'] = np.round(image * 255, 0).astype(np.uint8)
        state['wealth'] = np.array(
            [self.engine.agents_state[a]['wealth']
             for a in self.possible_agents])
        state['has_resource'] = np.array(
            [self.engine.agents_state[a]['inventory']['resource']
             for a in self.possible_agents]).astype(int)
        state['has_trash'] = np.array(
            [self.engine.agents_state[a]['inventory']['trash']
             for a in self.possible_agents]).astype(int)
        return state

    def render(self) -> np.ndarray:
        """Render global state

        Render global state as numpy image array

        Returns:
            np.ndarray: global visual state, shape:
                (self.state_size, self.state_size, 3)
        """
        return self.state()['image']

    def close(self) -> None:
        """Close all rendering windows"""
        pass

    def step(self, actions: Dict[str, int]) -> TimestampType:
        """Perform simulation step

        Perform simulation step in parallel multi-agent RL style.
        In case of conflicting actions (for example two agents
        picking up the same resource, priorities are assigned
        randomly). Note: both terminations and truncations occur
        simultaneously for all agents participating in the
        simulation.

        Args:
            actions: dictionary with action IDs for each agent
        Returns:
            TimestampType: environment time stamp as a tuple of:
                - Dict[str, Dict[str, np.ndarray]]: composite observations
                    for each agent
                - Dict[str, float]: rewards for each agent
                - Dict[str, bool]: terminations for each agent
                - Dict[str, bool]: truncations for each agent
                - Dict[str, dict]: infos for each agent"""
        self.num_moves += 1
        # Cache money to calculate rewards
        old_w = {a: self.engine.agents_state[a]['wealth'] for a in self.agents}
        # Apply actions in random order (to reconcile possible conflicts)
        agents = self.agents.copy()
        self.rng.shuffle(agents)
        for agent in agents:
            self._make_action(agent_id=agent, action_id=actions[agent])
        # Get rewards from updated wealth
        rewards = {a: self.engine.agents_state[a]['wealth'] - old_w[a]
                   for a in self.agents}
        # Apply rules to update ecology scores
        trash_by_agent = self._update_ecology_scores()
        # Terminate or truncate for those neeeded
        terminations = self._get_terminations()
        truncation = self.num_moves >= self.max_cycles
        truncations = {a: truncation for a in self.agents}
        # Respawn resources according to respawn probabilities
        for a, p in self._get_resource_respawn_probs().items():
            u = self.rng.uniform(low=0, high=1)
            if u < p:
                self.engine.sample_resource(agent_id=a)
        # Render next observations
        state = self.engine.get_state()
        self.current_state = state
        # Local perspective state
        local_agents_map = self.engine.agents_map(local_mode=True)
        local_state = state * np.logical_not(
            local_agents_map > 0)
        local_state = local_state + local_agents_map
        # Render local observations
        state_pad = self.engine.pad_state(local_state)
        observations = {a: self.engine.local_obs(
            state_pad=state_pad, agent_id=a) for a in self.agents}
        # Log information
        infos = {a: {
            'ecology_score': self.ecology_scores[a],
            'num_trash': trash_by_agent[a],
            'num_resource': self.engine.resource_by_segment(agent_id=a),
            'dead_ecology': self.engine.blocked[a]
        } for a in self.agents}
        # Delete truncated and terminated Agents
        if truncation or any(terminations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos
