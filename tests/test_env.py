import pytest
from pettingzoo import AECEnv
from pettingzoo.test import (max_cycles_test, parallel_api_test,
                             parallel_seed_test, performance_benchmark,
                             render_test)
from pettingzoo.test import test_save_obs as save_obs_test
from pettingzoo.utils import parallel_to_aec

from aij_multiagent_rl.env import AijMultiagentEnv


@pytest.fixture
def env_class():
    return AijMultiagentEnv


@pytest.fixture
def mixed_env(env_class):

    def aec_env(max_cycles=1000, render_mode='rgb_array') -> AECEnv:
        return parallel_to_aec(
            env_class(max_cycles=max_cycles, render_mode=render_mode))

    class TestEnv:
        def __init__(self):
            self.parallel_env = env_class
            self.env = aec_env
    return TestEnv()


def test_parallel_api(env_class):
    env = env_class()
    parallel_api_test(env, num_cycles=5000)


def test_parallel_seed(env_class):
    def env_fn():
        return env_class()
    parallel_seed_test(env_fn, num_cycles=5000)


def test_max_cycles(mixed_env):
    max_cycles_test(mixed_env)


def test_rendering(env_class):
    def aec_env(max_cycles=1000, render_mode='rgb_array') -> AECEnv:
        return parallel_to_aec(
            env_class(max_cycles=max_cycles, render_mode=render_mode))
    render_test(env_fn=aec_env)


def test_performance(env_class):
    def aec_env(max_cycles=1000, render_mode='rgb_array') -> AECEnv:
        return parallel_to_aec(
            env_class(max_cycles=max_cycles, render_mode=render_mode))
    performance_benchmark(aec_env(max_cycles=5000))


def test_saving_obs(env_class):
    def aec_env(max_cycles=1000, render_mode='rgb_array') -> AECEnv:
        return parallel_to_aec(
            env_class(max_cycles=max_cycles, render_mode=render_mode))
    save_obs_test(aec_env())
