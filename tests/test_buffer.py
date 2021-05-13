import gym
from torchvision.transforms.functional import resize
import numpy as np
import torch
import pytest

@pytest.mark.skip()
def test_add_exp_in_buffer(new_buffer):
    for i in range(120):
        new_buffer.add(i)
    assert len(new_buffer.memory) == 100

@pytest.mark.skip()
def test_add_img_from_env(new_buffer):
    env = gym.make("CarRacing-v0")
    observation = env.reset()

    for _ in range(50):
        #env.render()
        #env.step(env.action_space.sample()) 
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        new_buffer.add(observation)
        if done:
            observation = env.reset()
    env.close()
    assert len(new_buffer.memory) == 50
    assert new_buffer.memory[42].shape == (96, 96, 3)

@pytest.mark.skip()
def test_shuffle_buffer(new_buffer):
    new_buffer.shuffle()
