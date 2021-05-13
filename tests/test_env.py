import gym
from torchvision.transforms.functional import resize
import numpy as np
import torch
import pytest

@pytest.mark.skip()
def test_env():

    env = gym.make("CarRacing-v0")
    observation = env.reset()
    for _ in range(10):
        #env.render()
        #env.step(env.action_space.sample()) 
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print (action)

        if done:
            observation = env.reset()
    env.close()
    print (observation.shape)

    

if __name__ == "__main__":
    test_env()