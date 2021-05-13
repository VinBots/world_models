from src.vae.train import train_vae
from src.utils import preprocess
import gym
import pytest

@pytest.mark.skip()
def test_train_vae_net(test_config, vae_net, new_buffer):
    
    env = gym.make("CarRacing-v0")
    observation = env.reset()

    for _ in range(1000):
        #env.render()
        #env.step(env.action_space.sample()) 
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        new_buffer.add(observation)
        if done:
            observation = env.reset()
    env.close()

    test = train_vae (vae_net, new_buffer, preprocess, test_config)
    assert test == True

