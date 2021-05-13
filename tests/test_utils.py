import gym
from torchvision.transforms.functional import resize
import numpy as np
import torch
from src.utils import preprocess, create_gif, tensor_to_image
import pytest
import gym


@pytest.mark.skip()
def test_preprocess():
    img_list = [np.random.rand(96, 96, 3), np.random.rand(96, 96, 3)]
    new_img = preprocess (img_list[0], (64, 64))
    new_img_list = torch.stack(
        [preprocess(img, (64, 64)) for img in img_list])
    assert new_img_list.shape == (2, 3, 64, 64)
    

@pytest.mark.skip()
def test_preprocess_in_env():

    env = gym.make("CarRacing-v0")
    observation = env.reset()
    for _ in range(10):
        env.render()
        #env.step(env.action_space.sample()) 
        action = env.action_space.sample() # your agent here (this takes random actions)
        print (action)

        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()
    
    reshaped_observation = preprocess (np.copy(observation), (64, 64))
    assert reshaped_observation.shape == (3, 64, 64)

@pytest.mark.skip()
def test_create_gif(new_buffer):
    env = gym.make("CarRacing-v0")
    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        new_buffer.add(observation)
        if done:
            observation = env.reset()
    env.close()

    create_gif(new_buffer)

if __name__ == "__main__":
    test_preprocess()