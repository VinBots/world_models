import numpy as np
import torch
from src.vae.train import train_vae
from src.utils import preprocess, create_gif, tensor_to_image
import gym
import pytest
from src.vae.buffer import Buffer

#@pytest.mark.skip()
def test_reconstruct_img(test_config, vae_net, new_buffer, device):
    
    env = gym.make("CarRacing-v0")
    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        new_buffer.add(observation)
        if done:
            observation = env.reset()
    env.close()
    create_gif (new_buffer)

    test = train_vae (vae_net, new_buffer, preprocess, test_config, device)
    
    vae_net.save_weights()

    reconstruct_buffer = Buffer("reconstruct", 96)
    img_list = list(new_buffer.memory)

    x = torch.stack(
        [preprocess(np.copy(img), test_config.resize) for img in img_list]).to(device)
    
    x_hat, _, _ = vae_net.forward(x)
    
    for new_img in torch.unbind(x_hat, dim = 0):
        convert_img = tensor_to_image(new_img.cpu().detach().numpy())
        reconstruct_buffer.add (convert_img)
    
    create_gif (reconstruct_buffer)
    assert reconstruct_buffer.memory[14].shape == (64, 64, 3)
    assert len(reconstruct_buffer.memory) == 96
    