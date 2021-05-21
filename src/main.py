import gym

from vae.model import VAE
from rnn import rnn
from controller import controller
from config import Configuration

# Choose the task
task = "CarRacing-v0"
env = gym.make(task)

# initialize env



def rollout(controller):
  '''
  env, rnn, vae are global variables  
  '''
  vae = (params=vae_config, device = device).to(device)


  obs = env.reset()
  h = rnn.initial_state()
  
  done = False
  cumulative_reward = 0

  while not done:
      z = vae.encode(obs)
      a = controller.action([z, h])
      obs, reward, done = env.step(a)
      cumulative_reward += reward
      h = rnn.forward([a, z, h])
  return cumulative_reward

if __name__ == "__main__":
  rollout(controller)