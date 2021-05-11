from vae import vae
from rnn import rnn
from controller import controller

# Choose the task
task = "car_racing"

# initialize env


def rollout(controller):
  '''
  env, rnn, vae are global variables  
  '''

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