# Nestor Arana Arexolaleiba
# Antonio Serrano
# Mondragon University
# 2022/12/8
import gym

# load envrironment
env = gym.make("CartPole-v1", render_mode="human")
# wrap the environment
from skrl.envs.torch import wrap_env
env = wrap_env(env, wrapper ="gym")

import torch
import torch.nn as nn
import torch.nn.functional as F

"""# Define computing device"""
device = torch.device("cpu") 
env.device = device
print(device)

"""# Neural Networks"""
# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin

# Instantiate the agent's models (function approximators) using the model instantiator utility
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#spaces-and-models

class DeterministicNetwork (DeterministicMixin, Model):
  def __init__(self, observation_space, action_space, device, clip_actions = False):
    Model.__init__(self, observation_space, action_space, device)
    DeterministicMixin.__init__(self, clip_actions)

    self.linear_layer_1 = # include layer 1 
    self.linear_layer_2 = # include layer 2 with 64 neuros 
    self.action_layer = nn.Linear(64, self.num_actions)

  def compute (self, states):
      x = F.relu (self.linear_layer_1(states))
      x = # include a layer
      return self.action_layer (x)

"""# Agent"""
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#spaces-and-models
models_dqn = {}
models_dqn ["q_network"] = DeterministicNetwork(env.observation_space, env.action_space, device)
models_dqn ["target_q_network"] = DeterministicNetwork(env.observation_space, env.action_space, device)

"""# Memory"""
from skrl.memories.torch import RandomMemory

# include a memory of at least 2000

"""# Agent Configuration"""
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#configuration-and-hyperparameters
cfg_dqn = DQN_DEFAULT_CONFIG.copy()
cfg_dqn["learning_starts"] = 100
cfg_dqn["exploration"]["final_epsilon"] = 0.04
cfg_dqn["learning_rate"] = 1e-3
cfg_dqn["discount_factor"] = 0.99
cfg_dqn["batch_size"] = 64
# logging to TensorBoard and write checkpoints
cfg_dqn["experiment"]["directory"] = "../tensorboard_session"
cfg_dqn["experiment"]["write_interval"] = 300
cfg_dqn["experiment"]["checkpoint_interval"] = 1500

"""# Agent instantiation"""
agent_dqn = DQN(models=models_dqn, 
                  memory=memory, 
                  cfg=cfg_dqn, 
                  observation_space= env.observation_space, 
                  action_space=env.action_space, 
                  device=device)

"""# Trainning"""
from skrl.trainers.torch import SequentialTrainer
cfg_trainer = {"timesteps" : 2000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, 
                            env=env, 
                            agents=agent_dqn)

# Train the agent
