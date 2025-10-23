import numpy as np
import torch
import json
import ast

#minigrid
import minigrid

#gym
import gymnasium as gym
from gymnasium import RewardWrapper


ENV_NAME = "MiniGrid-RedBlueDoors-8x8-v0"
NUM_ACTIONS = 7 #amend according to env
#TOTAL_FRAMES = 50000
#FRAMES_PER_BATCH = 2048 #file size equivalent
#FRAMES_PER_SUBBATCH = 256 #mini_batch size, but you dont iterate through as much? 

DEVICE = torch.device("cpu")

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.device = DEVICE

    def load_cache(self):
        with open("cache.json") as f:
            cache = json.load(f)
        self.cache = {ast.literal_eval(key): value for key, value in cache.items()}

    def reward(self, reward):
        ax, ay = self.env.unwrapped.agent_pos
        direction = self.env.unwrapped.agent_dir
        reward = reward + self.cache[(ax, ay, direction)]

        return reward

def create_env():
    base_env = gym.make(ENV_NAME)
    obs, _ = base_env.reset()
    env = CustomRewardWrapper(base_env)
    return env, obs['mission']