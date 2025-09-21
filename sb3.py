import gymnasium as gym
import minigrid
import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium import RewardWrapper

from backend import query_feedback, query_evaluation


DEVICE = torch.device("cpu")
class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.device = DEVICE

    def reward(self, reward):
        dir, mapp = self.summarize_state()
        mission = self.env.mission
        feedback = query_feedback(mission=mission, dir=dir, state=mapp, noise=30)
        evaluation = query_evaluation(state=mapp, mission=mission, feedback=feedback)
        reward = reward + evaluation

        return reward

    def summarize_state(self):
        env = self.env
        view_size = env.unwrapped.agent_view_size
        grid_map = np.full((view_size, view_size), '.', dtype=object)

        # Agent position and direction
        ax, ay = env.unwrapped.agent_pos
        dir_map = {0: "right", 1: "down", 2: "left", 3: "up"}
        agent_dir_str = dir_map[env.unwrapped.agent_dir]

        # Compute top-left of agent view
        top = max(0, ay - view_size // 2)
        left = max(0, ax - view_size // 2)

        for y in range(view_size):
            for x in range(view_size):
                gx, gy = left + x, top + y
                if gx >= env.unwrapped.width or gy >= env.unwrapped.height:
                    grid_map[y, x] = ' '  # out-of-bounds
                    continue

                cell = env.unwrapped.grid.get(gx, gy)
                if cell is None:
                    grid_map[y, x] = '.'
                elif cell.type == 'wall':
                    grid_map[y, x] = 'W'
                elif cell.type == 'door':
                    grid_map[y, x] = 'RD' if cell.color == 'red' else 'BD'
                elif cell.type == 'goal':
                    grid_map[y, x] = 'G'
                else:
                    grid_map[y, x] = f"{cell.color[0].upper()}{cell.type[0].upper()}"

        # Place agent in the center
        center = view_size // 2
        grid_map[center, center] = 'A'

        return grid_map, agent_dir_str

env = gym.make("MiniGrid-RedBlueDoors-8x8-v0")
env = CustomRewardWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)