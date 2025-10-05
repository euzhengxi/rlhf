import numpy as np
import torch

#minigrid
import minigrid

#gym
import gymnasium as gym
from gymnasium import RewardWrapper

from backend import query_feedback, query_evaluation

#amend according to env
ENV_NAME = "MiniGrid-RedBlueDoors-8x8-v0"
ACTIONDICT = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up object",
    4: "drop object",
    5: "toggle / activate object",
    6: "done"
}
NUM_ACTIONS = 7

DEVICE = torch.device("cpu")

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, mission):
        super().__init__(env)
        self.env = env
        self.device = DEVICE
        self.mission = mission

    def reward(self, reward):
        mapp, dir = self.summarize_state()
        mission = self.mission
        feedback = query_feedback(mission=mission, dir=dir, state=mapp, noise=30)
        evaluation = query_evaluation(state=mapp, mission=mission, feedback=feedback)
        reward = reward + 0.5 * evaluation

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

def create_env():
    base_env = gym.make(ENV_NAME, render_mode="rgb_array")
    #base_env = RGBImgPartialObsWrapper(base_env) 
    obs, _ = base_env.reset()
    env = CustomRewardWrapper(base_env, obs['mission'])
    return env