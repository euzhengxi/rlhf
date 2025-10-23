import json
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI, RateLimitError 
import os
import time

DMAP = {0: "right", 1: "down", 2: "left", 3: "up"}

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API")
client = OpenAI(api_key=OPENAI_API)

def ask_with_retry(messages, model="gpt-4o-mini", max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response
        except RateLimitError as e:
            wait_time = 2 ** attempt  # exponential backoff: 1, 2, 4, 8, 16 seconds
            print(f"Rate limit hit, retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Failed after multiple retries due to rate limits.")

#feedback on current state
def query_feedback(mission:str, dir: str, state: list, noise: int) -> str:
    prompt = f'You are providing feedback to help an agent complete the mission: {mission}. The agent is facing {dir} and sees: \n {state}. \
\nProvide feedback to help the agent. There is a probability {noise} that you add conflicting and random information into the feedback. \
The final feedback should at most be 40 words. An example of a conflicting feedback: go up, maybe dont go up, maybe left?'

    message = [{"role": "user", "content": prompt}]
    response = ask_with_retry(messages=message)

    return response.choices[0].message.content

#evaluate current feedback using LLM
def query_evaluation(state:list, mission:str, feedback: str) -> float:
    prompt = f'Evaluate the following feedback: \n {feedback} \
\nUse these criteria: \
\n1. Clarity (1 - 5): how detailed is it and is the feedback contradictory? \
\n2. Relevance (1 - 5): is the feedback relevant given this mission: {mission} and state: \n {state} \
\n3. Overall score: +1 (good state) or -1 (bad state) \n \
\n\nlow clarity: go the blue door, actually go to the red door instead \
\nhigh clarity: go up 1 square, then go left 1 square and open the door \
\nRespond strictly in this format: Clarity: <score>, Relevance: <score>, Overall: <+1/-1>'
    message = [{"role": "user", "content": prompt}]
    response = ask_with_retry(messages=message)

    feedback = [] #clarity, relevance, sentiment
    for response in  response.choices[0].message.content.split(", "):
        k, v = response.split(":", 1)
        feedback.append(int(v.strip().replace("+", "")))
    return (feedback[0] / 5) * ((feedback[1]) / 5) * feedback[2]

def summarize_state(env):
    view_size = env.unwrapped.agent_view_size
    grid_map = np.full((view_size, view_size), '.', dtype=object)

    # Agent position and direction
    ax, ay = env.unwrapped.agent_pos
    agent_dir_str = DMAP[env.unwrapped.agent_dir]

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

def build_cache(env, mission):
    cache = dict()
    obs, info = env.reset()
    env.unwrapped.agent_pos = (0, 0)

    #build cache
    for r in range(env.unwrapped.height):
        for c in range(env.unwrapped.width):
            env.unwrapped.agent_pos = (c, r)
            for i in DMAP:
                env.unwrapped.agent_dir = i
                grid, dir = summarize_state(env)
                feedback = query_feedback(mission=mission, dir=dir, state=grid, noise=0.3)
                cache[f'{(c, r, i)}'] = query_evaluation(state=grid, mission=mission, feedback=feedback)

    with open("cache.json", 'w') as f:
        json.dump(cache, f, indent=4)