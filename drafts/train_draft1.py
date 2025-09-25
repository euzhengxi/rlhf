import numpy as np
import os
import torch
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict

from minigrid_env import create_env
from SoftA2C import create_actor, create_critic, create_advantage

'''
PPO 
- actor critic with clipped gradient updates, 
- update the critic every epoch but keep it under control

AFIs:
1. multiprocessing 
2. scheduler 
'''

#encasing minigrid within torch rl env
#creation of actor
#creation of value
#instantiating loss function

#collecting reward 
#updating of actor model
#updating of value model (once every 100 frames?)


#what are the things i should plot? 
#step sizes, run

ENV_NAME = "MiniGrid-RedBlueDoors-8x8-v0"
TOTAL_FRAMES = 10
FRAMES_PER_BATCH = 5 #how many steps you want to take
FRAMES_PER_SUBBATCH = 5 #256 #mini_batch size, but you dont iterate through as much? 
MINIBATCHES = FRAMES_PER_BATCH // FRAMES_PER_SUBBATCH
NUM_EPOCHS = 10 #on policy, so it still needs to be relevant
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cpu")
logs = defaultdict(list)
EPOCH_FILEPATH = "epochs"

def initialise_critic(critic):
    dummy_obs = torch.zeros(1, 3, 7, 7, device=DEVICE)  #adjust shape to match obs
    dummy_dir = torch.zeros(1, dtype=torch.long, device=DEVICE)
    dummy_mission = torch.zeros(1, dtype=torch.long, device=DEVICE)

    dummy_td = TensorDict({
        "observation": {
            "image": dummy_obs,
            "direction": dummy_dir,
            "mission": dummy_mission,
        }
    }, batch_size=[1])
    _ = critic(dummy_td) 

def collect_samples(obs_images, obs_directions, obs_missions, actions, action_log_probs, rewards, next_obs_images, next_obs_directions, next_obs_missions, dones, batchSize, frameCount, env, actor, device):
    for i in range(batchSize):
            td = TensorDict({
                "observation": TensorDict({
                    "image": torch.tensor(obs["image"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0),
                }, batch_size=[1])
            }, batch_size=[1])
            with torch.no_grad():
                outputt = actor(td)    
            next_obs, reward, terminated, truncated, info = env.step(outputt['action'].item())
            done = terminated or truncated
            action = outputt['action'].item()
            action_log_prob = outputt['action_log_prob'].item()    
            
            obs_images[i] = torch.tensor(obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
            obs_directions[i] = obs["direction"]
            actions[i] = action
            action_log_probs[i] = action_log_prob
            rewards[i] = reward
            next_obs_images[i] = torch.tensor(next_obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
            next_obs_directions[i] = next_obs["direction"]
            dones[i] = done

            obs = next_obs
            frame_count += 1

            if done:
                obs, _ = env.reset()

def multi_sync_collector(workers, framesPerBatch, totalFrames, device, env, actor):
    obs, _ = env.reset()
    frameCount = 0
    #instantiate multiple workers 
    while frameCount < totalFrames:
        batchSize = min(framesPerBatch, totalFrames - frameCount)

        obs_images = torch.empty(batchSize, 3, 7, 7, device=device)
        obs_directions = torch.empty(batchSize, dtype=torch.long, device=device)
        obs_missions = torch.zeros(batchSize, dtype=torch.long, device=device)
        actions = torch.empty(batchSize, dtype=torch.long, device=device)
        action_log_probs = torch.empty(batchSize, dtype=torch.float, device=device)
        rewards = torch.empty(batchSize, dtype=torch.float, device=device)
        next_obs_images = torch.empty(batchSize, 3, 7, 7, device=device)
        next_obs_directions = torch.empty(batchSize, dtype=torch.long, device=device)
        next_obs_missions = torch.zeros(batchSize, dtype=torch.long, device=device)
        dones = torch.empty(batchSize, dtype=torch.bool, device=device)

        for i in range(workers):
            #spawn worker processes
            collect_samples(obs_images, obs_directions, obs_missions, actions, action_log_probs, rewards, next_obs_images, next_obs_directions, next_obs_missions, dones, batchSize, frameCount, env, actor, device)
        
        #join worker processes
        frameCount += workers * batchSize

        obs = TensorDict({
                "image": obs_images,
                "direction": obs_directions,
                "mission": obs_missions,
            }, batch_size=[batchSize])
        next_obs = TensorDict({
                "image": next_obs_images,
                "direction": next_obs_directions,
                "mission": next_obs_missions,
            }, batch_size=[batchSize])
    
        buffer_td = TensorDict({
            "observation": obs,
            "action": actions,
            'action_log_prob': action_log_probs,
            "next": TensorDict({
                "observation": next_obs,
                "reward": rewards,
                "done": dones,
            }, batch_size=[batchSize])
        }, batch_size=[batchSize])

        yield buffer_td

def create_dataCollector(env, actor, frames_per_batch, total_frames, device):
    obs, _ = env.reset()
    frame_count = 0
    #instantiate multiple workers 
    while frame_count < total_frames:
        batchSize = min(frames_per_batch, total_frames - frame_count)

        obs_images = torch.empty(batchSize, 3, 7, 7, device=DEVICE)
        obs_directions = torch.empty(batchSize, dtype=torch.long, device=DEVICE)
        obs_missions = torch.zeros(batchSize, dtype=torch.long, device=DEVICE)
        actions = torch.empty(batchSize, dtype=torch.long, device=DEVICE)
        action_log_probs = torch.empty(batchSize, dtype=torch.float, device=DEVICE)
        rewards = torch.empty(batchSize, dtype=torch.float, device=DEVICE)
        next_obs_images = torch.empty(batchSize, 3, 7, 7, device=DEVICE)
        next_obs_directions = torch.empty(batchSize, dtype=torch.long, device=DEVICE)
        next_obs_missions = torch.zeros(batchSize, dtype=torch.long, device=DEVICE)
        dones = torch.empty(batchSize, dtype=torch.bool, device=DEVICE)

        for i in range(batchSize):
            td = TensorDict({
                "observation": TensorDict({
                    "image": torch.tensor(obs["image"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0),
                }, batch_size=[1])
            }, batch_size=[1])
            with torch.no_grad():
                outputt = actor(td)    
            next_obs, reward, terminated, truncated, info = env.step(outputt['action'].item())
            done = terminated or truncated
            action = outputt['action'].item()
            action_log_prob = outputt['action_log_prob'].item()    
            
            obs_images[i] = torch.tensor(obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
            obs_directions[i] = obs["direction"]
            actions[i] = action
            action_log_probs[i] = action_log_prob
            rewards[i] = reward
            next_obs_images[i] = torch.tensor(next_obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
            next_obs_directions[i] = next_obs["direction"]
            dones[i] = done

            obs = next_obs
            frame_count += 1

            if done:
                obs, _ = env.reset()

        obs = TensorDict({
                "image": obs_images,
                "direction": obs_directions,
                "mission": obs_missions,
            }, batch_size=[batchSize])
        next_obs = TensorDict({
                "image": next_obs_images,
                "direction": next_obs_directions,
                "mission": next_obs_missions,
            }, batch_size=[batchSize])
    
        buffer_td = TensorDict({
            "observation": obs,
            "action": actions,
            'action_log_prob': action_log_probs,
            "next": TensorDict({
                "observation": next_obs,
                "reward": rewards,
                "done": dones,
            }, batch_size=[batchSize])
        }, batch_size=[batchSize])

        yield buffer_td
            

def training(actor_filePath=None, critic_filePath=None):
    print()
    print(">>> initialising variables")
    env = create_env(env_name=ENV_NAME)

    actor = create_actor(env=env, device=DEVICE)
    if actor_filePath:
        actor.load_state_dict(torch.load(actor_filePath, weights_only=True)) 
    
    critic = create_critic(device=DEVICE)
    if critic_filePath:
        critic.load_state_dict(torch.load(critic_filePath, weights_only=True))
    
    initialise_critic(critic=critic)
    advantage = create_advantage(critic=critic, device=DEVICE)

    loss_function = ClipPPOLoss( #using default clip and entropy values: 0.1 and 0.01 respectively
        actor_network=actor,
        critic_network=critic,
        entropy_bonus=True,
    )

    optim = torch.optim.Adam(loss_function.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MINIBATCHES, 0.0)
    
    collector = create_dataCollector(env=env, actor=actor, frames_per_batch=FRAMES_PER_BATCH, total_frames=TOTAL_FRAMES, device=DEVICE)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH), #multi dimenional buffer? 
        sampler=SamplerWithoutReplacement(),                  #time slicing to create trajectories? 
    )
    
    print(">>> starting training")
    print()
    for i, tensordict_data in enumerate(collector):
        for j in range(NUM_EPOCHS):
            advantage(tensordict_data) #appends advantage attribute to be used by PPO
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu()) 
            for k in range(MINIBATCHES):
                subdata = replay_buffer.sample(FRAMES_PER_SUBBATCH)
                loss_vals = loss_function(subdata.to(DEVICE))
                loss_value = (
                    loss_vals["loss_objective"] #actor
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_function.parameters(), MAX_GRAD_NORM) #good to have
                optim.step()
                optim.zero_grad()

                step = i * NUM_EPOCHS * MINIBATCHES + j * MINIBATCHES + k

                 #logging
                logs["reward"].append(tensordict_data["next", "reward"].mean().item())
                logs["steps"].append(step) #tensordict_data["step_count"].max().item())
                logs["lr"].append(optim.param_groups[0]["lr"])

                print(f'{step}: Current reward: {logs["reward"][-1]: 4.4f} LR: {logs["lr"][-1]: 4.6f}')
                
                #evaluation and saving state every 10 epochs
                if (step + 1) % 5 == 0:
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        next_obs = subdata['next']
                        outputt = actor(next_obs)
                        next_obs, reward, terminated, truncated, info = env.step(outputt['action'][0])
                        logs["eval_reward"].append(reward)
                        logs["eval_reward_sum"].append(reward)
                        logs["eval_steps"].append(step) #eval_rollout["step_count"].max().item())
                        
                        print(f'EV: Current reward: {logs["eval_reward"][-1]: 4.4f}. Trajectory reward: {logs["eval_reward_sum"][0]: 4.4f}')
                        print()

                    torch.save(actor.state_dict(), os.path.join(EPOCH_FILEPATH, f"actor_step_{logs['steps'][-1]}.pt"))
                    torch.save(critic.state_dict(), os.path.join(EPOCH_FILEPATH, f"value_step_{logs['steps'][-1]}.pt"))
            scheduler.step()

if __name__ == "__main__":  
    training()