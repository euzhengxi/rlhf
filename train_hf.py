import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl.data.replay_buffers.samplers")

import numpy as np
import os
import torch
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from collections import defaultdict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict
from multiprocessing import Pipe, Process
import json

#from other scripts
from minigrid_env import NUM_ACTIONS, create_env
from SoftA2C import create_actor, create_critic, create_advantage
from backend import build_cache


NUM_WORKERS = 5
TOTAL_FRAMES = 20000
FRAMES_PER_BATCH = 100 #total frames per iteration = FRAMES_PER_BATCH * NUM_WORKERS, should this be the max number of frames in an episode? 
SUBBATCH = 64 #TD lambda
TRAJECTORY_LENGTH = 8
NUM_EPOCHS = 10 #on policy, so it still needs to be relevant
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5
ISCACHED = True

DEVICE = torch.device("cpu")
EPOCH_FILEPATH = "epochs/hf"

def initialise_critic(critic):
    dummy_obs = torch.zeros(1, 3, 7, 7, device=DEVICE) 

    dummy_td = TensorDict({
        "observation": {
            "image": dummy_obs
        }
    }, batch_size=[1])

    _ = critic(dummy_td) 

def collect_samples(workerRemote, env, actor, device, framesPerBatch):
    obs, _ = env.reset()
    while True:
        cmd, data = workerRemote.recv()
        if cmd == "collect":
            obs_image = torch.empty(framesPerBatch, 3, 7, 7, device=device)
            obs_direction = torch.empty(framesPerBatch, dtype=torch.float, device=device)
            combined = torch.empty(framesPerBatch, 148, device=device) #3x7x7 + 1
            action = torch.empty(framesPerBatch, dtype=torch.long, device=device)
            action_log_prob = torch.empty(framesPerBatch, dtype=torch.float, device=device)
            reward = torch.empty(framesPerBatch, dtype=torch.float, device=device)
            next_obs_image = torch.empty(framesPerBatch, 3, 7, 7, device=device)
            next_obs_direction = torch.empty(framesPerBatch, dtype=torch.float, device=device)
            done = torch.empty(framesPerBatch, dtype=torch.bool, device=device)

            for i in range(framesPerBatch):
                    img = torch.tensor(obs["image"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                    direction = torch.tensor(obs["direction"], dtype=torch.float32, device=device).view(1, 1) 
                    scombined = torch.cat([img.flatten(start_dim=1), direction], dim=1)  
                    td = TensorDict({
                        "observation": TensorDict({
                            "combined": scombined
                        }, batch_size=[1])
                    }, batch_size=[1])

                    with torch.no_grad():
                        outputt = actor(td)    
                    
                    next_obs, sreward, terminated, truncated, info = env.step(outputt['action'].item())
                    sdone = terminated or truncated

                    combined[i] = scombined
                    obs_image[i] = torch.tensor(obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
                    obs_direction[i] = obs["direction"]
                    action[i] = outputt['action'].item()
                    action_log_prob[i] = outputt['action_log_prob'].item() 
                    reward[i] = sreward
                    next_obs_image[i] = torch.tensor(next_obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
                    next_obs_direction[i] = next_obs["direction"]
                    done[i] = sdone

                    obs = next_obs

                    if sdone:
                        obs, _ = env.reset()
                    
            workerRemote.send((combined, obs_image, obs_direction, action, action_log_prob, next_obs_image, next_obs_direction, reward, done))

        elif cmd == "update":
            actor.load_state_dict(data)
        elif cmd == "close":
            workerRemote.close()
            break
        
        cmd = ""

    
def multi_sync_collector(numWorkers, parentRemotes, framesPerBatch, totalFrames):
    print("creating generator")
    frameCount = 0
    
    while frameCount < totalFrames:
        for pr in parentRemotes:
            pr.send(("collect", None))
        
        print("waiting for results")
        results = [pr.recv() for pr in parentRemotes]
        combined, obs_images, obs_direction, actions, action_log_probs, next_obs_images, next_obs_direction, rewards, dones = zip(*results)
        print("sampling done")

        frameCount += numWorkers * framesPerBatch
        batch_size = numWorkers * framesPerBatch

        obs = TensorDict({
                "image": torch.cat(obs_images, dim=0),
                "direction": torch.cat(obs_direction, dim=0),
                "combined": torch.cat(combined, dim=0), #for backprop subsequently
            }, batch_size=[batch_size])
        
        next_obs = TensorDict({
                "image": torch.cat(next_obs_images, dim=0),
                "direction": torch.cat(next_obs_direction, dim=0),
            }, batch_size=[batch_size])
        
        next_td = TensorDict({
                "observation": next_obs,
                "reward": torch.cat(rewards, dim=0),
                "done": torch.cat(dones, dim=0),
            }, batch_size=[batch_size])

        buffer_td = TensorDict({
            "observation": obs,
            "action": torch.cat(actions, dim=0),
            'action_log_prob': torch.cat(action_log_probs, dim=0),
            "next": next_td
        }, batch_size=[batch_size])

        yield buffer_td

def update_actor_in_workers(parentRemotes, actorStateDict):
    for pr in parentRemotes:
        pr.send(("update", actorStateDict))
    
def env_rollout(env, actor): #consider multiple env rollouts and averaging
    obs, _ = env.reset()
    done = False
    totalReward = 0.0
    stepCount = 0

    while not done:
        obs_image = torch.tensor(obs["image"], dtype=torch.float32, device=DEVICE).permute(2, 0, 1).unsqueeze(0)
        obs_direction = torch.tensor(obs["direction"], dtype=torch.float32, device=DEVICE).view(1, 1) 
        combined = torch.cat([obs_image.flatten(start_dim=1), obs_direction], dim=1)  
        td = TensorDict({
            "observation": TensorDict({
                "combined": combined
            }, batch_size=[1])
        }, batch_size=[1])

        with torch.no_grad():
            outputt = actor(td)    
        
        next_obs, reward, terminated, truncated, info = env.step(outputt['action'].item())
        done = terminated or truncated
        totalReward += reward
        stepCount += 1
        obs = next_obs

    return totalReward / stepCount, totalReward, stepCount

def training(env, actor, critic, advantage, collector, replayBuffer, parentRemotes, device):
    logs = defaultdict(list)
    loss_function = ClipPPOLoss( 
        actor_network=actor,
        critic_network=critic,
        entropy_bonus=True,
        entropy_coeff=0.01
    )

    optim = torch.optim.Adam(loss_function.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, TOTAL_FRAMES // (NUM_WORKERS * FRAMES_PER_BATCH), 0.0)

    print(">>> starting training")
    print()
    for i, tensordict_data in enumerate(collector):
        advantage(tensordict_data)  #fixing the landscape for more stable updates in PPO
        data_view = tensordict_data.reshape(-1)
        replayBuffer.extend(data_view.cpu()) 

        for j in range(NUM_EPOCHS):
            for k in range(NUM_WORKERS * FRAMES_PER_BATCH // SUBBATCH):
                subdata = replayBuffer.sample(SUBBATCH)
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

        step = i + 1

        #logging
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["steps"].append(step) #tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])

        print(f'{step}: Current reward: {logs["reward"][-1]: 4.4f} LR: {logs["lr"][-1]: 4.6f}')
                
        update_actor_in_workers(parentRemotes=parentRemotes, actorStateDict=actor.state_dict())
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            averageReward, trajectoryReward, stepCount = env_rollout(env=env, actor=actor)
            logs["eval_reward"].append(averageReward)
            logs["eval_reward_sum"].append(trajectoryReward)
            logs["eval_steps"].append(stepCount) #eval_rollout["step_count"].max().item())
                         
            print(f'EV: Current reward: {logs["eval_reward"][-1]: 4.4f}. Trajectory reward: {logs["eval_reward_sum"][0]: 4.4f}')
            print()

            torch.save(actor.state_dict(), os.path.join(EPOCH_FILEPATH, f"actor_step_{logs['steps'][-1]}.pt"))
            torch.save(critic.state_dict(), os.path.join(EPOCH_FILEPATH, f"value_step_{logs['steps'][-1]}.pt"))
        scheduler.step()
    
    with open("results_hf.json", "w") as f:
        json.dump(dict(logs), f)


if __name__ == "__main__":  
    
    env, mission = create_env()
    if not ISCACHED:
        print(">>> building cache")
        build_cache(env, mission)
    env.load_cache()

    print("cache loaded")
    print(">>> initialising variables")
    actor = create_actor(numActions=NUM_ACTIONS, filePath="", device=DEVICE) 
    critic = create_critic(filePath="", device=DEVICE)
    initialise_critic(critic=critic)
    advantage = create_advantage(critic=critic, device=DEVICE)
    
    #initialising worker processes for concurrent env sampling 
    parentRemotes, workerRemotes = zip(*[Pipe() for _ in range(NUM_WORKERS)])
    processes = []

    for pr, wr in zip(parentRemotes, workerRemotes):
        p = Process(target=collect_samples, args=(wr, env, actor, DEVICE, FRAMES_PER_BATCH))
        p.start()
        processes.append(p)
    
    collector = multi_sync_collector(numWorkers=NUM_WORKERS, parentRemotes=parentRemotes, framesPerBatch=FRAMES_PER_BATCH, totalFrames=TOTAL_FRAMES)

    replayBuffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=NUM_WORKERS * FRAMES_PER_BATCH),
        sampler=SliceSampler(num_slices=TRAJECTORY_LENGTH) 
    )

    training(env=env, actor=actor, critic=critic, advantage=advantage, collector=collector, replayBuffer=replayBuffer, parentRemotes=parentRemotes, device=DEVICE)