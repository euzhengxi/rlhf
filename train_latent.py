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
import clip
from PIL import Image

#from other scripts
from latentModel import latentModel, EMBEDDING_SIZE
from minigrid_env_latent import NUM_ACTIONS, ACTIONDICT, create_env
from SoftA2C import create_actor, create_critic, create_advantage

DEVICE = torch.device("cpu")
logs = defaultdict(list)
EPOCH_FILEPATH = "epochs/latent"

NUM_WORKERS = 3
TOTAL_FRAMES = 10
FRAMES_PER_BATCH = 3 #steps in a trajectory
FRAMES_PER_SUBBATCH = 5 #is this even required?  
MINIBATCHES = 2 #should be less than NUM_WORKERS * FRAMES_PER_BATCH
NUM_EPOCHS = 10 #on policy, so it still needs to be relevant
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5

CLIP_MODEL_NAME = "ViT-B/32"
CLIPMODEL, PREPROCESS = clip.load(CLIP_MODEL_NAME, device=DEVICE)

def initialise_critic(critic):
    dummy_obs = torch.zeros(1, 3, 7, 7, device=DEVICE) 

    dummy_td = TensorDict({
        "observation": {
            "image": dummy_obs
        }
    }, batch_size=[1])

    _ = critic(dummy_td) 

def collect_samples(workerRemote, env, actor, latentModel, device, framesPerBatch):
    obs, _ = env.reset()
    text_tokens = clip.tokenize([obs['mission']]).to(device)
    missionEmbedding = CLIPMODEL.encode_text(text_tokens)

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
            latent_obs = torch.empty(framesPerBatch, EMBEDDING_SIZE, dtype=torch.float, device=device)
            latent_actions = torch.empty(framesPerBatch, EMBEDDING_SIZE, dtype=torch.float, device=device)
            latent_nextObs = torch.empty(framesPerBatch, EMBEDDING_SIZE, dtype=torch.float, device=device)

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
                    next_obs_image[i] = torch.tensor(next_obs["image"], dtype=torch.float32, device=DEVICE).permute(2,0,1)
                    next_obs_direction[i] = next_obs["direction"]
                    done[i] = sdone

                    with torch.no_grad():
                        resized_img = Image.fromarray(obs["image"]).resize((224, 224))
                        resized_next_img = Image.fromarray(next_obs["image"]).resize((224, 224))
                        stateEmbedding = CLIPMODEL.encode_image(PREPROCESS(resized_img).unsqueeze(0).to(DEVICE))
                        actionEmbedding = CLIPMODEL.encode_text(clip.tokenize([ACTIONDICT[outputt["action"].item()]]).to(device))
                        nextStateEmbedding = CLIPMODEL.encode_image(PREPROCESS(resized_next_img).unsqueeze(0)).to(DEVICE)
                        latentReward = latentModel(stateEmbedding, missionEmbedding) 
                        reward[i] = sreward + 0.5 * (1 - torch.nn.functional.cosine_similarity(latentReward, missionEmbedding))

                    latent_obs[i] = stateEmbedding
                    latent_actions[i] = actionEmbedding
                    latent_nextObs[i] = nextStateEmbedding
                    

                    obs = next_obs

                    if sdone:
                        obs, _ = env.reset()
                    
            workerRemote.send((combined, obs_image, obs_direction, action, action_log_prob, next_obs_image, next_obs_direction, reward, done, latent_obs.cpu(), latent_actions.cpu(), latent_nextObs.cpu()))
            print("sampling done")

        elif cmd == "update_actor":
            actor.load_state_dict(data)
        elif cmd == "update_latent":
            latentModel.load_state_dict(data)
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
        combined, obs_images, obs_direction, actions, action_log_probs, next_obs_images, next_obs_direction, rewards, dones, latent_obs, latent_actions, latent_nextObs = zip(*results)

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
        
        latent_td = TensorDict({
                "current": torch.cat(latent_obs, dim=0),
                "action": torch.cat(latent_actions, dim=0),
                "next": torch.cat(latent_nextObs, dim=0),
            }, batch_size=[batch_size])

        buffer_td = TensorDict({
            "observation": obs,
            "action": torch.cat(actions, dim=0),
            'action_log_prob': torch.cat(action_log_probs, dim=0),
            "next": next_td,
            "latent": latent_td
        }, batch_size=[batch_size])

        yield buffer_td

def update_state_dicts_in_workers(parentRemotes, actorStateDict, latentStateDict):
    for pr in parentRemotes:
        pr.send(("update_actor", actorStateDict))
        pr.send(("update_latent", latentStateDict))
    
def env_rollout(env, actor): #consider multiple env rollouts and averaging
    obs, _ = env.reset()
    done = False
    totalReward = 0.0
    stepCount = 0

    while stepCount <= 50 and not done:
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
        print(stepCount)

    return totalReward / stepCount, totalReward, stepCount

def training(env, actor, critic, advantage, collector, replayBuffer, parentRemotes, device):
    loss_function = ClipPPOLoss( 
        actor_network=actor,
        critic_network=critic,
        entropy_bonus=True,
    )

    optim = torch.optim.Adam(loss_function.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MINIBATCHES, 0.0)

    loss_function_mlp = torch.nn.MSELoss()
    optimMLP = torch.optim.Adam(loss_function.parameters(), LEARNING_RATE)
    schedulerMLP = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MINIBATCHES, 0.0)

    print(">>> starting training")
    print()
    for i, tensordict_data in enumerate(collector):
        advantage(tensordict_data)  #fixing the landscape for more stable updates in PPO
        data_view = tensordict_data.reshape(-1)
        replayBuffer.extend(data_view.cpu()) 

        for j in range(NUM_EPOCHS):
            for k in range(MINIBATCHES):
                subdata = replayBuffer.sample(FRAMES_PER_BATCH)
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

                #print(subdata["latent"]["current"].shape, subdata["latent"]["action"].shape, subdata["latent"]["next"].shape)
                latent_prediction = latentModel(subdata["latent"]["current"], subdata["latent"]["action"])
                loss_mlp = loss_function_mlp(latent_prediction, subdata["latent"]["next"])
                loss_mlp.backward()
                optimMLP.step()
                optimMLP.zero_grad()

        step = i * NUM_EPOCHS * MINIBATCHES + j * MINIBATCHES

        #logging
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["steps"].append(step) #tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])

        print(f'{step}: Current reward: {logs["reward"][-1]: 4.4f} LR: {logs["lr"][-1]: 4.6f}')
                
        #evaluation and saving state every 10 epochs
        if (step) % 2 == 0:
            update_state_dicts_in_workers(parentRemotes=parentRemotes, actorStateDict=actor.state_dict(), latentStateDict=latentModel.state_dict())
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
        schedulerMLP.step()


if __name__ == "__main__":  
    print(">>> initialising variables")
    env = create_env()
    actor = create_actor(numActions=NUM_ACTIONS, filePath="", device=DEVICE) 
    critic = create_critic(filePath="", device=DEVICE)
    initialise_critic(critic=critic)
    latentModel = latentModel().to(DEVICE)
    advantage = create_advantage(critic=critic, device=DEVICE)
    
    #initialising worker processes for concurrent env sampling 
    parentRemotes, workerRemotes = zip(*[Pipe() for _ in range(NUM_WORKERS)])
    processes = []

    for pr, wr in zip(parentRemotes, workerRemotes):
        p = Process(target=collect_samples, args=(wr, env, actor, latentModel, DEVICE, FRAMES_PER_BATCH))
        p.start()
        processes.append(p)
    
    collector = multi_sync_collector(numWorkers=NUM_WORKERS, parentRemotes=parentRemotes, framesPerBatch=FRAMES_PER_BATCH, totalFrames=TOTAL_FRAMES)

    replayBuffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH),
        sampler=SliceSampler(num_slices=FRAMES_PER_BATCH) #double check on usage
    )

    training(env=env, actor=actor, critic=critic, advantage=advantage, collector=collector, replayBuffer=replayBuffer, parentRemotes=parentRemotes, device=DEVICE)