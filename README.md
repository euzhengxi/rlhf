This is a github repo for my Reinforcement Learning research that is still in progress.

Do note that this repo remains under development hence there are still many bugs!

things to take note:
rollout = action, step, MDP, it can be indexed
its possible to reverse the action performed during transform using inverse
PPO needs sufficient samples to be stable and GAE requires trajectories rather than individual episodes

to review:
double check the implementation for discrete distributions for actor
stepCount tracking - can be omitted for now
critic network - trajectory vs individual state reward & trajectory reward computation
backend - review prompt and results


to read up before training:
1. advantage parameters - gamma, lambda
2. scheduler & how it works 
3. calculations for advantage module - what is used as the baseline? 


nice to haves:
standardise naming convention


