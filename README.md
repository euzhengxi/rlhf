This is a github repo for my Reinforcement Learning research that is still in progress.

Do note that this repo remains under development hence there are still many bugs!

things to take note:
rollout = action, step, MDP, it can be indexed
its possible to reverse the action performed during transform using inverse

to review:
why isnt the model learning? 
double check the implementation for discrete distributions for actor
backend - review prompt and results 
latent feedback computation
parameter sharing -> loss function that is defined in terms of actor, critic and entropy loss


to read up before training:
1. scheduler & how it works 
2. calculations for advantage module - what is used as the baseline?  and how does it compute return? read the GAE paper
3. 

nice to haves:
standardise naming convention


