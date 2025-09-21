from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.objectives.value import GAE
from torch.distributions import Categorical

NUM_CELLS = 256
GAMMA = 0.99
LAMBDA = 0.95
class ToFloat(nn.Module):
    def forward(self, x):
        return x.float()

def create_actor(env, device):
    actor_net = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.ReLU(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.ReLU(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.ReLU(),
        nn.LazyLinear(env.action_space.n, device=device),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["image"], out_keys=["logits"]
    )

    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True
    )
    return actor

def create_critic(device):
    value_net = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    critic = ValueOperator(
        module=value_net,
        in_keys=[("observation", "image")]
    )

    return critic

def create_advantage(critic, device):
    advantage_module = GAE(
        gamma=GAMMA, lmbda=LAMBDA, value_network=critic, average_gae=True, device=device,
    )
    return advantage_module