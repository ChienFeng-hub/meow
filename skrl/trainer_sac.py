import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

def _train(cfg):
    # seed for reproducibility
    set_seed()  # e.g. `set_seed(42)` for fixed seed

    # load and wrap the Omniverse Isaac Gym environment
    env = load_omniverse_isaacgym_env(
        task_name=cfg['task_name'],
        headless=True,
        num_envs=cfg['num_envs'],
        parse_args=False,
    )
    env = wrap_env(env)
    device = env.device
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=cfg["memory_size"], num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    
    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": cfg["timesteps"], "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def main():
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["task_name"] = "AllegroHand"
    cfg["batch_size"] = 256
    cfg["num_envs"] = 512
    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4
    cfg["grad_norm_clip"] = 0
    cfg["learn_entropy"] = False
    cfg["polyak"] = 0.0025
    cfg["initial_entropy_value"] = 0.1
    cfg["timesteps"] = 1000000
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 0
    cfg["memory_size"] = 15000
    cfg["experiment"]["directory"] = "/workspace/skrl/runs/results_allegro/sac"
    _train(cfg)

if __name__ == '__main__':
    main()

