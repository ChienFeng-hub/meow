import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed


# Seed for reproducibility
seed = set_seed()  # e.g. `set_seed(42)` for fixed seed


# Define only the policy for evaluation
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


# instance VecEnvBase and setup task
headless = False  # set headless to False for rendering
env = get_env_instance(headless=headless)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from reaching_franka_omniverse_isaacgym_env import ReachingFrankaTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 64
TASK_CFG["task"]["env"]["controlSpace"] = "joint"  # "joint" or "cartesian"

sim_config = SimConfig(TASK_CFG)
task = ReachingFrankaTask(name="ReachingFranka", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device


# Instantiate the agent's policy.
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard each 32 timesteps an ignore checkpoints
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

agent = PPO(models=models_ppo,
            memory=None,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# load checkpoints
if TASK_CFG["task"]["env"]["controlSpace"] == "joint":
    agent.load("./agent_joint.pt")
elif TASK_CFG["task"]["env"]["controlSpace"] == "cartesian":
    agent.load("./agent_cartesian.pt")


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 5000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start evaluation
trainer.eval()
