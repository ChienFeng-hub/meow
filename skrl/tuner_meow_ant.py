import ray
from ray import tune
import os
from trainer_meow import _train
from skrl.agents.torch.meow import MEow_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

def trainer(tuner):
    id = tuner['id']
    grad_clip = tuner['grad_clip']
    tau = tuner['tau']
    alpha = tuner['alpha']
    lr = tuner['lr']
    loading = tuner['loading']
    num_envs = tuner['num_envs']
    bs = int(loading / num_envs)
    timesteps = tuner['timesteps']
    path = tuner['path']
    task_name = tuner['task_name']
    sigma_max = tuner['sigma_max']
    sigma_min = tuner['sigma_min']

    description = path + str(id)
    
    # rewrite base config
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = MEow_DEFAULT_CONFIG.copy()
    cfg["task_name"] = task_name
    cfg["polyak"] = tau
    cfg["entropy_value"] = alpha
    cfg["grad_norm_clip"] = grad_clip
    cfg["learning_rate"] = lr
    cfg["batch_size"] = bs
    cfg["num_envs"] = num_envs
    cfg["timesteps"] = timesteps
    cfg["sigma_max"] = sigma_max
    cfg["sigma_min"] = sigma_min
    cfg["experiment"]["directory"] = description
    # --------
    cfg["gradient_steps"] = 1
    cfg["discount_factor"] = 0.99
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["memory_size"] = 15000
    cfg["experiment"]["write_interval"] = 5000
    cfg["experiment"]["checkpoint_interval"] = timesteps
    
    _train(cfg)

# ====================================

def main():
    ray.init(num_gpus=1)
    
    search_space = {
        "task_name": tune.grid_search(["Ant"]),
        "grad_clip": tune.grid_search([30]),
        "tau": tune.grid_search([0.0005]),
        "alpha": tune.grid_search([0.075]),
        "lr": tune.grid_search([1e-3]),
        "loading": tune.grid_search([131072]),
        "num_envs": tune.grid_search([128]),
        "timesteps": tune.grid_search([1000000]),
        "sigma_max": tune.grid_search([2.0]),
        "sigma_min": tune.grid_search([-5.0]),
        "id": tune.grid_search([0,1]),
        "path": tune.grid_search(["/workspace/skrl/runs/results_ant/meow/"]),
    }
    
    analysis = tune.run(
        trainer, 
        num_samples=1,
        resources_per_trial={'cpu': 2, 'gpu': 0.5},
        config=search_space,
    )

if __name__ == '__main__':
    main()