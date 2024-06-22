import ray
from ray import tune
import os
from cleanrl.meow_continuous_action import train as train_
from cleanrl.meow_continuous_action import Args

def trainer(tuner):
    dirpath = os.path.join("/app/result_MEow")
    seed = tuner['seed']
    
    alpha = tuner['alpha']
    tau = tuner['tau']
    sigma_max = tuner['sigma_max']
    sigma_min = tuner['sigma_min']
    learning_starts = tuner['learning_starts']
    deterministic_action = tuner['deterministic_action']

    description = "Hopper-v4/" + "seed=" + str(seed) + \
                                      "_tau=" + str(tau) + \
                                      "_alpha=" + str(alpha) + \
                                      "_min=" + str(sigma_min) + \
                                      "_max=" + str(sigma_max) + \
                                      "_det=" + str(deterministic_action) + \
                                      "_warmup=" + str(learning_starts)
    
    # rewrite base config
    args = Args
    args.description = os.path.join(dirpath, description) # 
    args.device = 'cuda'
    args.seed = seed
    args.total_timesteps = 2000000
    args.env_id = "Hopper-v4"
    args.deterministic_action = deterministic_action

    args.tau = tau
    args.alpha = alpha
    args.sigma_min = sigma_min
    args.sigma_max = sigma_max
    args.learning_starts = learning_starts

    # Start tuning
    train_(args)

# ====================================

def main():
    ray.init(num_gpus=1)
    
    search_space = {
        "seed": tune.grid_search([1,2]),
        "alpha": tune.grid_search([0.25]),
        "tau": tune.grid_search([0.005]),
        "sigma_max": tune.grid_search([-0.3]),
        "sigma_min": tune.grid_search([-5.0]),
        "learning_starts": tune.grid_search([5000]),
        "deterministic_action": tune.grid_search([False]),
    }

    analysis = tune.run(
        trainer, 
        num_samples=1,
        # local_dir=dirpath,
        resources_per_trial={'cpu': 2, 'gpu': 0.5},
        config=search_space,
    )
if __name__ == '__main__':
    main()