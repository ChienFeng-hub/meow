import ray
from ray import tune
import os
from cleanrl.ddpg_continuous_action import train as train_
from cleanrl.ddpg_continuous_action import Args

def trainer(tuner):
    _dirpath = "/lancechao/cleanrl2" #os.path.dirname(os.path.realpath(__file__))
    dirpath = os.path.join(_dirpath, "result_DDPG_CleanRL")
    seed = tuner['seed']
    description = "Humanoid-v4/" + str(seed)
    
    # rewrite base config
    args = Args
    args.description = os.path.join(dirpath, description) # 
    args.device = 'cuda'
    args.seed = seed
    args.total_timesteps = 5000000
    args.env_id = "Humanoid-v4"

    # # Start tuning
    train_(args)

# ====================================

def main():
    _dirpath = "/lancechao/cleanrl2" #os.path.dirname(os.path.realpath(__file__))
    dirpath = os.path.join(_dirpath, "result_DDPG_CleanRL")
    ray.init(num_gpus=1, _temp_dir=dirpath)
    
    search_space = {
        "seed": tune.grid_search([0,1,2,3,4]),
    }

    analysis = tune.run(
        trainer, 
        num_samples=1,
        local_dir=dirpath,
        resources_per_trial={'cpu': 2, 'gpu': 0.2},
        config=search_space,
    )

if __name__ == '__main__':
    main()