import os
import torch

import random
import hydra
import toy_envs
import argparse

from omegaconf import DictConfig
from modules import flatten_cfg
from modules.utils import plot_value, plot_traj_multigoal
import numpy as np

def set_deterministic(seed):
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Numpy
    np.random.seed(seed)
    # Random
    random.seed(seed)
    # OS
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    # parse args
    cfg = flatten_cfg(cfg) # flatten the nested Dict structure from hydra
    args = argparse.Namespace(**cfg)

    set_deterministic(32)

    # environment init
    args.state_sizes = 2
    args.action_sizes = 2

    path_load = args.path_load
    policy = torch.load(path_load)

    figdir = ''
    _, _ = plot_traj_multigoal(policy, figdir+'traj', runs=11) 
    plot_value(policy, figdir+'value')
    print("Finish!")

if __name__ == '__main__':
    main()