from collections.abc import MutableMapping
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def flatten_cfg(cfg):
    items = []
    for key, value in cfg.items():
        if isinstance(value, MutableMapping):
            items.extend(flatten_cfg(value).items())
        else:
            items.append((key, value))
    return dict(items)

def log(result, global_step, logger):
    for k, v in result.items():
        logger.add_scalar(k, v, global_step)

def outputdir_make_and_add(outputdir, title=None):
    #creates outputdir
    os.makedirs(outputdir,exist_ok=True)
    folder_num = len(next(os.walk(outputdir))[1]) #counts how many folders already there 
    if folder_num == 0:
        folder_num = 1
    elif folder_num == 1 and next(os.walk(outputdir))[1][0][0] == ".":
        folder_num = 1
    else:
        folder_num = max([int(i.split('-')[0]) for i in next(os.walk(outputdir))[1] if i[0] != '.'],default=0) + 1 # this looks for max folder num and adds one... this works even if title is used (because we index at 1) (dot check to ignore .ipynb) 
        #currently returns error when a subfolder contains anything other than a number (exept dot handle) 
        #so essentially this assumes the outputdir structure with numbers (and possible titles). will need to fix if i want to use it later for something else
        
    if title == None:
        outputdir += '/' + str(folder_num) #adds one
    else:
        outputdir += '/' + str(folder_num) + f'-{title}' #adds one and appends title
    os.makedirs(outputdir,exist_ok=True)
    return outputdir

import gymnasium as gym
def plot_traj_multigoal(critic, file_name='_.png', runs=8, deterministic=False, rewards_thres=-10000000, render_plot=True):
    with torch.no_grad():
        paths = []
        rewards = []
        env = gym.make("MultiGoal")
        for i in range(runs):
            rewards_ = np.zeros((1,))
            dones = np.zeros((1,)).astype(bool)
            s, info = env.reset(seed=range(1))
            t = 0
            path = []
            reward = []
            path += [info['pos']]
            while not all(dones):
                a, _ = critic.sample(num_samples=1, obs=np.expand_dims(s, axis=0), deterministic=deterministic) #
                a = a.cpu().detach().numpy()
                s_, r, terminated, truncated, info = env.step(a)
                done = terminated | truncated
                rewards_ += r * (1-dones)
                dones |= done
                s = s_
                t += 1
                path += [info['pos']]
                reward += [r * (1-dones)]
            if rewards_.mean() > rewards_thres:
                paths.append(path)
                rewards.append(reward)
        if render_plot:
            env.render_rollouts(paths, file_name+'.png')
    return paths, rewards

def plot_value(critic, file_name='_.png'):
    critic.eval()
    grid_size = 100
    xx, yy = torch.meshgrid(torch.linspace(-8, 8, grid_size), torch.linspace(-8, 8, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2).to('cuda')

    neg_energy = critic.get_v(obs=torch.cat((zz, zz), dim=0))
    neg_energy = neg_energy[:neg_energy.shape[0]//2].view(*xx.shape).detach().cpu().numpy()
        
    plt.pcolormesh(xx, yy, neg_energy)
    plt.xlabel('s0')
    plt.ylabel('s1')
    
    plt.savefig(file_name+'.png', bbox_inches='tight')
