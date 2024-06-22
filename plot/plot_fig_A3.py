from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

import os 
from tbparse import SummaryReader
import numpy as np

def plot(env="Hopper-v4", steps=1000000, yticks=None, xticks=None, palette=None):
    print("plotting...")
    os.makedirs('fig_A3', exist_ok = True)
    log_dir = os.path.join("smoothed", env)
    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars
    
    df = df[["Steps", "Test/return", "dir_name"]]
    df = df.assign(dir_name=df["dir_name"].apply(lambda s: s.split('/')[0]))

    fig = plt.figure(figsize=(5,5.5))
    ax = plt.gca()
    sns.set_theme(style='whitegrid')
    plt.grid(color='lightgray')
     
    g = sns.lineplot(data=df, x='Steps', y='Test/return', hue='dir_name', palette=palette) #

    g.set(xlim=(0, steps))
    if env == "Hopper-v4":
        g.set(ylim=(0, 3750))
    else:
        g.set(ylim=(yticks[0], yticks[-1]))
    if xticks is not None:
        g.set_xticks(xticks)
    if yticks is not None:
        g.set_yticks(yticks)
    plt.legend([],[], frameon=False)
    
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('fig_A3/'+env+'.png')
    plt.close(fig)
    print("Finish plotting.")

def main(env, steps, yticks, xticks, palette=None):
    plot(env=env, steps=steps, yticks=yticks, xticks=xticks, palette=palette)

if __name__ == '__main__':
    for env in ["HalfCheetah-v4", "Ant-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4"]:        
        if env == "HalfCheetah-v4":
            steps = 1500000
            yticks = np.arange(-1500, 13500+2500, 2500)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Ant-v4":
            steps = 4000000
            yticks = np.arange(0, 7500+1500, 1500)
            xticks = np.arange(0, steps+1, 1000000)
        elif env == "Hopper-v4":
            steps = 1500000
            yticks = np.arange(0, 3500+500, 500)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Walker2d-v4":
            steps = 4000000
            yticks = np.arange(0, 6000+1000, 1000)
            xticks = np.arange(0, steps+1, 1000000)
        elif env == "Humanoid-v4":
            steps = 5000000
            yticks = np.arange(0, 8000+1000, 1000)
            xticks = np.arange(0, steps+1, 1000000)
        palette = ['xkcd:jade', 'xkcd:orange', 'xkcd:coral', 'xkcd:violet', 'xkcd:deep sky blue']
        main(env, steps, yticks, xticks, palette=palette)