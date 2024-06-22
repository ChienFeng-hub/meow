from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

import os 
from tbparse import SummaryReader
import numpy as np

def plot(env="Ant", steps=1000000, yticks=None, xticks=None, palette=None):
    print("plotting...")
    os.makedirs('fig_4', exist_ok = True)
    log_dir = os.path.join("smoothed", env)
    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars
    
    df = df[["Step", "return", "dir_name"]]
    df = df.assign(dir_name=df["dir_name"].apply(lambda s: s.split('/')[0]))

    fig = plt.figure(figsize=(5,5.5))
    ax = plt.gca()
    sns.set_theme(style='whitegrid')
    plt.grid(color='lightgray')
     
    g = sns.lineplot(data=df, x='Step', y='return', hue='dir_name', palette=palette) #
    g.set(xlim=(0, steps))
    if env == "Ingenuity":
        g.set(ylim=(-500, yticks[-1]))
    elif env == "Anymal":
        g.set(ylim=(-5, yticks[-1]))
    else:
        g.set(ylim=(yticks[0], yticks[-1]))
    if xticks is not None:
        g.set_xticks(xticks)
    if yticks is not None:
        g.set_yticks(yticks)
    plt.legend([],[], frameon=False)
    
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('fig_4/'+env+'.png')
    plt.close(fig)
    print("Finish plotting.")

def main(env, steps, yticks, xticks, palette=None):
    plot(env=env, steps=steps, yticks=yticks, xticks=xticks, palette=palette)

if __name__ == '__main__':
    for env in ["Allegro", "Ant", "Anymal", "Franka", "Humanoid", "Ingenuity"]:
        if env == "Allegro":
            steps = 1000000
            yticks = np.arange(-250, 750+200, 200)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Ant":
            steps = 1000000
            yticks = np.arange(-1000, 9000+2000, 2000)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Anymal":
            steps = 1000000
            yticks = np.arange(0, 60+15, 15)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Franka":
            steps = 1000000
            yticks = np.arange(0, 4000+1000, 1000)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Humanoid":
            steps = 1000000
            yticks = np.arange(-500, 5500+1500, 1500)
            xticks = np.arange(0, steps+1, 250000)
        elif env == "Ingenuity":
            steps = 500000
            yticks = np.arange(0, 6000+1500, 1500)
            xticks = np.arange(0, steps+1, 125000)

        palette = ['xkcd:deep sky blue', 'xkcd:orange']
        main(env, steps, yticks, xticks, palette=palette)