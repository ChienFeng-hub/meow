import os
import numpy as np
import torch
from modules.utils import log, plot_traj_multigoal, plot_value

def evaluate(envs, agent, deterministic=True):
    with torch.no_grad():
        num_envs = envs.unwrapped.num_envs
        rewards = np.zeros((num_envs,))
        dones = np.zeros((num_envs,)).astype(bool)
        s, _ = envs.reset(seed=range(num_envs))
        while not all(dones):
            a, _ = agent.act(s, deterministic=deterministic)
            a = a.cpu().detach().numpy()
            s_, r, terminated, truncated, _ = envs.step(a)
            done = terminated | truncated
            rewards += r * (1-dones)
            dones |= done
            s = s_
    return rewards.mean()

def train_loop(agent, args, buffer, train_envs, test_envs, logger):
    best_test_return = -np.inf
    best_train_return = -np.inf
    episode_return = 0
    episode = 0
    obs, _ = train_envs.reset(seed=args.seed)
    for t in range(args.steps+1):
        if t < args.warmup_steps:
            act = train_envs.action_space.sample()
        else:
            agent.policy.eval()
            act, _ = agent.act(obs)
            act = act.cpu().detach().numpy()
        
        next_obs, reward, terminated, truncated, info = train_envs.step(act)

        # Be careful of final observation in vectorized env, see: https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]

        # Note that done now change to "termination. see: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        buffer.store(
            s=obs,
            a=act,
            r=reward,
            s_=real_next_obs,
            d=terminated*1. 
        )
        episode_return += reward
        obs = next_obs

        if t >= args.warmup_steps:
            result = agent.update()
            if t % args.eval_every == 0:
                agent.policy.eval()
                test_return = evaluate(test_envs, agent, deterministic=args.deterministic)
                print(f"return = {test_return} at step {t}")
                if test_return > best_test_return:
                    best_test_return = test_return
                    torch.save(agent.policy, os.path.join(args.save_path, 'best.pt'))
                    print(f"save agent to: {args.save_path} with best return {best_test_return} at step {t}")
                log({
                    **result,
                    "Test/return": test_return,
                    "Test/best_return": best_test_return,
                    "Steps": t,
                }, t, logger)
            if t % args.plot_every == 0:
                figdir = os.path.join(args.save_path, 'figures', str(t))
                _, _ = plot_traj_multigoal(agent.policy, figdir+'_traj')
                plot_value(agent.policy, figdir+'_value')
                torch.save(agent.policy, os.path.join(args.save_path, 'best.pt'))
        
        if terminated or truncated:
            best_train_return = max(best_train_return, episode_return)
            log({
                "Train/return": episode_return,
                "Train/best_return": best_train_return,
                "Episodes": episode,
            }, episode, logger)
            episode_return = 0
            episode += 1

    torch.save(agent.policy, os.path.join(args.save_path, 'last.pt'))
    print(f"save last agent to: {args.save_path}")