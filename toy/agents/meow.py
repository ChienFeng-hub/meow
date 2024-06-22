from copy import deepcopy
import torch
from torch import nn
from modules.buffer import ReplayBuffer
from modules.policy import FlowPolicy
from modules.train_loop import train_loop
from torch.utils.tensorboard import SummaryWriter


class MEOW():
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.device = args.device
        self.logger = SummaryWriter(log_dir=args.save_path)
        self.buffer = ReplayBuffer(int(args.buffer_size), args.state_sizes, args.action_sizes)
        self.state_sizes = args.state_sizes
        self.action_sizes = args.action_sizes
        
        policy = FlowPolicy(args.alpha, args.sigma_max, args.sigma_min, args.action_sizes, args.state_sizes, args.device)
        self.policy, self.policy_old = policy, deepcopy(policy)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def train(self, train_envs, test_envs):
        self.unit_test()
        train_loop(self, self.args, self.buffer, train_envs, test_envs, self.logger)

    def unit_test(self):
        self.policy.eval()
        num_samples = 10
        state = torch.randn((num_samples, self.state_sizes), device=self.device)
        action = torch.clamp(torch.randn((num_samples, self.action_sizes), device=self.device), min=-1, max=1)
        state = torch.cat((state, state), dim=0)
        action = torch.cat((action, action), dim=0)
        log_prob = self.policy.log_prob(obs=state, act=action)
        q, v = self.policy.get_qv(obs=state, act=action)
        v_ = self.policy.get_v(obs=state)
        assert torch.allclose(log_prob * self.args.alpha, ((q-v)).squeeze())
        print("Pass Test 1: (q - v) = alpha * log p")
        assert torch.allclose(v, v_)
        print("Pass Test 2: v is a constant w.r.t. a")

    def act(self, s, deterministic=False):
        act, log_prob = self.policy.sample(num_samples=s.shape[0], obs=s, deterministic=deterministic)
        return act, log_prob[:, None]
    
    def update_q(self, batch, mseloss=nn.MSELoss(reduction='none')):
        (s, a, r, s_n, d) = batch
        with torch.no_grad():
            self.policy.eval()
            v_old = self.policy_old.get_v(torch.cat((s_n, s_n), dim=0))
            exact_v_old = torch.min(v_old[:v_old.shape[0]//2], v_old[v_old.shape[0]//2:])
            target_q = r + (1-d) * self.args.gamma * exact_v_old

        self.policy.train()
        current_q1, _ = self.policy.get_qv(torch.cat((s, s), dim=0), torch.cat((a, a), dim=0))
        target_q = torch.cat((target_q, target_q), dim=0)
        c1_loss = mseloss(current_q1.flatten(), target_q.flatten())
        c1_loss[c1_loss != c1_loss] = 0.0
        c1_loss = c1_loss.mean()
        self.optim.zero_grad(set_to_none=True)
        c1_loss.backward()
        # Gradient clipping
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip)
        self.optim.step()
        # check gradient norm
        weights = None
        for p in list(filter(lambda p: p.grad is not None, self.policy.parameters())):
            weights = torch.cat((weights, p.grad.flatten()), 0) if weights is not None else p.grad.flatten()
        norm = torch.sqrt((weights**2).sum())

        result = {
            "Loss/critic1_loss": c1_loss.item(),
            "Grad_norm/norm": norm.item(),
            "Value/exact_v_old": exact_v_old.mean().item(),
            "Q/current_q1": current_q1.mean().item(),
            "Q/target_q": target_q.mean().item(),
        }
        return result

    def update(self):
        batch = self.buffer.sample(self.args.batch_size, device=self.device)
        (s, a, r, s_n, d) = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
        # Update policy
        result1 = self.update_q(batch=(s, a, r, s_n, d))
        # sync weights
        self.soft_update(self.policy_old, self.policy)
        return {**result1}
    
    def soft_update(self, tgt: nn.Module, src: nn.Module):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(self.args.tau * src_param.data + (1-self.args.tau) * tgt_param.data)

    def save(self, path, best_return):
        torch.save({
            'best_return': best_return,
            'policy': self.policy.state_dict(),
            'optim': self.optim.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optim.load_state_dict(checkpoint['optim'])
        return checkpoint['best_return']
    
