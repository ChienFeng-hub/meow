from typing import Any, Mapping, Tuple, Union

import gym
import gymnasium

import torch
from torch.distributions import Normal


class FlowMixin:
    def __init__(self, reduction: str = "sum") -> None:
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = torch.mean if reduction == "mean" else torch.sum if reduction == "sum" \
            else torch.prod if reduction == "prod" else None

    def unit_test(self, num_samples=10, scale=1):
        self.eval()
        state = torch.randn((num_samples, self.num_observations), device=self.device)
        action = torch.clamp(torch.randn((num_samples, self.num_actions), device=self.device), min=-scale+1e-5, max=scale-1e-5)
        state = torch.cat((state, state), dim=0)
        action = torch.cat((action, action), dim=0)
        log_prob = self.log_prob(obs=state, act=action)
        q, v = self.get_qv(obs=state, act=action)
        v_ = self.get_v(obs=state)
        assert torch.allclose(log_prob * self.alpha, ((q-v)).squeeze())
        print("Pass Test 1: (q - v) = alpha * log p")
        assert torch.allclose(v, v_)
        print("Pass Test 2: v is a constant w.r.t. a")
        
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        obs_ = inputs['states']
        obs = torch.as_tensor(obs_, dtype=torch.float32, device=self.device)
        noises, log_prob = self.prior.sample(num_samples=obs_.shape[0], context=obs)
        actions, log_det = self.forward(obs=obs, act=noises)
        log_prob = log_prob + log_det
        return actions, log_prob, noises

    def forward(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in self.flows:
            z, log_det = flow.forward(z, context=obs)
            log_q -= log_det
        return z, log_q
    
    def inverse(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, log_det = flow.inverse(z, context=obs)
            log_q += log_det
        return z, log_q
    
    def log_prob(self, obs, act):
        z, log_q = self.inverse(obs=obs, act=act)
        log_q += self.prior.log_prob(z, context=obs)
        return log_q
    
    def get_qv(self, obs, act):
        q = torch.zeros((act.shape[0]), device=act.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, q_, v_ = flow.get_qv(z, context=obs)
            q += q_
            v += v_
        q_, v_ = self.prior.get_qv(z, context=obs)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
        return q[:, None], v[:, None]
    
    def get_v(self, obs):
        act = torch.zeros((obs.shape[0], self.num_actions), device=self.device)
        v = torch.zeros((obs.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, _, v_ = flow.get_qv(z, context=obs)
            v += v_
        _, v_ = self.prior.get_qv(z, context=obs)
        v += v_
        v = v * self.alpha
        return v[:, None]
