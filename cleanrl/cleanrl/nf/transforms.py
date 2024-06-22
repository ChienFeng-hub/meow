import torch
import numpy as np
from . import flows

class arcTanh(flows.Flow):
    def __init__(self):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, z):
        z_ = torch.tanh(z)
        log_det = torch.log(1-z_.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

    @torch.jit.export
    def inverse(self, z):
        z_ = torch.atanh(z)
        log_det = -torch.log(1-z.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

class Clip(flows.Flow):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z_):
        # (Generation direaction) output must be [-1, 1] (this is defined according to the env.)
        z_ = torch.clamp(z_, -1, 1) 
        return z_, torch.zeros(z_.shape[0], device=z_.device)
    
    @torch.jit.export
    def inverse(self, z):
        # (Density estimation direction) input must be [-1+esp, 1-esp] (prevent NAN outputs after preprocessing operation.)
        z = torch.clamp(z, -1+self.eps, 1-self.eps) 
        return z, torch.zeros(z.shape[0], device=z.device)
    
class Preprocessing(flows.Flow):
    def __init__(self):
        super().__init__()
        trans = [arcTanh(), Clip(eps=1e-5)]
        self.trans = torch.nn.ModuleList(trans)

    def forward(self, z, context=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.trans:
            z, log_d = flow.forward(z)
            log_det += log_d
        return z, log_det

    @torch.jit.export
    def inverse(self, z, context=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.trans[::-1]:
            z, log_d = flow.inverse(z)
            log_det += log_d
        return z, log_det
    
    @torch.jit.export
    def get_qv(self, z, context):
        z_, q = self.inverse(z, context)
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v