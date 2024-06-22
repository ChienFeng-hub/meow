import torch
import numpy as np
from . import flows

class Shift(flows.Flow):
    """Shift data by a fixed constant

    Default is -0.5 to shift data from
    interval [0, 1] to [-0.5, 0.5]
    """

    def __init__(self, shift=-0.5, MaP=False):
        """Constructor

        Args:
          shift: Shift to apply to the data
        """
        super().__init__()
        self.shift = shift
        self.MaP = MaP

    def forward(self, z):
        z = z - self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det

    @torch.jit.export
    def inverse(self, z):
        z = z + self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det

# (Lance implemented - 20230117)
class Scale(flows.Flow):
    def __init__(self, scale=0.5, MaP=False):
        super().__init__()
        # forward: in / scale
        # inverse: in * scale
        self.scale = scale
        self.MaP = MaP

    def forward(self, z):
        z = z / (self.scale)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            # log |det| = -D * log (self.scale)
            log_det = -torch.ones(z.shape[0], device=z.device) * np.log(np.abs(self.scale)) * z.shape[1]
        return z, log_det

    @torch.jit.export
    def inverse(self, z):
        z = z * (self.scale)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            # log |det| = D * log (self.scale)
            log_det = torch.ones(z.shape[0], device=z.device) * np.log(np.abs(self.scale)) * z.shape[1]
        return z, log_det

class arcTanh(flows.Flow):
    def __init__(self, MaP=False):
        super().__init__()
        self.MaP = MaP
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, z):
        z_ = torch.tanh(z)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = torch.log(1-z_.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

    @torch.jit.export
    def inverse(self, z):
        z_ = torch.atanh(z)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = -torch.log(1-z.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

# (Lance implemented - 20240121)
class Clip(flows.Flow):
    def __init__(self, scale=1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = 1/scale

    def forward(self, z_):
        # (Generation direaction) output must be [-1, 1] (this is defined according to the env.)
        # z_ = torch.clamp(z_, -1+self.eps, 1-self.eps)
        z_ = torch.clamp(z_, -self.scale, self.scale) 
        return z_, torch.zeros(z_.shape[0], device=z_.device)
    
    @torch.jit.export
    def inverse(self, z):
        # (Density estimation direction) input must be [-1+esp, 1-esp] (prevent having nan in fwd passing logit or tanh...)
        z = torch.clamp(z, -self.scale+self.eps, self.scale-self.eps) 
        return z, torch.zeros(z.shape[0], device=z.device)
    
class Preprocessing(flows.Flow):
    def __init__(self, option='eye', clip=True, scale=1, eps=1e-5):
        super().__init__()
        self.MaP = False

        if option == 'eye':
            trans = [Shift(shift=0.)]
        elif option == 'atanh':
            trans = [arcTanh(MaP=self.MaP)]
        elif option == 'scaleatanh':
            trans = [arcTanh(MaP=self.MaP), Scale(scale=scale, MaP=self.MaP)]
        else:
            raise NotImplementedError("Sorry, not implemented!")

        if clip:
            trans += [Clip(scale=scale, eps=eps)]
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

    def get_qv_fwd(self, z, context):
        z_, q = self.forward(z, context)
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v