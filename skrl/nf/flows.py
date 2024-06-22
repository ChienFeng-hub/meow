import numpy as np
import torch
from torch import nn

class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Args:
          z: input variable, first dimension is batch dim

        Returns:
          transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")

class MaskedCondAffineFlow(Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)
        self.s = s
        self.t = t

    def get_st(self, z_masked, context):
        tmp = torch.cat([z_masked, context], dim=1)
        scale = self.s(tmp) if self.s is not None else torch.zeros_like(z_masked)
        trans = self.t(tmp) if self.t is not None else torch.zeros_like(z_masked)
        return scale, trans

    def forward(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        z_, log_det = self.inverse(z, context)
        q = log_det
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v


class CondScaling(Flow):
    '''
    Transformation: z_ = z * exp(s) / exp(s)
    '''
    def __init__(self, s1, s2=None):
        super().__init__()
        self.scale1 = s1
        self.scale2 = s2

    def forward(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det
    
    @torch.jit.export
    def inverse(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        if self.scale2 is not None:
            s1 = self.scale1(context[:context.shape[0]//2])
            s2 = self.scale2(context[:context.shape[0]//2])
            q = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
            v = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
        else:
            s1 = self.scale1(context)
            q = s1[:, 0]
            v = s1[:, 0]
        return z, q, v
    
class Scaling(Flow):
    '''
    Transformation: z_ = z * exp(s) / exp(s)
    '''
    def __init__(self, init_value=100):
        super().__init__()
        self.scale1 = torch.nn.Parameter(torch.ones(1)*init_value)
        self.scale2 = torch.nn.Parameter(torch.ones(1)*init_value)
        self.scale1.requires_grad = True
        self.scale2.requires_grad = True

    def forward(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det
    
    @torch.jit.export
    def inverse(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        if self.scale2 is not None:
            s1 = self.scale1.view(1,1).repeat(z.shape[0]//2, 1)
            s2 = self.scale2.view(1,1).repeat(z.shape[0]//2, 1)
            q = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
            v = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
        else:
            s1 = self.scale1
            q = s1[:, 0]
            v = s1[:, 0]
        return z, q, v