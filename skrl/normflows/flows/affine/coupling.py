import numpy as np
import torch
from torch import nn

from ..base import Flow, zero_log_det_like_z
from ..reshape import Split, Merge


class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """

    def __init__(self, shape, num_classes):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.s_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.t_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = z * torch.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = (z - t) * torch.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2; ```z = [z1, z2]```
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1

        Args:
          z
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 - param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det


class MaskedAffineFlow(Flow):
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

        if s is None:
            self.s = torch.zeros_like
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like
        else:
            self.add_module("t", t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


# (Roy implemented - 20240128)
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
        # TODO (Jonhson): check if this is neccessary
        # nan = torch.tensor(np.nan, dtype=z_masked.dtype, device=z_masked.device)
        # scale = torch.where(torch.isfinite(scale), scale, nan)
        # trans = torch.where(torch.isfinite(trans), trans, nan)
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
    
    def get_qv_fwd(self, z, context):
        z_, log_det = self.forward(z, context)
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
            # assert torch.all(context[:context.shape[0]//2] == context[context.shape[0]//2:])
            s1 = self.scale1(context[:context.shape[0]//2])
            s2 = self.scale2(context[:context.shape[0]//2])
            q = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
            v = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
        else:
            s1 = self.scale1(context)
            q = s1[:, 0]
            v = s1[:, 0]

        return z, q, v
    
    def get_qv_fwd(self, z, context):
        z, q, v = self.get_qv(z, context)
        return z, q, v

# class CondAffineLinear(Flow):
#     """
#     Affine flow layer conditioned on context
#     """
#     def __init__(self, latent_size, hidden_layers, hidden_units, context_size, init_zeros=False):
#         super().__init__()
#         self.scale = MLP([context_size] + [hidden_units]*hidden_layers + [latent_size], init_zeros=init_zeros) #, output_fn='tanh', output_scale=10)
#         self.translate = MLP([context_size] + [hidden_units]*hidden_layers + [latent_size], init_zeros=init_zeros)
#         self.eps = 1e-3

#     def logabsdet(self, s):
#         # return torch.sum(s, dim=list(range(1, s.dim())))
#         return torch.sum(torch.log(s), dim=list(range(1, s.dim())))
    
#     def get_st(self, context):
#         # s = self.scale(context)
#         s = F.softplus(self.scale(context)) + self.eps
#         t = self.translate(context)
#         return s, t

#     def forward(self, z, context):
#         s, t = self.get_st(context)
#         z_ = (z - t) / s # torch.exp(s)
#         log_det = -self.logabsdet(s)
#         return z_, log_det
    
#     def inverse(self, z, context):
#         s, t = self.get_st(context)
#         z_ = t + z * s # torch.exp(s)
#         log_det = self.logabsdet(s)
#         return z_, log_det
    
# class CondAffineCoupling(Flow):
#     """
#     Affine flow layer conditioned on x and context
#     """
#     def __init__(self, input_size, output_size, hidden_layers, hidden_units, context_size, init_zeros=False):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.scale = MLP([context_size+input_size] + [hidden_units]*hidden_layers + [output_size], init_zeros=init_zeros) #, output_fn='tanh', output_scale=10)
#         self.translate = MLP([context_size+input_size] + [hidden_units]*hidden_layers + [output_size], init_zeros=init_zeros)

#     # diag
#     def forward(self, z, context):
#         z1, z2 = z[:, :self.input_size, ...], z[:, self.input_size:, ...]
#         tmp = torch.cat([z1, context], dim=1)
#         s = self.scale(tmp)
#         t = self.translate(tmp)
#         z2_ = (z2 - t) * torch.exp(-s)
#         z_ = torch.cat([z1, z2_], dim=1)
#         log_det = -torch.sum(s, dim=list(range(1, t.dim())))
#         return z_, log_det
    
#     # diag
#     def inverse(self, z, context):
#         z1, z2 = z[:, :self.input_size, ...], z[:, self.input_size:, ...]
#         tmp = torch.cat([z1, context], dim=1)
#         s = self.scale(tmp)
#         t = self.translate(tmp)
#         z2_ = z2 * torch.exp(s) + t
#         z_ = torch.cat([z1, z2_], dim=1)
#         log_det = torch.sum(s, dim=list(range(1, t.dim())))
#         return z_, log_det