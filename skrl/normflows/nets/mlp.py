import torch
from torch import nn
from .. import utils

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(
        self,
        layers,
        dropout_rate=None,
        init=False,
        layernorm=False
    ):
        """
        layers: list of layer sizes from start to end
        leaky: slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        init_zeros: Flag, if true, weights and biases of last layer are initialized with zeros (helpful for deep models, see [arXiv 1807.03039](https://arxiv.org/abs/1807.03039))
        dropout_rate: Float, if specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        
        for k in range(len(layers) - 2):
            # Linear
            net.append(nn.Linear(layers[k], layers[k + 1]))

            # Set Initial values
            if init == "zero":
                nn.init.zeros_(net[-1].weight)
            elif init == "orthogonal":
                nn.init.orthogonal_(net[-1].weight)
            elif init == "xavier_uniform":
                nn.init.xavier_uniform_(net[-1].weight)
            elif init == "xavier_normal_":
                nn.init.xavier_normal_(net[-1].weight)
            else:
                NotImplementedError("This output function is not implemented.")

            if layernorm:
                net.append(nn.LayerNorm(layers[k + 1]))
            # Non-linear
            net.append(Swish(dim=layers[k + 1]))
            # Dropout
            if dropout_rate is not None:
                net.append(nn.Dropout(p=dropout_rate))
        
        net.append(nn.Linear(layers[-2], layers[-1]))

        # Set Initial values
        if init == "zero":
            nn.init.zeros_(net[-1].weight)
        elif init == "orthogonal":
            nn.init.orthogonal_(net[-1].weight)
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(net[-1].weight)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(net[-1].weight)
        else:
            NotImplementedError("This output function is not implemented.")

        # Construct the model
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
    
class Swish(nn.Module):
  def __init__(self, dim=-1):
    """
    Swish from: https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299
    """
    super().__init__()
    self.beta = nn.Parameter(torch.ones((dim,)))

  def forward(self, x):
    return x * torch.sigmoid(self.beta[None, :] * x)
