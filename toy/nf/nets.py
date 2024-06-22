import torch
from torch import nn

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
