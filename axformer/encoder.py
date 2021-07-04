from torch import nn

from axformer.transformer_utils import clones
from axformer.layer_norm import LayerNorm

class Encoder(nn.Module):
    "Core encoder is a stack of n_layers"
    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)