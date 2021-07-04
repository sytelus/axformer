from torch import nn

from axformer.transformer_utils import clones
from axformer.layer_norm import LayerNorm

class Decoder(nn.Module):
    "Generic n_layers decoder with masking."
    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)