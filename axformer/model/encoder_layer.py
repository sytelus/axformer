from axformer.model.positionwise_feedforward import PositionwiseFeedForward
from axformer.model.multi_head_attention import MultiHeadedAttention
from torch import nn

from axformer.model.sublayer_connection import SublayerConnection
from axformer.transformer_utils import clones

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model:int, self_attn:MultiHeadedAttention, feed_forward:PositionwiseFeedForward, dropout:float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask): # x: shape [batch, sentence_size, d_model], mask: shape [batch, 1, d_model]
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)