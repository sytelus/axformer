
from torch import nn

from axformer.model.sublayer_connection import SublayerConnection
from axformer.transformer_utils import clones
from axformer.model.positionwise_feedforward import PositionwiseFeedForward
from axformer.model.multi_head_attention import MultiHeadedAttention

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, self_attn:MultiHeadedAttention, src_attn, feed_forward:PositionwiseFeedForward, dropout:float):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)