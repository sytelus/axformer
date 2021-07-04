import copy

from torch import nn

from axformer.model.multi_head_attention import MultiHeadedAttention
from axformer.model.positionwise_feedforward import PositionwiseFeedForward
from axformer.model.positional_encoding import PositionalEncoding
from axformer.model.encoder import Encoder
from axformer.model.decoder import Decoder
from axformer.model.embeddings import Embeddings
from axformer.model.generator import Generator
from axformer.model.encoder_layer import EncoderLayer
from axformer.model.decoder_layer import DecoderLayer

class Axformer(nn.Module):
    def __init__(self, src_vocab_size:int, tgt_vocab_size:int, n_layers=6,
            d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab_size), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position))
        self.generator = Generator(d_model, tgt_vocab_size)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)