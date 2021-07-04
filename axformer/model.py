import copy

from torch import nn

from axformer.multi_head_attention import MultiHeadedAttention
from axformer.positionwise_feedforward import PositionwiseFeedForward
from axformer.positional_encoding import PositionalEncoding
from axformer.encoder import Encoder
from axformer.decoder import Decoder
from axformer.encoder_decoder import EncoderDecoder
from axformer.embeddings import Embeddings
from axformer.generator import Generator
from axformer.encoder_layer import EncoderLayer
from axformer.decoder_layer import DecoderLayer

def make_model(src_vocab_size, tgt_vocab_size, n_layers=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1)->EncoderDecoder:
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), n_layers),
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model