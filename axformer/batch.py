import torch

from axformer.transformer_utils import subsequent_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1] # input target data with all but the last token
            self.trg_y = trg[:, 1:] # output target with all but the first token
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt:torch.Tensor, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & \
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) # create mask for the target seq
        return tgt_mask