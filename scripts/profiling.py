import torch
#from torch.profiler import profile, record_function, ProfilerActivity
from axformer.model.axformer import Axformer

vocab_size, d_model, n_layers = 11, 512, 2
model = Axformer(vocab_size, vocab_size, n_layers=n_layers, d_model=d_model)

batch_size, sentense_size = 30,10
src = torch.rand(batch_size, sentense_size, dtype=torch.int32)
tgt = torch.rand(batch_size, sentense_size-1, dtype=torch.int32)
src_mask = torch.rand(batch_size, 1, sentense_size, dtype=torch.bool)
tgt_mask = torch.rand(batch_size, sentense_size-1, sentense_size-1, dtype=torch.bool)

for n, p in model.named_parameters():
    print(n, torch.numel(p))

model(src, tgt, src_mask, tgt_mask)
