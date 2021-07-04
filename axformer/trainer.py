import time
from typing import List

import torch
from torch import nn

from axformer.label_smoothing import LabelSmoothing
from axformer.model.axformer import Axformer
from axformer.noam_opt import NoamOpt
from axformer.simple_loss_compute import SimpleLossCompute
from axformer.data_utils import toy_data_gen, MyIterator, rebatch
from axformer.multi_gpu_loss_compute import MultiGPULossCompute

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def batch_size_fn(new, count, sofar, max_src_in_batch, max_tgt_in_batch):
    "Keep augmenting batch and calculate total number of tokens + padding."
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements), max_src_in_batch, max_tgt_in_batch


# Train the simple copy task.
def toy_train():
    vocab_size, d_model, n_layers = 11, 512, 2
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)

    model = Axformer(vocab_size, vocab_size, n_layers=n_layers, d_model=d_model)
    model_opt = NoamOpt(d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(toy_data_gen(vocab_size, batch_size=30, nbatches=20), model,
                SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(toy_data_gen(vocab_size, batch_size=30, nbatches=5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

def create_iterators(devices:List[int], SRC, TGT, train, val):
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = Axformer(len(SRC.vocab), len(TGT.vocab), n_layers=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    return train_iter, valid_iter, model_par

def train(devices, criterion, pad_idx, model, train_iter, valid_iter, model_par, epochs=10):
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(epochs):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                          model_par,
                          MultiGPULossCompute(model.generator, criterion,
                          devices=devices, opt=None))
        print(loss)