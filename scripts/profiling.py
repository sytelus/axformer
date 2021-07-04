from axformer.model.axformer import Axformer

vocab_size, d_model, n_layers = 11, 512, 2
model = Axformer(vocab_size, vocab_size, n_layers=n_layers, d_model=d_model)