import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "./data/"
batch_size = 8
seq_len = 20000
epochs = 10
learning_rate = 1e-3
