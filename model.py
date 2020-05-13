from torch import nn
import torch.nn.functional as F


class ConvText(nn.Module):
    def __init__(self, n_emb, emb_dim, seq_len, n_kernel, kernel_size, n_outputs, pad_idx=0):
        super().__init__()
        self.n_kernel = n_kernel

        self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=emb_dim,
                                padding_idx=pad_idx)

        self.conv = nn.Conv1d(seq_len, n_kernel, kernel_size)

        self.pool = nn.MaxPool1d(kernel_size)

        self.hidden_size = int(((emb_dim - kernel_size + 1) - kernel_size) / kernel_size + 1)

        self.linear = nn.Linear(self.hidden_size * self.n_kernel, n_outputs)

    def forward(self, x):
        x = F.relu(self.emb(x))

        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.hidden_size * self.n_kernel)

        return self.linear(x)
