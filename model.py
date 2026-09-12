import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    def generate(self, index, max_new_tokens: int, block_size: int):
        for _ in range(max_new_tokens):
            logits, _ = self(index[:, -block_size:])
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            index = torch.cat((index, torch.multinomial(probabilities, 1)), dim=1)
        return index


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None):
        _, time = index.shape
        x = self.token_embedding(index) + self.position_embedding(torch.arange(time, device=index.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln(x))
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    def generate(self, index, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(index[:, -self.block_size :])
            index = torch.cat((index, torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)), dim=1)
        return index


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, time, channels = x.shape
        weights = self.query(x) @ self.key(x).transpose(-2, -1) * self.head_size ** -0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        return self.dropout(F.softmax(weights, dim=-1)) @ self.value(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = nn.ModuleList([CausalSelfAttention(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.feed_forward = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.projection(torch.cat([head(self.ln1(x)) for head in self.attention], dim=-1))
        return x + self.feed_forward(self.ln2(x))
