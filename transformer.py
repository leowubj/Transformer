# add all  your Encoder and Decoder code here

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_hidden, n_layer, block_size, n_output):
        super(TransformerEncoder, self).__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.attentionLayers = nn.ModuleList([Block(n_embd, n_head, block_size, "Encoder") for _ in range(n_layer)])
        self.layerNorm = nn.LayerNorm(n_embd)
        self.feedforward = FinalFeedForward(n_embd, n_hidden, n_output)

    def forward(self, x):
        attention_maps = []

        x = self.embedding(x) + self.positional_encoding(torch.arange(self.block_size, device=device))
        for layer in self.attentionLayers:
            x, attention = layer(x)
            attention_maps.append(attention)

        x = x.mean(dim=1)

        x = self.feedforward(x)

        return x, attention_maps


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_hidden, n_layer, block_size, n_output):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.attentionLayers = nn.ModuleList([Block(n_embd, n_head, block_size, "Decoder") for _ in range(n_layer)])
        self.feedforward = FinalFeedForward(n_embd, n_hidden, n_output)

    def forward(self, x):
        attention_maps = []

        x = self.embedding(x) + self.positional_encoding(torch.arange(self.block_size, device=device))
        for layer in self.attentionLayers:
            x, attention = layer(x)
            attention_maps.append(attention)

        x = self.feedforward(x)

        return x, attention_maps


class FinalFeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden, n_output):
        super().__init__()
        self.hidden_layer = nn.Linear(n_embd, n_hidden)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(n_hidden, n_output)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.fc_out(x)
        # x = self.log_softmax(x)
        return x


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, name):
        super().__init__()
        self.name = name
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.name == "Decoder":
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out, wei


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, name):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, name) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x)[0] for h in self.heads], dim=-1)
        out = self.proj(out)

        h1 = self.heads[0]
        return out, h1(x)[1]


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, name):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, name)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y, attn = self.sa(self.ln1(x))
        x = x + y
        x = x + self.ffwd(self.ln2(x))
        return x, attn


class AlibiHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, head_idx):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.n_head = n_embd // head_size
        self.head_idx = head_idx

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        m = 2 ** (-8 / self.n_head)
        m = m ** (self.head_idx + 1)

        indices = torch.arange(T, device=device)
        bias_pattern = (indices.view(1, -1) - indices.view(-1, 1)).clamp(max=0)  # Only keep values where k < j
        bias_pattern = bias_pattern.unsqueeze(0).expand(B, -1, -1)  # Expand pattern to [B, T, T]

        wei = wei + bias_pattern * m

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out, wei


class AlibiMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([AlibiHead(head_size, n_embd, block_size, i) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x)[0] for h in self.heads], dim=-1)
        out = self.proj(out)

        h1 = self.heads[0]
        return out, h1(x)[1]


class AlibiBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = AlibiMultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y, attn = self.sa(self.ln1(x))
        x = x + y
        x = x + self.ffwd(self.ln2(x))
        return x, attn


class AlibiDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_hidden, n_layer, block_size, n_output):
        super(AlibiDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.attentionLayers = nn.ModuleList([AlibiBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.feedforward = FinalFeedForward(n_embd, n_hidden, n_output)

    def forward(self, x):
        attention_maps = []

        x = self.embedding(x)
        for layer in self.attentionLayers:
            x, attention = layer(x)
            attention_maps.append(attention)

        x = self.feedforward(x)

        return x, attention_maps

# class AttentionLayer(nn.Module):
#     def __init__(self, n_embd, n_head):
#         super(AttentionLayer, self).__init__()
#
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head
#
#         self.query = nn.Linear(n_embd, n_embd)
#         self.key = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#
#         self.norm1 = nn.LayerNorm(n_embd)
#         self.norm2 = nn.LayerNorm(n_embd)
#
#         self.hidden = nn.Linear(n_embd, 4 * n_embd)
#         self.relu = nn.ReLU()
#         self.fc_out = nn.Linear(4 * n_embd, n_embd)
#
#     def forward(self, x):
#         y = self.norm1(x)
#
#         batch_size, seq_len, n_embd = y.shape
#
#         Q = self.query(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         K = self.key(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         V = self.value(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#
#         energy = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
#
#         attention = torch.softmax(energy, dim=-1)
#         out = torch.matmul(attention, V)
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
#
#         x = out + x
#
#         out2 = self.norm2(x)
#         out2 = self.hidden(out2)
#         out2 = self.relu(out2)
#         out2 = self.fc_out(out2)
#
#         x = x + out2
#
#         return x, attention

# class MaskedAttentionLayer(nn.Module):
#     def __init__(self, n_embd, n_head):
#         super(MaskedAttentionLayer, self).__init__()
#
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head
#
#         self.query = nn.Linear(n_embd, n_embd)
#         self.key = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.norm1 = nn.LayerNorm(n_embd)
#         self.norm2 = nn.LayerNorm(n_embd)
#
#         self.hidden = nn.Linear(n_embd, 4 * n_embd)
#         self.relu = nn.ReLU()
#         self.fc_out = nn.Linear(4 * n_embd, n_embd)
#
#     def forward(self, x):
#         y = self.norm1(x)
#
#         batch_size, seq_len, n_embd = y.shape
#
#         Q = self.query(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         K = self.key(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#         V = self.value(y).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
#
#         energy = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
#
#         mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
#         energy = energy.masked_fill(mask == 0, float('-inf'))
#
#         attention = torch.softmax(energy, dim=-1)
#         out = torch.matmul(attention, V)
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
#
#         x = out + x
#
#         out2 = self.norm2(x)
#         out2 = self.hidden(out2)
#         out2 = self.relu(out2)
#         out2 = self.fc_out(out2)
#
#         x = x + out2
#
#         return x, attention
