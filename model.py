import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class Block(nn.Module):
    def __init__(self, block_embd: int, num_heads: int, dropout: float, device='cpu'):
        super().__init__()
        self.device = device

        self.ln_1 = nn.LayerNorm(block_embd)
        self.ln_2 = nn.LayerNorm(block_embd)
        self.mlp = nn.Sequential(
            nn.Linear(block_embd, 4 * block_embd),
            nn.GELU(),
            nn.Linear(4 * block_embd, block_embd),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            block_embd, num_heads, dropout, batch_first=True)

    def forward(self, X):
        sequence_length = X.shape[1]
        mask = torch.triu(torch.ones(sequence_length, sequence_length) *
                          float('-inf'), diagonal=1).to(self.device)

        X = X + self.attn(self.ln_1(X), self.ln_1(X),
                          self.ln_1(X), attn_mask=mask)[0]
        X = X + self.mlp(self.ln_2(X))
        return X


class PoseFlowModel(nn.Module):
    def __init__(
        self,
        block_size: int,
        pose_embd: int,
        block_embd: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_embd, 256),
            nn.GELU(),
            nn.Linear(256, block_embd)
        )
        self.position_embedding = nn.Embedding(block_size, block_embd)
        self.blocks = nn.Sequential(
            *[Block(block_embd, num_heads, dropout, device) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(block_embd)
        self.lm_head = nn.Sequential(
            nn.Linear(block_embd, 256),
            nn.GELU(),
            nn.Linear(256, pose_embd)
        )

        self.apply(self._init_weights)

    # from andrej karpathy gpt-2 code
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X):
        pose_emb = self.pose_encoder(X)
        pos_emb = self.position_embedding(
            torch.arange(X.shape[1], device=self.device))
        X = pose_emb + pos_emb
        X = X + self.blocks(X)
        X = self.ln_f(X)
        X = self.lm_head(X)
        return X

    def stream(self, X):
        while True:
            X = X[:, -self.block_size:, :]
            logits = self.forward(X)
            last_frame = logits[:, -1, :]
            X = torch.cat([X, last_frame.unsqueeze(0)], dim=1)
            yield last_frame
    
    def parameter_count(self):
        parameter_count = 0
        for parameter in self.parameters():
            parameter_count += parameter.numel()
        return parameter_count


