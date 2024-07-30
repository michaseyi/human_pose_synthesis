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
            nn.Linear(pose_embd, 50),
            nn.GELU(),
            nn.Linear(50, block_embd)
        )
        self.position_embedding = nn.Embedding(block_size, block_embd)
        self.blocks = nn.Sequential(
            *[Block(block_embd, num_heads, dropout, device) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(block_embd)
        self.lm_head = nn.Sequential(
            nn.Linear(block_embd, 50),
            nn.GELU(),
            nn.Linear(50, pose_embd)
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
        B, T, C, D = X.shape
        pose_emb = self.pose_encoder(X.view(B, T, C * D))
        pos_emb = self.position_embedding(
            torch.arange(X.shape[1], device=self.device))
        X = pose_emb + pos_emb
        X = X + self.blocks(X)
        X = self.ln_f(X)
        X = self.lm_head(X)
        X = F.normalize(X.view(B, T, C, D), p=2, dim=-1)
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


# def linear_beta_schedule(timesteps):
#     scale = 1000 / timesteps
#     start = scale * 0.0001
#     end = scale * 0.02
#     return torch.linspace(start, end, timesteps)


# def extract(a, t, x_shape):
#     b = t.shape[0]
#     out = a.gather(-1, t)
#     print(out.shape, b)
#     return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


# class GaussianDiffusion(nn.Module):
#     def __init__(
#             self,
#             denoiser,
#             timesteps=300,
#     ):
#         super().__init__()
#         betas = linear_beta_schedule(timesteps)
#         alphas = 1. - betas

#         alphas_cumprod = torch.cumprod(alphas, 0)
#         alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

#         self.register_buffer("betas", betas)
#         self.register_buffer("alphas_cumprod", alphas_cumprod)
#         self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
#         self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
#         self.register_buffer("sqrt_one_minus_alphas_cumprod",
#                              torch.sqrt(1. - alphas_cumprod))

#     def forward_diffusion_sample(self, x_0, t):
#         noise = torch.randn_like(x_0)
#         return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise, noise
